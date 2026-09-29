"""Order recommendations for one depot.

Each depot has one priority (config: depots.<id>.priority):

- service_level (Beverage & Perishable): order up to the service-level
  quantile of demand until the delivery after next. Alcohol at 99%. Perishables
  use a newsvendor quantity and are cut back until expected spoilage is within
  limit. If the cooler or truck is full, cut non-alcohol first.
- presence (Luxury): keep at least one of every active SKU, order up to a 90%
  quantile, and phase out a SKU when it keeps declining while a sibling in the
  same family is taking its place.
- cost (Dry goods & tobacco): skip the truck when stock safely lasts until the
  next one; if the load is small but a truck is needed, pull the next cycle's
  demand forward to make the trip worth it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from forecasting.data import complete_series
from forecasting.diagnostics import trend_status
from forecasting.pipeline import SkuForecast
from ordering.demand import WindowDemand, window_demand
from ordering.schedule import delivery_dates


@dataclass
class DepotPlan:
    depot_id: str
    name: str
    asof: pd.Timestamp
    order_by: pd.Timestamp
    delivery: pd.Timestamp
    next_delivery: pd.Timestamp
    decision: str
    lines: pd.DataFrame
    pallets: float
    truck_pallets: float
    trip_cost: float
    load_value: float
    notes: list[str] = field(default_factory=list)

    @property
    def fill_pct(self) -> float:
        return 100 * self.pallets / self.truck_pallets if self.truck_pallets else 0.0

    @property
    def units(self) -> int:
        return int(self.lines["order_units"].sum()) if not self.lines.empty else 0

    @property
    def cost_per_unit(self) -> float | None:
        return self.trip_cost / self.units if self.units else None


def _case_round(qty: float, pack: int, max_qty: float | None = None) -> int:
    """Round up to whole cases; if that breaks a capacity limit, round down."""
    if qty <= 0:
        return 0
    q = math.ceil(qty / pack) * pack
    if max_qty is not None and q > max_qty:
        q = max(0, math.floor(max_qty / pack) * pack)
    return int(q)


class _Line:
    """Working state for one SKU while a plan is built."""

    def __init__(self, sku, product, inv, fc: SkuForecast, cutoff, asof, d1, d2, conf):
        self.sku = sku
        self.p = product
        self.category = product["category"]
        self.pack = int(product["case_pack"])
        self.upp = int(product["case_pack"] * product["cases_per_pallet"])  # units per pallet
        self.on_hand = int(inv["on_hand"])
        self.overflow = int(inv["overflow_units"])
        self.on_order = int(inv["on_order"])
        self.fc = fc
        self.cutoff, self.asof, self.d1, self.d2, self.conf = cutoff, asof, d1, d2, conf
        self.cover_to = d2
        self.window = self._window(asof, d2)
        self.to_first = self._window(asof, d1)
        self.target = 0
        self.qty = 0
        self.notes: list[str] = []
        self.stopped = False  # phased out / dead: not replenished

    def _window(self, start, end) -> WindowDemand:
        return window_demand(self.fc.horizons, self.cutoff, start, end, self.conf)

    @property
    def position(self) -> int:
        return self.on_hand + self.overflow + self.on_order

    def set_cover(self, end):
        self.cover_to = end
        self.window = self._window(self.asof, end)

    def order_up_to(self, service: float, minimum: int = 0, max_stock: float | None = None):
        self.target = max(self.window.quantile(service), minimum)
        room = None if max_stock is None else max_stock - self.position
        self.qty = _case_round(self.target - self.position, self.pack, room)

    def pallets(self, qty=None) -> float:
        return (self.qty if qty is None else qty) / self.upp

    def row(self) -> dict:
        daily = self.fc.horizons[0].forecast / (self.cutoff + pd.DateOffset(months=1)).days_in_month
        risk = self.window.prob_exceeds(self.position + self.qty)
        early = self.to_first.prob_exceeds(self.on_hand + self.overflow)
        notes = list(self.notes)
        if early > 0.2 and not self.stopped:
            notes.insert(0, f"{early:.0%} chance of running out before {self.d1:%b %d}")
        return {
            "sku": self.sku,
            "item": self.p["description"],
            "on_hand": self.on_hand,
            "overflow": self.overflow,
            "sales_per_day": round(daily, 1 if daily >= 1 else 2),
            "cover_until": self.cover_to.strftime("%b %d"),
            "target": self.target,
            "order_units": self.qty,
            "cases": self.qty // self.pack,
            "pallets": round(self.pallets(), 2),
            "stockout_risk": "-" if self.stopped else f"{risk:.0%}",
            "note": "; ".join(notes),
        }


# ---------------------------------------------------------------------------
# Shared constraints
# ---------------------------------------------------------------------------

def _cut_to_fit(lines: list[_Line], limit: float, measure, order: list[str], protect=None) -> float:
    """Remove whole cases, least important category first and largest line
    first, until measure(lines) <= limit. Returns the remaining excess."""
    for cat in order:
        group = sorted([l for l in lines if l.category == cat], key=lambda l: -l.pallets())
        for line in group:
            while measure(lines) > limit + 1e-9 and line.qty > 0:
                floor = protect(line) if protect else 0
                if line.qty - line.pack < floor:
                    break
                line.qty -= line.pack
                if "cut to fit" not in line.notes:
                    line.notes.append("cut to fit")
    return max(0.0, measure(lines) - limit)


def _apply_overflow(lines: list[_Line], free_pallets: float, overflow_cats: list[str]) -> float:
    """Stock above shelf_max must go to overflow (allowed categories only).
    Allocates free overflow space in line order; trims what does not fit."""
    for line in lines:
        excess = line.on_hand + line.qty - line.p["shelf_max"]
        if excess <= 0:
            continue
        if line.category in overflow_cats:
            fits = int(min(excess, free_pallets * line.upp))
            free_pallets -= fits / line.upp
            if fits:
                line.notes.append(f"{fits} to overflow")
            excess -= fits
        if excess > 0:
            line.qty = max(0, line.qty - math.ceil(excess / line.pack) * line.pack)
            line.notes.append("limited by shelf space")
    return free_pallets


# ---------------------------------------------------------------------------
# Depot priorities
# ---------------------------------------------------------------------------

def _plan_service_level(lines, depot, store, free_overflow, notes):
    sl = depot["service_level"]
    spoil = depot["perishable"]["max_spoilage_share"]
    for line in lines:
        if line.category == "perishable":
            p = line.p
            life = int(p["shelf_life_days"])
            per = depot["perishable"]
            if life > per["long_life_days"]:
                # Unsold stock carries over to the next cycle, so a newsvendor
                # quantity would under-order. Use a normal service level.
                line.order_up_to(per["long_life_service_level"])
            else:
                # Newsvendor: unsold units are lost, so the critical ratio is
                # margin / price (no salvage value).
                line.order_up_to((p["unit_price"] - p["unit_cost"]) / p["unit_price"])
            sell_by = line.d1 + pd.Timedelta(days=life)
            demand_life = line._window(line.asof, sell_by)
            # Expected units left to spoil must stay under a share of expected sales.
            allowed = spoil * demand_life.mean
            q = line.qty
            while q > 0 and demand_life.expected_leftover(line.position + q) > allowed:
                q -= line.pack
            if q < line.qty:
                line.notes.append(f"trimmed for {life}-day shelf life")
            line.qty = max(q, 0)
        else:
            line.order_up_to(sl.get(line.category, 0.95))

    order = depot["cut_order"]
    _apply_overflow(sorted(lines, key=lambda l: order.index(l.category), reverse=True),
                    free_overflow, store["overflow_categories"])

    cooler = [l for l in lines if l.p["needs_cooler"]]
    cooler_use = lambda ls: sum((l.on_hand + l.qty) / l.upp for l in ls if l.p["needs_cooler"])
    left = _cut_to_fit(cooler, store["cooler_pallets"], cooler_use, order)
    if left > 0:
        notes.append(f"Cooler over capacity by {left:.2f} pallets even with no cold order.")

    truck = lambda ls: sum(l.pallets() for l in ls)
    left = _cut_to_fit(lines, depot["truck_pallets"], truck, order)
    if left > 0:
        notes.append(f"Load exceeds the truck by {left:.2f} pallets: book a second trip.")

    for line in lines:
        if line.category == "alcohol":
            risk = line.window.prob_exceeds(line.position + line.qty)
            if risk > 1 - sl["alcohol"] + 1e-9:
                notes.append(f"{line.p['description']}: {risk:.0%} stockout risk after "
                             "constraints. Keep-stocked item: add a trip or free space.")
    return "Order"


def _yoy(hist: pd.Series) -> float | None:
    """Last 6 months of sales relative to the same 6 months a year earlier."""
    if len(hist) < 18 or hist.iloc[-18:-12].sum() == 0:
        return None
    return float(hist.tail(6).sum() / hist.iloc[-18:-12].sum())


def _successor_at(sku: str, siblings: list[str], monthly, cutoff, runs: int) -> str | None:
    """Did the cycling rule fire at this cutoff? Uses sales data only, so it can
    be evaluated for past months too."""
    for k in range(runs):
        hist = complete_series(monthly, sku, cutoff - pd.DateOffset(months=k))
        if not len(hist) or trend_status(hist) not in ("Dying", "Dead"):
            return None
    hist = complete_series(monthly, sku, cutoff)
    yoy = _yoy(hist)
    if yoy is not None and yoy >= 0.8:   # guard against ordinary seasonal dips
        return None
    mine = hist.tail(6).sum()
    for other in siblings:
        h = complete_series(monthly, other, cutoff)
        if not len(h):
            continue
        t = trend_status(h)
        if t == "Rising" or (t == "Stable" and h.tail(6).sum() > mine):
            return other
    return None


def _phase_out_check(line: _Line, family: list[_Line], monthly, depot):
    """Phase-out is sticky: once the rule has fired in the last `lookback`
    months, the SKU stays phased out while it is still down year over year.
    (The Rising/Dying trend flips month to month on small counts, so a
    single-month test would switch a SKU on and off.)"""
    cyc = depot["cycling"]
    siblings = [l.sku for l in family if l.sku != line.sku]
    if not siblings:
        return None
    yoy_now = _yoy(line.fc.history)
    if yoy_now is not None and yoy_now >= 0.8:
        return None
    for k in reversed(range(cyc.get("lookback_months", 12))):  # earliest first
        when = line.cutoff - pd.DateOffset(months=k)
        other = _successor_at(line.sku, siblings, monthly, when, cyc["dying_runs"])
        if other:
            desc = next(l.p["description"] for l in family if l.sku == other)
            return desc, when
    return None


def _plan_presence(lines, depot, store, free_overflow, notes, monthly):
    sl = depot["service_level"]["luxury"]
    minimum = depot["min_display_units"]
    by_family: dict[str, list[_Line]] = {}
    for line in lines:
        by_family.setdefault(line.p["family"], []).append(line)

    for line in lines:
        phase_out = _phase_out_check(line, by_family[line.p["family"]], monthly, depot)
        if phase_out:
            successor, since = phase_out
            line.target, line.qty, line.stopped = 0, 0, True
            line.notes.append(f"PHASE OUT (since {since:%b %Y}): replaced by {successor}. "
                              "Do not reorder; keep 1 on display, move the rest to overflow "
                              "and mark down.")
            continue
        if line.fc.trend == "Dead":
            line.target, line.qty, line.stopped = 0, 0, True
            line.notes.append("no sales in 6 months: not reordered")
            continue
        line.order_up_to(sl, minimum=minimum)
        if line.on_hand == 0 and line.qty > 0:
            line.notes.append("none on display")

    _apply_overflow(lines, free_overflow, store["overflow_categories"])
    value = lambda ls: sum(l.qty * l.p["unit_cost"] for l in ls)
    cats = ["luxury"]
    keep_one = lambda l: max(0, minimum - l.position)
    if value(lines) > depot["insured_value_cap"]:
        _cut_to_fit(lines, depot["insured_value_cap"], value, cats, protect=keep_one)
        notes.append("Load trimmed to the insured value cap.")
    _cut_to_fit(lines, depot["truck_pallets"], lambda ls: sum(l.pallets() for l in ls), cats,
                protect=keep_one)
    return "Order"


def _plan_cost(lines, depot, store, free_overflow, notes, dates):
    sl = depot["service_level"]
    d1, d2, d3 = dates[:3]

    # Can we skip this truck? Stock must last until the next truck arrives.
    risks = [l._window(l.asof, d2).prob_exceeds(l.position) for l in lines]
    if all(r <= 1 - sl.get(l.category, 0.95) for r, l in zip(risks, lines)):
        for l in lines:
            l.set_cover(d2)
            l.notes.append("stock lasts to next truck")
        notes.append(f"Skip this truck: stock covers until the {d2:%b %d} delivery "
                     f"(saves ${depot['trip_cost']:,.0f}).")
        return "Skip"

    for line in lines:
        line.order_up_to(sl.get(line.category, 0.95), max_stock=line.p["shelf_max"])
    load = sum(l.pallets() for l in lines)
    if load < depot["min_economic_pallets"]:
        # Pull the next cycle forward only if every SKU can then last to the
        # truck after next within its shelf space, so next week's trip can be
        # skipped. Otherwise a bigger order would just crowd the shelves.
        targets = {l.sku: l._window(l.asof, d3).quantile(sl.get(l.category, 0.95)) for l in lines}
        if all(targets[l.sku] <= l.p["shelf_max"] + l.overflow + l.on_order for l in lines):
            for line in lines:
                line.set_cover(d3)
                line.order_up_to(sl.get(line.category, 0.95), max_stock=line.p["shelf_max"])
            new_load = sum(l.pallets() for l in lines)
            notes.append(f"Small load ({load:.2f} pallets): ordered two weeks at once "
                         f"({new_load:.2f} pallets) so the {d2:%b %d} truck can be skipped.")
        else:
            short = [l.p["description"] for l in lines if targets[l.sku] > l.p["shelf_max"]]
            notes.append(f"Small load ({load:.2f} pallets), but shelf space for "
                         f"{', '.join(short)} is too small to order two weeks at once.")
    _cut_to_fit(lines, depot["truck_pallets"], lambda ls: sum(l.pallets() for l in ls),
                list(sl.keys()))
    return "Order"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def recommend(depot_id: str, cutoff: pd.Timestamp, forecasts: dict[str, SkuForecast],
              products: pd.DataFrame, inventory: pd.DataFrame, monthly: pd.DataFrame,
              cfg: dict) -> DepotPlan:
    depot = cfg["depots"][depot_id]
    store = cfg["store"]
    conf = cfg["forecast"]["confidence"]
    cutoff = pd.Timestamp(cutoff)
    asof = cutoff + pd.DateOffset(months=1)
    dates = delivery_dates(depot, asof, 3)
    d1, d2 = dates[0], dates[1]

    notes: list[str] = []
    upp_all = products["case_pack"] * products["cases_per_pallet"]
    used = (inventory["overflow_units"] / upp_all.reindex(inventory.index)).sum()
    free_overflow = max(0.0, store["overflow_pallets"] - used)

    lines = []
    for sku, p in products[products["depot"] == depot_id].iterrows():
        if sku not in forecasts:
            notes.append(f"{p['description']}: no sales history yet, not planned.")
            continue
        lines.append(_Line(sku, p, inventory.loc[sku], forecasts[sku], cutoff, asof, d1, d2, conf))

    priority = depot["priority"]
    if priority == "service_level":
        decision = _plan_service_level(lines, depot, store, free_overflow, notes)
    elif priority == "presence":
        decision = _plan_presence(lines, depot, store, free_overflow, notes, monthly)
    elif priority == "cost":
        decision = _plan_cost(lines, depot, store, free_overflow, notes, dates)
    else:
        raise ValueError(f"unknown priority {priority!r}")

    table = pd.DataFrame([l.row() for l in lines])
    pallets = float(sum(l.pallets() for l in lines))
    value = float(sum(l.qty * l.p["unit_cost"] for l in lines))
    if decision == "Order" and pallets == 0:
        decision = "Nothing to order"
    trip_cost = depot["trip_cost"] if decision == "Order" else 0.0
    return DepotPlan(depot_id, depot["name"], asof, d1 - pd.Timedelta(days=depot["lead_time_days"]),
                     d1, d2, decision, table, pallets, depot["truck_pallets"], trip_cost,
                     value, notes)
