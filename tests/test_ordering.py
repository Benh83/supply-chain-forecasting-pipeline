import pandas as pd
import pytest

from forecasting.config import load_config
from forecasting.diagnostics import Diagnostics
from forecasting.models import HorizonForecast
from forecasting.pipeline import SkuForecast
from forecasting.selection import ModelChoice
from ordering import recommend
from ordering.demand import WindowDemand, window_demand
from ordering.schedule import delivery_dates

CFG = load_config()
CUTOFF = pd.Timestamp("2024-05-01")


def make_forecast(sku, monthly_units, trend="Stable", history=None, spread=0.2):
    hs = [HorizonForecast(h, monthly_units, monthly_units * (1 - spread), monthly_units * (1 + spread), "test")
          for h in (1, 2, 3)]
    idx = pd.date_range("2022-01-01", CUTOFF, freq="MS")
    hist = history if history is not None else pd.Series(monthly_units, index=idx, dtype=float)
    diag = Diagnostics(n=len(hist), sparsity=0, non_zero=len(hist), mean=monthly_units, std=1, cv=0.1)
    return SkuForecast(sku, CUTOFF, hist, diag, ModelChoice("ets"), trend, hs)


def product(sku, category, depot, family="f", pack=1, per_pallet=100, shelf=100, cooler=0,
            life=None, cost=10.0, price=20.0):
    return dict(sku=sku, description=sku, category=category, family=family, depot=depot,
                case_pack=pack, cases_per_pallet=per_pallet, unit_cost=cost, unit_price=price,
                shelf_max=shelf, needs_cooler=cooler, shelf_life_days=life,
                launch_date=pd.Timestamp("2019-01-01"))


def run(depot, prods, forecasts, stock, monthly=None):
    products = pd.DataFrame(prods).set_index("sku")
    inv = pd.DataFrame([{"sku": s, "on_hand": q, "overflow_units": 0, "on_order": 0}
                        for s, q in stock.items()]).set_index("sku")
    monthly = monthly if monthly is not None else pd.DataFrame(columns=["sku", "yearmonth", "units"])
    return recommend(depot, CUTOFF, forecasts, products, inv, monthly, CFG)


def test_delivery_dates_respect_lead_time_and_weekdays():
    dates = delivery_dates(CFG["depots"]["BEV_PER"], pd.Timestamp("2024-06-01"), 3)  # a Saturday
    assert [d.dayofweek for d in dates] == [1, 4, 1]
    lux = delivery_dates(CFG["depots"]["LUX"], pd.Timestamp("2024-06-01"), 2)
    assert (lux[1] - lux[0]).days == 14 and lux[0] >= pd.Timestamp("2024-06-05")


def test_window_demand_spans_months():
    fc = make_forecast("A", 310).horizons
    wd = window_demand(fc, CUTOFF, pd.Timestamp("2024-06-25"), pd.Timestamp("2024-07-05"), 0.9)
    assert wd.mean == pytest.approx(310 / 30 * 6 + 310 / 31 * 4)


def test_nb_distribution_quantiles():
    wd = WindowDemand(10, 5)
    assert wd.quantile(0.5) <= wd.quantile(0.9) <= wd.quantile(0.99)
    assert 0 < wd.prob_exceeds(10) < 1
    assert wd.expected_leftover(30) > 19


def test_alcohol_gets_higher_service_than_water():
    prods = [product("ALC", "alcohol", "BEV_PER", shelf=1000), product("WTR", "nonalc_beverage", "BEV_PER", shelf=1000)]
    fcs = {"ALC": make_forecast("ALC", 300), "WTR": make_forecast("WTR", 300)}
    plan = run("BEV_PER", prods, fcs, {"ALC": 0, "WTR": 0})
    t = plan.lines.set_index("sku")["target"]
    assert t["ALC"] > t["WTR"]


def test_perishable_order_limited_by_shelf_life():
    prods = [product("FRT", "perishable", "BEV_PER", life=2, cost=3, price=4, shelf=1000)]
    plan = run("BEV_PER", prods, {"FRT": make_forecast("FRT", 300, spread=0.6)}, {"FRT": 0})
    line = plan.lines.iloc[0]
    assert "shelf life" in line["note"]


def test_truck_capacity_respected():
    prods = [product(f"W{i}", "nonalc_beverage", "BEV_PER", per_pallet=1, shelf=10_000) for i in range(4)]
    fcs = {p["sku"]: make_forecast(p["sku"], 300) for p in prods}
    plan = run("BEV_PER", prods, fcs, {p["sku"]: 0 for p in prods})
    assert plan.pallets <= CFG["depots"]["BEV_PER"]["truck_pallets"] + 1e-9


def test_luxury_keeps_one_of_each():
    prods = [product("LUX1", "luxury", "LUX")]
    plan = run("LUX", prods, {"LUX1": make_forecast("LUX1", 0.1)}, {"LUX1": 0})
    assert plan.lines.iloc[0]["order_units"] >= 1


def test_luxury_phase_out_when_sibling_takes_over():
    idx = pd.date_range("2022-01-01", CUTOFF, freq="MS")
    old = pd.Series([30.0] * 12 + [30 * 0.85 ** k for k in range(1, len(idx) - 11)], index=idx)
    new = pd.Series([0.0] * 12 + [3.0 * k for k in range(1, len(idx) - 11)], index=idx)
    monthly = pd.concat([pd.DataFrame({"sku": s, "yearmonth": idx, "units": v.values})
                         for s, v in {"OLD": old, "NEW": new}.items()])
    prods = [product("OLD", "luxury", "LUX", family="sun"), product("NEW", "luxury", "LUX", family="sun")]
    fcs = {"OLD": make_forecast("OLD", old.iloc[-1], trend="Dying", history=old),
           "NEW": make_forecast("NEW", new.iloc[-1], trend="Rising", history=new)}
    plan = run("LUX", prods, fcs, {"OLD": 5, "NEW": 2}, monthly)
    lines = plan.lines.set_index("sku")
    assert lines.loc["OLD", "order_units"] == 0
    assert "PHASE OUT" in lines.loc["OLD", "note"]


def test_cost_depot_skips_truck_when_stock_lasts():
    prods = [product("CIG", "tobacco", "DRY", shelf=10_000)]
    plan = run("DRY", prods, {"CIG": make_forecast("CIG", 30)}, {"CIG": 500})
    assert plan.decision == "Skip"
    assert plan.trip_cost == 0
