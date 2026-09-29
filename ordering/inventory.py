"""Store inventory at the start of the forecast month.

If data/inventory_snapshot.csv exists (columns: asof, sku, on_hand,
overflow_units, on_order) and has rows for the date, it is used. Otherwise a
synthetic snapshot is made: stock equal to 40-130% of one replenishment cycle
(lead time + days between deliveries) at last month's sales rate. It is seeded
by SKU and month, so a selection always shows the same inventory.
"""

from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd


def cycle_days(depot: dict) -> float:
    if "delivery_weekdays" in depot:
        review = 7 / len(depot["delivery_weekdays"])
    else:
        review = 7 * depot["delivery_every_n_weeks"]
    return depot["lead_time_days"] + review


def synthetic_inventory(products: pd.DataFrame, monthly: pd.DataFrame,
                        cutoff: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    overflow_cats = cfg["store"]["overflow_categories"]
    rows = []
    for sku, p in products.iterrows():
        rng = np.random.default_rng(zlib.crc32(f"{sku}|{cutoff:%Y-%m}".encode()))
        sold = monthly.loc[(monthly["sku"] == sku) & (monthly["yearmonth"] == cutoff), "units"].sum()
        rate = sold / cutoff.days_in_month
        cover = cycle_days(cfg["depots"][p["depot"]]) * rng.uniform(0.4, 1.3)
        if not pd.isna(p["shelf_life_days"]):
            cover = min(cover, float(p["shelf_life_days"]) * 0.6)
        on_hand = int(min(rng.poisson(rate * cover), p["shelf_max"]))
        overflow = 0
        if p["category"] in overflow_cats and sold > 0:
            unit = 1 if p["category"] == "luxury" else int(p["case_pack"])
            overflow = int(rng.integers(0, 3)) * unit
        rows.append({"sku": sku, "on_hand": on_hand, "overflow_units": overflow, "on_order": 0})
    return pd.DataFrame(rows).set_index("sku")


def load_inventory(path: str, products: pd.DataFrame, monthly: pd.DataFrame,
                   cutoff: pd.Timestamp, cfg: dict) -> tuple[pd.DataFrame, bool]:
    """Returns (inventory, is_synthetic)."""
    asof = cutoff + pd.DateOffset(months=1)
    if os.path.exists(path):
        snap = pd.read_csv(path, parse_dates=["asof"])
        snap = snap[snap["asof"] == asof]
        if not snap.empty:
            return snap.set_index("sku")[["on_hand", "overflow_units", "on_order"]], False
    return synthetic_inventory(products, monthly, cutoff, cfg), True
