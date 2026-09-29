"""Generate synthetic daily sales for one duty-free store (df_store_1).

Demand for each SKU is driven by one monthly border-crossing series (with an
elasticity), its own month-of-year pattern, a lifecycle/trend curve, weekday
effects and autocorrelated monthly noise. Output is written as transaction-style
rows (days with zero sales are omitted, as in a real POS extract).

    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.data import load_border  # noqa: E402

SEED = 20190101
START, END = "2019-01-01", "2025-09-30"
WEEKDAY = np.array([0.85, 0.80, 0.85, 0.95, 1.20, 1.30, 1.05])  # Mon..Sun
WEEKDAY = WEEKDAY / WEEKDAY.mean()

BUS, PED, PV = "Bus Passengers", "Pedestrians", "Personal Vehicle Passengers"

# month factors: {month: multiplier}; unspecified months = 1.0
PRODUCTS = [
    # sku, description, category, family, depot, driver, elasticity, base/day,
    # months, yearly trend, dispersion (NB size; None = Poisson), master data
    dict(sku="BEV-TEQ-01", description="Premium Tequila 750ml", category="alcohol",
         family="spirits", depot="BEV_PER", driver=PV, elasticity=1.0, base=7.0,
         months={7: 1.15, 11: 1.1, 12: 1.5}, trend=0.03, size=25,
         case_pack=6, cases_per_pallet=60, unit_cost=22.0, unit_price=38.0,
         shelf_max=72, needs_cooler=0, shelf_life_days=None),
    dict(sku="BEV-WHS-01", description="Blended Whisky 1L", category="alcohol",
         family="spirits", depot="BEV_PER", driver=PV, elasticity=0.9, base=3.5,
         months={6: 1.2, 11: 1.2, 12: 1.9}, trend=0.0, size=20,
         case_pack=6, cases_per_pallet=50, unit_cost=18.0, unit_price=32.0,
         shelf_max=48, needs_cooler=0, shelf_life_days=None),
    dict(sku="BEV-BER-12", description="Imported Beer 12-pack", category="alcohol",
         family="beer", depot="BEV_PER", driver=PV, elasticity=1.1, base=9.0,
         months={6: 1.25, 7: 1.4, 8: 1.3, 12: 1.1}, trend=0.01, size=25,
         case_pack=2, cases_per_pallet=80, unit_cost=11.0, unit_price=18.0,
         shelf_max=60, needs_cooler=1, shelf_life_days=None),
    dict(sku="BEV-WTR-01", description="Bottled Water 1L", category="nonalc_beverage",
         family="water", depot="BEV_PER", driver=PED, elasticity=1.0, base=14.0,
         months={1: 0.75, 2: 0.8, 4: 1.1, 5: 1.3, 6: 1.45, 7: 1.5, 8: 1.45, 9: 1.2, 12: 0.8},
         trend=0.0, size=30, case_pack=12, cases_per_pallet=84, unit_cost=0.5,
         unit_price=1.5, shelf_max=240, needs_cooler=1, shelf_life_days=None),
    dict(sku="BEV-NRG-01", description="Energy Drink 16oz", category="nonalc_beverage",
         family="energy", depot="BEV_PER", driver=PV, elasticity=0.8, base=6.0,
         months={}, trend=0.18, size=30, case_pack=24, cases_per_pallet=90,
         unit_cost=1.1, unit_price=2.9, shelf_max=144, needs_cooler=1, shelf_life_days=None),
    dict(sku="PER-CHO-01", description="Boxed Chocolates", category="perishable",
         family="confection", depot="BEV_PER", driver=PED, elasticity=0.9, base=4.0,
         months={2: 1.5, 5: 1.2, 11: 1.2, 12: 2.3}, trend=0.0, size=20,
         case_pack=12, cases_per_pallet=100, unit_cost=6.0, unit_price=14.0,
         shelf_max=60, needs_cooler=0, shelf_life_days=180),
    dict(sku="PER-FRT-01", description="Fresh Fruit Cup", category="perishable",
         family="fresh", depot="BEV_PER", driver=PED, elasticity=1.0, base=6.0,
         months={5: 1.2, 6: 1.4, 7: 1.4, 8: 1.3, 12: 0.85, 1: 0.8}, trend=0.02, size=15,
         case_pack=12, cases_per_pallet=70, unit_cost=1.2, unit_price=3.5,
         shelf_max=60, needs_cooler=1, shelf_life_days=5),
    dict(sku="LUX-FRG-A", description="Fragrance A 100ml", category="luxury",
         family="fragrance", depot="LUX", driver=BUS, elasticity=1.0, base=1.3,
         months={5: 1.3, 11: 1.2, 12: 1.9}, trend=0.0, size=None,
         case_pack=1, cases_per_pallet=300, unit_cost=55.0, unit_price=98.0,
         shelf_max=40, needs_cooler=0, shelf_life_days=None),
    dict(sku="LUX-FRG-B", description="Fragrance B 50ml", category="luxury",
         family="fragrance", depot="LUX", driver=BUS, elasticity=1.0, base=0.9,
         months={2: 1.3, 5: 1.7, 12: 1.3}, trend=0.02, size=None,
         case_pack=1, cases_per_pallet=400, unit_cost=48.0, unit_price=85.0,
         shelf_max=30, needs_cooler=0, shelf_life_days=None),
    dict(sku="LUX-SUN-X", description="Sunglasses Model X", category="luxury",
         family="sunglasses", depot="LUX", driver=BUS, elasticity=0.7, base=1.1,
         months={3: 1.3, 4: 1.3, 6: 1.3, 7: 1.4, 8: 1.3}, trend=0.0, size=None,
         lifecycle=("decline", "2022-10-01", 0.075),
         case_pack=1, cases_per_pallet=200, unit_cost=70.0, unit_price=145.0,
         shelf_max=24, needs_cooler=0, shelf_life_days=None),
    dict(sku="LUX-SUN-Y", description="Sunglasses Model Y", category="luxury",
         family="sunglasses", depot="LUX", driver=BUS, elasticity=0.7, base=1.2,
         months={3: 1.3, 4: 1.3, 6: 1.3, 7: 1.4, 8: 1.3}, trend=0.0, size=None,
         lifecycle=("launch", "2023-03-01", 6.0),
         case_pack=1, cases_per_pallet=200, unit_cost=75.0, unit_price=155.0,
         shelf_max=24, needs_cooler=0, shelf_life_days=None),
    dict(sku="LUX-WAL-01", description="Designer Leather Wallet", category="luxury",
         family="leather", depot="LUX", driver=BUS, elasticity=0.5, base=0.035,
         months={5: 1.5, 12: 2.0}, trend=0.0, size=None,
         case_pack=1, cases_per_pallet=150, unit_cost=160.0, unit_price=320.0,
         shelf_max=4, needs_cooler=0, shelf_life_days=None),
    dict(sku="DRY-CIG-10", description="Cigarettes Carton", category="tobacco",
         family="tobacco", depot="DRY", driver=PED, elasticity=1.0, base=18.0,
         months={}, trend=-0.04, size=40, case_pack=50, cases_per_pallet=40,
         unit_cost=28.0, unit_price=42.0, shelf_max=450, needs_cooler=0, shelf_life_days=None),
    dict(sku="DRY-CND-01", description="Travel Candy Tin", category="dry",
         family="candy", depot="DRY", driver=PV, elasticity=0.9, base=5.0,
         months={10: 1.3, 12: 1.3}, trend=0.0, size=20, case_pack=24, cases_per_pallet=60,
         unit_cost=2.0, unit_price=5.5, shelf_max=168, needs_cooler=0, shelf_life_days=None),
    dict(sku="DRY-TRV-01", description="Travel Neck Pillow", category="dry",
         family="travel", depot="DRY", driver=BUS, elasticity=1.2, base=1.6,
         months={6: 1.3, 7: 1.3, 12: 1.2}, trend=0.0, size=None, case_pack=12,
         cases_per_pallet=20, unit_cost=7.0, unit_price=19.0, shelf_max=84,
         needs_cooler=0, shelf_life_days=None),
    # Not border-driven: bought mostly by store staff and locals
    dict(sku="DRY-USB-01", description="USB Charging Cable", category="dry",
         family="electronics", depot="DRY", driver=None, elasticity=0.0, base=3.0,
         months={}, trend=0.05, size=None, case_pack=20, cases_per_pallet=80,
         unit_cost=3.0, unit_price=12.0, shelf_max=120, needs_cooler=0, shelf_life_days=None),
]


def lifecycle_factor(days: pd.DatetimeIndex, spec: tuple | None) -> np.ndarray:
    if spec is None:
        return np.ones(len(days))
    kind, start, param = spec
    t = (days - pd.Timestamp(start)).days.to_numpy() / 30.44  # months since start
    if kind == "decline":  # flat, then exponential decay to `param` after 32 months
        rate = np.log(param) / 32
        return np.where(t <= 0, 1.0, np.exp(rate * t))
    if kind == "launch":   # zero before launch, logistic ramp with midpoint `param` months
        return np.where(t < 0, 0.0, 1 / (1 + np.exp(-(t - param) / 1.8)))
    raise ValueError(kind)


def main() -> None:
    rng = np.random.default_rng(SEED)
    border = load_border(str(ROOT / "data" / "border_crossings_monthly.csv"))
    days = pd.date_range(START, END, freq="D")
    months = days.to_period("M").to_timestamp()
    ref = border.loc["2019"].mean()

    frames = []
    for p in PRODUCTS:
        if p["driver"] is None:
            driver = np.ones(len(days))
        else:
            driver = (border[p["driver"]] / ref[p["driver"]]).reindex(months).to_numpy()
        month_idx = pd.date_range(START, END, freq="MS")
        # AR(1) monthly noise gives realistic month-to-month persistence
        noise = np.zeros(len(month_idx))
        for i in range(1, len(noise)):
            noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 0.07)
        noise_daily = pd.Series(noise, index=month_idx).reindex(months).to_numpy()
        season = np.array([p["months"].get(m, 1.0) for m in days.month])
        years = (days - pd.Timestamp(START)).days.to_numpy() / 365.25
        rate = (p["base"] * driver ** p["elasticity"] * season
                * (1 + p["trend"]) ** years
                * lifecycle_factor(days, p.get("lifecycle"))
                * WEEKDAY[days.dayofweek] * np.exp(noise_daily))
        if p["size"] is None:
            units = rng.poisson(rate)
        else:
            n = p["size"]
            units = rng.negative_binomial(n, n / (n + rate))
        df = pd.DataFrame({"date": days, "sku": p["sku"], "units": units})
        frames.append(df[df["units"] > 0])

    sales = pd.concat(frames).sort_values(["date", "sku"])
    master = pd.DataFrame(PRODUCTS)
    sales = sales.merge(master[["sku", "description", "category", "depot"]], on="sku")
    sales.insert(1, "store", "df_store_1")
    sales = sales[["date", "store", "sku", "description", "category", "depot", "units"]]
    sales.to_csv(ROOT / "data" / "df_store_1_daily_sales.csv", index=False)

    launch = {p["sku"]: p["lifecycle"][1] for p in PRODUCTS
              if p.get("lifecycle") and p["lifecycle"][0] == "launch"}
    master["launch_date"] = master["sku"].map(launch).fillna(START)
    cols = ["sku", "description", "category", "family", "depot", "case_pack",
            "cases_per_pallet", "unit_cost", "unit_price", "shelf_max", "needs_cooler",
            "shelf_life_days", "launch_date"]
    master[cols].to_csv(ROOT / "data" / "products.csv", index=False)
    print(f"wrote {len(sales):,} sales rows for {len(PRODUCTS)} SKUs")


if __name__ == "__main__":
    main()
