"""Run forecasts for one or many SKUs at a cutoff month.

    python -m forecasting.pipeline --cutoff 2024-06
    python -m forecasting.pipeline --cutoff 2024-06 --depot LUX --out forecasts.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from forecasting.config import load_config
from forecasting.data import (complete_series, load_border, load_products, load_sales,
                              monthly_units)
from forecasting.diagnostics import Diagnostics, diagnose, trend_status
from forecasting.models import HorizonForecast, forecast_series
from forecasting.selection import ModelChoice, select_model

log = logging.getLogger(__name__)


@dataclass
class SkuForecast:
    sku: str
    cutoff: pd.Timestamp
    history: pd.Series
    diagnostics: Diagnostics
    choice: ModelChoice
    trend: str
    horizons: list[HorizonForecast]

    def to_row(self) -> dict:
        d = self.diagnostics
        row = {
            "sku": self.sku,
            "cutoff": self.cutoff.strftime("%Y-%m"),
            "trend_status": self.trend,
            "avg_monthly_sales": round(d.mean, 2),
            "sales_volatility": round(d.cv, 3),
            "total_last_6_months": int(self.history.tail(6).sum()),
            "non_zero_months": d.non_zero,
            "sparsity": round(d.sparsity, 3),
            "rule": self.choice.kind,
            "rule_reasons": "; ".join(self.choice.reasons),
            "forecast_model": self.horizons[0].model,
            "border_corr": round(d.border_corr, 3),
            "border_feature": d.border_feature,
            "border_p": round(d.border_p, 4),
            "granger_p": None if d.granger_p is None else round(d.granger_p, 4),
            "autocorr": round(d.autocorr, 3),
            "ljung_box_p": round(d.ljung_box_p, 4),
            "seasonal_strength": round(d.seasonal_strength, 3),
            "seasonal_evidence_decembers": d.decembers_seen,
            "d": d.d,
            "D": d.D,
            "demand_class": d.demand_class,
        }
        for f in self.horizons:
            month = (self.cutoff + pd.DateOffset(months=f.horizon)).strftime("%Y-%m")
            row[f"month_{f.horizon}_date"] = month
            row[f"month_{f.horizon}_forecast"] = round(f.forecast, 2)
            row[f"month_{f.horizon}_lower"] = round(f.lower, 2)
            row[f"month_{f.horizon}_upper"] = round(f.upper, 2)
            row[f"month_{f.horizon}_note"] = f.note
        return row


@lru_cache(maxsize=1)
def load_all(config_path: str | None = None):
    cfg = load_config(config_path) if config_path else load_config()
    sales = load_sales(cfg["paths"]["sales"])
    return cfg, monthly_units(sales), load_border(cfg["paths"]["border"]), \
        load_products(cfg["paths"]["products"])


def forecast_sku(sku: str, cutoff: str | pd.Timestamp, cfg: dict, monthly: pd.DataFrame,
                 border: pd.DataFrame) -> SkuForecast:
    cutoff = pd.Timestamp(cutoff).to_period("M").to_timestamp()
    series = complete_series(monthly, sku, cutoff)
    border_hist = border.reindex(series.index)
    has_border = not border_hist.isna().any().any()
    diag = diagnose(series, border_hist if has_border else None, cfg)
    choice = select_model(diag, has_border, cfg)
    horizons = forecast_series(series, choice, diag, cfg, border if has_border else None)
    trend = trend_status(series, cfg["forecast"]["trend_window"])
    return SkuForecast(sku, cutoff, series, diag, choice, trend, horizons)


def forecast_many(skus: list[str], cutoff, cfg, monthly, border) -> list[SkuForecast]:
    out = []
    for sku in skus:
        if complete_series(monthly, sku, pd.Timestamp(cutoff)).empty:
            log.info("%s has no sales before %s; skipped", sku, cutoff)
            continue
        out.append(forecast_sku(sku, cutoff, cfg, monthly, border))
    return out


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", required=True, help="last month of history, YYYY-MM")
    parser.add_argument("--depot", help="only SKUs supplied by this depot")
    parser.add_argument("--sku", nargs="*", help="only these SKUs")
    parser.add_argument("--out", help="write results to this CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg, monthly, border, products = load_all()
    skus = args.sku or list(products.index)
    if args.depot:
        skus = [s for s in skus if products.loc[s, "depot"] == args.depot]
    results = pd.DataFrame([f.to_row() for f in forecast_many(skus, args.cutoff, cfg,
                                                               monthly, border)])
    results.insert(1, "description", results["sku"].map(products["description"]))
    cols = ["sku", "description", "trend_status", "forecast_model",
            "month_1_forecast", "month_2_forecast", "month_3_forecast"]
    print(results[cols].to_string(index=False))
    if args.out:
        results.to_csv(args.out, index=False)
        print(f"\nsaved {len(results)} rows to {args.out}")
    return results


if __name__ == "__main__":
    main()
