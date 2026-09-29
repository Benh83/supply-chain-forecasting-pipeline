"""Rolling-origin backtest.

For each cutoff, fit on history up to the cutoff and score the 1-3 month
forecasts against what actually sold. Reports MASE (scaled by the in-sample
seasonal-naive error), bias, and how often actuals fell inside the 90% interval.
A seasonal-naive forecast is scored alongside as a baseline.

    python -m forecasting.backtest --start 2022-03 --end 2025-06 --step 3
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from forecasting.data import complete_series
from forecasting.pipeline import forecast_sku, load_all


def mase_scale(history: pd.Series) -> float:
    lag = 12 if len(history) >= 24 else 1
    scale = history.diff(lag).abs().mean()
    return float(scale) if scale and np.isfinite(scale) and scale > 0 else float(history.mean() or 1)


def run(start: str, end: str, step: int, skus: list[str] | None = None) -> pd.DataFrame:
    cfg, monthly, border, products = load_all()
    skus = skus or list(products.index)
    last_month = monthly["yearmonth"].max()
    rows = []
    for cutoff in pd.date_range(start, end, freq=f"{step}MS"):
        for sku in skus:
            if complete_series(monthly, sku, cutoff).shape[0] < 12:
                continue
            fc = forecast_sku(sku, cutoff, cfg, monthly, border)
            actual = complete_series(monthly, sku, last_month, start=fc.history.index[0])
            scale = mase_scale(fc.history)
            for h in fc.horizons:
                month = cutoff + pd.DateOffset(months=h.horizon)
                if month > last_month:
                    continue
                y = actual[month]
                snaive = fc.history.get(month - pd.DateOffset(months=12), fc.history.iloc[-1])
                rows.append({
                    "cutoff": cutoff, "sku": sku, "horizon": h.horizon, "rule": fc.choice.kind,
                    "model": h.model, "actual": y, "forecast": h.forecast,
                    "lower": h.lower, "upper": h.upper,
                    "ase": abs(y - h.forecast) / scale,
                    "ase_snaive": abs(y - snaive) / scale,
                    "error": h.forecast - y,
                    "covered": h.lower <= y <= h.upper,
                })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    agg = dict(n=("ase", "size"), MASE=("ase", "mean"), MASE_seasonal_naive=("ase_snaive", "mean"),
               bias=("error", "mean"), coverage_90=("covered", "mean"))
    by_sku = bt.groupby("sku").agg(**agg)
    total = bt.agg({"ase": "mean", "ase_snaive": "mean", "error": "mean", "covered": "mean"})
    by_sku.loc["ALL"] = [len(bt), total["ase"], total["ase_snaive"], total["error"], total["covered"]]
    return by_sku.round(3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2022-03")
    parser.add_argument("--end", default="2025-06")
    parser.add_argument("--step", type=int, default=3, help="months between cutoffs")
    parser.add_argument("--sku", nargs="*")
    parser.add_argument("--out", default="backtest_results.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    bt = run(args.start, args.end, args.step, args.sku)
    bt.to_csv(args.out, index=False)
    print(summarize(bt).to_string())
    print(f"\nrow-level results saved to {args.out}")


if __name__ == "__main__":
    main()
