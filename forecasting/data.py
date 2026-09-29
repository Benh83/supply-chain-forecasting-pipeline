"""Data loading and cleaning."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BORDER_COLS = ["Bus Passengers", "Pedestrians", "Personal Vehicle Passengers"]


def clean_border(raw: pd.DataFrame, z_threshold: float = 10.0,
                 ratio_threshold: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag gross recording errors in the border series and interpolate over them.

    A month is flagged when it is BOTH a robust-z outlier (log ratio to a centred
    5-month rolling median, scaled by MAD) AND off by more than `ratio_threshold`x
    from that median. The two conditions together keep genuine shocks (the 2020
    closure was a ~60% drop) while removing recording errors (a month at <1%
    of its neighbours). Missing values are interpolated too.

    Returns the cleaned frame (indexed by month) and a frame of what was changed.
    """
    df = raw.copy()
    df["yearmonth"] = pd.to_datetime(df["yearmonth"])
    df = df.sort_values("yearmonth").set_index("yearmonth")
    df.index.freq = "MS"
    changes = []
    for col in BORDER_COLS:
        s = df[col].astype(float)
        for month in s.index[s.isna()]:
            changes.append({"yearmonth": month, "column": col, "original": np.nan,
                            "reason": "missing"})
        med = s.rolling(5, center=True, min_periods=3).median()
        log_ratio = np.log(s / med)
        centred = log_ratio - log_ratio.median()
        mad = centred.abs().median() * 1.4826
        z = centred / mad if mad > 0 else centred * 0
        gross = (z.abs() > z_threshold) & (log_ratio.abs() > np.log(ratio_threshold))
        for month in s.index[gross.fillna(False)]:
            changes.append({"yearmonth": month, "column": col, "original": s[month],
                            "reason": f"outlier (robust z={z[month]:.0f})"})
        s[gross.fillna(False)] = np.nan
        df[col] = s.interpolate(method="linear", limit_direction="both")
    changes_df = pd.DataFrame(changes, columns=["yearmonth", "column", "original", "reason"])
    for _, row in changes_df.iterrows():
        log.info("border %s %s: %s -> %.0f", row["column"], row["yearmonth"].date(),
                 row["reason"], df.loc[row["yearmonth"], row["column"]])
    return df, changes_df


def load_border(path: str) -> pd.DataFrame:
    cleaned, _ = clean_border(pd.read_csv(path))
    return cleaned


def load_products(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["launch_date"]).set_index("sku")


def load_sales(path: str) -> pd.DataFrame:
    sales = pd.read_csv(path, parse_dates=["date"])
    sales["yearmonth"] = sales["date"].dt.to_period("M").dt.to_timestamp()
    return sales


def monthly_units(sales: pd.DataFrame) -> pd.DataFrame:
    """SKU x month table of units (months with no rows are absent, not zero)."""
    return sales.groupby(["sku", "yearmonth"], as_index=False)["units"].sum()


def complete_series(monthly: pd.DataFrame, sku: str, end: pd.Timestamp,
                    start: pd.Timestamp | None = None) -> pd.Series:
    """Monthly units for one SKU from its first sale (or `start`) to `end`.

    Months without sales are filled with 0: a missing month means no sales,
    not missing data.
    """
    rows = monthly[(monthly["sku"] == sku) & (monthly["yearmonth"] <= end)]
    if start is None:
        if rows.empty:
            return pd.Series(dtype=float, name="units")
        start = rows["yearmonth"].min()
    idx = pd.date_range(start, end, freq="MS")
    series = rows.set_index("yearmonth")["units"].reindex(idx, fill_value=0).astype(float)
    series.index.freq = "MS"
    series.name = "units"
    return series
