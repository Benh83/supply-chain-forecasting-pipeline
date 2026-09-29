"""Series diagnostics used by model selection.

Each function returns plain numbers so the selection logic stays readable and
the values can be shown next to the forecast.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, grangercausalitytests, kpss

from forecasting.data import BORDER_COLS


@dataclass
class Diagnostics:
    n: int
    sparsity: float
    non_zero: int
    mean: float
    std: float
    cv: float
    # original rule inputs
    border_corr: float = 0.0
    border_feature: str | None = None
    border_p: float = 1.0
    granger_p: float | None = None
    autocorr: float = 0.0
    ljung_box_p: float = 1.0
    seasonal_strength: float = 0.0
    has_december: bool = False
    has_summer: bool = False
    decembers_seen: int = 0
    # added tests
    d: int = 0
    adf_p: float | None = None
    D: int = 0
    stl_seasonal_strength: float | None = None
    demand_class: str = "smooth"
    adi: float | None = None
    cv2: float | None = None
    notes: list[str] = field(default_factory=list)


def basic_stats(series: pd.Series) -> dict:
    n = len(series)
    zeros = int((series == 0).sum())
    mean = float(series.mean()) if n else 0.0
    std = float(series.std()) if n > 1 else 0.0
    return {
        "n": n,
        "sparsity": zeros / n if n else 1.0,
        "non_zero": n - zeros,
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else 0.0,
    }


def border_correlation(series: pd.Series,
                       border: pd.DataFrame | None) -> tuple[float, str | None, float, float | None]:
    """Strongest correlation between sales and a border series.

    Both series are first-differenced so that a shared trend or level shift
    (e.g. 2020) does not by itself produce a high correlation. The p-value is
    returned so the caller can require significance. A lag-1 Granger test is
    reported for the chosen series as supporting evidence.

    Returns (|corr|, feature, p_value, granger_p).
    """
    if border is None:
        return 0.0, None, 1.0, None
    df = pd.concat([series.rename("sales"), border[BORDER_COLS]], axis=1, join="inner").dropna()
    diff = df.diff().dropna()
    if len(diff) < 10 or diff["sales"].std() == 0:
        return 0.0, None, 1.0, None
    best = (0.0, None, 1.0)
    for col in BORDER_COLS:
        if diff[col].std() == 0:
            continue
        r, p = stats.pearsonr(diff["sales"], diff[col])
        if abs(r) > best[0]:
            best = (abs(float(r)), col, float(p))
    granger_p = None
    if best[1] is not None and len(diff) >= 15:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(diff[["sales", best[1]]], maxlag=1)
        granger_p = float(res[1][0]["ssr_ftest"][1])
    return best[0], best[1], best[2], granger_p


def autocorrelation(series: pd.Series, max_lag: int = 3) -> tuple[float, float]:
    """Largest |ACF| over lags 1..max_lag, and the Ljung-Box p-value at max_lag."""
    if len(series) <= max_lag + 5 or series.std() == 0:
        return 0.0, 1.0
    values = acf(series, nlags=max_lag, fft=False)[1:]
    lb = acorr_ljungbox(series, lags=[max_lag], return_df=True)
    return float(np.max(np.abs(values))), float(lb["lb_pvalue"].iloc[0])


def seasonality_strength(series: pd.Series) -> float:
    """Share of variance explained by month-of-year means (original rule input)."""
    if len(series) < 12:
        return 0.0
    df = pd.DataFrame({"sales": series.values, "month": series.index.month})
    total = df["sales"].var()
    if not total or np.isnan(total):
        return 0.0
    return float(df.groupby("month")["sales"].mean().var() / total)


def detect_strong_seasonality(series: pd.Series) -> tuple[bool, bool, int]:
    """Original December/Summer rules, unchanged. Also returns how many
    Decembers the December call was based on."""
    if len(series) < 12:
        return False, False, 0
    mean = series.mean()
    if mean == 0:
        return False, False, 0

    dec = series[series.index.month == 12]
    strong_dec = False
    if len(dec) > 0:
        if (dec.iloc[-1] >= mean * 1.5
                or (dec.max() == series.max() and dec.max() > mean * 1.2)
                or dec.mean() >= mean * 1.3):
            strong_dec = True

    summer = series[series.index.month.isin([6, 7, 8])]
    strong_summer = False
    if len(summer) > 0:
        if (summer.iloc[-1] >= mean * 1.4
                or (summer.max() == series.max() and summer.max() > mean * 1.15)
                or summer.mean() >= mean * 1.3):
            strong_summer = True
    return strong_dec, strong_summer, len(dec)


def n_diffs(series: pd.Series, alpha: float = 0.05, max_d: int = 1) -> tuple[int, float | None]:
    """Order of differencing, decided by KPSS (null: stationary), the
    auto.arima convention. The ADF p-value (null: unit root) on the undifferenced
    series is returned alongside so disagreements between the two are visible.
    """
    x = series.astype(float)
    if len(x) <= 12 or x.std() == 0:
        return 0, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = float(adfuller(x, autolag="AIC")[1])
    d = 0
    while d < max_d and len(x) > 12 and x.std() > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p = kpss(x, regression="c", nlags="auto")[1]
        if kpss_p >= alpha:
            break
        x = x.diff().dropna()
        d += 1
    return d, adf_p


def stl_seasonal_strength(series: pd.Series) -> float | None:
    """Wang-Smith-Hyndman seasonal strength F_s = max(0, 1 - Var(R)/Var(S+R))."""
    if len(series) < 24 or series.std() == 0:
        return None
    res = STL(series, period=12, robust=True).fit()
    denom = np.var(res.seasonal + res.resid)
    return float(max(0.0, 1 - np.var(res.resid) / denom)) if denom > 0 else 0.0


def demand_class(series: pd.Series) -> tuple[str, float | None, float | None]:
    """Syntetos-Boylan-Croston classification (ADI cut 1.32, CV^2 cut 0.49)."""
    nz = series[series > 0]
    if len(nz) < 2:
        return "lumpy", None, None
    positions = np.flatnonzero(series.to_numpy() > 0)
    adi = float(len(series) / len(nz)) if len(positions) < 2 else float(
        np.mean(np.diff(np.r_[-1, positions])))
    cv2 = float((nz.std() / nz.mean()) ** 2) if nz.mean() > 0 else 0.0
    if adi < 1.32:
        cls = "smooth" if cv2 < 0.49 else "erratic"
    else:
        cls = "intermittent" if cv2 < 0.49 else "lumpy"
    return cls, adi, cv2


def diagnose(series: pd.Series, border: pd.DataFrame | None, cfg: dict) -> Diagnostics:
    tests = cfg["forecast"]["tests"]
    diag = Diagnostics(**basic_stats(series))
    if diag.n == 0 or series.sum() == 0:
        return diag
    diag.border_corr, diag.border_feature, diag.border_p, diag.granger_p = border_correlation(
        series, border)
    diag.autocorr, diag.ljung_box_p = autocorrelation(series, tests["max_acf_lag"])
    diag.seasonal_strength = seasonality_strength(series)
    diag.has_december, diag.has_summer, diag.decembers_seen = detect_strong_seasonality(series)
    diag.stl_seasonal_strength = stl_seasonal_strength(series)
    diag.D = int(diag.stl_seasonal_strength is not None
                 and diag.stl_seasonal_strength >= tests["seasonal_diff_strength"])
    base = series.diff(12).dropna() if diag.D else series
    diag.d, diag.adf_p = n_diffs(base, tests["alpha"], tests["max_d"])
    diag.demand_class, diag.adi, diag.cv2 = demand_class(series)
    return diag


def trend_status(series: pd.Series, window: int = 6) -> str:
    """Original Rising / Stable / Dying / Dead classification, unchanged."""
    if len(series) < window:
        return "Insufficient Data"
    recent = series.tail(window)
    if recent.sum() == 0:
        return "Dead"
    slope = stats.linregress(np.arange(window), recent.to_numpy()).slope
    early, late = recent.iloc[:3].mean(), recent.iloc[-3:].mean()
    if early == 0:
        pct = 0.0 if late == 0 else 100.0
    else:
        pct = (late - early) / early * 100
    if pct > 10 and slope > 0:
        return "Rising"
    if pct < -10 and slope < 0:
        return "Dying"
    return "Stable"
