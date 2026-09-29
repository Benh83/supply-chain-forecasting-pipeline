"""Model fitting and prediction intervals.

Every model returns one `HorizonForecast` per horizon. All intervals pass
through `_finalize`, so the forecast, lower and upper bounds are always
consistent (0 <= lower <= forecast <= upper <= cap).
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.diagnostics import Diagnostics
from forecasting.selection import ModelChoice

log = logging.getLogger(__name__)


@dataclass
class HorizonForecast:
    horizon: int
    forecast: float
    lower: float
    upper: float
    model: str
    note: str = ""


# ---------------------------------------------------------------------------
# Safety cap (original rule, unchanged in substance)
# ---------------------------------------------------------------------------

def safety_cap(series: pd.Series, choice: ModelChoice) -> float:
    """Upper limit on any forecast. December products get the permissive cap
    (largest candidate); everything else gets the conservative one (smallest)."""
    hist_max, hist_mean, hist_std = series.max(), series.mean(), series.std()
    hist_95 = series.quantile(0.95)
    recent = series.tail(12) if len(series) >= 12 else series.tail(6) if len(series) >= 6 else series
    rec_max, rec_mean, rec_std = recent.max(), recent.mean(), recent.std()
    hist_std = 0.0 if np.isnan(hist_std) else hist_std
    rec_std = 0.0 if np.isnan(rec_std) else rec_std

    if choice.december_product:
        b = choice.seasonal_boost
        candidates = [hist_max * 2.0 * b, rec_max * 3.0 * b, hist_mean + 5 * hist_std,
                      hist_95 * 2.0 * b, rec_mean + 6 * rec_std]
        cap = max([c for c in candidates if c > 0], default=0.0)
    else:
        candidates = [hist_max * 1.5, rec_max * 2.0, hist_mean + 3 * hist_std,
                      hist_95 * 1.3, rec_mean + 4 * rec_std]
        cap = min([c for c in candidates if c > 0], default=0.0)
        if hist_mean < 10:
            cap = min(cap, hist_max * 1.2, rec_max * 1.5)
        if (series == 0).mean() > 0.5:
            cap = min(cap, hist_max * 1.3, rec_max * 1.8)
        cap = min(cap, hist_max * 2.0)
    if cap <= 0:
        cap = max(10.0, hist_mean * 2)
    return float(cap)


def _finalize(h: int, f: float, lo: float, hi: float, cap: float, model: str,
              note: str = "") -> HorizonForecast:
    f = float(np.clip(f, 0, cap))
    lo = float(np.clip(lo, 0, f))
    hi = float(np.clip(hi, f, cap))
    if note:
        log.info("%s h=%d: %s", model, h, note)
    return HorizonForecast(h, f, lo, hi, model, note)


def _z(confidence: float) -> float:
    return float(stats.norm.ppf((1 + confidence) / 2))


# ---------------------------------------------------------------------------
# Simple models
# ---------------------------------------------------------------------------

def _sparse_recent(series, horizons, cap, conf):
    """Fewer than 3 non-zero months: mean of the last 4 months (zeros included),
    interval widened by 1.5x (original rule)."""
    recent = np.r_[np.zeros(max(0, 4 - len(series))), series.tail(4).to_numpy()]
    mean = recent.mean()
    std = np.std(recent, ddof=1) if len(set(recent)) > 1 else (mean * 0.5 if mean > 0 else 0.5)
    std *= 1.5
    z = _z(conf)
    return [_finalize(h, mean, mean - z * std, mean + z * std, cap, "Sparse_Recent")
            for h in horizons]


def _moving_avg(series, horizons, cap, conf, window=3):
    mean = series.tail(window).mean()
    std = series.std() if len(series) > 1 else mean * 0.5
    z = _z(conf)
    return [_finalize(h, mean, mean - z * std * np.sqrt(h), mean + z * std * np.sqrt(h),
                      cap, "MovingAvg") for h in horizons]


def croston_sba(series: pd.Series, alpha: float) -> tuple[float, np.ndarray]:
    """Croston's method with the Syntetos-Boylan bias correction.

    Returns the per-period demand rate and the in-sample one-step forecasts.
    """
    y = series.to_numpy(dtype=float)
    nz = np.flatnonzero(y > 0)
    size, interval = y[nz[0]], float(nz[0] + 1)
    fitted = np.full(len(y), np.nan)
    q = 1
    for t in range(nz[0] + 1, len(y)):
        fitted[t] = (1 - alpha / 2) * size / interval
        if y[t] > 0:
            size += alpha * (y[t] - size)
            interval += alpha * (q - interval)
            q = 1
        else:
            q += 1
    return (1 - alpha / 2) * size / interval, fitted


def _croston(series, horizons, cap, conf):
    """Intermittent demand. Smoothing constant chosen by in-sample MSE; interval
    from empirical residual quantiles, widened with horizon as for SES."""
    best = None
    for alpha in (0.05, 0.1, 0.2, 0.3):
        rate, fitted = croston_sba(series, alpha)
        resid = series.to_numpy() - fitted
        mse = np.nanmean(resid ** 2)
        if best is None or mse < best[0]:
            best = (mse, alpha, rate, resid[~np.isnan(resid)])
    _, alpha, rate, resid = best
    q_lo, q_hi = np.quantile(resid, [(1 - conf) / 2, (1 + conf) / 2]) if len(resid) >= 5 else (-rate, rate)
    out = []
    for h in horizons:
        widen = np.sqrt(1 + (h - 1) * alpha ** 2)
        out.append(_finalize(h, rate, rate + q_lo * widen, rate + q_hi * widen, cap,
                             "Croston_SBA", f"alpha={alpha}"))
    return out


def _ets(series, horizons, cap, conf):
    """ETS with estimated smoothing parameters (replaces the fixed alpha=0.3
    hand-rolled smoother). Chooses between A,N,N and A,Ad,N by AICc."""
    specs = [dict(trend=None)]
    if len(series) >= 12:
        specs.append(dict(trend="add", damped_trend=True))
    best = None
    for spec in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ETSModel(series, error="add", **spec).fit(disp=False)
        if best is None or res.aicc < best.aicc:
            best = res
    pred = best.get_prediction(start=len(series), end=len(series) + max(horizons) - 1)
    frame = pred.summary_frame(alpha=1 - conf)
    name = "ETS(A,Ad,N)" if best.model.trend else "ETS(A,N,N)"
    return [_finalize(h, frame["mean"].iloc[h - 1], frame["pi_lower"].iloc[h - 1],
                      frame["pi_upper"].iloc[h - 1], cap, name) for h in horizons]


# ---------------------------------------------------------------------------
# ARIMA family
# ---------------------------------------------------------------------------

def _fit_one(series, exog, order, s_order, trend):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            res = SARIMAX(series, exog=exog, order=order, seasonal_order=s_order, trend=trend,
                          enforce_stationarity=True, enforce_invertibility=True
                          ).fit(disp=False, maxiter=200)
    except (np.linalg.LinAlgError, ValueError) as exc:
        log.debug("ARIMA%s%s failed: %s", order, s_order, exc)
        return None
    return res if np.isfinite(res.aicc) else None


def fit_arima(series: pd.Series, d: int, D: int, seasonal: bool, cfg: dict,
              exog: np.ndarray | None = None):
    """Choose ARIMA orders by AICc in two steps: the non-seasonal (p, q) grid
    first, then seasonal (P, Q) terms on top of the best (p, q).

    Stationarity and invertibility are enforced, which removes most of the
    explosive forecasts the original safety checks were guarding against.
    Models with more than n/5 parameters are not considered.
    """
    g = cfg["forecast"]["arima_grid"]
    trend = "c" if d + D == 0 else "n"
    k_exog = exog.shape[1] if exog is not None else 0
    too_big = lambda k: k + k_exog + 2 > len(series) / 5
    best = None

    def consider(order, s_order):
        nonlocal best
        if too_big(order[0] + order[2] + s_order[0] + s_order[2]):
            return
        res = _fit_one(series, exog, order, s_order, trend)
        if res is not None and (best is None or res.aicc < best[0].aicc):
            best = (res, order, s_order)

    no_season = (0, D, 0, 12) if seasonal else (0, 0, 0, 0)
    for p, q in itertools.product(range(g["max_p"] + 1), range(g["max_q"] + 1)):
        if p + q <= g["max_pq"]:
            consider((p, d, q), no_season)
    if seasonal and best is not None:
        order = best[1]
        for P, Q in g["seasonal_PQ"]:
            if P or Q:
                consider(order, (P, D, Q, 12))
    if best is None:
        raise RuntimeError("no ARIMA candidate could be fitted")
    return best


def seasonal_naive_exog(border: pd.DataFrame, cutoff: pd.Timestamp, horizons_max: int,
                        cols: list[str]) -> pd.DataFrame:
    """Future border values = same month last year, scaled by the year-over-year
    change of the last 3 known months. Plain seasonal-naive badly under-predicts
    during a recovery (2022 vs 2021) or a slump; the scaling tracks the current
    level while keeping the monthly shape. Uses only data known at the cutoff."""
    future = pd.date_range(cutoff + pd.DateOffset(months=1), periods=horizons_max, freq="MS")
    last3 = pd.date_range(cutoff - pd.DateOffset(months=2), cutoff, freq="MS")
    now = border[cols].reindex(last3).sum()
    year_ago = border[cols].reindex(last3 - pd.DateOffset(months=12)).sum()
    growth = (now / year_ago).where(year_ago > 0, 1.0).fillna(1.0)
    vals = border[cols].reindex(future - pd.DateOffset(months=12)) * growth
    vals.index = future
    return vals


def _arima_family(series, horizons, cap, conf, choice: ModelChoice, diag: Diagnostics,
                  cfg, border: pd.DataFrame | None):
    min_seasonal = cfg["forecast"]["tests"]["min_obs_seasonal"]
    seasonal = choice.seasonal and len(series) >= min_seasonal
    notes = []
    if choice.seasonal and not seasonal:
        notes.append(f"history < {min_seasonal} months: seasonal terms dropped")
    D = diag.D if seasonal else 0
    d = diag.d

    exog = exog_future = None
    name = "ARIMA"
    if choice.kind == "sarimax_border" and border is not None and diag.border_feature:
        cols = [diag.border_feature]
        hist = border[cols].reindex(series.index)
        mu, sd = hist.mean(), hist.std().replace(0, 1)
        exog = ((hist - mu) / sd).to_numpy()
        fut = seasonal_naive_exog(border, series.index[-1], max(horizons), cols)
        exog_future = ((fut - mu) / sd).to_numpy()
        name = "ARIMAX"

    res, order, s_order = fit_arima(series, d, D, seasonal, cfg, exog)
    spec = f"{name}({order[0]},{order[1]},{order[2]})"
    if seasonal and any(s_order[:3]):
        spec = "S" + spec + f"({s_order[0]},{s_order[1]},{s_order[2]})12"
    if exog is not None:
        spec += f" + {diag.border_feature.lower()}"
    fc = res.get_forecast(steps=max(horizons), exog=exog_future)
    mean = fc.predicted_mean.to_numpy()
    ci = fc.conf_int(alpha=1 - conf).to_numpy()

    z = _z(conf)
    # Fallback level: the last 12 months, not the whole history. On a
    # growing or declining product the all-history mean is badly off.
    recent = series.tail(12)
    fb_mean, fb_std = recent.mean(), recent.std()
    out = []
    for h in horizons:
        m, lo, hi = mean[h - 1], ci[h - 1, 0], ci[h - 1, 1]
        note = "; ".join(notes)
        if not np.isfinite(m) or m > cap:
            # Original rule: explosive forecast -> fall back. Now logged.
            note = "; ".join(filter(None, [note, f"model gave {m:.0f} (above cap {cap:.0f}), "
                                                 "fell back to 12-month mean"]))
            m = fb_mean
            lo, hi = m - z * fb_std, m + z * fb_std
        elif hi - lo > 2 * cap:
            # Original rule: absurdly wide interval -> replaced. Now logged.
            note = "; ".join(filter(None, [note, "interval too wide, replaced by 12-month sd"]))
            lo, hi = m - z * fb_std, m + z * fb_std
        # A slightly negative mean on a near-zero series is simply clipped to 0
        # in _finalize rather than replaced by a historical average.
        out.append(_finalize(h, m, lo, hi, cap, spec, note))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def forecast_series(series: pd.Series, choice: ModelChoice, diag: Diagnostics, cfg: dict,
                    border: pd.DataFrame | None = None) -> list[HorizonForecast]:
    horizons = cfg["forecast"]["horizons"]
    conf = cfg["forecast"]["confidence"]
    dead = cfg["forecast"]["rules"]["dead_months"]

    if choice.kind == "zero" or (len(series) >= dead and series.tail(dead).sum() == 0):
        return [HorizonForecast(h, 0.0, 0.0, 0.0, "Dead_Product") for h in horizons]

    cap = safety_cap(series, choice)
    simple = {"sparse_recent": _sparse_recent, "moving_avg": _moving_avg,
              "croston": _croston, "ets": _ets}
    try:
        if choice.kind in simple:
            return simple[choice.kind](series, horizons, cap, conf)
        return _arima_family(series, horizons, cap, conf, choice, diag, cfg, border)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        log.warning("%s failed (%s); falling back to ETS", choice.kind, exc)
        try:
            out = _ets(series, horizons, cap, conf)
        except (ValueError, np.linalg.LinAlgError):
            out = _moving_avg(series, horizons, cap, conf)
        for f in out:
            f.note = f"{choice.kind} failed: {exc}"
            f.model = f"Fallback_{f.model}"
        return out
