"""Demand over an arbitrary window of days, derived from monthly forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from forecasting.models import HorizonForecast


@dataclass
class WindowDemand:
    mean: float
    sd: float

    def _dist(self):
        """Negative binomial matched to mean and variance (Poisson if the
        variance is not above the mean). Counts are whole units, which matters
        for slow luxury items where a normal approximation is poor."""
        mu = max(self.mean, 1e-9)
        var = self.sd ** 2
        if var > mu * 1.0001:
            n = mu ** 2 / (var - mu)
            return stats.nbinom(n, n / (n + mu))
        return stats.poisson(mu)

    def quantile(self, p: float) -> int:
        return int(self._dist().ppf(p)) if self.mean > 0 else 0

    def prob_exceeds(self, stock: float) -> float:
        """P(demand > stock): chance of running out within the window."""
        if self.mean <= 0:
            return 0.0
        return float(self._dist().sf(np.floor(stock)))

    def expected_leftover(self, stock: float) -> float:
        """E[(stock - demand)+]."""
        if stock <= 0:
            return 0.0
        k = np.arange(0, int(stock) + 1)
        return float(np.sum((stock - k) * self._dist().pmf(k)))


def monthly_sd(f: HorizonForecast, confidence: float) -> float:
    """Back out a standard deviation from the forecast's prediction interval."""
    z = stats.norm.ppf((1 + confidence) / 2)
    return max((f.upper - f.lower) / (2 * z), 0.0)


def window_demand(forecasts: list[HorizonForecast], cutoff: pd.Timestamp,
                  start: pd.Timestamp, end: pd.Timestamp, confidence: float) -> WindowDemand:
    """Demand from `start` (inclusive) to `end` (exclusive).

    Each month's forecast is spread evenly over its days. For a fraction f of a
    month the mean is f*mu. The variance has two parts:
      - forecast (level) error, which scales with the window: (f*sigma)^2
      - day-to-day sales randomness, taken as Poisson: f*mu
    Days beyond the last forecast month reuse the last month.
    """
    by_month = {cutoff + pd.DateOffset(months=f.horizon): f for f in forecasts}
    last = forecasts[-1]
    mean = var = 0.0
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    counts = pd.Series(1, index=days).groupby(days.to_period("M").to_timestamp()).sum()
    for month, n_days in counts.items():
        f = by_month.get(month, last)
        frac = n_days / month.days_in_month
        mean += f.forecast * frac
        var += (frac * monthly_sd(f, confidence)) ** 2 + frac * f.forecast
    return WindowDemand(mean, float(np.sqrt(var)))
