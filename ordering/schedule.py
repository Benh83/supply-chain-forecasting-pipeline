"""Delivery calendars for each depot."""

from __future__ import annotations

import pandas as pd


def delivery_dates(depot: dict, asof: pd.Timestamp, count: int = 3) -> list[pd.Timestamp]:
    """Next `count` delivery dates that can still be ordered for, as of `asof`.

    A delivery is reachable when asof + lead time <= delivery date.
    """
    earliest = asof + pd.Timedelta(days=depot["lead_time_days"])
    days = pd.date_range(earliest, earliest + pd.Timedelta(days=90), freq="D")
    if "delivery_weekdays" in depot:
        ok = days[days.dayofweek.isin(depot["delivery_weekdays"])]
    else:
        anchor = pd.Timestamp(depot["delivery_anchor"])
        period = 7 * depot["delivery_every_n_weeks"]
        ok = days[((days - anchor).days % period) == 0]
    return list(ok[:count])
