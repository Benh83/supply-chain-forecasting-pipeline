import numpy as np
import pandas as pd
import pytest

from forecasting.config import load_config
from forecasting.data import clean_border, complete_series
from forecasting.diagnostics import demand_class, diagnose, trend_status
from forecasting.models import croston_sba, forecast_series
from forecasting.selection import select_model

CFG = load_config()
IDX = pd.date_range("2019-01-01", periods=48, freq="MS")


def series(values, idx=None):
    idx = IDX[: len(values)] if idx is None else idx
    s = pd.Series(np.asarray(values, dtype=float), index=idx, name="units")
    s.index.freq = "MS"
    return s


def test_border_cleaning_flags_recording_error_but_keeps_covid_drop():
    raw = pd.read_csv(CFG["paths"]["border"])
    cleaned, changes = clean_border(raw)
    flagged = set(zip(changes["yearmonth"].dt.strftime("%Y-%m"), changes["column"]))
    assert ("2023-10", "Pedestrians") in flagged
    assert ("2020-04", "Pedestrians") not in flagged
    assert cleaned.loc["2023-10-01", "Pedestrians"] > 200_000
    assert not cleaned.isna().any().any()


def test_complete_series_fills_missing_months_with_zero():
    monthly = pd.DataFrame({"sku": ["A", "A"], "yearmonth": pd.to_datetime(["2024-01-01", "2024-04-01"]),
                            "units": [5, 7]})
    s = complete_series(monthly, "A", pd.Timestamp("2024-05-01"))
    assert list(s) == [5, 0, 0, 7, 0]


def test_all_zero_series_forecasts_zero():
    s = series([0] * 24)
    diag = diagnose(s, None, CFG)
    choice = select_model(diag, False, CFG)
    out = forecast_series(s, choice, diag, CFG)
    assert choice.kind == "zero"
    assert all(f.forecast == 0 for f in out)


def test_sparse_series_uses_croston():
    rng = np.random.default_rng(1)
    s = series(rng.poisson(0.8, 36))
    diag = diagnose(s, None, CFG)
    assert diag.sparsity > CFG["forecast"]["rules"]["sparsity"]
    assert select_model(diag, False, CFG).kind == "croston"


def test_december_rule_takes_priority():
    base = np.tile([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 30], 3)
    s = series(base + np.random.default_rng(0).integers(0, 3, 36))
    diag = diagnose(s, None, CFG)
    choice = select_model(diag, False, CFG)
    assert diag.has_december and choice.december_product
    assert choice.kind == "sarima"


def test_intervals_are_ordered_and_non_negative():
    rng = np.random.default_rng(3)
    for values in (rng.poisson(40, 36), rng.poisson(3, 36), np.r_[rng.poisson(50, 30), [0] * 3, [1] * 3]):
        s = series(values)
        diag = diagnose(s, None, CFG)
        for f in forecast_series(s, select_model(diag, False, CFG), diag, CFG):
            assert 0 <= f.lower <= f.forecast <= f.upper


def test_dead_product_rule():
    s = series(list(np.full(30, 20)) + [0] * 6)
    diag = diagnose(s, None, CFG)
    out = forecast_series(s, select_model(diag, False, CFG), diag, CFG)
    assert out[0].model == "Dead_Product"
    assert trend_status(s) == "Dead"


def test_croston_rate_close_to_true_rate():
    rng = np.random.default_rng(5)
    y = series((rng.random(48) < 0.4) * rng.integers(1, 4, 48))
    rate, _ = croston_sba(y, 0.1)
    assert rate == pytest.approx(y.mean(), rel=0.5)


def test_demand_class():
    assert demand_class(series([5, 6, 5, 7, 6, 5]))[0] == "smooth"
    assert demand_class(series([0, 0, 3, 0, 0, 3, 0, 0, 3]))[0] == "intermittent"
