"""Model selection: the original priority rules, fed by tested diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

from forecasting.diagnostics import Diagnostics


@dataclass
class ModelChoice:
    kind: str                       # zero | sparse_recent | croston | moving_avg | ets | arima | sarima | sarimax_border
    seasonal: bool = False
    december_product: bool = False
    seasonal_boost: float = 1.0
    reasons: list[str] = field(default_factory=list)


def select_model(diag: Diagnostics, has_border: bool, cfg: dict) -> ModelChoice:
    """Apply the original priority order.

    Differences from the original are only in what the thresholds are compared
    against: border correlation and autocorrelation must also be statistically
    significant, and sparse series go to Croston/SBA instead of a flat average.
    """
    r = cfg["forecast"]["rules"]
    alpha = cfg["forecast"]["tests"]["alpha"]
    n = diag.n

    if diag.n == 0 or diag.non_zero == 0:
        return ModelChoice("zero", reasons=["no sales in history"])
    if diag.non_zero < r["min_nonzero"]:
        return ModelChoice("sparse_recent", reasons=[f"only {diag.non_zero} non-zero months"])
    if diag.sparsity > r["sparsity"]:
        return ModelChoice("croston", reasons=[
            f"sparsity {diag.sparsity:.2f} > {r['sparsity']} ({diag.demand_class} demand)"])

    border_ok = (has_border and diag.border_corr > r["border_corr"]
                 and diag.border_p < alpha)
    autocorr_ok = diag.autocorr > r["autocorr"] and diag.ljung_box_p < alpha

    # Priority 1: December products (overrides everything else)
    if diag.has_december and n >= 8:
        why = f"strong December ({diag.decembers_seen} Decembers of evidence)"
        boost = r["december_boost"]
        if has_border and n >= 12:
            return ModelChoice("sarimax_border", True, True, boost, [why, "border data available"])
        return ModelChoice("sarima", True, True, boost, [why])

    # Priority 2: summer seasonality
    if diag.has_summer and n >= 12:
        if border_ok:
            return ModelChoice("sarimax_border", True, reasons=[
                "strong summer", f"border corr {diag.border_corr:.2f} ({diag.border_feature})"])
        return ModelChoice("sarima", True, reasons=["strong summer"])

    # Priority 3: border correlation
    if border_ok and n >= 8:
        return ModelChoice("sarimax_border", diag.seasonal_strength > r["seasonal_strength_soft"],
                           reasons=[f"border corr {diag.border_corr:.2f} with {diag.border_feature} "
                                    f"(p={diag.border_p:.3f})"])

    # Priority 4: autocorrelation
    if autocorr_ok and n >= 8:
        if diag.seasonal_strength > r["seasonal_strength_soft"] and n >= 12:
            return ModelChoice("sarima", True, reasons=[
                f"autocorr {diag.autocorr:.2f}", f"seasonal strength {diag.seasonal_strength:.2f}"])
        return ModelChoice("arima", reasons=[
            f"autocorr {diag.autocorr:.2f} (Ljung-Box p={diag.ljung_box_p:.3f})"])

    # Priority 5: generic seasonality
    if diag.seasonal_strength > r["seasonal_strength"] and n >= 12:
        return ModelChoice("sarima", True, reasons=[f"seasonal strength {diag.seasonal_strength:.2f}"])

    if n >= 4:
        return ModelChoice("ets", reasons=["default"])
    return ModelChoice("moving_avg", reasons=["short history"])
