"""Ordering layer: turns monthly demand forecasts into depot orders."""

from ordering.recommend import DepotPlan, recommend

__all__ = ["DepotPlan", "recommend"]
