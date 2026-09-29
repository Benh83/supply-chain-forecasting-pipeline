"""Configuration loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read settings.yaml and resolve data paths relative to the repo root."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: str(REPO_ROOT / v) for k, v in cfg["paths"].items()}
    return cfg
