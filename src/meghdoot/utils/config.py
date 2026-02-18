"""
Configuration loader – merges YAML defaults with CLI overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load YAML config and apply optional overrides.

    Parameters
    ----------
    path : str or Path, optional
        Path to a YAML config file. Falls back to ``configs/default.yaml``.
    overrides : dict, optional
        Dot-separated key overrides, e.g. ``{"vae.fine_tune.epochs": 50}``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    path = Path(path) if path else _DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Dump config dict back to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
