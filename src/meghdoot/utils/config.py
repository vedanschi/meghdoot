"""
Configuration loader – merges YAML defaults with CLI overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolve_default_config() -> Path:
    """Find the default config in either a source checkout or a packaged image."""
    candidates = [
        Path.cwd() / "configs" / "default.yaml",
        Path.cwd() / "default.yaml",
    ]

    module_path = Path(__file__).resolve()
    candidates.extend(
        parent / "configs" / "default.yaml" for parent in module_path.parents
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate configs/default.yaml. Set MEGHDOOT_CONFIG_PATH or pass an explicit path."
    )


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
    if path:
        path = Path(path)
    else:
        path = _resolve_default_config()

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
