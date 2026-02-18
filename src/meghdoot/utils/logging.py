"""
Logging helpers – Rich console + W&B integration.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


console = Console()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a Rich-powered logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def setup_wandb(cfg: dict) -> None:
    """Initialise Weights & Biases run from config dict."""
    try:
        import wandb

        wandb_cfg = cfg.get("logging", {}).get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "meghdoot-ai"),
            entity=wandb_cfg.get("entity"),
            config=cfg,
            save_code=True,
        )
    except ImportError:
        get_logger(__name__).warning("wandb not installed – skipping W&B init")
