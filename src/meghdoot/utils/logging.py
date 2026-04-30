"""
Logging helpers – Rich console + W&B integration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


console = Console()


class SafeRichHandler(RichHandler):
    """Rich handler that falls back to plain logging if Rich rendering fails."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except Exception:
            try:
                fallback = logging.StreamHandler(sys.stdout)
                fallback.setFormatter(logging.Formatter("%(message)s"))
                fallback.emit(record)
            except Exception:
                pass


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a Rich-powered logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler: logging.Handler = SafeRichHandler(
            console=console,
            show_path=False,
            rich_tracebacks=False,
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
            mode="offline",
            save_code=False,
        )
    except ImportError:
        get_logger(__name__).warning("wandb not installed – skipping W&B init")
