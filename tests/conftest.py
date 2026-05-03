"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def sample_frame():
    """A reusable 512×512 random frame."""
    return np.random.rand(512, 512).astype(np.float32) * 2 - 1


@pytest.fixture(scope="session")
def sample_latent():
    """A reusable 4×64×64 latent tensor."""
    return np.random.randn(4, 64, 64).astype(np.float32)
