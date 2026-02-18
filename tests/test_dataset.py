"""Tests for dataset classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal .npy dataset for testing."""
    channel_dir = tmp_path / "TIR1"
    channel_dir.mkdir()
    for i in range(6):
        arr = np.random.rand(512, 512).astype(np.float32) * 2 - 1
        np.save(channel_dir / f"frame_{i:04d}.npy", arr)
    return tmp_path


@pytest.fixture
def tmp_latent_dataset(tmp_path):
    """Create a minimal latent .npy dataset for testing."""
    channel_dir = tmp_path / "TIR1"
    channel_dir.mkdir()
    for i in range(6):
        arr = np.random.randn(4, 64, 64).astype(np.float32)
        np.save(channel_dir / f"frame_{i:04d}.npy", arr)
    return tmp_path


class TestINSATSequenceDataset:
    def test_length(self, tmp_dataset):
        from meghdoot.data.dataset import INSATSequenceDataset

        ds = INSATSequenceDataset(tmp_dataset, channel="TIR1", num_history=3)
        assert len(ds) == 3  # 6 - 3

    def test_sample_shape(self, tmp_dataset):
        from meghdoot.data.dataset import INSATSequenceDataset

        ds = INSATSequenceDataset(tmp_dataset, channel="TIR1", num_history=3)
        sample = ds[0]
        assert sample["history"].shape == (3, 1, 512, 512)
        assert sample["target"].shape == (512, 512)

    def test_insufficient_frames_raises(self, tmp_path):
        from meghdoot.data.dataset import INSATSequenceDataset

        channel_dir = tmp_path / "TIR1"
        channel_dir.mkdir()
        np.save(channel_dir / "frame_0000.npy", np.zeros((64, 64)))

        with pytest.raises(ValueError):
            INSATSequenceDataset(tmp_path, channel="TIR1", num_history=3)


class TestLatentSequenceDataset:
    def test_length(self, tmp_latent_dataset):
        from meghdoot.data.dataset import LatentSequenceDataset

        ds = LatentSequenceDataset(tmp_latent_dataset, channel="TIR1", num_history=3)
        assert len(ds) == 3

    def test_sample_shape(self, tmp_latent_dataset):
        from meghdoot.data.dataset import LatentSequenceDataset

        ds = LatentSequenceDataset(tmp_latent_dataset, channel="TIR1", num_history=3)
        sample = ds[0]
        assert sample["history"].shape == (3, 4, 64, 64)
        assert sample["target"].shape == (4, 64, 64)
