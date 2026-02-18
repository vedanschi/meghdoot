"""
dataset.py – PyTorch Dataset for temporal satellite sequences
=============================================================

Loads pre-processed .npy frames and groups them into sliding-window
sequences of ``(N_history + 1)`` consecutive frames for conditional
diffusion training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


class INSATSequenceDataset(Dataset):
    """Sliding-window dataset over pre-processed satellite frames.

    Each sample is a dict with:
        - ``"history"``  : Tensor [N_history, C, H, W]  – conditioning frames
        - ``"target"``   : Tensor [C, H, W]             – frame to predict
        - ``"timestamps"``: list[str]                    – filenames as proxy

    Supports both single-channel and multi-channel (5-band fusion) modes.
    """

    def __init__(
        self,
        data_dir: str | Path,
        channel: str | list[str] = "TIR1",
        num_history: int = 3,
        transform=None,
    ) -> None:
        super().__init__()
        self.num_history = num_history
        self.transform = transform

        # Support multi-channel input
        if isinstance(channel, str):
            self.channels = [channel]
        else:
            self.channels = list(channel)

        self.data_dir = Path(data_dir)
        self.multi_channel = len(self.channels) > 1

        # Validate that all channel directories exist
        self._channel_dirs = []
        for ch in self.channels:
            ch_dir = self.data_dir / ch
            if not ch_dir.exists():
                raise FileNotFoundError(f"Channel directory not found: {ch_dir}")
            self._channel_dirs.append(ch_dir)

        # Use first channel's file list as the reference timeline
        self.files = sorted(self._channel_dirs[0].glob("*.npy"))
        if len(self.files) < num_history + 1:
            raise ValueError(
                f"Need at least {num_history + 1} frames, found {len(self.files)}"
            )

        log.info(
            f"INSATSequenceDataset: {len(self.files)} frames, "
            f"{len(self)} sequences (history={num_history}, "
            f"channels={self.channels})"
        )

    def __len__(self) -> int:
        return len(self.files) - self.num_history

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[str]]:
        frames = []
        names = []
        for i in range(self.num_history + 1):
            fp = self.files[idx + i]

            if self.multi_channel:
                # Load all channels and stack → [C, H, W]
                channel_arrays = []
                for ch_dir in self._channel_dirs:
                    ch_file = ch_dir / fp.name
                    if not ch_file.exists():
                        # Fall back to first-channel data if file missing
                        ch_file = fp
                    arr = np.load(ch_file).astype(np.float32)
                    if arr.ndim == 2:
                        channel_arrays.append(arr)
                    else:
                        channel_arrays.append(arr[0] if arr.ndim == 3 else arr)
                frame = np.stack(channel_arrays, axis=0)  # [C, H, W]
            else:
                arr = np.load(fp).astype(np.float32)
                frame = arr[np.newaxis, :, :] if arr.ndim == 2 else arr  # [1, H, W]

            frames.append(frame)
            names.append(fp.stem)

        # Stack → [N_history+1, C, H, W]
        stack = np.stack(frames, axis=0)

        if self.transform:
            stack = self.transform(stack)

        tensor = torch.from_numpy(stack)

        return {
            "history": tensor[: self.num_history],       # [N, C, H, W]
            "target": tensor[self.num_history],           # [C, H, W]
            "timestamps": names,
        }


class LatentSequenceDataset(Dataset):
    """Same sliding-window logic but for pre-computed VAE latents.

    Each file is a [C, h, w] latent tensor (e.g. [4, 64, 64]).
    """

    def __init__(
        self,
        latent_dir: str | Path,
        channel: str = "TIR1",
        num_history: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dir = Path(latent_dir) / channel
        self.num_history = num_history

        if not self.latent_dir.exists():
            raise FileNotFoundError(f"Latent directory not found: {self.latent_dir}")

        self.files = sorted(self.latent_dir.glob("*.npy"))
        if len(self.files) < num_history + 1:
            raise ValueError(
                f"Need at least {num_history + 1} latent frames, found {len(self.files)}"
            )

        log.info(
            f"LatentSequenceDataset: {len(self.files)} latents, "
            f"{len(self)} sequences"
        )

    def __len__(self) -> int:
        return len(self.files) - self.num_history

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        latents = []
        for i in range(self.num_history + 1):
            arr = np.load(self.files[idx + i]).astype(np.float32)
            latents.append(arr)

        # Each arr is [C, h, w]  →  stack → [N+1, C, h, w]
        stack = torch.from_numpy(np.stack(latents, axis=0))

        return {
            "history": stack[: self.num_history],       # [3, 4, 64, 64]
            "target": stack[self.num_history],           # [4, 64, 64]
        }
