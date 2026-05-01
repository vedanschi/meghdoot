"""
dataset.py – PyTorch Dataset for temporal satellite sequences
=============================================================

Loads pre-processed .pt tensors and groups them into sliding-window
sequences of ``(N_history + 1)`` consecutive frames for conditional
diffusion training.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch.utils.data import Dataset

from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


class INSATSequenceDataset(Dataset):
    """Sliding-window dataset over pre-processed satellite .pt tensors.

    Each sample is a dict with:
        - ``"history"``  : Tensor [N_history, C, H, W]  – conditioning frames
        - ``"target"``   : Tensor [C, H, W]             – frame to predict
        - ``"timestamps"``: list[str]                   – filenames as proxy
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_history: int = 3,
        transform=None,
        **kwargs  # Swallows legacy 'channel' args from older training scripts
    ) -> None:
        super().__init__()
        self.num_history = num_history
        self.transform = transform
        self.data_dir = Path(data_dir)

        # Resolve whichever tensor cache layout actually exists in the workspace.
        candidate_dirs = [
            Path("/home/jupyter/local_data/stacked_tensors"),
            self.data_dir / "vae_tensors",
            self.data_dir / "stacked_tensors",
            self.data_dir,
            Path("/home/jupyter/meghdoot_data/processed/vae_tensors"),
            Path("/home/jupyter/meghdoot_data/processed"),
        ]
        target_dir = next((path for path in candidate_dirs if path.exists()), self.data_dir)
        self.target_dir = target_dir

        self.files = sorted(target_dir.glob("*.pt"))
        if len(self.files) < num_history + 1:
            log.warning(f"Found {len(self.files)} files, need {num_history + 1}. Run preprocessing first.")

        log.info(
            f"INSATSequenceDataset: {len(self.files)} tensors, "
            f"{max(0, len(self.files) - self.num_history)} sequences (history={num_history})"
        )

    def __len__(self) -> int:
        return max(0, len(self.files) - self.num_history)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[str]]:
        frames = []
        names = []
        for i in range(self.num_history + 1):
            fp = self.files[idx + i]
            tensor = self._load_tensor(fp)
            frames.append(tensor)
            names.append(fp.stem)

        # Stack → [N_history+1, C, H, W]
        stack = torch.stack(frames, dim=0)

        if self.transform:
            stack = self.transform(stack)

        return {
            "history": stack[: self.num_history],       # [N, C, H, W]
            "target": stack[self.num_history],          # [C, H, W]
            "timestamps": names,
        }

    def _load_tensor(self, fp: Path) -> torch.Tensor:
        """Load a tensor file, trying legacy tuple-string filename variants if needed."""
        try:
            return torch.load(fp, weights_only=True)
        except FileNotFoundError:
            fallback = self._legacy_tuple_filename(fp)
            if fallback is not None and fallback.exists():
                log.warning(f"Falling back to legacy tensor filename: {fallback.name}")
                return torch.load(fallback, weights_only=True)
            raise

    @staticmethod
    def _legacy_tuple_filename(fp: Path) -> Path | None:
        """Convert tuple-style filenames like "('foo',).pt" into plain "foo.pt"."""
        stem = fp.stem
        if stem.startswith("(") and stem.endswith(")"):
            try:
                parsed = ast.literal_eval(stem)
            except (ValueError, SyntaxError):
                return None
            if isinstance(parsed, tuple) and parsed:
                inner = parsed[0]
                if isinstance(inner, str):
                    return fp.with_name(f"{inner}.pt")
        return None


class LatentSequenceDataset(Dataset):
    """Same sliding-window logic but for pre-computed VAE latents.

    Each file is a [C, h, w] latent tensor (e.g. [4, 64, 64]).
    """

    def __init__(
        self,
        latent_dir: str | Path,
        num_history: int = 3,
        cache_in_memory: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.latent_dir = Path(latent_dir)
        self.num_history = num_history
        self.cache_in_memory = cache_in_memory

        # Support both direct latent_dir/*.pt and latent_dir/stacked_tensors/*.pt
        candidate_dirs = [
            Path("/home/jupyter/local_data/stacked_tensors"),
            self.latent_dir / "stacked_tensors",
            self.latent_dir,
            Path("/home/jupyter/meghdoot_data/processed/vae_tensors"),
            Path("/home/jupyter/meghdoot_data/processed"),
        ]
        self.latent_dir = next((path for path in candidate_dirs if path.exists()), self.latent_dir)

        self.files = sorted(self.latent_dir.glob("*.pt"))
        if len(self.files) < num_history + 1:
            log.warning(f"Found {len(self.files)} latents, need {num_history + 1}")

        self._cache = None
        if self.cache_in_memory:
            log.info(f"Loading all {len(self.files)} latents into memory...")
            self._cache = [self._load_tensor(fp) for fp in self.files]
            log.info(f"✓ Cache complete: {len(self._cache)} tensors in RAM")

        log.info(
            f"LatentSequenceDataset: {len(self.files)} latents, "
            f"{max(0, len(self.files) - self.num_history)} sequences"
        )

    def __len__(self) -> int:
        return max(0, len(self.files) - self.num_history)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        latents = []
        for i in range(self.num_history + 1):
            if self._cache is not None:
                tensor = self._cache[idx + i]
            else:
                tensor = self._load_tensor(self.files[idx + i])
            latents.append(tensor)

        # Each tensor is [C, h, w]  →  stack → [N+1, C, h, w]
        stack = torch.stack(latents, dim=0)

        return {
            "history": stack[: self.num_history],       # [3, 4, 64, 64]
            "target": stack[self.num_history],          # [4, 64, 64]
        }

    def _load_tensor(self, fp: Path) -> torch.Tensor:
        try:
            return torch.load(fp, weights_only=True)
        except FileNotFoundError:
            fallback = INSATSequenceDataset._legacy_tuple_filename(fp)
            if fallback is not None and fallback.exists():
                log.warning(f"Falling back to legacy latent filename: {fallback.name}")
                return torch.load(fallback, weights_only=True)
            raise
