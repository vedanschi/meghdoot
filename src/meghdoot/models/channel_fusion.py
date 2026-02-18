"""
channel_fusion.py – Multi-Channel Integration Layer
====================================================

Fuses multiple INSAT spectral bands (VIS, NIR, SWIR, TIR, WV)
into a unified representation before VAE encoding.

Architecture (from the Meghdoot architecture diagram):

    VIS (0.55-0.75 μm)  ─┐
    NIR (0.725-1.0 μm)  ─┤
    SWIR (1.55-1.70 μm) ─┼─→ Channel Integration Layer ─→ [B, 3, 512, 512]
    TIR (10.3-11.3 μm)  ─┤                                    ↓
    WV  (6.5-7.1 μm)    ─┘                               VAE Encoder

The layer learns to project N spectral channels down to 3 channels
(matching the RGB VAE input) via 1×1 convolutions with residual
attention, preserving physically meaningful cross-channel correlations
like cloud-top height (TIR–WV) and optical depth (VIS–SWIR).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from meghdoot.utils.logging import get_logger

log = get_logger(__name__)

# Default spectral bands from INSAT-3DR/3DS Imager
DEFAULT_CHANNELS = ["VIS", "NIR", "SWIR", "TIR", "WV"]


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention.

    Learns inter-band importance weights so the model can
    emphasise TIR+WV for cloud features or VIS for daytime
    context adaptively per-sample.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        w = self.pool(x).flatten(1)  # [B, C]
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * w


class ChannelIntegrationLayer(nn.Module):
    """Fuses N spectral bands → 3-channel pseudo-RGB for the VAE.

    Architecture::

        [B, N, H, W]
            ↓
        1×1 Conv (N → 32)  +  BatchNorm + GELU
            ↓
        ChannelAttention (32)
            ↓
        3×3 Conv (32 → 32)  +  BatchNorm + GELU  (local spatial mixing)
            ↓
        1×1 Conv (32 → 3)   (project to VAE input dim)
            ↓
        [B, 3, H, W]

    Parameters
    ----------
    in_channels : int
        Number of input spectral bands.  Default 5 for INSAT
        (VIS, NIR, SWIR, TIR, WV).
    hidden_dim : int
        Intermediate feature channels.
    out_channels : int
        Output channels (3 for RGB VAE).
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 32,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projection = nn.Sequential(
            # 1×1 pointwise: fuse spectral channels
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # Channel attention: learn band importance
            ChannelAttention(hidden_dim),
            # 3×3 spatial: capture local cross-band patterns
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 1×1 project to output dim
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.Tanh(),  # output in [-1, 1] matching VAE input range
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"ChannelIntegrationLayer: {in_channels} bands → "
            f"{out_channels} channels ({n_params:,} params)"
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, N, H, W]
            N spectral bands, each normalised to [-1, 1].

        Returns
        -------
        Tensor [B, 3, H, W]
            Fused pseudo-RGB ready for the VAE encoder.
        """
        return self.projection(x)


class MultiChannelINSATDataset(torch.utils.data.Dataset):
    """Dataset that loads all 5 spectral channels per timestep.

    Assumes directory layout::

        data/processed/
        ├── VIS/
        │   ├── frame_0000.npy   # [H, W]
        │   └── ...
        ├── NIR/
        ├── SWIR/
        ├── TIR/
        └── WV/

    Returns per-sample: ``{"channels": [5, H, W], "target_idx": int}``
    """

    def __init__(
        self,
        data_dir: str,
        channels: list[str] | None = None,
        num_history: int = 3,
    ) -> None:
        from pathlib import Path
        import numpy as np

        super().__init__()
        self.data_dir = Path(data_dir)
        self.channels = channels or DEFAULT_CHANNELS
        self.num_history = num_history

        # Use the first channel's file list as the time axis
        ref_dir = self.data_dir / self.channels[0]
        if not ref_dir.exists():
            raise FileNotFoundError(f"Channel directory not found: {ref_dir}")

        self.files = sorted(ref_dir.glob("*.npy"))
        if len(self.files) < num_history + 1:
            raise ValueError(
                f"Need ≥{num_history + 1} frames, found {len(self.files)}"
            )

        # Verify all channels have the same number of frames
        for ch in self.channels[1:]:
            ch_dir = self.data_dir / ch
            ch_files = sorted(ch_dir.glob("*.npy"))
            if len(ch_files) != len(self.files):
                log.warning(
                    f"Channel {ch} has {len(ch_files)} files vs "
                    f"{len(self.files)} in {self.channels[0]}"
                )

        log.info(
            f"MultiChannelINSATDataset: {len(self.files)} timesteps × "
            f"{len(self.channels)} channels, {len(self)} sequences"
        )

    def __len__(self) -> int:
        return len(self.files) - self.num_history

    def __getitem__(self, idx: int) -> dict:
        import numpy as np

        history_frames = []  # will be [num_history, N_channels, H, W]
        for t in range(self.num_history):
            bands = []
            for ch in self.channels:
                fp = self.data_dir / ch / self.files[idx + t].name
                bands.append(np.load(fp).astype(np.float32))
            history_frames.append(np.stack(bands, axis=0))  # [N_ch, H, W]

        # Target: all channels at time idx + num_history
        target_bands = []
        for ch in self.channels:
            fp = self.data_dir / ch / self.files[idx + self.num_history].name
            target_bands.append(np.load(fp).astype(np.float32))

        history = torch.from_numpy(np.stack(history_frames, axis=0))  # [N_hist, N_ch, H, W]
        target = torch.from_numpy(np.stack(target_bands, axis=0))    # [N_ch, H, W]

        return {
            "history": history,
            "target": target,
            "timestamps": [self.files[idx + i].stem for i in range(self.num_history + 1)],
        }
