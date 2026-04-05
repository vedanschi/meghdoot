"""
baselines.py – Baseline Models for Comparative Evaluation
==========================================================

Implements:
  1. ConvLSTM  – convolutional LSTM encoder-decoder
  2. PySTEPS optical flow wrapper

These serve as the "predictive blur" baselines that
Meghdoot-AI's latent diffusion model must beat.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════
#  1. ConvLSTM Baseline
# ═══════════════════════════════════════════════════

class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMPredictor(nn.Module):
    """ConvLSTM encoder-decoder for next-frame prediction.

    Takes a sequence of frames and predicts the next frame.
    This is the **standard baseline** known for "predictive blur".
    """

    def __init__(
        self,
        in_channels: int = 2,  # Updated for TIR1 + WV
        hidden_dims: Sequence[int] = (64, 64, 64),
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.hidden_dims = list(hidden_dims)

        # Encoder cells
        self.encoder_cells = nn.ModuleList()
        for i, hd in enumerate(hidden_dims):
            inp = in_channels if i == 0 else hidden_dims[i - 1]
            self.encoder_cells.append(ConvLSTMCell(inp, hd, kernel_size))

        # Decoder: project last hidden state to output
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq : Tensor [B, T, C, H, W]
            Sequence of T input frames.

        Returns
        -------
        Tensor [B, C, H, W]
            Predicted next frame.
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Initialise hidden states
        h = [torch.zeros(B, hd, H, W, device=device) for hd in self.hidden_dims]
        c = [torch.zeros(B, hd, H, W, device=device) for hd in self.hidden_dims]

        # Encode sequence
        for t in range(T):
            inp = x_seq[:, t]
            for i, cell in enumerate(self.encoder_cells):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]

        # Decode from last hidden state
        return self.decoder(h[-1])


# ═══════════════════════════════════════════════════
#  2. PySTEPS Optical Flow Baseline
# ═══════════════════════════════════════════════════

def pysteps_forecast(
    frames: np.ndarray,
    n_leadtimes: int = 1,
    method: str = "lucaskanade",
) -> np.ndarray:
    """Generate nowcast using PySTEPS optical flow."""
    try:
        from pysteps.motion.lucaskanade import dense_lucaskanade
        from pysteps.nowcasts.extrapolation import forecast as extrap_forecast
    except ImportError:
        raise ImportError(
            "pysteps is required for the optical flow baseline. "
            "Install with: pip install pysteps"
        )

    # PySTEPS expects [T, H, W]; use last 2+ frames for motion
    motion = dense_lucaskanade(frames[-3:])
    forecast = extrap_forecast(frames[-1], motion, n_leadtimes)
    return forecast
