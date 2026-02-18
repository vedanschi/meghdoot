"""
temporal_loss.py – Optical-Flow Temporal Consistency Loss
=========================================================

Penalises predictions that are temporally inconsistent with the
conditioning sequence.  Uses optical flow (estimated via OpenCV's
Farneback method) to warp the last observed frame forward in time,
then measures the discrepancy against the model's prediction.

This encourages smooth, physically plausible cloud advection
rather than abrupt "popping" artefacts.

Architecture diagram reference:
    Loss Function Block → Temporal Loss (Optical Flow)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


class OpticalFlowLoss(nn.Module):
    """Temporal consistency loss via optical-flow warping.

    Given two consecutive conditioning frames (t-1, t) and the
    predicted frame (t+1), this loss:
    1. Estimates optical flow from frame t-1 → t  (using OpenCV)
    2. Warps frame t forward by the flow to produce a "naive" t+1
    3. Penalises the L1 distance between the warp and the prediction

    This acts as a soft physics prior: clouds should move smoothly.

    Parameters
    ----------
    weight : float
        Loss weight (multiplied in the total loss).
    use_latent : bool
        If True, operates on latent tensors (4-channel, 64×64).
        If False, operates on pixel-space (1-channel, 512×512).
    """

    def __init__(self, weight: float = 0.1, use_latent: bool = True) -> None:
        super().__init__()
        self.weight = weight
        self.use_latent = use_latent

    def _estimate_flow_farneback(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
    ) -> np.ndarray:
        """Estimate dense optical flow between two greyscale frames.

        Parameters
        ----------
        prev, curr : ndarray [H, W]  float32 in [0, 255] range

        Returns
        -------
        ndarray [H, W, 2]  – (dx, dy) flow vectors
        """
        import cv2

        # Farneback expects uint8
        prev_u8 = np.clip(prev, 0, 255).astype(np.uint8)
        curr_u8 = np.clip(curr, 0, 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            prev_u8,
            curr_u8,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow  # [H, W, 2]

    def _warp_with_flow(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Warp a frame using optical flow via grid_sample.

        Parameters
        ----------
        frame : Tensor [B, C, H, W]
        flow : Tensor [B, 2, H, W]  – (dx, dy)

        Returns
        -------
        Tensor [B, C, H, W]  – warped frame
        """
        B, C, H, W = frame.shape

        # Build base grid in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device),
            torch.linspace(-1, 1, W, device=frame.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Normalise flow to [-1, 1] range
        flow_norm = flow.permute(0, 2, 3, 1).clone()  # [B, H, W, 2]
        flow_norm[..., 0] /= W / 2
        flow_norm[..., 1] /= H / 2

        sample_grid = base_grid + flow_norm

        return F.grid_sample(
            frame, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def forward(
        self,
        predicted: torch.Tensor,
        frame_t_minus_1: torch.Tensor,
        frame_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal consistency loss.

        Parameters
        ----------
        predicted : Tensor [B, C, H, W]
            Model's predicted frame (t+1).
        frame_t_minus_1 : Tensor [B, C, H, W]
            Conditioning frame at t-1.
        frame_t : Tensor [B, C, H, W]
            Conditioning frame at t (most recent).

        Returns
        -------
        Tensor (scalar)
            Weighted L1 between flow-warped t and predicted t+1.
        """
        B, C, H, W = predicted.shape
        device = predicted.device

        # Estimate flow on first channel (or greyscale summary)
        # We need to go CPU → NumPy for OpenCV
        with torch.no_grad():
            prev_np = frame_t_minus_1[:, 0].cpu().numpy()  # [B, H, W]
            curr_np = frame_t[:, 0].cpu().numpy()

            flows = []
            for b in range(B):
                # Convert [-1,1] → [0,255] for OpenCV
                p = (prev_np[b] + 1.0) * 127.5
                c = (curr_np[b] + 1.0) * 127.5
                flow = self._estimate_flow_farneback(p, c)
                flows.append(flow)

            # [B, H, W, 2] → [B, 2, H, W]
            flow_tensor = torch.from_numpy(np.stack(flows, axis=0)).to(device)
            flow_tensor = flow_tensor.permute(0, 3, 1, 2).float()

        # Warp frame_t forward by the estimated flow → "expected" t+1
        warped = self._warp_with_flow(frame_t, flow_tensor)

        # L1 loss between warp prediction and model prediction
        loss = F.l1_loss(predicted, warped)

        return self.weight * loss


class TemporalGradientLoss(nn.Module):
    """Lightweight alternative: penalise abrupt temporal changes.

    No OpenCV dependency.  Simply measures the frame-to-frame
    difference magnitude, encouraging smooth transitions.

    loss = |predicted - frame_t| - |frame_t - frame_t_minus_1|

    If the model's change is much larger than the observed change,
    the loss is high.
    """

    def __init__(self, weight: float = 0.05) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predicted: torch.Tensor,
        frame_t_minus_1: torch.Tensor,
        frame_t: torch.Tensor,
    ) -> torch.Tensor:
        observed_delta = (frame_t - frame_t_minus_1).abs().mean()
        predicted_delta = (predicted - frame_t).abs().mean()

        # Penalise if predicted change >> observed change
        excess = F.relu(predicted_delta - observed_delta)
        return self.weight * excess
