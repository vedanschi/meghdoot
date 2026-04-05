"""
temporal_loss.py – Optical-Flow Temporal Consistency Loss
=========================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


class TemporalConsistencyLoss(nn.Module): # Renamed to match diffusion.py import
    """Temporal consistency loss via optical-flow warping."""

    def __init__(self, warp_weight: float = 1.0, flow_smooth_weight: float = 0.1, flow_mag_weight: float = 0.01) -> None:
        super().__init__()
        self.warp_weight = warp_weight
        self.flow_smooth_weight = flow_smooth_weight
        self.flow_mag_weight = flow_mag_weight

    def _estimate_flow_farneback(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        import cv2
        # Safely normalize latents to 0-255 for OpenCV using min/max scaling per frame
        def to_uint8(arr):
            v_min, v_max = arr.min(), arr.max()
            if v_max - v_min < 1e-5: return np.zeros_like(arr, dtype=np.uint8)
            return ((arr - v_min) / (v_max - v_min) * 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            to_uint8(prev), to_uint8(curr), None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        return flow

    def _warp_with_flow(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = frame.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device),
            torch.linspace(-1, 1, W, device=frame.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        flow_norm = flow.permute(0, 2, 3, 1).clone()
        flow_norm[..., 0] /= W / 2
        flow_norm[..., 1] /= H / 2
        return F.grid_sample(frame, base_grid + flow_norm, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, predicted: torch.Tensor, frame_t_minus_1: torch.Tensor, frame_t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = predicted.shape
        device = predicted.device

        with torch.no_grad():
            prev_np = frame_t_minus_1[:, 0].cpu().numpy()
            curr_np = frame_t[:, 0].cpu().numpy()
            flows = [self._estimate_flow_farneback(prev_np[b], curr_np[b]) for b in range(B)]
            flow_tensor = torch.from_numpy(np.stack(flows, axis=0)).to(device).permute(0, 3, 1, 2).float()

        warped = self._warp_with_flow(frame_t, flow_tensor)
        return self.warp_weight * F.l1_loss(predicted, warped)
