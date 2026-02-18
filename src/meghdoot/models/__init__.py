"""Phase 2–3 – Model modules: VAE, Diffusion, Channel Fusion, Temporal Loss."""

from meghdoot.models.vae import SatelliteVAE, SSIMLoss, VGGPerceptualLoss
from meghdoot.models.diffusion import MeghdootDiffusion, MassConservationLoss, EMAModel
from meghdoot.models.channel_fusion import (
    ChannelFusionNet,
    MultiChannelPreprocessor,
    SpectralAttention,
)
from meghdoot.models.temporal_loss import (
    TemporalConsistencyLoss,
    OpticalFlowEstimator,
    WarpedFrameLoss,
)

__all__ = [
    "SatelliteVAE",
    "SSIMLoss",
    "VGGPerceptualLoss",
    "MeghdootDiffusion",
    "MassConservationLoss",
    "EMAModel",
    "ChannelFusionNet",
    "MultiChannelPreprocessor",
    "SpectralAttention",
    "TemporalConsistencyLoss",
    "OpticalFlowEstimator",
    "WarpedFrameLoss",
]
