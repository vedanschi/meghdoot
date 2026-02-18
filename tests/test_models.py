"""Tests for VAE loss components."""

from __future__ import annotations

import torch
import pytest


class TestSSIMLoss:
    def test_identical_inputs_loss_near_zero(self):
        from meghdoot.models.vae import SSIMLoss

        loss_fn = SSIMLoss(channels=1)
        x = torch.randn(2, 1, 64, 64)
        loss = loss_fn(x, x)
        assert loss.item() < 0.01

    def test_different_inputs_positive_loss(self):
        from meghdoot.models.vae import SSIMLoss

        loss_fn = SSIMLoss(channels=1)
        x = torch.randn(2, 1, 64, 64)
        y = torch.randn(2, 1, 64, 64)
        loss = loss_fn(x, y)
        assert loss.item() > 0.0


class TestVGGPerceptualLoss:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="VGG test needs GPU or is slow on CPU")
    def test_identical_inputs(self):
        from meghdoot.models.vae import VGGPerceptualLoss

        loss_fn = VGGPerceptualLoss()
        x = torch.randn(1, 3, 64, 64)
        loss = loss_fn(x, x)
        assert loss.item() < 0.01

    def test_different_inputs_cpu(self):
        from meghdoot.models.vae import VGGPerceptualLoss

        loss_fn = VGGPerceptualLoss()
        x = torch.randn(1, 3, 64, 64)
        y = torch.randn(1, 3, 64, 64)
        loss = loss_fn(x, y)
        assert loss.item() > 0.0


class TestMassConservationLoss:
    def test_identical_mass(self):
        from meghdoot.models.diffusion import MassConservationLoss

        loss_fn = MassConservationLoss()
        x = torch.randn(2, 4, 64, 64)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-5

    def test_different_mass(self):
        from meghdoot.models.diffusion import MassConservationLoss

        loss_fn = MassConservationLoss()
        x = torch.randn(2, 4, 64, 64)
        y = torch.randn(2, 4, 64, 64)
        loss = loss_fn(x, y)
        assert loss.item() > 0.0
