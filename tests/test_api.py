"""Tests for FastAPI inference endpoints."""

from __future__ import annotations

import pytest


@pytest.fixture
def mock_diffusion(monkeypatch):
    """Patch the global diffusion model so the app doesn't need real weights."""
    import numpy as np
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.sample.return_value = np.random.randn(1, 4, 64, 64).astype(np.float32)

    import meghdoot.deploy.api as api_module
    monkeypatch.setattr(api_module, "diffusion", mock)
    return mock


@pytest.fixture
def client(mock_diffusion):
    """Create a TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient
    from meghdoot.deploy.api import app

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "scheduler" in data


class TestPredictJSON:
    def test_predict_json_missing_body(self, client):
        resp = client.post("/predict/json")
        assert resp.status_code == 422  # validation error
