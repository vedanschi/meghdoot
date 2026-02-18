"""Tests for MOSDAC data acquisition client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meghdoot.data.mdapi import MOSDACClient, _format_size, run_official_mdapi


# ── Fixtures ────────────────────────────────────────
@pytest.fixture
def minimal_cfg():
    """Minimal Meghdoot YAML config dict for MOSDACClient."""
    return {
        "data": {
            "paths": {"raw": "data/raw"},
            "dataset_ids": ["3RIMG_L1B_STD"],
            "date_range": {"start": "2024-06-01", "end": "2024-08-31"},
            "region": {
                "lat_min": 6.0,
                "lat_max": 38.0,
                "lon_min": 66.0,
                "lon_max": 100.0,
            },
        }
    }


@pytest.fixture
def mosdac_json(tmp_path):
    """Write a temporary official-format config.json."""
    conf = {
        "user_credentials": {
            "username/email": "test_user",
            "password": "test_pass",
        },
        "search_parameters": {
            "datasetId": "3RIMG_L1B_STD",
            "startTime": "2024-06-01",
            "endTime": "2024-08-31",
            "count": "",
            "boundingBox": "66.0,6.0,100.0,38.0",
            "gId": "",
        },
        "download_settings": {
            "download_path": str(tmp_path / "downloads"),
            "organize_by_date": True,
            "skip_user_input": False,
            "generate_error_logs": False,
            "error_logs_dir": "",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(conf, indent=2))
    return path


# ── Construction ────────────────────────────────────
class TestMOSDACClientInit:
    def test_from_meghdoot_cfg(self, minimal_cfg):
        client = MOSDACClient(minimal_cfg)
        assert client._dataset_ids == ["3RIMG_L1B_STD"]
        assert client._start_time == "2024-06-01"
        assert client._end_time == "2024-08-31"
        assert "66.0" in client._bounding_box
        assert "38.0" in client._bounding_box

    def test_from_mosdac_json(self, mosdac_json):
        client = MOSDACClient.from_mosdac_json(mosdac_json)
        assert client.username == "test_user"
        assert client.password == "test_pass"
        assert client._dataset_ids == ["3RIMG_L1B_STD"]
        assert client._start_time == "2024-06-01"
        assert client.organize_by_date is True

    def test_from_mosdac_json_via_init(self, minimal_cfg, mosdac_json):
        client = MOSDACClient(minimal_cfg, mosdac_config=mosdac_json)
        assert client.username == "test_user"
        assert client._dataset_ids == ["3RIMG_L1B_STD"]

    def test_env_vars_override(self, minimal_cfg, monkeypatch):
        monkeypatch.setenv("MOSDAC_USERNAME", "env_user")
        monkeypatch.setenv("MOSDAC_PASSWORD", "env_pass")
        client = MOSDACClient(minimal_cfg)
        assert client.username == "env_user"
        assert client.password == "env_pass"


# ── Authentication ──────────────────────────────────
class TestAuthentication:
    def test_authenticate_success(self, minimal_cfg, monkeypatch):
        monkeypatch.setenv("MOSDAC_USERNAME", "user")
        monkeypatch.setenv("MOSDAC_PASSWORD", "pass")
        client = MOSDACClient(minimal_cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "abc123",
            "refresh_token": "ref456",
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("meghdoot.data.mdapi.requests.post", return_value=mock_resp):
            assert client.authenticate() is True
            assert client._access_token == "abc123"
            assert client._refresh_token == "ref456"

    def test_authenticate_401(self, minimal_cfg, monkeypatch):
        monkeypatch.setenv("MOSDAC_USERNAME", "user")
        monkeypatch.setenv("MOSDAC_PASSWORD", "wrong")
        client = MOSDACClient(minimal_cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": "Invalid credentials"}

        with patch("meghdoot.data.mdapi.requests.post", return_value=mock_resp):
            assert client.authenticate() is False

    def test_authenticate_no_credentials(self, minimal_cfg):
        client = MOSDACClient(minimal_cfg)
        assert client.authenticate() is False


# ── Search ──────────────────────────────────────────
class TestSearch:
    def test_search_success(self, minimal_cfg, monkeypatch):
        monkeypatch.setenv("MOSDAC_USERNAME", "u")
        monkeypatch.setenv("MOSDAC_PASSWORD", "p")
        client = MOSDACClient(minimal_cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "totalResults": 150,
            "totalSizeMB": 2048.5,
            "itemsPerPage": 100,
            "entries": [{"id": "1", "identifier": "file.h5", "updated": None}],
        }

        with patch("meghdoot.data.mdapi.requests.get", return_value=mock_resp):
            result = client.search("3RIMG_L1B_STD")
            assert result["totalResults"] == 150

    def test_search_error_status(self, minimal_cfg, monkeypatch):
        monkeypatch.setenv("MOSDAC_USERNAME", "u")
        monkeypatch.setenv("MOSDAC_PASSWORD", "p")
        client = MOSDACClient(minimal_cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"message": ["Invalid datasetId"]}

        with patch("meghdoot.data.mdapi.requests.get", return_value=mock_resp):
            result = client.search("BAD_ID")
            assert result == {}


# ── Helpers ─────────────────────────────────────────
class TestHelpers:
    def test_format_size_mb(self):
        assert "512.00 MB" == _format_size(512)

    def test_format_size_gb(self):
        assert "2.00 GB" == _format_size(2048)

    def test_format_size_tb(self):
        assert "1.00 TB" == _format_size(1024**2)


# ── Official script launcher ───────────────────────
class TestRunOfficialMdapi:
    def test_missing_script(self, tmp_path):
        rc = run_official_mdapi(project_root=tmp_path)
        assert rc == 1

    def test_runs_script(self, tmp_path):
        # Create a dummy mdapi.py that exits 0
        script = tmp_path / "mdapi.py"
        script.write_text("import sys; sys.exit(0)")

        # Also needs config.json
        (tmp_path / "config.json").write_text("{}")

        rc = run_official_mdapi(project_root=tmp_path)
        assert rc == 0
