"""Tests for the data preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestNormalization:
    """Test brightness temperature normalisation."""

    def test_normalize_to_range(self):
        from meghdoot.data.preprocessing import normalize_brightness_temp

        arr = np.array([200.0, 250.0, 300.0], dtype=np.float32)
        normed = normalize_brightness_temp(arr)

        assert normed.min() >= -1.0
        assert normed.max() <= 1.0
        assert np.isclose(normed[0], -1.0)
        assert np.isclose(normed[-1], 1.0)

    def test_normalize_with_clipping(self):
        from meghdoot.data.preprocessing import normalize_brightness_temp

        arr = np.array([100.0, 200.0, 250.0, 300.0, 400.0], dtype=np.float32)
        normed = normalize_brightness_temp(arr, clip_low=200.0, clip_high=300.0)

        assert normed.min() >= -1.0
        assert normed.max() <= 1.0

    def test_constant_value_returns_zeros(self):
        from meghdoot.data.preprocessing import normalize_brightness_temp

        arr = np.full((10, 10), 250.0, dtype=np.float32)
        normed = normalize_brightness_temp(arr)

        assert np.allclose(normed, 0.0)


class TestResize:
    """Test OpenCV-based array resizing."""

    def test_resize_to_target(self):
        from meghdoot.data.preprocessing import resize_array

        arr = np.random.rand(100, 100).astype(np.float32)
        resized = resize_array(arr, (512, 512))

        assert resized.shape == (512, 512)
        assert resized.dtype == np.float32

    def test_no_op_if_same_size(self):
        from meghdoot.data.preprocessing import resize_array

        arr = np.random.rand(512, 512).astype(np.float32)
        resized = resize_array(arr, (512, 512))

        assert np.array_equal(arr, resized)


class TestCropToRegion:
    """Test regional cropping with xarray."""

    def test_crop_with_coords(self):
        import xarray as xr
        from meghdoot.data.preprocessing import crop_to_region

        lat = np.linspace(0, 40, 100)
        lon = np.linspace(60, 110, 100)
        data = np.random.rand(100, 100).astype(np.float32)

        da = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        cropped = crop_to_region(da, lat_min=6.0, lat_max=38.0, lon_min=66.0, lon_max=100.0)

        assert cropped.lat.min() >= 6.0
        assert cropped.lat.max() <= 38.0
        assert cropped.lon.min() >= 66.0
        assert cropped.lon.max() <= 100.0
