"""
preprocessing.py – Geospatial Processing Pipeline
==================================================

Converts raw INSAT-3DR / INSAT-3DS satellite files into analysis-ready tensors.

Supported input formats:
  - HDF5  (.hdf5 / .h5)  –  via h5py + xarray
  - NetCDF (.nc / .nc4)   –  via xarray / netCDF4

Pipeline:
  1. Open HDF5/NetCDF with xarray
  2. Reproject using GDAL/Rasterio onto a consistent spatial grid
  3. Regional crop to the Indian subcontinent
  4. Resize to 512x512 with OpenCV (bilinear)
  5. Normalize 10-bit values to [-1, 1]
  6. Stack TIR1 and WV channels into a [2, 512, 512] PyTorch Tensor
  7. Save as .pt for Hugging Face Diffusers pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import xarray as xr
import torch
from tqdm import tqdm

from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import ensure_dir
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


_DATASET_KEYS = {
    "TIR1": "IMG_TIR1",
    "TIR2": "IMG_TIR2",
    "MIR":  "IMG_MIR",
    "WV":   "IMG_WV",
    "VIS":  "IMG_VIS",
    "SWIR": "IMG_SWIR",
}

_LAT_KEY = "Latitude"
_LON_KEY = "Longitude"
_TARGET_CRS = "EPSG:4326"


def open_satellite_file(filepath: str | Path, channel: str) -> xr.DataArray:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".nc", ".nc4"):
        return _open_netcdf(filepath, channel)
    elif suffix in (".hdf5", ".h5"):
        return _open_hdf5(filepath, channel)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _open_netcdf(filepath: Path, channel: str) -> xr.DataArray:
    ds = xr.open_dataset(filepath, engine="netcdf4")
    ds_key = _DATASET_KEYS.get(channel, f"IMG_{channel}")

    if ds_key not in ds:
        available = list(ds.data_vars)
        raise KeyError(
            f"Variable '{ds_key}' not found in {filepath.name}. "
            f"Available: {available}"
        )

    da = ds[ds_key].astype(np.float32)
    da.attrs["channel"] = channel
    da.attrs["source_file"] = filepath.name
    return da


def _open_hdf5(filepath: Path, channel: str) -> xr.DataArray:
    ds_key = _DATASET_KEYS.get(channel, f"IMG_{channel}")

    with h5py.File(filepath, "r") as f:
        if ds_key not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset '{ds_key}' not found in {filepath.name}. "
                f"Available keys: {available}"
            )
        data = f[ds_key][:]

        lat = f[_LAT_KEY][:] if _LAT_KEY in f else None
        lon = f[_LON_KEY][:] if _LON_KEY in f else None

    if lat is not None and lon is not None:
        if lat.ndim == 2:
            lat_1d = lat[:, lat.shape[1] // 2]
            lon_1d = lon[lon.shape[0] // 2, :]
        else:
            lat_1d, lon_1d = lat, lon

        da = xr.DataArray(
            data.astype(np.float32),
            dims=["lat", "lon"],
            coords={"lat": lat_1d, "lon": lon_1d},
            attrs={"channel": channel, "source_file": filepath.name},
        )
    else:
        da = xr.DataArray(
            data.astype(np.float32),
            dims=["y", "x"],
            attrs={"channel": channel, "source_file": filepath.name},
        )
        log.warning(f"No lat/lon found in {filepath.name}; using pixel indices")

    return da


def crop_to_region(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    if "lat" in da.coords and "lon" in da.coords:
        lat_sorted = da.lat.values[0] < da.lat.values[-1]
        if lat_sorted:
            da = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        else:
            da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        log.warning("No geo-coords – skipping regional crop")
    return da


def resize_array(arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    if arr.shape == target_size:
        return arr
    resized = cv2.resize(
        arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )
    return resized.astype(np.float32)


def reproject_to_equirectangular(
    arr: np.ndarray,
    src_bounds: tuple[float, float, float, float],
    target_size: tuple[int, int],
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:4326",
) -> np.ndarray:
    src_h, src_w = arr.shape
    dst_h, dst_w = target_size

    src_transform = from_bounds(*src_bounds, src_w, src_h)
    dst_transform = from_bounds(*src_bounds, dst_w, dst_h)

    destination = np.zeros(target_size, dtype=np.float32)

    reproject(
        source=arr.astype(np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    return destination


def normalize_for_vae(arr: np.ndarray) -> np.ndarray:
    return (arr / 511.5) - 1.0


def process_channel(filepath: Path, channel: str, region: dict, target_size: tuple[int, int]) -> np.ndarray:
    da = open_satellite_file(filepath, channel)
    da = crop_to_region(
        da,
        lat_min=region["lat_min"],
        lat_max=region["lat_max"],
        lon_min=region["lon_min"],
        lon_max=region["lon_max"],
    )

    arr = da.values

    if "lat" in da.coords and "lon" in da.coords:
        src_bounds = (
            float(da.lon.min()), float(da.lat.min()),
            float(da.lon.max()), float(da.lat.max()),
        )
        arr = reproject_to_equirectangular(arr, src_bounds, target_size)
    else:
        arr = resize_array(arr, target_size)

    arr = normalize_for_vae(arr)
    return arr


def process_single_file(
    filepath: Path,
    region: dict,
    target_size: tuple[int, int],
    out_dir: Path,
) -> Path | None:
    try:
        arr_tir1 = process_channel(filepath, "TIR1", region, target_size)
        arr_wv = process_channel(filepath, "WV", region, target_size)

        stacked_array = np.stack([arr_tir1, arr_wv], axis=0)
        tensor = torch.from_numpy(stacked_array).float()

        stem = filepath.stem
        out_path = ensure_dir(out_dir / "vae_tensors") / f"{stem}.pt"
        torch.save(tensor, out_path)
        return out_path

    except Exception as e:
        log.error(f"Failed to process {filepath.name}: {e}")
        return None


def run_preprocessing(cfg: dict) -> None:
    data_cfg = cfg["data"]
    region = data_cfg["region"]
    target_size = tuple(data_cfg["crop_size"])
    raw_dir = Path(data_cfg["paths"]["raw"])
    processed_dir = Path(data_cfg["paths"]["processed"])

    files = sorted(
        list(raw_dir.rglob("*.h*5"))
        + list(raw_dir.rglob("*.nc"))
        + list(raw_dir.rglob("*.nc4"))
    )
    
    if not files:
        log.warning(f"No files found in {raw_dir}")
        return

    log.info(f"Processing {len(files)} files into stacked PyTorch tensors (TIR1 + WV)")

    for fp in tqdm(files, desc="Converting to .pt"):
        process_single_file(
            filepath=fp,
            region=region,
            target_size=target_size,
            out_dir=processed_dir,
        )

    log.info("Preprocessing complete ✓")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pre-process INSAT-3DR/3DS data into PyTorch Tensors")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()

