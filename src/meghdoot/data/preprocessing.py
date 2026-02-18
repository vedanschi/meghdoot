"""
preprocessing.py – Geospatial Processing Pipeline
==================================================

Converts raw INSAT-3DR / INSAT-3DS satellite files into analysis-ready tensors.

Supported input formats:
  - HDF5  (.hdf5 / .h5)  –  via h5py + xarray
  - NetCDF (.nc / .nc4)   –  via xarray / netCDF4

Pipeline:
  1. Open HDF5/NetCDF with xarray  →  data-cube (Time × Lat × Lon × Channel)
  2. Reproject using GDAL/Rasterio onto a consistent spatial grid
  3. Regional crop to the Indian subcontinent
  4. Resize to 512×512 with OpenCV (bilinear)
  5. Normalise brightness temperatures to [-1, 1]
  6. Save as .npy for fast data loading

Usage
-----
    python -m meghdoot.data.preprocessing --config configs/default.yaml
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
from tqdm import tqdm

from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import ensure_dir
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)


# ── Constants ──────────────────────────────────────────
# Typical dataset keys inside INSAT-3DR / INSAT-3DS L1C files
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

# Target CRS for reprojection (Equirectangular / Plate Carrée)
_TARGET_CRS = "EPSG:4326"


# ── File Opener (HDF5 + NetCDF) ───────────────────
def open_satellite_file(filepath: str | Path, channel: str) -> xr.DataArray:
    """Open an INSAT-3DR/3DS file in HDF5 or NetCDF format.

    Dispatches to the appropriate reader based on file extension.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".nc", ".nc4"):
        return _open_netcdf(filepath, channel)
    elif suffix in (".hdf5", ".h5"):
        return _open_hdf5(filepath, channel)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _open_netcdf(filepath: Path, channel: str) -> xr.DataArray:
    """Read a NetCDF file into an xarray DataArray."""
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


# ── HDF5 reader ─────────────────────────────────
def _open_hdf5(filepath: Path, channel: str) -> xr.DataArray:
    """Read a single INSAT-3DR/3DS HDF5 file into an xarray DataArray.

    Parameters
    ----------
    filepath : Path
        Path to the .hdf5 / .h5 file.
    channel : str
        Channel name ("TIR1", "WV", etc.) used to locate the dataset.

    Returns
    -------
    xr.DataArray
        2-D array with ``lat`` and ``lon`` coordinates.
    """
    ds_key = _DATASET_KEYS.get(channel, f"IMG_{channel}")

    with h5py.File(filepath, "r") as f:
        # Read image data
        if ds_key not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset '{ds_key}' not found in {filepath.name}. "
                f"Available keys: {available}"
            )
        data = f[ds_key][:]

        # Read geolocation if available
        lat = f[_LAT_KEY][:] if _LAT_KEY in f else None
        lon = f[_LON_KEY][:] if _LON_KEY in f else None

    # Build xarray
    if lat is not None and lon is not None:
        # If lat/lon are 2-D grids, take centre row/col as 1-D coords
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


# ── Regional Crop ──────────────────────────────────
def crop_to_region(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    """Slice a DataArray to a lat/lon bounding box.

    If the DataArray has proper ``lat``/``lon`` coordinates the slice
    is exact.  Otherwise we fall back to a proportional pixel crop.
    """
    if "lat" in da.coords and "lon" in da.coords:
        # xarray smart indexing (handles descending lat)
        lat_sorted = da.lat.values[0] < da.lat.values[-1]
        if lat_sorted:
            da = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        else:
            da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        log.debug(f"Cropped to region: {da.shape}")
    else:
        log.warning("No geo-coords – skipping regional crop")
    return da


# ── Resize (OpenCV) ─────────────────────────────
def resize_array(arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize 2-D array to ``target_size`` using OpenCV bilinear interpolation.

    OpenCV provides faster and higher-quality resampling than
    scipy for large satellite imagery arrays.
    """
    if arr.shape == target_size:
        return arr
    # OpenCV uses (width, height) ordering
    resized = cv2.resize(
        arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )
    return resized.astype(np.float32)


# ── GDAL / Rasterio Reprojection ─────────────
def reproject_to_equirectangular(
    arr: np.ndarray,
    src_bounds: tuple[float, float, float, float],
    target_size: tuple[int, int],
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:4326",
) -> np.ndarray:
    """Reproject and resample an array onto a regular lat/lon grid.

    Uses Rasterio (backed by GDAL) to handle satellite-native
    projections and ensure georeferencing consistency.

    Parameters
    ----------
    arr : ndarray [H, W]
        Source imagery.
    src_bounds : (left, bottom, right, top)
        Geographic bounds of the source array.
    target_size : (H, W)
        Output array dimensions.
    src_crs, dst_crs : str
        Coordinate reference systems.

    Returns
    -------
    ndarray [target_H, target_W]
    """
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


# ── Normalisation ──────────────────────────────────
def normalize_brightness_temp(
    arr: np.ndarray,
    clip_low: float | None = None,
    clip_high: float | None = None,
) -> np.ndarray:
    """Scale brightness temperatures to [-1, 1].

    Parameters
    ----------
    arr : ndarray
        Raw brightness temperature values (Kelvin).
    clip_low, clip_high : float, optional
        If given, clip to these values before normalising.
        Typically computed from percentile statistics of the dataset.

    Returns
    -------
    ndarray
        Values in [-1, 1].
    """
    if clip_low is not None and clip_high is not None:
        arr = np.clip(arr, clip_low, clip_high)

    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-6:
        log.warning("Constant-value frame detected – returning zeros")
        return np.zeros_like(arr)

    # Map [vmin, vmax] → [-1, 1]
    return (2.0 * (arr - vmin) / (vmax - vmin) - 1.0).astype(np.float32)


def compute_dataset_statistics(
    file_paths: Sequence[Path],
    channel: str,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> dict[str, float]:
    """Compute global min/max/percentile stats across all files.

    Used for robust normalisation (clip outlier pixels).
    """
    all_vals: list[float] = []
    for fp in tqdm(file_paths, desc=f"Stats ({channel})", leave=False):
        try:
            da = open_satellite_file(fp, channel)
            vals = da.values.ravel()
            # sub-sample for speed
            idx = np.random.choice(len(vals), min(50_000, len(vals)), replace=False)
            all_vals.append(vals[idx])
        except Exception as e:
            log.warning(f"Skipping {fp.name}: {e}")

    if not all_vals:
        return {"p_low": 0.0, "p_high": 1.0, "global_min": 0.0, "global_max": 1.0}

    merged = np.concatenate(all_vals)
    p_low, p_high = np.percentile(merged, percentiles)
    return {
        "p_low": float(p_low),
        "p_high": float(p_high),
        "global_min": float(merged.min()),
        "global_max": float(merged.max()),
    }


# ── Full Processing Pipeline ──────────────────────
def process_single_file(
    filepath: Path,
    channel: str,
    region: dict,
    target_size: tuple[int, int],
    clip_low: float | None,
    clip_high: float | None,
    out_dir: Path,
) -> Path | None:
    """Process one HDF5 file → normalised .npy tensor.

    Returns the path to the saved .npy, or None on failure.
    """
    try:
        da = open_satellite_file(filepath, channel)

        # Crop
        da = crop_to_region(
            da,
            lat_min=region["lat_min"],
            lat_max=region["lat_max"],
            lon_min=region["lon_min"],
            lon_max=region["lon_max"],
        )

        arr = da.values

        # Reproject onto a consistent spatial grid via Rasterio/GDAL
        if "lat" in da.coords and "lon" in da.coords:
            src_bounds = (
                float(da.lon.min()), float(da.lat.min()),
                float(da.lon.max()), float(da.lat.max()),
            )
            arr = reproject_to_equirectangular(arr, src_bounds, target_size)
        else:
            # Fallback: resize with OpenCV
            arr = resize_array(arr, target_size)

        # Normalise
        arr = normalize_brightness_temp(arr, clip_low, clip_high)

        # Save
        stem = filepath.stem
        out_path = ensure_dir(out_dir / channel) / f"{stem}.npy"
        np.save(out_path, arr)
        return out_path

    except Exception as e:
        log.error(f"Failed to process {filepath.name}: {e}")
        return None


def run_preprocessing(cfg: dict) -> None:
    """Run the full preprocessing pipeline from config."""
    data_cfg = cfg["data"]
    region = data_cfg["region"]
    target_size = tuple(data_cfg["crop_size"])
    raw_dir = Path(data_cfg["paths"]["raw"])
    processed_dir = Path(data_cfg["paths"]["processed"])
    pct = data_cfg["normalization"]["clip_percentile"]

    for channel in data_cfg["channels"]:
        chan_dir = raw_dir / channel
        if not chan_dir.exists():
            log.warning(f"No raw data directory for {channel}: {chan_dir}")
            continue

        files = sorted(
            list(chan_dir.glob("*.h*5"))
            + list(chan_dir.glob("*.nc"))
            + list(chan_dir.glob("*.nc4"))
        )  # HDF5 + NetCDF
        if not files:
            log.warning(f"No HDF5 files found in {chan_dir}")
            continue

        log.info(f"Processing {len(files)} files for channel {channel}")

        # Global statistics for robust normalisation
        stats = compute_dataset_statistics(files, channel, percentiles=tuple(pct))
        log.info(f"  Stats: clip=[{stats['p_low']:.1f}, {stats['p_high']:.1f}]")

        for fp in tqdm(files, desc=channel):
            process_single_file(
                filepath=fp,
                channel=channel,
                region=region,
                target_size=target_size,
                clip_low=stats["p_low"],
                clip_high=stats["p_high"],
                out_dir=processed_dir,
            )

    log.info("Preprocessing complete ✓")


# ── CLI ────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pre-process INSAT-3DR HDF5 data")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
