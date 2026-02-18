#!/usr/bin/env python3
"""
sync_gcs.py – Google Cloud Storage Sync Utility
================================================

Bi-directional sync between local ``data/`` directory and a GCS bucket
for storing large INSAT-3DR/3DS satellite files and pre-computed latents.

Usage
-----
    # Upload processed data to GCS
    python scripts/sync_gcs.py upload --bucket meghdoot-data --prefix data/

    # Download data from GCS to local
    python scripts/sync_gcs.py download --bucket meghdoot-data --prefix data/

    # Sync (upload missing, skip existing)
    python scripts/sync_gcs.py sync --bucket meghdoot-data

Requirements
------------
    pip install google-cloud-storage
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from tqdm import tqdm

try:
    from google.cloud import storage
except ImportError:
    storage = None  # type: ignore[assignment]


# ── Defaults ───────────────────────────────────────
DEFAULT_BUCKET = "meghdoot-data"
DEFAULT_LOCAL_DIR = "data"
DEFAULT_GCS_PREFIX = "meghdoot"
SKIP_EXTENSIONS = {".pyc", ".log", ".tmp", ".DS_Store"}


def _md5(filepath: Path) -> str:
    """Compute MD5 of a local file (for change detection)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_client() -> "storage.Client":
    if storage is None:
        raise ImportError(
            "google-cloud-storage not installed.  "
            "Run: pip install google-cloud-storage"
        )
    return storage.Client()


def _list_local_files(local_dir: Path) -> list[Path]:
    """Recursively list data files, skipping junk."""
    files = []
    for p in sorted(local_dir.rglob("*")):
        if p.is_file() and p.suffix not in SKIP_EXTENSIONS:
            files.append(p)
    return files


# ── Upload ─────────────────────────────────────────
def upload(
    bucket_name: str,
    local_dir: str = DEFAULT_LOCAL_DIR,
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
    skip_existing: bool = True,
) -> None:
    """Upload local files to GCS bucket."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_dir)

    if not local_path.exists():
        print(f"Local directory {local_path} does not exist.")
        return

    files = _list_local_files(local_path)
    print(f"Found {len(files)} files in {local_path}")

    uploaded, skipped = 0, 0
    for fp in tqdm(files, desc="Uploading"):
        blob_name = f"{gcs_prefix}/{fp.relative_to(local_path)}"
        blob = bucket.blob(blob_name)

        if skip_existing and blob.exists():
            skipped += 1
            continue

        blob.upload_from_filename(str(fp))
        uploaded += 1

    print(f"Done: {uploaded} uploaded, {skipped} skipped (already exist)")


# ── Download ───────────────────────────────────────
def download(
    bucket_name: str,
    local_dir: str = DEFAULT_LOCAL_DIR,
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
    skip_existing: bool = True,
) -> None:
    """Download files from GCS bucket to local directory."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_dir)

    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    print(f"Found {len(blobs)} blobs under gs://{bucket_name}/{gcs_prefix}")

    downloaded, skipped = 0, 0
    for blob in tqdm(blobs, desc="Downloading"):
        if blob.name.endswith("/"):
            continue  # skip directory markers

        rel = blob.name.removeprefix(f"{gcs_prefix}/")
        dest = local_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and dest.exists():
            skipped += 1
            continue

        blob.download_to_filename(str(dest))
        downloaded += 1

    print(f"Done: {downloaded} downloaded, {skipped} skipped (already exist)")


# ── Sync (upload missing + download missing) ──────
def sync(
    bucket_name: str,
    local_dir: str = DEFAULT_LOCAL_DIR,
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
) -> None:
    """Bi-directional sync: upload local-only files, download remote-only."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_dir)

    # Collect remote blobs
    remote_blobs = {
        b.name.removeprefix(f"{gcs_prefix}/"): b
        for b in bucket.list_blobs(prefix=gcs_prefix)
        if not b.name.endswith("/")
    }

    # Collect local files
    local_files = _list_local_files(local_path) if local_path.exists() else []
    local_map = {str(f.relative_to(local_path)): f for f in local_files}

    # Upload local-only
    upload_count = 0
    for rel, fp in tqdm(local_map.items(), desc="Uploading local-only"):
        if rel not in remote_blobs:
            blob = bucket.blob(f"{gcs_prefix}/{rel}")
            blob.upload_from_filename(str(fp))
            upload_count += 1

    # Download remote-only
    download_count = 0
    for rel, blob in tqdm(remote_blobs.items(), desc="Downloading remote-only"):
        if rel not in local_map:
            dest = local_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
            download_count += 1

    print(f"Sync complete: {upload_count} uploaded, {download_count} downloaded")


# ── CLI ────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meghdoot-AI GCS data sync utility"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd, fn in [("upload", upload), ("download", download), ("sync", sync)]:
        p = sub.add_parser(cmd)
        p.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket name")
        p.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help="Local data dir")
        p.add_argument("--prefix", default=DEFAULT_GCS_PREFIX, help="GCS prefix")

    args = parser.parse_args()

    kwargs = dict(
        bucket_name=args.bucket,
        local_dir=args.local_dir,
        gcs_prefix=args.prefix,
    )

    if args.command == "upload":
        upload(**kwargs)
    elif args.command == "download":
        download(**kwargs)
    elif args.command == "sync":
        sync(**kwargs)


if __name__ == "__main__":
    main()
