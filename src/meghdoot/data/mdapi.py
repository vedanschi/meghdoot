"""
mdapi.py – MOSDAC Data Acquisition Wrapper
===========================================

Integration layer between Meghdoot-AI and the **official MOSDAC
Data Download API** (``mdapi.py`` + ``config.json``).

MOSDAC distributes a standalone downloader as a zip:
    https://mosdac.gov.in/software/mdapi.zip
Docs: https://www.mosdac.gov.in/downloadapi-manual

The official files live at the **project root**::

    meghdoot/
    ├── mdapi.py        ← official MOSDAC script (unmodified)
    ├── config.json     ← official MOSDAC config  (edit with your creds)
    └── src/meghdoot/data/mdapi.py   ← THIS FILE (wrapper)

This wrapper provides two modes:

1. **Run official script** — ``run_official_mdapi()`` launches the
   root-level ``mdapi.py`` via subprocess so it reads ``config.json``
   from the project root exactly as MOSDAC intended.

2. **Programmatic client** — ``MOSDACClient`` class talks to the
   same API endpoints for use within the Meghdoot pipeline / CLI.

Usage
-----
    # Mode 1: Run official script directly
    cd meghdoot/ && python mdapi.py

    # Mode 2: Via Meghdoot CLI (uses MOSDACClient)
    meghdoot download --mosdac-config config.json

    # Mode 3: Programmatic
    from meghdoot.data.mdapi import MOSDACClient
    client = MOSDACClient.from_mosdac_json("config.json")
    results = client.search("3RIMG_L1B_STD")

Credentials
-----------
    Option A: Fill ``config.json`` → ``user_credentials`` section.
    Option B: Export ``MOSDAC_USERNAME`` / ``MOSDAC_PASSWORD`` env vars.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from meghdoot.utils.config import load_config
from meghdoot.utils.helpers import ensure_dir
from meghdoot.utils.logging import get_logger

log = get_logger(__name__)

# ── Official MOSDAC API endpoints ───────────────────
# (same URLs hardcoded in the official mdapi.py)
_TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
_SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
_DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
_REFRESH_URL = "https://mosdac.gov.in/download_api/refresh-token"
_LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"
_CHECK_INTERNET_URL = "https://mosdac.gov.in/download_api/check-internet"

_BATCH_SIZE = 100  # MOSDAC pagination page size
_DAILY_DOWNLOAD_LIMIT = 5000
_CHUNK_SIZE = 1_048_576  # 1 MiB download chunks
_RETRY_DELAYS = [10, 20, 30, 60, 90, 120]

# ── Locate project root (where official mdapi.py lives) ──
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/meghdoot/data → root


# ═══════════════════════════════════════════════════
# Mode 1: Launch the official MOSDAC script as-is
# ═══════════════════════════════════════════════════
def run_official_mdapi(
    project_root: Path | str | None = None,
    config_json: Path | str | None = None,
) -> int:
    """Run the unmodified official ``mdapi.py`` via subprocess.

    The official script expects ``config.json`` in its cwd.
    We ``chdir`` into the directory containing the config before
    launching so everything works exactly as MOSDAC intended.

    Parameters
    ----------
    project_root : Path, optional
        Directory containing ``mdapi.py`` + ``config.json``.
        Defaults to the Meghdoot project root.
    config_json : Path, optional
        Path to a custom ``config.json``.  If provided, it is
        **copied** (symlinked) next to ``mdapi.py`` before launch.

    Returns
    -------
    int
        Process return code (0 = success).
    """
    root = Path(project_root) if project_root else _PROJECT_ROOT
    script = root / "mdapi.py"

    if not script.exists():
        log.error(
            f"Official mdapi.py not found at {script}. "
            f"Download it from https://mosdac.gov.in/software/mdapi.zip"
        )
        return 1

    # If a custom config was passed, copy it alongside the script
    cwd = root
    if config_json:
        src = Path(config_json).resolve()
        dst = root / "config.json"
        if src != dst:
            import shutil
            shutil.copy2(src, dst)
            log.info(f"Copied {src} → {dst}")

    log.info(f"Launching official MOSDAC mdapi.py from {cwd}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd),
        # stdin stays connected so the user can answer Y/N prompts
    )
    return result.returncode


# ═══════════════════════════════════════════════════
# Mode 2: Programmatic MOSDACClient
# ═══════════════════════════════════════════════════
class MOSDACClient:
    """Programmatic client for the official MOSDAC Data Download API.

    Mirrors the workflow described in the MOSDAC manual:
    1. Search  (no auth needed)
    2. Authenticate → access + refresh tokens
    3. Paginated download with automatic token refresh
    4. Logout

    Can be constructed from:
    - A Meghdoot YAML config  (``__init__``)
    - The official ``config.json``  (``from_mosdac_json``)
    """

    def __init__(self, cfg: dict, mosdac_config: Path | str | None = None) -> None:
        self.cfg = cfg
        self.raw_dir = Path(cfg["data"]["paths"]["raw"])
        self.organize_by_date: bool = True

        # ── Resolve credentials ───────────────────────
        self.username: str = ""
        self.password: str = ""

        # ── Resolve search parameters ─────────────────
        self._dataset_ids: list[str] = []
        self._start_time: str = ""
        self._end_time: str = ""
        self._count: str = ""
        self._bounding_box: str = ""
        self._gid: str = ""

        if mosdac_config is not None:
            self._load_mosdac_json(Path(mosdac_config))
        else:
            self._load_from_meghdoot_cfg(cfg)

        # Override with env vars if present
        self.username = os.environ.get("MOSDAC_USERNAME", self.username)
        self.password = os.environ.get("MOSDAC_PASSWORD", self.password)

        # ── Token state ───────────────────────────────
        self._access_token: str | None = None
        self._refresh_token: str | None = None

        if not self.username:
            log.warning(
                "MOSDAC credentials not set. Register at "
                "https://mosdac.gov.in/signup/ and export "
                "MOSDAC_USERNAME / MOSDAC_PASSWORD env vars, or "
                "provide a mosdac_config.json."
            )

    # ── Config loaders ──────────────────────────────
    def _load_mosdac_json(self, path: Path) -> None:
        """Load official-format ``config.json``."""
        with open(path) as f:
            conf = json.load(f)

        creds = conf.get("user_credentials", {})
        self.username = creds.get("username/email", creds.get("username", ""))
        self.password = creds.get("password", "")

        sp = conf.get("search_parameters", {})
        ds_id = sp.get("datasetId", "")
        self._dataset_ids = [ds_id] if ds_id else []
        self._start_time = sp.get("startTime", "")
        self._end_time = sp.get("endTime", "")
        self._count = str(sp.get("count", ""))
        self._bounding_box = sp.get("boundingBox", "")
        self._gid = sp.get("gId", "")

        dl = conf.get("download_settings", {})
        if dl.get("download_path"):
            self.raw_dir = Path(dl["download_path"])
        self.organize_by_date = dl.get("organize_by_date", False)
        log.info(f"Loaded MOSDAC config from {path}")

    def _load_from_meghdoot_cfg(self, cfg: dict) -> None:
        """Derive MOSDAC search params from the Meghdoot YAML config."""
        data = cfg["data"]
        self._dataset_ids = data.get("dataset_ids", [])
        dr = data.get("date_range", {})
        self._start_time = dr.get("start", "")
        self._end_time = dr.get("end", "")

        region = data.get("region", {})
        if all(k in region for k in ("lon_min", "lat_min", "lon_max", "lat_max")):
            self._bounding_box = (
                f"{region['lon_min']},{region['lat_min']},"
                f"{region['lon_max']},{region['lat_max']}"
            )

    @classmethod
    def from_mosdac_json(
        cls,
        config_json: Path | str,
        download_path: str = "data/raw",
    ) -> "MOSDACClient":
        """Construct a client directly from the official ``config.json``.

        This is a convenience constructor that doesn't require a full
        Meghdoot YAML config.  Useful for standalone data acquisition.

        Parameters
        ----------
        config_json : Path or str
            Path to MOSDAC's official ``config.json``.
        download_path : str
            Fallback download directory if not set in the JSON.
        """
        # Build a minimal cfg dict so __init__ doesn't break
        minimal_cfg: dict[str, Any] = {
            "data": {
                "paths": {"raw": download_path},
                "date_range": {},
                "region": {},
            }
        }
        return cls(minimal_cfg, mosdac_config=config_json)

    # ── Authentication ──────────────────────────────
    def authenticate(self) -> bool:
        """Obtain access + refresh tokens from MOSDAC token endpoint.

        Returns ``True`` on success.  On failure logs the error and
        returns ``False``.

        Endpoint: ``POST /download_api/gettoken``
        """
        if not self.username or not self.password:
            log.error(
                "Cannot authenticate: username/password missing. "
                "Set MOSDAC_USERNAME & MOSDAC_PASSWORD env vars."
            )
            return False

        payload = {"username": self.username, "password": self.password}

        try:
            resp = requests.post(_TOKEN_URL, json=payload, timeout=30)

            if resp.status_code == 503:
                msg = resp.json().get("message", "Service unavailable")
                log.error(f"MOSDAC server maintenance: {msg}")
                return False

            if resp.status_code == 400:
                err = resp.json().get("error", "Validation error")
                log.error(f"MOSDAC auth validation error: {err}")
                return False

            if resp.status_code == 401:
                err = resp.json().get("error", "Invalid credentials")
                log.error(f"MOSDAC auth failed: {err}")
                return False

            resp.raise_for_status()
            tokens = resp.json()
            self._access_token = tokens.get("access_token")
            self._refresh_token = tokens.get("refresh_token")
            log.info(f"Authenticated with MOSDAC as '{self.username}'")
            return True

        except requests.RequestException as e:
            log.error(f"MOSDAC authentication request failed: {e}")
            return False

    def _refresh_access_token(self) -> bool:
        """Refresh an expired access token.

        Endpoint: ``POST /download_api/refresh-token``
        """
        if not self._refresh_token:
            return False

        try:
            resp = requests.post(
                _REFRESH_URL,
                json={"refresh_token": self._refresh_token},
                timeout=30,
            )
            if resp.status_code == 400:
                log.error(f"Token refresh validation error: {resp.json().get('error')}")
                return False

            resp.raise_for_status()
            tokens = resp.json()
            self._access_token = tokens.get("access_token")
            self._refresh_token = tokens.get("refresh_token")
            log.info("MOSDAC access token refreshed")
            return True

        except requests.RequestException as e:
            log.error(f"Token refresh failed: {e}")
            return False

    def logout(self) -> None:
        """End the MOSDAC session.

        Endpoint: ``POST /download_api/logout``
        """
        for attempt, delay in enumerate(_RETRY_DELAYS):
            try:
                resp = requests.post(
                    _LOGOUT_URL,
                    json={"username": self.username},
                    timeout=10,
                )
                if resp.status_code == 400:
                    log.warning(f"Logout validation error: {resp.json().get('error')}")
                    return
                resp.raise_for_status()
                log.info(f"Logout successful. Goodbye {self.username}!")
                return
            except (requests.ConnectionError, requests.Timeout):
                if attempt < len(_RETRY_DELAYS) - 1:
                    log.warning(f"Logout network error, retrying in {delay}s…")
                    time.sleep(delay)
                else:
                    log.error("Logout failed after multiple attempts")
            except requests.RequestException as e:
                log.error(f"Logout error: {e}")
                return

    # ── Search ──────────────────────────────────────
    def search(self, dataset_id: str) -> dict[str, Any]:
        """Search MOSDAC catalogue for a given ``datasetId``.

        No authentication is required for searching.

        Endpoint: ``GET /apios/datasets.json``

        Returns
        -------
        dict
            Raw JSON with keys: ``totalResults``, ``totalSizeMB``,
            ``itemsPerPage``, ``entries`` (list of granules).
        """
        params: dict[str, str] = {"datasetId": dataset_id}
        for key, val in [
            ("startTime", self._start_time),
            ("endTime", self._end_time),
            ("count", self._count),
            ("boundingBox", self._bounding_box),
            ("gId", self._gid),
        ]:
            if val:
                params[key] = val

        try:
            resp = requests.get(_SEARCH_URL, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("totalResults", 0)
                size_mb = data.get("totalSizeMB", 0)
                log.info(
                    f"Search '{dataset_id}': {total:,} files, "
                    f"{_format_size(size_mb)}"
                )
                return data
            else:
                err = resp.json().get("message", [resp.text])
                log.error(
                    f"Search failed (HTTP {resp.status_code}): "
                    f"{err[0] if isinstance(err, list) else err}"
                )
                return {}
        except (requests.ConnectionError, requests.Timeout):
            log.error("Search failed: no network connection")
            return {}
        except requests.RequestException as e:
            log.error(f"Search request error: {e}")
            return {}

    # ── Single-file download ────────────────────────
    def _download_file(
        self,
        record_id: str,
        identifier: str,
        prod_date: str | None,
        dataset_id: str,
        counter: int,
        total: int,
    ) -> Path | None:
        """Download one granule by its record ID.

        Endpoint: ``GET /download_api/download?id=<record_id>``

        Handles token refresh, rate limiting (429), retries on
        network errors, and deduplication (skip existing files).
        """
        # ── Resolve destination path ──────────────────
        if self.organize_by_date and prod_date:
            try:
                dt = datetime.strptime(prod_date, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                dt = None
            if dt:
                day_month = dt.strftime("%d") + dt.strftime("%b").upper()
                dest_dir = self.raw_dir / dataset_id / dt.strftime("%Y") / day_month
            else:
                dest_dir = self.raw_dir / dataset_id
        else:
            dest_dir = self.raw_dir / dataset_id

        ensure_dir(dest_dir)
        dest = dest_dir / identifier

        if dest.exists():
            log.debug(f"Already exists, skipping: {identifier}")
            return None  # skip

        headers = {"Authorization": f"Bearer {self._access_token}"}
        params = {"id": record_id}
        tmp_path = dest.with_suffix(dest.suffix + ".part")

        for attempt, delay in enumerate(_RETRY_DELAYS + [None]):  # type: ignore[list-item]
            try:
                resp = requests.get(
                    _DOWNLOAD_URL,
                    headers=headers,
                    params=params,
                    stream=True,
                    timeout=30,
                )

                # ── Handle error codes ────────────────
                if resp.status_code == 401:
                    code = resp.json().get("code", "")
                    if code == "INVALID_TOKEN":
                        if self._refresh_access_token():
                            headers["Authorization"] = f"Bearer {self._access_token}"
                            continue
                        log.error("Token refresh failed, aborting download")
                        return None
                    log.error(f"Auth error during download: {code}")
                    return None

                if resp.status_code == 404:
                    code = resp.json().get("code", "")
                    if code == "NOT_RELEASED":
                        log.warning(
                            f"Product not released on MOSDAC: {identifier}"
                        )
                        return None

                if resp.status_code == 429:
                    body = resp.json()
                    err_type = body.get("type", "")
                    msg = body.get("message", "Rate limited")
                    if err_type == "daily_limit":
                        log.error(f"Daily download quota reached: {msg}")
                        return None
                    # minute_limit → wait and retry
                    log.warning(f"Rate limited ({err_type}): {msg}. Waiting 20s…")
                    time.sleep(20)
                    continue

                if resp.status_code == 400:
                    err = resp.json().get("error", "Validation error")
                    log.error(f"Download validation error: {err}")
                    return None

                resp.raise_for_status()

                # ── Check Content-Disposition ─────────
                cd = resp.headers.get("Content-Disposition", "")
                if "filename=" not in cd:
                    log.warning(f"{identifier}: not available on server, skipping")
                    return None

                total_size = int(resp.headers.get("Content-Length", 0))

                # ── Clean up previous partial download ─
                if tmp_path.exists():
                    tmp_path.unlink()

                log.info(
                    f"[{counter}/{total}] Downloading: {identifier} "
                    f"({total_size / (1024 * 1024):.2f} MB)"
                )

                # ── Stream to disk ────────────────────
                with open(tmp_path, "wb") as f:
                    if HAS_TQDM:
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=identifier,
                            leave=False,
                        ) as pbar:
                            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        downloaded = 0
                        for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                pct = downloaded / total_size * 100 if total_size else 0
                                sys.stdout.write(f"\r  Progress: {pct:.1f}%")
                                sys.stdout.flush()
                        sys.stdout.write("\n")

                # Rename .part → final
                tmp_path.rename(dest)
                return dest

            except (requests.ConnectionError, requests.Timeout) as e:
                if delay is None:
                    log.error(
                        f"Download failed after {len(_RETRY_DELAYS)} retries: "
                        f"{identifier}"
                    )
                    if tmp_path.exists():
                        tmp_path.unlink()
                    return None
                log.warning(f"Network error, retrying in {delay}s… ({e})")
                if tmp_path.exists():
                    tmp_path.unlink()
                time.sleep(delay)

            except requests.RequestException as e:
                log.error(f"Download error for {identifier}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                return None

        return None

    # ── Paginated bulk download ─────────────────────
    def bulk_download(
        self,
        dataset_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Path]:
        """Download all matching granules via paginated search.

        If ``dataset_id`` is not provided, iterates over all
        ``dataset_ids`` from the config.

        Parameters
        ----------
        dataset_id : str, optional
            MOSDAC datasetId (e.g. ``"3RIMG_L1B_STD"``).
        start_date, end_date : str, optional
            Override the date range from config.

        Returns
        -------
        list[Path]
            Paths to all successfully downloaded files.
        """
        # Temporarily override date range if provided
        orig_start, orig_end = self._start_time, self._end_time
        if start_date:
            self._start_time = start_date
        if end_date:
            self._end_time = end_date

        ids = [dataset_id] if dataset_id else self._dataset_ids
        if not ids:
            log.error(
                "No dataset_ids configured. Set 'data.dataset_ids' in "
                "default.yaml or provide a mosdac_config.json."
            )
            return []

        all_downloaded: list[Path] = []

        try:
            for ds_id in ids:
                downloaded = self._download_dataset(ds_id)
                all_downloaded.extend(downloaded)
        finally:
            # Restore original date range
            self._start_time, self._end_time = orig_start, orig_end

        log.info(
            f"Bulk download complete: {len(all_downloaded)} files across "
            f"{len(ids)} dataset(s)"
        )
        return all_downloaded

    def _download_dataset(self, dataset_id: str) -> list[Path]:
        """Search + paginated download for a single datasetId."""
        # Initial search to get total count
        search_result = self.search(dataset_id)
        if not search_result:
            return []

        total_files = search_result.get("totalResults", 0)
        if total_files == 0:
            log.warning(f"No files found for dataset '{dataset_id}'")
            return []

        log.info(f"Starting download of {total_files:,} files for '{dataset_id}'")

        downloaded: list[Path] = []
        counter = 1
        start_index = 1

        # Build search params
        params: dict[str, str] = {"datasetId": dataset_id}
        for key, val in [
            ("startTime", self._start_time),
            ("endTime", self._end_time),
            ("count", self._count),
            ("boundingBox", self._bounding_box),
            ("gId", self._gid),
        ]:
            if val:
                params[key] = val

        while counter <= total_files:
            params["startIndex"] = str(start_index)

            try:
                resp = requests.get(_SEARCH_URL, params=params, timeout=60)
                if resp.status_code != 200:
                    log.error(f"Paginated search failed (HTTP {resp.status_code})")
                    break

                page = resp.json()
                entries = page.get("entries", [])
                if not entries:
                    break

                for entry in entries:
                    identifier = entry.get("identifier", "")
                    record_id = entry.get("id", "")
                    prod_date = entry.get("updated")

                    path = self._download_file(
                        record_id=record_id,
                        identifier=identifier,
                        prod_date=prod_date,
                        dataset_id=dataset_id,
                        counter=counter,
                        total=total_files,
                    )

                    if path and path.exists():
                        downloaded.append(path)

                    counter += 1

                start_index += _BATCH_SIZE

            except KeyboardInterrupt:
                log.warning("Download interrupted by user")
                break
            except requests.RequestException as e:
                log.error(f"Paginated fetch error: {e}")
                break

        return downloaded


# ── Helpers ─────────────────────────────────────────
def _format_size(size_mb: float) -> str:
    """Human-readable size string."""
    if size_mb < 1024:
        return f"{size_mb:,.2f} MB"
    elif size_mb < 1024**2:
        return f"{size_mb / 1024:,.2f} GB"
    else:
        return f"{size_mb / (1024**2):,.2f} TB"


# ── CLI entry point ─────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download INSAT-3DR/3DS data from MOSDAC"
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--mosdac-config",
        default=None,
        help="Path to official MOSDAC config.json",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Override datasetId (e.g. 3RIMG_L1B_STD)",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help=(
            "Run the unmodified official MOSDAC mdapi.py script "
            "instead of the programmatic client. Uses config.json "
            "from the project root."
        ),
    )
    args = parser.parse_args()

    # ── Mode 1: Official script ───────────────────
    if args.official:
        rc = run_official_mdapi(
            config_json=args.mosdac_config,
        )
        sys.exit(rc)

    # ── Mode 2: Programmatic client ───────────────
    if args.mosdac_config:
        client = MOSDACClient.from_mosdac_json(args.mosdac_config)
    else:
        cfg = load_config(args.config)
        client = MOSDACClient(cfg)

    if not client.authenticate():
        log.error("Cannot proceed without authentication. Exiting.")
        return

    try:
        client.bulk_download(dataset_id=args.dataset_id)
    finally:
        client.logout()


if __name__ == "__main__":
    main()
