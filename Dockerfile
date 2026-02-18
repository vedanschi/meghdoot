# ─────────────────────────────────────────────────
# Meghdoot-AI  –  Production Dockerfile
# ─────────────────────────────────────────────────
# Uses NVIDIA CUDA base for GPU inference.
# Handles the "dependency hell" of GDAL + geospatial libs.
#
# Build:   docker build -t meghdoot-ai .
# Run:     docker run --gpus all -p 8000:8000 meghdoot-ai
# ─────────────────────────────────────────────────

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System dependencies (GDAL, HDF5, OpenCV, etc.) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip python3.11-dev \
        gdal-bin libgdal-dev \
        libhdf5-dev libnetcdf-dev \
        libgl1-mesa-glx libglib2.0-0 \
        curl git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set GDAL environment for pip build
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# ── Python dependencies ──────────────────────────
COPY pyproject.toml .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ".[dev]"

# ── Application code ─────────────────────────────
COPY configs/ configs/
COPY src/ src/
RUN python -m pip install -e .

# ── Default: run FastAPI inference server ────────
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "meghdoot.deploy.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
