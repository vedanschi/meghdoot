# ── Stage 1: Builder ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder

# Install Python, build tools, and the runtime libraries needed by OpenCV, rasterio, and h5py.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip python3.11-dev \
      libgdal-dev libhdf5-dev libnetcdf-dev \
      libxcb1 libx11-6 libsm6 libxext6 libxrender1 \
      gcc g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md requirements-pinned.txt ./
COPY src ./src

# Build the virtual environment with pinned backend dependencies.
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-pinned.txt && \
    /opt/venv/bin/pip install --no-cache-dir --no-deps .

# ── Stage 2: Runtime ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /app

# Copy the prepared venv and the system Python runtime it expects.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/bin/python3.11 /usr/bin/python3.11
COPY --from=builder /usr/bin/python3 /usr/bin/python3
COPY --from=builder /usr/lib/python3.11 /usr/lib/python3.11

# Copy the native libraries needed by the pinned Python wheels.
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu

# Copy the app code and the MOSDAC config used by the runtime pipeline.
COPY configs ./configs
COPY config.json ./config.json
COPY mdapi.py ./mdapi.py
COPY src ./src
COPY pyproject.toml README.md requirements-pinned.txt ./

ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Use the venv interpreter directly so Cloud Run gets a deterministic startup command.
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-m", "uvicorn", "meghdoot.deploy.api:app", "--host", "0.0.0.0", "--port", "8080"]
