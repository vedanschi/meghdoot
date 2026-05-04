# ── Stage 1: Builder ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder

# Force HTTPS apt sources before any package operation; Cloud Build is failing on HTTP archive access.
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.edge.kernel.org/ubuntu|g; s|https://archive.ubuntu.com/ubuntu|https://mirrors.edge.kernel.org/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true

# Install Python, build tools, and the runtime libraries needed by OpenCV, rasterio, and h5py.
RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=3 && \
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

# Rewrite apt sources to the working HTTPS mirror BEFORE any apt command.
# This prevents Cloud Run from timing out on HTTP archive.ubuntu.com.
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.edge.kernel.org/ubuntu|g; s|https://archive.ubuntu.com/ubuntu|https://mirrors.edge.kernel.org/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true

# CRITICAL FIX: Install X11/graphics runtime libraries directly in the runtime stage.
# This avoids the fragile and incomplete approach of copying /usr/lib/x86_64-linux-gnu from builder.
# cv2's native bindings require libx11-6, libxcb1, libsm6, libxext6, libxrender1, and libglib2.0-0.
# By installing them here via apt, we ensure all symlinks and dependencies are correct.
RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends \
      libx11-6 libxcb1 libsm6 libxext6 libxrender1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the prepared venv and the system Python runtime it expects.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/bin/python3.11 /usr/bin/python3.11
COPY --from=builder /usr/bin/python3 /usr/bin/python3
COPY --from=builder /usr/lib/python3.11 /usr/lib/python3.11

# Copy the app code.
COPY configs ./configs
COPY src ./src
COPY pyproject.toml README.md requirements-pinned.txt ./

ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Use the venv interpreter directly so Cloud Run gets a deterministic startup command.
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-m", "uvicorn", "meghdoot.deploy.api:app", "--host", "0.0.0.0", "--port", "8080"]
