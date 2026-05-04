# ── Stage 1: Builder ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder
# Prefer HTTPS apt sources (helps when HTTP port 80 is blocked) and force IPv4
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list || true

# Install system dependencies with retry logic and IPv4/HTTPS options
RUN apt-get update -o Acquire::ForceIPv4=true && \
    apt-get install -y -o Acquire::Retries=3 -o Acquire::ForceIPv4=true --no-install-recommends \
      python3.11 python3.11-venv python3-pip python3.11-dev \
      libgdal-dev libhdf5-dev libnetcdf-dev \
      libxcb1 libx11-6 libsm6 libxext6 libxrender1 \
      gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Create and activate virtual environment, install with retries
RUN python3.11 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir . || \
    (echo "First pip install failed, retrying..." && sleep 10 && \
     pip install --no-cache-dir .)

# ── Stage 2: Runtime ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Ensure HTTPS apt sources in runtime as well
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list || true

# No runtime apt-get required: system libraries will be copied from the builder stage

# Create non-root user for security
RUN useradd -m appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# Copy system libraries from builder to avoid runtime apt network dependency
# (copies only the directories that contain the needed shared libraries)
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu

# Copy application files
COPY . .

ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8080
USER appuser

EXPOSE 8080

# Cloud Run will handle health checks via PORT env var and TCP probe

CMD ["sh", "-c", "uvicorn meghdoot.deploy.api:app --host 0.0.0.0 --port ${PORT}"]
