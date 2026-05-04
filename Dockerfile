# ── Stage 1: Builder ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder

# Install system dependencies with retry logic
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip python3.11-dev \
      libgdal-dev libhdf5-dev libnetcdf-dev \
      gcc g++ && \
    rm -rf /var/lib/apt/lists/* || \
    (echo "First apt-get failed, retrying..." && sleep 10 && \
     apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip python3.11-dev \
      libgdal-dev libhdf5-dev libnetcdf-dev \
      gcc g++ && \
     rm -rf /var/lib/apt/lists/*)

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

# Install runtime dependencies with retry logic  
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 libgdal30 libhdf5-103 libnetcdf19 \
      libxcb1 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/* || \
    (echo "First apt-get failed, retrying..." && sleep 10 && \
     apt-get update && apt-get install -y --no-install-recommends \
      python3.11 libgdal30 libhdf5-103 libnetcdf19 \
      libxcb1 libsm6 libxext6 libxrender1 && \
     rm -rf /var/lib/apt/lists/*)

# Create non-root user for security
RUN useradd -m appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .

ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8080
USER appuser

EXPOSE 8080

# Cloud Run will handle health checks via PORT env var and TCP probe

CMD ["sh", "-c", "uvicorn meghdoot.deploy.api:app --host 0.0.0.0 --port ${PORT}"]
