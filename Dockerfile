# ── Stage 1: Builder ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
  gdal-bin libgdal-dev libhdf5-dev libnetcdf-dev \
    gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
# Install everything into a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

# ── Stage 2: Runtime ──────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 libgdal30 libhdf5-103 libnetcdf19 \
  curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .

ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8080
USER appuser

EXPOSE 8080

# Healthcheck for Cloud Run
HEALTHCHECK --interval=30s --timeout=3s \
  CMD sh -c 'curl -f http://localhost:${PORT:-8080}/health || exit 1'

CMD ["sh", "-c", "uvicorn meghdoot.deploy.api:app --host 0.0.0.0 --port ${PORT}"]
