#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/frontend"

if [[ -f "${FRONTEND_DIR}/.env.production" ]]; then
  # shellcheck disable=SC1091
  source "${FRONTEND_DIR}/.env.production"
fi

: "${PROJECT_ID:?PROJECT_ID is required}"
: "${FRONTEND_SERVICE_NAME:?FRONTEND_SERVICE_NAME is required}"
: "${REGION:?REGION is required}"

IMAGE_NAME="${FRONTEND_IMAGE_NAME:-meghdoot-frontend}"
IMAGE_TAG="${FRONTEND_IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/meghdoot/${IMAGE_NAME}:${IMAGE_TAG}"

NEXT_PUBLIC_GCS_BUCKET="${NEXT_PUBLIC_GCS_BUCKET:-meghdoot-satellite-data}"
NEXT_PUBLIC_FORECAST_PREFIX="${NEXT_PUBLIC_FORECAST_PREFIX:-forecasts/latest}"
NEXT_PUBLIC_REFRESH_MS="${NEXT_PUBLIC_REFRESH_MS:-60000}"
NEXT_PUBLIC_STALE_AFTER_MINUTES="${NEXT_PUBLIC_STALE_AFTER_MINUTES:-90}"

gcloud config set project "${PROJECT_ID}" >/dev/null

cd "${FRONTEND_DIR}"
gcloud builds submit --tag "${IMAGE_URI}" .

gcloud run deploy "${FRONTEND_SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 512Mi \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8080 \
  --set-env-vars "PORT=8080,NEXT_PUBLIC_GCS_BUCKET=${NEXT_PUBLIC_GCS_BUCKET},NEXT_PUBLIC_FORECAST_PREFIX=${NEXT_PUBLIC_FORECAST_PREFIX},NEXT_PUBLIC_REFRESH_MS=${NEXT_PUBLIC_REFRESH_MS},NEXT_PUBLIC_STALE_AFTER_MINUTES=${NEXT_PUBLIC_STALE_AFTER_MINUTES}"

FRONTEND_URL=$(gcloud run services describe "${FRONTEND_SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')
echo "Frontend deployed: ${FRONTEND_URL}"
