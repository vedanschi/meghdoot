#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env.gcp" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.gcp"
fi

if [[ -f "${REPO_ROOT}/.env.backend" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.backend"
fi

if [[ -f "${REPO_ROOT}/frontend/.env.production" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/frontend/.env.production"
fi

: "${PROJECT_ID:?PROJECT_ID is required (set in .env.gcp or shell)}"
: "${REGION:?REGION is required (set in .env.gcp or shell)}"

BACKEND_SERVICE_NAME="${BACKEND_SERVICE_NAME:-meghdoot-backend}"
FRONTEND_SERVICE_NAME="${FRONTEND_SERVICE_NAME:-meghdoot-frontend}"
SCHEDULER_JOB_NAME="${SCHEDULER_JOB_NAME:-meghdoot-nowcast}"
SCHEDULER_CRON="${SCHEDULER_CRON:-*/30 * * * *}"
SCHEDULER_TIME_ZONE="${SCHEDULER_TIME_ZONE:-UTC}"

BACKEND_RUNTIME_SA_NAME="${BACKEND_RUNTIME_SA_NAME:-meghdoot-runtime}"
SCHEDULER_SA_NAME="${SCHEDULER_SA_NAME:-meghdoot-scheduler}"

BACKEND_RUNTIME_SA_EMAIL="${BACKEND_RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA_EMAIL="${SCHEDULER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

ARTIFACT_REPO="meghdoot"

echo "Using project: ${PROJECT_ID}"
echo "Using region: ${REGION}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# 1) Enable required APIs.
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com \
  iamcredentials.googleapis.com

# 2) Ensure Artifact Registry repo exists.
if ! gcloud artifacts repositories describe "${ARTIFACT_REPO}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Meghdoot container images"
fi

# 3) Ensure service accounts exist.
if ! gcloud iam service-accounts describe "${BACKEND_RUNTIME_SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${BACKEND_RUNTIME_SA_NAME}" \
    --display-name="Meghdoot Backend Runtime"
fi

if ! gcloud iam service-accounts describe "${SCHEDULER_SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SCHEDULER_SA_NAME}" \
    --display-name="Meghdoot Scheduler"
fi

# 4) Grant storage rights for model download + forecast uploads.
# Bucket-level is tighter than project-wide role grants.
if [[ -n "${MEGHDOOT_GCS_BUCKET:-}" ]]; then
  gsutil iam ch "serviceAccount:${BACKEND_RUNTIME_SA_EMAIL}:objectAdmin" "gs://${MEGHDOOT_GCS_BUCKET}" || true
fi

# 5) Deploy backend.
export SERVICE_NAME="${BACKEND_SERVICE_NAME}"
export SERVICE_ACCOUNT="${BACKEND_RUNTIME_SA_EMAIL}"
export SCHEDULER_SERVICE_ACCOUNT="${SCHEDULER_SA_EMAIL}"
export CHECKPOINT_GCS_URI="${CHECKPOINT_GCS_URI:-${MEGHDOOT_DIFFUSION_CHECKPOINT_GCS_URI:-}}"
export GCS_BUCKET="${GCS_BUCKET:-${MEGHDOOT_GCS_BUCKET:-}}"

if [[ -n "${MOSDAC_USERNAME_SECRET:-}" ]]; then
  export MOSDAC_USERNAME_SECRET
fi
if [[ -n "${MOSDAC_PASSWORD_SECRET:-}" ]]; then
  export MOSDAC_PASSWORD_SECRET
fi

"${SCRIPT_DIR}/deploy_cloud_run_backend.sh"

BACKEND_URL="$(gcloud run services describe "${BACKEND_SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')"

# 6) Deploy frontend.
export FRONTEND_SERVICE_NAME
"${SCRIPT_DIR}/deploy_cloud_run_frontend.sh"

FRONTEND_URL="$(gcloud run services describe "${FRONTEND_SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')"

# 7) Allow scheduler SA to invoke backend.
gcloud run services add-iam-policy-binding "${BACKEND_SERVICE_NAME}" \
  --region "${REGION}" \
  --member "serviceAccount:${SCHEDULER_SA_EMAIL}" \
  --role "roles/run.invoker" >/dev/null

# 8) Create or update Cloud Scheduler job.
if gcloud scheduler jobs describe "${SCHEDULER_JOB_NAME}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "${SCHEDULER_JOB_NAME}" \
    --location "${REGION}" \
    --schedule "${SCHEDULER_CRON}" \
    --time-zone "${SCHEDULER_TIME_ZONE}" \
    --uri "${BACKEND_URL}/forecast/nowcast" \
    --http-method POST \
    --oidc-service-account-email "${SCHEDULER_SA_EMAIL}" \
    --oidc-token-audience "${BACKEND_URL}"
else
  gcloud scheduler jobs create http "${SCHEDULER_JOB_NAME}" \
    --location "${REGION}" \
    --schedule "${SCHEDULER_CRON}" \
    --time-zone "${SCHEDULER_TIME_ZONE}" \
    --uri "${BACKEND_URL}/forecast/nowcast" \
    --http-method POST \
    --oidc-service-account-email "${SCHEDULER_SA_EMAIL}" \
    --oidc-token-audience "${BACKEND_URL}"
fi

echo ""
echo "Setup complete"
echo "Backend URL:  ${BACKEND_URL}"
echo "Frontend URL: ${FRONTEND_URL}"
echo "Scheduler:    ${SCHEDULER_JOB_NAME} (${SCHEDULER_CRON})"
echo ""
echo "To test scheduler once now:"
echo "gcloud scheduler jobs run ${SCHEDULER_JOB_NAME} --location ${REGION}"
