#!/usr/bin/env bash

# Deploy Meghdoot-AI backend to Cloud Run with an L4 GPU.
#
# Required environment variables:
#   PROJECT_ID                 GCP project ID
#   SERVICE_NAME               Cloud Run service name
#   REGION                     GCP region (e.g. us-central1)
#   SERVICE_ACCOUNT            Runtime service account email
#   CHECKPOINT_GCS_URI         gs://... URI for the diffusion checkpoint
#   GCS_BUCKET                 Forecast output bucket name
#
# Optional environment variables:
#   MOSDAC_USERNAME_SECRET     Secret Manager secret name for MOSDAC username
#   MOSDAC_PASSWORD_SECRET     Secret Manager secret name for MOSDAC password
#   SCHEDULER_SERVICE_ACCOUNT  Cloud Scheduler service account email for invoker binding
#   IMAGE_NAME                 Artifact Registry image name (default: meghdoot-backend)
#   IMAGE_TAG                  Image tag (default: latest)
#
# Example:
#   PROJECT_ID=my-project \
#   SERVICE_NAME=meghdoot-backend \
#   REGION=us-central1 \
#   SERVICE_ACCOUNT=meghdoot-runtime@my-project.iam.gserviceaccount.com \
#   CHECKPOINT_GCS_URI=gs://my-bucket/checkpoints/diffusion_epoch290.pt \
#   GCS_BUCKET=megdhoot-satellite-data \
#   ./scripts/deploy_cloud_run_backend.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env.backend" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.backend"
fi

: "${PROJECT_ID:?PROJECT_ID is required}"
: "${SERVICE_NAME:?SERVICE_NAME is required}"
: "${REGION:?REGION is required}"
: "${SERVICE_ACCOUNT:?SERVICE_ACCOUNT is required}"
: "${CHECKPOINT_GCS_URI:?CHECKPOINT_GCS_URI is required}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"

VAE_CHECKPOINT_GCS_URI="${VAE_CHECKPOINT_GCS_URI:-${MEGHDOOT_VAE_CHECKPOINT_GCS_URI:-}}"
if [[ -z "${VAE_CHECKPOINT_GCS_URI}" ]]; then
  echo "VAE_CHECKPOINT_GCS_URI or MEGHDOOT_VAE_CHECKPOINT_GCS_URI must be set"
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-meghdoot-backend}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/meghdoot/${IMAGE_NAME}:${IMAGE_TAG}"

# Build and push the container image.
echo "Building container image: ${IMAGE_URI}"
gcloud config set project "${PROJECT_ID}" >/dev/null

gcloud builds submit --tag "${IMAGE_URI}" .

# Build the secret flag list only when secrets are supplied.
SECRET_FLAGS=()
if [[ -n "${MOSDAC_USERNAME_SECRET:-}" ]]; then
  SECRET_FLAGS+=("--set-secrets=MOSDAC_USERNAME=${MOSDAC_USERNAME_SECRET}:latest")
fi
if [[ -n "${MOSDAC_PASSWORD_SECRET:-}" ]]; then
  SECRET_FLAGS+=("--set-secrets=MOSDAC_PASSWORD=${MOSDAC_PASSWORD_SECRET}:latest")
fi

# Deploy the service.
echo "Deploying Cloud Run service: ${SERVICE_NAME}"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --execution-environment gen2 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --concurrency 1 \
  --timeout 3600 \
  --min-instances 0 \
  --max-instances 1 \
  --port 8080 \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "PORT=8080,MEGHDOOT_DEVICE=cuda,MEGHDOOT_GCS_BUCKET=${GCS_BUCKET},MEGHDOOT_VAE_CHECKPOINT_GCS_URI=${VAE_CHECKPOINT_GCS_URI},MEGHDOOT_DIFFUSION_CHECKPOINT_GCS_URI=${CHECKPOINT_GCS_URI},MEGHDOOT_RUNTIME_DIR=/tmp/meghdoot" \
  "${SECRET_FLAGS[@]}"


# Optionally grant Cloud Scheduler invoker permissions.
if [[ -n "${SCHEDULER_SERVICE_ACCOUNT:-}" ]]; then
  echo "Granting Cloud Run invoker to ${SCHEDULER_SERVICE_ACCOUNT}"
  gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
    --region "${REGION}" \
    --member "serviceAccount:${SCHEDULER_SERVICE_ACCOUNT}" \
    --role "roles/run.invoker"
fi

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')
echo ""
echo "Deployment complete. Service URL: ${SERVICE_URL}"
echo ""
echo "Next steps:"
echo "1. Verify the service health: curl ${SERVICE_URL}/health"
echo "2. Trigger the forecast endpoint: curl -X POST ${SERVICE_URL}/forecast/nowcast"
echo "3. Point Cloud Scheduler at ${SERVICE_URL}/forecast/nowcast"
