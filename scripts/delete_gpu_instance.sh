#!/usr/bin/env bash

set -euo pipefail

if [[ -f ".env.gcp" ]]; then
  # shellcheck disable=SC1091
  source .env.gcp
fi

INSTANCE_NAME="${INSTANCE_NAME:-}"
ZONE="${ZONE:-}"
PROJECT_ID="${PROJECT_ID:-}"
KEEP_DISKS="${KEEP_DISKS:-0}"
YES="${YES:-0}"

usage() {
  echo "Usage: INSTANCE_NAME=name ZONE=zone PROJECT_ID=project ./scripts/delete_gpu_instance.sh"
  echo "Optional: KEEP_DISKS=1 YES=1"
}

if [[ -z "${INSTANCE_NAME}" || -z "${ZONE}" || -z "${PROJECT_ID}" ]]; then
  usage
  exit 1
fi

gcloud config set project "${PROJECT_ID}" >/dev/null

if ! gcloud compute instances describe "${INSTANCE_NAME}" --zone "${ZONE}" >/dev/null 2>&1; then
  echo "Instance not found: ${INSTANCE_NAME} in ${ZONE}"
  exit 1
fi

echo "Instance details:"
gcloud compute instances describe "${INSTANCE_NAME}" --zone "${ZONE}" \
  --format='table(name,zone,status,machineType.basename(),guestAccelerators[].acceleratorType.basename(),disks[].source.basename())'

if [[ "${YES}" != "1" ]]; then
  echo ""
  echo "Type the instance name to confirm deletion:"
  read -r CONFIRM
  if [[ "${CONFIRM}" != "${INSTANCE_NAME}" ]]; then
    echo "Confirmation mismatch. Aborting."
    exit 1
  fi
fi

DELETE_ARGS=("${INSTANCE_NAME}" "--zone" "${ZONE}" "--quiet")
if [[ "${KEEP_DISKS}" == "1" ]]; then
  DELETE_ARGS+=("--keep-disks=all")
fi

gcloud compute instances delete "${DELETE_ARGS[@]}"

echo "Deleted instance: ${INSTANCE_NAME}"
if [[ "${KEEP_DISKS}" == "1" ]]; then
  echo "Attached disks were kept."
else
  echo "Attached disks used by the instance were deleted according to default policy."
fi
