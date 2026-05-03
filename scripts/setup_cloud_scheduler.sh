#!/bin/bash
# setup_cloud_scheduler.sh
#
# Automates Cloud Scheduler setup for Meghdoot nowcasting pipeline
#
# Usage:
#   ./setup_cloud_scheduler.sh --project YOUR_PROJECT_ID \
#                             --instance your-instance-name \
#                             --zone us-central1-a \
#                             --interval 30
#
# This script will:
# 1. Get the internal IP of your Compute Engine instance
# 2. Create a service account for Cloud Scheduler
# 3. Grant necessary GCS permissions
# 4. Create the Cloud Scheduler job

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Defaults
INTERVAL=30  # minutes

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --instance)
            INSTANCE_NAME="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --help)
            cat << 'EOF'
Usage: ./setup_cloud_scheduler.sh [OPTIONS]

Options:
  --project PROJECT_ID       GCP project ID (required)
  --instance INSTANCE_NAME   Compute Engine instance name (required)
  --zone ZONE                GCP zone (default: us-central1-a)
  --interval MINUTES         Scheduler interval in minutes (default: 30)
  --help                     Show this help message

Example:
  ./setup_cloud_scheduler.sh --project my-gcp-project \
                            --instance meghdoot-instance \
                            --zone us-central1-a
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PROJECT_ID" || -z "$INSTANCE_NAME" ]]; then
    echo -e "${RED}Error: --project and --instance are required${NC}"
    exit 1
fi

ZONE="${ZONE:-us-central1-a}"
REGION="${ZONE%-*}"  # Extract region from zone (e.g., us-central1 from us-central1-a)

echo -e "${YELLOW}Meghdoot Cloud Scheduler Setup${NC}"
echo "=========================================="
echo "Project:  $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Zone:     $ZONE"
echo "Region:   $REGION"
echo "Interval: $INTERVAL minutes"
echo ""

# Set default project
gcloud config set project "$PROJECT_ID"

# Step 1: Get instance internal IP
echo -e "${YELLOW}[1/5]${NC} Fetching instance internal IP..."
INSTANCE_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --format="get(networkInterfaces[0].networkIP)" 2>/dev/null)

if [[ -z "$INSTANCE_IP" ]]; then
    echo -e "${RED}Error: Could not find instance $INSTANCE_NAME in zone $ZONE${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Instance IP: $INSTANCE_IP"

# Step 2: Create service account
SA_NAME="meghdoot-scheduler"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo -e "${YELLOW}[2/5]${NC} Creating service account..."
if gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
    echo -e "${GREEN}✓${NC} Service account already exists: $SA_EMAIL"
else
    gcloud iam service-accounts create "$SA_NAME" \
      --display-name="Meghdoot Scheduler" &>/dev/null
    echo -e "${GREEN}✓${NC} Created service account: $SA_EMAIL"
fi

# Step 3: Grant GCS permissions
echo -e "${YELLOW}[3/5]${NC} Granting GCS bucket permissions..."
BUCKET_NAME="megdhoot-satellite-data"

if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    gsutil iam ch "serviceAccount:${SA_EMAIL}:objectAdmin" "gs://${BUCKET_NAME}" 2>/dev/null || true
    echo -e "${GREEN}✓${NC} Granted objectAdmin on gs://$BUCKET_NAME"
else
    echo -e "${YELLOW}⚠${NC} Warning: Bucket gs://$BUCKET_NAME not found or not accessible"
    echo "   You can grant permissions manually later:"
    echo "   gsutil iam ch serviceAccount:${SA_EMAIL}:objectAdmin gs://${BUCKET_NAME}"
fi

# Step 4: Create or update Cloud Scheduler job
JOB_NAME="meghdoot-nowcast"
SCHEDULE="*/${INTERVAL} * * * *"
URL="http://${INSTANCE_IP}:8000/forecast/nowcast"

echo -e "${YELLOW}[4/5]${NC} Creating Cloud Scheduler job..."

if gcloud scheduler jobs describe "$JOB_NAME" --location="$REGION" &>/dev/null; then
    echo -e "${GREEN}✓${NC} Job already exists; updating..."
    gcloud scheduler jobs update http "$JOB_NAME" \
      --location="$REGION" \
      --schedule="$SCHEDULE" \
      --uri="$URL" \
      --http-method=POST \
      --oidc-service-account-email="$SA_EMAIL" \
      --oidc-token-audience="$URL" \
      --time-zone=UTC \
      --attempt-deadline=600s \
      2>/dev/null
else
    echo -e "${GREEN}✓${NC} Creating new job..."
    gcloud scheduler jobs create http "$JOB_NAME" \
      --location="$REGION" \
      --schedule="$SCHEDULE" \
      --uri="$URL" \
      --http-method=POST \
      --oidc-service-account-email="$SA_EMAIL" \
      --oidc-token-audience="$URL" \
      --time-zone=UTC \
      --attempt-deadline=600s \
      2>/dev/null
fi

# Step 5: Test the job (optional)
echo -e "${YELLOW}[5/5]${NC} Testing scheduler job..."
if gcloud scheduler jobs run "$JOB_NAME" --location="$REGION" &>/dev/null; then
    echo -e "${GREEN}✓${NC} Scheduled job triggered successfully"
    echo ""
    echo -e "${YELLOW}Checking execution status...${NC}"
    sleep 2
    gcloud scheduler jobs describe "$JOB_NAME" \
      --location="$REGION" \
      --format="value(lastAttemptTime, status)"
else
    echo -e "${YELLOW}⚠${NC} Job creation succeeded but test execution failed"
    echo "   This may be normal; check Cloud Scheduler logs for details"
fi

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Start the FastAPI server on your Compute Engine instance:"
echo "   cd /home/vedanschi/meghdoot && source .venv/bin/activate"
echo "   uvicorn meghdoot.deploy.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "2. Monitor scheduler executions:"
echo "   gcloud scheduler jobs describe $JOB_NAME --location=$REGION"
echo ""
echo "3. View recent logs:"
echo "   gcloud scheduler jobs run $JOB_NAME --location=$REGION"
echo ""
echo "Useful commands:"
echo "  Describe job:     gcloud scheduler jobs describe $JOB_NAME --location=$REGION"
echo "  Manual trigger:   gcloud scheduler jobs run $JOB_NAME --location=$REGION"
echo "  View history:     gcloud scheduler jobs describe $JOB_NAME --location=$REGION --format=json | jq '.lastAttemptTime'"
echo "  Delete job:       gcloud scheduler jobs delete $JOB_NAME --location=$REGION"
