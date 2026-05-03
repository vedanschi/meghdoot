# Cloud Scheduler Setup for Meghdoot Nowcasting Pipeline

Since your infrastructure is on Google Cloud (Compute Engine + GCS bucket), you can use **Cloud Scheduler** to trigger the forecast pipeline every 30 minutes. This is simpler than cron because it has built-in monitoring, retries, and logging.

## Architecture

```
┌──────────────────────────────────┐
│ Cloud Scheduler (every 30 min)   │
└────────────────┬─────────────────┘
                 │ HTTP POST
                 v
  ┌─────────────────────────────────┐
  │ Compute Engine Instance          │
  │ FastAPI Server (port 8000)       │
  │                                  │
  │ POST /forecast/nowcast ─────────┐│
  │   ├─ Download INSAT frames     │ │
  │   ├─ Preprocess to tensors     │ │
  │   ├─ Run diffusion inference   │ │
  │   └─ Publish to GCS bucket     │ │
  │                                 │ │
  └─────────────────────────────────┘ │
           │                           │
           └───> Cloud Logging ────────┘
```

## Setup Steps

### Step 1: Verify Your Compute Engine Instance is Running

Make sure your instance is running and has the FastAPI server up:

```bash
# SSH into your Compute Engine instance
gcloud compute ssh your-instance-name --zone your-zone

# Inside instance: start the API server
cd /home/vedanschi/meghdoot
source .venv/bin/activate
uvicorn meghdoot.deploy.api:app --host 0.0.0.0 --port 8000 &
```

Or add this to your instance startup script so it runs automatically on boot:

```bash
#!/bin/bash
cd /home/vedanschi/meghdoot
source .venv/bin/activate
nohup uvicorn meghdoot.deploy.api:app --host 0.0.0.0 --port 8000 > /var/log/meghdoot_api.log 2>&1 &
```

### Step 2: Get Your Instance's Internal IP

```bash
gcloud compute instances describe your-instance-name \
  --zone your-zone \
  --format="get(networkInterfaces[0].networkIP)"
```

Note this IP address (e.g., `10.128.0.2`).

### Step 3: Create Cloud Scheduler Job

#### Option A: Using gcloud CLI (Recommended)

```bash
# Set variables
PROJECT_ID="your-gcp-project-id"
INSTANCE_IP="10.128.0.2"          # From step 2
INSTANCE_ZONE="us-central1-a"
SERVICE_ACCOUNT="your-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Create Cloud Scheduler job
gcloud scheduler jobs create http meghdoot-nowcast \
  --location=us-central1 \
  --schedule="*/30 * * * *" \
  --uri="http://${INSTANCE_IP}:8000/forecast/nowcast" \
  --http-method=POST \
  --oidc-service-account-email="${SERVICE_ACCOUNT}" \
  --oidc-token-audience="http://${INSTANCE_IP}:8000" \
  --time-zone="UTC" \
  --project="${PROJECT_ID}"
```

#### Option B: Using Terraform

Create `scheduler.tf`:

```hcl
resource "google_cloud_scheduler_job" "meghdoot_nowcast" {
  name             = "meghdoot-nowcast"
  description      = "Trigger Meghdoot nowcasting pipeline every 30 minutes"
  schedule         = "*/30 * * * *"  # Every 30 minutes
  time_zone        = "UTC"
  attempt_deadline = "600s"           # 10 minutes
  region           = "us-central1"

  http_target {
    uri        = "http://${google_compute_instance.my_instance.network_interface[0].network_ip}:8000/forecast/nowcast"
    http_method = "POST"

    oidc_token {
      service_account_email = google_service_account.meghdoot_scheduler.email
      audience              = "http://${google_compute_instance.my_instance.network_interface[0].network_ip}:8000"
    }
  }
}

# Service account for scheduler
resource "google_service_account" "meghdoot_scheduler" {
  account_id   = "meghdoot-scheduler"
  display_name = "Meghdoot Scheduler Service Account"
}

# Grant permissions to access GCS bucket
resource "google_storage_bucket_iam_member" "scheduler_bucket_access" {
  bucket = "megdhoot-satellite-data"
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.meghdoot_scheduler.email}"
}
```

Then apply:
```bash
terraform apply
```

#### Option C: Google Cloud Console

1. Go to [Cloud Scheduler](https://console.cloud.google.com/cloudscheduler)
2. Click **Create Job**
3. Fill in:
   - **Name**: `meghdoot-nowcast`
   - **Frequency**: `*/30 * * * *` (every 30 minutes)
   - **Timezone**: UTC
   - **Execution timeout**: 600 seconds (10 minutes)
4. Click **Continue**
5. Select **HTTP** as execution type
6. Fill in:
   - **URL**: `http://10.128.0.2:8000/forecast/nowcast` (replace IP)
   - **HTTP method**: POST
   - **Authentication**: Add OIDC token
     - Service account email: `meghdoot-scheduler@your-project.iam.gserviceaccount.com`
     - Audience: `http://10.128.0.2:8000`
7. Click **Create**

### Step 4: Set Up Authentication

The API server needs to trust Cloud Scheduler's service account. Update your FastAPI server to validate OIDC tokens:

```python
# In api.py, add token validation middleware

from google.auth.transport import requests
from google.oauth2 import id_token

SCHEDULER_SERVICE_ACCOUNT = "meghdoot-scheduler@YOUR_PROJECT.iam.gserviceaccount.com"

@app.middleware("http")
async def validate_scheduler_token(request: Request, call_next):
    """Validate Cloud Scheduler OIDC token."""
    if request.url.path == "/forecast/nowcast":
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse({"error": "Missing authorization token"}, status_code=401)
        
        token = auth_header[7:]  # Remove "Bearer "
        try:
            claims = id_token.verify_oauth2_token(
                token,
                requests.Request(),
                audience="http://YOUR_INSTANCE_IP:8000"
            )
            # Verify token is from Cloud Scheduler service account
            if claims.get("email") != SCHEDULER_SERVICE_ACCOUNT:
                return JSONResponse({"error": "Invalid service account"}, status_code=403)
        except Exception as e:
            log.warning(f"Token validation failed: {e}")
            return JSONResponse({"error": "Invalid token"}, status_code=401)
    
    return await call_next(request)
```

**For now**, you can skip token validation during development by keeping the endpoint open. In production, add proper authentication.

### Step 5: Test the Scheduler

#### Manual Trigger (Test)

```bash
gcloud scheduler jobs run meghdoot-nowcast --location=us-central1
```

Check logs:
```bash
gcloud scheduler jobs describe meghdoot-nowcast --location=us-central1
```

#### View Execution History

In Cloud Console → Cloud Scheduler → meghdoot-nowcast → **Execution history** tab

You'll see:
- Execution time
- Response status
- Response body (success/failure details)
- Latency

#### View Application Logs

On your Compute Engine instance:
```bash
# If running in foreground, check stdout
# If running as service, check logs:
tail -f /var/log/meghdoot_api.log
```

Or via Cloud Logging:
```bash
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_id=YOUR_INSTANCE_ID" \
  --limit 50 --format json
```

## Monitoring & Alerts

### Set Up Cloud Monitoring Alert

In [Cloud Monitoring](https://console.cloud.google.com/monitoring):

1. **Alerting** → **Create Policy**
2. **Select a metric**: `cloudscheduler.googleapis.com/job/execution_times`
3. **Add threshold**: Alert if execution fails
4. **Add notification channel**: Email, Slack, PagerDuty, etc.

### Example: Alert on Pipeline Failure

Create a notification policy:
```bash
gcloud alpha monitoring policies create \
  --notification-channels=YOUR_CHANNEL_ID \
  --display-name="Meghdoot Pipeline Failed" \
  --condition-display-name="Scheduler job failed" \
  --condition-threshold-value=1 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=60s
```

## Troubleshooting

### 1. "Connection refused" or "Network unreachable"

- **Cause**: Compute Engine firewall rule blocking traffic from Cloud Scheduler
- **Solution**: 
  ```bash
  # Ensure firewall allows internal traffic on port 8000
  gcloud compute firewall-rules create allow-scheduler \
    --allow=tcp:8000 \
    --source-ranges=0.0.0.0/0  # Or restrict to Cloud Scheduler IPs
  ```

### 2. "401 Unauthorized"

- **Cause**: OIDC token validation is strict and token is invalid
- **Solution**: Temporarily disable token validation in middleware (for testing)

### 3. "503 Model not loaded"

- **Cause**: FastAPI server didn't load the models in `startup()`
- **Solution**: Check server logs for model loading errors; ensure GPU is available

### 4. "GCS bucket not found"

- **Cause**: Service account doesn't have bucket access
- **Solution**: Grant permissions to the scheduler service account:
  ```bash
  gsutil iam ch serviceAccount:meghdoot-scheduler@YOUR_PROJECT.iam.gserviceaccount.com:roles/storage.objectAdmin gs://megdhoot-satellite-data
  ```

## Cost Optimization

**Why Cloud Scheduler + Compute Engine is cheaper than alternatives:**

| Option | Compute | Storage | Cost/Month |
|--------|---------|---------|-----------|
| **Cloud Scheduler + Compute Engine** (your setup) | L4 GPU (30min/hr) | GCS bucket | ~$100–200 |
| **Cloud Run** | No GPU support (yet) | ❌ | N/A |
| **Cloud Functions** | No GPU, cold start lag | ❌ | N/A |
| **App Engine** | Standard/Flexible, no GPU | ❌ | N/A |
| **Kubernetes (GKE)** | Overkill, expensive | High overhead | $300+ |

Your setup is optimal: run GPU only during forecast cycles (~1 min every 30 min = 2% utilization), not 24/7.

## Next Steps

1. **Verify pipeline runs successfully** via manual scheduler trigger
2. **Add monitoring** (Cloud Logging + Monitoring alerts)
3. **Implement token validation** in production (use `google-auth` library)
4. **Frontend deployment** (reads forecast from GCS bucket; see FRONTEND_SETUP.md)
5. **Cost tracking** (enable GCP budget alerts)

## Useful Commands

```bash
# List all scheduler jobs
gcloud scheduler jobs list --location=us-central1

# Describe a job
gcloud scheduler jobs describe meghdoot-nowcast --location=us-central1

# Manually run (for testing)
gcloud scheduler jobs run meghdoot-nowcast --location=us-central1

# Update schedule to different interval
gcloud scheduler jobs update meghdoot-nowcast \
  --location=us-central1 \
  --schedule="0,30 * * * *"

# Delete job
gcloud scheduler jobs delete meghdoot-nowcast --location=us-central1

# View recent executions with gcloud
gcloud logging read \
  'resource.type="cloud_scheduler_job" AND resource.labels.job_id="meghdoot-nowcast"' \
  --limit=10 --format=json --project=YOUR_PROJECT_ID
```

---

**Questions?** Check Cloud Scheduler logs in the [Google Cloud Console](https://console.cloud.google.com/cloudscheduler) or run `gcloud scheduler jobs describe meghdoot-nowcast --location=us-central1 --format=json`.
