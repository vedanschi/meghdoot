#!/bin/bash
# run_forecast_cycle.sh – Cron wrapper for nowcasting pipeline
# 
# Run via cron every 30 minutes:
#   0,30 * * * * /path/to/run_forecast_cycle.sh
#
# Or via systemd timer:
#   [Unit]
#   Description=Meghdoot Nowcasting Pipeline
#   
#   [Timer]
#   OnBootSec=1min
#   OnUnitActiveSec=30min
#   Persistent=true
#   
#   [Install]
#   WantedBy=timers.target

set -e

REPO_ROOT="/home/vedanschi/meghdoot"
VENV="${REPO_ROOT}/.venv"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate virtual environment and run pipeline
{
    echo "Starting pipeline at $(date)"
    source "${VENV}/bin/activate"
    cd "$REPO_ROOT"
    python -m meghdoot.deploy.pipeline --config configs/default.yaml
    exit_code=$?
    echo "Pipeline exited with code: $exit_code"
    exit $exit_code
} >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "Pipeline run succeeded; see $LOG_FILE"
else
    echo "Pipeline run failed; see $LOG_FILE"
    # Optionally send alert here (email, Slack, etc.)
fi
