#!/usr/bin/env bash
# Daily pipeline refresh script.
# Add to crontab: 0 8 * * 1-5 /path/to/gold_forecast/refresh.sh
# This runs Monday-Friday at 8am to pick up the previous day's market close.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/logs/refresh_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$SCRIPT_DIR/logs"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Gold Forecast Refresh Started: $(date) ==="

source "$SCRIPT_DIR/venv/bin/activate"

echo "[1/5] Collecting raw data..."
python pipeline/collect.py

echo "[2/5] Loading bronze layer..."
python pipeline/bronze.py

echo "[3/5] Building silver layer..."
python pipeline/silver.py

echo "[4/5] Building gold layer..."
python pipeline/gold.py

echo "[5/6] Generating forecasts..."
python models/predict.py

echo "[6/6] Reconciling live predictions + logging tomorrow's..."
python models/track_predictions.py

echo "=== Refresh Complete: $(date) ==="
