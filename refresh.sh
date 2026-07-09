#!/usr/bin/env bash
# Daily pipeline refresh script. Runs the full pipeline, then publishes the refreshed
# database to GitHub so the Streamlit Cloud deploy picks it up automatically.
#
# Triggered by a systemd --user timer (gold-forecast-refresh.timer), not plain cron --
# Persistent=true means it catches up on next login/wake if the laptop was asleep or off
# at the scheduled time, instead of silently skipping the day.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/logs/refresh_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$SCRIPT_DIR/logs"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Gold Forecast Refresh Started: $(date) ==="

source "$SCRIPT_DIR/venv/bin/activate"

echo "[1/7] Collecting raw data..."
python pipeline/collect.py

echo "[2/7] Loading bronze layer..."
python pipeline/bronze.py

echo "[3/7] Building silver layer..."
python pipeline/silver.py

echo "[4/7] Building gold layer..."
python pipeline/gold.py

echo "[5/7] Generating forecasts..."
python models/predict.py

echo "[6/7] Reconciling live predictions + logging tomorrow's..."
python models/track_predictions.py

echo "[7/7] Publishing updated data to GitHub..."
if [ -n "$(git status --porcelain -- gold_forecast.duckdb)" ]; then
    git add gold_forecast.duckdb
    git commit -m "Automated daily data refresh: $(date +%Y-%m-%d)"
    git push
    echo "[publish] Pushed updated data to GitHub -- Streamlit Cloud will auto-redeploy."
else
    echo "[publish] No changes to gold_forecast.duckdb, nothing to push."
fi

echo "=== Refresh Complete: $(date) ==="
