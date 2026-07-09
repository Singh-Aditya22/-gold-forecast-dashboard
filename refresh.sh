#!/usr/bin/env bash
# Daily pipeline refresh script. Runs the full pipeline, then publishes the refreshed
# database to GitHub so the Streamlit Cloud deploy picks it up automatically.
#
# Triggered by a systemd --user timer (gold-forecast-refresh.timer), not plain cron --
# Persistent=true means it catches up on next login/wake if the laptop was asleep or off
# at the scheduled time, instead of silently skipping the day.
#
# Multi-machine safe: this script also runs manually/via its own timer on the office
# laptop. It always pulls the latest data and re-fetches prices (cheap), then checks
# whether the LATEST TRADING DAY now available already has a logged prediction
# (predicted_on in gold.live_predictions tracks MAX(silver.prices.date), not the
# wall-clock date -- they differ whenever today's close isn't in yet, e.g. mornings or
# weekends). If another machine already processed that trading day today, this run
# skips the expensive model-retraining steps and just publishes/no-ops, instead of
# redundantly retraining every model AND racing to git-push a conflicting binary
# database file. Pass --force to bypass the skip and refresh anyway.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/logs/refresh_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$SCRIPT_DIR/logs"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Gold Forecast Refresh Started: $(date) ==="

source "$SCRIPT_DIR/venv/bin/activate"

echo "[1/10] Pulling latest data (in case another machine already refreshed today)..."
git pull --quiet || echo "[git] pull failed (offline or conflict?) -- continuing with local database state."

echo "[2/10] Collecting raw data..."
python pipeline/collect.py

echo "[3/10] Loading bronze layer..."
python pipeline/bronze.py

echo "[4/10] Building silver layer..."
python pipeline/silver.py

echo "[5/10] Checking whether the latest trading day already has predictions logged..."
ALREADY_DONE=$(python -c "
import duckdb
try:
    con = duckdb.connect('gold_forecast.duckdb', read_only=True)
    latest_trading_day = con.execute('SELECT MAX(date) FROM silver.prices').fetchone()[0]
    n = con.execute('SELECT COUNT(*) FROM gold.live_predictions WHERE predicted_on = ?', [latest_trading_day]).fetchone()[0]
    print(n)
except Exception:
    print(0)
")

if [ "${ALREADY_DONE:-0}" -gt 0 ] && [ "${1:-}" != "--force" ]; then
    echo "[skip] The latest trading day already has predictions logged (most likely from"
    echo "       another machine earlier today) -- skipping model retraining."
    echo "       Run './refresh.sh --force' to refresh again anyway."
else
    echo "[6/10] Building gold layer..."
    python pipeline/gold.py

    echo "[7/10] Generating forecasts..."
    python models/predict.py

    echo "[8/10] Reconciling live predictions + logging tomorrow's..."
    python models/track_predictions.py

    echo "[9/10] Backfilling any missed days (laptop asleep/off)..."
    python models/backfill_live_predictions.py
fi

echo "[10/10] Publishing updated data to GitHub..."
if [ -n "$(git status --porcelain -- gold_forecast.duckdb)" ]; then
    git add gold_forecast.duckdb
    git commit -m "Automated daily data refresh: $(date +%Y-%m-%d)"
    git push
    echo "[publish] Pushed updated data to GitHub -- Streamlit Cloud will auto-redeploy."
else
    echo "[publish] No changes to gold_forecast.duckdb, nothing to push."
fi

echo "=== Refresh Complete: $(date) ==="
