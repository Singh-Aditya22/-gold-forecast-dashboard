#Requires -Version 5.1
# refresh.ps1 — Windows equivalent of refresh.sh
#
# Run manually : .\refresh.ps1
# Force re-run : .\refresh.ps1 -Force
# Automated via Windows Task Scheduler (registered by setup-task.ps1)
#
# Same multi-machine safety as refresh.sh:
#   git pull first → cheap collect/bronze/silver → check if latest trading day already
#   has predictions logged → skip expensive model steps if so (skips in <1 min).
#   Pass -Force to bypass the skip and retrain anyway.

param([switch]$Force)

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $SCRIPT_DIR

$LOG_DIR  = Join-Path $SCRIPT_DIR "logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$LOG_FILE = Join-Path $LOG_DIR "refresh_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}

Log "=== Gold Forecast Refresh Started ==="

# Activate venv
$activateScript = Join-Path $SCRIPT_DIR "venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Log "[ERROR] venv not found at $activateScript"
    Log "        Run: python -m venv venv; venv\Scripts\pip install -r requirements.txt"
    Log "        Then: venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu"
    exit 1
}
. $activateScript

Log "[1/10] Pulling latest data..."
$pullOut = git pull --quiet 2>&1
Log "[git]  $pullOut"

Log "[2/10] Collecting raw data..."
python pipeline\collect.py

Log "[3/10] Loading bronze layer..."
python pipeline\bronze.py

Log "[4/10] Building silver layer..."
python pipeline\silver.py

Log "[5/10] Checking whether the latest trading day already has predictions logged..."
$ALREADY_DONE = python -c "
import duckdb
try:
    con = duckdb.connect('gold_forecast.duckdb', read_only=True)
    latest = con.execute('SELECT MAX(date) FROM silver.prices').fetchone()[0]
    n = con.execute('SELECT COUNT(*) FROM gold.live_predictions WHERE predicted_on = ?', [latest]).fetchone()[0]
    print(n)
except Exception:
    print(0)
"

if (([int]($ALREADY_DONE.Trim()) -gt 0) -and (-not $Force)) {
    Log "[skip] Latest trading day already has predictions logged (another machine ran first)."
    Log "       Run '.\refresh.ps1 -Force' to refresh anyway."
} else {
    Log "[6/10] Building gold layer..."
    python pipeline\gold.py

    Log "[7/10] Generating forecasts..."
    python models\predict.py

    Log "[8/10] Reconciling live predictions + logging tomorrow's..."
    python models\track_predictions.py

    Log "[9/10] Backfilling any missed days (laptop off/asleep)..."
    python models\backfill_live_predictions.py
}

Log "[10/10] Publishing updated data to GitHub..."
$gitStatus = git status --porcelain -- gold_forecast.duckdb
if ($gitStatus) {
    git add gold_forecast.duckdb
    git commit -m "Automated daily data refresh: $(Get-Date -Format 'yyyy-MM-dd')"
    git push
    Log "[publish] Pushed to GitHub -- Streamlit Cloud will auto-redeploy."
} else {
    Log "[publish] No changes to gold_forecast.duckdb -- nothing to push."
}

Log "=== Refresh Complete ==="
