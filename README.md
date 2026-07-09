# Gold Forecast Dashboard

An end-to-end data engineering + ML project that tracks non-physical gold investment
options in India (Gold Futures, Gold ETFs, SBI Gold Fund), forecasts prices with multiple
models, and helps identify historical and predicted buying dips.

## Stack

- **Data:** yfinance (gold futures, USD/INR, Gold ETFs, VIX, crude oil, USD Index, 10Y
  Treasury yield) + mftool (SBI Gold Fund NAV)
- **Storage:** DuckDB, medallion architecture (bronze → silver → gold schemas)
- **Models:** Naive (random-walk baseline), Prophet, ARIMA (with macro exogenous
  regressors), XGBoost, LightGBM (both hyperparameter-tuned), LSTM, and a simple ensemble
- **Dashboard:** Streamlit + Plotly

## Why a naive baseline is one of the 7 models

Daily asset prices behave close to a random walk — "tomorrow's price = today's price" is
a genuinely hard benchmark to beat, and treating it as the accuracy bar (rather than an
arbitrary MAPE%) is the standard practice in financial forecasting. The dashboard reports
every model's *skill score vs. this baseline*, not just its raw error, so a model is only
trusted if it actually adds value over doing nothing.

## Notable design decisions

- **Return-space modeling, not price-level:** XGBoost/LightGBM/LSTM predict the next-day
  *return*, then reconstruct price by compounding — training directly on price level
  means a tree-based model can't extrapolate past the highest price it saw in training,
  which silently produces terrible forecasts once the market moves into new territory.
- **Damped-trend compounding:** a recursive multi-step forecast can turn a small,
  unremarkable daily bias into a wildly unrealistic long-horizon move (a steady +0.4%/day
  compounds to +45% over 90 days). Predicted returns are damped toward zero as the horizon
  lengthens, converging toward "no further change" for distant days instead of assuming a
  constant rate holds indefinitely.
- **Live forward-testing:** separate from the one-time historical backtest
  (`evaluate.py`), `track_predictions.py` logs each model's real next-day prediction
  before the outcome is known, then reconciles it once the actual price arrives — a
  genuine day-by-day track record, not a replay of already-known history.
- **Outlier-guarded ingestion:** a rolling-median check in the silver layer catches vendor
  data glitches (a real one: GOLDBEES.NS briefly printed ~1% of its actual price for two
  days in Dec 2019) before they corrupt technical indicators or training data.

## Project layout

```
pipeline/     collect.py → bronze.py → silver.py → gold.py   (data pipeline)
models/       train.py → evaluate.py → predict.py → track_predictions.py
dashboard/    app.py (Streamlit UI), charts.py, queries.py, insights.py
refresh.sh    runs the full daily pipeline end to end
```

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# CPU-only PyTorch (smaller download, no GPU needed):
pip install torch --index-url https://download.pytorch.org/whl/cpu

python pipeline/collect.py
python pipeline/bronze.py
python pipeline/silver.py
python pipeline/gold.py
python models/train.py       # ~15-20 min: hyperparameter search + LSTM training
python models/evaluate.py
python models/predict.py
python models/track_predictions.py

streamlit run dashboard/app.py
```

A pre-built `gold_forecast.duckdb` (with all layers and trained-model forecasts already
computed) is included, so you can run just `streamlit run dashboard/app.py` and explore
the dashboard immediately without waiting on the full pipeline.

## Deploying the dashboard (Streamlit Community Cloud)

`dashboard/app.py` and everything it imports (`charts.py`, `queries.py`, `insights.py`)
only need `streamlit`, `duckdb`, `pandas`, `numpy`, and `plotly` — the heavier libraries
in `requirements.txt` (prophet, xgboost, lightgbm, pmdarima, torch, yfinance, mftool) are
only used by the training pipeline scripts, not by the dashboard itself, since it just
reads the pre-built `gold_forecast.duckdb`. Use **`requirements-dashboard.txt`** for a
much faster, lighter cloud deploy:

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
2. "New app" → this repo → branch `main` → main file path `dashboard/app.py`
3. Under "Advanced settings", set the requirements file to `requirements-dashboard.txt`
4. Deploy — auto-redeploys on every push to `main`. Delete the app from the same dashboard
   to take it down (or just leave it — Streamlit's free tier auto-sleeps idle apps).

## Dashboard pages

- **Overview** — normalized return comparison across instruments + correlation heatmap
- **Individual Instrument** — candlestick/line chart with technical indicators, forecast
  overlay, and a trend line for the selected window
- **Dip Tracker** — historical and forecast-projected buying dips
- **Forecast** — pick any of the 7 models, compare side by side, with confidence bands
- **Model Comparison** — full accuracy/skill-score table
- **SGB Calculator** — Sovereign Gold Bond return calculator

## Disclaimer

This is a personal/educational project, not financial advice. Forecasts are model
outputs with documented limitations (see design decisions above), not investment
recommendations.
