"""
Generate 90-day forward forecasts for EVERY candidate model (naive, prophet, arima,
xgboost, lightgbm, lstm, ensemble) per instrument, so the dashboard's Forecast page can
let the user pick any of them to view — not just the one evaluate.py picked as the
statistically best choice (gold.model_scores.selected = true).

The holdout-trained models in models/saved/ are for scoring only (evaluate.py) — they're
frozen as of 2024-12-31 and can't produce a genuine "next 90 days from today" forecast.
So here we retrain every architecture per instrument on the FULL dataset (through the
most recent actual date) and use those fresh models purely for the live forecast.
XGBoost/LightGBM reuse their cached tuned hyperparameters (from train.py's
RandomizedSearchCV) rather than re-searching. "ensemble" is a simple average of the
other 6 already-computed forecasts (not retrained a second time).

Future values for the macro/geopolitical-risk regressors (VIX, oil, USD index, 10Y
yield) are carried forward flat from their last known value over the 90-day horizon —
a standard simplification, since those series aren't themselves being forecast here.

Writes in-sample fitted values + future predictions to gold.forecasts.
Run after evaluate.py.
"""

import os
import sys
import warnings
import duckdb
import pandas as pd
import numpy as np
from prophet import Prophet

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import (
    DB_PATH, MACRO_COLS, FEATURE_COLS, load_instrument_data, business_days_forward,
    load_best_params, DEFAULT_XGB_PARAMS, DEFAULT_LGBM_PARAMS, log_return_clip_bounds,
    DAMPING_PHI, historical_daily_vol, random_walk_confidence_band,
)
from train import train_prophet, train_arima, train_xgboost, train_lightgbm
import lstm_model

warnings.filterwarnings("ignore")

FORECAST_DAYS = 90


def train_production_tree_model(model_name: str, df: pd.DataFrame, instrument: str):
    if model_name == "xgboost":
        params = load_best_params(instrument, "xgboost", DEFAULT_XGB_PARAMS)
        return train_xgboost(df, params=params)
    params = load_best_params(instrument, "lightgbm", DEFAULT_LGBM_PARAMS)
    return train_lightgbm(df, params=params)


def forecast_prophet(model: Prophet, df: pd.DataFrame, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=forecast_days)

    hist_macro = df[["date"] + MACRO_COLS].rename(columns={"date": "ds"})
    hist_macro["ds"] = pd.to_datetime(hist_macro["ds"]).dt.tz_localize(None)
    future = future.merge(hist_macro, on="ds", how="left")

    last_macro = df[MACRO_COLS].iloc[-1]
    for col in MACRO_COLS:
        future[col] = future[col].fillna(last_macro[col])

    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result.columns = ["date", "yhat", "yhat_lower", "yhat_upper"]
    result["date"] = pd.to_datetime(result["date"])
    historical_dates = set(df["date"].dt.date)
    result["is_future"] = result["date"].dt.date.apply(lambda d: d not in historical_dates)
    return result


def forecast_arima(model, df: pd.DataFrame, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    exog_hist = df[MACRO_COLS].values
    fitted = pd.DataFrame({
        "date": df["date"],
        "yhat": list(model.predict_in_sample(X=exog_hist)),
        "yhat_lower": None,
        "yhat_upper": None,
        "is_future": False,
    })

    last_macro = df[MACRO_COLS].iloc[-1].values
    future_exog = np.tile(last_macro, (forecast_days, 1))
    future_preds, conf_int = model.predict(n_periods=forecast_days, X=future_exog, return_conf_int=True)
    future_dates = business_days_forward(df["date"].max().date(), forecast_days)
    future_df = pd.DataFrame({
        "date": pd.to_datetime(future_dates),
        "yhat": future_preds,
        "yhat_lower": conf_int[:, 0],
        "yhat_upper": conf_int[:, 1],
        "is_future": True,
    })
    return pd.concat([fitted, future_df], ignore_index=True)


def forecast_tree_model(model, df: pd.DataFrame, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """
    Model predicts log_return; reconstruct price by compounding onto a running estimate.
    Fitted (in-sample) values reconstruct from the true previous actual close (walk-forward,
    no compounding error). ma_50/ma_200 are recomputed each future step from the rolling
    price window; rsi_14/bollinger bands/rolling_vol_30d/drawdown_pct are held flat at their
    last known actual value over the horizon — recomputing those from a window of
    increasingly synthetic future prices would compound approximation error for uncertain
    benefit, so this follows the same carry-forward simplification already used for macro.
    """
    df_lags = common.add_lag_features(df)

    X_hist = df_lags[FEATURE_COLS].values
    pred_log_returns_hist = model.predict(X_hist)
    prev_actual_close_hist = df_lags["lag_1"].values
    fitted = pd.DataFrame({
        "date": df_lags["date"],
        "yhat": prev_actual_close_hist * np.exp(pred_log_returns_hist),
        "yhat_lower": None,
        "yhat_upper": None,
        "is_future": False,
    })

    last_row = df_lags.iloc[-1].copy()

    def _flat(col, default):
        return float(last_row[col]) if col in last_row and pd.notna(last_row[col]) else default

    flat_rsi = _flat("rsi_14", 50.0)
    flat_bb_upper = _flat("bb_upper", float(last_row["close_inr"]) if "close_inr" in last_row else 0.0)
    flat_bb_lower = _flat("bb_lower", float(last_row["close_inr"]) if "close_inr" in last_row else 0.0)
    flat_vol = _flat("rolling_vol_30d", 0.0)
    flat_drawdown = _flat("drawdown_pct", 0.0)
    last_macro = [float(last_row[c]) for c in MACRO_COLS]
    clip_lo, clip_hi = log_return_clip_bounds(df)
    sigma = historical_daily_vol(df)

    future_rows = []
    price_window = list(df["close_inr"].tail(30))
    running_price = float(df["close_inr"].iloc[-1])

    for step, day in enumerate(business_days_forward(df["date"].max().date(), forecast_days), start=1):
        lag_1 = price_window[-1]
        lag_7 = price_window[-7] if len(price_window) >= 7 else price_window[0]
        lag_30 = price_window[-30] if len(price_window) >= 30 else price_window[0]
        ma_50 = float(np.mean(price_window[-50:])) if len(price_window) >= 50 else float(np.mean(price_window))
        ma_200 = float(np.mean(price_window[-200:])) if len(price_window) >= 200 else float(np.mean(price_window))

        X = np.array([[lag_1, lag_7, lag_30, ma_50, ma_200, flat_rsi,
                        flat_bb_upper, flat_bb_lower, flat_vol, flat_drawdown] + last_macro])
        pred_log_return = float(np.clip(model.predict(X)[0], clip_lo, clip_hi))
        pred_log_return *= DAMPING_PHI ** step
        running_price = running_price * np.exp(pred_log_return)
        price_window.append(running_price)
        lo, hi = random_walk_confidence_band(running_price, step, sigma)

        future_rows.append({
            "date": pd.Timestamp(day),
            "yhat": running_price,
            "yhat_lower": lo,
            "yhat_upper": hi,
            "is_future": True,
        })

    future_df = pd.DataFrame(future_rows)
    return pd.concat([fitted, future_df], ignore_index=True)


def forecast_naive(df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    """
    Random-walk baseline: fitted = previous day's actual, future = flat at last close.
    Confidence band widens as sqrt(step) (the closed-form random-walk interval), scaled by
    the instrument's own historical daily volatility — naive has no built-in uncertainty
    estimate otherwise, but "flat line, growing uncertainty" is exactly what a random walk
    forecast's interval should look like.
    """
    fitted = pd.DataFrame({
        "date": df["date"],
        "yhat": df["close_inr"].shift(1),
        "yhat_lower": None,
        "yhat_upper": None,
        "is_future": False,
    }).dropna(subset=["yhat"])

    last_close = float(df["close_inr"].iloc[-1])
    sigma = historical_daily_vol(df)
    future_dates = business_days_forward(df["date"].max().date(), forecast_days)
    lowers, uppers = [], []
    for step in range(1, len(future_dates) + 1):
        lo, hi = random_walk_confidence_band(last_close, step, sigma)
        lowers.append(lo)
        uppers.append(hi)
    future_df = pd.DataFrame({
        "date": pd.to_datetime(future_dates),
        "yhat": last_close,
        "yhat_lower": lowers,
        "yhat_upper": uppers,
        "is_future": True,
    })
    return pd.concat([fitted, future_df], ignore_index=True)


def average_forecasts(members: dict) -> pd.DataFrame:
    """Simple average of already-computed member forecasts (fitted + future), by date —
    used for "ensemble" so it doesn't retrain every member a second time."""
    combined = pd.concat(
        [mf[["date", "yhat", "is_future"]].assign(member=name) for name, mf in members.items()],
        ignore_index=True,
    )
    avg = combined.groupby(["date", "is_future"], as_index=False)["yhat"].mean()
    avg["yhat_lower"] = None
    avg["yhat_upper"] = None
    return avg[["date", "yhat", "yhat_lower", "yhat_upper", "is_future"]]


def run_predictions() -> None:
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    con.execute("DROP TABLE IF EXISTS gold.forecasts")

    all_forecasts = []

    for instrument in common.INSTRUMENTS:
        print(f"\n[predict] === {instrument} ===")
        df = load_instrument_data(con, instrument)

        members = {}
        print(f"[predict] {instrument} → naive")
        members["naive"] = forecast_naive(df, FORECAST_DAYS)

        print(f"[predict] {instrument} → prophet")
        members["prophet"] = forecast_prophet(train_prophet(df), df)

        print(f"[predict] {instrument} → arima")
        members["arima"] = forecast_arima(train_arima(df), df)

        print(f"[predict] {instrument} → xgboost")
        members["xgboost"] = forecast_tree_model(train_production_tree_model("xgboost", df, instrument), df)

        print(f"[predict] {instrument} → lightgbm")
        members["lightgbm"] = forecast_tree_model(train_production_tree_model("lightgbm", df, instrument), df)

        print(f"[predict] {instrument} → lstm")
        lstm_artifact = lstm_model.train_lstm(df)
        members["lstm"] = lstm_model.forecast_lstm(lstm_artifact, df, FORECAST_DAYS)

        print(f"[predict] {instrument} → ensemble")
        members["ensemble"] = average_forecasts(members)

        for model_name, forecast_df in members.items():
            forecast_df = forecast_df.copy()
            forecast_df["instrument"] = instrument
            forecast_df["model_name"] = model_name
            all_forecasts.append(forecast_df)

    if not all_forecasts:
        print("[predict] No forecasts generated.")
        con.close()
        return

    final_df = pd.concat(all_forecasts, ignore_index=True)
    final_df = final_df[["date", "instrument", "model_name", "yhat", "yhat_lower", "yhat_upper", "is_future"]]
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.date
    # pd.to_numeric before rounding: concat of per-model frames (some with None bands,
    # some with floats) yields an object-dtype column holding literal None -- pandas 3.0's
    # Series.round raises on None instead of coercing it to NaN like float64 dtype did.
    final_df["yhat"] = pd.to_numeric(final_df["yhat"], errors="coerce").round(4)
    final_df["yhat_lower"] = pd.to_numeric(final_df["yhat_lower"], errors="coerce").round(4)
    final_df["yhat_upper"] = pd.to_numeric(final_df["yhat_upper"], errors="coerce").round(4)

    con.execute("CREATE TABLE gold.forecasts AS SELECT * FROM final_df")
    print(f"[predict] gold.forecasts written — {len(final_df)} rows")

    con.close()


if __name__ == "__main__":
    run_predictions()
    print("\n[predict] Done.")
