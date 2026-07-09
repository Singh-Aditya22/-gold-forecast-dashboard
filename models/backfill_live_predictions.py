"""
Backfill Live Track Record entries for any trading day that has real data but no logged
prediction -- happens when the laptop was asleep/off and the daily refresh never ran on
those specific days. For each gap day, retrains every model using ONLY data strictly
before that day (a genuine walk-forward 1-step-ahead forecast, not hindsight), then logs
+ immediately reconciles it since the actual outcome is already known (the day already
happened).

"Gap days" are scoped to the window SINCE live tracking actually started (the earliest
predicted_for date already in gold.live_predictions for that instrument) -- not the
entire multi-decade historical dataset, which predates this feature and isn't a "missed
day" in any meaningful sense.

No artificial limit on how far back a gap can be backfilled: each gap day is a genuine
retrain per model (naive is instant; tree models reuse cached tuned hyperparameters;
Prophet/ARIMA/LSTM are the slow ones, LSTM particularly at ~1 min each) -- a long gap
means real compute time, by design, since there's no shortcut to an honest walk-forward
prediction.

Run standalone (python models/backfill_live_predictions.py) or as part of refresh.sh,
after track_predictions.py.
"""

import os
import sys
import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DB_PATH, INSTRUMENTS, load_instrument_data, load_best_params, DEFAULT_XGB_PARAMS, DEFAULT_LGBM_PARAMS
from train import train_prophet, train_arima, train_xgboost, train_lightgbm
import lstm_model
import predict as predict_mod

MEMBER_MODELS = ["naive", "prophet", "arima", "xgboost", "lightgbm", "lstm"]
MIN_TRAIN_ROWS = 90  # LSTM needs at least seq_len(30) + val_days(60) to fit meaningfully


def _one_day_ahead(model_name: str, train_df: pd.DataFrame, instrument: str) -> float:
    """Genuine walk-forward 1-step-ahead point forecast, using ONLY train_df (strictly
    before the day being predicted -- no lookahead)."""
    if model_name == "naive":
        return float(train_df["close_inr"].iloc[-1])

    if model_name == "prophet":
        model = train_prophet(train_df)
        fc = predict_mod.forecast_prophet(model, train_df, forecast_days=1)
    elif model_name == "arima":
        model = train_arima(train_df)
        fc = predict_mod.forecast_arima(model, train_df, forecast_days=1)
    elif model_name == "xgboost":
        params = load_best_params(instrument, "xgboost", DEFAULT_XGB_PARAMS)
        model = train_xgboost(train_df, params=params)
        fc = predict_mod.forecast_tree_model(model, train_df, forecast_days=1)
    elif model_name == "lightgbm":
        params = load_best_params(instrument, "lightgbm", DEFAULT_LGBM_PARAMS)
        model = train_lightgbm(train_df, params=params)
        fc = predict_mod.forecast_tree_model(model, train_df, forecast_days=1)
    elif model_name == "lstm":
        artifact = lstm_model.train_lstm(train_df)
        fc = lstm_model.forecast_lstm(artifact, train_df, 1)
    else:
        raise ValueError(f"unknown model_name: {model_name}")

    future_row = fc[fc["is_future"] == True].sort_values("date").iloc[0]
    return float(future_row["yhat"])


def backfill_predictions() -> None:
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    con.execute("""
        CREATE TABLE IF NOT EXISTS gold.live_predictions (
            instrument VARCHAR, model_name VARCHAR, predicted_on DATE, predicted_for DATE,
            predicted_price DOUBLE, actual_price DOUBLE, abs_error DOUBLE, pct_error DOUBLE
        )
    """)

    total_backfilled = 0
    for instrument in INSTRUMENTS:
        tracking_start_row = con.execute(f"""
            SELECT MIN(predicted_for) FROM gold.live_predictions WHERE instrument = '{instrument}'
        """).fetchone()
        if tracking_start_row[0] is None:
            print(f"[backfill] {instrument}: live tracking hasn't logged anything yet, nothing to backfill.")
            continue
        tracking_start = pd.Timestamp(tracking_start_row[0])

        full_df = load_instrument_data(con, instrument)
        all_dates = full_df["date"].tolist()

        logged = con.execute(f"""
            SELECT DISTINCT predicted_for FROM gold.live_predictions WHERE instrument = '{instrument}'
        """).fetchdf()["predicted_for"]
        logged_set = set(pd.to_datetime(logged))

        gap_dates = [d for d in all_dates if d >= tracking_start and d not in logged_set]
        if not gap_dates:
            print(f"[backfill] {instrument}: no gaps since tracking started ({tracking_start.date()}).")
            continue

        print(f"[backfill] {instrument}: {len(gap_dates)} gap day(s) since {tracking_start.date()}: "
              f"{gap_dates[0].date()} .. {gap_dates[-1].date()}")

        for target_date in gap_dates:
            train_df = full_df[full_df["date"] < target_date].reset_index(drop=True)
            if len(train_df) < MIN_TRAIN_ROWS:
                print(f"[backfill]   {target_date.date()}: skipping, not enough history yet "
                      f"({len(train_df)} rows).")
                continue
            actual_price = float(full_df.loc[full_df["date"] == target_date, "close_inr"].iloc[0])

            member_preds = {}
            for m in MEMBER_MODELS:
                try:
                    member_preds[m] = _one_day_ahead(m, train_df, instrument)
                except Exception as e:
                    print(f"[backfill]   {target_date.date()} {m}: FAILED ({e}), skipping this model.")
            if member_preds:
                member_preds["ensemble"] = float(np.mean(list(member_preds.values())))

            for model_name, pred_price in member_preds.items():
                abs_err = abs(actual_price - pred_price)
                pct_err = (abs_err / actual_price * 100) if actual_price else None
                con.execute("""
                    INSERT INTO gold.live_predictions
                        (instrument, model_name, predicted_on, predicted_for, predicted_price,
                         actual_price, abs_error, pct_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [instrument, model_name, train_df["date"].iloc[-1].date(), target_date.date(),
                      pred_price, actual_price, abs_err, pct_err])
                total_backfilled += 1
            print(f"[backfill]   {target_date.date()}: logged {len(member_preds)} model(s).")

    print(f"[backfill] Done — {total_backfilled} prediction(s) backfilled.")
    con.close()


if __name__ == "__main__":
    backfill_predictions()
