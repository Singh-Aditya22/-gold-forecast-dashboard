"""
Evaluate all trained models on the 2025 holdout set.
Computes RMSE, MAE, MAPE, skill-score-vs-naive, and directional accuracy per model
per instrument. Writes results to gold.model_scores.

Domain-standard accuracy bar: a model only "passes" if it beats a naive random-walk
baseline (yhat = previous day's actual close) — the standard benchmark in financial
forecasting, since an arbitrary MAPE% threshold isn't meaningful for a volatile asset
like gold. If nothing beats naive for an instrument, naive itself is selected — that's
the honest, correct outcome, not a bug.

Run after train.py.
"""

import os
import sys
import warnings
import duckdb
import pandas as pd
import numpy as np
from prophet import Prophet

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DB_PATH, SAVED_DIR, INSTRUMENTS, TEST_START, TEST_END, MACRO_COLS, FEATURE_COLS, TARGET_COL,
    load_instrument_data, add_lag_features, load_model,
)
import lstm_model

warnings.filterwarnings("ignore")

CANDIDATE_MODELS = ["prophet", "arima", "xgboost", "lightgbm", "lstm"]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred, dtype=float)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred, dtype=float))))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true, y_pred, y_prev) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred, dtype=float)
    y_prev = np.array(y_prev, dtype=float)
    actual_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    mask = actual_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(pred_dir[mask] == actual_dir[mask]) * 100)


def predict_prophet(model: Prophet, test_df: pd.DataFrame) -> list:
    future = test_df[["date"] + MACRO_COLS].rename(columns={"date": "ds"})
    future["ds"] = future["ds"].dt.tz_localize(None)
    forecast = model.predict(future)
    return forecast["yhat"].tolist()


def predict_arima(model, test_df: pd.DataFrame) -> list:
    exog = test_df[MACRO_COLS].values
    return list(model.predict(n_periods=len(test_df), X=exog))


def predict_tree_model(model, full_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """
    Model predicts log_return; reconstruct price as prev_actual_close * exp(pred_log_return)
    (walk-forward against known history, matching naive/arima/prophet's one-step scoring —
    avoids compounding error into the backtest). lag_1 is exactly the previous actual close.
    """
    df_with_lags = add_lag_features(full_df)
    test_dates = set(test_df["date"].dt.date)
    test_rows = df_with_lags[df_with_lags["date"].dt.date.isin(test_dates)]
    if test_rows.empty:
        return []
    X = test_rows[FEATURE_COLS].values
    pred_log_returns = model.predict(X)
    prev_actual_close = test_rows["lag_1"].values
    return list(prev_actual_close * np.exp(pred_log_returns))


def evaluate_all() -> None:
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    con.execute("DROP TABLE IF EXISTS gold.model_scores")

    records = []

    for instrument in INSTRUMENTS:
        print(f"\n[evaluate] === {instrument} ===")
        full_df = load_instrument_data(con, instrument)
        test_df = full_df[(full_df["date"] >= TEST_START) & (full_df["date"] <= TEST_END)].copy()

        if test_df.empty:
            print(f"[evaluate] No test data for {instrument}, skipping.")
            continue

        y_true = test_df["close_inr"].values
        prev_actual = full_df["close_inr"].shift(1)
        naive_pred = prev_actual.loc[test_df.index].values

        preds = {"naive": naive_pred}

        for model_name in CANDIDATE_MODELS:
            ext = "pt" if model_name == "lstm" else "pkl"
            model_path = os.path.join(SAVED_DIR, f"{instrument}__{model_name}.{ext}")
            if not os.path.exists(model_path):
                print(f"[evaluate] SKIP: {instrument}__{model_name}.{ext} not found")
                continue

            if model_name == "prophet":
                y_pred = predict_prophet(load_model(instrument, "prophet"), test_df)
            elif model_name == "arima":
                y_pred = predict_arima(load_model(instrument, "arima"), test_df)
            elif model_name == "lstm":
                artifact = lstm_model.load_lstm(instrument)
                y_pred = lstm_model.predict_at_dates(artifact, full_df, test_df["date"])
            else:
                y_pred = predict_tree_model(load_model(instrument, model_name), full_df, test_df)

            if not y_pred or len(y_pred) != len(y_true) or any(v is None for v in y_pred):
                print(f"[evaluate] SKIP: bad predictions for {instrument}__{model_name}")
                continue

            preds[model_name] = y_pred

        member_names = [m for m in CANDIDATE_MODELS if m in preds]
        if member_names:
            preds["ensemble"] = np.mean([np.array(preds[m], dtype=float) for m in member_names], axis=0)

        naive_mape = mape(y_true, preds["naive"])

        for model_name, y_pred in preds.items():
            r = rmse(y_true, y_pred)
            m = mae(y_true, y_pred)
            p = mape(y_true, y_pred)
            skill = 0.0 if model_name == "naive" else (round(1 - (p / naive_mape), 6) if naive_mape else 0.0)
            d_acc = directional_accuracy(y_true, y_pred, naive_pred)
            records.append({
                "instrument": instrument,
                "model_name": model_name,
                "rmse": round(r, 4),
                "mae": round(m, 4),
                "mape": round(p, 4),
                "skill_score_vs_naive": skill,
                "directional_accuracy": round(d_acc, 2) if not np.isnan(d_acc) else None,
                "selected": False,
            })
            print(f"[evaluate] {model_name}: RMSE={r:.2f}  MAE={m:.2f}  MAPE={p:.4f}%  "
                  f"skill_vs_naive={skill:+.4f}  dir_acc={d_acc:.1f}%")

    if not records:
        print("[evaluate] No results to write.")
        con.close()
        return

    scores_df = pd.DataFrame(records)

    def pick(group):
        passing = group[group["skill_score_vs_naive"] > 0]
        winner = "naive" if passing.empty else passing.loc[passing["skill_score_vs_naive"].idxmax(), "model_name"]
        group["selected"] = group["model_name"] == winner
        return group

    scores_df = scores_df.groupby("instrument", group_keys=False).apply(pick)

    con.execute("CREATE TABLE gold.model_scores AS SELECT * FROM scores_df")
    print("\n[evaluate] gold.model_scores written.")
    print(scores_df.to_string(index=False))

    con.close()


if __name__ == "__main__":
    evaluate_all()
    print("\n[evaluate] Done.")
