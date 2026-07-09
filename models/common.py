"""
Shared constants and functions used by train.py, evaluate.py, and predict.py.
Centralized so the feature list (especially MACRO_COLS) stays consistent
across training, evaluation, and live forecasting.
"""

import os
import json
import pickle
import duckdb
import pandas as pd
import numpy as np
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "gold_forecast.duckdb")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

INSTRUMENTS = ["gold_futures", "goldbees_etf", "hdfc_gold_etf"]

LAG_COLS = ["lag_1", "lag_7", "lag_30"]
TECH_COLS = ["ma_50", "ma_200", "rsi_14", "bb_upper", "bb_lower", "rolling_vol_30d", "drawdown_pct"]
MACRO_COLS = ["vix_close", "oil_close", "usd_index_close", "us10y_yield_close", "usdinr_close"]
FEATURE_COLS = LAG_COLS + TECH_COLS + MACRO_COLS
TARGET_COL = "log_return"

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
}
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
}


def load_instrument_data(con: duckdb.DuckDBPyConnection, instrument: str) -> pd.DataFrame:
    """
    Per-instrument prices + technical features + macro proxies, with TECH_COLS/MACRO_COLS
    shifted back 1 day so every feature at row t reflects only information known as of t-1.

    Without this shift there's same-day leakage two ways: (1) gold.technical_features'
    rolling windows are defined "N PRECEDING AND CURRENT ROW", so e.g. ma_50 at date t
    already includes date t's own close_inr -- using it to predict close_inr[t]/log_return[t]
    leaks the target into its own feature; (2) the macro ASOF JOIN matches "inst.date >= mf.date",
    which can match same-day macro data. lag_1/7/30 (built in add_lag_features) already
    reference the past by construction and don't need this additional shift.
    """
    df = con.execute(f"""
        WITH inst AS (
            SELECT sp.date, sp.close_inr, sp.log_return,
                   tf.ma_50, tf.ma_200, tf.rsi_14, tf.bb_upper, tf.bb_lower,
                   tf.rolling_vol_30d, tf.drawdown_pct
            FROM silver.prices sp
            JOIN gold.technical_features tf
              ON sp.date = tf.date AND sp.instrument = tf.instrument
            WHERE sp.instrument = '{instrument}'
              AND sp.close_inr IS NOT NULL
        )
        SELECT
            inst.*, mf.vix_close, mf.oil_close, mf.usd_index_close, mf.us10y_yield_close,
            mf.usdinr_close
        FROM inst
        ASOF LEFT JOIN silver.macro_features mf
          ON inst.date >= mf.date
        ORDER BY inst.date
    """).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = fill_macro_gaps(df)

    shift_cols = TECH_COLS + MACRO_COLS
    df[shift_cols] = df[shift_cols].shift(1)
    # A few leading rows can still be NaN after the shift: row 0 has nothing to shift from,
    # and gold.technical_features' own first row(s) are NaN for stddev-based indicators
    # (bb_upper/bb_lower/rolling_vol_30d) since a rolling stddev of a single point is
    # undefined. Drop rather than fabricate a value -- downstream consumers that don't go
    # through add_lag_features' dropna (train_prophet, train_arima) would otherwise choke.
    return df.dropna(subset=shift_cols).reset_index(drop=True)


def fill_macro_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Handle rows before the first available macro data point (ASOF has no match yet)."""
    df = df.copy()
    df[MACRO_COLS] = df[MACRO_COLS].ffill().bfill()
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1"] = df["close_inr"].shift(1)
    df["lag_7"] = df["close_inr"].shift(7)
    df["lag_30"] = df["close_inr"].shift(30)
    return df.dropna(subset=FEATURE_COLS + [TARGET_COL])


# Damped-trend factor for recursive multi-step forecasts (tree models, LSTM): each step's
# predicted return is scaled by DAMPING_PHI ** step_index before compounding. Clipping
# alone doesn't stop compounding drift when the model outputs a persistent, individually-
# unremarkable return (e.g. a steady +0.4%/day is nowhere near an outlier day for gold,
# but repeated for 90 straight days it compounds to +45%). Damping shrinks the predicted
# move toward zero as the horizon lengthens -- standard practice in forecasting theory
# (Gardner & McKenzie's damped trend) since genuine confidence in a specific direction
# should decay over a multi-month horizon, not compound at a constant rate forever.
DAMPING_PHI = 0.90


def historical_daily_vol(df: pd.DataFrame) -> float:
    """Std dev of this instrument's actual daily log_return — the volatility used to size
    a growing confidence band around any point forecast that has no built-in uncertainty
    estimate (naive, tree models, LSTM all just produce a single number per day)."""
    return float(df["log_return"].std())


def random_walk_confidence_band(center_price: float, step: int, sigma: float, z: float = 1.96):
    """
    Closed-form random-walk confidence interval: a random walk's variance grows linearly
    with time, so its h-step-ahead standard deviation is sigma*sqrt(h) -- the interval
    widens as sqrt(step), not linearly and not constant. Returns (lower, upper) around
    center_price at forecast step `step` (1-indexed).
    """
    half_width = z * sigma * (step ** 0.5)
    return center_price * np.exp(-half_width), center_price * np.exp(half_width)


def log_return_clip_bounds(df: pd.DataFrame, lower_pct: float = 1.0, upper_pct: float = 99.0):
    """
    Historical 1st/99th percentile of this instrument's actual daily log_return.
    Used to clip predicted returns during a RECURSIVE multi-step forecast (tree models,
    LSTM) -- without this, even a tiny systematic bias in the 1-step model compounds
    multiplicatively over 90 chained steps into an unrealistic price trajectory (observed:
    a consistent +0.4%/day bias alone produces a ~45% run-up by day 90). Clipping keeps
    each step within the range of moves the instrument has actually made historically.
    """
    return (
        float(df["log_return"].quantile(lower_pct / 100)),
        float(df["log_return"].quantile(upper_pct / 100)),
    )


def business_days_forward(start: date, n: int) -> list:
    days = []
    current = start
    while len(days) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days


def save_model(model, instrument: str, model_name: str) -> None:
    path = os.path.join(SAVED_DIR, f"{instrument}__{model_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(instrument: str, model_name: str):
    path = os.path.join(SAVED_DIR, f"{instrument}__{model_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _params_path(instrument: str, model_name: str) -> str:
    return os.path.join(SAVED_DIR, f"{instrument}__{model_name}_params.json")


def save_best_params(instrument: str, model_name: str, params: dict) -> None:
    with open(_params_path(instrument, model_name), "w") as f:
        json.dump(params, f, indent=2)


def load_best_params(instrument: str, model_name: str, default: dict) -> dict:
    path = _params_path(instrument, model_name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)
