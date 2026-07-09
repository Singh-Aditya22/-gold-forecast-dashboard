"""
Train Prophet, ARIMA (with macro exogenous regressors), XGBoost, LightGBM (both
hyperparameter-tuned), and LSTM for each OHLCV instrument.
SBI Gold Fund (NAV-only) is excluded — insufficient features for lag-based models.

Train/test split: train on data up to 2024-12-31, test on 2025-01-01 to 2025-12-31.
This is the occasional/manual step (not part of the daily refresh.sh cron) — it does
the expensive hyperparameter search and persists the winning params to models/saved/
so predict.py's daily production retrain can reuse them without re-searching.

Trained models are saved to models/saved/ as pickle files.
Run after gold.py, before evaluate.py.
"""

import os
import sys
import warnings
import duckdb
import numpy as np
from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DB_PATH, SAVED_DIR, TRAIN_END, INSTRUMENTS, MACRO_COLS, FEATURE_COLS, TARGET_COL,
    DEFAULT_XGB_PARAMS, DEFAULT_LGBM_PARAMS,
    load_instrument_data, add_lag_features, save_model, save_best_params,
)
import lstm_model

warnings.filterwarnings("ignore")

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
}
LGBM_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8, -1],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "num_leaves": [15, 31, 63, 127],
}
N_ITER_SEARCH = 30


def _cv_splits(n_rows: int) -> int:
    # Smaller-history instruments (e.g. hdfc_gold_etf) may not support 5 expanding splits.
    return 5 if n_rows >= 600 else 3


def train_prophet(train_df) -> Prophet:
    prophet_df = train_df[["date", "close_inr"] + MACRO_COLS].rename(
        columns={"date": "ds", "close_inr": "y"})
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    for col in MACRO_COLS:
        model.add_regressor(col)
    model.fit(prophet_df)
    return model


def train_arima(train_df):
    exog = train_df[MACRO_COLS].values
    return auto_arima(
        train_df["close_inr"], X=exog,
        seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore",
        max_p=5, max_q=5, max_d=2,
    )


def train_xgboost(train_df, params: dict = None) -> XGBRegressor:
    """
    Plain fit with given (or default) hyperparams — used for the daily production retrain.
    Predicts TARGET_COL (log_return), not raw price level: tree models can't extrapolate
    beyond the target range seen during training, and gold's price today is far outside
    2003-2024 training data. Returns are stationary regardless of the price trend, so the
    model never has to extrapolate — price is reconstructed downstream from the predicted
    return (see predict.py/evaluate.py).
    """
    params = params or DEFAULT_XGB_PARAMS
    df = add_lag_features(train_df)
    X, y = df[FEATURE_COLS].values, df[TARGET_COL].values
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X, y)
    return model


def train_lightgbm(train_df, params: dict = None) -> LGBMRegressor:
    """Plain fit with given (or default) hyperparams — used for the daily production retrain."""
    params = params or DEFAULT_LGBM_PARAMS
    df = add_lag_features(train_df)
    X, y = df[FEATURE_COLS].values, df[TARGET_COL].values
    model = LGBMRegressor(**params, random_state=42, verbose=-1)
    model.fit(X, y)
    return model


def tune_xgboost(train_df):
    # n_jobs=1 on the estimator: RandomizedSearchCV(n_jobs=-1) already parallelizes
    # across candidates/folds — letting XGBoost ALSO multi-thread internally causes
    # massive thread oversubscription (search workers x model threads) and can make
    # the search dramatically slower, not faster.
    # scoring is MAE not MAPE: log_return values hover near zero, so MAPE's division
    # by the true value is unstable/undefined here (fine for price-level targets, not returns).
    df = add_lag_features(train_df)
    X, y = df[FEATURE_COLS].values, df[TARGET_COL].values
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0, n_jobs=1), XGB_PARAM_DIST,
        n_iter=N_ITER_SEARCH, cv=TimeSeriesSplit(n_splits=_cv_splits(len(df))),
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_lightgbm(train_df):
    df = add_lag_features(train_df)
    X, y = df[FEATURE_COLS].values, df[TARGET_COL].values
    search = RandomizedSearchCV(
        LGBMRegressor(random_state=42, verbose=-1, n_jobs=1), LGBM_PARAM_DIST,
        n_iter=N_ITER_SEARCH, cv=TimeSeriesSplit(n_splits=_cv_splits(len(df))),
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_all() -> None:
    os.makedirs(SAVED_DIR, exist_ok=True)
    con = duckdb.connect(DB_PATH)

    for instrument in INSTRUMENTS:
        print(f"\n[train] === {instrument} ===")
        df = load_instrument_data(con, instrument)
        train_df = df[df["date"] <= TRAIN_END].copy()
        print(f"[train] Training rows: {len(train_df)}")

        print("[train] Training Prophet (with macro regressors)...")
        save_model(train_prophet(train_df), instrument, "prophet")
        print(f"[train] Saved {instrument}__prophet.pkl")

        print("[train] Training ARIMA (with macro exogenous features)...")
        save_model(train_arima(train_df), instrument, "arima")
        print(f"[train] Saved {instrument}__arima.pkl")

        print("[train] Tuning XGBoost (RandomizedSearchCV + TimeSeriesSplit)...")
        xgb_model, xgb_params = tune_xgboost(train_df)
        save_model(xgb_model, instrument, "xgboost")
        save_best_params(instrument, "xgboost", xgb_params)
        print(f"[train] Saved {instrument}__xgboost.pkl — best params: {xgb_params}")

        print("[train] Tuning LightGBM (RandomizedSearchCV + TimeSeriesSplit)...")
        lgbm_model, lgbm_params = tune_lightgbm(train_df)
        save_model(lgbm_model, instrument, "lightgbm")
        save_best_params(instrument, "lightgbm", lgbm_params)
        print(f"[train] Saved {instrument}__lightgbm.pkl — best params: {lgbm_params}")

        print("[train] Training LSTM...")
        lstm_artifact = lstm_model.train_lstm(train_df)
        lstm_model.save_lstm(lstm_artifact, instrument)
        print(f"[train] Saved {instrument}__lstm.pt")

    con.close()


if __name__ == "__main__":
    train_all()
    print("\n[train] Done. All models saved.")
