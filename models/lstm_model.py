"""
LSTM candidate model — sequence-based forecaster using close_inr + macro features.
Two prediction modes mirror how tree models are already handled in this project:
- predict_at_dates: non-recursive, uses true historical windows (evaluate.py's backtest)
- forecast_lstm: recursive multi-step rollout (predict.py's live 90-day forecast)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from common import (
    MACRO_COLS, SAVED_DIR, TARGET_COL, business_days_forward, log_return_clip_bounds, DAMPING_PHI,
    historical_daily_vol, random_walk_confidence_band,
)

# Predicts TARGET_COL (log_return), not raw close_inr: same extrapolation problem as the
# tree models otherwise (a neural net trained only on 2003-2024 price levels has no basis
# for 2025-2026 prices far outside that range). Returns are stationary, so the model never
# needs to extrapolate; price is reconstructed downstream by compounding predicted returns
# onto a running price estimate (see forecast_lstm / predict_at_dates).
INPUT_COLS = [TARGET_COL] + MACRO_COLS
SEQ_LEN = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
VAL_DAYS = 60


class GoldLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _make_windows(feat: np.ndarray, target: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(feat) - seq_len):
        X.append(feat[i:i + seq_len])
        y.append(target[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(train_df: pd.DataFrame) -> dict:
    torch.manual_seed(42)
    df = train_df[["date"] + INPUT_COLS].dropna(subset=INPUT_COLS).reset_index(drop=True)

    fit_df = df.iloc[:-VAL_DAYS]
    val_df = df.iloc[-(VAL_DAYS + SEQ_LEN):]  # include lookback overlap for windowing

    scaler_X = MinMaxScaler().fit(fit_df[INPUT_COLS])
    scaler_y = MinMaxScaler().fit(fit_df[[TARGET_COL]])

    fit_X = scaler_X.transform(fit_df[INPUT_COLS])
    fit_y = scaler_y.transform(fit_df[[TARGET_COL]]).ravel()
    val_X = scaler_X.transform(val_df[INPUT_COLS])
    val_y = scaler_y.transform(val_df[[TARGET_COL]]).ravel()

    Xtr, ytr = _make_windows(fit_X, fit_y, SEQ_LEN)
    Xval, yval = _make_windows(val_X, val_y, SEQ_LEN)

    model = GoldLSTM(input_size=len(INPUT_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xval_t, yval_t = torch.tensor(Xval), torch.tensor(yval)

    best_val, best_state, patience_ctr = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()
        if val_loss < best_val:
            best_val, best_state, patience_ctr = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return {
        "state_dict": best_state, "scaler_X": scaler_X, "scaler_y": scaler_y,
        "input_cols": INPUT_COLS, "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
    }


def _rebuild_model(artifact: dict) -> GoldLSTM:
    model = GoldLSTM(len(artifact["input_cols"]), artifact["hidden_size"],
                      artifact["num_layers"], artifact["dropout"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def save_lstm(artifact: dict, instrument: str) -> None:
    torch.save(artifact, os.path.join(SAVED_DIR, f"{instrument}__lstm.pt"))


def load_lstm(instrument: str) -> dict:
    # weights_only=False: artifact bundles sklearn scalers, not just tensors.
    return torch.load(os.path.join(SAVED_DIR, f"{instrument}__lstm.pt"), weights_only=False)


def predict_at_dates(artifact: dict, full_df: pd.DataFrame, dates) -> list:
    """
    Non-recursive: true-history window ending the day before each target date.
    Model predicts log_return; reconstruct price as prev_actual_close * exp(pred_log_return)
    (walk-forward against known history — matches how naive/arima/prophet are scored, and
    avoids compounding error in the backtest).
    """
    model = _rebuild_model(artifact)
    df = full_df.dropna(subset=artifact["input_cols"] + ["close_inr"]).reset_index(drop=True)
    scaler_X, scaler_y, seq_len = artifact["scaler_X"], artifact["scaler_y"], artifact["seq_len"]
    input_cols = artifact["input_cols"]

    idx_by_date = {d: i for i, d in enumerate(df["date"])}
    windows, valid_positions, prev_closes = [], [], []
    for pos, d in enumerate(pd.Series(dates).reset_index(drop=True)):
        i = idx_by_date.get(d)
        if i is None or i < seq_len:
            continue
        window = df.iloc[i - seq_len:i][input_cols].values
        windows.append(scaler_X.transform(window))
        valid_positions.append(pos)
        prev_closes.append(df["close_inr"].iloc[i - 1])

    result = [None] * len(dates)
    if not windows:
        return result

    X = torch.tensor(np.array(windows, dtype=np.float32))
    with torch.no_grad():
        preds_scaled = model(X).numpy()
    pred_log_returns = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    pred_prices = np.array(prev_closes) * np.exp(pred_log_returns)
    for pos, val in zip(valid_positions, pred_prices):
        result[pos] = float(val)
    return result


def forecast_lstm(artifact: dict, df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    """
    Recursive multi-step forecast, mirrors forecast_tree_model's iterative pattern.
    Model predicts log_return at each step; price is reconstructed by compounding
    predicted returns onto a running price estimate (fitted values use the true previous
    actual close instead, avoiding compounding error in the historical/fit view).
    """
    model = _rebuild_model(artifact)
    scaler_X, scaler_y, seq_len = artifact["scaler_X"], artifact["scaler_y"], artifact["seq_len"]
    input_cols = artifact["input_cols"]

    df2 = df.dropna(subset=input_cols + ["close_inr"]).reset_index(drop=True)

    # In-sample fitted values (true windows, walk-forward reconstruction)
    fitted_rows = []
    for i in range(seq_len, len(df2)):
        window = scaler_X.transform(df2.iloc[i - seq_len:i][input_cols].values)
        with torch.no_grad():
            pred_scaled = model(torch.tensor(window[None].astype(np.float32))).item()
        pred_log_return = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        prev_actual_close = df2["close_inr"].iloc[i - 1]
        pred_price = prev_actual_close * np.exp(pred_log_return)
        fitted_rows.append({"date": df2["date"].iloc[i], "yhat": pred_price,
                             "yhat_lower": None, "yhat_upper": None, "is_future": False})
    fitted = pd.DataFrame(fitted_rows)

    # Recursive future forecast — macro columns held flat at their last known value,
    # predicted returns (clipped to the instrument's actual historical range, so a small
    # per-step bias can't compound into an unrealistic 90-day price swing) onto a running
    # price estimate.
    clip_lo, clip_hi = log_return_clip_bounds(df2)
    sigma = historical_daily_vol(df2)
    window_buffer = df2.tail(seq_len)[input_cols].values.tolist()
    last_macro = df2[MACRO_COLS].iloc[-1].values.tolist()
    running_price = float(df2["close_inr"].iloc[-1])
    future_rows = []
    for step, day in enumerate(business_days_forward(df2["date"].max().date(), forecast_days), start=1):
        scaled_window = scaler_X.transform(np.array(window_buffer))
        with torch.no_grad():
            pred_scaled = model(torch.tensor(scaled_window[None].astype(np.float32))).item()
        pred_log_return = float(np.clip(scaler_y.inverse_transform([[pred_scaled]])[0][0], clip_lo, clip_hi))
        # Note: the (undamped) predicted return still feeds the window buffer so the
        # model's own recurrent state evolves normally -- only the compounded PRICE uses
        # the damped value, so damping doesn't fight the model's internal dynamics.
        damped_log_return = pred_log_return * (DAMPING_PHI ** step)
        running_price = running_price * np.exp(damped_log_return)
        window_buffer.append([pred_log_return] + last_macro)
        window_buffer.pop(0)
        lo, hi = random_walk_confidence_band(running_price, step, sigma)
        future_rows.append({"date": pd.Timestamp(day), "yhat": running_price,
                             "yhat_lower": lo, "yhat_upper": hi, "is_future": True})

    return pd.concat([fitted, pd.DataFrame(future_rows)], ignore_index=True)
