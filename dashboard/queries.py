"""
DuckDB query functions used by the Streamlit dashboard.
All functions return pandas DataFrames.
"""

import os
import duckdb
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "gold_forecast.duckdb")

INSTRUMENT_LABELS = {
    "gold_futures":  "Gold Futures (GC=F)",
    "goldbees_etf":  "Nippon Gold BeES ETF",
    "hdfc_gold_etf": "HDFC Gold ETF",
    "sbi_gold_nav":  "SBI Gold Fund (NAV)",
}

ALL_INSTRUMENTS = list(INSTRUMENT_LABELS.keys())
OHLCV_INSTRUMENTS = ["gold_futures", "goldbees_etf", "hdfc_gold_etf"]

# Every model predict.py generates a forecast for, per instrument.
ALL_MODEL_NAMES = ["naive", "prophet", "arima", "xgboost", "lightgbm", "lstm", "ensemble"]


def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH, read_only=True)


def get_prices(instrument: str, start_date: str, end_date: str) -> pd.DataFrame:
    with _con() as con:
        return con.execute(f"""
            SELECT date, open, high, low, close_inr AS close, volume, daily_return_pct, log_return
            FROM silver.prices
            WHERE instrument = '{instrument}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """).fetchdf()


def get_technical_features(instrument: str, start_date: str, end_date: str) -> pd.DataFrame:
    with _con() as con:
        return con.execute(f"""
            SELECT date, close_inr, ma_50, ma_200, bb_upper, bb_lower,
                   rsi_14, rolling_vol_30d, drawdown_pct, is_dip_historical
            FROM gold.technical_features
            WHERE instrument = '{instrument}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """).fetchdf()


def get_normalized_returns(instruments: list, start_date: str, end_date: str) -> pd.DataFrame:
    inst_list = ", ".join(f"'{i}'" for i in instruments)
    with _con() as con:
        df = con.execute(f"""
            SELECT date, instrument, return_from_inception_pct
            FROM gold.normalized_returns
            WHERE instrument IN ({inst_list})
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """).fetchdf()
    df["instrument_label"] = df["instrument"].map(INSTRUMENT_LABELS)
    return df


def get_forecast_normalized_returns(instruments: list, model_name: str = None) -> pd.DataFrame:
    """
    Projects each instrument's forecast into the same "% return from inception" terms as
    get_normalized_returns (same inception-price formula gold.py uses), so the Overview
    comparison chart can show a forecasted continuation of each instrument's growth line.
    Defaults to each instrument's own selected (best-choice) model if model_name is None.
    """
    inst_list = ", ".join(f"'{i}'" for i in instruments)
    model_join = (
        "" if model_name else
        "JOIN model_pick mp ON f.instrument = mp.instrument AND f.model_name = mp.model_name"
    )
    model_filter = f"AND f.model_name = '{model_name}'" if model_name else ""
    with _con() as con:
        df = con.execute(f"""
            WITH first_prices AS (
                SELECT instrument, MIN(date) AS inception_date
                FROM silver.prices WHERE instrument IN ({inst_list})
                GROUP BY instrument
            ),
            inception_close AS (
                SELECT sp.instrument, sp.close_inr AS inception_close
                FROM silver.prices sp
                JOIN first_prices fp ON sp.instrument = fp.instrument AND sp.date = fp.inception_date
            ),
            model_pick AS (
                SELECT instrument, model_name FROM gold.model_scores WHERE selected = true
            )
            SELECT f.date, f.instrument,
                   ((f.yhat - ic.inception_close) / ic.inception_close) * 100 AS return_from_inception_pct
            FROM gold.forecasts f
            JOIN inception_close ic ON f.instrument = ic.instrument
            {model_join}
            WHERE f.instrument IN ({inst_list}) AND f.is_future = true {model_filter}
            ORDER BY f.date
        """).fetchdf()
    df["instrument_label"] = df["instrument"].map(INSTRUMENT_LABELS)
    return df


def get_correlation_matrix(start_date: str, end_date: str) -> pd.DataFrame:
    with _con() as con:
        df = con.execute(f"""
            SELECT date, instrument, daily_return_pct
            FROM silver.prices
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
              AND daily_return_pct IS NOT NULL
            ORDER BY date
        """).fetchdf()
    pivot = df.pivot(index="date", columns="instrument", values="daily_return_pct")
    pivot = pivot.rename(columns=INSTRUMENT_LABELS)
    return pivot.corr()


def get_forecasts(instrument: str, horizon_days: int, model_name: str = None) -> pd.DataFrame:
    model_filter = f"AND model_name = '{model_name}'" if model_name else ""
    with _con() as con:
        return con.execute(f"""
            SELECT date, yhat, yhat_lower, yhat_upper, is_future, model_name
            FROM gold.forecasts
            WHERE instrument = '{instrument}' {model_filter}
            ORDER BY date
            LIMIT (SELECT COUNT(*) FROM silver.prices WHERE instrument = '{instrument}')
                 + {horizon_days}
        """).fetchdf()


def get_forecasts_future_only(instrument: str, model_name: str = None) -> pd.DataFrame:
    model_filter = f"AND model_name = '{model_name}'" if model_name else ""
    with _con() as con:
        return con.execute(f"""
            SELECT date, yhat, yhat_lower, yhat_upper, model_name
            FROM gold.forecasts
            WHERE instrument = '{instrument}' AND is_future = true {model_filter}
            ORDER BY date
        """).fetchdf()


def get_selected_model(instrument: str) -> str:
    """The model evaluate.py picked as the honest accuracy winner for this instrument."""
    with _con() as con:
        row = con.execute(f"""
            SELECT model_name FROM gold.model_scores
            WHERE instrument = '{instrument}' AND selected = true
        """).fetchone()
    return row[0] if row else None


def get_dip_tracker(instrument: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Historical only -- gold.technical_features has no future rows, so a forecast can't be
    joined onto it this way (an earlier version tried; it silently never matched anything).
    The forecast overlay for Dip Tracker instead comes from get_forecasts_future_only,
    drawn via the shared _add_forecast_series helper alongside this historical data.
    """
    with _con() as con:
        return con.execute(f"""
            SELECT date, close_inr, drawdown_pct, is_dip_historical
            FROM gold.technical_features
            WHERE instrument = '{instrument}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """).fetchdf()


def get_last_ma_200(instrument: str) -> float:
    """Last known 200-day average price -- used as the flat baseline for flagging a
    'possible future dip' in the forecast (same carry-forward convention the models
    themselves use for technical indicators over the forecast horizon)."""
    with _con() as con:
        row = con.execute(f"""
            SELECT ma_200 FROM gold.technical_features
            WHERE instrument = '{instrument}' AND ma_200 IS NOT NULL
            ORDER BY date DESC LIMIT 1
        """).fetchone()
    return row[0] if row else None


def get_model_scores() -> pd.DataFrame:
    with _con() as con:
        df = con.execute("""
            SELECT instrument, model_name, rmse, mae, mape,
                   skill_score_vs_naive, directional_accuracy, selected
            FROM gold.model_scores
            ORDER BY instrument, mape
        """).fetchdf()
    df["instrument_label"] = df["instrument"].map(INSTRUMENT_LABELS)
    return df


def get_date_range(instrument: str) -> tuple:
    with _con() as con:
        row = con.execute(f"""
            SELECT MIN(date), MAX(date)
            FROM silver.prices
            WHERE instrument = '{instrument}'
        """).fetchone()
    return row[0], row[1]


def get_live_predictions(instrument: str) -> pd.DataFrame:
    """
    Real day-by-day forward-test track record for this instrument's best-choice model:
    what was predicted for a given trading day (logged the day before it happened),
    versus what actually happened, once known. Different from gold.model_scores, which
    is a one-time backtest over the already-known 2025 holdout -- this accumulates live,
    one real row per trading day, going forward from whenever tracking started.
    """
    with _con() as con:
        try:
            return con.execute(f"""
                SELECT predicted_on, predicted_for, model_name, predicted_price,
                       actual_price, abs_error, pct_error
                FROM gold.live_predictions
                WHERE instrument = '{instrument}'
                ORDER BY predicted_for DESC
            """).fetchdf()
        except Exception:
            return pd.DataFrame()


def get_macro_snapshot() -> dict:
    """Latest macro/geopolitical-risk proxy reading + its own 1-year average -- see
    dashboard.insights.macro_commentary for how this becomes plain-language text."""
    from dashboard import insights
    with _con() as con:
        return insights.get_macro_snapshot(con)


def get_all_instruments_summary() -> pd.DataFrame:
    with _con() as con:
        return con.execute("""
            SELECT instrument,
                   COUNT(*) AS trading_days,
                   MIN(date) AS first_date,
                   MAX(date) AS last_date,
                   ROUND(MIN(close_inr), 2) AS min_price_inr,
                   ROUND(MAX(close_inr), 2) AS max_price_inr,
                   ROUND(AVG(close_inr), 2) AS avg_price_inr
            FROM silver.prices
            GROUP BY instrument
            ORDER BY instrument
        """).fetchdf()


def get_last_close(instrument: str) -> float:
    """Most recent actual close -- the reference point the consensus panel measures each
    model's forecast against."""
    with _con() as con:
        row = con.execute(f"""
            SELECT close_inr FROM silver.prices
            WHERE instrument = '{instrument}'
            ORDER BY date DESC LIMIT 1
        """).fetchone()
    return row[0] if row else None


def get_dip_forward_returns(instrument: str) -> pd.DataFrame:
    """
    Forward returns from EVERY historical day, tagged by whether that day was a dip
    (is_dip_historical) -- lets the Dip Tracker answer "was buying the dip actually a
    good idea?" with this instrument's own history rather than folklore. Horizons are
    trading-day offsets (21/63/126/252 rows ~= 1/3/6/12 months), the standard convention.
    Deliberately full-history: a backtest shouldn't be limited to the chart's window.
    """
    with _con() as con:
        return con.execute(f"""
            WITH px AS (
                SELECT date, close_inr, is_dip_historical,
                       LEAD(close_inr, 21)  OVER (ORDER BY date) AS close_30d,
                       LEAD(close_inr, 63)  OVER (ORDER BY date) AS close_90d,
                       LEAD(close_inr, 126) OVER (ORDER BY date) AS close_180d,
                       LEAD(close_inr, 252) OVER (ORDER BY date) AS close_365d
                FROM gold.technical_features
                WHERE instrument = '{instrument}'
            )
            SELECT date, is_dip_historical,
                   (close_30d  / close_inr - 1) * 100 AS fwd_30d,
                   (close_90d  / close_inr - 1) * 100 AS fwd_90d,
                   (close_180d / close_inr - 1) * 100 AS fwd_180d,
                   (close_365d / close_inr - 1) * 100 AS fwd_365d
            FROM px
            ORDER BY date
        """).fetchdf()


def get_monthly_returns(instrument: str) -> pd.DataFrame:
    """
    Month-over-month % returns (month-end close to month-end close) over full history,
    for the seasonality heatmap. The current in-progress calendar month is excluded --
    its "return" isn't final and would read as a misleading cold/hot cell.
    """
    with _con() as con:
        df = con.execute(f"""
            WITH monthly AS (
                SELECT date_trunc('month', date) AS month_start,
                       max_by(close_inr, date) AS month_close
                FROM silver.prices
                WHERE instrument = '{instrument}'
                GROUP BY 1
            )
            SELECT CAST(EXTRACT(year FROM month_start) AS INT)  AS year,
                   CAST(EXTRACT(month FROM month_start) AS INT) AS month,
                   (month_close / LAG(month_close) OVER (ORDER BY month_start) - 1) * 100
                       AS monthly_return_pct
            FROM monthly
            ORDER BY year, month
        """).fetchdf()
    df = df.dropna(subset=["monthly_return_pct"])
    today = pd.Timestamp.today()
    return df[~((df["year"] == today.year) & (df["month"] == today.month))].reset_index(drop=True)


EVENTS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "events.csv")


def get_events(start_date: str, end_date: str) -> pd.DataFrame:
    """Curated market events (import-duty changes, crises, policy pivots) from the
    git-tracked data/events.csv -- pure file read, no DuckDB. Empty DataFrame if the
    file is missing so the dashboard degrades gracefully."""
    try:
        events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    except Exception:
        return pd.DataFrame(columns=["date", "label", "description"])
    mask = (events["date"] >= pd.Timestamp(start_date)) & (events["date"] <= pd.Timestamp(end_date))
    return events[mask].sort_values("date").reset_index(drop=True)
