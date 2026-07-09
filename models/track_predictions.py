"""
Live forward-testing for EVERY model, per instrument (not just whichever
gold.model_scores.selected picks) -- so any chart in the dashboard that shows any model's
forecast can also show that model's real day-by-day track record. This is deliberately
different from evaluate.py's 2025 holdout backtest, which replays a whole year of
already-known history at once. This script builds a genuine day-by-day track record:

1. Reconcile: for any previously-logged prediction whose target date has now actually
   happened (new data arrived via the day's collect->bronze->silver run), fill in the
   real closing price and compute the error.
2. Log a new prediction: for every (instrument, model_name) combination present in
   gold.forecasts, read tomorrow's forecast (the first future row) and record it, to be
   reconciled once tomorrow's actual data arrives.

Run daily, after predict.py (needs its freshly generated gold.forecasts). Idempotent --
running it twice on the same day won't double-log a prediction for the same
(instrument, model_name, target date).
"""

import os
import sys
import duckdb

sys.path.insert(0, os.path.dirname(__file__))
from common import DB_PATH


def track_predictions() -> None:
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    con.execute("""
        CREATE TABLE IF NOT EXISTS gold.live_predictions (
            instrument VARCHAR,
            model_name VARCHAR,
            predicted_on DATE,
            predicted_for DATE,
            predicted_price DOUBLE,
            actual_price DOUBLE,
            abs_error DOUBLE,
            pct_error DOUBLE
        )
    """)

    # 1. Reconcile any predictions whose target date has now actually happened.
    # (model-agnostic -- the actual price is the same regardless of which model predicted it)
    updated = con.execute("""
        UPDATE gold.live_predictions lp
        SET actual_price = sp.close_inr,
            abs_error = ABS(sp.close_inr - lp.predicted_price),
            pct_error = ABS(sp.close_inr - lp.predicted_price) / sp.close_inr * 100
        FROM silver.prices sp
        WHERE lp.actual_price IS NULL
          AND sp.instrument = lp.instrument
          AND sp.date = lp.predicted_for
    """)
    print(f"[track] Reconciled {updated.fetchone()[0] if updated else 0} prior prediction(s) with actuals.")

    # 2. Log a new prediction for the next trading day, for every (instrument, model) pair.
    combos = con.execute("""
        SELECT DISTINCT instrument, model_name FROM gold.forecasts ORDER BY 1, 2
    """).fetchdf()
    today = con.execute("SELECT MAX(date) FROM silver.prices").fetchone()[0]

    logged, skipped = 0, 0
    for _, row in combos.iterrows():
        instrument, model_name = row["instrument"], row["model_name"]
        next_row = con.execute(f"""
            SELECT date, yhat FROM gold.forecasts
            WHERE instrument = '{instrument}' AND model_name = '{model_name}' AND is_future = true
            ORDER BY date LIMIT 1
        """).fetchone()
        if next_row is None:
            continue
        predicted_for, predicted_price = next_row

        already_logged = con.execute(f"""
            SELECT COUNT(*) FROM gold.live_predictions
            WHERE instrument = '{instrument}' AND model_name = '{model_name}'
              AND predicted_for = '{predicted_for}'
        """).fetchone()[0]
        if already_logged:
            skipped += 1
            continue

        con.execute("""
            INSERT INTO gold.live_predictions
                (instrument, model_name, predicted_on, predicted_for, predicted_price,
                 actual_price, abs_error, pct_error)
            VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
        """, [instrument, model_name, today, predicted_for, predicted_price])
        logged += 1

    print(f"[track] Logged {logged} new prediction(s), {skipped} already logged for today.")
    con.close()


if __name__ == "__main__":
    track_predictions()
    print("\n[track] Done.")
