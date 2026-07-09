"""
Transform bronze tables into silver.prices — a single unified table.

Transformations applied:
- Unify all instruments into one table with a consistent schema
- Convert GC=F (USD) close price to INR using USDINR=X
- NAV-only instruments (sbi_gold_nav) get nulls for open/high/low/volume
- Drop weekends and rows with null close/nav
- Compute daily_return_pct and log_return from close_inr
Run after bronze.py.
"""

import os
import duckdb
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "gold_forecast.duckdb")


def build_silver(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute("DROP TABLE IF EXISTS silver.prices")

    # Build unified table from all instruments
    con.execute("""
        CREATE TABLE silver.prices AS

        WITH gold_futures_inr AS (
            SELECT
                gf.date,
                'gold_futures' AS instrument,
                gf.open * fx.close    AS open,
                gf.high * fx.close    AS high,
                gf.low  * fx.close    AS low,
                gf.close * fx.close   AS close,
                gf.volume             AS volume,
                gf.close * fx.close   AS close_inr
            FROM bronze.gold_futures gf
            JOIN bronze.usd_inr fx ON gf.date = fx.date
            WHERE gf.close IS NOT NULL AND fx.close IS NOT NULL
        ),

        goldbees AS (
            SELECT
                date,
                'goldbees_etf'  AS instrument,
                open, high, low, close, volume,
                close           AS close_inr
            FROM bronze.goldbees_etf
            WHERE close IS NOT NULL
        ),

        hdfc_gold AS (
            SELECT
                date,
                'hdfc_gold_etf' AS instrument,
                open, high, low, close, volume,
                close           AS close_inr
            FROM bronze.hdfc_gold_etf
            WHERE close IS NOT NULL
        ),

        sbi_nav AS (
            SELECT
                date,
                'sbi_gold_nav'  AS instrument,
                NULL::DOUBLE    AS open,
                NULL::DOUBLE    AS high,
                NULL::DOUBLE    AS low,
                nav             AS close,
                NULL::DOUBLE    AS volume,
                nav             AS close_inr
            FROM bronze.sbi_gold_nav
            WHERE nav IS NOT NULL
        ),

        combined AS (
            SELECT * FROM gold_futures_inr
            UNION ALL
            SELECT * FROM goldbees
            UNION ALL
            SELECT * FROM hdfc_gold
            UNION ALL
            SELECT * FROM sbi_nav
        ),

        -- Data-quality guard: drop bad vendor prints (e.g. a 1-2 day price collapse that
        -- fully reverts) by comparing each close to the local 11-day rolling median for
        -- that instrument. A real market move doesn't snap back to the prior level within
        -- days, so anything more than 4x away from its own neighborhood is a vendor glitch,
        -- not a price. Found via: GOLDBEES.NS printed ~0.30-0.34 INR on 2019-12-19/20
        -- (vs. ~33.5 before and after, with 100x normal volume) before recovering fully.
        with_rolling_median AS (
            SELECT *,
                MEDIAN(close_inr) OVER (
                    PARTITION BY instrument ORDER BY date
                    ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING
                ) AS rolling_median_close
            FROM combined
        ),

        cleaned AS (
            SELECT * EXCLUDE (rolling_median_close)
            FROM with_rolling_median
            WHERE rolling_median_close IS NULL
               OR close_inr BETWEEN 0.25 * rolling_median_close AND 4.0 * rolling_median_close
        ),

        with_prev AS (
            SELECT *,
                LAG(close_inr) OVER (
                    PARTITION BY instrument ORDER BY date
                ) AS prev_close_inr
            FROM cleaned
        )

        SELECT
            date,
            instrument,
            CAST(open AS DOUBLE)       AS open,
            CAST(high AS DOUBLE)       AS high,
            CAST(low  AS DOUBLE)       AS low,
            CAST(close AS DOUBLE)      AS close,
            CAST(volume AS DOUBLE)     AS volume,
            CAST(close_inr AS DOUBLE)  AS close_inr,
            CASE
                WHEN prev_close_inr IS NOT NULL AND prev_close_inr != 0
                THEN ROUND(((close_inr - prev_close_inr) / prev_close_inr) * 100, 6)
                ELSE NULL
            END AS daily_return_pct,
            CASE
                WHEN prev_close_inr IS NOT NULL AND prev_close_inr > 0 AND close_inr > 0
                THEN ROUND(LN(close_inr / prev_close_inr), 8)
                ELSE NULL
            END AS log_return
        FROM with_prev
        ORDER BY instrument, date
    """)

    row_count = con.execute("SELECT COUNT(*) FROM silver.prices").fetchone()[0]
    print(f"[silver] Built silver.prices — {row_count} rows")

    summary = con.execute("""
        SELECT instrument, COUNT(*) AS row_cnt, MIN(date) AS first_date, MAX(date) AS last_date
        FROM silver.prices
        GROUP BY instrument
        ORDER BY instrument
    """).fetchdf()
    print(summary.to_string(index=False))


def build_macro_features(con: duckdb.DuckDBPyConnection) -> None:
    """
    Macro/geopolitical-risk proxy series (VIX, crude oil, USD index, 10Y yield),
    unified by date. Deliberately keeps NULLs where a series didn't trade on a given
    date (different market calendars) -- carry-forward happens downstream (ASOF JOIN
    in models/common.py) when this is attached to per-instrument feature rows.
    """
    con.execute("DROP TABLE IF EXISTS silver.macro_features")
    con.execute("""
        CREATE TABLE silver.macro_features AS
        WITH vix AS (SELECT date, close AS vix_close FROM bronze.vix WHERE close IS NOT NULL),
        oil  AS (SELECT date, close AS oil_close FROM bronze.crude_oil WHERE close IS NOT NULL),
        usd  AS (SELECT date, close AS usd_index_close FROM bronze.usd_index WHERE close IS NOT NULL),
        y10  AS (SELECT date, close AS us10y_yield_close FROM bronze.us10y_yield WHERE close IS NOT NULL),
        -- USD/INR: a *direct* driver of INR-priced gold (INR gold ~ USD gold x USDINR),
        -- not just a conversion rate -- surfaced here so the dashboard commentary and
        -- (eventually) the models can use it like any other macro series.
        inr  AS (SELECT date, close AS usdinr_close FROM bronze.usd_inr WHERE close IS NOT NULL),
        all_dates AS (
            SELECT date FROM vix
            UNION SELECT date FROM oil
            UNION SELECT date FROM usd
            UNION SELECT date FROM y10
            UNION SELECT date FROM inr
        )
        SELECT
            ad.date,
            v.vix_close,
            o.oil_close,
            u.usd_index_close,
            t.us10y_yield_close,
            i.usdinr_close
        FROM all_dates ad
        LEFT JOIN vix v ON ad.date = v.date
        LEFT JOIN oil  o ON ad.date = o.date
        LEFT JOIN usd  u ON ad.date = u.date
        LEFT JOIN y10  t ON ad.date = t.date
        LEFT JOIN inr  i ON ad.date = i.date
        ORDER BY ad.date
    """)
    row_count = con.execute("SELECT COUNT(*) FROM silver.macro_features").fetchone()[0]
    date_range = con.execute("SELECT MIN(date), MAX(date) FROM silver.macro_features").fetchone()
    print(f"[silver] Built silver.macro_features — {row_count} rows ({date_range[0]} to {date_range[1]})")


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    build_silver(con)
    build_macro_features(con)
    con.close()
    print("[silver] Done.")
