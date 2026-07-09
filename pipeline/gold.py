"""
Build the gold analytics layer from silver.prices.

Tables produced:
  gold.technical_features  — MA50, MA200, RSI, Bollinger Bands, rolling vol, drawdown, dip flag
  gold.normalized_returns  — % return from each instrument's inception date
  gold.etf_premium         — each ETF's price ratio to international gold (INR), z-scored
Run after silver.py.
"""

import os
import duckdb

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "gold_forecast.duckdb")


def build_technical_features(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS gold.technical_features")
    con.execute("""
        CREATE TABLE gold.technical_features AS

        WITH base AS (
            SELECT
                date,
                instrument,
                close_inr,
                open, high, low, volume
            FROM silver.prices
            ORDER BY instrument, date
        ),

        with_mas AS (
            SELECT *,
                AVG(close_inr) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                ) AS ma_50,
                AVG(close_inr) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
                ) AS ma_200,
                STDDEV(close_inr) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS bb_stddev,
                AVG(close_inr) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS bb_mid,
                STDDEV(close_inr) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS rolling_vol_30d
            FROM base
        ),

        -- RSI: 14-period using Wilder's method approximated with simple avg of gains/losses
        with_change AS (
            SELECT *,
                close_inr - LAG(close_inr) OVER (
                    PARTITION BY instrument ORDER BY date
                ) AS price_change
            FROM with_mas
        ),

        with_rsi_components AS (
            SELECT *,
                CASE WHEN price_change > 0 THEN price_change ELSE 0 END AS gain,
                CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END AS loss
            FROM with_change
        ),

        with_avg_gl AS (
            SELECT *,
                AVG(gain) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) AS avg_gain,
                AVG(loss) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) AS avg_loss
            FROM with_rsi_components
        ),

        -- Rolling peak for drawdown calculation
        with_peak AS (
            SELECT *,
                MAX(close_inr) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS rolling_peak
            FROM with_avg_gl
        )

        SELECT
            date,
            instrument,
            ROUND(close_inr, 4)     AS close_inr,
            open, high, low, volume,
            ROUND(ma_50, 4)         AS ma_50,
            ROUND(ma_200, 4)        AS ma_200,
            ROUND(bb_mid + 2 * bb_stddev, 4)  AS bb_upper,
            ROUND(bb_mid - 2 * bb_stddev, 4)  AS bb_lower,
            ROUND(rolling_vol_30d, 6)          AS rolling_vol_30d,
            -- RSI: 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
            CASE
                WHEN avg_loss = 0 THEN 100
                ELSE ROUND(100 - (100 / (1 + avg_gain / NULLIF(avg_loss, 0))), 2)
            END AS rsi_14,
            -- Drawdown: how far below the rolling peak
            CASE
                WHEN rolling_peak > 0
                THEN ROUND(((close_inr - rolling_peak) / rolling_peak) * 100, 4)
                ELSE NULL
            END AS drawdown_pct,
            -- Historical dip: price more than 5% below 200-day MA
            CASE
                WHEN ma_200 IS NOT NULL AND ma_200 > 0
                    AND close_inr < ma_200 * 0.95
                THEN TRUE
                ELSE FALSE
            END AS is_dip_historical
        FROM with_peak
        ORDER BY instrument, date
    """)

    row_count = con.execute("SELECT COUNT(*) FROM gold.technical_features").fetchone()[0]
    print(f"[gold] Built gold.technical_features — {row_count} rows")


def build_normalized_returns(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS gold.normalized_returns")
    con.execute("""
        CREATE TABLE gold.normalized_returns AS

        WITH first_prices AS (
            SELECT instrument, MIN(date) AS inception_date
            FROM silver.prices
            GROUP BY instrument
        ),

        inception_close AS (
            SELECT sp.instrument, sp.close_inr AS inception_close
            FROM silver.prices sp
            JOIN first_prices fp
              ON sp.instrument = fp.instrument AND sp.date = fp.inception_date
        )

        SELECT
            sp.date,
            sp.instrument,
            ROUND(
                ((sp.close_inr - ic.inception_close) / ic.inception_close) * 100,
                4
            ) AS return_from_inception_pct
        FROM silver.prices sp
        JOIN inception_close ic ON sp.instrument = ic.instrument
        ORDER BY sp.instrument, sp.date
    """)

    row_count = con.execute("SELECT COUNT(*) FROM gold.normalized_returns").fetchone()[0]
    print(f"[gold] Built gold.normalized_returns — {row_count} rows")


def build_etf_premium(con: duckdb.DuckDBPyConnection) -> None:
    """
    Is the ETF rich or cheap vs. international gold parity? The raw ratio of ETF price to
    INR-converted futures is meaningless in absolute terms (ETF units track ~0.01g vs.
    futures per-ounce, and expense-ratio drag adds a slow secular drift) -- the signal is
    the ratio's deviation from its OWN 1-year rolling mean, which the z-score and relative
    premium capture while the rolling window absorbs the drift.
    """
    con.execute("DROP TABLE IF EXISTS gold.etf_premium")
    con.execute("""
        CREATE TABLE gold.etf_premium AS

        WITH fut AS (
            SELECT date, close_inr AS fut_inr
            FROM silver.prices WHERE instrument = 'gold_futures'
        ),

        etf AS (
            SELECT date, instrument, close_inr AS etf_inr
            FROM silver.prices
            WHERE instrument IN ('goldbees_etf', 'hdfc_gold_etf')
        ),

        ratio AS (
            SELECT e.date, e.instrument, e.etf_inr / f.fut_inr AS ratio
            FROM etf e
            JOIN fut f ON e.date = f.date
            WHERE f.fut_inr > 0
        ),

        stats AS (
            SELECT *,
                AVG(ratio)    OVER w AS ratio_avg_1y,
                STDDEV(ratio) OVER w AS ratio_std_1y
            FROM ratio
            WINDOW w AS (
                PARTITION BY instrument ORDER BY date
                ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
            )
        )

        SELECT
            date,
            instrument,
            ratio,
            ROUND((ratio / ratio_avg_1y - 1) * 100, 4) AS premium_vs_1y_avg_pct,
            CASE WHEN ratio_std_1y > 0
                 THEN ROUND((ratio - ratio_avg_1y) / ratio_std_1y, 4)
            END AS premium_zscore
        FROM stats
        ORDER BY instrument, date
    """)

    row_count = con.execute("SELECT COUNT(*) FROM gold.etf_premium").fetchone()[0]
    print(f"[gold] Built gold.etf_premium — {row_count} rows")


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    build_technical_features(con)
    build_normalized_returns(con)
    build_etf_premium(con)
    con.close()
    print("[gold] Done.")
