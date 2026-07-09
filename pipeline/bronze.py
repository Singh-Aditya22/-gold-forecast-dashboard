"""
Load raw CSV files into DuckDB bronze schema, exactly as collected.
No transformations — raw data preserved for auditability.
Run after collect.py.
"""

import os
import duckdb
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "gold_forecast.duckdb")

OHLCV_TABLES = {
    "gold_futures": "gold_futures",
    "usd_inr": "usd_inr",
    "goldbees_etf": "goldbees_etf",
    "hdfc_gold_etf": "hdfc_gold_etf",
    "vix": "vix",
    "crude_oil": "crude_oil",
    "usd_index": "usd_index",
    "us10y_yield": "us10y_yield",
}


def load_ohlcv(con: duckdb.DuckDBPyConnection, file_key: str, table_name: str) -> None:
    csv_path = os.path.join(RAW_DIR, f"{file_key}.csv")
    if not os.path.exists(csv_path):
        print(f"[bronze] SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date

    expected = ["date", "open", "high", "low", "close", "volume"]
    for col in expected:
        if col not in df.columns:
            df[col] = None

    df = df[expected]

    con.execute(f"DROP TABLE IF EXISTS bronze.{table_name}")
    con.execute(f"""
        CREATE TABLE bronze.{table_name} AS
        SELECT * FROM df
    """)
    print(f"[bronze] Loaded {len(df)} rows → bronze.{table_name}")


def load_sbi_gold_nav(con: duckdb.DuckDBPyConnection) -> None:
    csv_path = os.path.join(RAW_DIR, "sbi_gold_nav.csv")
    if not os.path.exists(csv_path):
        print(f"[bronze] SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df[["date", "nav"]]

    con.execute("DROP TABLE IF EXISTS bronze.sbi_gold_nav")
    con.execute("""
        CREATE TABLE bronze.sbi_gold_nav AS
        SELECT * FROM df
    """)
    print(f"[bronze] Loaded {len(df)} rows → bronze.sbi_gold_nav")


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")

    for file_key, table_name in OHLCV_TABLES.items():
        load_ohlcv(con, file_key, table_name)

    load_sbi_gold_nav(con)

    con.close()
    print("[bronze] Done.")
