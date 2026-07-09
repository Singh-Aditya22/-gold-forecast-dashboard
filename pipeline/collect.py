"""
Fetch raw data from yfinance and mftool and save to data/raw/ as CSV files.
Run this first in the pipeline before bronze.py.
"""

import os
import pandas as pd
import yfinance as yf
from mftool import Mftool
from datetime import date

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
START_DATE = "2000-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

YFINANCE_TICKERS = {
    "gold_futures": "GC=F",
    "usd_inr": "USDINR=X",
    "goldbees_etf": "GOLDBEES.NS",
    "hdfc_gold_etf": "HDFCGOLD.NS",
    # Macro/geopolitical-risk proxies (exogenous features) — gold is a well-documented
    # hedge against risk-off events (wars, shocks): VIX = fear gauge, crude oil often
    # co-moves with geopolitical tension, USD index has an inverse correlation with gold,
    # and the 10Y yield drives real interest rates (a key gold price driver).
    "vix": "^VIX",
    "crude_oil": "CL=F",
    "usd_index": "DX-Y.NYB",
    "us10y_yield": "^TNX",
}

SBI_GOLD_FUND_CODE = "119598"


def fetch_yfinance(name: str, ticker: str) -> None:
    print(f"[collect] Fetching {name} ({ticker}) from {START_DATE} to {END_DATE}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if df.empty:
        print(f"[collect] WARNING: No data returned for {ticker}")
        return
    df = df.reset_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.rename(columns={"price": "close"}) if "price" in df.columns else df
    out_path = os.path.join(RAW_DIR, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"[collect] Saved {len(df)} rows → {out_path}")


def fetch_sbi_gold_fund() -> None:
    print(f"[collect] Fetching SBI Gold Fund NAV (code: {SBI_GOLD_FUND_CODE})...")
    mf = Mftool()
    data = mf.get_scheme_historical_nav(SBI_GOLD_FUND_CODE, as_Dataframe=True)
    if data is None or data.empty:
        print("[collect] WARNING: No NAV data returned for SBI Gold Fund")
        return
    data = data.reset_index()
    data.columns = [c.lower() for c in data.columns]
    data = data.rename(columns={"nav": "nav", "date": "date"})
    data["date"] = pd.to_datetime(data["date"], dayfirst=True)
    data["nav"] = pd.to_numeric(data["nav"], errors="coerce")
    data = data.dropna(subset=["nav"])
    data = data.sort_values("date")
    out_path = os.path.join(RAW_DIR, "sbi_gold_nav.csv")
    data.to_csv(out_path, index=False)
    print(f"[collect] Saved {len(data)} rows → {out_path}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, ticker in YFINANCE_TICKERS.items():
        fetch_yfinance(name, ticker)
    fetch_sbi_gold_fund()
    print("[collect] Done.")
