"""
Plain-language interpretation helpers: trend-line math, and the explanatory text shown
below each chart (what it shows, what current conditions suggest for buying/selling, and
how macro/geopolitical factors are reading right now). These are simple rule-based
comparisons against historical averages -- not another predictive model, and not the same
thing as the ML forecasts elsewhere in the dashboard.
"""

import numpy as np
import pandas as pd


def compute_trend(dates, values) -> dict:
    """
    Ordinary least-squares linear trend over whatever window is currently selected --
    summarizes the *realized* direction of the selected period in one line, separate from
    any model's forecast. Returns None if there's not enough data to fit a line.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    valid = values.notna()
    dates, values = dates[valid], values[valid]
    if len(dates) < 2:
        return None

    x = (dates - dates.min()).dt.days.values.astype(float)
    y = values.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept

    start_val = fitted[0]
    pct_change = ((fitted[-1] - start_val) / start_val) * 100 if start_val else 0.0
    days_span = max(x[-1] - x[0], 1)
    annualized_pct = pct_change * (365.25 / days_span)
    direction = "up" if pct_change > 1 else "down" if pct_change < -1 else "flat"

    return {
        "dates": dates, "fitted": fitted,
        "pct_change": pct_change, "annualized_pct": annualized_pct, "direction": direction,
    }


def trend_badge(trend: dict) -> tuple:
    """
    (arrow, label, value, delta) -- value/delta are separate short strings meant for
    st.metric's value= and delta= slots (a single long combined string gets truncated by
    st.metric's compact display). Callers building a sentence instead of a metric can just
    join value and delta with " over this window (" + delta + ")".
    """
    if trend is None:
        return "➡️", "Not enough data", "", ""
    arrow = {"up": "📈", "down": "📉", "flat": "➡️"}[trend["direction"]]
    label = {"up": "Uptrend", "down": "Downtrend", "flat": "Roughly flat"}[trend["direction"]]
    value = f"{trend['pct_change']:+.1f}%"
    delta = f"{trend['annualized_pct']:+.1f}%/yr annualized"
    return arrow, label, value, delta


def buy_sell_read(trend: dict, rsi: float = None, drawdown_pct: float = None) -> str:
    """
    A short, rule-based "what does this look like right now" line combining trend
    direction, momentum (RSI), and how far below its recent peak the price is. This is an
    interpretation of current conditions, not a recommendation or a prediction.
    """
    parts = []
    if trend is not None:
        if trend["direction"] == "up":
            parts.append(f"trending upward (+{trend['pct_change']:.1f}% over this window)")
        elif trend["direction"] == "down":
            parts.append(f"trending downward ({trend['pct_change']:.1f}% over this window)")
        else:
            parts.append("roughly flat over this window")

    read = "Currently " + " and ".join(parts) if parts else "Not enough data in this window to read a trend."

    signals = []
    if rsi is not None:
        if rsi < 30:
            signals.append(f"momentum is in the **cheap zone** (RSI {rsi:.0f}) — historically where dips have been worth watching for buying")
        elif rsi > 70:
            signals.append(f"momentum is in the **expensive zone** (RSI {rsi:.0f}) — historically where prices have been more stretched, worth watching before buying more")
        else:
            signals.append(f"momentum is in a **neutral range** (RSI {rsi:.0f}) — no strong cheap/expensive signal either way")
    if drawdown_pct is not None and drawdown_pct <= -5:
        signals.append(f"price is **{abs(drawdown_pct):.1f}% below its recent 200-day peak** — the kind of dip that has historically been a better entry point than a stretched high")

    if signals:
        read += "; " + ", and ".join(signals) + "."
    else:
        read += "."
    return read


def get_macro_snapshot(con) -> dict:
    """Latest macro reading + its own 1-year average, so callers can describe current
    conditions as elevated/subdued/near-average without another query round trip.
    USD/INR is queried with a fallback: an older committed gold_forecast.duckdb won't
    have the usdinr_close column yet, and the dashboard must not crash on it."""
    try:
        row = con.execute("""
            SELECT date, vix_close, oil_close, usd_index_close, us10y_yield_close, usdinr_close
            FROM silver.macro_features ORDER BY date DESC LIMIT 1
        """).fetchone()
        has_inr = True
    except Exception:
        row = con.execute("""
            SELECT date, vix_close, oil_close, usd_index_close, us10y_yield_close
            FROM silver.macro_features ORDER BY date DESC LIMIT 1
        """).fetchone()
        has_inr = False
    if row is None:
        return {}
    latest_date = row[0]
    inr_cols = ", AVG(usdinr_close)" if has_inr else ""
    avgs = con.execute(f"""
        SELECT AVG(vix_close), AVG(oil_close), AVG(usd_index_close), AVG(us10y_yield_close){inr_cols}
        FROM silver.macro_features
        WHERE date >= DATE '{latest_date}' - INTERVAL 365 DAY
    """).fetchone()
    return {
        "date": row[0], "vix": row[1], "oil": row[2], "usd_index": row[3], "us10y_yield": row[4],
        "usdinr": row[5] if has_inr else None,
        "vix_avg": avgs[0], "oil_avg": avgs[1], "usd_index_avg": avgs[2], "us10y_yield_avg": avgs[3],
        "usdinr_avg": avgs[4] if has_inr else None,
    }


def macro_commentary(snapshot: dict) -> str:
    """
    Rule-based read of the 4 macro/geopolitical-risk proxies these models actually use as
    features (see [[project_gold_forecast]] in memory): compares the latest reading to its
    own 1-year average and explains the historically-documented direction of its effect on
    gold, rather than treating it as a hidden black-box input.
    """
    if not snapshot or snapshot.get("vix") is None:
        return "Macro data not available."

    lines = []
    vix, vix_avg = snapshot["vix"], snapshot["vix_avg"]
    if vix_avg:
        if vix > vix_avg * 1.15:
            lines.append(f"- **Market fear (VIX)** is elevated at {vix:.1f} vs. its 1-year average of "
                          f"{vix_avg:.1f} — higher fear has historically coincided with stronger gold "
                          f"demand as a safe-haven asset.")
        elif vix < vix_avg * 0.85:
            lines.append(f"- **Market fear (VIX)** is subdued at {vix:.1f} vs. its 1-year average of "
                          f"{vix_avg:.1f} — calmer markets typically reduce gold's safe-haven appeal.")

    usd, usd_avg = snapshot["usd_index"], snapshot["usd_index_avg"]
    if usd_avg:
        if usd < usd_avg * 0.98:
            lines.append(f"- **US Dollar Index** is weaker than its 1-year average ({usd:.1f} vs "
                          f"{usd_avg:.1f}) — a weaker dollar tends to support gold, since gold is "
                          f"priced in dollars.")
        elif usd > usd_avg * 1.02:
            lines.append(f"- **US Dollar Index** is stronger than its 1-year average ({usd:.1f} vs "
                          f"{usd_avg:.1f}) — a stronger dollar tends to pressure gold prices.")

    y10, y10_avg = snapshot["us10y_yield"], snapshot["us10y_yield_avg"]
    if y10_avg:
        if y10 > y10_avg * 1.05:
            lines.append(f"- **10-Year Treasury yield** is above its 1-year average ({y10:.2f}% vs "
                          f"{y10_avg:.2f}%) — higher real yields raise the opportunity cost of holding "
                          f"non-yielding gold, historically a headwind.")
        elif y10 < y10_avg * 0.95:
            lines.append(f"- **10-Year Treasury yield** is below its 1-year average ({y10:.2f}% vs "
                          f"{y10_avg:.2f}%) — lower yields reduce that opportunity cost, historically a "
                          f"tailwind for gold.")

    oil, oil_avg = snapshot["oil"], snapshot["oil_avg"]
    if oil_avg and oil > oil_avg * 1.1:
        lines.append(f"- **Crude oil** is running well above its 1-year average (${oil:.1f} vs "
                      f"${oil_avg:.1f}) — often a sign of geopolitical tension or inflation pressure, "
                      f"both historically supportive of gold.")

    inr, inr_avg = snapshot.get("usdinr"), snapshot.get("usdinr_avg")
    if inr and inr_avg:
        if inr > inr_avg * 1.02:
            lines.append(f"- **Rupee (USD/INR)** is weaker than its 1-year average (₹{inr:.1f} vs "
                          f"₹{inr_avg:.1f} per dollar) — a weaker rupee directly raises the INR price "
                          f"of gold, supporting domestic gold ETFs and funds even when the dollar gold "
                          f"price is flat.")
        elif inr < inr_avg * 0.98:
            lines.append(f"- **Rupee (USD/INR)** is stronger than its 1-year average (₹{inr:.1f} vs "
                          f"₹{inr_avg:.1f} per dollar) — a stronger rupee directly lowers the INR price "
                          f"of gold, a headwind for domestic gold ETFs and funds independent of the "
                          f"dollar gold price.")

    if not lines:
        return ("Macro conditions (market fear, USD, yields, oil) are all broadly near their recent "
                "averages right now — no strong signal either way from these factors.")
    return "\n".join(lines)


def forecast_consensus(future_df: pd.DataFrame, last_close: float, horizon_days: int,
                       flat_band_pct: float = 0.5) -> dict:
    """
    "What do the models agree on?" -- classify every model's forecast at the given horizon
    as up/down/flat relative to the last actual close. The flat band matters: naive always
    predicts ~0% change by construction, so without it the baseline would randomly pollute
    the up/down counts on floating-point noise.
    """
    if future_df is None or future_df.empty or not last_close:
        return None

    per_model = []
    for model_name, mdf in future_df.groupby("model_name"):
        mdf = mdf.sort_values("date").head(horizon_days)
        if mdf.empty:
            continue
        yhat = float(mdf["yhat"].iloc[-1])
        pct = (yhat - last_close) / last_close * 100
        direction = "up" if pct > flat_band_pct else "down" if pct < -flat_band_pct else "flat"
        per_model.append({"model_name": model_name, "yhat": yhat,
                          "pct_change": pct, "direction": direction})
    if not per_model:
        return None

    n = len(per_model)
    n_up = sum(r["direction"] == "up" for r in per_model)
    n_down = sum(r["direction"] == "down" for r in per_model)
    n_flat = n - n_up - n_down
    top_share = max(n_up, n_down, n_flat) / n
    agreement = "strong" if top_share >= 0.7 else "lean" if top_share > 0.5 else "split"

    return {
        "n_models": n, "n_up": n_up, "n_down": n_down, "n_flat": n_flat,
        "median_pct_change": float(np.median([r["pct_change"] for r in per_model])),
        "per_model": per_model, "agreement": agreement,
    }


def consensus_text(consensus: dict, horizon_label: str) -> str:
    """Plain-language summary of the consensus dict, with the two honesty caveats that
    stop "X of 7 models" from overstating independence."""
    n, up, down = consensus["n_models"], consensus["n_up"], consensus["n_down"]
    flat = consensus["n_flat"]
    med = consensus["median_pct_change"]

    if up > down and up > flat:
        lead = f"**{up} of {n} models** expect the price to be *higher* in {horizon_label} than it is today"
    elif down > up and down > flat:
        lead = f"**{down} of {n} models** expect the price to be *lower* in {horizon_label} than it is today"
    else:
        lead = f"The models are **split** on where the price will be in {horizon_label}"

    return (
        f"{lead} (median call across all models: **{med:+.1f}%**). Two of the votes aren't "
        f"independent: the naive baseline always says \"no change\" by construction, and the "
        f"ensemble is just the average of the other six — so read this as context, not a poll "
        f"of seven independent experts."
    )


DIP_BACKTEST_HORIZONS = [
    ("fwd_30d", "1 month"), ("fwd_90d", "3 months"),
    ("fwd_180d", "6 months"), ("fwd_365d", "1 year"),
]


def dip_backtest_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median + quartile forward returns from dip days vs. all other days, per horizon.
    Medians (not means) on purpose -- a couple of crisis rebounds shouldn't carry the
    whole verdict.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for col, label in DIP_BACKTEST_HORIZONS:
        sub = df[df[col].notna()]
        dip = sub[sub["is_dip_historical"] == True][col]
        other = sub[sub["is_dip_historical"] != True][col]
        if dip.empty or other.empty:
            continue
        rows.append({
            "horizon": label, "dip_days": int(dip.count()),
            "dip_median": dip.median(), "dip_q25": dip.quantile(0.25), "dip_q75": dip.quantile(0.75),
            "other_median": other.median(), "other_q25": other.quantile(0.25), "other_q75": other.quantile(0.75),
            "median_advantage": dip.median() - other.median(),
        })
    return pd.DataFrame(rows)


def dip_backtest_verdict(summary: pd.DataFrame, instrument_label: str) -> str:
    """One-paragraph plain-language verdict on the dip rule, with the statistical caveat:
    dip days cluster into episodes, so these are overlapping, non-independent samples --
    evidence about the past, not proof about the future."""
    if summary is None or summary.empty:
        return "Not enough dip history to judge for this instrument."
    wins = summary[summary["median_advantage"] > 0]
    n_h = len(summary)
    if len(wins) == n_h:
        lead = (f"For {instrument_label}, buying on a dip day beat buying on an ordinary day "
                f"on **all {n_h} horizons** tested")
    elif len(wins) == 0:
        lead = (f"For {instrument_label}, buying on a dip day did **not** beat buying on an "
                f"ordinary day on any horizon tested")
    else:
        lead = (f"For {instrument_label}, buying on a dip day beat buying on an ordinary day "
                f"on **{len(wins)} of {n_h} horizons** tested")
    adv_lo, adv_hi = summary["median_advantage"].min(), summary["median_advantage"].max()
    return (
        f"{lead} (median advantage ranged from {adv_lo:+.1f} to {adv_hi:+.1f} percentage points). "
        f"One caveat: dip days cluster into episodes (a single long dip contributes many "
        f"overlapping samples), so treat this as historical evidence, not statistical proof."
    )


def etf_premium_read(latest_zscore: float, latest_premium_pct: float) -> str:
    """One-line plain-language read of the ETF's current premium/discount vs.
    international gold parity."""
    if latest_zscore is None or pd.isna(latest_zscore):
        return "Not enough overlapping history to read the ETF's premium right now."
    if latest_zscore >= 2:
        verdict = ("**noticeably rich** vs. international gold — buying today pays an unusual "
                   "premium over parity; historically this has tended to normalize back down")
    elif latest_zscore >= 1:
        verdict = "**slightly rich** vs. international gold — a mild premium over its usual level"
    elif latest_zscore <= -2:
        verdict = ("**noticeably cheap** vs. international gold — an unusual discount to parity; "
                   "historically this has tended to normalize back up")
    elif latest_zscore <= -1:
        verdict = "**slightly cheap** vs. international gold — a mild discount to its usual level"
    else:
        verdict = "**near its usual level** vs. international gold — no meaningful premium or discount"
    return (
        f"Right now this ETF is {verdict} (z = {latest_zscore:+.1f}, "
        f"{latest_premium_pct:+.2f}% vs. its 1-year average ratio). This reflects tracking "
        f"error, local demand, and expense drag — context for entry timing, not an arbitrage "
        f"signal."
    )


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def seasonality_text(monthly_df: pd.DataFrame) -> str:
    """Best/worst average calendar months + the India demand-season context, with the
    small-sample caveat that stops a heatmap cell from being read as a trading rule."""
    if monthly_df is None or monthly_df.empty:
        return "Not enough history to read a seasonal pattern."
    means = monthly_df.groupby("month")["monthly_return_pct"].mean()
    best, worst = int(means.idxmax()), int(means.idxmin())
    return (
        f"On average, **{MONTH_NAMES[best - 1]}** has been the strongest month "
        f"({means[best]:+.1f}% avg) and **{MONTH_NAMES[worst - 1]}** the weakest "
        f"({means[worst]:+.1f}% avg). Indian gold demand has a real seasonal rhythm — "
        f"festival and wedding-season buying runs roughly October–December, plus Akshaya "
        f"Tritiya in April/May — but each cell above is one month of one year (a small "
        f"sample), so treat this as context, not a trading rule."
    )
