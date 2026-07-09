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
    conditions as elevated/subdued/near-average without another query round trip."""
    row = con.execute("""
        SELECT date, vix_close, oil_close, usd_index_close, us10y_yield_close
        FROM silver.macro_features ORDER BY date DESC LIMIT 1
    """).fetchone()
    if row is None:
        return {}
    latest_date = row[0]
    avgs = con.execute(f"""
        SELECT AVG(vix_close), AVG(oil_close), AVG(usd_index_close), AVG(us10y_yield_close)
        FROM silver.macro_features
        WHERE date >= DATE '{latest_date}' - INTERVAL 365 DAY
    """).fetchone()
    return {
        "date": row[0], "vix": row[1], "oil": row[2], "usd_index": row[3], "us10y_yield": row[4],
        "vix_avg": avgs[0], "oil_avg": avgs[1], "usd_index_avg": avgs[2], "us10y_yield_avg": avgs[3],
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

    if not lines:
        return ("Macro conditions (market fear, USD, yields, oil) are all broadly near their recent "
                "averages right now — no strong signal either way from these factors.")
    return "\n".join(lines)
