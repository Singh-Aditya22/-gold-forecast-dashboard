"""
Streamlit multi-page Gold Forecasting Dashboard.
Launch with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard import queries, charts, insights

st.set_page_config(
    page_title="Gold Forecast Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Overview",
    "Individual Instrument",
    "Dip Tracker",
    "Forecast",
    "Model Comparison",
    "SGB Calculator",
]

OHLCV_INSTRUMENTS = queries.OHLCV_INSTRUMENTS
ALL_INSTRUMENTS = queries.ALL_INSTRUMENTS
INSTRUMENT_LABELS = queries.INSTRUMENT_LABELS


# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title("Gold Dashboard")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Yahoo Finance · mftool · DuckDB")


# ── Helper ───────────────────────────────────────────────────────────────────
# A single slider spanning the full ~25-year history (9000+ days) makes it nearly
# impossible to drag precisely to something like "last 3 months" -- one pixel of mouse
# movement covers dozens of days. Preset buttons for the ranges people actually want,
# with an exact-date fallback for anything else, is far more usable.
DATE_PRESETS = {
    "1M": 30, "3M": 90, "6M": 182, "1Y": 365, "5Y": 365 * 5, "All": None,
}


def date_range_picker(key: str, min_date: date, max_date: date, default: str = "1Y"):
    options = list(DATE_PRESETS.keys()) + ["Custom"]
    preset = st.segmented_control(
        "Date range", options=options, default=default, key=f"{key}_preset",
    )

    if preset == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start date", value=min_date, min_value=min_date,
                                  max_value=max_date, key=f"{key}_start")
        with col2:
            end = st.date_input("End date", value=max_date, min_value=min_date,
                                max_value=max_date, key=f"{key}_end")
        return start, end

    days = DATE_PRESETS.get(preset)
    start = min_date if days is None else max(min_date, max_date - timedelta(days=days))
    return start, max_date


# ── Page: Overview ───────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Overview — All Instruments")

    instruments = st.multiselect(
        "Select instruments to compare",
        options=ALL_INSTRUMENTS,
        default=OHLCV_INSTRUMENTS,
        format_func=lambda x: INSTRUMENT_LABELS[x],
    )

    global_min, global_max = date(2000, 1, 1), date.today()
    start, end = date_range_picker("overview_range", global_min, global_max, default="5Y")

    show_overview_forecast = st.checkbox(
        "Extend each line with its forecast (using each instrument's best-choice model)",
        value=False,
    )
    show_overview_labels = (
        st.checkbox("Show predicted values on the chart", value=False, key="overview_show_labels")
        if show_overview_forecast else False
    )

    if instruments:
        with st.spinner("Loading returns..."):
            ret_df = queries.get_normalized_returns(instruments, str(start), str(end))
            forecast_ret_df = (
                queries.get_forecast_normalized_returns(instruments) if show_overview_forecast else None
            )

        if not ret_df.empty:
            st.plotly_chart(
                charts.normalized_returns_chart(ret_df, forecast_ret_df, show_overview_labels),
                use_container_width=True,
            )

            trend_lines = []
            for inst in instruments:
                inst_df = ret_df[ret_df["instrument"] == inst]
                trend = insights.compute_trend(inst_df["date"], inst_df["return_from_inception_pct"])
                if trend:
                    arrow, trend_label, value, _ = insights.trend_badge(trend)
                    trend_lines.append(f"- {arrow} **{INSTRUMENT_LABELS[inst]}**: {trend_label} "
                                       f"({value} over this window)")
            macro_snapshot = queries.get_macro_snapshot()
            st.markdown(f"""
**What this shows:** each instrument's growth since the day you could first have bought it,
so a fair side-by-side comparison regardless of when each one started trading — the dotted
line is the straight-line trend over your selected window (the button above), separate from
the dashed forecast continuation (if turned on).

**Trend this window:**
{chr(10).join(trend_lines) if trend_lines else "Not enough data in this window."}

**External factors currently affecting gold:**
{insights.macro_commentary(macro_snapshot)}
            """)
        else:
            st.warning("No data for selected range.")

        st.subheader("Daily Return Correlation")
        with st.spinner("Computing correlation..."):
            corr = queries.get_correlation_matrix(str(start), str(end))
        if not corr.empty:
            st.plotly_chart(charts.correlation_heatmap(corr), use_container_width=True)
            st.caption(
                "**What this shows:** how much each instrument's day-to-day price moves match "
                "each other over your selected window. Close to 1.0 means they move almost "
                "together (little diversification benefit from holding both); close to 0 means "
                "they move independently; negative means they tend to move opposite each other."
            )

        st.subheader("Instrument Summary")
        summary = queries.get_all_instruments_summary()
        summary["instrument_label"] = summary["instrument"].map(INSTRUMENT_LABELS)
        st.dataframe(summary.drop(columns=["instrument"]).set_index("instrument_label"),
                     use_container_width=True)
    else:
        st.info("Select at least one instrument.")


# ── Page: Individual Instrument ──────────────────────────────────────────────
elif page == "Individual Instrument":
    st.title("Individual Instrument")

    instrument = st.selectbox(
        "Instrument",
        options=ALL_INSTRUMENTS,
        format_func=lambda x: INSTRUMENT_LABELS[x],
    )
    label = INSTRUMENT_LABELS[instrument]
    min_date, max_date = queries.get_date_range(instrument)
    start, end = date_range_picker("individual_range", min_date, max_date, default="1Y")

    show_mas = st.sidebar.checkbox("Show 50/200-day average price lines", value=True)
    show_bb = st.sidebar.checkbox("Show typical price range (Bollinger Bands)", value=True)

    is_nav_only = instrument == "sbi_gold_nav"
    forecast_series, display_name_overrides, show_labels = {}, {}, False
    if not is_nav_only:
        selected_model = queries.get_selected_model(instrument)
        chosen_models = st.multiselect(
            "Overlay forecast from",
            options=queries.ALL_MODEL_NAMES,
            default=[selected_model] if selected_model else [],
            format_func=lambda m: charts.MODEL_DISPLAY_NAMES.get(m, m)
                                  + ("  ⭐ best choice" if m == selected_model else ""),
            key="individual_forecast_models",
        )
        show_labels = st.checkbox("Show predicted values on the chart", value=False, key="individual_show_labels")
        with st.spinner("Loading forecast overlay..."):
            forecast_series = {
                m: queries.get_forecasts_future_only(instrument, model_name=m) for m in chosen_models
            }
        if selected_model in forecast_series:
            display_name_overrides[selected_model] = (
                f"{charts.MODEL_DISPLAY_NAMES.get(selected_model, selected_model)} (best choice)"
            )

    with st.spinner(f"Loading {label}..."):
        price_df = queries.get_prices(instrument, str(start), str(end))
        tech_df = queries.get_technical_features(instrument, str(start), str(end))

    if price_df.empty:
        st.warning("No price data for selected range.")
    else:
        if is_nav_only:
            st.plotly_chart(charts.line_chart(price_df, label, tech_df), use_container_width=True)
        else:
            st.plotly_chart(
                charts.candlestick_chart(price_df, label, show_mas, show_bb, tech_df,
                                        forecast_series, display_name_overrides, show_labels),
                use_container_width=True,
            )

        trend = insights.compute_trend(price_df["date"], price_df["close"])
        arrow, trend_label, trend_value, trend_delta = insights.trend_badge(trend)
        last_rsi = tech_df["rsi_14"].iloc[-1] if not tech_df.empty and "rsi_14" in tech_df.columns else None
        last_drawdown = (tech_df["drawdown_pct"].iloc[-1]
                        if not tech_df.empty and "drawdown_pct" in tech_df.columns else None)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(f"{arrow} {trend_label}", trend_value, trend_delta)
        with col2:
            st.markdown(f"**What's happening:** {insights.buy_sell_read(trend, last_rsi, last_drawdown)}")

        st.markdown(
            f"**What the price chart shows:** the candlesticks are each trading day's open/high/low/close "
            f"— the dashed lines are 50/200-day moving averages (smoothed trend), the shaded band is the "
            f"typical recent price range (Bollinger Bands), and the dash-dot line is the straight-line "
            f"trend over your selected window. The panel below shows momentum (RSI) with cheap/expensive "
            f"zones shaded."
        )
        if not is_nav_only:
            macro_snapshot = queries.get_macro_snapshot()
            st.markdown(f"**External factors currently affecting this instrument's model:**\n\n"
                       f"{insights.macro_commentary(macro_snapshot)}")

        if not tech_df.empty:
            st.plotly_chart(charts.volatility_chart(tech_df, label), use_container_width=True)
            # rolling_vol_30d is a std dev of the raw price LEVEL (INR), not of returns --
            # normalize by price so this reads as a genuine "% of price" swing regardless
            # of the instrument's absolute price level (matches what the chart itself plots).
            vol_pct = (tech_df["rolling_vol_30d"] / tech_df["close_inr"]) * 100
            vol_trend = insights.compute_trend(tech_df["date"], vol_pct)
            if vol_trend:
                vol_arrow, vol_label, _, _ = insights.trend_badge(vol_trend)
                st.caption(
                    f"**What this shows:** how much the daily price has been swinging (30-day rolling "
                    f"volatility, as a % of price) — {vol_arrow} **{vol_label.lower()}** over your "
                    f"selected window ({vol_trend['pct_change']:+.1f}%). Rising volatility often means "
                    f"bigger moves in either direction are becoming more likely, not a directional signal "
                    f"by itself."
                )

        if not is_nav_only:
            with st.expander("📋 Live Track Record — real predictions vs. what actually happened"):
                live_df = queries.get_live_predictions(instrument)
                if live_df.empty:
                    st.info("No live predictions logged yet — this builds up one row per day going forward.")
                else:
                    if chosen_models:
                        live_df = live_df[live_df["model_name"].isin(chosen_models)]
                    display_live = live_df[["predicted_for", "model_name", "predicted_price",
                                            "actual_price", "pct_error"]].copy()
                    display_live.columns = ["Trading Day", "Model", "Predicted (INR)", "Actual (INR)", "Error (%)"]
                    st.dataframe(display_live, use_container_width=True)

        with st.expander("Raw data"):
            st.dataframe(price_df, use_container_width=True)


# ── Page: Dip Tracker ────────────────────────────────────────────────────────
elif page == "Dip Tracker":
    st.title("Dip Tracker")
    st.caption(
        "Green dots = past moments this was noticeably cheaper than its recent trend "
        "(more than 5% below its 200-day average price) — historically a good time to buy. "
        "Orange diamonds = a possible future dip, projected from whichever forecast "
        "model(s) you pick below."
    )

    instrument = st.selectbox(
        "Instrument",
        options=OHLCV_INSTRUMENTS,
        format_func=lambda x: INSTRUMENT_LABELS[x],
        key="dip_instrument",
    )
    label = INSTRUMENT_LABELS[instrument]
    min_date, max_date = queries.get_date_range(instrument)
    start, end = date_range_picker("dip_range", min_date, max_date, default="1Y")

    selected_model = queries.get_selected_model(instrument)
    chosen_models = st.multiselect(
        "Project future dips using",
        options=queries.ALL_MODEL_NAMES,
        default=[selected_model] if selected_model else [],
        format_func=lambda m: charts.MODEL_DISPLAY_NAMES.get(m, m)
                              + ("  ⭐ best choice" if m == selected_model else ""),
        key="dip_forecast_models",
    )
    show_dip_labels = st.checkbox("Show predicted values on the chart", value=False, key="dip_show_labels")

    with st.spinner("Loading dip data..."):
        dip_df = queries.get_dip_tracker(instrument, str(start), str(end))
        forecast_series = {
            m: queries.get_forecasts_future_only(instrument, model_name=m) for m in chosen_models
        }
        last_ma_200 = queries.get_last_ma_200(instrument)

    display_name_overrides = {}
    if selected_model in forecast_series:
        display_name_overrides[selected_model] = (
            f"{charts.MODEL_DISPLAY_NAMES.get(selected_model, selected_model)} (best choice)"
        )

    if dip_df.empty:
        st.warning("No data for selected range.")
    else:
        st.plotly_chart(
            charts.drawdown_chart(dip_df, label, forecast_series, display_name_overrides,
                                 show_dip_labels, last_ma_200),
            use_container_width=True,
        )

        hist_dip_count = dip_df["is_dip_historical"].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Historical Dip Days", int(hist_dip_count))
        col2.metric("Max Drawdown", f"{dip_df['drawdown_pct'].min():.2f}%")
        col3.metric("Current Drawdown", f"{dip_df['drawdown_pct'].iloc[-1]:.2f}%"
                    if not dip_df.empty else "N/A")

        dip_trend = insights.compute_trend(dip_df["date"], dip_df["close_inr"])
        last_drawdown = dip_df["drawdown_pct"].iloc[-1] if not dip_df.empty else None
        st.markdown(
            f"**What's happening:** {insights.buy_sell_read(dip_trend, None, last_drawdown)} "
            f"The dash-dot line on the price chart is the straight-line trend over your selected window."
        )
        macro_snapshot = queries.get_macro_snapshot()
        st.markdown(f"**External factors currently affecting this instrument's model:**\n\n"
                   f"{insights.macro_commentary(macro_snapshot)}")

        with st.expander("📋 Live Track Record — real predictions vs. what actually happened"):
            live_df = queries.get_live_predictions(instrument)
            if live_df.empty:
                st.info("No live predictions logged yet — this builds up one row per day going forward.")
            else:
                if chosen_models:
                    live_df = live_df[live_df["model_name"].isin(chosen_models)]
                display_live = live_df[["predicted_for", "model_name", "predicted_price",
                                        "actual_price", "pct_error"]].copy()
                display_live.columns = ["Trading Day", "Model", "Predicted (INR)", "Actual (INR)", "Error (%)"]
                st.dataframe(display_live, use_container_width=True)


# ── Page: Forecast ───────────────────────────────────────────────────────────
elif page == "Forecast":
    st.title("90-Day Price Forecast")

    instrument = st.selectbox(
        "Instrument",
        options=OHLCV_INSTRUMENTS,
        format_func=lambda x: INSTRUMENT_LABELS[x],
        key="forecast_instrument",
    )
    label = INSTRUMENT_LABELS[instrument]
    HORIZON_PRESETS = {"1W": 7, "2W": 14, "1M": 30, "2M": 60, "3M": 90}
    horizon_label = st.segmented_control(
        "Forecast horizon", options=list(HORIZON_PRESETS.keys()) + ["Custom"],
        default="1M", key="forecast_horizon",
    )
    if horizon_label == "Custom":
        # 90 is a hard ceiling, not just a UI choice -- predict.py only ever generates
        # 90 days of future forecast per model, so there's no data beyond that to show.
        horizon = st.number_input("Custom horizon (days)", min_value=1, max_value=90,
                                  value=30, step=1, key="forecast_horizon_custom")
    else:
        horizon = HORIZON_PRESETS.get(horizon_label, 30)
    show_candles = st.checkbox(
        "Show daily trading range (candlesticks) instead of a plain price line", value=False,
    )
    show_forecast_labels = st.checkbox("Show predicted values on the chart", value=False, key="forecast_show_labels")
    selected_model = queries.get_selected_model(instrument)

    chosen_models = st.multiselect(
        "Models to show",
        options=queries.ALL_MODEL_NAMES,
        default=list(dict.fromkeys(["naive", "lstm", selected_model])),  # dedupe, keep order
        format_func=lambda m: charts.MODEL_DISPLAY_NAMES.get(m, m)
                              + ("  ⭐ best choice" if m == selected_model else ""),
        key="forecast_models",
    )

    st.caption(
        "**If the price simply stays the same** (naive) assumes tomorrow's price stays "
        "the same as today's — surprisingly, that simple guess beats every fancier model "
        "we tried on this data (a well-known result for daily asset prices), which is why "
        "it's the benchmark the ⭐ best-choice model had to beat. Add other models above "
        "to compare their trend estimates side by side."
    )

    with st.spinner("Loading forecast..."):
        hist_df = queries.get_prices(instrument, "2000-01-01", str(date.today()))
        series = {
            m: queries.get_forecasts_future_only(instrument, model_name=m).head(horizon)
            for m in chosen_models
        }

    display_name_overrides = {}
    if selected_model in series:
        display_name_overrides[selected_model] = (
            f"{charts.MODEL_DISPLAY_NAMES.get(selected_model, selected_model)} (best choice)"
        )

    if not chosen_models or all(df.empty for df in series.values()):
        st.warning("No forecast data available. Run models/predict.py first, or pick a model above.")
    else:
        chart_fn = charts.forecast_candlestick_chart if show_candles else charts.forecast_chart
        st.plotly_chart(
            chart_fn(hist_df.tail(365), series, label, display_name_overrides, show_forecast_labels),
            use_container_width=True,
        )

        st.subheader(f"Next {horizon} Days — Predicted Prices")
        tab_labels = [display_name_overrides.get(m, charts.MODEL_DISPLAY_NAMES.get(m, m)) for m in chosen_models]
        tabs = st.tabs(tab_labels)
        for tab, m in zip(tabs, chosen_models):
            with tab:
                df = series[m]
                if df.empty:
                    st.info("No data.")
                else:
                    display_df = df[["date", "yhat", "yhat_lower", "yhat_upper"]].copy()
                    display_df.columns = ["Date", "Predicted (INR)", "Lower Bound", "Upper Bound"]
                    st.dataframe(display_df, use_container_width=True)

        hist_recent = hist_df.tail(365)
        fc_trend = insights.compute_trend(hist_recent["date"], hist_recent["close"])
        arrow, trend_label, trend_value, trend_delta = insights.trend_badge(fc_trend)
        st.markdown(
            f"**What the chart shows:** the yellow line is the actual price so far; the dash-dot "
            f"line is the realized trend over the last year (separate from any model's forward "
            f"projection); each model's forecast line/band extends from there. {arrow} Recently, "
            f"the price has been **{trend_label.lower()}** ({trend_value} over the last year, "
            f"~{trend_delta})."
        )
        macro_snapshot = queries.get_macro_snapshot()
        st.markdown(f"**External factors these models use as inputs:**\n\n"
                   f"{insights.macro_commentary(macro_snapshot)}")

    st.markdown("---")
    with st.expander("📋 Live Track Record — real predictions vs. what actually happened"):
        st.caption(
            f"Every day, the **{selected_model}** model (currently the statistically best "
            f"choice for {label}) predicts the very next trading day's price, and that "
            f"prediction is logged here before the outcome is known. Once the actual price "
            f"is available, it's filled in below — this is a genuine forward test, not a "
            f"backtest replaying already-known history. Rows with a blank 'Actual' are "
            f"awaiting tomorrow's data."
        )
        live_df = queries.get_live_predictions(instrument)
        if live_df.empty:
            st.info("No live predictions logged yet — this builds up one row per day going forward.")
        else:
            if chosen_models:
                live_df = live_df[live_df["model_name"].isin(chosen_models)]
            display_live = live_df[["predicted_for", "model_name", "predicted_price",
                                    "actual_price", "pct_error"]].copy()
            display_live.columns = ["Trading Day", "Model", "Predicted (INR)", "Actual (INR)", "Error (%)"]
            st.dataframe(display_live, use_container_width=True)


# ── Page: Model Comparison ───────────────────────────────────────────────────
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption(
        "Tested on a full year (Jan–Dec 2025) the models never saw during training. "
        "The bar chart below: shorter bars = smaller average error = better."
    )
    st.info(
        "**\"Skill vs Naive\"** compares each model to the simplest possible guess — "
        "\"tomorrow's price = today's price.\" Positive means a model actually beats that "
        "simple guess; negative (most rows below) means it does worse. This sounds like a "
        "low bar, but for daily prices of investments like gold, beating it is genuinely "
        "hard — which is exactly why we test against it instead of just picking whichever "
        "model has the fanciest name.",
    )

    with st.spinner("Loading model scores..."):
        scores_df = queries.get_model_scores()

    if scores_df.empty:
        st.warning("No model scores found. Run models/evaluate.py first.")
    else:
        st.plotly_chart(charts.model_scores_chart(scores_df), use_container_width=True)

        st.subheader("Full Scores Table")
        display_df = scores_df[["instrument_label", "model_name", "rmse", "mae", "mape",
                                 "skill_score_vs_naive", "directional_accuracy", "selected"]].copy()
        display_df.columns = ["Instrument", "Model", "RMSE", "MAE", "Avg. Error (%)",
                               "Skill vs Naive", "Correct Direction (%)", "Best Choice"]
        st.dataframe(
            display_df.style.applymap(
                lambda v: "background-color: #2d5a27" if v is True else "",
                subset=["Best Choice"],
            ),
            use_container_width=True,
        )


# ── Page: SGB Calculator ─────────────────────────────────────────────────────
elif page == "SGB Calculator":
    st.title("SGB Return Calculator")
    st.caption(
        "Sovereign Gold Bonds earn gold price appreciation + 2.5% p.a. fixed interest (semi-annual). "
        "Compare vs. Gold ETF over the same period."
    )

    col1, col2 = st.columns(2)
    with col1:
        investment = st.number_input("Investment amount (₹)", min_value=1000, value=100000, step=5000)
    with col2:
        invest_date = st.date_input(
            "Investment start date",
            value=date(2020, 1, 1),
            min_value=date(2015, 1, 1),
            max_value=date.today() - timedelta(days=30),
        )

    end_date = st.date_input(
        "End / Redemption date",
        value=date.today(),
        min_value=invest_date + timedelta(days=1),
        max_value=date.today(),
    )

    with st.spinner("Fetching ETF prices..."):
        etf_df = queries.get_prices("goldbees_etf", str(invest_date), str(end_date))

    if etf_df.empty:
        st.warning("No ETF data for selected range.")
    else:
        start_price = etf_df["close"].iloc[0]
        end_price = etf_df["close"].iloc[-1]
        years = (end_date - invest_date).days / 365.25

        # ETF return
        etf_return_pct = ((end_price - start_price) / start_price) * 100
        etf_value = investment * (end_price / start_price)

        # SGB return (same gold price change + 2.5% p.a. simple interest)
        sgb_interest = investment * 0.025 * years
        sgb_value = etf_value + sgb_interest
        sgb_return_pct = ((sgb_value - investment) / investment) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Holding Period", f"{years:.1f} years")
        col2.metric("Gold Price Return", f"{etf_return_pct:.2f}%")
        col3.metric("ETF Value (₹)", f"₹{etf_value:,.0f}")
        col4.metric("SGB Value (₹)", f"₹{sgb_value:,.0f}",
                    delta=f"+₹{sgb_interest:,.0f} interest bonus")

        st.markdown("---")
        st.markdown(f"""
        | | Gold ETF | Sovereign Gold Bond |
        |---|---|---|
        | **Investment** | ₹{investment:,} | ₹{investment:,} |
        | **End Value** | ₹{etf_value:,.0f} | ₹{sgb_value:,.0f} |
        | **Total Return** | {etf_return_pct:.2f}% | {sgb_return_pct:.2f}% |
        | **Interest Earned** | — | ₹{sgb_interest:,.0f} |
        | **Maturity Tax** | 12.5% LTCG | Tax-free (if held to 8yr maturity) |
        """)

        st.info(
            "Note: SGB interest bonus becomes fully tax-free only if you hold to the 8-year maturity "
            "as an original subscriber (primary issue). Secondary market purchases are taxed at 12.5%."
        )
