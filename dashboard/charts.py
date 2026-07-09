"""
Plotly chart builders for the Streamlit dashboard.
Each function takes a DataFrame (from queries.py) and returns a plotly Figure.

Colors follow a validated categorical/status palette (dark-mode steps, since the
dashboard uses template="plotly_dark"): trend lines use fixed categorical hues
(never cycled), "good"/"caution" zones use fixed status colors (never reused for
a plain series), and instrument identity colors stay consistent across charts.
Titles, zone shading, and annotations favor plain language over financial jargon
(e.g. "Cheap zone (oversold)" instead of just "RSI < 30") since this dashboard is
meant to be readable by someone with no investing background.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dashboard import insights

TREND_LINE_COLOR = "#c3c2b7"  # neutral ink -- distinct from any model/instrument color

# Instrument identity colors (categorical, fixed assignment — never cycled/repainted)
COLORS = {
    "gold_futures":  "#FFD700",
    "goldbees_etf":  "#FF6B35",
    "hdfc_gold_etf": "#4ECDC4",
    "sbi_gold_nav":  "#A8DADC",
}

# Trend/indicator lines — fixed categorical hues from the validated dark-mode palette
MA_SHORT_COLOR = "#3987e5"    # blue
MA_LONG_COLOR = "#199e70"     # aqua
BB_BAND_COLOR = "rgba(100,100,200,0.4)"
BB_FILL_COLOR = "rgba(100,100,200,0.12)"

# Status colors (fixed meaning, never reused for a plain series)
STATUS_GOOD = "#0ca30c"       # cheap / oversold / historical dip
STATUS_CRITICAL = "#d03b3b"   # expensive / overbought
STATUS_WARNING = "#fab219"    # predicted dip zone (uncertain, forward-looking)

# Forecast model colors — fixed per model type so a chosen model looks the same
# everywhere. naive is deliberately muted (it's a "nothing changes" baseline, not a
# competing trend claim); every real model gets its own fixed categorical hue.
NAIVE_COLOR = "#898781"       # muted ink — flat/no-change baseline
LSTM_COLOR = "#d95926"        # orange — kept for backward-compatible callers
MODEL_COLORS = {
    "naive":    NAIVE_COLOR,
    "prophet":  "#3987e5",    # blue
    "arima":    "#199e70",    # aqua
    "xgboost":  "#c98500",    # yellow
    "lightgbm": "#9085e9",    # violet
    "lstm":     LSTM_COLOR,   # orange
    "ensemble": "#d55181",    # magenta
}
MODEL_DISPLAY_NAMES = {
    "naive": "If the price simply stays the same",
    "prophet": "Prophet",
    "arima": "ARIMA",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lstm": "Trend-based estimate (LSTM model)",
    "ensemble": "Ensemble (average of all models)",
}


def candlestick_chart(df: pd.DataFrame, instrument_label: str, show_mas: bool = True,
                      show_bb: bool = True, tech_df: pd.DataFrame = None,
                      forecast_series: dict = None, display_name_overrides: dict = None,
                      show_labels: bool = False, show_trend: bool = True) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.04,
        subplot_titles=[
            instrument_label, "Volume Traded",
            "Momentum — is it \"cheap\" or \"expensive\" right now?",
        ],
    )

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Daily price range", increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ), row=1, col=1)

    if show_trend:
        add_trend_line(fig, df["date"], df["close"], row=1, col=1)

    if tech_df is not None and show_mas:
        fig.add_trace(go.Scatter(
            x=tech_df["date"], y=tech_df["ma_50"], name="50-day average price",
            line=dict(color=MA_SHORT_COLOR, width=1.5, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=tech_df["date"], y=tech_df["ma_200"], name="200-day average price",
            line=dict(color=MA_LONG_COLOR, width=1.5, dash="dash"),
        ), row=1, col=1)

    if tech_df is not None and show_bb:
        fig.add_trace(go.Scatter(
            x=tech_df["date"], y=tech_df["bb_upper"], name="Typical high range",
            line=dict(color=BB_BAND_COLOR, width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=tech_df["date"], y=tech_df["bb_lower"], name="Typical low range",
            line=dict(color=BB_BAND_COLOR, width=1),
            fill="tonexty", fillcolor=BB_FILL_COLOR,
        ), row=1, col=1)

    if df["volume"].notna().any():
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"], name="Volume",
            marker_color="rgba(100,100,200,0.5)", showlegend=False,
        ), row=2, col=1)

    if tech_df is not None and "rsi_14" in tech_df.columns:
        _add_rsi_panel(fig, tech_df, row=3)

    if forecast_series:
        _add_forecast_series(fig, forecast_series, display_name_overrides,
                             row=1, col=1, show_labels=show_labels)

    fig.update_layout(
        height=780, xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=60),
    )
    return fig


def _add_rsi_panel(fig: go.Figure, tech_df: pd.DataFrame, row: int) -> None:
    """
    Shaded "cheap"/"expensive" zones instead of a bare dashed line at 70/30 — a
    shaded band reads as a zone to a non-technical viewer; a dashed line with no
    label does not. RSI itself is still plotted so a curious user can see the raw
    indicator, but the zones carry the actual meaning.
    """
    x0, x1 = tech_df["date"].iloc[0], tech_df["date"].iloc[-1]
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=30,
                  fillcolor=STATUS_GOOD, opacity=0.12, line_width=0, row=row, col=1)
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=70, y1=100,
                  fillcolor=STATUS_CRITICAL, opacity=0.12, line_width=0, row=row, col=1)
    fig.add_trace(go.Scatter(
        x=tech_df["date"], y=tech_df["rsi_14"], name="Momentum score (RSI)",
        line=dict(color="#FF6B35", width=1.5),
    ), row=row, col=1)
    fig.add_annotation(x=x1, y=15, text="Cheap zone", showarrow=False,
                        font=dict(color=STATUS_GOOD, size=11), xanchor="right", row=row, col=1)
    fig.add_annotation(x=x1, y=85, text="Expensive zone", showarrow=False,
                        font=dict(color=STATUS_CRITICAL, size=11), xanchor="right", row=row, col=1)


def line_chart(df: pd.DataFrame, instrument_label: str, tech_df: pd.DataFrame = None,
               forecast_series: dict = None, display_name_overrides: dict = None,
               show_labels: bool = False, show_trend: bool = True) -> go.Figure:
    """
    For NAV-only instruments that don't have OHLCV. Note: SBI Gold Fund (the only current
    NAV-only instrument) isn't in common.INSTRUMENTS, so it has no trained model / no
    gold.forecasts rows -- forecast_series will always be empty for it today, but the
    parameter is here so this function isn't a special case if that changes.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05,
                        subplot_titles=[instrument_label, "Momentum — cheap or expensive right now?"])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], name="Fund value (NAV)",
        line=dict(color=COLORS["sbi_gold_nav"], width=2),
    ), row=1, col=1)

    if show_trend:
        add_trend_line(fig, df["date"], df["close"], row=1, col=1)

    if tech_df is not None:
        if "ma_50" in tech_df.columns:
            fig.add_trace(go.Scatter(
                x=tech_df["date"], y=tech_df["ma_50"], name="50-day average",
                line=dict(color=MA_SHORT_COLOR, width=1.5, dash="dot"),
            ), row=1, col=1)
        if "rsi_14" in tech_df.columns:
            _add_rsi_panel(fig, tech_df, row=2)

    if forecast_series:
        _add_forecast_series(fig, forecast_series, display_name_overrides,
                             row=1, col=1, show_labels=show_labels)

    fig.update_layout(height=610, template="plotly_dark",
                      legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
                      margin=dict(l=40, r=40, t=60, b=60))
    return fig


def normalized_returns_chart(df: pd.DataFrame, forecast_df: pd.DataFrame = None,
                             show_labels: bool = False, show_trend: bool = True) -> go.Figure:
    """
    forecast_df (optional): same shape as df (date, instrument, return_from_inception_pct,
    instrument_label) but future-only, from queries.get_forecast_normalized_returns -- drawn
    as a dashed continuation of each instrument's own line, in that instrument's own color,
    so it reads as "this line keeps going" rather than a separate competing series.
    show_trend draws each instrument's own linear trend (dotted, its own color, thinner)
    over the selected window -- distinct from the forecast continuation (dashed).
    """
    fig = go.Figure()
    for instrument in df["instrument"].unique():
        inst_df = df[df["instrument"] == instrument]
        label = inst_df["instrument_label"].iloc[0]
        color = COLORS.get(instrument, None)
        fig.add_trace(go.Scatter(
            x=inst_df["date"], y=inst_df["return_from_inception_pct"],
            name=label, line=dict(color=color, width=2),
        ))

        if show_trend:
            trend = insights.compute_trend(inst_df["date"], inst_df["return_from_inception_pct"])
            if trend is not None:
                fig.add_trace(go.Scatter(
                    x=trend["dates"], y=trend["fitted"], name=f"{label} trend",
                    line=dict(color=color, width=1, dash="dot"), opacity=0.6,
                    showlegend=False, hoverinfo="skip",
                ))

        if forecast_df is not None and not forecast_df.empty:
            fc_df = forecast_df[forecast_df["instrument"] == instrument]
            if not fc_df.empty:
                trace_kwargs = dict(
                    x=fc_df["date"], y=fc_df["return_from_inception_pct"],
                    name=f"{label} (forecast)", line=dict(color=color, width=2, dash="dash"),
                    showlegend=False,
                    hovertemplate="%{y:.1f}%<extra>" + label + " (forecast)</extra>",
                )
                if show_labels:
                    trace_kwargs.update(
                        mode="lines+markers+text",
                        text=_thinned_pct_labels(fc_df["return_from_inception_pct"].tolist()),
                        textposition="top center", textfont=dict(size=9, color=color),
                        marker=dict(size=4, color=color),
                    )
                fig.add_trace(go.Scatter(**trace_kwargs))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    fig.update_layout(
        title="Which option would have grown your money the most?" +
              (" (dashed = forecast)" if forecast_df is not None and not forecast_df.empty else ""),
        yaxis_title="Growth since you first invested (%)", xaxis_title="Date",
        height=520, template="plotly_dark",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=80),
    )
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Move\ntogether?"),
    ))
    fig.update_layout(
        title="Do these move together, or independently? (1.0 = perfectly together, 0 = unrelated, -1.0 = opposite)",
        height=450, template="plotly_dark",
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def drawdown_chart(dip_df: pd.DataFrame, instrument_label: str, forecast_series: dict = None,
                   display_name_overrides: dict = None, show_labels: bool = False,
                   last_ma_200: float = None, show_trend: bool = True) -> go.Figure:
    """
    forecast_series draws each model's forward price line/band onto the price row (row 1).
    last_ma_200, if given, flags future days where a model's forecast dips >5% below the
    last known 200-day average as a "possible future dip" -- the same flat-carry-forward
    convention the models themselves use for technical indicators over the forecast
    horizon, so a genuine forward dip signal instead of the always-empty version before.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05,
                        subplot_titles=[f"{instrument_label} — Price", "How far below its recent peak (%)"])

    fig.add_trace(go.Scatter(
        x=dip_df["date"], y=dip_df["close_inr"], name="Price (INR)",
        line=dict(color="#FFD700", width=1.5),
    ), row=1, col=1)

    if show_trend:
        add_trend_line(fig, dip_df["date"], dip_df["close_inr"], row=1, col=1)

    hist_dips = dip_df[dip_df["is_dip_historical"] == True]
    if not hist_dips.empty:
        fig.add_trace(go.Scatter(
            x=hist_dips["date"], y=hist_dips["close_inr"],
            mode="markers", name="Past buying opportunity",
            marker=dict(color=STATUS_GOOD, size=6, symbol="circle"),
        ), row=1, col=1)

    if forecast_series:
        _add_forecast_series(fig, forecast_series, display_name_overrides,
                             row=1, col=1, show_labels=show_labels)

        if last_ma_200:
            display_names = dict(MODEL_DISPLAY_NAMES)
            display_names.update(display_name_overrides or {})
            threshold = last_ma_200 * 0.95
            for key, fdf in forecast_series.items():
                if fdf is None or fdf.empty:
                    continue
                dip_rows = fdf[fdf["yhat"] < threshold]
                if not dip_rows.empty:
                    fig.add_trace(go.Scatter(
                        x=dip_rows["date"], y=dip_rows["yhat"], mode="markers",
                        name=f"Possible future dip ({display_names.get(key, key)})",
                        marker=dict(color=STATUS_WARNING, size=8, symbol="diamond",
                                    line=dict(color=MODEL_COLORS.get(key, STATUS_WARNING), width=2)),
                    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dip_df["date"], y=dip_df["drawdown_pct"], name="Below recent peak (%)",
        fill="tozeroy", line=dict(color=STATUS_CRITICAL, width=1),
        fillcolor="rgba(208,59,59,0.2)",
    ), row=2, col=1)
    fig.add_hline(y=-5, line_dash="dash", line_color=STATUS_WARNING, opacity=0.7, row=2, col=1,
                  annotation_text="5% below peak — notable dip", annotation_position="bottom right")

    fig.update_layout(height=650, template="plotly_dark",
                      legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                      margin=dict(l=40, r=40, t=60, b=60))
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _thinned_label_mask(n: int, max_labels: int = 10) -> list:
    """
    True for roughly max_labels positions out of n, regardless of how long the series is --
    labeling every single day of a 90-day forecast is unreadable clutter, but a fixed
    "every Nth point" also looks sparse on a 7-day view. Scaling the stride to the series
    length keeps label density visually similar across any horizon. Full precision is
    always still on hover; these are just the always-visible on-chart callouts.
    """
    if n == 0:
        return []
    stride = max(1, round(n / max_labels))
    return [i % stride == 0 or i == n - 1 for i in range(n)]


def _thinned_currency_labels(values) -> list:
    mask = _thinned_label_mask(len(values))
    return [f"₹{v:,.0f}" if keep else "" for v, keep in zip(values, mask)]


def _thinned_pct_labels(values) -> list:
    mask = _thinned_label_mask(len(values))
    return [f"{v:+.1f}%" if keep else "" for v, keep in zip(values, mask)]


def add_trend_line(fig: go.Figure, dates, values, row: int = None, col: int = None) -> dict:
    """
    Draws a linear (OLS) trend line over whatever window is currently on screen -- this
    describes the *realized* direction of the selected period, separate from any model's
    forecast (which projects forward from here). Returns the computed trend dict (or None)
    so the caller can also render a "Trend: up/down X%" badge next to the chart.
    """
    trend = insights.compute_trend(dates, values)
    if trend is None:
        return None
    kwargs = {"row": row, "col": col} if row is not None else {}
    fig.add_trace(go.Scatter(
        x=trend["dates"], y=trend["fitted"], name="Trend (this period)",
        line=dict(color=TREND_LINE_COLOR, width=1.5, dash="dashdot"),
        hoverinfo="skip",
    ), **kwargs)
    return trend


def _add_forecast_series(fig: go.Figure, forecast_series: dict, display_name_overrides: dict = None,
                         row: int = None, col: int = None, show_labels: bool = False) -> None:
    """
    Shared by forecast_chart, forecast_candlestick_chart, candlestick_chart, line_chart,
    drawdown_chart: draws each model's forecast line + confidence band onto an existing
    figure (row/col for subplot placement).
    forecast_series: {model_name: future-only DataFrame with date/yhat/yhat_lower/yhat_upper}.
    model_name is one of naive/prophet/arima/xgboost/lightgbm/lstm/ensemble -- each has a
    fixed color (MODEL_COLORS) and default label (MODEL_DISPLAY_NAMES). Each series' band
    is tinted to match its own line color (not a single shared color) so overlapping bands
    stay visually distinguishable, and bands skip hover entirely -- the bound values show
    on the line's own tooltip instead, avoiding a confusing floating "<name> — uncertainty
    range" label with no value. show_labels adds the predicted value directly on the chart
    at a thinned set of points (see _thinned_currency_labels) -- full precision is always on hover
    regardless of this toggle.
    """
    display_names = dict(MODEL_DISPLAY_NAMES)
    display_names.update(display_name_overrides or {})
    kwargs = {"row": row, "col": col} if row is not None else {}

    for key, future_df in forecast_series.items():
        if future_df is None or future_df.empty:
            continue
        color = MODEL_COLORS.get(key, "#d95926")
        style = dict(color=color, dash="dot" if key == "naive" else "dash", width=2)
        display_name = display_names.get(key, key)
        has_band = future_df["yhat_upper"].notna().any()

        if has_band:
            fig.add_trace(go.Scatter(
                x=pd.concat([future_df["date"], future_df["date"].iloc[::-1]]),
                y=pd.concat([future_df["yhat_upper"], future_df["yhat_lower"].iloc[::-1]]),
                fill="toself", fillcolor=_hex_to_rgba(style["color"], 0.15),
                line=dict(color="rgba(255,255,255,0)"),
                name=display_name, showlegend=False, hoverinfo="skip",
            ), **kwargs)

        hover = "%{y:,.0f} INR"
        customdata = None
        if has_band:
            hover = "%{y:,.0f} INR<br>Likely range: %{customdata[0]:,.0f} – %{customdata[1]:,.0f}"
            customdata = future_df[["yhat_lower", "yhat_upper"]].values

        trace_kwargs = dict(
            x=future_df["date"], y=future_df["yhat"], name=display_name,
            line=style, customdata=customdata,
            hovertemplate=f"{hover}<extra>{display_name}</extra>",
        )
        if show_labels:
            trace_kwargs.update(
                mode="lines+markers+text",
                text=_thinned_currency_labels(future_df["yhat"].tolist()),
                textposition="top center",
                textfont=dict(size=9, color=style["color"]),
                marker=dict(size=4, color=style["color"]),
            )
        fig.add_trace(go.Scatter(**trace_kwargs), **kwargs)


def forecast_chart(hist_df: pd.DataFrame, forecast_series: dict, instrument_label: str,
                   display_name_overrides: dict = None, show_labels: bool = False,
                   show_trend: bool = True) -> go.Figure:
    """Plain closing-price line for history + forecast lines/bands. See forecast_candlestick_chart
    for a version that shows daily trading range (OHLC) instead of just the close."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["close"], name="Actual price so far",
        line=dict(color="#FFD700", width=1.5),
        hovertemplate="%{y:,.0f} INR<extra>Actual price so far</extra>",
    ))

    if show_trend:
        add_trend_line(fig, hist_df["date"], hist_df["close"])

    _add_forecast_series(fig, forecast_series, display_name_overrides, show_labels=show_labels)

    fig.update_layout(
        title=f"{instrument_label} — What might happen next?",
        yaxis_title="Price (INR)", xaxis_title="Date",
        height=520, template="plotly_dark",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=80),
    )
    return fig


def forecast_candlestick_chart(hist_df: pd.DataFrame, forecast_series: dict, instrument_label: str,
                               display_name_overrides: dict = None, show_labels: bool = False,
                               show_trend: bool = True) -> go.Figure:
    """
    Same forecast overlay as forecast_chart, but history is shown as candlesticks
    (open/high/low/close) instead of a plain closing-price line -- candles show each day's
    actual trading range, useful context for judging how volatile the recent action has
    been before trusting a forecast. The forecast itself can only ever be close-price
    lines (models here don't predict a full future OHLC range), so candles stop where
    history ends and the forecast lines/bands pick up from there.
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist_df["date"], open=hist_df["open"], high=hist_df["high"],
        low=hist_df["low"], close=hist_df["close"],
        name="Daily price range", increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ))

    if show_trend:
        add_trend_line(fig, hist_df["date"], hist_df["close"])

    _add_forecast_series(fig, forecast_series, display_name_overrides, show_labels=show_labels)

    fig.update_layout(
        title=f"{instrument_label} — Recent trading range + what might happen next",
        yaxis_title="Price (INR)", xaxis_title="Date",
        height=560, template="plotly_dark", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=60, b=80),
    )
    return fig


def model_scores_chart(scores_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        scores_df, x="model_name", y="mape", color="instrument_label",
        barmode="group", text_auto=".2f",
        labels={"mape": "Average error (%) — lower is better", "model_name": "Model",
                "instrument_label": "Instrument"},
        title="How far off was each model's price guess, on average? (lower bars are better)",
        template="plotly_dark",
        color_discrete_sequence=["#3987e5", "#199e70", "#c98500", "#9085e9"],
    )
    fig.update_layout(height=450, margin=dict(l=40, r=40, t=70, b=40))
    return fig


def volatility_chart(tech_df: pd.DataFrame, instrument_label: str, show_trend: bool = True) -> go.Figure:
    # rolling_vol_30d is a std dev of the raw price LEVEL (INR), not of returns -- normalize
    # by price so this reads as a genuine "% of price" swing regardless of the instrument's
    # absolute price level (without this, gold_futures at ~4 lakh INR shows a std dev in the
    # tens of thousands, which * 100 looks like millions on the axis instead of a percent).
    vol_pct = (tech_df["rolling_vol_30d"] / tech_df["close_inr"]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tech_df["date"], y=vol_pct,
        name="How much the price has been swinging",
        fill="tozeroy", line=dict(color="#6A4C93", width=1.5),
        fillcolor="rgba(106,76,147,0.2)",
    ))
    if show_trend:
        add_trend_line(fig, tech_df["date"], vol_pct)
    fig.update_layout(
        title=f"{instrument_label} — How bumpy has the ride been? (last 30 days)",
        yaxis_title="Typical daily swing (%)", xaxis_title="Date",
        height=380, template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
