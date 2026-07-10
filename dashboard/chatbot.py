"""
"Ask AI" chatbot: a local-LLM agent (via Ollama) that answers questions by querying the
DuckDB database and explains the dashboard's pages -- no paid API, fully on-device.

Layout: pure agent core first (system prompt, tools, executors, loop -- no Streamlit
imports used), Streamlit UI wrapper at the bottom. The core is testable from a terminal:

    python -m dashboard.chatbot "what's the latest gold price?"

The Ollama model is swappable via the OLLAMA_MODEL env var (default llama3.1:8b) -- this
is also the seam where a stronger local model or a hosted API backend would plug in
later, without touching the tools or the UI.
"""

import os
import re
import sys
import json
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import pandas as pd

from dashboard import queries, charts, insights

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
# Ollama's default context window (4096) silently truncates -- our system prompt alone is
# ~1K tokens and tool results add up fast. Low temperature keeps tool calls / SQL reliable.
OLLAMA_OPTIONS = {"num_ctx": 8192, "temperature": 0.2}
MAX_TOOL_ROUNDS = 5
SQL_ROW_CAP = 200
SQL_CELL_CAP = 300


def ollama_status() -> str:
    """'ok' | 'package_missing' | 'server_down' | 'model_missing' -- drives the chat
    page's graceful-degradation notices (Streamlit Cloud has no Ollama server, so the
    deployed app naturally lands on 'server_down' -> a 'local mode only' message)."""
    if not OLLAMA_AVAILABLE:
        return "package_missing"
    try:
        models = ollama.list()
    except Exception:
        return "server_down"
    names = [m.model for m in models.models]
    # `ollama pull llama3.1:8b` registers as "llama3.1:8b"; a bare tag matches too.
    if any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL + ":") for n in names):
        return "ok"
    return "model_missing"


# ── System prompt ────────────────────────────────────────────────────────────
# Frozen at import time. Dynamic context (today's date, latest data date) is prepended
# to the first user turn instead, so it never bloats or destabilizes this prompt.

_instrument_lines = "\n".join(
    f"- {key} = {label}" for key, label in queries.INSTRUMENT_LABELS.items()
)
_model_lines = ", ".join(
    f"{name} ({charts.MODEL_DISPLAY_NAMES[name]})" for name in queries.ALL_MODEL_NAMES
)

SYSTEM_PROMPT = f"""You are the built-in assistant of a personal "Gold Forecast Dashboard" -- a Streamlit app that tracks non-physical gold investment options in India and forecasts prices with machine-learning models. Answer questions using the tools provided, explain the dashboard's pages and charts, and help users think through gold investment decisions at their level (plain language for beginners, metrics for experienced investors).

## Data you can query (DuckDB, read-only)
- silver.prices(date, instrument, open, high, low, close, close_inr, volume, daily_return_pct, log_return) -- daily prices; close_inr is the INR price (use it).
- silver.macro_features(date, vix_close, oil_close, usd_index_close, us10y_yield_close, usdinr_close) -- daily macro indicators.
- gold.technical_features(date, instrument, close_inr, ma_50, ma_200, bb_upper, bb_lower, rolling_vol_30d, rsi_14, drawdown_pct, is_dip_historical) -- technical indicators.
- gold.forecasts(date, instrument, model_name, yhat, yhat_lower, yhat_upper, is_future) -- model forecasts; is_future=true rows are the live forward forecast.
- gold.model_scores(instrument, model_name, rmse, mae, mape, skill_score_vs_naive, directional_accuracy, selected) -- backtest scores; selected=true = best model.
- gold.live_predictions(instrument, model_name, predicted_on, predicted_for, predicted_price, actual_price, abs_error, pct_error) -- real daily predictions reconciled against actual outcomes.
- gold.etf_premium(date, instrument, ratio, premium_vs_1y_avg_pct, premium_zscore).
- gold.normalized_returns(date, instrument, return_from_inception_pct).

Instruments (exact keys):
{_instrument_lines}
Caveats: hdfc_gold_etf has history only since June 2023 (noisier stats); sbi_gold_nav is NAV-only with NO forecasts or technical features.

Models: {_model_lines}. skill_score_vs_naive > 0 means the model beats the naive random-walk baseline (tomorrow = today) -- the honest bar in finance. Backtest scores test models frozen at end-2024 against 2025; the live track record logs real daily predictions -- they can disagree, and the live record matters more as it grows.

## Dashboard pages (for "explain this page" questions)
1. Overview -- normalized %-since-inception returns for all instruments on one chart (fair cross-instrument comparison) + correlation matrix of daily returns.
2. Individual Instrument -- candlestick chart with 50/200-day moving averages (golden/death cross markers), Bollinger Bands, RSI, volatility, ETF premium/discount vs international gold, monthly seasonality heatmap, market-event annotations.
3. Dip Tracker -- drawdown from the 200-day peak, historical dip days, and a backtest of whether buying dips beat ordinary days (forward returns at 1/3/6/12 months).
4. Forecast -- up to 90-day forecast per model with confidence bands, a consensus panel (models saying up/down/flat), and the Live Track Record of past predictions vs actuals.
5. Model Comparison -- the backtest scoreboard across all models and instruments.
6. SGB Calculator -- Sovereign Gold Bond return calculator (2.5%/yr interest + gold appreciation, 8-year tenure; NOT forecast -- the secondary market is too illiquid).

## Rules
- NEVER state a price, return, score, or forecast without getting it from a tool first. If a tool errors, read the error and call the tool again with a fixed input -- do not guess numbers.
- After a tool returns, ANSWER THE USER'S QUESTION DIRECTLY from its data. Do not show SQL to the user, do not describe what query could be run -- give the actual numbers and what they mean.
- get_market_snapshot already contains the latest price of every instrument -- if it answers the question, answer immediately from it without calling more tools.
- Prefer the specific tools; use run_sql only for questions they don't cover.
- run_sql: ONE single SELECT statement, always include LIMIT (max {SQL_ROW_CAP}), date literals as 'YYYY-MM-DD'. DuckDB strftime uses % codes: strftime(date,'%Y') gives the year -- 'YYYY' as a format string will NOT work. Examples:
  SELECT date, close_inr FROM silver.prices WHERE instrument='gold_futures' ORDER BY date DESC LIMIT 5
  SELECT strftime(date,'%Y') AS yr, MIN(close_inr) AS low, MAX(close_inr) AS high FROM silver.prices WHERE instrument='goldbees_etf' GROUP BY yr ORDER BY yr LIMIT 30
- Adapt to the user: plain language and analogies for beginners; RSI/drawdown/skill-score detail for pros.
- Forecasts are uncertain: always mention the confidence interval or consensus split, never present a single number as fact.
- End every answer involving an investment decision with: "This is educational context from a personal analytics tool, not financial advice."
"""


# ── Tools ────────────────────────────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "get_market_snapshot",
        "description": (
            "Current state of the gold market: latest INR close for every instrument, "
            "how fresh the data is, which forecast model is currently best per "
            "instrument, and a plain-language macro read (VIX, USD index, 10Y yield, "
            "oil, USD/INR vs 1-yr averages). ALWAYS call this first for any 'how are "
            "things now', price, or investment question."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_forecast_summary",
        "description": (
            "Price forecasts from all 7 models for ONE instrument at a horizon, plus "
            "the cross-model consensus (how many say up/down/flat, median % change). "
            "Call when the user asks about future prices or investing over a time "
            "period. Not available for sbi_gold_nav."),
        "parameters": {"type": "object", "properties": {
            "instrument": {"type": "string", "enum": queries.OHLCV_INSTRUMENTS},
            "horizon_days": {"type": "integer",
                             "description": "Trading days ahead, 1-90"}},
            "required": ["instrument", "horizon_days"]}}},
    {"type": "function", "function": {
        "name": "get_model_performance",
        "description": (
            "How accurate the forecast models are: backtest scores (RMSE, MAPE, skill "
            "vs naive baseline, directional accuracy, which is selected best) and the "
            "recent live next-day prediction track record. Call when asked how "
            "trustworthy the forecasts are."),
        "parameters": {"type": "object", "properties": {
            "instrument": {"type": "string", "enum": queries.OHLCV_INSTRUMENTS,
                           "description": "Optional -- omit for all instruments"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_dip_analysis",
        "description": (
            "'Buy the dip' analysis for ONE instrument: current drawdown from peak and "
            "RSI, plus the historical backtest of forward returns (1/3/6/12 months) "
            "after dip days vs ordinary days. Call for questions about entry timing or "
            "whether now is a good time to buy."),
        "parameters": {"type": "object", "properties": {
            "instrument": {"type": "string", "enum": queries.OHLCV_INSTRUMENTS}},
            "required": ["instrument"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": (
            "Run ONE read-only SELECT on the DuckDB database (schema in your "
            "instructions). Only for questions the other tools don't answer. Single "
            "statement, include LIMIT (max 200 rows), dates as 'YYYY-MM-DD'."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "A single SELECT/WITH statement with LIMIT"}},
            "required": ["query"]}}},
]


# ── Tool executors ───────────────────────────────────────────────────────────

def _check_instrument(instrument: str) -> str | None:
    """Whitelist-validate before anything reaches queries.py's f-string SQL."""
    if instrument not in queries.OHLCV_INSTRUMENTS:
        return (f"ERROR: unknown instrument '{instrument}'. "
                f"Valid: {', '.join(queries.OHLCV_INSTRUMENTS)}")
    return None


def _get_market_snapshot() -> str:
    lines = ["## Market snapshot"]
    for inst in queries.ALL_INSTRUMENTS:
        label = queries.INSTRUMENT_LABELS[inst]
        last_close = queries.get_last_close(inst)
        _, max_date = queries.get_date_range(inst)
        line = f"- {label} ({inst}): last close ₹{last_close:,.2f} (data through {max_date})"
        if inst in queries.OHLCV_INSTRUMENTS:
            line += f"; best forecast model right now: {queries.get_selected_model(inst)}"
        lines.append(line)
    snapshot = queries.get_macro_snapshot()
    if snapshot:
        lines.append("\n## Macro conditions\n" + insights.macro_commentary(snapshot))
    return "\n".join(lines)


def _get_forecast_summary(instrument: str, horizon_days: int) -> str:
    if err := _check_instrument(instrument):
        return err
    horizon_days = max(1, min(int(horizon_days), 90))
    last_close = queries.get_last_close(instrument)
    frames = []
    lines = [f"## Forecasts for {queries.INSTRUMENT_LABELS[instrument]} "
             f"~{horizon_days} trading days ahead (last close ₹{last_close:,.2f})"]
    for model in queries.ALL_MODEL_NAMES:
        df = queries.get_forecasts_future_only(instrument, model_name=model)
        if df.empty:
            continue
        df = df.assign(model_name=model)
        frames.append(df)
        row = df.iloc[min(horizon_days, len(df)) - 1]
        pct = (row["yhat"] / last_close - 1) * 100
        band = ""
        if pd.notna(row.get("yhat_lower")) and pd.notna(row.get("yhat_upper")):
            band = f" (range ₹{row['yhat_lower']:,.0f}-₹{row['yhat_upper']:,.0f})"
        lines.append(f"- {model}: ₹{row['yhat']:,.2f} ({pct:+.1f}%){band}")
    if not frames:
        return f"ERROR: no forecasts available for {instrument}."
    all_future = pd.concat(frames, ignore_index=True)
    consensus = insights.forecast_consensus(all_future, last_close, horizon_days)
    if consensus:
        horizon_label = f"{horizon_days} trading days"
        lines.append("\n## Model consensus\n" + insights.consensus_text(consensus, horizon_label))
    return "\n".join(lines)


def _get_model_performance(instrument: str = None) -> str:
    scores = queries.get_model_scores()
    if instrument:
        if err := _check_instrument(instrument):
            return err
        scores = scores[scores["instrument"] == instrument]
    lines = ["## Backtest scores (2025 holdout; skill > 0 beats the naive baseline)",
             scores.round(4).to_markdown(index=False)]
    for inst in scores["instrument"].unique():
        live = queries.get_live_predictions(inst)
        live = live[live["actual_price"].notna()] if not live.empty else live
        if live.empty:
            continue
        recent = live.sort_values("predicted_for").groupby("model_name").tail(10)
        acc = recent.groupby("model_name")["pct_error"].mean().round(3)
        lines.append(f"\n## Live track record for {inst} "
                     f"(mean abs % error, last {min(10, len(recent))} reconciled days)\n"
                     + acc.to_markdown())
    return "\n".join(lines)


def _get_dip_analysis(instrument: str) -> str:
    if err := _check_instrument(instrument):
        return err
    label = queries.INSTRUMENT_LABELS[instrument]
    fwd = queries.get_dip_forward_returns(instrument)
    if fwd.empty:
        return f"ERROR: no dip data for {instrument}."
    summary = insights.dip_backtest_summary(fwd)
    verdict = insights.dip_backtest_verdict(summary, label)
    min_d, max_d = queries.get_date_range(instrument)
    tech = queries.get_technical_features(instrument, str(min_d), str(max_d))
    latest = tech.dropna(subset=["drawdown_pct"]).iloc[-1]
    lines = [
        f"## Dip analysis for {label}",
        f"Current drawdown from 200-day peak: {latest['drawdown_pct']:.1f}%",
        f"Current RSI(14): {latest['rsi_14']:.0f}" if pd.notna(latest["rsi_14"]) else "",
        f"Currently flagged as a dip day: {bool(latest['is_dip_historical'])}",
        "\n## Historical 'buy the dip' backtest (median forward returns %)",
        summary.round(2).to_markdown(index=False),
        "\n" + verdict,
    ]
    return "\n".join(l for l in lines if l)


_SQL_ALLOWED = re.compile(r"^\s*(select|with|describe|show|summarize)\b", re.IGNORECASE)


def _run_sql(query: str) -> str:
    q = re.sub(r"--[^\n]*", "", query)              # strip line comments
    q = re.sub(r"/\*.*?\*/", "", q, flags=re.DOTALL)  # strip block comments
    q = q.strip().rstrip(";").strip()
    if ";" in q:
        return "ERROR: a single SQL statement only -- remove the extra ';'."
    if not _SQL_ALLOWED.match(q):
        return ("ERROR: read-only queries only "
                "(SELECT / WITH / DESCRIBE / SHOW / SUMMARIZE).")
    con = duckdb.connect(queries.DB_PATH, read_only=True)
    try:
        con.execute("SET enable_external_access=false")
        df = con.execute(q).fetchdf().head(SQL_ROW_CAP)
    finally:
        con.close()
    if df.empty:
        return "Query ran fine but returned 0 rows."
    df = df.astype(object).where(df.notna(), None)
    for col in df.columns:
        df[col] = df[col].map(
            lambda v: (str(v)[:SQL_CELL_CAP] + "…")
            if isinstance(v, str) and len(v) > SQL_CELL_CAP else v)
    return f"{len(df)} row(s):\n{df.to_markdown(index=False)}"


def _arg(args: dict, *names, required=True):
    """Small local models sometimes use a near-miss key ('sql' instead of 'query').
    Accept known aliases; if truly missing, return a corrective message the model can
    act on instead of a cryptic KeyError."""
    for n in names:
        if args.get(n) not in (None, ""):
            return args[n], None
    if not required:
        return None, None
    return None, (f"ERROR: missing argument '{names[0]}'. "
                  f"Call this tool again with '{names[0]}' set.")


def execute_tool(name: str, args: dict) -> str:
    """Dispatch a tool call. Never raises: errors come back as text so the model can
    read them and retry (important for a small local model's SQL misses)."""
    try:
        if name == "get_market_snapshot":
            return _get_market_snapshot()
        if name == "get_forecast_summary":
            inst, err = _arg(args, "instrument")
            if err:
                return err
            horizon, err = _arg(args, "horizon_days", "horizon", "days")
            if err:
                return err
            return _get_forecast_summary(inst, horizon)
        if name == "get_model_performance":
            inst, _ = _arg(args, "instrument", required=False)
            return _get_model_performance(inst)
        if name == "get_dip_analysis":
            inst, err = _arg(args, "instrument")
            if err:
                return err
            return _get_dip_analysis(inst)
        if name == "run_sql":
            query, err = _arg(args, "query", "sql", "q", "statement")
            if err:
                return err
            return _run_sql(query)
        return f"ERROR: unknown tool '{name}'."
    except duckdb.IOException:
        return ("ERROR: the database is busy (the daily data refresh is probably "
                "writing right now). Try again in a minute.")
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ── Agent core (no Streamlit) ────────────────────────────────────────────────

def first_turn_context() -> str:
    """Dynamic context line prepended to the first user message of a conversation --
    kept out of SYSTEM_PROMPT so the prompt stays frozen."""
    try:
        _, latest = queries.get_date_range("gold_futures")
        return f"[Context: today is {date.today()}; market data available through {latest}]"
    except Exception:
        return f"[Context: today is {date.today()}]"


def run_agent_turn(messages: list, on_text=None, on_tool=None) -> str:
    """Drive one user turn to completion, mutating `messages` in place through up to
    MAX_TOOL_ROUNDS tool rounds. `on_text(delta)` / `on_tool(name, args)` let the UI
    stream output without this core importing Streamlit. Returns the assistant's text
    joined across ALL rounds -- a model may put the substance in one round and only a
    closing line in the next, so keeping just the last round loses the answer."""
    round_texts = []
    for _ in range(MAX_TOOL_ROUNDS):
        content, tool_calls = "", []
        for chunk in ollama.chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS,
                                 stream=True, options=OLLAMA_OPTIONS):
            if chunk.message.content:
                content += chunk.message.content
                if on_text:
                    on_text(chunk.message.content)
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)

        # Persist plain dicts (not pydantic objects) so session_state stays serializable.
        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        messages.append(assistant_msg)
        if content.strip():
            round_texts.append(content.strip())
        final_text = "\n\n".join(round_texts)

        if not tool_calls:
            break
        for call in tool_calls:
            args = call.function.arguments or {}
            if isinstance(args, str):  # some models emit JSON strings
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if on_tool:
                on_tool(call.function.name, args)
            messages.append({"role": "tool",
                             "tool_name": call.function.name,
                             "content": execute_tool(call.function.name, args)})
    else:
        # Round cap hit without a final answer -- tell the user instead of going silent.
        fallback = (final_text or "I couldn't get a clean answer from the database after "
                    "several attempts -- try rephrasing the question more simply.")
        if on_text and not final_text:
            on_text(fallback)
        return fallback
    return final_text


def new_conversation() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


MAX_EXCHANGES = 6


def trim_history(messages: list) -> list:
    """Keep the system message + roughly the last MAX_EXCHANGES user turns. Trim only
    at plain user-text boundaries so a tool result is never orphaned from its call."""
    user_turn_idx = [i for i, m in enumerate(messages)
                     if m["role"] == "user"]
    if len(user_turn_idx) <= MAX_EXCHANGES:
        return messages
    cut = user_turn_idx[len(user_turn_idx) - MAX_EXCHANGES]
    return [messages[0]] + messages[cut:]


if __name__ == "__main__":
    # Terminal smoke test: python -m dashboard.chatbot "what's the latest gold price?"
    question = " ".join(sys.argv[1:]) or "What's the latest gold price?"
    status = ollama_status()
    if status != "ok":
        sys.exit(f"Ollama not ready: {status}")
    msgs = new_conversation()
    msgs.append({"role": "user", "content": f"{first_turn_context()}\n{question}"})
    run_agent_turn(
        msgs,
        on_text=lambda d: print(d, end="", flush=True),
        on_tool=lambda n, a: print(f"\n[tool: {n}({a})]", flush=True),
    )
    print()


# ── Streamlit UI ─────────────────────────────────────────────────────────────
# Imported lazily-ish: streamlit is always available in the dashboard venvs, and
# importing it here is harmless for the CLI path too.

import streamlit as st  # noqa: E402


_STATUS_NOTICES = {
    "package_missing": ("The `ollama` Python package isn't installed in this "
                        "environment. Run `pip install ollama` and reload."),
    "server_down": ("**AI chat runs in local mode only.** No Ollama server is "
                    "reachable here. On your own machine: install Ollama "
                    "(ollama.com/download), make sure it's running, then reload. "
                    "On the cloud deployment this feature is unavailable by design."),
    "model_missing": (f"Ollama is running but the model isn't downloaded yet. "
                      f"Run `ollama pull {OLLAMA_MODEL}` (a few GB, one time)."),
}


def sidebar_explain_button(current_page: str):
    """One sidebar button that pre-seeds the chat with an explain-this-page question
    and navigates to Ask AI. on_click callbacks run before widgets instantiate on the
    next rerun, so setting the nav radio's session key there is legal."""
    if current_page == "Ask AI":
        return

    def _go():
        st.session_state["chat_pending_prompt"] = (
            f"Explain what the '{current_page}' page of this dashboard shows, "
            f"how to read its charts, and what I should look at first."
        )
        st.session_state["nav_page"] = "Ask AI"

    st.sidebar.button("🤖 Explain this page", on_click=_go,
                      help="Ask the AI assistant to walk you through this page")


def render_chat_page():
    st.title("🤖 Ask AI")
    st.caption(f"A local AI assistant ({OLLAMA_MODEL} via Ollama) that answers from "
               "this dashboard's own database. Nothing leaves your machine.")

    status = ollama_status()
    if status != "ok":
        st.info(_STATUS_NOTICES[status])
        return

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = new_conversation()
        st.session_state["chat_display"] = []

    if st.session_state["chat_display"]:
        if st.button("🗑️ Clear chat"):
            st.session_state["chat_messages"] = new_conversation()
            st.session_state["chat_display"] = []
            st.rerun()

    for msg in st.session_state["chat_display"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    prompt = st.chat_input("Ask about gold prices, forecasts, or this dashboard...")
    if not prompt:
        prompt = st.session_state.pop("chat_pending_prompt", None)
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    # Work on a copy; commit to session_state only after the turn fully completes, so a
    # mid-generation rerun can't persist an assistant message with unanswered tool calls.
    messages = list(st.session_state["chat_messages"])
    is_first = not any(m["role"] == "user" for m in messages)
    messages.append({"role": "user",
                     "content": f"{first_turn_context()}\n{prompt}" if is_first else prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = {"text": ""}

        def on_text(delta: str):
            streamed["text"] += delta
            placeholder.markdown(streamed["text"] + "▌")

        def on_tool(name: str, args: dict):
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            st.caption(f"🔍 Querying: `{name}({arg_str})`")
            if streamed["text"]:
                streamed["text"] += "\n\n"  # keep earlier rounds' text visible

        try:
            with st.spinner("Thinking (local model -- first response can take a minute)..."):
                final_text = run_agent_turn(messages, on_text=on_text, on_tool=on_tool)
        except Exception as e:
            placeholder.error(f"The local model failed: {e}. "
                              "Check that Ollama is running (`systemctl status ollama` "
                              "or `ollama serve`).")
            return
        placeholder.markdown(final_text or "*(no answer produced -- try rephrasing)*")

    st.session_state["chat_messages"] = trim_history(messages)
    st.session_state["chat_display"].append({"role": "user", "text": prompt})
    st.session_state["chat_display"].append({"role": "assistant",
                                             "text": final_text or "*(no answer)*"})
