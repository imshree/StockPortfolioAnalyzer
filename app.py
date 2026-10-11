"""Stock Portfolio Performance Analyzer - Streamlit app.

Run with ./run.sh (or: streamlit run app.py).
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio import market_data
from portfolio.calculations import (apply_splits, build_positions, holdings_table,
                                    portfolio_summary)
from portfolio.explain import explain
from portfolio.history import RANGES, daily_values, range_view
from portfolio.recommend import recommend
from portfolio.transactions import SAMPLE_CSV, FileRejected, parse_transactions

DATA_DIR = Path(__file__).resolve().parent / "data"
TXN_FILE = DATA_DIR / "transactions.csv"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"
ET = ZoneInfo("America/New_York")
BAR_COLOR = "#7E1FB5"  # matches the reference history chart
PIE_COLORS = ["#7E1FB5", "#4263EB", "#1098AD", "#F59F00", "#AE3EC9", "#748FFC", "#3BC9DB",
              "#FAB005", "#5F3DC4", "#228BE6", "#0C8599", "#E67700"] * 5
DISCLAIMER = ("This is an automated analysis for information only, not financial advice. It does "
              "not consider your full financial situation, goals or taxes. Past performance does "
              "not predict future results.")

st.set_page_config(page_title="Stock Portfolio Analyzer", page_icon=":material/monitoring:",
                   layout="wide")


# ---------------------------------------------------------------- persistence

def load_saved() -> pd.DataFrame | None:
    if not TXN_FILE.exists():
        return None
    df = pd.read_csv(TXN_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def save(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(TXN_FILE, index=False)


if "txns" not in st.session_state:
    st.session_state.txns = load_saved()


# ---------------------------------------------------------------- market data (cached)

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_known(symbols: tuple[str, ...]) -> set[str]:
    return market_data.known_symbols(symbols)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(symbols: tuple[str, ...], start: date):
    return market_data.daily_prices(symbols, start)


@st.cache_data(ttl=60, show_spinner=False)
def cached_quotes(symbols: tuple[str, ...]):
    return market_data.quotes(symbols), datetime.now(ET)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_fundamentals(symbol: str) -> dict:
    return market_data.fundamentals(symbol)


def market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(ET)
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)


def load_portfolio(txns: pd.DataFrame):
    """Prices, split-adjusted positions and the holdings table for the saved transactions."""
    symbols = tuple(sorted(txns["Symbol"].unique()))
    start = min(txns["Date"].min(), date.today() - timedelta(days=420))
    closes, splits = cached_prices(symbols + (market_data.BENCHMARK,), start)
    adjusted = apply_splits(txns, splits)
    positions = build_positions(adjusted)
    open_syms = tuple(s for s, p in positions.items() if p.qty > 0)

    live, fetched_at = cached_quotes(open_syms)
    quotes, stale = {}, []
    for s in open_syms:
        if s in live:
            quotes[s] = live[s]
            st.session_state.setdefault("last_quotes", {})[s] = (live[s], fetched_at)
        elif s in st.session_state.get("last_quotes", {}):
            quotes[s], _ = st.session_state.last_quotes[s]
            stale.append(s)
        else:  # fall back to the last two daily closes
            c = closes[s].dropna() if s in closes else pd.Series(dtype=float)
            if len(c) >= 2:
                quotes[s] = {"last": float(c.iloc[-1]), "prev_close": float(c.iloc[-2])}
                stale.append(s)
    table = holdings_table(positions, quotes)
    return adjusted, closes, positions, table, fetched_at, stale


# ---------------------------------------------------------------- formatting

def money(x: float, signed: bool = False) -> str:
    sign = "+" if signed and x > 0 else "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def md(text: str) -> str:
    return text.replace("$", "\\$")  # keep Streamlit markdown from reading $...$ as LaTeX


def tone(text: str, x: float) -> str:
    """Green with + for gains, red with - for losses: colour is never the only signal."""
    text = md(text)
    return f":green[{text}]" if x > 0 else f":red[{text}]" if x < 0 else text


# ---------------------------------------------------------------- tab 1: upload

def upload_tab() -> None:
    st.subheader("Upload transactions")
    st.caption("One CSV of BUY, SELL and DIVIDEND transactions. Accepts .csv up to 5 MB or "
               "10,000 rows. Your data stays on this computer.")
    st.download_button("Download sample CSV", SAMPLE_CSV, "sample_transactions.csv", "text/csv",
                       icon=":material/download:")

    file = st.file_uploader("Transactions CSV", type=["csv"], label_visibility="collapsed")
    if file is None:
        if st.session_state.txns is not None:
            st.info(f"Current portfolio: {len(st.session_state.txns)} transactions imported. "
                    "Upload a new file to replace it.")
            if st.button("Clear my data", icon=":material/delete:"):
                TXN_FILE.unlink(missing_ok=True)
                st.session_state.txns = None
                st.rerun()
        return

    cache_key = (file.name, file.size, file.file_id)
    if st.session_state.get("parsed_key") != cache_key:
        with st.spinner("Checking your transactions..."):
            try:
                result = parse_transactions(file.getvalue(), file.name, known_symbols=cached_known)
            except FileRejected as e:
                st.error(str(e))
                return
        st.session_state.parsed_key = cache_key
        st.session_state.parsed = result
        st.session_state.imported_key = None
    result = st.session_state.parsed

    with st.container(border=True):
        cols = st.columns(5)
        cols[0].metric("Rows read", result.rows_read)
        cols[1].metric("Imported", result.rows_imported)
        cols[2].metric("Skipped", len(result.errors))
        cols[3].metric("Holdings found", result.holdings_found)
        rng = result.date_range
        cols[4].metric("Date range", f"{rng[0]:%b %d, %Y} to {rng[1]:%b %d, %Y}" if rng else "-")

    if result.errors:
        st.warning(f"{len(result.errors)} rows were skipped. Fix them in your file and upload again.")
        st.dataframe(pd.DataFrame([{"Row": e.row, "Value": e.value, "Reason": e.reason}
                                   for e in result.errors]), hide_index=True, width="stretch")
    for w in result.warnings:
        st.caption(f"Note: {w.message}")

    if result.rows_imported == 0:
        st.error("No valid transactions to import.")
        return

    already = st.session_state.get("imported_key") == cache_key
    if not already:
        if st.session_state.txns is not None:
            st.warning("This replaces your current portfolio.")
            label = "Replace portfolio"
        else:
            label = "Import transactions"
        if st.button(label, type="primary"):
            save(result.transactions)
            st.session_state.txns = result.transactions
            st.session_state.imported_key = cache_key
            st.rerun()
    else:
        st.success("Imported.")

        def go_to_portfolio():
            st.session_state.main_tab = "Portfolio"
        st.button("View portfolio", type="primary", on_click=go_to_portfolio)


# ---------------------------------------------------------------- tab 2: portfolio

SORT_COLUMNS = ["Symbol", "Last Price $", "Change $", "Change %", "Qty #", "Price Paid $",
                "Day's Gain $", "Total Gain $", "Total Gain %", "Value $"]
WIDTHS = [0.9, 1.9, 1.1, 1, 1, 0.8, 1.1, 1.1, 1.1, 1.1, 1.1]


@st.dialog("Recommendation", width="large")
def recommendation_dialog(symbol: str) -> None:
    txns = st.session_state.txns
    adjusted, closes, positions, table, _, _ = load_portfolio(txns)
    row = table.set_index("Symbol").loc[symbol]
    total_value = table["Value $"].sum()
    weight = row["Value $"] / total_value * 100 if total_value else 0.0

    with st.spinner("Analysing..."):
        f = cached_fundamentals(symbol)
        series = closes[symbol].dropna()
        bench = closes[market_data.BENCHMARK].dropna() if market_data.BENCHMARK in closes else None
        rec = recommend(symbol, series, bench, row["Last Price $"], f, weight, row["Total Gain %"])
        key = ("explained", symbol, rec.verdict, round(rec.score, 1))
        if key not in st.session_state:
            st.session_state[key] = explain(rec)
        reasons, triggers, source = st.session_state[key]

    # 1. Header
    st.markdown(f"### {symbol} · {md(f['name'])}")
    change = f"{money(row['Change $'], True)} ({row['Change %']:+.2f}%)"
    st.markdown(f"**{md(money(row['Last Price $']))}** &nbsp; {tone(change, row['Change $'])} today")

    # 2. Verdict badge
    color = {"Buy more": "green", "Hold": "orange", "Sell": "red"}[rec.verdict]
    st.markdown(f":{color}-badge[**{rec.verdict}**] &nbsp; Confidence: **{rec.confidence}** "
                f"&nbsp; · &nbsp; Score {rec.score:+.0f} of ±100")

    # 3. Your position
    st.markdown("##### Your position")
    c = st.columns(4)
    c[0].metric("Shares", f"{row['Qty #']:,.4g}")
    c[1].metric("Price paid", money(row["Price Paid $"]))
    c[2].metric("Total gain", money(row["Total Gain $"], True), f"{row['Total Gain %']:+.2f}%")
    c[3].metric("Of portfolio", f"{weight:.1f}%")

    # 4. Market trend: 6-month price with 50- and 200-day averages, returns vs S&P 500
    st.markdown("##### Market trend")
    six = series[series.index >= series.index[-1] - pd.DateOffset(months=6)]
    fig = go.Figure()
    fig.add_scatter(x=six.index, y=six, name="Price", line=dict(width=2))
    for n, dash in ((50, "dot"), (200, "dash")):
        ma = series.rolling(n).mean()
        fig.add_scatter(x=six.index, y=ma.loc[six.index], name=f"{n}-day avg",
                        line=dict(width=1.25, dash=dash))
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=8, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0), yaxis_tickprefix="$")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    def ret(s, days):
        s = s.dropna()
        return (s.iloc[-1] / s.iloc[-1 - days] - 1) * 100 if len(s) > days else None
    rows = []
    for label, days in (("1M", 21), ("3M", 63), ("1Y", 252)):
        a, b = ret(series, days), ret(bench, days) if bench is not None else None
        rows.append({"Period": label, symbol: f"{a:+.1f}%" if a is not None else "-",
                     "S&P 500": f"{b:+.1f}%" if b is not None else "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="content")

    # 5. Why  6. What would change this view
    st.markdown("##### Why")
    for r in reasons:
        st.markdown(f"- {md(r)}")
    if triggers:
        st.markdown("##### What would change this view")
        for t in triggers:
            st.markdown(f"- {md(t)}")
    if source == "template":
        st.caption("Reasons generated from the scoring rules.")

    # 7. Disclaimer and feedback
    st.divider()
    st.caption(DISCLAIMER)
    vote = st.feedback("thumbs", key=f"fb_{symbol}")
    if vote is not None and st.session_state.get(f"fb_saved_{symbol}") != vote:
        DATA_DIR.mkdir(exist_ok=True)
        new = not FEEDBACK_FILE.exists()
        with FEEDBACK_FILE.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "symbol", "verdict", "score", "helpful"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), symbol, rec.verdict,
                        round(rec.score, 1), bool(vote)])
        st.session_state[f"fb_saved_{symbol}"] = vote
        st.toast("Thanks for the feedback.")


def allocation_pie(table: pd.DataFrame, total: float):
    share = table.set_index("Symbol")["Value $"] / total * 100
    big = share[share >= 2].sort_values(ascending=False)
    small = share[share < 2]
    labels = list(big.index) + (["Other"] if len(small) else [])
    values = list(table.set_index("Symbol").loc[big.index, "Value $"]) + (
        [table.set_index("Symbol").loc[small.index, "Value $"].sum()] if len(small) else [])
    hover = ["%{label}<br>$%{value:,.2f}<br>%{percent}"] * len(big)
    if len(small):
        hover.append("Other: " + ", ".join(f"{s} {v:.1f}%" for s, v in small.items())
                     + "<br>$%{value:,.2f}<br>%{percent}")
    colors = PIE_COLORS[:len(big)] + (["#ADB5BD"] if len(small) else [])
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45, sort=False,
                           marker=dict(colors=colors),
                           textinfo="label+percent", hovertemplate=[h + "<extra></extra>" for h in hover]))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


@st.fragment(run_every=60 if market_open() else None)
def portfolio_tab() -> None:
    txns = st.session_state.txns
    if txns is None:
        st.info("Upload your transactions on the Upload tab to see your portfolio.")
        return
    with st.spinner("Loading prices..."):
        try:
            _, _, positions, table, fetched_at, stale = load_portfolio(txns)
        except Exception as e:
            st.error(f"We couldn't load market data: {e}")
            return
    if table.empty:
        st.info("You have no open positions.")
        return

    summary = portfolio_summary(table, positions)
    c = st.columns(4)
    c[0].metric("Total value", money(summary["total_value"]))
    c[1].metric("Today's change", money(summary["day_change"], True),
                f"{summary['day_change_pct']:+.2f}%")
    c[2].metric("Total gain", money(summary["total_gain"], True),
                f"{summary['total_gain_pct']:+.2f}%")
    if market_open():
        c[3].metric("Last updated", f"{fetched_at:%I:%M %p} ET", "Delayed 15 min",
                    delta_color="off")
    else:
        c[3].metric("Last updated", "Market closed", f"Prices as of {fetched_at:%b %d, %I:%M %p} ET",
                    delta_color="off")
    if stale:
        st.warning(f"Live prices unavailable for {', '.join(stale)}; showing last known prices.")

    st.markdown("##### Allocation")
    left, right = st.columns([1, 1])
    with left:
        sel = st.plotly_chart(allocation_pie(table, summary["total_value"]), width="stretch",
                              config={"displayModeBar": False}, on_select="rerun",
                              selection_mode="points", key="pie")
        picked = None
        try:
            pts = sel.selection.points
            picked = pts[0].get("label") if pts else None
        except AttributeError:
            pass
    with right:
        alloc = table[["Symbol", "Value $"]].assign(
            **{"% of portfolio": table["Value $"] / summary["total_value"] * 100}
        ).sort_values("Value $", ascending=False)
        st.dataframe(alloc, hide_index=True, width="stretch", column_config={
            "Value $": st.column_config.NumberColumn(format="$%,.2f"),
            "% of portfolio": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                              max_value=100),
        })

    h1, h2, h3 = st.columns([4, 1.4, 1], vertical_alignment="center")
    h1.markdown("##### Holdings")
    sort_col = h2.selectbox("Sort by", SORT_COLUMNS, key="sort_col")
    h3.toggle("Descending", key="sort_desc")
    view = table.sort_values(sort_col, ascending=not st.session_state.sort_desc, kind="stable")

    with st.container(border=True):
        headers = ["Symbol", "Actions", "Last Price $", "Change $", "Change %", "Qty #",
                   "Price Paid $", "Day's Gain $", "Total Gain $", "Total Gain %", "Value $"]
        hdr = st.columns(WIDTHS, vertical_alignment="bottom")
        for col, name in zip(hdr, headers):
            col.markdown(f"**{md(name)}**")
        for r in view.to_dict("records"):
            cols = st.columns(WIDTHS, vertical_alignment="center")
            sym = r["Symbol"]
            cols[0].markdown(f":violet-background[**{sym}**]" if sym == picked else f"**{sym}**")
            if cols[1].button("Recommendation", key=f"rec_{sym}", type="primary",
                               icon=":material/insights:"):
                recommendation_dialog(sym)
            cols[2].markdown(f"{r['Last Price $']:,.2f}")
            cols[3].markdown(tone(f"{r['Change $']:+,.2f}", r["Change $"]))
            cols[4].markdown(tone(f"{r['Change %']:+.2f}%", r["Change %"]))
            cols[5].markdown(f"{r['Qty #']:,.4g}")
            cols[6].markdown(f"{r['Price Paid $']:,.2f}")
            day_gain = r["Day's Gain $"]
            cols[7].markdown(tone(f"{day_gain:+,.2f}", day_gain))
            cols[8].markdown(tone(f"{r['Total Gain $']:+,.2f}", r["Total Gain $"]))
            cols[9].markdown(tone(f"{r['Total Gain %']:+.2f}%", r["Total Gain %"]))
            cols[10].markdown(f"{r['Value $']:,.2f}")
        tot = st.columns(WIDTHS)
        tot[0].markdown("**Total**")
        day_total = table["Day's Gain $"].sum()
        tot[7].markdown(tone(f"**{day_total:+,.2f}**", day_total))
        tot[8].markdown(tone(f"**{table['Total Gain $'].sum():+,.2f}**", table["Total Gain $"].sum()))
        tot[10].markdown(f"**{table['Value $'].sum():,.2f}**")

    with st.expander("Lot history"):
        sym = st.selectbox("Symbol", list(view["Symbol"]), key="lot_sym")
        lots = pd.DataFrame(positions[sym].lots)[["Date", "Action", "Quantity", "Price", "Fees", "Amount"]]
        st.dataframe(lots, hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab 3: history

def history_tab() -> None:
    txns = st.session_state.txns
    if txns is None:
        st.info("Upload your transactions on the Upload tab to see your history.")
        return
    key = st.segmented_control("Range", list(RANGES), default="1Y", key="range",
                               label_visibility="collapsed") or "1Y"
    with st.spinner("Building history..."):
        try:
            adjusted, closes, _, table, _, _ = load_portfolio(txns)
        except Exception as e:
            st.error(f"We couldn't load market data: {e}")
            return
        # End on the same live prices the Portfolio tab shows.
        today = pd.Timestamp(date.today())
        if len(closes) and closes.index[-1] == today:
            closes = closes.copy()
            for sym, last in zip(table["Symbol"], table["Last Price $"]):
                closes.loc[today, sym] = last
        held = [s for s in adjusted["Symbol"].unique() if s in closes]
        daily = daily_values(adjusted, closes[held])
        view = range_view(daily, key)

    if view.bars.empty:
        st.info("No history yet.")
        return

    c = st.columns(4)
    c[0].metric("Total portfolio value", money(view.end_value),
                f"Start of range: {money(view.start_value)}", delta_color="off",
                delta_arrow="off")
    c[1].metric("Total investment", money(view.investment), "Buys in this range, incl. fees",
                delta_color="off", delta_arrow="off")
    c[2].metric("Total sales", money(view.sales), "Sells in this range, net of fees",
                delta_color="off", delta_arrow="off")
    c[3].metric("Return", money(view.return_usd, True),
                f"{view.return_pct:+.2f}% time-weighted" if view.return_pct is not None else None)

    s = st.columns(4)
    s[0].caption(f"Start value: **{md(money(view.start_value))}** · End value: **{md(money(view.end_value))}**")
    if view.best:
        s[1].caption(f"Best period: **{view.best[0]}** ({md(money(view.best[1], True))})")
        s[2].caption(f"Worst period: **{view.worst[0]}** ({md(money(view.worst[1], True))})")
    divs = view.bars["dividends"].sum()
    if divs:
        s[3].caption(f"Dividends received: **{md(money(divs))}** (not in Return)")

    bars = view.bars
    custom = list(zip(bars.index.strftime("%b %d, %Y"),
                      bars["change"], bars["change_pct"].fillna(0), bars["flow"], bars["dividends"]))
    fig = go.Figure(go.Bar(
        x=bars["label"], y=bars["value"], marker_color=BAR_COLOR, width=0.18,
        customdata=custom,
        hovertemplate=("<b>%{customdata[0]}</b><br>Value: $%{y:,.2f}<br>"
                       "Change: $%{customdata[1]:+,.2f} (%{customdata[2]:+.2f}%)<br>"
                       "Net cash added: $%{customdata[3]:+,.2f}<br>"
                       "Dividends: $%{customdata[4]:,.2f}<extra></extra>"),
    ))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), bargap=0.6,
                      yaxis=dict(rangemode="tozero", tickprefix="$", nticks=5, gridcolor="#bbb"),
                      xaxis=dict(type="category", showgrid=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if view.started_on:
        st.caption(f"Portfolio started on {view.started_on:%b %d, %Y}.")


# ---------------------------------------------------------------- layout

# Recommendation CTA: blue, unlike the purple theme colour. White on #1971C2 meets WCAG AA.
st.markdown("""<style>
div[class*="st-key-rec_"] button {
    background-color: #1971C2; border-color: #1971C2; color: #FFFFFF; font-weight: 600;
}
div[class*="st-key-rec_"] button:hover { background-color: #1864AB; border-color: #1864AB; color: #FFFFFF; }
div[class*="st-key-rec_"] button:active { background-color: #145591; border-color: #145591; }
div[class*="st-key-rec_"] button:focus-visible { box-shadow: 0 0 0 3px rgba(25, 113, 194, 0.4); }
</style>""", unsafe_allow_html=True)

st.title("Stock Portfolio Analyzer")
default_tab = "Portfolio" if st.session_state.txns is not None else "Upload"
upload, portfolio, history = st.tabs(["Upload", "Portfolio", "History"], key="main_tab",
                                     default=default_tab, on_change="rerun")
with upload:
    upload_tab()
with portfolio:
    portfolio_tab()
with history:
    history_tab()
st.caption("US markets · Quotes delayed 15 minutes · Market data from Yahoo Finance. " + DISCLAIMER)
