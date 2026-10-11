"""Portfolio value over time and per-range summaries (PRD: Tab 3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from portfolio.calculations import sort_transactions

RANGES = {
    "1W": "1 week",
    "1M": "1 month",
    "1Q": "1 quarter",
    "1Y": "1 year",
}


def daily_values(txns: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Portfolio value and cash flows for every trading day from the first transaction.

    `closes` is indexed by trading date with one column of closing prices per symbol.
    Columns returned: value, investment (buys incl. fees), sales (sells net of fees),
    flow (investment - sales) and dividends. A trade on a non-trading day counts on the
    next trading day.
    """
    if txns.empty or closes.empty:
        return pd.DataFrame(columns=["value", "investment", "sales", "flow", "dividends"])

    closes = closes.sort_index().ffill()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    first = pd.Timestamp(txns["Date"].min())
    days = closes.index[closes.index >= first]
    if len(days) == 0:
        return pd.DataFrame(columns=["value", "investment", "sales", "flow", "dividends"])

    shares = pd.DataFrame(0.0, index=days, columns=closes.columns)
    cash = pd.DataFrame(0.0, index=days, columns=["investment", "sales", "dividends"])

    for t in sort_transactions(txns).to_dict("records"):
        pos = min(days.searchsorted(pd.Timestamp(t["Date"])), len(days) - 1)
        day = days[pos]
        if t["Action"] == "BUY":
            shares.loc[day:, t["Symbol"]] += t["Quantity"]
            cash.loc[day, "investment"] += t["Quantity"] * t["Price"] + t["Fees"]
        elif t["Action"] == "SELL":
            shares.loc[day:, t["Symbol"]] -= t["Quantity"]
            cash.loc[day, "sales"] += t["Quantity"] * t["Price"] - t["Fees"]
        else:
            cash.loc[day, "dividends"] += t["Amount"]

    value = (shares * closes.loc[days, shares.columns].fillna(0.0)).sum(axis=1)
    out = cash.assign(value=value)
    out["flow"] = out["investment"] - out["sales"]
    return out[["value", "investment", "sales", "flow", "dividends"]]


@dataclass
class RangeView:
    bars: pd.DataFrame  # index = bar date; label, value, change, change_pct, flow, dividends
    start_value: float
    end_value: float
    investment: float
    sales: float
    return_usd: float
    return_pct: float | None
    best: tuple[str, float] | None
    worst: tuple[str, float] | None
    started_on: date | None  # set when the portfolio began inside this range


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _label(ts: pd.Timestamp, key: str, first: bool) -> str:
    if key == "1W":
        return f"{ts.strftime('%a')} {ts.day}"
    if key == "1Y":
        name = MONTHS[ts.month - 1]
        return f"{name} '{ts.strftime('%y')}" if first or ts.month == 1 else name
    return f"{MONTHS[ts.month - 1]} {ts.day}"


def _periods(daily: pd.DataFrame, key: str) -> pd.DataFrame:
    """Aggregate daily rows to one row per bar: last value, summed flows."""
    agg = {"value": "last", "investment": "sum", "sales": "sum", "flow": "sum", "dividends": "sum"}
    if key in ("1W", "1M"):
        return daily
    period = daily.index.to_period("W-FRI" if key == "1Q" else "M")
    # Each bar is dated by its last trading day, so the current period ends today.
    g = daily.assign(_day=daily.index).groupby(period).agg({**agg, "_day": "last"})
    g.index = pd.DatetimeIndex(g.pop("_day"))
    return g


def range_view(daily: pd.DataFrame, key: str, today: date | None = None) -> RangeView:
    today = today or date.today()
    periods = _periods(daily, key)
    t = pd.Timestamp(today)
    if key == "1W":
        bars = periods.tail(5)
    elif key == "1M":
        bars = periods[periods.index > t - pd.DateOffset(months=1)]
    elif key == "1Q":
        bars = periods.tail(13)
    else:
        bars = periods.tail(12)

    if bars.empty:
        return RangeView(bars, 0, 0, 0, 0, 0, None, None, None, None)

    earlier = periods[periods.index < bars.index[0]]
    start_value = float(earlier["value"].iloc[-1]) if len(earlier) else 0.0
    started_on = None if len(earlier) else daily.index[0].date()

    prev = pd.concat([pd.Series([start_value]), bars["value"].iloc[:-1]], ignore_index=True)
    prev.index = bars.index
    bars = bars.copy()
    bars["change"] = bars["value"] - prev - bars["flow"]
    base = prev + bars["investment"]
    bars["change_pct"] = (bars["change"] / base.where(base > 0)) * 100
    bars["label"] = [_label(ts, key, i == 0) for i, ts in enumerate(bars.index)]

    # Time-weighted return over the daily rows in the range: new money is not growth.
    in_range = daily[daily.index > (earlier.index[-1] if len(earlier) else pd.Timestamp.min)]
    in_range = in_range[in_range.index <= bars.index[-1]]
    growth, prev_v = 1.0, start_value
    for v, inv, sales in zip(in_range["value"], in_range["investment"], in_range["sales"]):
        start = prev_v + inv  # treat buys as made at the start of the day
        if start > 0:
            growth *= (v + sales) / start
        prev_v = v
    has_return = (start_value > 0) or (in_range["investment"].sum() > 0)

    investment = float(bars["investment"].sum())
    sales = float(bars["sales"].sum())
    end_value = float(bars["value"].iloc[-1])
    best_i, worst_i = bars["change"].idxmax(), bars["change"].idxmin()
    return RangeView(
        bars=bars,
        start_value=start_value,
        end_value=end_value,
        investment=investment,
        sales=sales,
        return_usd=end_value - start_value - investment + sales,
        return_pct=(growth - 1) * 100 if has_return else None,
        best=(bars.loc[best_i, "label"], float(bars.loc[best_i, "change"])) if len(bars) > 1 else None,
        worst=(bars.loc[worst_i, "label"], float(bars.loc[worst_i, "change"])) if len(bars) > 1 else None,
        started_on=started_on,
    )
