"""Average-cost positions and the holdings table (PRD: Tab 2, Key calculations)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

ACTION_ORDER = {"BUY": 0, "DIVIDEND": 1, "SELL": 2}


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    cost: float = 0.0  # cost basis of the shares still held, fees included
    realised_gain: float = 0.0
    dividends: float = 0.0
    lots: list[dict] = field(default_factory=list)  # every transaction for this symbol

    @property
    def avg_cost(self) -> float:
        return self.cost / self.qty if self.qty > 0 else 0.0


def sort_transactions(txns: pd.DataFrame) -> pd.DataFrame:
    """Date order, with buys before sells on the same day."""
    return txns.assign(_o=txns["Action"].map(ACTION_ORDER)).sort_values(
        ["Date", "_o", "Row"], kind="stable").drop(columns="_o")


def build_positions(txns: pd.DataFrame) -> dict[str, Position]:
    """Replay transactions with the average-cost method.

    Buys add qty x price + fees to cost; sells remove shares at the current average
    cost, so the average is unchanged by a sale. A position that goes to zero resets.
    """
    positions: dict[str, Position] = {}
    for t in sort_transactions(txns).to_dict("records"):
        p = positions.setdefault(t["Symbol"], Position(t["Symbol"]))
        p.lots.append(t)
        if t["Action"] == "BUY":
            p.cost += t["Quantity"] * t["Price"] + t["Fees"]
            p.qty += t["Quantity"]
        elif t["Action"] == "SELL":
            avg = p.avg_cost
            proceeds = t["Quantity"] * t["Price"] - t["Fees"]
            p.realised_gain += proceeds - avg * t["Quantity"]
            p.qty -= t["Quantity"]
            p.cost -= avg * t["Quantity"]
            if p.qty <= 1e-9:
                p.qty, p.cost = 0.0, 0.0
        else:
            p.dividends += t["Amount"]
    return positions


HOLDING_COLUMNS = ["Symbol", "Last Price $", "Change $", "Change %", "Qty #", "Price Paid $",
                   "Day's Gain $", "Total Gain $", "Total Gain %", "Value $"]


def holdings_table(positions: dict[str, Position], quotes: dict[str, dict],
                   today: date | None = None) -> pd.DataFrame:
    """One row per open position, in the PRD's column order (Actions is added by the UI).

    `quotes` maps symbol -> {"last": float, "prev_close": float}.
    """
    today = today or date.today()
    rows = []
    for sym, p in sorted(positions.items()):
        if p.qty <= 0:
            continue
        q = quotes.get(sym)
        if not q:
            continue
        last, prev = q["last"], q["prev_close"]
        # Shares bought today gain from their purchase price, not yesterday's close.
        bought_today = [l for l in p.lots if l["Action"] == "BUY" and l["Date"] == today]
        today_qty = min(sum(l["Quantity"] for l in bought_today), p.qty)
        day_gain = (last - prev) * (p.qty - today_qty) + sum(
            (last - l["Price"]) * l["Quantity"] for l in bought_today)
        total_gain = (last - p.avg_cost) * p.qty
        rows.append({
            "Symbol": sym,
            "Last Price $": last,
            "Change $": last - prev,
            "Change %": (last - prev) / prev * 100 if prev else 0.0,
            "Qty #": p.qty,
            "Price Paid $": p.avg_cost,
            "Day's Gain $": day_gain,
            "Total Gain $": total_gain,
            "Total Gain %": total_gain / p.cost * 100 if p.cost else 0.0,
            "Value $": last * p.qty,
        })
    return pd.DataFrame(rows, columns=HOLDING_COLUMNS)


def portfolio_summary(table: pd.DataFrame, positions: dict[str, Position]) -> dict:
    """Summary-bar tiles: total value, today's change, total gain."""
    value = float(table["Value $"].sum())
    day = float(table["Day's Gain $"].sum())
    cost = sum(positions[s].cost for s in table["Symbol"])
    prior = value - day
    return {
        "total_value": value,
        "day_change": day,
        "day_change_pct": day / prior * 100 if prior else 0.0,
        "total_gain": value - cost,
        "total_gain_pct": (value - cost) / cost * 100 if cost else 0.0,
    }


def apply_splits(txns: pd.DataFrame, splits: dict[str, pd.Series]) -> pd.DataFrame:
    """Restate pre-split trades in post-split shares so they match split-adjusted prices.

    `splits` maps symbol -> Series of split ratios indexed by date (e.g. 4.0 for 4:1).
    """
    if txns.empty or not splits:
        return txns
    out = txns.copy()
    for sym, series in splits.items():
        for split_date, ratio in series.items():
            if not ratio or ratio == 1:
                continue
            d = split_date.date() if hasattr(split_date, "date") else split_date
            mask = (out["Symbol"] == sym) & (out["Date"] < d) & (out["Action"] != "DIVIDEND")
            out.loc[mask, "Quantity"] *= ratio
            out.loc[mask, "Price"] /= ratio
    return out
