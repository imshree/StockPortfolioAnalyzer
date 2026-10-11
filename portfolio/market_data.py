"""US market data from Yahoo Finance via yfinance (quotes are ~15 minutes delayed)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

BENCHMARK = "SPY"

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "yfinance"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _download(symbols: list[str], **kwargs) -> pd.DataFrame:
    df = yf.download(symbols, auto_adjust=False, actions=True, progress=False,
                     group_by="column", multi_level_index=True, **kwargs)
    return df if df is not None else pd.DataFrame()


def daily_prices(symbols: Iterable[str], start: date) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Daily closes (split-adjusted) from `start`, plus each symbol's split history.

    Batched in one request; any symbol that comes back empty (yfinance's cache can
    fail under concurrency) is retried on its own.
    """
    symbols = sorted(set(symbols))
    if not symbols:
        return pd.DataFrame(), {}
    df = _download(symbols, start=start.isoformat(), threads=True)
    closes = df["Close"].reindex(columns=symbols) if not df.empty else pd.DataFrame(columns=symbols)
    splits_df = df["Stock Splits"] if "Stock Splits" in df else pd.DataFrame()

    for sym in [s for s in symbols if closes.get(s) is None or closes[s].isna().all()]:
        one = _download([sym], start=start.isoformat(), threads=False)
        if not one.empty and one["Close"][sym].notna().any():
            closes = closes.drop(columns=sym).join(one["Close"][[sym]], how="outer")
            if "Stock Splits" in one:
                splits_df = splits_df.drop(columns=sym, errors="ignore").join(
                    one["Stock Splits"][[sym]], how="outer")

    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    splits = {}
    for sym in splits_df.columns if not splits_df.empty else []:
        s = splits_df[sym].dropna()
        s = s[s > 0]
        if not s.empty:
            s.index = pd.to_datetime(s.index).tz_localize(None)
            splits[sym] = s
    return closes[symbols].sort_index(), splits


def known_symbols(symbols: Iterable[str]) -> set[str]:
    """The subset of `symbols` that Yahoo Finance has recent prices for."""
    symbols = sorted(set(symbols))
    if not symbols:
        return set()
    df = _download(symbols, period="10d", threads=True)
    found = set()
    if not df.empty:
        found = {s for s in symbols if s in df["Close"] and df["Close"][s].notna().any()}
    for sym in set(symbols) - found:  # retry individually before calling it unknown
        one = _download([sym], period="10d", threads=False)
        if not one.empty and one["Close"][sym].notna().any():
            found.add(sym)
    return found


def _quote(sym: str) -> tuple[str, dict | None]:
    try:
        fi = yf.Ticker(sym).fast_info
        last, prev = fi["lastPrice"], fi["previousClose"]
        if last and prev:
            return sym, {"last": float(last), "prev_close": float(prev)}
    except Exception:
        pass
    return sym, None


def quotes(symbols: Iterable[str]) -> dict[str, dict]:
    """Latest (delayed) price and previous close per symbol."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        return {s: q for s, q in pool.map(_quote, sorted(set(symbols))) if q}


def fundamentals(symbol: str) -> dict:
    """Name, sector, valuation and analyst data used by the recommendation engine."""
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    out = {
        "name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg": info.get("trailingPegRatio") or info.get("pegRatio"),
        "rating_key": info.get("recommendationKey"),
        "rating_mean": info.get("recommendationMean"),
        "analysts": info.get("numberOfAnalystOpinions"),
        "earnings_surprise_pct": None,
    }
    try:
        eh = t.earnings_history
        if eh is not None and not eh.empty and "surprisePercent" in eh:
            val = eh.sort_index()["surprisePercent"].dropna()
            if not val.empty:
                out["earnings_surprise_pct"] = float(val.iloc[-1]) * 100
    except Exception:
        pass
    return out
