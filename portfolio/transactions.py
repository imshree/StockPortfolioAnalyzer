"""Parse and validate an uploaded transactions CSV (PRD: Tab 1)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Iterable

import pandas as pd

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10_000

REQUIRED_COLUMNS = ["Date", "Symbol", "Action", "Quantity", "Price"]
OPTIONAL_COLUMNS = ["Fees", "Amount"]
ACTIONS = {"BUY", "SELL", "DIVIDEND"}

TXN_COLUMNS = ["Date", "Symbol", "Action", "Quantity", "Price", "Fees", "Amount", "Row"]


class FileRejected(Exception):
    """The whole file is unusable (wrong type, empty, too big, missing a column)."""


@dataclass
class RowIssue:
    row: int  # 1-based row number as the user sees it in a spreadsheet (header = row 1)
    value: str
    reason: str

    @property
    def message(self) -> str:
        return f"Row {self.row}: {self.reason}"


@dataclass
class ParseResult:
    transactions: pd.DataFrame
    errors: list[RowIssue] = field(default_factory=list)  # skipped rows
    warnings: list[RowIssue] = field(default_factory=list)  # imported, but flagged
    rows_read: int = 0

    @property
    def rows_imported(self) -> int:
        return len(self.transactions)

    @property
    def holdings_found(self) -> int:
        if self.transactions.empty:
            return 0
        from portfolio.calculations import build_positions

        return sum(1 for p in build_positions(self.transactions).values() if p.qty > 0)

    @property
    def date_range(self) -> tuple[date, date] | None:
        if self.transactions.empty:
            return None
        return self.transactions["Date"].min(), self.transactions["Date"].max()


def _parse_date(text: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_number(text: str) -> float | None:
    if text == "":
        return None
    try:
        return float(text.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def parse_transactions(
    data: bytes,
    filename: str = "upload.csv",
    known_symbols: Callable[[Iterable[str]], set[str]] | None = None,
    today: date | None = None,
) -> ParseResult:
    """Validate a CSV and return the importable transactions plus row-level issues.

    `known_symbols` receives every distinct symbol and returns the ones that exist in
    market data; rows with any other symbol are skipped. Pass None to skip the check.
    """
    today = today or date.today()

    if not filename.lower().endswith(".csv"):
        raise FileRejected("Please upload a .csv file.")
    if len(data) == 0:
        raise FileRejected("The file is empty.")
    if len(data) > MAX_BYTES:
        raise FileRejected("The file is larger than 5 MB.")

    try:
        raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, skipinitialspace=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        raise FileRejected("We couldn't read this file as a CSV.")

    if raw.empty:
        raise FileRejected("The file has a header but no transactions.")
    if len(raw) > MAX_ROWS:
        raise FileRejected(f"The file has {len(raw):,} rows; the limit is {MAX_ROWS:,}.")

    # Match column names case-insensitively, in any order.
    by_lower = {c.strip().lower(): c for c in raw.columns}
    for col in REQUIRED_COLUMNS:
        if col.lower() not in by_lower:
            raise FileRejected(
                f"Your file is missing the column: {col}. Download the sample to see the format."
            )
    raw = raw.rename(columns={by_lower[c.lower()]: c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
                              if c.lower() in by_lower})
    for col in OPTIONAL_COLUMNS:
        if col not in raw.columns:
            raw[col] = ""

    result = ParseResult(transactions=pd.DataFrame(columns=TXN_COLUMNS), rows_read=len(raw))
    rows: list[dict] = []
    seen: set[tuple] = set()

    for i, rec in enumerate(raw.to_dict("records")):
        row_no = i + 2  # header is row 1
        rec = {k: str(v).strip() for k, v in rec.items()}

        d = _parse_date(rec["Date"])
        if d is None:
            result.errors.append(RowIssue(row_no, rec["Date"], f"Date '{rec['Date']}' is not a valid date."))
            continue
        if d > today:
            result.errors.append(RowIssue(row_no, rec["Date"], f"Date {rec['Date']} is in the future."))
            continue

        symbol = rec["Symbol"].upper()
        if not (1 <= len(symbol) <= 6) or not symbol.replace(".", "").replace("-", "").isalnum():
            result.errors.append(RowIssue(row_no, rec["Symbol"], f"Symbol '{rec['Symbol']}' is not valid."))
            continue

        action = rec["Action"].upper()
        if action not in ACTIONS:
            result.errors.append(
                RowIssue(row_no, rec["Action"], f"Action '{rec['Action']}' must be BUY, SELL or DIVIDEND.")
            )
            continue

        fees = _parse_number(rec["Fees"]) if rec["Fees"] else 0.0
        if fees is None or fees < 0:
            result.errors.append(RowIssue(row_no, rec["Fees"], f"Fees '{rec['Fees']}' is not a number."))
            continue

        qty = price = amount = None
        if action in ("BUY", "SELL"):
            qty = _parse_number(rec["Quantity"])
            if qty is None or qty <= 0:
                what = "is missing" if rec["Quantity"] == "" else f"'{rec['Quantity']}' is not a positive number"
                result.errors.append(RowIssue(row_no, rec["Quantity"], f"Quantity {what}."))
                continue
            price = _parse_number(rec["Price"])
            if price is None or price <= 0:
                what = "is missing" if rec["Price"] == "" else f"'{rec['Price']}' is not a positive number"
                result.errors.append(RowIssue(row_no, rec["Price"], f"Price {what}."))
                continue
        else:
            amount = _parse_number(rec["Amount"])
            if amount is None or amount <= 0:
                what = "is missing" if rec["Amount"] == "" else f"'{rec['Amount']}' is not a positive number"
                result.errors.append(RowIssue(row_no, rec["Amount"], f"Dividend Amount {what}."))
                continue

        key = (d, symbol, action, qty, price, fees, amount)
        if key in seen:
            result.warnings.append(
                RowIssue(row_no, "", "Exact duplicate of an earlier row; imported once.")
            )
            continue
        seen.add(key)

        rows.append({"Date": d, "Symbol": symbol, "Action": action, "Quantity": qty or 0.0,
                     "Price": price or 0.0, "Fees": fees, "Amount": amount or 0.0, "Row": row_no})

    if rows and known_symbols is not None:
        known = known_symbols(sorted({r["Symbol"] for r in rows}))
        kept = []
        for r in rows:
            if r["Symbol"] in known:
                kept.append(r)
            else:
                result.errors.append(RowIssue(r["Row"], r["Symbol"], f"We couldn't find symbol {r['Symbol']}."))
        rows = kept

    # Oversold check: walk each symbol in date order (buys before sells on the same day).
    order = {"BUY": 0, "DIVIDEND": 1, "SELL": 2}
    rows.sort(key=lambda r: (r["Date"], order[r["Action"]], r["Row"]))
    held: dict[str, float] = {}
    kept = []
    for r in rows:
        if r["Action"] == "BUY":
            held[r["Symbol"]] = held.get(r["Symbol"], 0.0) + r["Quantity"]
        elif r["Action"] == "SELL":
            have = held.get(r["Symbol"], 0.0)
            if r["Quantity"] > have + 1e-9:
                result.errors.append(RowIssue(
                    r["Row"], str(r["Quantity"]),
                    f"Oversold: selling {r['Quantity']:g} {r['Symbol']} but only {have:g} held on {r['Date']}.",
                ))
                continue
            held[r["Symbol"]] = have - r["Quantity"]
        kept.append(r)

    result.errors.sort(key=lambda e: e.row)
    result.transactions = pd.DataFrame(kept, columns=TXN_COLUMNS)
    return result


SAMPLE_CSV = """Date,Symbol,Action,Quantity,Price,Fees,Amount
2026-01-05,AAPL,BUY,10,185.20,0,
2026-02-10,MSFT,BUY,5,410.00,1.00,
2026-05-16,AAPL,DIVIDEND,,,,2.50
2026-07-22,AAPL,SELL,4,214.75,0,
"""
