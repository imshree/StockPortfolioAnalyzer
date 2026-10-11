from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio.calculations import apply_splits, build_positions, holdings_table, portfolio_summary
from portfolio.explain import numbers_grounded
from portfolio.history import daily_values, range_view
from portfolio.recommend import recommend
from portfolio.transactions import SAMPLE_CSV, FileRejected, parse_transactions

TODAY = date(2026, 10, 9)


def parse(text, **kw):
    return parse_transactions(text.encode(), "t.csv", today=TODAY, **kw)


# ---------------------------------------------------------------- CSV parsing

def test_sample_csv_imports_cleanly():
    r = parse(SAMPLE_CSV)
    assert r.rows_read == 4 and r.rows_imported == 4 and not r.errors
    assert r.holdings_found == 2


def test_columns_matched_case_insensitively_in_any_order():
    r = parse("symbol,ACTION,date,price,quantity\nAAPL,buy,01/05/2026,100,2\n")
    assert r.rows_imported == 1
    t = r.transactions.iloc[0]
    assert t["Date"] == date(2026, 1, 5) and t["Action"] == "BUY" and t["Fees"] == 0


@pytest.mark.parametrize("data,name,msg", [
    (b"a,b\n1,2\n", "t.txt", "Please upload a .csv file."),
    (b"", "t.csv", "empty"),
    (b"Date,Symbol,Action,Quantity\n2026-01-05,AAPL,BUY,1\n", "t.csv", "missing the column: Price"),
])
def test_file_rejections(data, name, msg):
    with pytest.raises(FileRejected, match=msg):
        parse_transactions(data, name, today=TODAY)


def test_bad_rows_are_skipped_with_reasons_and_the_rest_import():
    csv = ("Date,Symbol,Action,Quantity,Price,Fees,Amount\n"
           "2026-01-05,AAPL,BUY,ten,100,0,\n"        # row 2
           "2027-01-05,AAPL,BUY,1,100,0,\n"          # row 3 future
           "2026-01-05,AAPL,HOLD,1,100,0,\n"         # row 4 bad action
           "2026-01-05,XYZQ,BUY,1,100,0,\n"          # row 5 unknown symbol
           "2026-01-05,AAPL,BUY,5,100,0,\n"          # row 6 ok
           "2026-02-05,AAPL,SELL,9,120,0,\n"         # row 7 oversold
           "2026-01-05,AAPL,BUY,5,100,0,\n"          # row 8 duplicate of 6
           "2026-03-01,AAPL,DIVIDEND,,,,\n")         # row 9 missing amount
    r = parse(csv, known_symbols=lambda syms: set(syms) - {"XYZQ"})
    reasons = {e.row: e.reason for e in r.errors}
    assert reasons[2] == "Quantity 'ten' is not a positive number."
    assert "in the future" in reasons[3]
    assert "BUY, SELL or DIVIDEND" in reasons[4]
    assert reasons[5] == "We couldn't find symbol XYZQ."
    assert reasons[7].startswith("Oversold")
    assert "Amount is missing" in reasons[9]
    assert [w.row for w in r.warnings] == [8]
    assert r.rows_imported == 1


def test_sell_on_same_day_as_buy_is_not_oversold():
    r = parse("Date,Symbol,Action,Quantity,Price\n2026-01-05,AAPL,SELL,1,110\n2026-01-05,AAPL,BUY,1,100\n")
    assert not r.errors and r.rows_imported == 2


# ---------------------------------------------------------------- average cost

def txns(*rows):
    return pd.DataFrame([{"Date": d, "Symbol": s, "Action": a, "Quantity": q, "Price": p,
                          "Fees": f, "Amount": amt, "Row": i}
                         for i, (d, s, a, q, p, f, amt) in enumerate(rows)])


def test_average_cost_includes_fees_and_is_unchanged_by_sells():
    t = txns((date(2026, 1, 1), "A", "BUY", 10, 100, 10, 0),
             (date(2026, 2, 1), "A", "BUY", 10, 120, 10, 0),
             (date(2026, 3, 1), "A", "SELL", 5, 150, 5, 0),
             (date(2026, 4, 1), "A", "DIVIDEND", 0, 0, 0, 7.5))
    p = build_positions(t)["A"]
    assert p.qty == 15
    assert p.avg_cost == pytest.approx(2220 / 20)  # 111.00
    assert p.realised_gain == pytest.approx((5 * 150 - 5) - 5 * 111)
    assert p.dividends == 7.5


def test_holdings_table_columns_and_totals():
    t = txns((date(2026, 1, 1), "A", "BUY", 10, 100, 0, 0),
             (date(2026, 1, 1), "B", "BUY", 4, 50, 0, 0))
    pos = build_positions(t)
    table = holdings_table(pos, {"A": {"last": 110, "prev_close": 105},
                                 "B": {"last": 45, "prev_close": 50}}, today=TODAY)
    a = table.set_index("Symbol").loc["A"]
    assert a["Change $"] == 5 and a["Day's Gain $"] == 50
    assert a["Total Gain $"] == 100 and a["Total Gain %"] == pytest.approx(10)
    s = portfolio_summary(table, pos)
    assert s["total_value"] == table["Value $"].sum() == 1280
    assert s["total_gain"] == pytest.approx(1280 - 1200)


def test_days_gain_for_shares_bought_today_uses_purchase_price():
    t = txns((date(2026, 1, 1), "A", "BUY", 10, 100, 0, 0),
             (TODAY, "A", "BUY", 10, 108, 0, 0))
    table = holdings_table(build_positions(t), {"A": {"last": 110, "prev_close": 105}}, today=TODAY)
    assert table.iloc[0]["Day's Gain $"] == pytest.approx(10 * 5 + 10 * 2)


def test_split_restates_earlier_trades():
    t = txns((date(2026, 1, 1), "A", "BUY", 10, 400, 0, 0))
    adj = apply_splits(t, {"A": pd.Series([4.0], index=[pd.Timestamp("2026-06-01")])})
    assert adj.iloc[0]["Quantity"] == 40 and adj.iloc[0]["Price"] == 100


# ---------------------------------------------------------------- history

def closes_for(prices: dict, start="2025-09-01", end="2026-10-09"):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({s: f(np.arange(len(idx))) for s, f in prices.items()}, index=idx)


def test_bar_value_is_shares_times_close():
    closes = closes_for({"A": lambda i: 100 + i * 0.0})
    t = txns((date(2026, 1, 5), "A", "BUY", 10, 100, 0, 0))
    daily = daily_values(t, closes)
    assert daily.loc["2026-03-02", "value"] == pytest.approx(1000)
    assert daily.index[0] == pd.Timestamp("2026-01-05")


def test_range_totals_investment_sales_and_return():
    # Flat price 100: a new buy alone must not show as a gain.
    closes = closes_for({"A": lambda i: np.full(len(i), 100.0)})
    t = txns((date(2025, 12, 1), "A", "BUY", 10, 100, 0, 0),
             (date(2026, 9, 21), "A", "BUY", 5, 100, 0, 0),
             (date(2026, 9, 28), "A", "SELL", 3, 100, 1, 0))
    view = range_view(daily_values(t, closes), "1M", today=TODAY)
    assert view.start_value == 1000
    assert view.investment == 500
    assert view.sales == 299
    assert view.end_value == 1200
    assert view.return_usd == pytest.approx(-1)  # only the sale fee
    assert abs(view.return_pct) < 0.1


def test_range_bar_counts():
    closes = closes_for({"A": lambda i: 100 + i * 0.1})
    t = txns((date(2025, 9, 2), "A", "BUY", 1, 100, 0, 0))
    daily = daily_values(t, closes)
    assert len(range_view(daily, "1W", TODAY).bars) == 5
    assert len(range_view(daily, "1Q", TODAY).bars) == 13
    y = range_view(daily, "1Y", TODAY)
    assert len(y.bars) == 12
    assert y.bars["label"].iloc[0] == "Nov '25" and "Jan '26" in list(y.bars["label"])
    assert y.bars.index[-1] == pd.Timestamp("2026-10-09")
    assert y.return_pct == pytest.approx((y.end_value / y.start_value - 1) * 100)


def test_portfolio_started_inside_range():
    closes = closes_for({"A": lambda i: np.full(len(i), 50.0)})
    view = range_view(daily_values(txns((date(2026, 7, 15), "A", "BUY", 2, 50, 0, 0)), closes),
                      "1Y", TODAY)
    assert view.started_on == date(2026, 7, 15)
    assert len(view.bars) == 4


# ---------------------------------------------------------------- recommendation

def series(f, n=300):
    idx = pd.bdate_range(end="2026-10-09", periods=n)
    return pd.Series(f(np.arange(n)), index=idx, dtype=float)


GOOD = {"peg": 0.8, "trailing_pe": 30, "forward_pe": 24, "rating_key": "buy", "analysts": 30,
        "earnings_surprise_pct": 4.0}
BAD = {"peg": 3.5, "trailing_pe": 20, "forward_pe": 26, "rating_key": "sell", "analysts": 12,
       "earnings_surprise_pct": -6.0}


def wavy_up(i):  # rising with pullbacks so RSI stays in a healthy range
    return 100 + i * 0.3 + 4 * np.sin(i / 3)


def test_strong_stock_small_position_is_buy_more():
    s, bench = series(wavy_up), series(lambda i: 100 + i * 0.05)
    rec = recommend("A", s, bench, s.iloc[-1], GOOD, weight_pct=8, gain_pct=20)
    assert rec.verdict == "Buy more" and rec.score >= 30
    assert 3 <= len(rec.reasons) <= 5 and 1 <= len(rec.triggers) <= 2


def test_concentrated_position_is_capped_at_hold():
    s, bench = series(wavy_up), series(lambda i: 100 + i * 0.05)
    rec = recommend("A", s, bench, s.iloc[-1], GOOD, weight_pct=35, gain_pct=20)
    assert rec.verdict == "Hold" and rec.capped
    assert "35.0%" in rec.reasons[0]


def test_weak_stock_is_sell():
    s, bench = series(lambda i: 200 - i * 0.3), series(lambda i: 100 + i * 0.05)
    rec = recommend("A", s, bench, s.iloc[-1], BAD, weight_pct=28, gain_pct=-30)
    assert rec.verdict == "Sell" and rec.confidence == "High"


def test_missing_data_does_not_crash():
    s = series(lambda i: 100 + i * 0.0, n=30)
    rec = recommend("A", s, None, 100.0, {}, weight_pct=10, gain_pct=0)
    assert rec.verdict in ("Buy more", "Hold", "Sell")


# ---------------------------------------------------------------- explanation guard

def test_numbers_must_come_from_the_input():
    src = "Price is 12.3% above its 200-day average of $180.50. Weight 31.0%."
    assert numbers_grounded("Price sits 12.3% over the 200-day average ($180.50).", src)
    assert not numbers_grounded("Price could reach $250 soon.", src)
    assert not numbers_grounded("Price is 12.9% above.", src)
