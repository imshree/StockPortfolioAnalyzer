# Stock Portfolio Analyzer

A local web app that turns a CSV of your stock trades into your current portfolio, its history over 1W / 1M / 1Q / 1Y, and a Buy more / Hold / Sell view for each holding. Built from the [PRD](https://claude.ai/code/artifact/19c57d72-1284-49d2-b913-17b987ea75fb).

## Run it

```bash
./run.sh
```

The app opens at http://localhost:8501. To try it straight away, upload `demo_transactions.csv`. Its trades are made up, but prices come from live market data.

**Optional:** to get Claude-written reasons in the Recommendation pop-up, set `ANTHROPIC_API_KEY` before you start the app. Without a key, the pop-up shows the same verdict with reasons generated from the scoring rules.

**Setup from scratch** (the `.venv` already exists in this folder):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`run.sh` sets `DYLD_LIBRARY_PATH` to Homebrew's `expat`. That works around a broken Homebrew Python 3.14 on this Mac, which otherwise can't load its XML module.

## Tabs

| Tab | What it does |
| --- | --- |
| Upload | Checks the CSV (Date, Symbol, Action, Quantity, Price, optional Fees and Amount), lists skipped rows with reasons, then imports. Data is saved to `data/transactions.csv` on this computer until you click **Clear my data**. |
| Portfolio | Summary tiles, allocation pie, and the holdings table in the reference column order. Each row has a **Recommendation** button. Prices refresh every 60 s during market hours. |
| History | Bar chart of portfolio value per day, week or month; total portfolio value, total investment, total sales and return for the selected range. |

## How the numbers work

- **Cost basis:** average cost including fees. A sale doesn't change the average.
- **Return $:** end value − start value − investment + sales. Money you add isn't counted as gain.
- **Return %:** time-weighted.
- **Stock splits:** trades before a split are restated in post-split shares.
- **Market data:** Yahoo Finance via `yfinance`, US-listed symbols, quotes about 15 minutes delayed.

## Recommendation engine

`portfolio/recommend.py` scores five signals from −100 to +100 and weights them:

| Signal | Weight | Uses |
| --- | --- | --- |
| Trend | 30% | Price vs 50- and 200-day averages, and how the two averages compare |
| Momentum | 20% | 1M and 3M return vs S&P 500 (SPY), 14-day RSI |
| Valuation | 20% | PEG ratio, forward vs trailing P/E |
| Position risk | 15% | Share of portfolio, drawdown with a falling trend |
| Analyst sentiment | 15% | Consensus rating, last earnings surprise |

A score of +30 or more is Buy more and −30 or less is Sell. A stock above 30% of the portfolio is capped at Hold.

`portfolio/explain.py` sends Claude Sonnet 5.5 the symbol, percentages and the scored facts. It never sends share counts or dollar cost. Claude rewords the reasons only. Its output is discarded if it contains any number that wasn't in its input. Server-side refusal fallbacks (`fallbacks: "default"`) are enabled.

## Differences from the PRD

- **Valuation:** the PRD compares P/E with the stock's 5-year average and its sector median. The free data source doesn't provide those, so this uses PEG ratio and forward vs trailing P/E.
- **Storage:** data is saved in a local file on this computer, not in browser storage. Streamlit runs a local Python server.
- **Sorting:** the holdings table sorts with a **Sort by** control rather than by clicking column headers, because Streamlit tables can't hold per-row buttons.
- **Not built (P1 items):** S&P 500 comparison on History, appending transactions, sector view on the pie, closed-positions view.

## Tests

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib .venv/bin/python -m pytest -q
```
