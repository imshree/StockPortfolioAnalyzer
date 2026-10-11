"""Rule-based Buy more / Hold / Sell scoring (PRD: Recommendation engine).

Five signals, each scored -100..+100, are combined with fixed weights into one score.
The rules decide the verdict; the LLM layer (explain.py) only rewords the reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

WEIGHTS = {"Trend": 0.30, "Momentum": 0.20, "Valuation": 0.20, "Position risk": 0.15,
           "Analyst sentiment": 0.15}
BUY_AT, SELL_AT = 30, -30
CONCENTRATION_CAP_PCT = 30  # never "Buy more" above this share of the portfolio


@dataclass
class Signal:
    name: str
    score: float  # -100..+100
    facts: list[str] = field(default_factory=list)  # each states one number
    available: bool = True

    @property
    def weight(self) -> float:
        return WEIGHTS[self.name]

    @property
    def contribution(self) -> float:
        return self.weight * self.score


@dataclass
class Recommendation:
    symbol: str
    verdict: str  # "Buy more" | "Hold" | "Sell"
    confidence: str  # "High" | "Medium" | "Low"
    score: float
    signals: list[Signal]
    reasons: list[str]
    triggers: list[str]
    capped: bool = False


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def rsi(closes: pd.Series, period: int = 14) -> float | None:
    delta = closes.diff().dropna()
    if len(delta) < period:
        return None
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return 100 - 100 / (1 + gain / loss)


def _ret(s: pd.Series, days: int) -> float | None:
    s = s.dropna()
    if len(s) <= days:
        return None
    return (s.iloc[-1] / s.iloc[-1 - days] - 1) * 100


def trend_signal(closes: pd.Series, last: float) -> tuple[Signal, float | None, float | None]:
    closes = closes.dropna()
    ma50 = closes.tail(50).mean() if len(closes) >= 50 else None
    ma200 = closes.tail(200).mean() if len(closes) >= 200 else None
    comps, facts = [], []
    for label, ma in (("50-day", ma50), ("200-day", ma200)):
        if ma is None:
            continue
        gap = (last / ma - 1) * 100
        comps.append(1 if last > ma else -1)
        facts.append(f"Price is {abs(gap):.1f}% {'above' if gap >= 0 else 'below'} its {label} "
                     f"average of ${ma:,.2f}.")
    if ma50 is not None and ma200 is not None:
        comps.append(1 if ma50 > ma200 else -1)
        facts.append(f"The 50-day average (${ma50:,.2f}) is {'above' if ma50 > ma200 else 'below'} "
                     f"the 200-day average (${ma200:,.2f}).")
    if not comps:
        return Signal("Trend", 0, ["Not enough price history for moving averages."], False), None, None
    return Signal("Trend", _mean(comps) * 100, facts), ma50, ma200


def momentum_signal(closes: pd.Series, bench: pd.Series | None) -> Signal:
    comps, facts = [], []
    for label, days in (("1-month", 21), ("3-month", 63)):
        r = _ret(closes, days)
        b = _ret(bench, days) if bench is not None else None
        if r is None or b is None:
            continue
        comps.append(1 if r > b else -1)
        facts.append(f"{label.capitalize()} return is {r:+.1f}% vs {b:+.1f}% for the S&P 500.")
    r14 = rsi(closes.dropna())
    if r14 is not None:
        if 40 <= r14 <= 70:
            comps.append(1)
            facts.append(f"14-day RSI is {r14:.0f}, a healthy range (40 to 70).")
        elif r14 > 75:
            comps.append(-1)
            facts.append(f"14-day RSI is {r14:.0f}, which suggests the stock is overbought.")
        else:
            comps.append(0)
            facts.append(f"14-day RSI is {r14:.0f}, outside the 40 to 70 range.")
    if not comps:
        return Signal("Momentum", 0, ["Not enough price history to judge momentum."], False)
    return Signal("Momentum", _mean(comps) * 100, facts)


def valuation_signal(f: dict) -> Signal:
    comps, facts = [], []
    peg = f.get("peg")
    if peg and peg > 0:
        comps.append(1 if peg < 1 else 0 if peg <= 2 else -0.5 if peg <= 2.5 else -1)
        facts.append(f"PEG ratio is {peg:.2f} ({'cheap' if peg < 1 else 'fair' if peg <= 2 else 'expensive'} "
                     f"for its expected growth).")
    tpe, fpe = f.get("trailing_pe"), f.get("forward_pe")
    if tpe and fpe and tpe > 0 and fpe > 0:
        if fpe < tpe * 0.9:
            comps.append(1)
            facts.append(f"Forward P/E of {fpe:.1f} is below the trailing P/E of {tpe:.1f}, "
                         f"so earnings are expected to grow.")
        elif fpe > tpe * 1.1:
            comps.append(-1)
            facts.append(f"Forward P/E of {fpe:.1f} is above the trailing P/E of {tpe:.1f}, "
                         f"so earnings are expected to shrink.")
        else:
            comps.append(0)
            facts.append(f"P/E is {tpe:.1f} now and {fpe:.1f} on next year's earnings.")
    if not comps:
        return Signal("Valuation", 0, ["Valuation data is not available for this symbol."], False)
    return Signal("Valuation", _mean(comps) * 100, facts)


def position_signal(weight_pct: float, gain_pct: float, trend_score: float) -> Signal:
    if weight_pct < 15:
        score = 100.0
        facts = [f"This stock is {weight_pct:.1f}% of your portfolio, leaving room to add."]
    elif weight_pct <= 25:
        score = 0.0
        facts = [f"This stock is {weight_pct:.1f}% of your portfolio, a sizeable position."]
    else:
        score = -100.0
        facts = [f"This stock is {weight_pct:.1f}% of your portfolio, a concentration risk."]
    if gain_pct <= -25 and trend_score < 0:
        score = -100.0
        facts.append(f"Your position is down {abs(gain_pct):.1f}% from cost and the trend is falling.")
    return Signal("Position risk", score, facts)


RATING_SCORES = {"strong_buy": 1, "buy": 0.6, "hold": 0, "underperform": -0.6, "sell": -1,
                 "strong_sell": -1}


def sentiment_signal(f: dict) -> Signal:
    comps, facts = [], []
    key = (f.get("rating_key") or "").lower()
    if key in RATING_SCORES:
        comps.append(RATING_SCORES[key])
        n = f.get("analysts")
        who = f"{n} analysts" if n else "Analysts"
        facts.append(f"{who} rate it {key.replace('_', ' ')} on average.")
    surprise = f.get("earnings_surprise_pct")
    if surprise is not None:
        comps.append(1 if surprise > 0 else -1 if surprise < 0 else 0)
        facts.append(f"Last quarter's earnings {'beat' if surprise >= 0 else 'missed'} estimates "
                     f"by {abs(surprise):.1f}%.")
    if not comps:
        return Signal("Analyst sentiment", 0, ["No analyst coverage is available."], False)
    return Signal("Analyst sentiment", _mean(comps) * 100, facts)


def recommend(symbol: str, closes: pd.Series, bench: pd.Series | None, last: float,
              fundamentals: dict, weight_pct: float, gain_pct: float) -> Recommendation:
    trend, ma50, ma200 = trend_signal(closes, last)
    signals = [
        trend,
        momentum_signal(closes, bench),
        valuation_signal(fundamentals),
        position_signal(weight_pct, gain_pct, trend.score),
        sentiment_signal(fundamentals),
    ]
    score = sum(s.contribution for s in signals)
    verdict = "Buy more" if score >= BUY_AT else "Sell" if score <= SELL_AT else "Hold"
    capped = False
    if verdict == "Buy more" and weight_pct > CONCENTRATION_CAP_PCT:
        verdict, capped = "Hold", True

    def agrees(s: Signal) -> bool:
        if verdict == "Buy more":
            return s.score > 0
        if verdict == "Sell":
            return s.score < 0
        return abs(s.score) <= 33
    agree = sum(agrees(s) for s in signals if s.available)
    confidence = "High" if agree >= 4 else "Medium" if agree == 3 else "Low"

    # Reasons: strongest contributors first, one fact each, 3 to 5 in total.
    reasons: list[str] = []
    ranked = sorted((s for s in signals if s.available), key=lambda s: -abs(s.contribution))
    for s in ranked:
        reasons.append(s.facts[0])
    for s in ranked:
        reasons.extend(s.facts[1:2])
    if capped:
        reasons.insert(0, f"Capped at Hold: this stock is {weight_pct:.1f}% of your portfolio, "
                          f"above the {CONCENTRATION_CAP_PCT}% limit for adding more.")
    reasons = reasons[:5]

    triggers: list[str] = []
    level = ma200 or ma50
    label = "200-day" if ma200 else "50-day"
    if level:
        if verdict == "Sell":
            triggers.append(f"A close back above the {label} average of ${level:,.2f} would turn the "
                            f"trend signal positive.")
        else:
            triggers.append(f"A close below the {label} average of ${level:,.2f} would turn the "
                            f"trend signal negative.")
    if weight_pct > 25:
        triggers.append(f"Trimming the position from {weight_pct:.1f}% to under 25% of the "
                        f"portfolio would remove the concentration penalty.")
    elif verdict != "Buy more" and score > 0:
        triggers.append(f"The score is {score:.0f}; it needs {BUY_AT} or more for Buy more.")
    elif verdict != "Sell" and score <= 0:
        triggers.append(f"The score is {score:.0f}; at {SELL_AT} or below it becomes Sell.")

    return Recommendation(symbol, verdict, confidence, score, signals, reasons, triggers[:2], capped)
