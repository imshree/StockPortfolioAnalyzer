"""Plain-language reasons from Claude (PRD: Explanation layer).

The rules in recommend.py decide the verdict. Claude only rewrites the scored facts as
3-5 short reasons plus 1-2 triggers. Its output is rejected, and the templated reasons
are shown instead, if it contains any number that was not in its input, or if the call
fails for any reason (no API key, network, refusal).
"""

from __future__ import annotations

import json
import re

from portfolio.recommend import Recommendation

MODEL = "claude-sonnet-5-5"

SYSTEM = """You explain an automated stock recommendation to an individual investor.

You receive a verdict that has already been decided by fixed rules, the signals behind it, and \
draft reasons and triggers. Rewrite them in plain, friendly language for someone who is not a \
professional analyst.

Rules:
- Never change, question or soften the verdict.
- Write 3 to 5 reasons. Each reason is one sentence, ties to one signal, and states one number.
- Write 1 or 2 triggers: what would change this view.
- Use only numbers that appear in the input. Do not compute, round differently or add new numbers.
- No advice beyond the verdict, no predictions, no disclaimers (the app adds its own)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasons": {"type": "array", "items": {"type": "string"}},
        "triggers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasons", "triggers"],
    "additionalProperties": False,
}

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return [m.replace(",", "") for m in _NUM.findall(text)]


def numbers_grounded(output: str, source: str) -> bool:
    """True if every number in `output` appears in `source` at the same precision."""
    allowed = {float(n) for n in _numbers(source)}
    for n in _numbers(output):
        decimals = len(n.split(".")[1]) if "." in n else 0
        value = float(n)
        if not any(round(a, decimals) == value for a in allowed):
            return False
    return True


def build_input(rec: Recommendation) -> dict:
    # Privacy (PRD): send the symbol and percentages only, never share counts or dollar cost.
    return {
        "symbol": rec.symbol,
        "verdict": rec.verdict,
        "confidence": rec.confidence,
        "signals": [
            {"name": s.name, "weight_pct": round(s.weight * 100), "facts": s.facts}
            for s in rec.signals if s.available
        ],
        "draft_reasons": rec.reasons,
        "draft_triggers": rec.triggers,
    }


def explain(rec: Recommendation) -> tuple[list[str], list[str], str]:
    """Return (reasons, triggers, source) where source is "claude" or "template"."""
    try:
        import anthropic

        payload = json.dumps(build_input(rec), indent=2)
        client = anthropic.Anthropic(timeout=20.0, max_retries=1)
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": payload}],
        )
        if response.stop_reason != "end_turn":
            return rec.reasons, rec.triggers, "template"
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        reasons = [r.strip() for r in data["reasons"] if r.strip()][:5]
        triggers = [t.strip() for t in data["triggers"] if t.strip()][:2]
        if len(reasons) < min(3, len(rec.reasons)) or (rec.triggers and not triggers):
            return rec.reasons, rec.triggers, "template"
        if not numbers_grounded(" ".join(reasons + triggers), payload):
            return rec.reasons, rec.triggers, "template"
        return reasons, triggers, "claude"
    except Exception:
        return rec.reasons, rec.triggers, "template"
