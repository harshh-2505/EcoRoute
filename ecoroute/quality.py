"""Answer-quality safeguards: a cheap answer check (for cascade escalation) and an LLM judge (for shadow evals).

The answer check is deliberately simple and transparent: it only looks for signs that a small model
clearly failed (empty, cut off, refused, far too short, or no code when code was asked for). It never
claims an answer is *good*; that is what shadow evaluations against the baseline model are for.
"""
from __future__ import annotations

import json
import re

REFUSAL = re.compile(
    r"\b(?:i(?:'m| am) (?:not sure|unable|not able)|i can(?:no|')t (?:help|answer|do|provide|assist)|"
    r"i don(?:'|)t know|as an ai(?: language model)?|i do not have (?:enough|the) (?:information|context))\b",
    re.IGNORECASE)
CODE_HINT = re.compile(r"```|^\s{4,}\S|\bdef \w+\(|\bfunction \w+\(|=>|\bclass \w+|;\s*$", re.MULTILINE)


def check_answer(cx: dict, text: str, finish_reason: str = "stop", tool_calls: list | None = None) -> dict:
    """Return ``{"ok", "issues", "confidence"}`` for an answer to a prompt with complexity ``cx``."""
    if tool_calls:
        return {"ok": True, "issues": [], "confidence": 1.0}
    issues: list[str] = []
    words = len(text.split())
    if not text.strip():
        issues.append("empty answer")
    if finish_reason == "length":
        issues.append("answer was cut off at the token limit")
    if REFUSAL.search(text[:400]):
        issues.append("model refused or said it was unsure")
    expected_words = cx.get("expected_output_tokens", 0) * 0.75
    if cx.get("tier", 1) >= 3 and text.strip() and words < max(25, 0.12 * expected_words):
        issues.append(f"much shorter than expected ({words} words)")
    if cx.get("task_type") == "coding" and text.strip() and not CODE_HINT.search(text):
        issues.append("coding task but no code in the answer")
    confidence = max(0.0, 1.0 - 0.45 * len(issues))
    return {"ok": not issues, "issues": issues, "confidence": round(confidence, 2)}


JUDGE_SYSTEM = (
    "You are a strict, impartial grader. You compare two answers to the same user request. Judge "
    "correctness first, then completeness and usefulness. Ignore length unless it hurts the answer. "
    "Reply with JSON only.")


def judge_prompt(question: str, routed: str, baseline: str) -> list[dict]:
    user = (f"USER REQUEST:\n{question[:6000]}\n\n=== ANSWER A ===\n{routed[:8000]}\n\n=== ANSWER B ===\n{baseline[:8000]}\n\n"
            'Score each answer from 1 to 10 and give a verdict. Reply exactly as JSON: '
            '{"score_a": <1-10>, "score_b": <1-10>, "verdict": "a_better" | "equal" | "b_better", "reason": "<one sentence>"}. '
            'Use "equal" when both answers would satisfy the user equally well.')
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_verdict(text: str) -> dict | None:
    """Pull the judge's JSON out of its reply; ``None`` if it can't be read."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        a, b = float(v["score_a"]), float(v["score_b"])
    except (ValueError, KeyError, TypeError):
        return None
    verdict = v.get("verdict")
    if verdict not in ("a_better", "equal", "b_better"):
        verdict = "equal" if abs(a - b) < 1 else "a_better" if a > b else "b_better"
    return {"score_routed": a, "score_baseline": b,
            "verdict": {"a_better": "routed_better", "b_better": "baseline_better"}.get(verdict, "equal"),
            "reason": str(v.get("reason", ""))[:300]}
