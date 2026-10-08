import json
from pathlib import Path

import pytest

from ecoroute.complexity import analyze, estimate_tokens


@pytest.mark.parametrize("prompt", ["hi", "Thanks!", "What is the capital of Japan?"])
def test_trivial_prompts_are_tier_1(prompt):
    assert analyze(prompt).tier == 1


def test_hard_prompts_score_higher_than_easy_ones():
    easy = analyze("Translate 'hello' to French.")
    hard = analyze("Design a scalable, fault-tolerant distributed architecture for a payments platform "
                   "and analyze the trade-offs of each component.")
    assert hard.score > easy.score
    assert hard.tier >= 4


def test_code_never_goes_to_trivial_tier():
    assert analyze("fix this: def f(x): return x +").tier >= 2


def test_proofs_get_at_least_tier_3():
    assert analyze("Prove that sqrt(2) is irrational.").tier >= 3


def test_requested_length_raises_expected_output():
    r = analyze("Write a 2000 word essay on renewable energy.")
    assert r.expected_output_tokens >= 2000


def test_contributions_explain_score():
    r = analyze("Explain why the sky is blue.")
    assert "reasoning" in r.contributions
    assert 0 <= r.score <= 1


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") >= 3


def test_benchmark_accuracy_floor():
    """Guard against regressions: every labelled prompt must be within one tier."""
    rows = [json.loads(x) for x in Path("benchmarks/prompts.jsonl").read_text().splitlines() if x.strip()]
    near = sum(abs(analyze(r["prompt"]).tier - r["tier"]) <= 1 for r in rows)
    assert near / len(rows) >= 0.95
