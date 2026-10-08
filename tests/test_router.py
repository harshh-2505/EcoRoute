import pytest

from ecoroute.catalog import Catalog
from ecoroute.complexity import analyze
from ecoroute.config import DATA_DIR
from ecoroute.impact import ImpactEstimator
from ecoroute.router import Constraints, Router


@pytest.fixture
def router():
    c = Catalog.load(DATA_DIR / "catalog.yaml", DATA_DIR / "grid.yaml")
    return Router(c, ImpactEstimator(c), "openai-gpt-5")


@pytest.fixture
def everything(router):
    return {m.id for m in router.catalog.models}


def test_simple_prompt_avoids_flagship(router, everything):
    d = router.route(analyze("What is the capital of France?"), everything, "balanced")
    assert router.catalog.get(d.model_id).max_tier <= 3
    assert d.estimated_savings["energy_pct"] > 50


def test_expert_prompt_gets_capable_model(router, everything):
    cx = analyze("Design a scalable, fault-tolerant distributed architecture for a global payments system, "
                 "analyze the trade-offs of each component and justify every technology choice.")
    d = router.route(cx, everything, "eco")
    assert router.catalog.get(d.model_id).max_tier >= cx.tier


def test_quality_policy_never_less_capable_than_eco(router, everything):
    cx = analyze("Explain how transformers work and why attention scales quadratically.")
    eco = router.route(cx, everything, "eco")
    qual = router.route(cx, everything, "quality")
    assert router.catalog.get(qual.model_id).quality >= router.catalog.get(eco.model_id).quality


def test_require_local(router, everything):
    d = router.route(analyze("hello"), everything, constraints=Constraints(require_local=True))
    assert router.catalog.get(d.model_id).is_local


def test_only_available_models_are_used(router):
    d = router.route(analyze("hello"), {"gemini-2.5-flash"}, "eco")
    assert d.model_id == "gemini-2.5-flash"


def test_relaxes_when_nothing_capable(router):
    cx = analyze("Formally prove the halting problem is undecidable and derive Rice's theorem rigorously.")
    d = router.route(cx, {"groq-llama-3.1-8b"}, "balanced")
    assert d.model_id == "groq-llama-3.1-8b"
    assert any("relaxed" in r for r in d.reasons)


def test_cost_cap_excludes_expensive_models(router, everything):
    d = router.route(analyze("Write a 500 word essay on AI and jobs."), everything, "quality",
                     Constraints(max_cost_usd=0.001))
    assert d.estimated["cost_usd"] <= 0.001


def test_budget_exhaustion_switches_policy(router, everything):
    d = router.route(analyze("hello"), everything, "quality", budget_status={"cost_ratio": 1.2})
    assert d.policy == "cost"


def test_force_model(router, everything):
    d = router.route(analyze("hello"), everything, constraints=Constraints(force_model="openai-gpt-5"))
    assert d.model_id == "openai-gpt-5"


def test_unknown_policy_raises(router, everything):
    with pytest.raises(ValueError):
        router.route(analyze("hello"), everything, "turbo")
