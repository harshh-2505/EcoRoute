import pytest

from ecoroute.catalog import Catalog
from ecoroute.config import DATA_DIR
from ecoroute.impact import (
    ImpactEstimator,
    compute_wh_per_1k,
    efficiency_label,
    energy_profile,
    log_score,
    price_wh_per_1k,
    sustainability_score,
)


@pytest.fixture
def catalog():
    return Catalog.load(DATA_DIR / "catalog.yaml", DATA_DIR / "grid.yaml")


def test_catalog_loads(catalog):
    assert len(catalog.models) >= 10
    assert any(m.is_local for m in catalog.models)


def test_compute_model_scales_with_parameters():
    assert compute_wh_per_1k(70, 1.1) == pytest.approx(compute_wh_per_1k(7, 1.1) * 10)


def test_price_proxy_is_monotonic_and_anchored():
    assert price_wh_per_1k(10.0) == pytest.approx(0.5)
    assert price_wh_per_1k(1.0) < price_wh_per_1k(10.0) < price_wh_per_1k(30.0)
    assert price_wh_per_1k(0) == 0


def test_closed_models_use_price_open_models_blend(catalog):
    assert energy_profile(catalog.get("openai-gpt-5")).method == "price"
    assert energy_profile(catalog.get("groq-llama-3.3-70b")).method == "compute+price"
    assert energy_profile(catalog.get("local-llama3.2-3b")).method == "device"


def test_bigger_models_use_more_energy(catalog):
    est = ImpactEstimator(catalog)
    small = est.estimate(catalog.get("groq-llama-3.1-8b"), 100, 500)
    big = est.estimate(catalog.get("anthropic-claude-sonnet-4.5"), 100, 500)
    assert big.energy_wh > small.energy_wh * 5
    assert big.cost_usd > small.cost_usd


def test_local_models_cost_nothing_and_skip_onsite_water(catalog):
    est = ImpactEstimator(catalog)
    m = catalog.get("local-qwen2.5-7b")
    imp = est.estimate(m, 100, 500)
    assert imp.cost_usd == 0
    region = catalog.region_for(m)
    assert imp.water_ml == pytest.approx(imp.energy_wh / 1000 * region.water_l_per_kwh * 1000)


def test_dirty_grid_means_more_carbon():
    clean = Catalog.load(DATA_DIR / "catalog.yaml", DATA_DIR / "grid.yaml", "nordics")
    dirty = Catalog.load(DATA_DIR / "catalog.yaml", DATA_DIR / "grid.yaml", "india")
    m = clean.get("local-llama3.2-3b")
    assert ImpactEstimator(dirty).estimate(m, 0, 1000).carbon_g > 10 * ImpactEstimator(clean).estimate(m, 0, 1000).carbon_g


def test_scores_and_labels():
    assert log_score(0.0001, (0.001, 1)) == 100
    assert log_score(10, (0.001, 1)) == 0
    assert efficiency_label(0.01) == "A" and efficiency_label(5) == "G"


def test_sustainability_score_prefers_small_models(catalog):
    est = ImpactEstimator(catalog)
    s_small = sustainability_score(est.estimate(catalog.get("groq-gpt-oss-20b"), 100, 500))
    s_big = sustainability_score(est.estimate(catalog.get("openai-gpt-5"), 100, 500))
    assert s_small > s_big
