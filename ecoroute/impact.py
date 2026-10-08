"""Energy, carbon, water and cost estimation for a single LLM call.

Energy per 1,000 output tokens is estimated with up to three signals (see docs/METHODOLOGY.md):

1. Compute model (open-weight models, where active parameters are public):
       FLOPs/token ≈ 2 × active_params
       Wh/1k = 2·N·1000 / EFFECTIVE_FLOP_PER_J / 3600 × PUE
2. Price proxy (closed models): price reflects GPU-time, and GPU-time ≈ energy, so
       Wh/1k = PRICE_ANCHOR_WH × (price_out / PRICE_ANCHOR_USD) ^ PRICE_EXPONENT
   The exponent is < 1 because list prices also include margin, R&D and demand effects.
3. Device model (local models): Wh/1k = device_watts / tokens_per_sec × 1000 / 3600

Input (prefill) tokens are compute-bound and batch very efficiently, so they are charged at
INPUT_TOKEN_RATIO of an output token.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .catalog import Catalog, ModelSpec

# --- Calibration constants (documented in docs/METHODOLOGY.md) ---
EFFECTIVE_FLOP_PER_J = 2.0e11   # delivered FLOPs per joule during batched decoding on H100-class GPUs
INPUT_TOKEN_RATIO = 0.10        # prefill token energy relative to a decode token
PRICE_ANCHOR_USD = 10.0         # a $10/M-output-token frontier model ...
PRICE_ANCHOR_WH = 0.5           # ... ≈ 0.5 Wh per 1k output tokens
PRICE_EXPONENT = 0.6
COMPUTE_WEIGHT_WHEN_KNOWN = 0.8  # blend weight for compute model vs price proxy

# Absolute scales for 0-100 scores (log-scaled; per request of ~typical size)
ENERGY_BOUNDS_WH = (0.003, 3.0)
CARBON_BOUNDS_G = (0.001, 2.0)
WATER_BOUNDS_ML = (0.01, 15.0)
COST_BOUNDS_USD = (0.000005, 0.05)

# Everyday equivalents
PHONE_CHARGE_WH = 15.0          # one full smartphone charge
LED_BULB_W = 10.0               # 10 W LED bulb
CAR_G_PER_KM = 120.0            # average petrol car
GOOGLE_SEARCH_WH = 0.3          # classic (2009) per-search estimate


@dataclass
class EnergyProfile:
    """Static per-model efficiency figures."""
    model_id: str
    wh_per_1k_output: float
    wh_compute: float | None
    wh_price: float | None
    method: str          # "compute+price" | "price" | "device"
    confidence: str      # "medium" | "low"
    label: str           # A..G efficiency class

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Impact:
    energy_wh: float
    carbon_g: float
    water_ml: float
    cost_usd: float
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict:
        return {k: round(v, 8) if isinstance(v, float) else v for k, v in asdict(self).items()}


def efficiency_label(wh_per_1k: float) -> str:
    """EU-energy-label-style class from Wh per 1k output tokens."""
    for limit, label in ((0.02, "A"), (0.05, "B"), (0.1, "C"), (0.25, "D"), (0.5, "E"), (1.0, "F")):
        if wh_per_1k < limit:
            return label
    return "G"


def compute_wh_per_1k(active_params_b: float, pue: float) -> float:
    joules = 2 * active_params_b * 1e9 * 1000 / EFFECTIVE_FLOP_PER_J
    return joules / 3600 * pue


def price_wh_per_1k(price_out: float) -> float:
    if price_out <= 0:
        return 0.0
    return PRICE_ANCHOR_WH * (price_out / PRICE_ANCHOR_USD) ** PRICE_EXPONENT


def energy_profile(m: ModelSpec) -> EnergyProfile:
    if m.is_local:
        wh = m.device_watts / max(m.tps, 0.1) * 1000 / 3600
        return EnergyProfile(m.id, wh, None, None, "device", "medium", efficiency_label(wh))
    wh_p = price_wh_per_1k(m.price_out)
    if m.params_active_b:
        wh_c = compute_wh_per_1k(m.params_active_b, m.pue)
        if wh_p > 0:
            w = COMPUTE_WEIGHT_WHEN_KNOWN
            wh = math.exp(w * math.log(wh_c) + (1 - w) * math.log(wh_p))
        else:
            wh = wh_c
        return EnergyProfile(m.id, wh, wh_c, wh_p, "compute+price", "medium", efficiency_label(wh))
    return EnergyProfile(m.id, wh_p, None, wh_p, "price", "low", efficiency_label(wh_p))


class ImpactEstimator:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.profiles = {m.id: energy_profile(m) for m in catalog.models}

    def estimate(self, model: ModelSpec, input_tokens: int, output_tokens: int) -> Impact:
        prof = self.profiles[model.id]
        eff_tokens = output_tokens + INPUT_TOKEN_RATIO * input_tokens
        energy_wh = prof.wh_per_1k_output * eff_tokens / 1000
        region = self.catalog.region_for(model)
        kwh = energy_wh / 1000
        carbon_g = kwh * region.carbon_g_per_kwh
        # On-site cooling water uses IT energy (energy / PUE); off-site uses total facility energy.
        it_kwh = kwh / model.pue if model.pue else kwh
        water_ml = (it_kwh * model.wue + kwh * region.water_l_per_kwh) * 1000
        cost = input_tokens / 1e6 * model.price_in + output_tokens / 1e6 * model.price_out
        return Impact(energy_wh, carbon_g, water_ml, cost, int(input_tokens), int(output_tokens))


def log_score(value: float, bounds: tuple[float, float]) -> float:
    """100 = at/below lower bound (best), 0 = at/above upper bound (worst)."""
    lo, hi = bounds
    if value <= lo:
        return 100.0
    if value >= hi:
        return 0.0
    return 100.0 * (1 - (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))


def sustainability_score(imp: Impact) -> float:
    """Composite 0-100 (higher = greener): 50% energy, 30% carbon, 20% water."""
    return round(0.5 * log_score(imp.energy_wh, ENERGY_BOUNDS_WH)
                 + 0.3 * log_score(imp.carbon_g, CARBON_BOUNDS_G)
                 + 0.2 * log_score(imp.water_ml, WATER_BOUNDS_ML), 2)


def cost_score(imp: Impact) -> float:
    return round(log_score(imp.cost_usd, COST_BOUNDS_USD), 2) if imp.cost_usd > 0 else 100.0


def equivalents(energy_wh: float, carbon_g: float, water_ml: float) -> dict:
    return {
        "phone_charges": energy_wh / PHONE_CHARGE_WH,
        "led_bulb_minutes": energy_wh / LED_BULB_W * 60,
        "google_searches": energy_wh / GOOGLE_SEARCH_WH,
        "car_km": carbon_g / CAR_G_PER_KM,
        "water_glasses": water_ml / 250.0,
    }
