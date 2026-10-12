"""Sustainability-aware routing: pick the greenest model that is capable enough for the prompt."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .catalog import Catalog, ModelSpec
from .complexity import ComplexityResult
from .impact import Impact, ImpactEstimator, cost_score, sustainability_score
from .providers import TOOL_PROVIDERS

# weights: sustainability, cost, quality, latency
POLICIES: dict[str, dict[str, float]] = {
    "eco":      {"sustainability": 0.60, "cost": 0.20, "quality": 0.15, "latency": 0.05},
    "balanced": {"sustainability": 0.35, "cost": 0.25, "quality": 0.30, "latency": 0.10},
    "quality":  {"sustainability": 0.15, "cost": 0.10, "quality": 0.65, "latency": 0.10},
    "cost":     {"sustainability": 0.20, "cost": 0.65, "quality": 0.10, "latency": 0.05},
}
POLICY_DESCRIPTIONS = {
    "eco": "Minimise energy, carbon and water; accept the smallest capable model.",
    "balanced": "Trade off sustainability, cost and answer quality.",
    "quality": "Prefer the strongest model, still skipping wasteful over-provisioning.",
    "cost": "Minimise spend within the capable set.",
}


@dataclass
class Constraints:
    max_cost_usd: float | None = None        # per request
    max_energy_wh: float | None = None       # per request
    require_local: bool = False              # privacy / offline
    prefer_local: bool = False               # small bonus for local models
    allowed_providers: list[str] | None = None
    exclude_models: list[str] | None = None
    min_tier: int | None = None              # force at least this tier
    force_model: str | None = None           # bypass routing
    needs_tools: bool = False                # request carries tool/function definitions

    @classmethod
    def from_dict(cls, d: dict | None) -> Constraints:
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Candidate:
    model_id: str
    display_name: str
    provider: str
    kind: str
    quality: float
    max_tier: int
    estimated: dict
    scores: dict
    total: float
    eligible: bool = True
    reason: str = ""


@dataclass
class Decision:
    model_id: str
    display_name: str
    provider: str
    policy: str
    complexity: dict
    required_tier: int
    estimated: dict
    baseline_model: str
    baseline_estimated: dict
    estimated_savings: dict
    candidates: list[Candidate] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    budget_status: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [asdict(c) for c in self.candidates]
        return d


class Router:
    def __init__(self, catalog: Catalog, estimator: ImpactEstimator, baseline_model: str):
        self.catalog = catalog
        self.estimator = estimator
        if not catalog.has(baseline_model):
            baseline_model = max(catalog.models, key=lambda m: m.quality).id
        self.baseline_model = baseline_model

    # ------------------------------------------------------------------
    def route(self, cx: ComplexityResult, available: set[str], policy: str = "balanced",
              constraints: Constraints | None = None, budget_status: dict | None = None) -> Decision:
        constraints = constraints or Constraints()
        budget_status = budget_status or {}
        reasons: list[str] = []
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy '{policy}'. Options: {sorted(POLICIES)}")

        # Budget pressure shifts the policy toward cost/eco automatically.
        effective_policy = policy
        if budget_status.get("cost_ratio", 0) >= 1.0 or budget_status.get("energy_ratio", 0) >= 1.0:
            effective_policy = "eco" if budget_status.get("energy_ratio", 0) >= 1.0 else "cost"
            reasons.append(f"Monthly budget exhausted → switched policy '{policy}' to '{effective_policy}'.")
        elif budget_status.get("cost_ratio", 0) >= 0.8 and policy == "quality":
            effective_policy = "balanced"
            reasons.append("Over 80% of monthly budget used → softened 'quality' to 'balanced'.")
        weights = dict(POLICIES[effective_policy])

        required = max(cx.tier, constraints.min_tier or 1)
        if constraints.needs_tools and required < 2:
            required = 2  # the smallest models are unreliable at choosing and filling tool calls
            reasons.append("Request defines tools → needs a tool-capable model of tier 2 or above.")
        in_tok, out_tok = cx.input_tokens, cx.expected_output_tokens

        cands: list[Candidate] = []
        for m in self.catalog.models:
            imp = self.estimator.estimate(m, in_tok, out_tok)
            elig, why = self._eligible(m, required, available, constraints, imp)
            cands.append(self._score(m, imp, weights, cx, required, elig, why, constraints))

        eligible = [c for c in cands if c.eligible]
        if constraints.force_model:
            forced = next((c for c in cands if c.model_id == constraints.force_model), None)
            if forced is None:
                raise ValueError(f"Unknown model '{constraints.force_model}'")
            if constraints.force_model not in available:
                raise ValueError(f"Model '{constraints.force_model}' is not available (missing credentials?)")
            forced.eligible, forced.reason = True, "forced by request"
            eligible = [forced]
            reasons.append("Model forced by request; routing bypassed.")
        if not eligible:
            # Relax the capability requirement: use the most capable available model.
            pool = [c for c in cands if c.model_id in available
                    and (not constraints.require_local or c.kind == "local")]
            if not pool:
                raise RuntimeError("No models available. Configure an API key, run Ollama, or use simulate mode.")
            best_tier = max(c.max_tier for c in pool)
            eligible = [c for c in pool if c.max_tier == best_tier]
            for c in eligible:
                c.eligible, c.reason = True, "fallback: no model met every constraint"
            reasons.append(f"No available model satisfied tier {required} + constraints; "
                           f"relaxed to the most capable available (tier {best_tier}).")

        eligible.sort(key=lambda c: c.total, reverse=True)
        cands.sort(key=lambda c: (not c.eligible, -c.total))
        chosen = eligible[0]
        model = self.catalog.get(chosen.model_id)

        base = self.catalog.get(self.baseline_model)
        base_imp = self.estimator.estimate(base, in_tok, out_tok)
        est = chosen.estimated
        savings = _savings(est, base_imp.to_dict())

        reasons.insert(0, f"Prompt judged tier {cx.tier} ({cx.tier_name}, {cx.task_type}); "
                          f"{len(eligible)} capable model(s) available.")
        reasons.append(f"Picked {model.display_name} under '{effective_policy}' policy "
                       f"(score {chosen.total:.1f}, sustainability {chosen.scores['sustainability']:.0f}/100).")
        if savings["energy_pct"] > 0:
            reasons.append(f"Estimated {savings['energy_pct']:.0f}% less energy and "
                           f"{savings['cost_pct']:.0f}% lower cost than always using {base.display_name}.")

        return Decision(
            model_id=model.id, display_name=model.display_name, provider=model.provider,
            policy=effective_policy, complexity=cx.to_dict(), required_tier=required,
            estimated=est, baseline_model=base.id, baseline_estimated=base_imp.to_dict(),
            estimated_savings=savings, candidates=cands,
            fallbacks=[c.model_id for c in eligible[1:4]], reasons=reasons, budget_status=budget_status,
        )

    # ------------------------------------------------------------------
    def _eligible(self, m: ModelSpec, required: int, available: set[str], k: Constraints,
                  imp: Impact) -> tuple[bool, str]:
        if m.id not in available:
            return False, "not available (no credentials / not running)"
        if k.exclude_models and m.id in k.exclude_models:
            return False, "excluded by request"
        if k.require_local and not m.is_local:
            return False, "local-only requested"
        if k.allowed_providers and m.provider not in k.allowed_providers:
            return False, "provider not allowed"
        if k.needs_tools and (m.provider not in TOOL_PROVIDERS or m.api == "responses"):
            return False, "no tool-calling support"
        if m.max_tier < required:
            return False, f"under-powered (max tier {m.max_tier} < {required})"
        if k.max_cost_usd is not None and imp.cost_usd > k.max_cost_usd:
            return False, "over per-request cost cap"
        if k.max_energy_wh is not None and imp.energy_wh > k.max_energy_wh:
            return False, "over per-request energy cap"
        return True, ""

    def _score(self, m: ModelSpec, imp: Impact, weights: dict, cx: ComplexityResult, required: int,
               eligible: bool, why: str, k: Constraints) -> Candidate:
        sust = sustainability_score(imp)
        cost = cost_score(imp)
        # Quality matters more as prompts get harder. Over-provisioning (a tier-5 model for a
        # tier-1 prompt) earns no extra quality credit — that headroom is wasted energy.
        q_needed = 40 + 12 * required
        quality = 100 * min(1.0, m.quality / q_needed) if q_needed else 100
        quality = 0.7 * quality + 0.3 * m.quality  # keep some absolute preference
        latency = max(0.0, min(100.0, 100 * (math.log10(max(m.tps, 1)) - 1) / 2))  # 10→0, 1000→100

        w = dict(weights)
        w["quality"] *= 0.6 + 0.8 * cx.score  # 0.6x for trivial prompts, 1.4x for hardest
        total_w = sum(w.values())
        total = (w["sustainability"] * sust + w["cost"] * cost + w["quality"] * quality
                 + w["latency"] * latency) / total_w
        if k.prefer_local and m.is_local:
            total += 5
        return Candidate(
            model_id=m.id, display_name=m.display_name, provider=m.provider, kind=m.kind,
            quality=m.quality, max_tier=m.max_tier, estimated=imp.to_dict(),
            scores={"sustainability": sust, "cost": cost, "quality": round(quality, 2),
                    "latency": round(latency, 2)},
            total=round(total, 3), eligible=eligible, reason=why,
        )


def _savings(est: dict, base: dict) -> dict:
    def pct(a: float, b: float) -> float:
        return round(100 * (b - a) / b, 2) if b > 0 else 0.0
    return {
        "energy_wh": base["energy_wh"] - est["energy_wh"],
        "carbon_g": base["carbon_g"] - est["carbon_g"],
        "water_ml": base["water_ml"] - est["water_ml"],
        "cost_usd": base["cost_usd"] - est["cost_usd"],
        "energy_pct": pct(est["energy_wh"], base["energy_wh"]),
        "carbon_pct": pct(est["carbon_g"], base["carbon_g"]),
        "water_pct": pct(est["water_ml"], base["water_ml"]),
        "cost_pct": pct(est["cost_usd"], base["cost_usd"]),
    }
