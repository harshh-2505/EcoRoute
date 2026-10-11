"""FastAPI app: routing API, OpenAI-compatible endpoint, metrics and the web dashboard."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .complexity import TIER_NAMES, analyze
from .impact import equivalents
from .router import POLICIES, POLICY_DESCRIPTIONS
from .service import EcoRouteService

STATIC = Path(__file__).parent / "static"


class Message(BaseModel):
    role: str = "user"
    content: str


class RouteRequest(BaseModel):
    prompt: str | None = None
    messages: list[Message] | None = None
    policy: str | None = Field(None, description="eco | balanced | quality | cost")
    constraints: dict | None = None


class ChatRequest(RouteRequest):
    use_cache: bool = True
    max_tokens: int | None = None


class AnalyzeRequest(BaseModel):
    prompt: str


class BudgetRequest(BaseModel):
    monthly_budget_usd: float | None = None
    monthly_energy_budget_wh: float | None = None


class OpenAIChatRequest(BaseModel):
    model: str = "ecoroute/auto"
    messages: list[Message]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


def create_app(service: EcoRouteService | None = None) -> FastAPI:
    svc = service or EcoRouteService()
    app = FastAPI(title="EcoRoute", version=__version__,
                  description="Sustainability-aware LLM router: the right-sized model for every prompt.")
    app.state.service = svc

    def _msgs(req: RouteRequest):
        return [m.model_dump() for m in req.messages] if req.messages else None

    def _bad(e: Exception):
        raise HTTPException(status_code=400, detail=str(e))

    @app.get("/health")
    async def health():
        available, mode = await svc.availability()
        return {"status": "ok", "version": __version__, "mode": mode, "available_models": len(available),
                "providers_configured": svc.provider_status()}

    @app.get("/v1/policies")
    async def policies():
        return [{"name": k, "weights": v, "description": POLICY_DESCRIPTIONS[k]} for k, v in POLICIES.items()]

    @app.get("/v1/models")
    async def models():
        available, mode = await svc.availability()
        return {"mode": mode, "baseline_model": svc.router.baseline_model,
                "pricing_snapshot": svc.catalog.pricing_snapshot, "sources": svc.catalog.sources,
                "models": svc.models_view(available)}

    @app.get("/v1/grid")
    async def grid():
        await svc.refresh_grid()
        return {"local_region": svc.catalog.local_region,
                "regions": [vars(r) for r in svc.catalog.regions.values()]}

    @app.post("/v1/analyze")
    async def analyze_prompt(req: AnalyzeRequest):
        return analyze(req.prompt).to_dict() | {"tiers": TIER_NAMES}

    @app.post("/v1/route")
    async def route(req: RouteRequest):
        try:
            d = await svc.decide(req.prompt, _msgs(req), req.policy, req.constraints)
        except (ValueError, RuntimeError) as e:
            _bad(e)
        return d.to_dict()

    @app.post("/v1/chat")
    async def chat(req: ChatRequest):
        try:
            out = await svc.chat(req.prompt, _msgs(req), req.policy, req.constraints, req.use_cache, req.max_tokens)
        except ValueError as e:
            _bad(e)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return out.to_dict()

    # Drop-in OpenAI-compatible endpoint: model = "ecoroute/<policy>" or a catalog id.
    @app.post("/v1/chat/completions")
    async def openai_compat(req: OpenAIChatRequest):
        policy, constraints = None, None
        name = req.model.removeprefix("ecoroute/")
        if name in POLICIES:
            policy = name
        elif name not in ("auto", ""):
            if not svc.catalog.has(name):
                raise HTTPException(404, f"Unknown model '{req.model}'. Use ecoroute/auto|eco|balanced|quality|cost.")
            constraints = {"force_model": name}
        try:
            out = await svc.chat(messages=[m.model_dump() for m in req.messages], policy=policy,
                                 constraints=constraints, max_tokens=req.max_completion_tokens or req.max_tokens)
        except ValueError as e:
            _bad(e)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return {
            "id": f"chatcmpl-eco-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": out.model_id,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": out.text}}],
            "usage": {"prompt_tokens": out.impact["input_tokens"], "completion_tokens": out.impact["output_tokens"],
                      "total_tokens": out.impact["input_tokens"] + out.impact["output_tokens"]},
            "ecoroute": {"source": out.source, "impact": out.impact, "savings": out.savings,
                         "tier": out.decision.complexity["tier"], "reasons": out.decision.reasons},
        }

    @app.get("/v1/metrics/summary")
    async def metrics_summary(days: float = Query(30, ge=0)):
        since = time.time() - days * 86400 if days else 0
        s = svc.ledger.summary(since)
        t = s["totals"]
        saved = {"energy_wh": t["b_energy"] - t["energy_wh"], "carbon_g": t["b_carbon"] - t["carbon_g"],
                 "water_ml": t["b_water"] - t["water_ml"], "cost_usd": t["b_cost"] - t["cost_usd"]}
        pct = {k: (100 * v / b if b else 0.0) for (k, v), b in
               zip(saved.items(), (t["b_energy"], t["b_carbon"], t["b_water"], t["b_cost"]))}
        return {**s, "saved": saved, "saved_pct": pct,
                "equivalents_saved": equivalents(max(saved["energy_wh"], 0), max(saved["carbon_g"], 0),
                                                 max(saved["water_ml"], 0)),
                "baseline_model": svc.router.baseline_model, "budget": svc.budget_status()}

    @app.get("/v1/metrics/timeseries")
    async def metrics_timeseries(days: float = Query(7, gt=0), bucket: str = Query("hour")):
        size = {"minute": 60, "hour": 3600, "day": 86400}.get(bucket)
        if not size:
            raise HTTPException(400, "bucket must be minute|hour|day")
        return svc.ledger.timeseries(time.time() - days * 86400, size)

    @app.get("/v1/metrics/requests")
    async def metrics_requests(limit: int = Query(50, ge=1, le=500)):
        return svc.ledger.recent(limit)

    @app.delete("/v1/metrics")
    async def metrics_reset():
        svc.ledger.clear()
        return {"status": "cleared"}

    @app.get("/v1/budget")
    async def get_budget():
        return svc.budget_status()

    @app.put("/v1/budget")
    async def put_budget(req: BudgetRequest):
        return svc.set_budget(req.monthly_budget_usd, req.monthly_energy_budget_wh)

    @app.post("/v1/demo/seed")
    async def demo_seed(n: int = Query(40, ge=1, le=200), spread_days: float = Query(7, ge=0, le=30)):
        """Populate the dashboard with the benchmark prompt mix (simulate mode only — never spends money)."""
        _, mode = await svc.availability()
        if mode != "simulate":
            raise HTTPException(409, "Demo seeding only runs in simulate mode (set ECOROUTE_MODE=simulate).")
        import json as _json
        import random
        bench = Path(__file__).resolve().parent / "data" / "demo_prompts.jsonl"
        rows = [_json.loads(x)["prompt"] for x in bench.read_text().splitlines() if x.strip()]
        rng = random.Random()
        policies = ["balanced"] * 5 + ["eco"] * 3 + ["quality", "cost"]
        prev = svc.simulator.delay
        svc.simulator.delay = False
        try:
            for i in range(n):
                out = await svc.chat(rng.choice(rows), policy=rng.choice(policies), use_cache=rng.random() < 0.3)
                if spread_days and out.request_id:  # spread demo traffic over the past few days
                    svc.ledger.set_ts(out.request_id, time.time() - rng.random() * spread_days * 86400)
        finally:
            svc.simulator.delay = prev
        return {"status": "ok", "seeded": n}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


def app_factory() -> FastAPI:  # for `uvicorn ecoroute.api:app_factory --factory`
    return create_app()
