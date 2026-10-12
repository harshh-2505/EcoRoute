"""FastAPI app: routing API, OpenAI-compatible endpoint, metrics and the web dashboard."""
from __future__ import annotations

import json
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .complexity import TIER_NAMES, analyze
from .impact import equivalents
from .router import POLICIES, POLICY_DESCRIPTIONS
from .service import EcoRouteService

STATIC = Path(__file__).parent / "static"


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "user"
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def as_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


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


class FeedbackRequest(BaseModel):
    request_id: int
    rating: str | int = Field(..., description="up | down (or 1 / -1)")
    comment: str | None = None
    escalate: bool = Field(False, description="On 'down', re-ask a stronger model and return its answer")

    def score(self) -> int:
        r = str(self.rating).lower()
        if r in ("up", "1", "+1", "good"):
            return 1
        if r in ("down", "-1", "bad"):
            return -1
        raise ValueError("rating must be 'up' or 'down'")


class BudgetRequest(BaseModel):
    monthly_budget_usd: float | None = None
    monthly_energy_budget_wh: float | None = None


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # accept (and ignore) other OpenAI fields
    model: str = "ecoroute/auto"
    messages: list[Message]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    user: str | None = None

    def options(self) -> dict:
        return {k: v for k, v in {"tools": self.tools, "tool_choice": self.tool_choice, "temperature": self.temperature,
                                  "top_p": self.top_p, "stop": self.stop}.items() if v is not None}


@dataclass
class Caller:
    team: str | None = None
    admin: bool = True       # open access (no keys configured) counts as admin
    scoped: bool = False     # metrics limited to the caller's own team


def _token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def caller(request: Request) -> Caller:
    """Resolve the API key (if keys are configured) to a team. Open access when none are configured."""
    settings = request.app.state.service.settings
    keys, admin_key = settings.access_keys, settings.admin_key
    header_team = (request.headers.get("x-ecoroute-team") or "").strip()[:64] or None
    if not keys and not admin_key:
        return Caller(team=header_team)
    tok = _token(request)
    if admin_key and tok and secrets.compare_digest(tok, admin_key):
        return Caller(team=header_team, admin=True)
    if tok and tok in keys:
        team = keys[tok]
        return Caller(team=team or header_team, admin=not admin_key, scoped=bool(team))
    raise HTTPException(401, "Missing or invalid API key (send 'Authorization: Bearer <key>').",
                        headers={"WWW-Authenticate": "Bearer"})


CallerDep = Annotated[Caller, Depends(caller)]


def admin(c: CallerDep) -> Caller:
    if not c.admin:
        raise HTTPException(403, "This action needs the admin key.")
    return c


def create_app(service: EcoRouteService | None = None) -> FastAPI:
    svc = service or EcoRouteService()
    app = FastAPI(title="EcoRoute", version=__version__,
                  description="Sustainability-aware LLM router: the right-sized model for every prompt.")
    app.state.service = svc

    def _msgs(req: RouteRequest):
        return [m.as_dict() for m in req.messages] if req.messages else None

    def team_scope(c: Caller, team: str | None) -> str | None:
        return c.team if c.scoped else team

    def _bad(e: Exception):
        raise HTTPException(status_code=400, detail=str(e))

    @app.get("/health")
    async def health():
        available, mode = await svc.availability()
        return {"status": "ok", "version": __version__, "mode": mode, "available_models": len(available),
                "providers_configured": svc.provider_status(),
                "auth": bool(svc.settings.access_keys or svc.settings.admin_key)}

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

    @app.post("/v1/analyze", dependencies=[Depends(caller)])
    async def analyze_prompt(req: AnalyzeRequest):
        return analyze(req.prompt).to_dict() | {"tiers": TIER_NAMES}

    @app.post("/v1/route", dependencies=[Depends(caller)])
    async def route(req: RouteRequest):
        try:
            d = await svc.decide(req.prompt, _msgs(req), req.policy, req.constraints)
        except (ValueError, RuntimeError) as e:
            _bad(e)
        return d.to_dict()

    def maybe_shadow(out, messages, tasks: BackgroundTasks):
        if svc.wants_shadow(out):
            tasks.add_task(svc.shadow_eval, out, messages)

    @app.post("/v1/chat")
    async def chat(req: ChatRequest, c: CallerDep, tasks: BackgroundTasks):
        try:
            out = await svc.chat(req.prompt, _msgs(req), req.policy, req.constraints, req.use_cache, req.max_tokens,
                                 team=c.team)
            maybe_shadow(out, _msgs(req) or [{"role": "user", "content": req.prompt}], tasks)
        except ValueError as e:
            _bad(e)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return out.to_dict()

    # Drop-in OpenAI-compatible endpoint: model = "ecoroute/<policy>" or a catalog id.
    @app.post("/v1/chat/completions")
    async def openai_compat(req: OpenAIChatRequest, c: CallerDep, tasks: BackgroundTasks):
        policy, constraints = None, None
        name = req.model.removeprefix("ecoroute/")
        if name in POLICIES:
            policy = name
        elif name not in ("auto", ""):
            if not svc.catalog.has(name):
                raise HTTPException(404, f"Unknown model '{req.model}'. Use ecoroute/auto|eco|balanced|quality|cost.")
            constraints = {"force_model": name}
        kwargs = {"messages": [m.as_dict() for m in req.messages], "policy": policy, "constraints": constraints,
                  "max_tokens": req.max_completion_tokens or req.max_tokens, "options": req.options(), "team": c.team}
        if req.stream:
            try:  # route before the stream opens, so bad requests still get a proper HTTP error
                await svc.decide(messages=kwargs["messages"], policy=policy, constraints=constraints,
                                 options=kwargs["options"])
            except ValueError as e:
                _bad(e)
            except RuntimeError as e:
                raise HTTPException(status_code=502, detail=str(e))
            return StreamingResponse(_sse(svc.stream_chat(**kwargs)), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            out = await svc.chat(**kwargs)
            maybe_shadow(out, kwargs["messages"], tasks)
        except ValueError as e:
            _bad(e)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        message = {"role": "assistant", "content": out.text or None}
        if out.tool_calls:
            message["tool_calls"] = out.tool_calls
        return {
            "id": f"chatcmpl-eco-{out.request_id or uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": out.model_id,
            "choices": [{"index": 0, "finish_reason": out.finish_reason, "message": message}],
            "usage": _usage(out), "ecoroute": _eco(out),
        }

    @app.post("/v1/feedback")
    async def feedback(req: FeedbackRequest, c: CallerDep):
        """Thumbs up/down on an answer. Down-votes steer future routing; ``escalate`` re-asks a stronger model."""
        row = svc.ledger.get(req.request_id)
        if not row or (c.scoped and row["team"] != c.team):
            raise HTTPException(404, f"No request {req.request_id}")
        try:
            return await svc.feedback(req.request_id, req.score(), req.comment, req.escalate)
        except ValueError as e:
            _bad(e)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.get("/v1/quality", dependencies=[Depends(caller)])
    async def quality(days: float = Query(30, ge=0)):
        """Answer-quality evidence: shadow evaluations vs the baseline, user feedback, cascade escalations."""
        since = time.time() - days * 86400 if days else 0
        q = svc.ledger.quality(since)
        return {**q, "baseline_model": svc.router.baseline_model, "shadow_rate": svc.settings.shadow_rate,
                "cascade": svc.settings.cascade, "judge_model": svc.settings.judge_model or svc.router.baseline_model,
                "feedback_penalties": svc.feedback_penalties()}

    @app.post("/v1/quality/eval/{request_id}", dependencies=[Depends(admin)])
    async def quality_eval(request_id: int):
        """Grade one earlier live answer against the baseline model now (makes two extra model calls)."""
        try:
            row = await svc.eval_request(request_id)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            _bad(e)
        if row is None:
            raise HTTPException(409, "The baseline or judge model isn't reachable in this mode.")
        return row

    @app.get("/v1/metrics/summary")
    async def metrics_summary(c: CallerDep, days: float = Query(30, ge=0), team: str | None = None):
        since = time.time() - days * 86400 if days else 0
        s = svc.ledger.summary(since, team_scope(c, team))
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
    async def metrics_timeseries(c: CallerDep, days: float = Query(7, gt=0), bucket: str = Query("hour"),
                                 team: str | None = None):
        size = {"minute": 60, "hour": 3600, "day": 86400}.get(bucket)
        if not size:
            raise HTTPException(400, "bucket must be minute|hour|day")
        return svc.ledger.timeseries(time.time() - days * 86400, size, team_scope(c, team))

    @app.get("/v1/metrics/requests")
    async def metrics_requests(c: CallerDep, limit: int = Query(50, ge=1, le=500), team: str | None = None):
        return svc.ledger.recent(limit, team_scope(c, team))

    @app.delete("/v1/metrics", dependencies=[Depends(admin)])
    async def metrics_reset():
        svc.ledger.clear()
        return {"status": "cleared"}

    @app.get("/v1/budget", dependencies=[Depends(caller)])
    async def get_budget():
        return svc.budget_status()

    @app.put("/v1/budget", dependencies=[Depends(admin)])
    async def put_budget(req: BudgetRequest):
        return svc.set_budget(req.monthly_budget_usd, req.monthly_energy_budget_wh)

    @app.post("/v1/demo/seed", dependencies=[Depends(admin)])
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


def _usage(out) -> dict:
    i, o = out.impact["input_tokens"], out.impact["output_tokens"]
    return {"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o}


def _eco(out) -> dict:
    return {"request_id": out.request_id, "source": out.source, "impact": out.impact, "savings": out.savings,
            "baseline_model": out.decision.baseline_model, "tier": out.decision.complexity["tier"],
            "policy": out.decision.policy, "reasons": out.decision.reasons, "team": out.team,
            **{k: v for k, v in out.extra.items() if k in ("escalation", "quality_check")}}


async def _sse(events):
    """Turn service stream events into OpenAI ``chat.completion.chunk`` server-sent events."""
    cid, created, model = f"chatcmpl-eco-{uuid.uuid4().hex[:12]}", int(time.time()), "ecoroute"

    def chunk(delta: dict, finish: str | None = None, **extra) -> str:
        body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(body)}\n\n"

    async for ev in events:
        if ev["type"] == "model":
            model = ev["model_id"]
            yield chunk({"role": "assistant", "content": ""})
        elif ev["type"] == "delta":
            yield chunk({"content": ev["text"]})
        elif ev["type"] == "done":
            out = ev["outcome"]
            if out.tool_calls:
                yield chunk({"tool_calls": [{"index": i, **tc} for i, tc in enumerate(out.tool_calls)]})
            yield chunk({}, out.finish_reason, usage=_usage(out), ecoroute=_eco(out))
        elif ev["type"] == "error":
            yield f"data: {json.dumps({'error': {'message': ev['message'], 'type': 'ecoroute_error'}})}\n\n"
    yield "data: [DONE]\n\n"


def app_factory() -> FastAPI:  # for `uvicorn ecoroute.api:app_factory --factory`
    return create_app()
