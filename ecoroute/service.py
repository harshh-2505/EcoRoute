"""EcoRoute service: analyse → route → (cache) → execute with fallback → measure → record."""
from __future__ import annotations

import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .catalog import Catalog
from .complexity import analyze, estimate_tokens
from .config import Settings
from .impact import Impact, ImpactEstimator, equivalents, sustainability_score
from .ledger import Ledger, cache_key, month_start_ts
from .providers import Provider, ProviderError, ProviderResult, SimulatedProvider, build_providers
from .quality import check_answer, judge_prompt, parse_verdict
from .router import Constraints, Decision, Router

log = logging.getLogger("ecoroute")

CONCISE_INSTRUCTION = ("Answer as concisely as possible while staying correct and complete. Lead with the answer, "
                       "skip preamble, restating the question and closing summaries.")


@dataclass
class ChatOutcome:
    decision: Decision
    model_id: str
    text: str
    source: str  # live | simulated | cache
    impact: dict
    baseline: dict
    savings: dict
    latency_ms: float
    fallback_used: bool
    request_id: int | None
    errors: list[str]
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"
    team: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id, "model_id": self.model_id, "source": self.source,
            "response": self.text, "tool_calls": self.tool_calls, "finish_reason": self.finish_reason,
            "team": self.team, "latency_ms": round(self.latency_ms, 1),
            "impact": self.impact, "baseline": self.baseline, "savings": self.savings,
            "equivalents_saved": equivalents(max(self.savings["energy_wh"], 0), max(self.savings["carbon_g"], 0),
                                             max(self.savings["water_ml"], 0)),
            "fallback_used": self.fallback_used, "errors": self.errors,
            "decision": self.decision.to_dict(), **self.extra,
        }


class EcoRouteService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        s = self.settings
        self.catalog = Catalog.load(s.catalog_path, s.grid_path, s.local_region)
        self.estimator = ImpactEstimator(self.catalog)
        self.router = Router(self.catalog, self.estimator, s.baseline_model)
        self.ledger = Ledger(s.db_path)
        self.providers: dict[str, Provider] = build_providers(s)
        self.simulator = SimulatedProvider()
        self._grid_checked = 0.0

    # ------------------------------------------------------------------ availability
    async def availability(self) -> tuple[set[str], str]:
        """Return (available model ids, effective mode)."""
        mode = self.settings.mode
        if mode == "simulate":
            return {m.id for m in self.catalog.models}, "simulate"
        live: set[str] = set()
        for p in self.providers.values():
            live |= await p.available_models(self.catalog.models)
        if mode == "live" or live:
            return live, "live"
        return {m.id for m in self.catalog.models}, "simulate"

    def provider_status(self) -> dict:
        return {name: p.configured() for name, p in self.providers.items() if name != "ollama"}

    # ------------------------------------------------------------------ budgets
    def budget(self) -> dict:
        b = self.ledger.get_setting("budget") or {}
        return {"monthly_budget_usd": b.get("monthly_budget_usd", self.settings.monthly_budget_usd),
                "monthly_energy_budget_wh": b.get("monthly_energy_budget_wh",
                                                  self.settings.monthly_energy_budget_wh)}

    def set_budget(self, usd: float | None = None, wh: float | None = None) -> dict:
        b = self.budget()
        if usd is not None:
            b["monthly_budget_usd"] = max(0.0, float(usd))
        if wh is not None:
            b["monthly_energy_budget_wh"] = max(0.0, float(wh))
        self.ledger.set_setting("budget", b)
        return self.budget_status()

    def budget_status(self) -> dict:
        b, mtd = self.budget(), self.ledger.month_to_date()
        usd, wh = b["monthly_budget_usd"], b["monthly_energy_budget_wh"]
        return {**b, "spent_usd": mtd["cost_usd"], "used_wh": mtd["energy_wh"],
                "cost_ratio": mtd["cost_usd"] / usd if usd else 0.0,
                "energy_ratio": mtd["energy_wh"] / wh if wh else 0.0,
                "month_start": month_start_ts()}

    # ------------------------------------------------------------------ live grid data
    async def refresh_grid(self, force: bool = False) -> None:
        """Pull live carbon intensity from Electricity Maps (if a key is configured)."""
        key = self.settings.electricitymaps_api_key
        if not key or (not force and time.time() - self._grid_checked < 900):
            return
        self._grid_checked = time.time()
        async with httpx.AsyncClient(timeout=5) as client:
            for region in self.catalog.regions.values():
                if not region.zone:
                    continue
                try:
                    r = await client.get("https://api.electricitymap.org/v3/carbon-intensity/latest",
                                         params={"zone": region.zone}, headers={"auth-token": key})
                    if r.status_code == 200 and r.json().get("carbonIntensity") is not None:
                        region.carbon_g_per_kwh = float(r.json()["carbonIntensity"])
                        region.live = True
                except (httpx.HTTPError, ValueError) as e:  # keep static value
                    log.warning("grid refresh failed for %s: %s", region.key, e)

    # ------------------------------------------------------------------ core
    @staticmethod
    def _normalise(prompt: str | None, messages: list[dict] | None) -> list[dict]:
        if messages:
            return [_norm_message(m) for m in messages]
        if not prompt:
            raise ValueError("Provide 'prompt' or 'messages'.")
        return [{"role": "user", "content": prompt}]

    async def decide(self, prompt: str | None = None, messages: list[dict] | None = None,
                     policy: str | None = None, constraints: dict | None = None,
                     options: dict | None = None) -> Decision:
        msgs = self._normalise(prompt, messages)
        await self.refresh_grid()
        last_user = last_prompt(msgs)
        history = sum(estimate_tokens(m["content"]) for m in msgs) - estimate_tokens(last_user)
        cx = analyze(last_user, history_tokens=history)
        available, _ = await self.availability()
        k = Constraints.from_dict(constraints)
        if (options or {}).get("tools"):
            k.needs_tools = True
        return self.router.route(cx, available, policy or self.settings.default_policy, k, self.budget_status(),
                                 self.feedback_penalties())

    def feedback_penalties(self, days: float = 30) -> dict[str, float]:
        """Quality points to subtract per model from recent thumbs-down votes (needs ≥3 votes; full weight at 10)."""
        out = {}
        for model_id, f in self.ledger.feedback_stats(time.time() - days * 86400).items():
            n = f["up"] + f["down"]
            if n >= 3 and f["down"]:
                out[model_id] = round(30 * f["down"] / n * min(1.0, n / 10), 2)
        return out

    def _concise(self, msgs, decision, constraints, max_tokens) -> tuple[list[dict], int | None, bool]:
        """Concise mode: ask for a short answer and cap output near the tier's expected length.

        Output tokens dominate inference energy, so a shorter answer is a direct saving.
        """
        k = Constraints.from_dict(constraints)
        if not (self.settings.concise if k.concise is None else k.concise):
            return msgs, max_tokens, False
        cap = max(128, int(decision.complexity["expected_output_tokens"] * 1.2))
        return [{"role": "system", "content": CONCISE_INSTRUCTION}] + msgs, min(max_tokens or cap, cap), True

    def _cached(self, msgs, decision, use_cache, options, variant: str = "") -> tuple[str, dict | None]:
        """Cache key and hit (if any). Requests with tools are never cached: tool results change."""
        key = cache_key(msgs, decision.required_tier, variant)
        if not (use_cache and self.settings.cache_enabled) or (options or {}).get("tools"):
            return key, None
        return key, self.ledger.cache_get(key, self.settings.cache_ttl_hours * 3600)

    def _from_cache(self, decision, hit, t0, errors, msgs, team) -> ChatOutcome:
        base = self.estimator.estimate(self.catalog.get(decision.baseline_model),
                                       hit["input_tokens"], hit["output_tokens"]).to_dict()
        zero = {"energy_wh": 0.0, "carbon_g": 0.0, "water_ml": 0.0, "cost_usd": 0.0,
                "input_tokens": hit["input_tokens"], "output_tokens": hit["output_tokens"]}
        res = ProviderResult(hit["response"], hit["input_tokens"], hit["output_tokens"], (time.perf_counter() - t0) * 1000)
        return self._finish(decision, hit["model_id"], res, "cache", zero, base, False, errors, msgs, team)

    def _measure(self, decision, model_id, res: ProviderResult, msgs, key, use_cache, options, fallback, errors, team):
        model = self.catalog.get(model_id)
        in_tok = res.input_tokens or sum(estimate_tokens(m["content"]) for m in msgs)
        out_tok = res.output_tokens or estimate_tokens(res.text) + 20 * len(res.tool_calls or [])
        imp = self.estimator.estimate(model, in_tok, out_tok).to_dict()
        base = self.estimator.estimate(self.catalog.get(decision.baseline_model), in_tok, out_tok).to_dict()
        if use_cache and self.settings.cache_enabled and res.text and not res.tool_calls and not (options or {}).get("tools"):
            self.ledger.cache_put(key, model_id, res.text, in_tok, out_tok)
        return self._finish(decision, model_id, res, "simulated" if res.simulated else "live",
                            imp, base, fallback, errors, msgs, team)

    def _target(self, model_id: str, cx: dict, max_tokens: int | None, mode: str) -> tuple[Provider, int]:
        model = self.catalog.get(model_id)
        if mode == "simulate":
            return self.simulator, cx["expected_output_tokens"]
        provider = self.providers.get(model.provider)
        if provider is None:
            raise ProviderError(f"no provider configured for {model.provider}")
        return provider, max_tokens or min(4096, max(256, int(cx["expected_output_tokens"] * 2.5)))

    async def chat(self, prompt: str | None = None, messages: list[dict] | None = None,
                   policy: str | None = None, constraints: dict | None = None,
                   use_cache: bool = True, max_tokens: int | None = None,
                   options: dict | None = None, team: str | None = None) -> ChatOutcome:
        msgs = self._normalise(prompt, messages)
        decision = await self.decide(messages=msgs, policy=policy, constraints=constraints, options=options)
        _, mode = await self.availability()
        errors: list[str] = []
        t0 = time.perf_counter()
        send, max_tokens, concise = self._concise(msgs, decision, constraints, max_tokens)

        # 1) Cache: a repeated prompt (or the same question worded slightly differently) costs (almost) nothing.
        key, hit = self._cached(msgs, decision, use_cache, options, "concise" if concise else "")
        if hit:
            return self._from_cache(decision, hit, t0, errors, msgs, team)

        # 2) Execute, falling back down the ranked list on provider errors.
        for i, model_id in enumerate([decision.model_id] + decision.fallbacks):
            try:
                provider, budget = self._target(model_id, decision.complexity, max_tokens, mode)
                res = await provider.complete(self.catalog.get(model_id), send, budget, options)
            except (ProviderError, httpx.HTTPError, KeyError, IndexError) as e:
                errors.append(f"{model_id}: {e}")
                log.warning("provider failed, falling back: %s", e)
                continue
            out = self._measure(decision, model_id, res, msgs, key, use_cache, options, i > 0, errors, team)
            out.extra["concise"] = concise
            k = Constraints.from_dict(constraints)
            if (self.settings.cascade if k.cascade is None else k.cascade) and not res.simulated:
                out = await self._cascade(out, res, send, mode, max_tokens, options, team)
            return out
        raise RuntimeError("All candidate models failed: " + "; ".join(errors))

    async def stream_chat(self, prompt: str | None = None, messages: list[dict] | None = None,
                          policy: str | None = None, constraints: dict | None = None,
                          use_cache: bool = True, max_tokens: int | None = None,
                          options: dict | None = None, team: str | None = None) -> AsyncIterator[dict]:
        """Like :meth:`chat`, but yields ``{"type": "model"|"delta"|"done", ...}`` events as text arrives.

        Fallback to the next model only happens before the first token; after that an error ends the stream.
        """
        msgs = self._normalise(prompt, messages)
        decision = await self.decide(messages=msgs, policy=policy, constraints=constraints, options=options)
        _, mode = await self.availability()
        errors: list[str] = []
        t0 = time.perf_counter()
        send, max_tokens, concise = self._concise(msgs, decision, constraints, max_tokens)

        key, hit = self._cached(msgs, decision, use_cache, options, "concise" if concise else "")
        if hit:
            yield {"type": "model", "model_id": hit["model_id"], "decision": decision}
            yield {"type": "delta", "text": hit["response"]}
            yield {"type": "done", "outcome": self._from_cache(decision, hit, t0, errors, msgs, team)}
            return

        for i, model_id in enumerate([decision.model_id] + decision.fallbacks):
            started = False
            try:
                provider, budget = self._target(model_id, decision.complexity, max_tokens, mode)
                async for ev in provider.stream(self.catalog.get(model_id), send, budget, options):
                    if not started:
                        started = True
                        yield {"type": "model", "model_id": model_id, "decision": decision}
                    if ev.done:
                        outcome = self._measure(decision, model_id, ev.result, msgs, key, use_cache, options,
                                                i > 0, errors, team)
                        yield {"type": "done", "outcome": outcome}
                        return
                    if ev.text:
                        yield {"type": "delta", "text": ev.text}
            except (ProviderError, httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                errors.append(f"{model_id}: {e}")
                log.warning("stream failed on %s: %s", model_id, e)
                if started:
                    yield {"type": "error", "message": str(e)}
                    return
        yield {"type": "error", "message": "All candidate models failed: " + "; ".join(errors)}

    async def _cascade(self, out: ChatOutcome, res: ProviderResult, msgs, mode, max_tokens, options, team) -> ChatOutcome:
        """Check the answer; if it clearly failed, ask the next stronger eligible model once."""
        decision = out.decision
        check = check_answer(decision.complexity, res.text, res.finish_reason, res.tool_calls)
        self.ledger.update(out.request_id, check_ok=int(check["ok"]))
        out.extra["quality_check"] = check
        if check["ok"]:
            return out
        current = self.catalog.get(out.model_id)
        stronger = [c for c in decision.candidates if c.eligible and c.model_id != out.model_id
                    and (c.quality > current.quality or c.max_tier > current.max_tier)]
        if not stronger:
            return out
        target = max(stronger, key=lambda c: c.total)
        try:
            provider, budget = self._target(target.model_id, decision.complexity, max_tokens, mode)
            res2 = await provider.complete(self.catalog.get(target.model_id), msgs, budget, options)
        except (ProviderError, httpx.HTTPError, KeyError, IndexError) as e:
            out.errors.append(f"escalation to {target.model_id} failed: {e}")
            return out
        key = cache_key(msgs, decision.required_tier)
        out2 = self._measure(decision, target.model_id, res2, msgs, key, True, options, False, out.errors, team)
        check2 = check_answer(decision.complexity, res2.text, res2.finish_reason, res2.tool_calls)
        self.ledger.update(out2.request_id, escalated=1, check_ok=int(check2["ok"]))
        out2.extra["quality_check"] = check2
        out2.extra["escalation"] = {"from": out.model_id, "to": target.model_id, "issues": check["issues"],
                                    "first_request_id": out.request_id,
                                    "extra_energy_wh": out.impact["energy_wh"], "extra_cost_usd": out.impact["cost_usd"]}
        return out2

    async def feedback(self, request_id: int, rating: int, comment: str | None = None,
                       escalate: bool = False) -> dict:
        """Record 👍/👎 on a request. With ``escalate`` on a 👎, re-ask a stronger model and return its answer."""
        row = self.ledger.get(request_id)
        if not row:
            raise LookupError(f"No request {request_id}")
        self.ledger.set_feedback(request_id, rating, (comment or "")[:500] or None)
        result: dict = {"request_id": request_id, "rating": rating}
        if escalate and rating < 0:
            msgs = self.ledger.payload_get(request_id)
            if not msgs:
                raise ValueError("The prompt for this request wasn't stored, so it can't be re-asked "
                                 "(ECOROUTE_STORE_PROMPTS is off or it expired).")
            out = await self.chat(messages=msgs, policy=row["policy"], use_cache=False, team=row["team"],
                                  constraints={"min_tier": min(5, (row["tier"] or 1) + 1),
                                               "exclude_models": [row["model_id"]], "cascade": False})
            self.ledger.update(out.request_id, escalated=1)
            result["escalated"] = out.to_dict()
        return result

    # ------------------------------------------------------------------ shadow evaluation
    def wants_shadow(self, out: ChatOutcome) -> bool:
        return (out.source == "live" and not out.tool_calls and self.settings.shadow_rate > 0
                and out.model_id != out.decision.baseline_model and random.random() < self.settings.shadow_rate)

    async def shadow_eval(self, out: ChatOutcome, messages: list[dict]) -> dict | None:
        """Send the same request to the baseline model, then have a judge grade both answers blind.

        Returns the stored evaluation, or ``None`` if the baseline or judge couldn't be reached.
        The extra calls are real energy; they are logged as evaluation overhead, not as routed traffic.
        """
        msgs = self._normalise(None, messages)
        decision = out.decision
        base = self.catalog.get(decision.baseline_model)
        judge = self.catalog.get(self.settings.judge_model) if self.settings.judge_model and \
            self.catalog.has(self.settings.judge_model) else base
        available, mode = await self.availability()
        if mode != "live" or base.id not in available or judge.id not in available:
            return None
        try:
            bp, budget = self._target(base.id, decision.complexity, None, mode)
            b = await bp.complete(base, msgs, budget)
            swap = random.random() < 0.5  # hide which answer is the baseline from the judge
            a_text, b_text = (b.text, out.text) if swap else (out.text, b.text)
            jp, _ = self._target(judge.id, decision.complexity, None, mode)
            j = await jp.complete(judge, judge_prompt(last_prompt(msgs), a_text, b_text), 400)
        except (ProviderError, httpx.HTTPError, KeyError, IndexError) as e:
            log.warning("shadow eval failed: %s", e)
            return None
        v = parse_verdict(j.text)
        if not v:
            return None
        if swap:
            v["score_routed"], v["score_baseline"] = v["score_baseline"], v["score_routed"]
            v["verdict"] = {"routed_better": "baseline_better", "baseline_better": "routed_better"}.get(v["verdict"], "equal")
        overhead = [self.estimator.estimate(base, b.input_tokens, b.output_tokens),
                    self.estimator.estimate(judge, j.input_tokens, j.output_tokens)]
        row = {"ts": time.time(), "request_id": out.request_id, "tier": decision.complexity["tier"],
               "task_type": decision.complexity["task_type"], "model_id": out.model_id, "baseline_model": base.id,
               "judge_model": judge.id, **v,
               "overhead_energy_wh": sum(x.energy_wh for x in overhead),
               "overhead_cost_usd": sum(x.cost_usd for x in overhead)}
        row["id"] = self.ledger.eval_put(row)
        return row

    async def eval_request(self, request_id: int) -> dict | None:
        """Run a shadow evaluation for an earlier live request, on demand."""
        row, p = self.ledger.get(request_id), self.ledger.payload(request_id)
        if not row or not p:
            raise LookupError(f"Request {request_id} or its stored prompt was not found")
        if row["source"] != "live":
            raise ValueError("Only live answers can be evaluated; simulated and cached answers are skipped.")
        decision = await self.decide(messages=p["messages"], policy=row["policy"])
        out = ChatOutcome(decision, row["model_id"], p["response"], "live", {}, {}, {}, 0.0, False, request_id, [])
        return await self.shadow_eval(out, p["messages"])

    def _finish(self, decision, model_id, res: ProviderResult, source, imp, base, fallback, errors, msgs, team):
        model = self.catalog.get(model_id)
        savings = {k: base[k] - imp[k] for k in ("energy_wh", "carbon_g", "water_ml", "cost_usd")}
        for k in ("energy_wh", "carbon_g", "water_ml", "cost_usd"):
            savings[k.split("_")[0] + "_pct"] = round(100 * savings[k] / base[k], 2) if base[k] else 0.0
        sust = 100.0 if source == "cache" else sustainability_score(Impact(**imp))
        cx = decision.complexity
        rid = self.ledger.record({
            "ts": time.time(),
            "prompt_preview": (last_prompt(msgs)[:120] if self.settings.store_prompts else None),
            "complexity": cx["score"], "tier": cx["tier"], "task_type": cx["task_type"],
            "policy": decision.policy, "model_id": model_id, "provider": model.provider, "source": source,
            "input_tokens": imp["input_tokens"], "output_tokens": imp["output_tokens"], "latency_ms": res.latency_ms,
            "energy_wh": imp["energy_wh"], "carbon_g": imp["carbon_g"], "water_ml": imp["water_ml"],
            "cost_usd": imp["cost_usd"], "baseline_model": decision.baseline_model,
            "baseline_energy_wh": base["energy_wh"], "baseline_carbon_g": base["carbon_g"],
            "baseline_water_ml": base["water_ml"], "baseline_cost_usd": base["cost_usd"],
            "sustainability_score": sust, "fallback_used": int(fallback), "team": team,
        })
        if self.settings.store_prompts and rid:
            self.ledger.payload_put(rid, msgs, res.text, 7 * 86400)
        return ChatOutcome(decision, model_id, res.text, source, imp, base, savings, res.latency_ms, fallback, rid,
                           errors, res.tool_calls, res.finish_reason, team)

    # ------------------------------------------------------------------ carbon-aware batches
    async def submit_batch(self, items: list[dict], deadline_hours: float = 24, team: str | None = None,
                           policy: str = "eco", max_carbon_g_per_kwh: float | None = None) -> dict:
        """Queue non-urgent requests to run when the grid is cleaner (or at the deadline, whichever comes first)."""
        if not items:
            raise ValueError("A batch needs at least one request.")
        if len(items) > 1000:
            raise ValueError("At most 1000 requests per batch.")
        for it in items:
            self._normalise(it.get("prompt"), it.get("messages"))  # validate now, not hours later
        await self.refresh_grid()
        first = await self.decide(items[0].get("prompt"), items[0].get("messages"), policy, items[0].get("constraints"))
        region = self.catalog.region_for(self.catalog.get(first.model_id))
        job = {"id": f"batch_{uuid.uuid4().hex[:16]}", "created": time.time(),
               "deadline": time.time() + max(0.0, deadline_hours) * 3600, "status": "queued", "team": team,
               "policy": policy, "max_carbon_g_per_kwh": max_carbon_g_per_kwh,
               "submit_g_per_kwh": region.carbon_g_per_kwh, "n": len(items)}
        self.ledger.batch_create(job, items)
        return self.ledger.batch_get(job["id"])

    def _grid_is_clean(self, region, threshold: float | None) -> bool:
        """Clean = live intensity at or below the threshold (default: 85% of the region's average)."""
        if not region.live:
            return False  # without live data there's no signal to wait for; jobs run at their deadline
        return region.carbon_g_per_kwh <= (threshold or 0.85 * region.average_g_per_kwh)

    async def run_batches(self, force: bool = False, limit: int = 200) -> dict:
        """Run queued batch items whose grid is clean now, or whose deadline is within the next hour."""
        await self.refresh_grid()
        ran = waiting = 0
        for job in self.ledger.batch_jobs():
            due = force or time.time() >= job["deadline"] - 3600
            full = self.ledger.batch_get(job["id"])
            for it in full["items"]:
                if it["status"] != "queued" or ran >= limit:
                    continue
                req = it["request"]
                d = await self.decide(req.get("prompt"), req.get("messages"), req.get("policy") or job["policy"],
                                      req.get("constraints"))
                region = self.catalog.region_for(self.catalog.get(d.model_id))
                if not (due or self._grid_is_clean(region, job["max_carbon_g_per_kwh"])):
                    waiting += 1
                    continue
                try:
                    out = await self.chat(req.get("prompt"), req.get("messages"), req.get("policy") or job["policy"],
                                          req.get("constraints"), team=job["team"])
                    self.ledger.batch_item_update(job["id"], it["idx"], status="done", request_id=out.request_id,
                                                  response=out.text, model_id=out.model_id,
                                                  grid_g_per_kwh=region.carbon_g_per_kwh, ran=time.time())
                except (ValueError, RuntimeError) as e:
                    self.ledger.batch_item_update(job["id"], it["idx"], status="failed", error=str(e)[:500],
                                                  ran=time.time())
                ran += 1
            items = self.ledger.batch_get(job["id"])["items"]
            done = sum(i["status"] != "queued" for i in items)
            finished = done == len(items)
            self.ledger.batch_update(job["id"], done=done, status="completed" if finished else "running" if done else "queued",
                                     **({"finished": time.time()} if finished else {}))
        return {"ran": ran, "waiting": waiting}

    # ------------------------------------------------------------------ catalog view
    def models_view(self, available: set[str]) -> list[dict]:
        out = []
        for m in self.catalog.models:
            prof = self.estimator.profiles[m.id]
            ref = self.estimator.estimate(m, 200, 500)  # a "typical" request
            region = self.catalog.region_for(m)
            out.append({
                **m.to_dict(), "available": m.id in available, "energy": prof.to_dict(),
                "typical_request": ref.to_dict(), "sustainability_score": sustainability_score(ref),
                "grid_region": region.key, "grid_carbon_g_per_kwh": region.carbon_g_per_kwh,
                "grid_live": region.live,
            })
        out.sort(key=lambda d: -d["sustainability_score"])
        return out


def last_prompt(msgs: list[dict]) -> str:
    return next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")


def _norm_message(m: dict) -> dict:
    """Keep the OpenAI fields providers need; flatten multi-part content to its text parts."""
    content = m.get("content")
    if isinstance(content, list):
        content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    out = {"role": m.get("role") or "user", "content": "" if content is None else str(content)}
    for k in ("tool_calls", "tool_call_id", "name"):
        if m.get(k):
            out[k] = m[k]
    return out
