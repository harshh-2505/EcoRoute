"""Command-line interface: `ecoroute serve | route | chat | models | analyze | bench`."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .complexity import analyze
from .service import EcoRouteService


def _fmt_wh(x: float) -> str:
    return f"{x * 1000:.2f} mWh" if x < 1 else f"{x:.3f} Wh"


def cmd_serve(a):
    import uvicorn

    from .api import create_app
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="info")


def cmd_analyze(a):
    print(json.dumps(analyze(a.prompt).to_dict(), indent=2))


def cmd_route(a):
    svc = EcoRouteService()
    d = asyncio.run(svc.decide(a.prompt, policy=a.policy, constraints=_constraints(a)))
    if a.json:
        print(json.dumps(d.to_dict(), indent=2))
        return
    cx = d.complexity
    print(f"Complexity : {cx['score']:.2f} → tier {cx['tier']} ({cx['tier_name']}, {cx['task_type']})")
    print(f"Chosen     : {d.display_name}  [{d.model_id}]  policy={d.policy}")
    e, b = d.estimated, d.baseline_estimated
    print(f"Estimate   : {_fmt_wh(e['energy_wh'])}, {e['carbon_g']:.4f} gCO2e, {e['water_ml']:.2f} mL water, "
          f"${e['cost_usd']:.6f}")
    print(f"Baseline   : {_fmt_wh(b['energy_wh'])}, ${b['cost_usd']:.6f}  ({d.baseline_model})")
    for r in d.reasons:
        print(f"  • {r}")
    print("\nTop candidates:")
    for c in [c for c in d.candidates if c.eligible][:5]:
        print(f"  {c.total:6.1f}  {c.display_name:28}  sust {c.scores['sustainability']:5.1f}  "
              f"cost {c.scores['cost']:5.1f}  quality {c.scores['quality']:5.1f}")


def cmd_chat(a):
    svc = EcoRouteService()
    out = asyncio.run(svc.chat(a.prompt, policy=a.policy, constraints=_constraints(a), use_cache=not a.no_cache))
    if a.json:
        print(json.dumps(out.to_dict(), indent=2))
        return
    print(out.text)
    i, s = out.impact, out.savings
    print(f"\n— {out.model_id} ({out.source}) · {_fmt_wh(i['energy_wh'])} · {i['carbon_g']:.4f} gCO2e · "
          f"{i['water_ml']:.2f} mL · ${i['cost_usd']:.6f} · saved {s['energy_pct']:.0f}% energy vs baseline",
          file=sys.stderr)


def cmd_models(a):
    svc = EcoRouteService()
    available, mode = asyncio.run(svc.availability())
    print(f"mode={mode}  baseline={svc.router.baseline_model}\n")
    print(f"{'model':30} {'label':5} {'Wh/1k out':>10} {'method':14} {'$in/M':>7} {'$out/M':>7} {'tier':>4} {'avail':>5}")
    for m in svc.models_view(available):
        e = m["energy"]
        print(f"{m['id']:30} {e['label']:5} {e['wh_per_1k_output']:10.4f} {e['method']:14} "
              f"{m['price_in']:7.2f} {m['price_out']:7.2f} {m['max_tier']:4} {'yes' if m['available'] else '-':>5}")


def cmd_bench(a):
    from benchmarks.run_benchmark import main as bench_main  # type: ignore
    bench_main(a.output)


def _constraints(a) -> dict:
    c = {}
    if getattr(a, "local", False):
        c["require_local"] = True
    if getattr(a, "model", None):
        c["force_model"] = a.model
    return c


def main(argv=None):
    p = argparse.ArgumentParser(prog="ecoroute", description="Sustainability-aware LLM router")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("analyze", help="score prompt complexity")
    s.add_argument("prompt")
    s.set_defaults(fn=cmd_analyze)

    for name, fn, hlp in (("route", cmd_route, "show the routing decision (no model call)"),
                          ("chat", cmd_chat, "route and answer")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("prompt")
        s.add_argument("--policy", choices=["eco", "balanced", "quality", "cost"])
        s.add_argument("--local", action="store_true", help="only use local models")
        s.add_argument("--model", help="force a specific catalog model id")
        s.add_argument("--json", action="store_true")
        if name == "chat":
            s.add_argument("--no-cache", action="store_true")
        s.set_defaults(fn=fn)

    s = sub.add_parser("models", help="list models with efficiency labels")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("bench", help="run the routing benchmark (simulate mode)")
    s.add_argument("--output", default="benchmarks/RESULTS.md")
    s.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
