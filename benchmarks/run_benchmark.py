"""Benchmark EcoRoute on the labelled prompt set (simulate mode, no API keys needed).

Reports
  1. complexity-classifier accuracy (train split used for threshold calibration, holdout untouched)
  2. per-policy estimated energy / carbon / water / cost versus always using the baseline model

Usage:  python -m benchmarks.run_benchmark [output.md]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ecoroute.complexity import analyze
from ecoroute.config import Settings
from ecoroute.impact import equivalents
from ecoroute.service import EcoRouteService

PROMPTS = Path(__file__).with_name("prompts.jsonl")


def load():
    return [json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()]


def classifier_report(rows):
    out = {}
    for split in ("train", "holdout"):
        rr = [r for r in rows if r.get("split", "train") == split]
        preds = [analyze(r["prompt"]).tier for r in rr]
        exact = sum(p == r["tier"] for p, r in zip(preds, rr))
        near = sum(abs(p - r["tier"]) <= 1 for p, r in zip(preds, rr))
        under = sum(p < r["tier"] for p, r in zip(preds, rr))
        out[split] = {"n": len(rr), "exact": exact / len(rr), "within_1": near / len(rr),
                      "under_routed": under / len(rr)}
    return out


async def policy_report(rows):
    results = {}
    for policy in ("eco", "balanced", "quality", "cost"):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings.from_env()
            s.mode, s.db_path, s.cache_enabled = "simulate", os.path.join(tmp, "b.db"), False
            svc = EcoRouteService(s)
            svc.simulator.delay = False
            picks = Counter()
            for r in rows:
                out = await svc.chat(r["prompt"], policy=policy, use_cache=False)
                picks[out.model_id] += 1
            summ = svc.ledger.summary()
            t = summ["totals"]
            svc.ledger.close()
        results[policy] = {
            "energy_wh": t["energy_wh"], "baseline_energy_wh": t["b_energy"],
            "carbon_g": t["carbon_g"], "baseline_carbon_g": t["b_carbon"],
            "water_ml": t["water_ml"], "baseline_water_ml": t["b_water"],
            "cost_usd": t["cost_usd"], "baseline_cost_usd": t["b_cost"],
            "picks": picks.most_common(),
        }
    return results


def pct(a, b):
    return 100 * (b - a) / b if b else 0.0


def render(cls, pol, n, baseline):
    lines = ["# EcoRoute benchmark results", "",
             f"{n} labelled prompts, simulate mode, baseline = always `{baseline}`.",
             "All impact figures are *estimates* from the methodology in docs/METHODOLOGY.md.", "",
             "## Complexity classifier", "",
             "| split | prompts | exact tier | within ±1 tier | under-routed |",
             "|---|---|---|---|---|"]
    for split, r in cls.items():
        lines.append(f"| {split} | {r['n']} | {r['exact']:.0%} | {r['within_1']:.0%} | {r['under_routed']:.0%} |")
    lines += ["", "Thresholds were calibrated on the *train* split; *holdout* prompts were written afterwards "
              "and never used for tuning.", "",
              "## Savings versus always using the baseline model", "",
              "| policy | energy (Wh) | baseline (Wh) | energy saved | carbon saved | water saved | cost saved |",
              "|---|---|---|---|---|---|---|"]
    for p, r in pol.items():
        lines.append(f"| {p} | {r['energy_wh']:.3f} | {r['baseline_energy_wh']:.3f} | "
                     f"{pct(r['energy_wh'], r['baseline_energy_wh']):.1f}% | "
                     f"{pct(r['carbon_g'], r['baseline_carbon_g']):.1f}% | "
                     f"{pct(r['water_ml'], r['baseline_water_ml']):.1f}% | "
                     f"{pct(r['cost_usd'], r['baseline_cost_usd']):.1f}% |")
    lines += ["", "## Models chosen per policy", ""]
    for p, r in pol.items():
        lines.append(f"- **{p}**: " + ", ".join(f"`{m}` × {c}" for m, c in r["picks"]))
    b = pol["balanced"]
    eq = equivalents(b["baseline_energy_wh"] - b["energy_wh"], b["baseline_carbon_g"] - b["carbon_g"],
                     b["baseline_water_ml"] - b["water_ml"])
    scale = 1_000_000 / n
    lines += ["", "## What this means at scale (balanced policy)", "",
              f"Extrapolated to **1 million** prompts with this mix: "
              f"~{(b['baseline_energy_wh'] - b['energy_wh']) * scale / 1000:.0f} kWh, "
              f"~{(b['baseline_carbon_g'] - b['carbon_g']) * scale / 1000:.0f} kg CO2e, "
              f"~{(b['baseline_water_ml'] - b['water_ml']) * scale / 1000:.0f} L of water and "
              f"~${(b['baseline_cost_usd'] - b['cost_usd']) * scale:,.0f} saved.",
              f"(Per {n} prompts: {eq['phone_charges']:.2f} phone charges' worth of energy.)", ""]
    return "\n".join(lines)


def main(output: str | None = None):
    rows = load()
    cls = classifier_report(rows)
    pol = asyncio.run(policy_report(rows))
    baseline = Settings.from_env().baseline_model
    md = render(cls, pol, len(rows), baseline)
    print(md)
    if output:
        Path(output).write_text(md)
        print(f"\nwritten to {output}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "benchmarks" / "RESULTS.md"))
