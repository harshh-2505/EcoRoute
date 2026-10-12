"""Monthly sustainability (ESG) report: totals, savings and uncertainty ranges, as JSON, CSV or printable HTML."""
from __future__ import annotations

import csv
import html
import io
from datetime import datetime, timezone

from .impact import UNCERTAINTY, equivalents

METRICS = ("energy_wh", "carbon_g", "water_ml", "cost_usd")


def month_bounds(month: str | None) -> tuple[str, float, float]:
    """``"2026-10"`` → (label, start_ts, end_ts) in UTC. ``None`` = the current month."""
    now = datetime.now(timezone.utc)
    if month:
        try:
            start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
        except ValueError:
            raise ValueError("month must look like 2026-10") from None
    else:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    return start.strftime("%Y-%m"), start.timestamp(), end.timestamp()


def build(svc, month: str | None = None, team: str | None = None) -> dict:
    label, start, end = month_bounds(month)
    rows = svc.ledger.period_by_model(start, end, team)
    by_model, totals = [], dict.fromkeys(METRICS, 0.0) | {"n": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
    base = dict.fromkeys(METRICS, 0.0)
    low, high = dict.fromkeys(METRICS[:3], 0.0), dict.fromkeys(METRICS[:3], 0.0)
    for r in rows:
        prof = svc.estimator.profiles.get(r["model_id"])
        method = prof.method if prof else "price"
        f = UNCERTAINTY.get(method, 2.5)
        for k in METRICS:
            totals[k] += r[k] or 0.0
            base[k] += r["baseline_" + k] or 0.0
        for k in METRICS[:3]:
            low[k] += (r[k] or 0.0) / f
            high[k] += (r[k] or 0.0) * f
        for k in ("n", "cache_hits", "input_tokens", "output_tokens"):
            totals[k] += r[k] or 0
        by_model.append({**r, "method": method, "confidence": prof.confidence if prof else "low",
                         "uncertainty_factor": f})
    saved = {k: base[k] - totals[k] for k in METRICS}
    return {
        "month": label, "team": team, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": {"start": start, "end": end}, "baseline_model": svc.router.baseline_model,
        "totals": totals, "baseline": base, "saved": saved,
        "saved_pct": {k: (100 * saved[k] / base[k] if base[k] else 0.0) for k in METRICS},
        "range": {"low": low, "high": high},
        "equivalents_saved": equivalents(max(saved["energy_wh"], 0), max(saved["carbon_g"], 0),
                                         max(saved["water_ml"], 0)),
        "by_model": by_model,
        "by_team": [] if team else svc.ledger.period_by_team(start, end),
        "quality": svc.ledger.quality(start)["evals"],
        "methodology": {
            "energy": "Open-weight models: 2 × active parameters FLOPs per token at 2e11 FLOP/J × PUE, blended with a "
                      "price signal. Closed models: list price as an energy proxy. Local models: device watts ÷ tokens/s.",
            "carbon": "Energy × grid carbon intensity of the serving region (live from Electricity Maps when configured).",
            "water": "On-site cooling (WUE × IT energy) + off-site water used to generate the electricity.",
            "baseline": f"The same requests, with the same token counts, sent to {svc.router.baseline_model}.",
            "uncertainty": "Ranges divide and multiply each model's figures by a factor for its estimation method: "
                           + ", ".join(f"{k} ×/÷{v}" for k, v in UNCERTAINTY.items()) + ".",
            "scope": "Inference only (no training, embodied hardware or networking). Cache hits count as zero.",
        },
    }


def to_csv(r: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["month", "model_id", "provider", "requests", "cache_hits", "input_tokens", "output_tokens",
                "energy_wh", "energy_wh_low", "energy_wh_high", "carbon_g", "water_ml", "cost_usd",
                "baseline_energy_wh", "baseline_carbon_g", "baseline_water_ml", "baseline_cost_usd", "estimate_method"])
    for m in r["by_model"]:
        f = m["uncertainty_factor"]
        w.writerow([r["month"], m["model_id"], m["provider"], m["n"], m["cache_hits"], m["input_tokens"],
                    m["output_tokens"], _r(m["energy_wh"]), _r(m["energy_wh"] / f), _r(m["energy_wh"] * f),
                    _r(m["carbon_g"]), _r(m["water_ml"]), _r(m["cost_usd"]), _r(m["baseline_energy_wh"]),
                    _r(m["baseline_carbon_g"]), _r(m["baseline_water_ml"]), _r(m["baseline_cost_usd"]), m["method"]])
    t, b = r["totals"], r["baseline"]
    w.writerow([r["month"], "TOTAL", "", t["n"], t["cache_hits"], t["input_tokens"], t["output_tokens"],
                _r(t["energy_wh"]), _r(r["range"]["low"]["energy_wh"]), _r(r["range"]["high"]["energy_wh"]),
                _r(t["carbon_g"]), _r(t["water_ml"]), _r(t["cost_usd"]), _r(b["energy_wh"]), _r(b["carbon_g"]),
                _r(b["water_ml"]), _r(b["cost_usd"]), ""])
    return buf.getvalue()


def _r(x) -> float:
    return round(x or 0.0, 6)


def _wh(x: float) -> str:
    return f"{x / 1000:,.3f} kWh" if x >= 1000 else f"{x:,.2f} Wh" if x >= 1 else f"{x * 1000:,.1f} mWh"


def _g(x: float) -> str:
    return f"{x / 1000:,.3f} kg" if x >= 1000 else f"{x:,.2f} g"


def _ml(x: float) -> str:
    return f"{x / 1000:,.2f} L" if x >= 1000 else f"{x:,.1f} mL"


def _usd(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 or x == 0 else f"${x:.5f}"


def to_html(r: dict) -> str:
    e = html.escape
    t, b, s, p, lo, hi = r["totals"], r["baseline"], r["saved"], r["saved_pct"], r["range"]["low"], r["range"]["high"]
    q = r["quality"]
    parity = (f"{100 * q['parity_rate']:.0f}% of {q['n']} graded answers were judged as good as or better than "
              f"{e(r['baseline_model'])}." if q.get("n") else "No shadow evaluations this month.")
    rows = "".join(
        f"<tr><td>{e(m['model_id'])}</td><td>{m['n']:,}</td><td>{_wh(m['energy_wh'])}</td>"
        f"<td>{_wh(m['energy_wh'] / m['uncertainty_factor'])} – {_wh(m['energy_wh'] * m['uncertainty_factor'])}</td>"
        f"<td>{_g(m['carbon_g'])}</td><td>{_ml(m['water_ml'])}</td><td>{_usd(m['cost_usd'])}</td>"
        f"<td>{e(m['method'])}</td></tr>" for m in r["by_model"]) or '<tr><td colspan="8">No requests.</td></tr>'
    teams = "".join(
        f"<tr><td>{e(x['team'])}</td><td>{x['n']:,}</td><td>{_wh(x['energy_wh'] or 0)}</td>"
        f"<td>{_g(x['carbon_g'] or 0)}</td><td>{_usd(x['cost_usd'] or 0)}</td>"
        f"<td>{_wh((x['baseline_energy_wh'] or 0) - (x['energy_wh'] or 0))}</td></tr>" for x in r["by_team"])
    meth = "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in r["methodology"].items())
    kpi = [("Requests", f"{t['n']:,}", f"{t['cache_hits']:,} from cache"),
           ("Energy", _wh(t["energy_wh"]), f"range {_wh(lo['energy_wh'])} – {_wh(hi['energy_wh'])}"),
           ("CO₂e", _g(t["carbon_g"]), f"range {_g(lo['carbon_g'])} – {_g(hi['carbon_g'])}"),
           ("Water", _ml(t["water_ml"]), f"range {_ml(lo['water_ml'])} – {_ml(hi['water_ml'])}"),
           ("Spend", _usd(t["cost_usd"]), f"baseline {_usd(b['cost_usd'])}")]
    saved = [("Energy avoided", _wh(s["energy_wh"]), p["energy_wh"]), ("CO₂e avoided", _g(s["carbon_g"]), p["carbon_g"]),
             ("Water saved", _ml(s["water_ml"]), p["water_ml"]), ("Spend saved", _usd(s["cost_usd"]), p["cost_usd"])]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EcoRoute sustainability report · {e(r['month'])}</title>
<style>
:root{{--ink:#0d1210;--muted:#5b6660;--line:#d9dfdb;--accent:#3d7a1f;--bg:#fbfcfa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:40px 24px}}h1{{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
.k{{border:1px solid var(--line);padding:12px 14px;background:#fff}}.k b{{display:block;font-size:20px;font-variant-numeric:tabular-nums}}
.k small{{color:var(--muted)}}.k.save b{{color:var(--accent)}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}}
th{{font-size:12px;color:var(--muted);font-weight:600}}.wrap{{overflow-x:auto}}dt{{font-weight:600;margin-top:8px;text-transform:capitalize}}dd{{margin:2px 0 0;color:var(--muted)}}
.bar{{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}}button{{font:inherit;padding:6px 12px;border:1px solid var(--ink);background:none;cursor:pointer}}
@media print{{button{{display:none}}main{{padding:0}}.k{{break-inside:avoid}}}}
</style></head><body><main>
<div class="bar"><div><h1>AI inference sustainability report</h1>
<div class="sub">{e(r['month'])}{' · team ' + e(r['team']) if r['team'] else ''} · generated {e(r['generated'])} UTC · by EcoRoute</div></div>
<button onclick="print()">Save as PDF</button></div>
<h2>Footprint</h2><div class="grid">{''.join(f'<div class="k"><small>{a}</small><b>{v}</b><small>{c}</small></div>' for a, v, c in kpi)}</div>
<h2>Avoided versus always using {e(r['baseline_model'])}</h2>
<div class="grid">{''.join(f'<div class="k save"><small>{a}</small><b>{v}</b><small>{pc:.0f}% lower</small></div>' for a, v, pc in saved)}</div>
<p class="sub">In everyday terms: {r['equivalents_saved']['phone_charges']:,.1f} phone charges, {r['equivalents_saved']['car_km']:,.2f} km not driven,
{r['equivalents_saved']['water_glasses']:,.1f} glasses of water.</p>
<h2>Answer quality</h2><p>{parity}</p>
<h2>By model</h2><div class="wrap"><table><thead><tr><th>Model</th><th>Requests</th><th>Energy</th><th>Range</th><th>CO₂e</th><th>Water</th><th>Spend</th><th>Estimate</th></tr></thead>
<tbody>{rows}</tbody></table></div>
{f'<h2>By team</h2><div class="wrap"><table><thead><tr><th>Team</th><th>Requests</th><th>Energy</th><th>CO₂e</th><th>Spend</th><th>Energy avoided</th></tr></thead><tbody>{teams}</tbody></table></div>' if teams else ''}
<h2>Methodology and scope</h2><dl>{meth}</dl>
<p class="sub">All figures are estimates. See docs/METHODOLOGY.md in the EcoRoute repository for formulas and constants.</p>
</main></body></html>"""
