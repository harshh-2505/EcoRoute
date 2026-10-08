/* EcoRoute dashboard — vanilla JS + Chart.js */
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const fmt = {
  wh: (x) => (x < 1 ? `${(x * 1000).toFixed(x < 0.01 ? 2 : 1)} mWh` : `${x.toFixed(2)} Wh`),
  g: (x) => (x < 1 ? `${(x * 1000).toFixed(1)} mg` : x < 1000 ? `${x.toFixed(2)} g` : `${(x / 1000).toFixed(2)} kg`),
  ml: (x) => (x < 1000 ? `${x.toFixed(x < 10 ? 2 : 1)} mL` : `${(x / 1000).toFixed(2)} L`),
  usd: (x) => (x === 0 ? "$0" : x < 0.01 ? `$${x.toFixed(5)}` : `$${x.toFixed(2)}`),
  pct: (x) => `${x >= 0 ? "" : "+"}${Math.abs(x).toFixed(0)}%`,
  n: (x) => Intl.NumberFormat().format(Math.round(x)),
};
const LABEL_COLORS = { A: "--l-A", B: "--l-B", C: "--l-C", D: "--l-D", E: "--l-E", F: "--l-F", G: "--l-G" };
const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const labelChip = (l) => `<span class="label" style="background:var(${LABEL_COLORS[l]})">${l}</span>`;

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

/* ---------------- tabs & segmented controls ---------------- */
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === t.dataset.tab));
  if (t.dataset.tab === "impact") loadImpact();
  if (t.dataset.tab === "models") loadModels();
}));
function seg(id, onChange) {
  const el = $(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    onChange && onChange(b.dataset.v);
  });
  return () => el.querySelector("button.on").dataset.v;
}
const policy = seg("#policy");
const range = seg("#range", () => loadImpact());

/* ---------------- status ---------------- */
async function loadStatus() {
  try {
    const h = await api("/health");
    const prov = Object.entries(h.providers_configured).filter(([, v]) => v).map(([k]) => k);
    $("#status").innerHTML = `mode <b>${h.mode}</b> · ${h.available_models} models` +
      (prov.length ? ` · ${prov.join(", ")}` : h.mode === "simulate" ? " · no API keys" : "");
  } catch { $("#status").textContent = "API unreachable"; }
}

/* ---------------- playground ---------------- */
const EXAMPLES = [
  "What is the capital of Japan?",
  "Explain why the sky is blue.",
  "Write a Python function to reverse a linked list and add unit tests.",
  "Analyze the trade-offs of training a model in a low-carbon region versus close to users.",
  "Design a distributed, fault-tolerant architecture for real-time smart-grid telemetry across 5 states, covering ingestion, storage, analytics and alerting, and justify each technology choice.",
];
$("#examples").innerHTML = EXAMPLES.map((e, i) => `<button data-i="${i}">${esc(e.length > 46 ? e.slice(0, 44) + "…" : e)}</button>`).join("");
$("#examples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { $("#prompt").value = EXAMPLES[b.dataset.i]; } });

function payload() {
  const c = {};
  if ($("#localOnly").checked) c.require_local = true;
  return { prompt: $("#prompt").value.trim(), policy: policy(), constraints: c, use_cache: $("#useCache").checked };
}

async function go(run) {
  const p = payload();
  if (!p.prompt) { $("#prompt").focus(); return; }
  const btns = [$("#btnRoute"), $("#btnRun")]; btns.forEach((b) => (b.disabled = true));
  try {
    if (run) {
      const out = await api("/v1/chat", { method: "POST", body: JSON.stringify(p) });
      renderDecision(out.decision, out);
    } else {
      const d = await api("/v1/route", { method: "POST", body: JSON.stringify(p) });
      renderDecision(d, null);
    }
  } catch (err) { alert(err.message); }
  finally { btns.forEach((b) => (b.disabled = false)); }
}
$("#btnRoute").addEventListener("click", () => go(false));
$("#btnRun").addEventListener("click", () => go(true));
$("#prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) go(true); });

function renderDecision(d, out) {
  $("#result").classList.remove("hidden");
  const cx = d.complexity;
  const modelId = out ? out.model_id : d.model_id;
  const cand = d.candidates.find((c) => c.model_id === modelId) || {};
  $("#rChosen").textContent = cand.display_name || d.display_name;
  $("#rMeta").innerHTML = `${esc(cand.provider || d.provider)} · ${cand.kind === "local" ? "on-device" : "hosted"} · policy <b>${esc(d.policy)}</b>` +
    (out && out.fallback_used ? ` · <span style="color:var(--warn)">fallback used</span>` : "");
  $("#rScore").textContent = `${cx.score.toFixed(2)} · tier ${cx.tier} ${cx.tier_name} · ${cx.task_type}`;
  $("#rScoreBar").style.width = `${Math.max(2, cx.score * 100)}%`;
  $("#rContrib").innerHTML = Object.entries(cx.contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([k, v]) => `<span class="${v < 0 ? "neg" : ""}">${esc(k.replace(/_/g, " "))} ${v > 0 ? "+" : ""}${v.toFixed(2)}</span>`).join("");
  $("#rReasons").innerHTML = d.reasons.map((r) => `<li>${esc(r)}</li>`).join("");

  const imp = out ? out.impact : d.estimated;
  const base = out ? out.baseline : d.baseline_estimated;
  $("#impactTitle").textContent = out ? (out.source === "cache" ? "Impact — served from cache" : "Measured impact vs. baseline") : "Estimated impact vs. baseline";
  const cells = [
    ["Energy", fmt.wh, "energy_wh"], ["Carbon (CO₂e)", fmt.g, "carbon_g"],
    ["Water", fmt.ml, "water_ml"], ["Cost", fmt.usd, "cost_usd"],
  ];
  $("#rImpact").innerHTML = cells.map(([k, f, key]) => {
    const s = base[key] ? (100 * (base[key] - imp[key])) / base[key] : 0;
    return `<div class="imp"><div class="k">${k}</div><div class="v">${f(imp[key])}</div>
      <div class="b">baseline ${f(base[key])} · <span class="save ${s < 0 ? "neg" : ""}">${s >= 0 ? "−" : "+"}${Math.abs(s).toFixed(0)}%</span></div></div>`;
  }).join("");
  const baseName = (d.candidates.find((c) => c.model_id === d.baseline_model) || {}).display_name || d.baseline_model;
  $("#rBaseline").textContent = `Baseline = sending the same request to ${baseName}. ${imp.input_tokens} input + ${imp.output_tokens} output tokens${out ? "" : " (expected)"}.`;

  $("#responseCard").classList.toggle("hidden", !out);
  if (out) {
    $("#rSource").textContent = out.source + (out.latency_ms ? ` · ${Math.round(out.latency_ms)} ms` : "");
    $("#rText").textContent = out.response;
  }

  const bar = (v) => `<span class="mono">${v.toFixed(0)}</span><span class="scorebar"><i style="width:${v}%"></i></span>`;
  $("#rCands").innerHTML = `<thead><tr><th>Model</th><th>Score</th><th>Sustainability</th><th>Cost</th><th>Quality</th><th>Est. energy</th><th>Est. cost</th><th>Status</th></tr></thead><tbody>` +
    d.candidates.map((c) => `<tr class="${c.model_id === modelId ? "win" : c.eligible ? "" : "dim"}">
      <td>${esc(c.display_name)}</td><td class="num">${c.eligible ? c.total.toFixed(1) : "—"}</td>
      <td>${bar(c.scores.sustainability)}</td><td>${bar(c.scores.cost)}</td><td>${bar(c.scores.quality)}</td>
      <td class="num">${fmt.wh(c.estimated.energy_wh)}</td><td class="num">${fmt.usd(c.estimated.cost_usd)}</td>
      <td>${c.model_id === modelId ? "✔ chosen" : c.eligible ? "eligible" : esc(c.reason)}</td></tr>`).join("") + "</tbody>";
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------------- impact dashboard ---------------- */
const charts = {};
function chart(id, cfg) {
  charts[id]?.destroy();
  const ink = cssv("--muted"), line = cssv("--line");
  Chart.defaults.color = ink; Chart.defaults.borderColor = line;
  Chart.defaults.font.family = "IBM Plex Sans";
  charts[id] = new Chart($(id), cfg);
}

async function loadImpact() {
  const days = Number(range());
  const [s, ts, recent] = await Promise.all([
    api(`/v1/metrics/summary?days=${days}`),
    api(`/v1/metrics/timeseries?days=${days || 3650}&bucket=${days && days <= 1 ? "minute" : days <= 7 ? "hour" : "day"}`),
    api("/v1/metrics/requests?limit=25"),
  ]);
  const t = s.totals;
  const k = (label, v, sub) => `<div class="kpi"><div class="k">${label}</div><div class="v">${v}</div><div class="s">${sub || "&nbsp;"}</div></div>`;
  $("#kpis").innerHTML = [
    k("Requests", fmt.n(t.n), `${(t.cache_hit_rate * 100).toFixed(0)}% served from cache`),
    k("Energy used", fmt.wh(t.energy_wh), `${s.saved_pct.energy_wh.toFixed(0)}% less than baseline`),
    k("Energy saved", fmt.wh(s.saved.energy_wh), `${fmt.wh(t.energy_per_1k_tokens_wh)} per 1k tokens`),
    k("CO₂e avoided", fmt.g(s.saved.carbon_g), `${s.saved_pct.carbon_g.toFixed(0)}% lower`),
    k("Water saved", fmt.ml(s.saved.water_ml), `${s.saved_pct.water_ml.toFixed(0)}% lower`),
    k("Spend", fmt.usd(t.cost_usd), `${fmt.usd(s.saved.cost_usd)} saved (${s.saved_pct.cost_usd.toFixed(0)}%)`),
    k("Avg sustainability", `${t.avg_sust.toFixed(0)}/100`, "per request, higher is greener"),
    k("Avg latency", `${fmt.n(t.avg_latency)} ms`, `${t.fallbacks} fallback(s)`),
  ].join("");
  const e = s.equivalents_saved;
  $("#equiv").innerHTML = t.n ? `<span class="eyebrow" style="margin:0">Savings, in everyday terms</span>
    <span><b>${e.phone_charges.toFixed(2)}</b> phone charges</span>
    <span><b>${e.led_bulb_minutes.toFixed(0)}</b> min of a 10 W LED</span>
    <span><b>${e.car_km.toFixed(3)}</b> km not driven</span>
    <span><b>${e.water_glasses.toFixed(2)}</b> glasses of water</span>`
    : `<span class="empty">No requests yet — run a prompt in the Playground or click “Load demo traffic”.</span>`;

  const acc = cssv("--accent"), warn = cssv("--warn"), muted = cssv("--muted");
  const labels = ts.map((r) => {
    const d = new Date(r.bucket * 1000);
    return days && days <= 1 ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : days <= 7 && days ? d.toLocaleString([], { weekday: "short", hour: "2-digit" }) : d.toLocaleDateString();
  });
  chart("#chTime", { type: "line", data: { labels, datasets: [
    { label: "EcoRoute", data: ts.map((r) => r.energy_wh), borderColor: acc, backgroundColor: acc + "33", fill: true, tension: .3, pointRadius: 2 },
    { label: "Baseline", data: ts.map((r) => r.baseline_energy_wh), borderColor: warn, borderDash: [5, 4], tension: .3, pointRadius: 0 },
  ] }, options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } } } });

  const tiers = ["trivial", "simple", "moderate", "complex", "expert"];
  chart("#chTier", { type: "bar", data: { labels: s.by_tier.map((r) => `${r.tier} ${tiers[r.tier - 1]}`), datasets: [
    { label: "EcoRoute", data: s.by_tier.map((r) => r.energy_wh), backgroundColor: acc },
    { label: "Baseline", data: s.by_tier.map((r) => r.baseline_energy_wh), backgroundColor: warn + "99" },
  ] }, options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } } });

  chart("#chModels", { type: "bar", data: { labels: s.by_model.map((r) => r.model_id), datasets: [
    { label: "requests", data: s.by_model.map((r) => r.n), backgroundColor: acc },
  ] }, options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } } });

  const b = s.budget;
  $("#budUsd").value = b.monthly_budget_usd || ""; $("#budWh").value = b.monthly_energy_budget_wh || "";
  const setBar = (bar, txt, ratio, used, cap, f) => {
    $(bar).style.width = `${Math.min(100, ratio * 100)}%`; $(bar).classList.toggle("hot", ratio >= 0.8);
    $(txt).textContent = cap ? `${f(used)} / ${f(cap)}` : `${f(used)} · no cap`;
  };
  setBar("#budUsdBar", "#budUsdTxt", b.cost_ratio, b.spent_usd, b.monthly_budget_usd, fmt.usd);
  setBar("#budWhBar", "#budWhTxt", b.energy_ratio, b.used_wh, b.monthly_energy_budget_wh, fmt.wh);

  $("#recent").innerHTML = recent.length ? `<thead><tr><th>When</th><th>Prompt</th><th>Tier</th><th>Policy</th><th>Model</th><th>Source</th><th>Energy</th><th>Saved</th><th>Cost</th></tr></thead><tbody>` +
    recent.map((r) => {
      const sv = r.baseline_energy_wh ? (100 * (r.baseline_energy_wh - r.energy_wh)) / r.baseline_energy_wh : 0;
      return `<tr><td>${new Date(r.ts * 1000).toLocaleTimeString()}</td><td title="${esc(r.prompt_preview)}">${esc((r.prompt_preview || "—").slice(0, 48))}</td>
      <td class="num">${r.tier}</td><td>${esc(r.policy)}</td><td>${esc(r.model_id)}</td><td>${esc(r.source)}</td>
      <td class="num">${fmt.wh(r.energy_wh)}</td><td class="num">${sv.toFixed(0)}%</td><td class="num">${fmt.usd(r.cost_usd)}</td></tr>`;
    }).join("") + "</tbody>" : `<tr><td class="empty">Nothing logged yet.</td></tr>`;
}

$("#btnSeed").addEventListener("click", async (e) => {
  e.target.disabled = true;
  try { await api("/v1/demo/seed?n=60", { method: "POST" }); await loadImpact(); }
  catch (err) { alert(err.message); } finally { e.target.disabled = false; }
});
$("#btnReset").addEventListener("click", async () => {
  if (!confirm("Delete all logged requests and cached responses?")) return;
  await api("/v1/metrics", { method: "DELETE" }); loadImpact();
});
$("#btnBudget").addEventListener("click", async () => {
  await api("/v1/budget", { method: "PUT", body: JSON.stringify({
    monthly_budget_usd: Number($("#budUsd").value || 0), monthly_energy_budget_wh: Number($("#budWh").value || 0) }) });
  loadImpact();
});

/* ---------------- models ---------------- */
async function loadModels() {
  const m = await api("/v1/models");
  $("#snapshot").textContent = `Mode: ${m.mode}. Baseline model: ${m.baseline_model}. Prices: ${m.pricing_snapshot}.`;
  $("#modelTbl").innerHTML = `<thead><tr><th>Label</th><th>Model</th><th>Where</th><th>Wh / 1k out</th><th>Estimate from</th><th>$ in / out per M</th><th>Typical energy</th><th>CO₂e</th><th>Water</th><th>Sust.</th><th>Max tier</th><th>Available</th></tr></thead><tbody>` +
    m.models.map((x) => `<tr class="${x.available ? "" : "dim"}">
      <td>${labelChip(x.energy.label)}</td><td>${esc(x.display_name)}</td>
      <td>${x.kind === "local" ? "your device" : esc(x.provider)} · ${esc(x.grid_region)}${x.grid_live ? " (live)" : ""}</td>
      <td class="num">${x.energy.wh_per_1k_output.toFixed(3)}</td><td>${esc(x.energy.method)} · ${esc(x.energy.confidence)}</td>
      <td class="num">${x.price_in.toFixed(2)} / ${x.price_out.toFixed(2)}</td>
      <td class="num">${fmt.wh(x.typical_request.energy_wh)}</td><td class="num">${fmt.g(x.typical_request.carbon_g)}</td>
      <td class="num">${fmt.ml(x.typical_request.water_ml)}</td><td class="num">${x.sustainability_score.toFixed(0)}</td>
      <td class="num">${x.max_tier}</td><td>${x.available ? "yes" : "—"}</td></tr>`).join("") + "</tbody>";

  const hosted = m.models.filter((x) => x.price_out > 0);
  const pts = (pred) => hosted.filter(pred).map((x) => ({ x: x.price_out, y: x.energy.wh_per_1k_output, name: x.display_name }));
  const acc = cssv("--accent");
  chart("#chPrice", { type: "scatter", data: { datasets: [
    { label: "compute estimate (open weights)", data: pts((x) => x.energy.method !== "price"), backgroundColor: acc, pointRadius: 6 },
    { label: "price estimate (closed)", data: pts((x) => x.energy.method === "price"), borderColor: acc, backgroundColor: "transparent", pointRadius: 6, borderWidth: 2 },
  ] }, options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" },
    tooltip: { callbacks: { label: (c) => `${c.raw.name}: $${c.raw.x}/M → ${c.raw.y.toFixed(3)} Wh/1k` } } },
    scales: { x: { type: "logarithmic", min: 0.05, max: 20, title: { display: true, text: "output price, $ per M tokens" } },
              y: { type: "logarithmic", min: 0.01, max: 1, title: { display: true, text: "Wh per 1k output tokens" } } } } });
}

loadStatus();
