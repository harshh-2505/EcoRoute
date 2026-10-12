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

// When the server has API keys configured, the dashboard asks for one once and keeps it in this browser.
const KEY_STORE = "ecoroute.key";
const storedKey = () => { try { return localStorage.getItem(KEY_STORE) || ""; } catch { return ""; } };
async function api(path, opts = {}, retried = false) {
  const headers = { "Content-Type": "application/json" };
  if (storedKey()) headers.Authorization = `Bearer ${storedKey()}`;
  const r = await fetch(path, { ...opts, headers });
  const body = await r.json().catch(() => ({}));
  if (r.status === 401 && !retried) {
    const k = prompt("This EcoRoute server needs an API key:");
    if (k) { try { localStorage.setItem(KEY_STORE, k.trim()); } catch { /* private mode */ } return api(path, opts, true); }
  }
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : r.statusText);
  return body;
}

async function apiBlob(path) {
  const headers = storedKey() ? { Authorization: `Bearer ${storedKey()}` } : {};
  const r = await fetch(path, { headers });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.blob();
}

/* ---------------- tabs & segmented controls ---------------- */
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === t.dataset.tab));
  if (t.dataset.tab === "impact") loadImpact();
  if (t.dataset.tab === "models") loadModels();
  if (t.dataset.tab === "quality") loadQuality();
  if (t.dataset.tab === "embed") loadEmbed();
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
const qRange = seg("#qRange", () => loadQuality());

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
  if ($("#concise").checked) c.concise = true;
  if ($("#cascade").checked) c.cascade = true;
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
  renderMeter(cx.score);
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
    $("#rText").textContent = out.response || (out.tool_calls ? JSON.stringify(out.tool_calls, null, 2) : "");
    renderNotes(out);
    lastOut = out;
    EcoReceipt.render("#rReceipt", out, { theme: "dark" });
    $$("#rFeedback .btn").forEach((b) => { b.disabled = false; b.classList.remove("picked"); });
    $("#rFbDone").textContent = "";
    $("#rFeedback").classList.toggle("hidden", !out.request_id);
  }

  lastDecision = { d, modelId };
  renderCands();
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

let lastOut = null;
function renderNotes(out) {
  const n = [];
  if (out.escalation) n.push(`<div class="note"><b>Escalated</b> from ${esc(out.escalation.from)}: ${esc(out.escalation.issues.join(", "))}. The first attempt's ${fmt.wh(out.escalation.extra_energy_wh)} is still counted.</div>`);
  else if (out.quality_check && out.quality_check.ok) n.push(`<div class="note ok"><b>Answer check passed.</b> No sign the model struggled.</div>`);
  else if (out.quality_check) n.push(`<div class="note"><b>Answer check:</b> ${esc(out.quality_check.issues.join(", "))}. No stronger model was available.</div>`);
  if (out.concise) n.push(`<div class="note ok"><b>Concise mode.</b> Asked for a brief answer and capped output tokens.</div>`);
  $("#rNotes").innerHTML = n.join("");
}

$("#rFeedback").addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b || !lastOut) return;
  const escalate = b.dataset.r === "escalate";
  $$("#rFeedback .btn").forEach((x) => (x.disabled = true));
  b.classList.add("picked");
  try {
    const r = await api("/v1/feedback", { method: "POST", body: JSON.stringify({
      request_id: lastOut.request_id, rating: b.dataset.r === "up" ? "up" : "down", escalate }) });
    if (r.escalated) {
      const prev = lastOut.model_id;
      renderDecision(r.escalated.decision, r.escalated);
      $("#rNotes").insertAdjacentHTML("afterbegin", `<div class="note"><b>Re-asked</b> after your ▼ on ${esc(prev)}. Future routing will weigh that vote.</div>`);
    } else {
      $("#rFbDone").textContent = b.dataset.r === "up" ? "Thanks. Logged." : "Logged. Down-votes steer future routing away from this model.";
    }
  } catch (err) { $("#rFbDone").textContent = err.message; $$("#rFeedback .btn").forEach((x) => (x.disabled = false)); }
});

// With dozens of models the full ranking is long: show the top rows, expandable to all.
const CAND_ROWS = 10;
let lastDecision = null, showAllCands = false;
const bar = (v) => `<span class="mono">${v.toFixed(0)}</span><span class="scorebar"><i style="width:${v}%"></i></span>`;
function renderCands() {
  const { d, modelId } = lastDecision;
  const all = d.candidates, n = all.length, elig = all.filter((c) => c.eligible).length;
  const rows = showAllCands ? all : all.slice(0, CAND_ROWS);
  $("#candCount").textContent = `${elig} of ${n} models eligible`;
  $("#candToggle").textContent = showAllCands ? `Top ${CAND_ROWS} only` : `Show all ${n}`;
  $("#candToggle").classList.toggle("hidden", n <= CAND_ROWS);
  $("#rCands").innerHTML = `<thead><tr><th>Model</th><th>Score</th><th>Sustainability</th><th>Cost</th><th>Quality</th><th>Est. energy</th><th>Est. cost</th><th>Status</th></tr></thead><tbody>` +
    rows.map((c) => `<tr class="${c.model_id === modelId ? "win" : c.eligible ? "" : "dim"}">
      <td>${esc(c.display_name)}</td><td class="num">${c.eligible ? c.total.toFixed(1) : "—"}</td>
      <td>${bar(c.scores.sustainability)}</td><td>${bar(c.scores.cost)}</td><td>${bar(c.scores.quality)}</td>
      <td class="num">${fmt.wh(c.estimated.energy_wh)}</td><td class="num">${fmt.usd(c.estimated.cost_usd)}</td>
      <td>${c.model_id === modelId ? "✔ chosen" : c.eligible ? "eligible" : esc(c.reason)}</td></tr>`).join("") +
    (rows.length < n ? `<tr class="more"><td colspan="8">+ ${n - rows.length} more models ranked below</td></tr>` : "") + "</tbody>";
}
$("#candToggle").addEventListener("click", () => { showAllCands = !showAllCands; renderCands(); });

// 50-cell LED meter, coloured by the tier band each cell falls in (cut points match the router's tiers).
function renderMeter(score) {
  const n = 50, lit = Math.max(1, Math.round(score * n));
  const band = (f) => (f < 0.2 ? "--accent-2" : f < 0.48 ? "--accent" : f < 0.74 ? "--warn" : "--danger");
  $("#rMeter").innerHTML = Array.from({ length: n }, (_, i) => {
    const on = i < lit;
    return `<i class="${on ? "on" : ""}${i === lit - 1 ? " edge" : ""}" style="--c:var(${band(i / n)});transition-delay:${on ? i * 12 : 0}ms"></i>`;
  }).join("");
}

/* ---------------- impact dashboard ---------------- */
const charts = {};
function chart(id, cfg) {
  charts[id]?.destroy();
  const ink = cssv("--muted"), line = cssv("--line");
  Chart.defaults.color = ink; Chart.defaults.borderColor = line;
  Chart.defaults.font.family = "Geist Mono"; Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 10; Chart.defaults.plugins.legend.labels.boxHeight = 10;
  Object.assign(Chart.defaults.plugins.tooltip, { backgroundColor: cssv("--raise"), borderColor: cssv("--line-2"), borderWidth: 1,
    titleColor: cssv("--ink"), bodyColor: cssv("--ink"), cornerRadius: 2, padding: 10 });
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
    { label: "EcoRoute", data: ts.map((r) => r.energy_wh), borderColor: acc, borderWidth: 2, fill: true, tension: .35, pointRadius: 0, pointHoverRadius: 4,
      backgroundColor: (c) => { const a = c.chart.chartArea; if (!a) return acc + "22";
        const g = c.chart.ctx.createLinearGradient(0, a.top, 0, a.bottom); g.addColorStop(0, acc + "55"); g.addColorStop(1, acc + "00"); return g; } },
    { label: "Baseline", data: ts.map((r) => r.baseline_energy_wh), borderColor: warn, borderDash: [5, 4], tension: .3, pointRadius: 0 },
  ] }, options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } } } });

  const tiers = ["trivial", "simple", "moderate", "complex", "expert"];
  chart("#chTier", { type: "bar", data: { labels: s.by_tier.map((r) => `${r.tier} ${tiers[r.tier - 1]}`), datasets: [
    { label: "EcoRoute", data: s.by_tier.map((r) => r.energy_wh), backgroundColor: acc },
    { label: "Baseline", data: s.by_tier.map((r) => r.baseline_energy_wh), backgroundColor: warn + "33", borderColor: warn, borderWidth: 1 },
  ] }, options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } } });

  const TOP = 12, byModel = [...s.by_model].sort((a, b) => b.n - a.n);
  const shown = byModel.slice(0, TOP), rest = byModel.slice(TOP).reduce((a, r) => a + r.n, 0);
  if (rest) shown.push({ model_id: `${byModel.length - TOP} other models`, n: rest });
  chart("#chModels", { type: "bar", data: { labels: shown.map((r) => r.model_id), datasets: [
    { label: "requests", data: shown.map((r) => r.n), backgroundColor: acc + "cc", borderRadius: 1, barPercentage: .7 },
  ] }, options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } } });

  const b = s.budget;
  $("#budUsd").value = b.monthly_budget_usd || ""; $("#budWh").value = b.monthly_energy_budget_wh || "";
  const setBar = (bar, txt, ratio, used, cap, f) => {
    $(bar).style.width = `${Math.min(100, ratio * 100)}%`; $(bar).classList.toggle("hot", ratio >= 0.8);
    $(txt).textContent = cap ? `${f(used)} / ${f(cap)}` : `${f(used)} · no cap`;
  };
  setBar("#budUsdBar", "#budUsdTxt", b.cost_ratio, b.spent_usd, b.monthly_budget_usd, fmt.usd);
  setBar("#budWhBar", "#budWhTxt", b.energy_ratio, b.used_wh, b.monthly_energy_budget_wh, fmt.wh);

  const teams = s.by_team || [];
  $("#teamCard").classList.toggle("hidden", !teams.length || (teams.length === 1 && teams[0].team === "unassigned"));
  $("#teams").innerHTML = `<thead><tr><th>Team</th><th>Requests</th><th>Energy</th><th>Saved vs baseline</th><th>CO₂e</th><th>Spend</th></tr></thead><tbody>` +
    teams.map((r) => {
      const sv = r.baseline_energy_wh ? (100 * (r.baseline_energy_wh - r.energy_wh)) / r.baseline_energy_wh : 0;
      return `<tr><td>${esc(r.team)}</td><td class="num">${fmt.n(r.n)}</td><td class="num">${fmt.wh(r.energy_wh || 0)}</td>
        <td class="num">${sv.toFixed(0)}%</td><td class="num">${fmt.g(r.carbon_g || 0)}</td><td class="num">${fmt.usd(r.cost_usd || 0)}</td></tr>`;
    }).join("") + "</tbody>";

  $("#recent").innerHTML = recent.length ? `<thead><tr><th>When</th><th>Prompt</th><th>Tier</th><th>Policy</th><th>Model</th><th>Source</th><th>Energy</th><th>Saved</th><th>Cost</th></tr></thead><tbody>` +
    recent.map((r) => {
      const sv = r.baseline_energy_wh ? (100 * (r.baseline_energy_wh - r.energy_wh)) / r.baseline_energy_wh : 0;
      return `<tr><td>${new Date(r.ts * 1000).toLocaleTimeString()}</td><td title="${esc(r.prompt_preview)}">${esc((r.prompt_preview || "—").slice(0, 48))}</td>
      <td class="num">${r.tier}</td><td>${esc(r.policy)}</td><td>${esc(r.model_id)}</td><td>${esc(r.source)}</td>
      <td class="num">${fmt.wh(r.energy_wh)}</td><td class="num">${sv.toFixed(0)}%</td><td class="num">${fmt.usd(r.cost_usd)}</td></tr>`;
    }).join("") + "</tbody>" : `<tr><td class="empty">Nothing logged yet.</td></tr>`;
}

async function openReport(format) {
  try {
    const blob = await apiBlob(`/v1/report?format=${format}`);
    const url = URL.createObjectURL(blob);
    if (format === "csv") {
      const a = Object.assign(document.createElement("a"), { href: url, download: `ecoroute-report-${new Date().toISOString().slice(0, 7)}.csv` });
      document.body.appendChild(a); a.click(); a.remove();
    } else window.open(url, "_blank", "noopener");
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (err) { alert(err.message); }
}
$("#btnReport").addEventListener("click", () => openReport("html"));
$("#btnCsv").addEventListener("click", () => openReport("csv"));

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

/* ---------------- quality ---------------- */
async function loadQuality() {
  const days = Number(qRange());
  const q = await api(`/v1/quality?days=${days}`);
  const ev = q.evals, fb = q.feedback;
  const k = (label, v, sub) => `<div class="kpi"><div class="k">${label}</div><div class="v">${v}</div><div class="s">${sub || "&nbsp;"}</div></div>`;
  const pctOr = (x, alt) => (x == null ? alt : `${(x * 100).toFixed(0)}%`);
  $("#qKpis").innerHTML = [
    k("At parity with baseline", pctOr(ev.parity_rate, "—"), ev.n ? `${fmt.n(ev.n)} answers graded blind` : "no graded answers yet"),
    k("Avg judge score", ev.n ? `${ev.avg_routed.toFixed(1)} / ${ev.avg_baseline.toFixed(1)}` : "—", "routed / baseline, out of 10"),
    k("User satisfaction", pctOr(fb.satisfaction, "—"), `${fmt.n(fb.up)} ▲ · ${fmt.n(fb.down)} ▼`),
    k("Escalations", fmt.n(fb.escalated), fb.checks_run ? `${fmt.n(fb.checks_failed)} of ${fmt.n(fb.checks_run)} checks failed` : "cascade checks off"),
  ].join("");
  $("#qConfig").innerHTML = `shadow rate <b>${(q.shadow_rate * 100).toFixed(0)}%</b> · judge <b>${esc(q.judge_model)}</b> · cascade <b>${q.cascade ? "on" : "per request"}</b>` +
    (ev.n ? ` · eval overhead ${fmt.wh(ev.overhead_energy_wh)}, ${fmt.usd(ev.overhead_cost_usd)}` : "");
  const tiers = ["trivial", "simple", "moderate", "complex", "expert"];
  $("#qTiers").innerHTML = q.evals_by_tier.length ? q.evals_by_tier.map((t) => {
    const p = t.n ? (100 * t.at_parity) / t.n : 0;
    return `<div class="row"><span>${t.tier} ${tiers[t.tier - 1]}</span><span class="track"><i style="width:${p}%"></i><i class="lose" style="width:${100 - p}%"></i></span><b>${p.toFixed(0)}% of ${t.n}</b></div>`;
  }).join("") : `<span class="empty">No shadow evaluations yet. They run on live traffic only, since simulated answers aren't real model output.</span>`;
  const pen = Object.entries(q.feedback_penalties || {});
  $("#qPenalties").innerHTML = pen.length ? `Feedback is currently lowering: ${pen.map(([m, v]) => `<b>${esc(m)}</b> −${v.toFixed(1)}`).join(" · ")}` : "";
  $("#qEvals").innerHTML = q.recent_evals.length ? `<thead><tr><th>When</th><th>Tier</th><th>Routed model</th><th>Score</th><th>Baseline</th><th>Verdict</th><th>Judge's reason</th></tr></thead><tbody>` +
    q.recent_evals.map((r) => `<tr><td>${new Date(r.ts * 1000).toLocaleString()}</td><td class="num">${r.tier}</td><td>${esc(r.model_id)}</td>
      <td class="num">${r.score_routed}</td><td class="num">${r.score_baseline}</td><td><span class="verdict ${esc(r.verdict)}">${esc(r.verdict.replace("_", " "))}</span></td>
      <td title="${esc(r.reason)}">${esc((r.reason || "").slice(0, 70))}</td></tr>`).join("") + "</tbody>"
    : `<tr><td class="empty">Nothing graded yet.</td></tr>`;
}

/* ---------------- embed ---------------- */
const SAMPLE = { model: "groq-llama-3.1-8b", ecoroute: { source: "live", baseline_model: "openai-gpt-5",
  impact: { energy_wh: 0.0124, carbon_g: 0.0046, water_ml: 0.041, cost_usd: 0.00002 }, savings: { energy_pct: 91 } } };
function loadEmbed() {
  const origin = location.origin;
  $("#embedCode").innerHTML = [
    `<span class="c">&lt;!-- 1. load the script --&gt;</span>`,
    `&lt;script src=<span class="s">"${esc(origin)}/static/receipt.js"</span>&gt;&lt;/script&gt;`,
    ``, `<span class="c">&lt;!-- 2. a place for the receipt --&gt;</span>`,
    `&lt;div id=<span class="s">"receipt"</span>&gt;&lt;/div&gt;`,
    ``, `<span class="c">&lt;!-- 3. render it from any EcoRoute response --&gt;</span>`,
    `&lt;script&gt;`,
    `  <span class="k">const</span> r = <span class="k">await</span> client.chat.completions.create({ model: <span class="s">"ecoroute/eco"</span>, messages });`,
    `  EcoReceipt.render(<span class="s">"#receipt"</span>, r);`,
    `&lt;/script&gt;`,
  ].join("\n");
  const src = lastOut || SAMPLE;
  EcoReceipt.render("#embedReceipt", src, { theme: "dark" });
  EcoReceipt.render("#embedReceiptCompact", src, { theme: "dark", compact: true });
}

/* ---------------- models ---------------- */
const PROVIDER_NAMES = { ollama: "local", openai: "OpenAI", anthropic: "Anthropic", gemini: "Gemini", groq: "Groq",
  mistral: "Mistral", deepseek: "DeepSeek", xai: "xAI" };
const provName = (p) => PROVIDER_NAMES[p] || p;
const MODEL_COLS = [
  ["Label", "label"], ["Model", "display_name"], ["Where", "provider"], ["Wh / 1k out", "wh", true], ["Estimate from", null],
  ["$ in / out per M", "price_out", true], ["Typical energy", "wh", true], ["CO₂e", "carbon", true], ["Water", "water", true],
  ["Sust.", "sustainability_score", true], ["Quality", "quality", true], ["Max tier", "max_tier", true], ["Available", "available"],
];
const sortVal = {
  label: (x) => x.energy.label, wh: (x) => x.energy.wh_per_1k_output, carbon: (x) => x.typical_request.carbon_g,
  water: (x) => x.typical_request.water_ml, provider: (x) => provName(x.provider),
};
const mState = { data: null, prov: "all", q: "", sort: "sustainability_score", desc: true };

async function loadModels() {
  const m = await api("/v1/models");
  mState.data = m;
  $("#snapshot").textContent = `Mode: ${m.mode}. Baseline model: ${m.baseline_model}. Prices: ${m.pricing_snapshot}.`;
  $("#sources").innerHTML = `${m.models.length} models from ${Object.keys(m.sources || {}).length} providers. Prices and model ids from: ` +
    Object.entries(m.sources || {}).map(([p, u]) => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(provName(p))}</a>`).join(" · ");
  const counts = {};
  m.models.forEach((x) => (counts[x.provider] = (counts[x.provider] || 0) + 1));
  $("#mProv").innerHTML = [["all", m.models.length], ...Object.entries(counts)].map(([p, n]) =>
    `<button data-p="${esc(p)}" class="${p === mState.prov ? "on" : ""}">${p === "all" ? "All" : esc(provName(p))}<i>${n}</i></button>`).join("");
  renderModelTable();

  const hosted = m.models.filter((x) => x.price_out > 0);
  const pts = (pred) => hosted.filter(pred).map((x) => ({ x: x.price_out, y: x.energy.wh_per_1k_output, name: x.display_name }));
  const acc = cssv("--accent");
  chart("#chPrice", { type: "scatter", data: { datasets: [
    { label: "compute estimate (open weights)", data: pts((x) => x.energy.method !== "price"), backgroundColor: acc, pointRadius: 5 },
    { label: "price estimate (closed)", data: pts((x) => x.energy.method === "price"), borderColor: acc, backgroundColor: "transparent", pointRadius: 5, borderWidth: 1.5 },
  ] }, options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" },
    tooltip: { callbacks: { label: (c) => `${c.raw.name}: $${c.raw.x}/M → ${c.raw.y.toFixed(3)} Wh/1k` } } },
    scales: { x: { type: "logarithmic", min: 0.05, max: 250, ticks: { callback: decadeTick }, title: { display: true, text: "output price, $ per M tokens" } },
              y: { type: "logarithmic", min: 0.01, max: 4, ticks: { callback: decadeTick }, title: { display: true, text: "Wh per 1k output tokens" } } } } });
}

const price = (v) => v.toFixed(Math.round(v * 100) === v * 100 ? 2 : 3);
const decadeTick = (v) => ([0.01, 0.1, 1, 10, 100].includes(v) ? v : "");

function renderModelTable() {
  const m = mState.data; if (!m) return;
  const q = mState.q.toLowerCase();
  const get = sortVal[mState.sort] || ((x) => x[mState.sort]);
  const rows = m.models
    .filter((x) => mState.prov === "all" || x.provider === mState.prov)
    .filter((x) => !q || `${x.display_name} ${x.model} ${x.provider} ${provName(x.provider)}`.toLowerCase().includes(q))
    .sort((a, b) => { const va = get(a), vb = get(b); const c = va < vb ? -1 : va > vb ? 1 : 0; return mState.desc ? -c : c; });
  $("#mCount").textContent = `${rows.length} of ${m.models.length} shown`;
  $("#modelTbl").innerHTML = `<thead><tr>${MODEL_COLS.map(([h, k]) => k
    ? `<th data-k="${k}" class="${k === mState.sort ? `sorted${mState.desc ? "" : " asc"}` : ""}">${h}</th>` : `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    (rows.length ? rows.map((x) => `<tr class="${x.available ? "" : "dim"}">
      <td>${labelChip(x.energy.label)}</td><td>${esc(x.display_name)}${x.status && x.status !== "active" ? `<span class="tag ${esc(x.status)}">${esc(x.status)}</span>` : ""}</td>
      <td>${x.kind === "local" ? "your device" : esc(provName(x.provider))} · ${esc(x.grid_region)}${x.grid_live ? " (live)" : ""}</td>
      <td class="num">${x.energy.wh_per_1k_output.toFixed(3)}</td><td>${esc(x.energy.method)} · ${esc(x.energy.confidence)}</td>
      <td class="num">${price(x.price_in)} / ${price(x.price_out)}</td>
      <td class="num">${fmt.wh(x.typical_request.energy_wh)}</td><td class="num">${fmt.g(x.typical_request.carbon_g)}</td>
      <td class="num">${fmt.ml(x.typical_request.water_ml)}</td><td class="num">${x.sustainability_score.toFixed(0)}</td>
      <td class="num">${x.quality}</td><td class="num">${x.max_tier}</td><td>${x.available ? "yes" : "—"}</td></tr>`).join("")
      : `<tr><td class="empty" colspan="${MODEL_COLS.length}">No models match.</td></tr>`) + "</tbody>";
}
$("#mSearch").addEventListener("input", (e) => { mState.q = e.target.value.trim(); renderModelTable(); });
$("#mProv").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  mState.prov = b.dataset.p;
  $$("#mProv button").forEach((x) => x.classList.toggle("on", x === b));
  renderModelTable();
});
$("#modelTbl").addEventListener("click", (e) => {
  const th = e.target.closest("th[data-k]"); if (!th) return;
  const k = th.dataset.k;
  if (mState.sort === k) mState.desc = !mState.desc;
  else { mState.sort = k; mState.desc = !["display_name", "provider", "label", "wh", "carbon", "water", "price_out"].includes(k); }
  renderModelTable();
});

loadStatus();
