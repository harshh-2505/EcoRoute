/*! EcoRoute eco receipt: a small, embeddable badge that shows which model answered and what it saved.
 *
 *   <script src="https://YOUR-ECOROUTE/static/receipt.js"></script>
 *   <div id="r"></div>
 *   <script>EcoReceipt.render("#r", completion);</script>
 *
 * `completion` can be a /v1/chat/completions response (it reads `model` and `ecoroute`), a /v1/chat
 * response, or the `ecoroute` block on its own. Options: { theme: "auto" | "dark" | "light", compact: bool,
 * label: "Answered sustainably" }. Styles live in a shadow root, so the host page's CSS is untouched.
 */
(function (global) {
  "use strict";

  var CSS = [
    ":host{all:initial;display:inline-block;max-width:100%;--bg:#0a0e0c;--ink:#e7efe9;--muted:#8a998f;--line:#22302a;--acc:#c4f24a;--acc2:#5ee0ff;--glow:rgba(196,242,74,.28)}",
    ":host([data-theme=light]){--bg:#f7faf5;--ink:#0d1410;--muted:#5d6b63;--line:#d6e0d8;--acc:#3f7d12;--acc2:#0b7f99;--glow:rgba(63,125,18,.18)}",
    "@media (prefers-color-scheme:light){:host([data-theme=auto]){--bg:#f7faf5;--ink:#0d1410;--muted:#5d6b63;--line:#d6e0d8;--acc:#3f7d12;--acc2:#0b7f99;--glow:rgba(63,125,18,.18)}}",
    ".r{box-sizing:border-box;font:12px/1.4 ui-monospace,'Geist Mono',SFMono-Regular,Menlo,monospace;color:var(--ink);background:var(--bg);",
    "border:1px solid var(--line);border-radius:6px;padding:10px 12px;display:grid;gap:8px;min-width:0;max-width:100%;position:relative;overflow:hidden}",
    ".r:before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--acc);box-shadow:0 0 12px var(--glow)}",
    ".h{display:flex;align-items:center;gap:8px;min-width:0}",
    ".dot{width:7px;height:7px;border-radius:50%;background:var(--acc);box-shadow:0 0 8px var(--glow);flex:none}",
    ".t{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);white-space:nowrap}",
    ".m{font-weight:600;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}",
    ".g{display:flex;flex-wrap:wrap;gap:6px 14px}",
    ".s{display:flex;flex-direction:column;min-width:64px}",
    ".s b{font-weight:600;font-size:13px;font-variant-numeric:tabular-nums}",
    ".s i{font-style:normal;color:var(--muted);font-size:10.5px}",
    ".s .p{color:var(--acc)}.s .w{color:var(--acc2)}",
    ".f{color:var(--muted);font-size:10.5px}",
    ".c .g,.c .f{display:none}.c{padding:6px 10px}"
  ].join("");

  function wh(x) { return x < 1 ? (x * 1000).toFixed(x < 0.01 ? 2 : 1) + " mWh" : x.toFixed(2) + " Wh"; }
  function g(x) { return x < 1 ? (x * 1000).toFixed(1) + " mg" : x.toFixed(2) + " g"; }
  function ml(x) { return x < 1000 ? x.toFixed(x < 10 ? 2 : 1) + " mL" : (x / 1000).toFixed(2) + " L"; }
  function pct(x) { return (x >= 0 ? "−" : "+") + Math.abs(x).toFixed(0) + "%"; }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }

  /** Pull model, impact and savings out of any EcoRoute response shape. */
  function normalise(src) {
    if (!src) return null;
    var eco = src.ecoroute || src;
    var impact = eco.impact || src.impact;
    if (!impact) return null;
    var savings = eco.savings || src.savings || {};
    return {
      model: src.model || src.model_id || eco.model_id || "",
      baseline: eco.baseline_model || (src.decision && src.decision.baseline_model) || "the flagship model",
      source: eco.source || src.source || "",
      impact: impact,
      savings: savings,
    };
  }

  function render(target, src, opts) {
    opts = opts || {};
    var el = typeof target === "string" ? document.querySelector(target) : target;
    if (!el) return null;
    var d = normalise(src);
    var root = el.shadowRoot || el.attachShadow({ mode: "open" });
    el.setAttribute("data-theme", opts.theme || "auto");
    if (!d) { root.innerHTML = ""; return null; }
    var i = d.impact, s = d.savings;
    var cached = d.source === "cache";
    root.innerHTML = "<style>" + CSS + "</style>" +
      '<div class="r' + (opts.compact ? " c" : "") + '" role="note" aria-label="Energy receipt for this answer">' +
      '<div class="h"><span class="dot"></span><span class="t">' + esc(opts.label || (cached ? "Served from cache" : "Answered sustainably")) +
      '</span><span class="m" title="' + esc(d.model) + '">' + esc(d.model) + "</span>" +
      (opts.compact && s.energy_pct != null ? '<span class="t">' + pct(s.energy_pct) + " energy</span>" : "") + "</div>" +
      '<div class="g">' +
      '<span class="s"><b>' + wh(i.energy_wh) + '</b><i>energy</i></span>' +
      '<span class="s"><b>' + g(i.carbon_g) + '</b><i>CO₂e</i></span>' +
      '<span class="s"><b class="w">' + ml(i.water_ml) + '</b><i>water</i></span>' +
      (s.energy_pct != null ? '<span class="s"><b class="p">' + pct(s.energy_pct) + "</b><i>vs " + esc(d.baseline) + "</i></span>" : "") +
      "</div>" +
      '<div class="f">Estimated by EcoRoute · routed to the smallest model that could handle this prompt</div></div>';
    return d;
  }

  /** Render a receipt into every element with data-ecoroute='{"model":…,"ecoroute":{…}}'. */
  function scan(scope) {
    var nodes = (scope || document).querySelectorAll("[data-ecoroute]");
    for (var k = 0; k < nodes.length; k++) {
      try { render(nodes[k], JSON.parse(nodes[k].getAttribute("data-ecoroute")), { compact: nodes[k].hasAttribute("data-compact") }); }
      catch (e) { /* ignore malformed blocks */ }
    }
  }

  global.EcoReceipt = { render: render, scan: scan, normalise: normalise };
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () { scan(); });
    else scan();
  }
})(typeof window !== "undefined" ? window : this);
