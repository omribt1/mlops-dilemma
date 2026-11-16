/* mlops-dilemma web UI (`dilemma ui`). Design: docs/ui-design-plan.md.
 * Inputs live in the rail; every output is recomputed client-side with window.DilemmaModel (model.js).
 * The server is only called to load/save sessions, re-plan hardware, upload files and run benchmarks. */
(function () {
  "use strict";

  const M = window.DilemmaModel;
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const enc = encodeURIComponent;

  // ---------------------------------------------------------------- formatting
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const trim0 = (s) => s.replace(/\.0(?=[KM]?$)/, "");
  function fmtUSD(x) {
    if (x == null || !isFinite(x)) return "—";
    const a = Math.abs(x);
    if (a < 1000) return "$" + Math.round(x).toLocaleString("en-US");
    if (a < 10000) return "$" + trim0((x / 1e3).toFixed(1)) + "K";
    if (a < 1e6) return "$" + Math.round(x / 1e3).toLocaleString("en-US") + "K";
    return "$" + trim0((x / 1e6).toFixed(a < 1e7 ? 2 : 1)) + "M";
  }
  const fmtUSDc = (x) => (x == null || !isFinite(x)) ? "—" : "$" + (x < 10 ? x.toFixed(2) : Math.round(x).toLocaleString("en-US"));
  function fmtTok(x) {
    if (x == null || !isFinite(x)) return "—";
    if (x < 1000) return String(Math.round(x));
    if (x < 1e6) return trim0((x / 1e3).toFixed(x < 10000 ? 1 : 0)) + "K";
    if (x < 1e9) return trim0((x / 1e6).toFixed(x < 1e7 ? 2 : x < 1e8 ? 1 : 0)) + "M";
    return trim0((x / 1e9).toFixed(x < 1e10 ? 2 : 1)) + "B";
  }
  const fmtInt = (x) => (x == null || !isFinite(x)) ? "—" : Math.round(x).toLocaleString("en-US");
  const pct = (x) => (x == null || !isFinite(x)) ? "—" : Math.round(x * 100) + "%";
  const pct1 = (x) => (x == null || !isFinite(x)) ? "—" : (x * 100 < 10 ? (x * 100).toFixed(1).replace(/\.0$/, "") : Math.round(x * 100)) + "%";
  const perM = (x) => (x == null || !isFinite(x)) ? "—" : "$" + (x < 1 ? x.toFixed(3) : x.toFixed(2));
  const clone = (o) => (o == null ? o : JSON.parse(JSON.stringify(o)));
  function fmtElapsed(sec) {
    const s = Math.max(0, Math.floor(sec)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    return h ? `${h}h ${m}m` : m ? `${m}m ${r}s` : `${r}s`;
  }
  function niceMax(v) {
    const p = Math.pow(10, Math.floor(Math.log10(v || 1)));
    for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  // ---------------------------------------------------------------- constants
  const TIER_ORDER = ["vertex", "vertex_cud_1y", "vertex_cud_3y", "cud_1y", "cud_3y"];
  const TIER_LABELS = {
    vertex: "Agent Platform, on demand",
    vertex_cud_1y: "Agent Platform + reservation, 1-yr",
    vertex_cud_3y: "Agent Platform + reservation, 3-yr",
    cud_1y: "Self-managed GKE, 1-yr",
    cud_3y: "Self-managed GKE, 3-yr",
  };
  const GEMINI_KEYS = ["gemini_flash_lite", "gemini_flash", "gemini_pro", "gemini_4"];
  const GEMINI_FALLBACK_NAMES = { gemini_flash_lite: "Gemini Flash-Lite", gemini_flash: "Gemini Flash", gemini_pro: "Gemini Pro", gemini_4: "Gemini 4" };
  // Display names for overflow / routing / failover targets; Gemini names come from the catalog ("Gemini 3.5 Flash-Lite").
  function targetName(t) {
    if (!t || t === "open_maas") return "open-model MaaS";
    const g = geminiCatalog()[t];
    if (g && g.name) return g.name;
    // A pinned model the catalog no longer lists is priced as its role alias (model.js aliasForId).
    if (isGeminiTarget(t) && M.aliasForId) { const a = geminiCatalog()[M.aliasForId(t)]; if (a && a.name) return a.name; }
    return GEMINI_FALLBACK_NAMES[t] || t;
  }
  const TARGETS = new Proxy({}, { get: (_, t) => targetName(t) });
  // User-facing naming: the product is "Agent Platform". Backend strings may still say Vertex AI.
  const ap = (s) => String(s == null ? "" : s)
    .replace(/Gemini on Vertex AI/g, "Gemini on Agent Platform")
    .replace(/Vertex AI/g, "Agent Platform")
    .replace(/\bVertex \+ /g, "Agent Platform + ")
    .replace(/\bVertex, on demand/g, "Agent Platform, on demand");
  const targetColor = (t) => (t && t.indexOf("gemini") === 0 ? "var(--s-gem)" : "var(--s-maas)");
  const targetClass = (t) => (t && t.indexOf("gemini") === 0 ? "fill-gem" : "fill-maas");
  const STEPS = [
    { id: "plan", name: "Plan", desc: "Size the hardware" },
    { id: "prompts", name: "Prompts", desc: "Prompt set" },
    { id: "deploy", name: "Deploy", desc: "Endpoint, 10–30 min" },
    { id: "load", name: "Load sweep", desc: "Ramp concurrency" },
    { id: "analyze", name: "Analyze", desc: "Peak in budget" },
    { id: "calibrate", name: "Calibrate", desc: "Short-only pass" },
    { id: "report", name: "Report", desc: "Report, dashboard" },
    { id: "teardown", name: "Tear down", desc: "Delete the endpoint" },
  ];
  const SOURCE_LABEL = { estimated: "Estimated", calibrated: "Calibrated", measured: "Measured", override: "Your value" };
  // gem: per-model Gemini price overrides {model id: {in, out}}; liteIn … g4In are the older per-role fields (kept).
  const DEFAULT_UI = { pricing: "auto", offPeak: 1, growth: 1, tputMult: 1, liteIn: null, liteOut: null, flashIn: null, flashOut: null, proIn: null, proOut: null, g4In: null, g4Out: null, gem: null, fleetOverride: null };
  const GEM_UI = { liteIn: ["gemini_flash_lite", "input_per_m"], liteOut: ["gemini_flash_lite", "output_per_m"], flashIn: ["gemini_flash", "input_per_m"], flashOut: ["gemini_flash", "output_per_m"], proIn: ["gemini_pro", "input_per_m"], proOut: ["gemini_pro", "output_per_m"], g4In: ["gemini_4", "input_per_m"], g4Out: ["gemini_4", "output_per_m"] };
  const HOURS_MONTH = 730;
  const SEC_MONTH = 3600 * HOURS_MONTH; // monthly input tokens = weekly average tok/s × SEC_MONTH
  const U = 0.85; // usable fraction of fleet capacity (matches model.js fixedFleet)

  // ---------------------------------------------------------------- state
  const S = {
    catalog: null,
    sessions: [],
    name: null,
    session: null,
    cfg: null,
    ui: Object.assign({}, DEFAULT_UI),
    dirty: false,
    data: null,
    p: null,
    base: null,
    r: null,
    profileError: null,
    tab: "costs",
    busy: "",
    job: null,
    sse: null,
    poll: null,
    tick: null,
    yamlDirty: false,
    models: null,
    modelsLoading: false,
    hw: null,
    hwLoading: false,
    hwReq: 0,
    hwFilter: "all",
    hwSort: { key: "usd", dir: 1 },
    drawer: null,
    drawerView: { model: "table", hw: "table" },
    drawerReturn: null,
    drawn: null,
    anim: 0,
    drag: null,
    native: undefined,
    autoEstimated: {},
    tpsEdit: undefined, // pending throughput override from the rail: number, null (clear) or undefined (no edit)
    simRunning: false,
    shape: { on: false, mode: "repeat", brush: "smooth", work: null, undo: [], cursor: 0, ymax: null, stroke: null },
    mdl: { q: "", family: "all", size: "all", fits: false, older: false, sort: "match", limit: 40 },
    hwView: "usd",
    describe: null,
  };

  // ---------------------------------------------------------------- API
  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail || j.error || detail; } catch (_) { /* not JSON */ }
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = res.status;
      throw err;
    }
    return res.json();
  }
  const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

  // ---------------------------------------------------------------- small UI helpers
  function svg(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const icon = (id, cls) => `<svg class="ico${cls ? " " + cls : ""}" aria-hidden="true" focusable="false"><use href="#${id}"/></svg>`;
  // An override is "Your value" only when the user set it; a preset figure (e.g. the example's) is "Assumed".
  const sourceText = (src) => {
    const t = throughput() || {};
    return (src === "override" && !t.pending && !/set by you/i.test(t.label || ""))
      ? "Assumed" : (SOURCE_LABEL[src] || "Estimated");
  };
  // Estimates name their tier (planner.reference_throughput): a reference benchmark, scaled from one, or GPU specs.
  const BASIS_TAG = { reference: "Reference est.", scaled: "Scaled est.", spec: "Rough estimate" };
  const BASIS_TIP = {
    reference: "Measured on this GPU with the reference workload (Gemma 4 26B-A4B FP8, ~5K-token prompts, 500-token answers, no thinking). Your workload may differ: run a benchmark with your prompts.",
    scaled: "Scaled from a reference benchmark on this GPU with another model or configuration. Your workload may differ: run a benchmark with your prompts.",
    spec: "No reference benchmark for this GPU yet: a rough estimate from its specs. Real throughput can differ by several times. Run a benchmark with your prompts.",
  };
  const needsCaution = (src) => src === "estimated" || src === "override";
  const tagHTML = (src, basis, note) => {
    const b = src === "estimated" ? (BASIS_TAG[basis] ? basis : "spec") : null;
    const text = b ? BASIS_TAG[b] : sourceText(src);
    const tipText = b ? (note || BASIS_TIP[b]) : (needsCaution(src) ? "Not measured on your workload. Run a benchmark with your prompts." : "");
    return `<span class="tag tag-${esc(src)}${b ? " tag-basis-" + b : ""}${needsCaution(src) ? " tag-caution" : ""}"${tipText ? ` title="${esc(tipText)}"` : ""}>${needsCaution(src) ? icon("i-alert", "ico-sm") : ""}${esc(text)}</span>`;
  };
  // Permanent inline caution next to throughput figures that aren't measured: links to the Benchmark tab.
  const cautionHTML = (text) => `<button type="button" class="caution" data-goto-tab="bench" title="Throughput here is an estimate, not measured on your workload. Open the Benchmark tab to measure it with your prompts.">${icon("i-alert", "ico-sm")}<span>${esc(text || "Estimate: not your workload")}</span></button>`;

  let toastTimer = null;
  function toast(msg, isErr) {
    const t = $("toast");
    t.textContent = msg;
    t.className = "toast" + (isErr ? " err" : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, isErr ? 7000 : 3200);
  }

  const tip = $("tip");
  function showTip(evt, title, rows) {
    tip.textContent = "";
    const t = document.createElement("div"); t.className = "t"; t.textContent = title; tip.appendChild(t);
    for (const [color, value, label] of rows) {
      const r = document.createElement("div"); r.className = "r";
      const i = document.createElement("i"); if (color) i.style.background = color; r.appendChild(i);
      const b = document.createElement("b"); b.textContent = value; r.appendChild(b);
      const s = document.createElement("span"); s.textContent = label; r.appendChild(s);
      tip.appendChild(r);
    }
    tip.hidden = false;
    const x = Math.min(window.innerWidth - tip.offsetWidth - 10, evt.clientX + 14);
    const y = Math.min(window.innerHeight - tip.offsetHeight - 10, evt.clientY + 14);
    tip.style.left = Math.max(10, x) + "px";
    tip.style.top = Math.max(10, y) + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  // Generic modal dialog. Resolves to the input value (or true) on OK, null on cancel.
  function dialog(opts) {
    return new Promise((resolve) => {
      const d = $("dlg");
      $("dlgTitle").textContent = opts.title;
      const body = $("dlgBody");
      body.innerHTML = opts.body || "";
      let input = null;
      if (opts.input) {
        const f = document.createElement("div"); f.className = "field";
        f.innerHTML = `<label for="dlgInput">${esc(opts.input.label)}</label><input type="text" id="dlgInput" spellcheck="false">`;
        body.appendChild(f);
        input = f.querySelector("input");
        input.value = opts.input.value || "";
      }
      const ok = $("dlgOk"), cancel = $("dlgCancel");
      ok.textContent = opts.okLabel || "OK";
      cancel.textContent = opts.cancelLabel || "Cancel";
      ok.className = "btn " + (opts.danger ? "btn-danger-solid" : "btn-primary");
      const done = (val) => {
        ok.onclick = cancel.onclick = null;
        d.onclose = null;
        if (d.open) d.close();
        resolve(val);
      };
      ok.onclick = (e) => {
        e.preventDefault();
        if (input) {
          const v = input.value.trim();
          if (!v) { input.focus(); return; }
          if (opts.input.validate) {
            const msg = opts.input.validate(v);
            if (msg) { input.setCustomValidity(msg); input.reportValidity(); input.oninput = () => input.setCustomValidity(""); return; }
          }
          done(v);
        } else done(true);
      };
      cancel.onclick = () => done(opts.cancelValue != null ? opts.cancelValue : null);
      d.onclose = () => done(null);
      d.onkeydown = (e) => {   // keep Tab inside the dialog (native modals let it escape to the browser chrome)
        if (e.key !== "Tab") return;
        const f = [...d.querySelectorAll("button, input, select, textarea, a[href], [tabindex]:not([tabindex='-1'])")].filter((x) => !x.disabled && x.offsetParent !== null);
        if (!f.length) return;
        const first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      };
      d.showModal();
      (input || (opts.danger ? cancel : ok)).focus();
      if (input) input.select();
    });
  }

  // ---------------------------------------------------------------- config helpers
  function getPath(o, path) {
    return path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
  }
  function setPath(o, path, v) {
    const ks = path.split(".");
    let cur = o;
    for (let i = 0; i < ks.length - 1; i++) {
      if (cur[ks[i]] == null || typeof cur[ks[i]] !== "object") cur[ks[i]] = /^\d+$/.test(ks[i + 1]) ? [] : {};
      cur = cur[ks[i]];
    }
    cur[ks[ks.length - 1]] = v;
  }
  const normRes = (v) => { v = +v; if (!(v >= 1)) return 1; return v >= 1.75 ? 2 : v >= 1.25 ? 1.5 : 1; };

  function normalizeCfg(c) {
    c.use_case = c.use_case || {};
    c.model = c.model || {};
    c.deployment = c.deployment || {};
    c.traffic = c.traffic || {};
    if (!Array.isArray(c.traffic.busy_hours_utc) || c.traffic.busy_hours_utc.length !== 2) c.traffic.busy_hours_utc = [13, 20];
    if (c.traffic.weekend_factor == null) c.traffic.weekend_factor = 0.15;
    if (monthlyAnchor(c.traffic)) c.traffic.average_input_tps = Math.round(+c.traffic.monthly_input_tokens / SEC_MONTH);
    c.requests = c.requests || {};
    const r = c.requests;
    if (r.pct_over_128k == null) r.pct_over_128k = r.pct_long || 0;
    if (r.pct_over_32k == null) r.pct_over_32k = (r.pct_medium || 0) + r.pct_over_128k;
    syncBuckets(c);
    if (c.traffic.preset && !TRAFFIC_PRESETS[c.traffic.preset]) c.traffic.preset = null;
    c.pricing = c.pricing || {};
    if (!c.pricing.tier) c.pricing.tier = "auto";
    c.maas = c.maas || {};
    c.hybrid = c.hybrid || {};
    c.constraints = c.constraints || {};
    c.constraints.resilience_multiplier = normRes(c.constraints.resilience_multiplier);
    if (!c.constraints.burst_headroom) c.constraints.burst_headroom = 1.3;
    c.prompts = c.prompts || {};
    c.prompts.synthetic = c.prompts.synthetic || {};
    c.loadtest = c.loadtest || {};
    if (!Array.isArray(c.loadtest.concurrency) || !c.loadtest.concurrency.length) c.loadtest.concurrency = [16, 32, 64, 128];
    if (!c.loadtest.requests_per_level) c.loadtest.requests_per_level = 300;
    return c;
  }
  function syncBuckets(c) {
    const r = c.requests;
    let o32 = Math.max(0, Math.min(100, +r.pct_over_32k || 0));
    let o128 = Math.max(0, Math.min(o32, +r.pct_over_128k || 0));
    r.pct_long = +o128.toFixed(2);
    r.pct_medium = +(o32 - o128).toFixed(2);
    r.pct_short = +(100 - o32).toFixed(2);
  }
  function collectCfg() {
    const c = clone(S.cfg);
    c.name = S.name;
    syncBuckets(c);
    return c;
  }
  function uiState() {
    const u = Object.assign({}, S.ui);
    return u;
  }

  // ---------------------------------------------------------------- weekly traffic (port of traffic.parametric)
  function windowShape(busy, ramp) {
    ramp = ramp || 5;
    const [start, end] = busy;
    const s = [];
    for (let i = 0; i < 24; i++) {
      const x = i + 0.5;
      let v = 0;
      if (x >= start && x <= end) v = 1;
      else if (x > start - ramp && x < start) v = 0.5 - 0.5 * Math.cos(Math.PI * (x - (start - ramp)) / ramp);
      else if (x > end && x < end + ramp) v = 0.5 + 0.5 * Math.cos(Math.PI * (x - end) / ramp);
      s.push(v);
    }
    return s;
  }
  function parametric(peak, avg, floor, busy, wk) {
    if (!(peak > 0 && avg > 0 && floor >= 0)) throw new Error("Enter the peak, average and quietest hour in tokens per second.");
    if (!(floor <= avg && avg <= peak)) throw new Error("The quietest hour must be at most the average, and the average at most the peak.");
    const base = windowShape(busy || [13, 20]);
    const w = wk == null ? 0.15 : wk;
    const week = (p) => {
      const wd = base.map((b) => floor + (peak - floor) * Math.pow(b, p));
      const we = wd.map((v) => floor + w * (v - floor));
      return [].concat(wd, wd, wd, wd, wd, we, we);
    };
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    let lo = 1e-3, hi = 50;
    for (let i = 0; i < 80; i++) {
      const mid = Math.sqrt(lo * hi);
      if (mean(week(mid)) > avg) lo = mid; else hi = mid;
    }
    const out = week(Math.sqrt(lo * hi));
    if (Math.abs(mean(out) - avg) / avg > 0.02) {
      throw new Error(`An average of ${fmtTok(avg)} tok/s can't be reached with this peak, quietest hour and weekend load ` +
        `(possible: about ${fmtTok(mean(week(50)))}–${fmtTok(mean(week(1e-3)))}).`);
    }
    return out;
  }
  const isCustom = (t) => !t.hourly_csv && Array.isArray(t.custom_hourly) && t.custom_hourly.length === 168;

  // Named traffic shapes (port of traffic.PRESETS / traffic.preset_week; keep the two identical).
  // Each region is busy over the same local hours, weighted by its share of users; Monday 00:00 UTC first.
  const REGION_UTC = { us_east: -5, us_west: -8, eu: 1, apac: 8 };
  const TRAFFIC_PRESETS = {
    us_core_hours: { mix: { us_east: 0.8, us_west: 0.2 }, floor: 0.015, weekend: 0.04, hours: [10, 16.5] },
    us_europe: { mix: { us_east: 0.35, us_west: 0.2, eu: 0.45 }, floor: 0.08, weekend: 0.2, hours: [8.5, 18] },
    follow_the_sun: { mix: { us_east: 0.3, us_west: 0.1, eu: 0.3, apac: 0.3 }, floor: 0.15, weekend: 0.35, hours: [8.5, 18] },
    steady: { mix: { us_east: 0.25, us_west: 0.25, eu: 0.25, apac: 0.25 }, floor: 0.7, weekend: 0.9, hours: [8.5, 18] },
  };
  const PRESET_PHRASE = { us_core_hours: "US business hours", us_europe: "US + Europe", follow_the_sun: "follow the sun", steady: "steady machine traffic" };
  function presetBump(start, end, h) {
    const sg = (x) => 1 / (1 + Math.exp(-x / (1.5 / 3)));
    h = ((h % 24) + 24) % 24;
    return Math.max(sg(h - start) * sg(end - h), sg(h + 24 - start) * sg(end - h - 24), sg(h - 24 - start) * sg(end - h + 24));
  }
  const presetShapes = {};
  function presetShape(key) {
    if (presetShapes[key]) return presetShapes[key];
    const P = TRAFFIC_PRESETS[key];
    const day = [];
    for (let h = 0; h < 24; h++) {
      let v = 0;
      for (const r in P.mix) v += P.mix[r] * presetBump(P.hours[0], P.hours[1], h + REGION_UTC[r]);
      day.push(v);
    }
    const mx = Math.max.apply(null, day);
    const wd = day.map((v) => P.floor + (1 - P.floor) * v / mx);
    const we = day.map((v) => P.floor + (1 - P.floor) * (v / mx) * P.weekend);
    presetShapes[key] = [].concat(wd, wd, wd, wd, wd, we, we);
    return presetShapes[key];
  }
  function presetWeek(key, avg) {
    const sh = presetShape(key), m = sh.reduce((a, v) => a + v, 0) / sh.length;
    return sh.map((v) => v * avg / m);
  }
  // Active preset: precedence hourly_csv > custom_hourly > preset > parametric (as traffic.weekly_profile).
  const presetOf = (t) => (t && !t.hourly_csv && !isCustom(t) && t.preset && TRAFFIC_PRESETS[t.preset] ? t.preset : null);
  const meanOf = (a) => a.reduce((x, v) => x + (+v || 0), 0) / Math.max(1, a.length);
  // Monthly anchor: the weekly profile is rescaled so its monthly total matches (contract: traffic.monthly_input_tokens).
  const monthlyAnchor = (t) => t.monthly_input_tokens != null && +t.monthly_input_tokens > 0;
  function currentProfile() {
    const s = S.session, t = S.cfg.traffic;
    if (t.hourly_csv) {
      S.profileError = null;
      return (s.traffic_profile && s.traffic_profile.length === 168) ? s.traffic_profile : (s.dashboard_data && s.dashboard_data.profile) || null;
    }
    if (isCustom(t)) {
      S.profileError = null;
      let prof = t.custom_hourly.map((v) => Math.max(0, +v || 0));
      if (monthlyAnchor(t)) {
        const m = meanOf(prof), want = +t.monthly_input_tokens / SEC_MONTH;
        if (m > 0) prof = prof.map((v) => v * want / m);
      }
      return prof;
    }
    const pk = presetOf(t);
    if (pk) {
      const avg = monthlyAnchor(t) ? +t.monthly_input_tokens / SEC_MONTH : +t.average_input_tps;
      if (avg > 0) { S.profileError = null; return presetWeek(pk, avg); }
    }
    try {
      const prof = parametric(+t.peak_input_tps, +t.average_input_tps, +t.floor_input_tps, t.busy_hours_utc.map(Number), +t.weekend_factor);
      S.profileError = null;
      return prof;
    } catch (e) {
      S.profileError = e.message;
      return (S.data && S.data.profile) || (s.traffic_profile && s.traffic_profile.length ? s.traffic_profile : null);
    }
  }

  // ---------------------------------------------------------------- catalog lookups
  // The engines' Gemini map {role alias or model id: entry}. Older catalogs sent a list or alias-only map: their
  // entries are filed under the alias they belong to.
  function normGemini(g) {
    if (!g) return {};
    if (!Array.isArray(g) && GEMINI_KEYS.some((k) => g[k])) return g;
    const out = {};
    const list = Array.isArray(g) ? g : Object.keys(g).map((k) => Object.assign({ id: k }, g[k]));
    for (const it of list) {
      const id = String(it.id || it.model_id || "").toLowerCase();
      const key = GEMINI_KEYS.indexOf(id) >= 0 ? id
        : /gemini[-_ ]?4(\b|[-_])/.test(id + " " + String(it.name || "").toLowerCase()) && id.indexOf("lite") < 0 ? "gemini_4"
        : id.indexOf("lite") >= 0 ? "gemini_flash_lite" : id.indexOf("pro") >= 0 ? "gemini_pro" : id.indexOf("flash") >= 0 ? "gemini_flash" : null;
      if (key && !out[key]) out[key] = it;
    }
    return out;
  }
  // Every live Gemini model (newest first within family) and the role aliases, from /api/catalog.
  function geminiModels() {
    const list = S.catalog && S.catalog.gemini_models;
    if (Array.isArray(list) && list.length) return list;
    const g = geminiCatalog();
    return Object.keys(g).filter((k) => GEMINI_KEYS.indexOf(k) < 0 && !g[k].alias)
      .map((k) => Object.assign({ id: k, roles: [] }, g[k]));
  }
  function geminiModel(id) { return geminiModels().find((m) => m.id === id) || null; }
  function geminiAliases() {
    const a = S.catalog && S.catalog.gemini_aliases;
    const g = geminiCatalog();
    return GEMINI_KEYS.map((k) => {
      const x = (a && a[k]) || {};
      const e = g[k] || {};
      return { key: k, label: x.label || (k === "gemini_4" ? "Gemini 4 flagship" : GEMINI_FALLBACK_NAMES[k]),
               model_id: x.model_id || e.model_id || null, name: x.name || e.name || GEMINI_FALLBACK_NAMES[k], rule: x.rule || e.rule || "" };
    });
  }
  const isGeminiTarget = (t) => !!t && String(t).indexOf("gemini") === 0;
  const stageTag = (m) => (m && m.stage === "preview" ? " (Preview)" : m && m.deprecated ? " (deprecated)" : m && m.stage === "unlisted" ? " (not in Model Garden yet)" : "");
  function shortModelName(n) {
    return String(n || "").replace(/\s*\((MoE|Dense)\)/gi, "").replace(/\s+Instruct\b/gi, "").replace(/\s+/g, " ").trim();
  }
  function modelInfo(mgid) {
    const list = (S.catalog && S.catalog.models) || [];
    const lc = String(mgid || "").toLowerCase();
    return list.find((m) => m.model_garden_id === mgid) ||
      list.find((m) => lc && ((m.hf_id && m.hf_id.toLowerCase() === lc) || (Array.isArray(m.aliases) && m.aliases.some((a) => String(a).toLowerCase() === lc)))) ||
      ((S.models && S.models.recommendations) || []).find((m) => m.model_garden_id === mgid) || null;
  }
  function modelName(m) {
    if (!m) return "No model";
    const info = modelInfo(m.model_garden_id);
    if (m.name) return shortModelName(m.name);
    if (info && info.name) return shortModelName(info.name);
    const id = m.model_garden_id || m.hf_id || "";
    return id.split("@").pop().split("/").pop() || "No model";
  }
  function accName(type) {
    const a = S.catalog && S.catalog.accelerators && S.catalog.accelerators[type];
    if (a && a.name) return a.name.replace(/^Cloud\s+/, "");
    const dd = S.session && S.session.dashboard_data;
    const me = S.session && S.session.measured && S.session.measured.endpoint;
    if (dd && me && me.accelerator_type === type && dd.meta && dd.meta.gpu) return dd.meta.gpu.replace(/^Cloud\s+/, "");
    return String(type || "accelerator").replace(/^NVIDIA_(TESLA_)?/, "").replace(/_/g, " ");
  }
  function accKind(type) {
    const a = S.catalog && S.catalog.accelerators && S.catalog.accelerators[type];
    if (a && a.kind) return a.kind;
    if (/TPU/i.test(type || "")) return "tpu";
    return (S.session && S.session.dashboard_data && S.session.dashboard_data.meta && S.session.dashboard_data.meta.kind) || "gpu";
  }
  const chipWord = (type, n) => accKind(type) === "tpu" ? (n === 1 ? "TPU chip" : "TPU chips") : (n === 1 ? "GPU" : "GPUs");

  // Single source of truth for what is planned. Prefer the contract field, fall back to what priced the numbers.
  function selected() {
    const s = S.session;
    // The contract field, also when nothing fits the chosen model (hardware null + no_fit): never fall back to
    // what priced an older selection.
    if (s && s.selected && s.selected.model && (s.selected.hardware || s.no_fit || s.plan_error)) return s.selected.hardware ? s.selected : Object.assign({}, s.selected, { hardware: {} });
    const cm = (s && s.config && s.config.model) || {};
    const ep = (s && s.measured && s.measured.endpoint) || (s && s.plan && s.plan.chosen) || (s && s.config && s.config.deployment) || {};
    return {
      model: { model_garden_id: cm.model_garden_id, hf_id: cm.hf_id, name: null },
      hardware: { accelerator_type: ep.accelerator_type, accelerator_count: ep.accelerator_count, machine_type: ep.machine_type },
      precision: cm.precision,
    };
  }
  function throughput() {
    const s = S.session;
    if (S.tpsEdit > 0) return { per_chip: +S.tpsEdit, source: "override", pending: true };
    if (S.tpsEdit === null && s && s.throughput && s.throughput.source === "override") {
      const ch = s.plan && s.plan.chosen;
      if (ch && ch.est_input_tps_per_gpu) return { per_chip: ch.est_input_tps_per_gpu, source: "estimated", basis: ch.est_basis, note: ch.est_basis_note, pending: true };
    }
    if (s && s.throughput && s.throughput.per_chip > 0) return s.throughput;
    const m = s && s.measured;
    if (m && m.input_tps_per_gpu > 0) return { per_chip: m.input_tps_per_gpu, source: m.estimated ? "estimated" : "measured" };
    const ch = s && s.plan && s.plan.chosen;
    if (ch && ch.est_input_tps_per_gpu) return { per_chip: ch.est_input_tps_per_gpu, source: "estimated", basis: ch.est_basis, note: ch.est_basis_note };
    return null;
  }
  function pickTiers(prices) {
    const out = {};
    if (!prices) return out;
    for (const t of TIER_ORDER) if (prices[t] > 0) out[t] = prices[t];
    return out;
  }
  function geminiCatalog() {
    return normGemini((S.catalog && S.catalog.gemini) || (S.session && S.session.dashboard_data && S.session.dashboard_data.gemini));
  }

  // Build the model.js DATA object for the current session.
  function buildData() {
    const s = S.session;
    const dd = s.dashboard_data;
    const ch = s.plan && s.plan.chosen;
    const tp = throughput();
    let prices = dd && dd.prices ? pickTiers(dd.prices) : null;
    if ((!prices || !Object.keys(prices).length) && ch && ch.prices) prices = pickTiers(ch.prices);
    if (!prices || !Object.keys(prices).length || !tp) { S.data = null; return; }
    const sel = selected();
    const prof = currentProfile();
    if (!prof) { S.data = null; return; }
    const gem = geminiCatalog();
    S.data = {
      profile: prof,
      perGpu: tp.per_chip,
      // Which prompt mix perGpu belongs to (1 = short prompts; a measured run records its own mix).
      mixFactorRef: tp.pending ? 1 : ((dd && dd.mixFactorRef) ?? 1),
      // Long-prompt rates measured by a benchmark (they belong to its short-prompt rate, not to a typed one).
      classTps: tp.pending ? null : ((dd && dd.classTps) || null),
      replica: (dd && dd.replica) || sel.hardware.accelerator_count || 1,
      prices,
      pricingLabels: Object.fromEntries(Object.keys(prices).map((k) => [k, TIER_LABELS[k] || k])),
      gemini: gem,
      meta: { model: sel.model.model_garden_id, gpu: accName(sel.hardware.accelerator_type), kind: accKind(sel.hardware.accelerator_type) },
    };
    S.native = undefined;
  }

  // ---------------------------------------------------------------- the model
  function gemMap(p) {
    if (M.getGeminiPrices) return M.getGeminiPrices(S.data, p);
    const g = (S.data && S.data.gemini) || {};
    const pro = g.gemini_pro || {};
    const proIn = pro.input_per_m != null ? pro.input_per_m : 1.25;
    return {
      gemini_flash_lite: { in: p.liteIn, out: p.liteOut, cached: p.liteIn * 0.25 },
      gemini_flash: { in: p.flashIn, out: p.flashOut, cached: p.flashIn * 0.25 },
      gemini_pro: { in: proIn, out: pro.output_per_m != null ? pro.output_per_m : 10, cached: proIn * 0.25 },
    };
  }
  function baseParams() {
    const c = S.cfg, g = (S.data && S.data.gemini) || {};
    const gv = (k, f, d) => (g[k] && g[k][f] != null ? +g[k][f] : d);
    const r = c.requests, m = c.maas, h = c.hybrid, k = c.constraints;
    const p = {
      resilience: normRes(k.resilience_multiplier),
      offPeak: +S.ui.offPeak || 1,
      growth: +S.ui.growth || 1,
      tputMult: +S.ui.tputMult || 1,
      avgInput: Math.max(1, +r.avg_input_tokens || 1),
      avgOutput: Math.max(0, +r.avg_output_tokens || 0),
      thinkingShare: Math.max(0, Math.min(1, +r.thinking_share || 0)),
      thinkingTokens: Math.max(0, +r.avg_thinking_tokens || 0),
      pctOver32: +r.pct_over_32k || 0,
      pctOver128: Math.min(+r.pct_over_128k || 0, +r.pct_over_32k || 0),
      mediumMean: +r.medium_mean || 69000,
      largeMean: +r.large_mean || 188000,
      mediumTps: (c.throughput && +c.throughput.medium_tps) || null,
      largeTps: (c.throughput && +c.throughput.large_tps) || null,
      systemPrompt: +r.system_prompt_tokens || 0,
      systemHit: (r.system_prompt_cache_hit != null ? +r.system_prompt_cache_hit : 0.9) * 100,
      cachedInput: r.cached_input_tokens != null && r.cached_input_tokens !== "" ? +r.cached_input_tokens : null,
      routeLong: !!h.route_long,
      routeLongTarget: h.route_long_target || "gemini_flash_lite",
      peakOverflowTarget: h.peak_overflow_target || "open_maas",
      maasRps: Math.max(0, +m.max_rps || 0),
      headroom: Math.max(1, +k.burst_headroom || 1.3),
      maasIn: +m.input_per_m || 0,
      maasOut: +m.output_per_m || 0,
      maasCached: m.cached_per_m != null ? +m.cached_per_m : 0,
      failoverTarget: h.failover_target || "gemini_flash_lite",
      gemOverrides: S.ui.gem || {},   // per-model Gemini price overrides from the rail
    };
    // Gemini prices come from DATA.gemini (Billing Catalog) unless the user overrides them in the rail.
    for (const k in GEM_UI) {
      if (S.ui[k] != null && isFinite(+S.ui[k])) p[k] = +S.ui[k];
      else if (!M.getGeminiPrices) p[k] = gv(GEM_UI[k][0], GEM_UI[k][1], undefined);
    }
    // Targets the engine can't price (no model of that kind) fall back to Flash-Lite; a pinned model the catalog no
    // longer lists is priced as its role alias (model.js geminiKey).
    const ok = (t) => t === "open_maas" || !!(M.geminiKey ? M.geminiKey(t, gemMap(p)) : gemMap(p)[t]);
    if (!ok(p.peakOverflowTarget)) p.peakOverflowTarget = "gemini_flash_lite";
    if (!ok(p.routeLongTarget)) p.routeLongTarget = "gemini_flash_lite";
    if (p.failoverTarget === "open_maas" || !ok(p.failoverTarget)) p.failoverTarget = "gemini_flash_lite";
    return withThinking(p);
  }
  // Thinking: model.js ≥ round 2 adds share × thinking tokens itself (M.effectiveOutput). Older engines only know
  // avgOutput, so pass the effective output in it instead, never both.
  const effectiveOutput = (p) => (M.effectiveOutput ? M.effectiveOutput(p) : p.avgOutput + p.thinkingShare * p.thinkingTokens);
  function withThinking(p) {
    if (!M.effectiveOutput) { p.answerOutput = p.avgOutput; p.avgOutput = effectiveOutput(p); p.thinkingShare = 0; }
    return p;
  }

  // Honest baseline: open-model MaaS can only serve the peak if its rate limit allows. When it can't, the baseline is
  // MaaS up to its limit plus overflow to the failover Gemini tier, priced per token. Prefer the engine's fields.
  function baselineOf(r, p) {
    p = p || S.p;
    if (!r) return null;
    if (r.baseline && r.baseline.total != null) {
      const b = r.baseline;
      return {
        total: +b.total, feasible: r.maasOnlyFeasible !== undefined ? !!r.maasOnlyFeasible : b.kind !== "maas_plus_overflow",
        maasCost: b.maas_cost, overflowCost: b.overflow_cost, target: b.overflow_target || p.failoverTarget,
        peakRps: r.maasPeakRps, capRps: r.maasCapRps != null ? r.maasCapRps : p.maasRps,
        vs: r.total != null && +b.total > 0 ? r.total / b.total : null, native: true, label: b.label,
      };
    }
    if (!r.demand || r.maasOnly == null) return null;
    const D = r.demand, scale = HOURS_MONTH / D.length;
    const perReq = Math.max(1, p.avgInput);
    const peakRps = Math.max.apply(null, D) / perReq;
    const capRps = Math.max(0, p.maasRps);
    const feasible = peakRps <= capRps + 1e-9;
    let total = r.maasOnly, maasCost = r.maasOnly, overflowCost = 0;
    const target = p.failoverTarget || "gemini_flash_lite";
    if (!feasible) {
      const capTps = capRps * perReq;
      const g = gemMap(p)[target] || gemMap(p).gemini_flash_lite;
      const mRate = M.maasRate(p, r.overallOut, r.overallCache), gRate = M.geminiRate(g, r.overallOut, r.overallCache);
      let onM = 0, over = 0;
      for (const d of D) { onM += Math.min(d, capTps); over += Math.max(0, d - capTps); }
      maasCost = onM * 3600 * scale * mRate; overflowCost = over * 3600 * scale * gRate;
      total = maasCost + overflowCost;
    }
    return { total, feasible, maasCost, overflowCost, target, peakRps, capRps, vs: r.total != null ? r.total / total : null, native: false };
  }
  const vsBase = (r, p) => { const b = baselineOf(r, p); return b ? b.vs : NaN; };
  const baseShort = (b) => (b && !b.feasible ? "MaaS + overflow" : "MaaS");
  const fmtRps = (x) => (x == null || !isFinite(x)) ? "—" : (x < 10 ? x.toFixed(1).replace(/\.0$/, "") : fmtInt(x));

  function cheapestTier(p) {
    let best = null, bestT = null;
    for (const t of Object.keys(S.data.prices)) {
      const r = evaluate(Object.assign({}, p, { pricing: t }), null);
      if (r && r.total != null && (best == null || r.total < best)) { best = r.total; bestT = t; }
    }
    return bestT || Object.keys(S.data.prices)[0];
  }
  function params() {
    const p = baseParams();
    p.pricing = S.ui.pricing && S.ui.pricing !== "auto" && S.data.prices[S.ui.pricing] ? S.ui.pricing : cheapestTier(Object.assign({}, p, { pricing: Object.keys(S.data.prices)[0] }));
    return p;
  }
  function peakFleet(base, p) {
    const replica = S.data.replica || 1;
    const D = base.routing ? base.demand.map((d) => d * base.buckets.small.tokenShare) : base.demand;
    const need = (Math.max.apply(null, D) * p.headroom) / Math.max(1, base.capPerGpu);
    return Math.max(replica, Math.ceil(need / replica) * replica);
  }
  // Detect whether model.js supports fleetOverride natively (backend contract), and in which unit.
  function nativeOverride(p, base) {
    if (S.native !== undefined) return S.native;
    if (M.evalFleet) { S.native = "replicas"; return S.native; } // model.js contract: fleetOverride in whole replicas
    S.native = false;
    try {
      if (!base.fleet) return false;
      const rep = S.data.replica || 1;
      const want = base.fleet.gpus + rep;
      const t = M.scenario(S.data, Object.assign({}, p, { fleetOverride: want }));
      if (t.fleet && t.fleet.gpus === want && t.fleet.gpus !== base.fleet.gpus) S.native = "chips";
      else if (t.fleet && t.fleet.gpus === want * rep && rep > 1) S.native = "replicas";
    } catch (_) { S.native = false; }
    return S.native;
  }
  // Fleet of exactly n chips (before redundancy). Spill above the MaaS rate limit is priced at Gemini Flash-Lite.
  function atFleet(p, base, n) {
    const nat = nativeOverride(p, base);
    if (nat) {
      const rep = S.data.replica || 1;
      const t = M.scenario(S.data, Object.assign({}, p, { fleetOverride: nat === "chips" ? n : n / rep }));
      if (t && t.fleet && t.fleet.gpus === n && t.total != null) { t.overridden = true; t.infeasible = null; return t; }
    }
    const b = base.buckets, routing = base.routing, D = base.demand, replica = S.data.replica || 1;
    const Ds = routing ? D.map((d) => d * b.small.tokenShare) : D;
    const cap = (base.maasCapTps || 0) / p.headroom, perGpu = base.capPerGpu, scale = 730 / D.length;
    const g = gemMap(p);
    const ratio = routing ? b.small.outRatio : base.overallOut;
    const cache = routing ? b.small.cached : base.overallCache;
    const ot = p.peakOverflowTarget || "open_maas";
    const rate = g[ot] ? M.geminiRate(g[ot], ratio, cache) : M.maasRate(p, ratio, cache);
    const failRate = M.geminiRate(g[p.failoverTarget] || g.gemini_flash_lite, ratio, cache);
    let onGpu = 0, spill = 0, excess = 0;
    for (const d of Ds) {
      const gg = Math.min(d, n * perGpu * U), s = d - gg;
      onGpu += gg; spill += Math.min(s, cap); excess += Math.max(0, s - cap);
    }
    const excessCost = excess * 3600 * scale * failRate;
    const fleet = {
      gpus: n, gpuCost: n * base.price * 730,
      maasCost: spill * 3600 * scale * rate + excessCost,
      excessCost, util: onGpu / Ds.length / (n * perGpu),
      gpuTokenShare: onGpu / Math.max(1e-9, onGpu + spill + excess),
    };
    const gpus = Math.ceil((n * p.resilience) / replica) * replica;
    const r = Object.assign({}, base, { fleet, gpus, gpuCost: gpus * base.price * 730, overridden: true, infeasible: null });
    r.total = r.gpuCost + fleet.maasCost + (base.longMaasCost || 0);
    r.perM = r.total / (base.totalTok / 1e6);
    return r;
  }
  // Evaluate a scenario; n = fleet override in chips or null for the cost-optimal fleet.
  function evaluate(p, n) {
    const base = M.scenario(S.data, p);
    if (!base.capPerGpu) return base; // long-prompt routing infeasible: no fleet to draw
    if (n != null && base.fleet && n === base.fleet.gpus) return base;
    if (n != null) return atFleet(p, base, n);
    if (base.infeasible && !base.fleet) {
      const r = atFleet(p, base, peakFleet(base, p));
      r.overridden = false;
      r.fallbackNote = base.infeasible;
      return r;
    }
    return base;
  }

  // Fleet policy (model.js evalFleet, costmodel.eval_fleet; governed by "Peak overflow goes to"): an overridden fleet
  // that needs Gemini at peaks while peak overflow goes to open-model MaaS, or sits below the burst headroom, is a
  // what-if, never presented as the better option. Returns short reasons ([] when the fleet is within the policy).
  function policyIssues(r, p) {
    p = p || S.p;
    if (!r || !r.overridden || r.policyOk !== false) return [];
    const out = [];
    if (r.needsGeminiAtPeaks) out.push({ key: "gemini", short: "needs Gemini at peaks", text: "Needs Gemini at peaks: not allowed while peak overflow goes to open-model MaaS." });
    if (r.overOverflowQuota) out.push({ key: "quota", short: "exceeds the Gemini overflow quota", text: `Exceeds the ${fmtInt(M.GEMINI_OVERFLOW_RPS || 150)} req/s ${targetName(p.peakOverflowTarget)} overflow quota at peaks.` });
    if (r.shortOfHeadroom) out.push({ key: "headroom", short: `is below the ${+p.headroom}× burst headroom`, text: `Below the ${+p.headroom}× burst headroom: the peak needs at least ${r.headroomGpus} chips with this overflow.` });
    return out;
  }

  // ---------------------------------------------------------------- recompute + render
  let rafPending = false;
  function scheduleRecompute() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; recompute({ animate: true }); });
  }

  // Long-prompt rate placeholders: the rate in use (a benchmark's, or the short-prompt rate / the assumed slowdown)
  function classPlaceholders() {
    const shortNow = S.data ? S.data.perGpu * (S.data.mixFactorRef || 1) : null;
    const ctps = (S.data && S.data.classTps) || {};
    [["f-medTps", "medium", 3.0], ["f-lrgTps", "large", 4.0]].forEach(([id, k, pen]) => {
      const el = $(id); if (!el) return;
      const v = ctps[k] || (shortNow ? shortNow / pen : null);
      el.placeholder = v ? String(Math.round(v)) : "";
      el.title = ctps[k] ? "Measured by your benchmark" : "Assumed from the short-prompt rate";
    });
  }

  function recompute(opts) {
    opts = opts || {};
    classPlaceholders();
    if (!S.session) return;
    if (!S.data) { renderNoData(); return; }
    const prof = currentProfile();
    if (prof) S.data.profile = prof;
    renderTrafficError();
    const p = params();
    S.p = p;
    const base = M.scenario(S.data, p);
    S.base = base;
    let r;
    if (S.ui.fleetOverride != null && base.capPerGpu) r = evaluate(p, +S.ui.fleetOverride);
    else r = evaluate(p, null);
    S.r = r;
    renderVerdict();
    renderRegionWarn();
    renderFigures();
    renderProvenance();
    renderWaterline(!!opts.animate);
    renderBucketShare();
    renderPriceSummaries();
    renderTab();
  }

  // Why the chosen model fits no single machine (planner.no_fit_explanation) and what to do, as clickable options.
  function noFitHTML(nf, idPrefix) {
    const opts = (nf.options || []).map((o, i) => {
      const a = o.action || {};
      if (a.type === "precision") return `<li><button type="button" class="link-btn" data-nofit-prec="${esc(a.value)}" data-needs-idle id="${idPrefix}-o${i}">${esc(ap(o.label))}</button></li>`;
      if (a.type === "models") return `<li><button type="button" class="link-btn" data-nofit-models id="${idPrefix}-o${i}">${esc(ap(o.label))}</button></li>`;
      return `<li>${esc(ap(o.label))}</li>`;
    }).join("");
    return `<span class="nofit-head">${icon("i-alert")}<span>${esc(ap(nf.message))}</span></span>${opts ? `<ul class="nofit-opts">${opts}</ul>` : ""}`;
  }
  function wireNoFit(root) {
    root.querySelectorAll("[data-nofit-prec]").forEach((b) => b.onclick = () => { if (S.drawer) closeDrawer(); setPrecision(b.dataset.nofitPrec); });
    root.querySelectorAll("[data-nofit-models]").forEach((b) => b.onclick = () => { if (S.drawer === "hw") closeDrawer(); if (S.drawer !== "model") openDrawer("model", $("chipModel")); });
  }
  function renderNoData() {
    const nf = S.session && S.session.no_fit;
    const v = $("verdict");
    v.classList.toggle("nofit", !!(nf && !S.busy));
    document.body.classList.toggle("is-nofit", !!nf);   // hides the chart and tabs (nothing to show for this model)
    if (nf && !S.busy) {
      v.innerHTML = noFitHTML(nf, "vnf");
      wireNoFit(v);
    } else if (S.session && S.session.plan_error && !S.busy) {
      v.innerHTML = `${esc(ap(S.session.plan_error))} <button type="button" class="link-btn" id="btnEstimateNow">Try again</button>`;
    } else {
      v.innerHTML = S.busy
        ? "Sizing hardware for this workload…"
        : `No estimate yet for this workload. <button type="button" class="link-btn" id="btnEstimateNow">Estimate it now</button>`;
    }
    const b = $("btnEstimateNow");
    if (b) b.onclick = () => reestimate();
    renderRegionWarn();
    $("figures").innerHTML = "";
    $("provLine").textContent = "";
    $("wlSvg").textContent = "";
    $("wlHandle").hidden = true;
    $("wlLegend").innerHTML = ""; S.legendHTML = null;
    renderChips();
    renderBench();
    renderConfig();
  }

  function renderTrafficError() {
    const el = $("trafficError");
    if (S.profileError && !S.cfg.traffic.hourly_csv) { el.textContent = S.profileError; el.hidden = false; }
    else el.hidden = true;
  }

  // The model the user chose (the config), shown at once; selected() is what the current numbers were computed for.
  function chosenModel() {
    const cm = S.cfg && S.cfg.model;
    const sm = selected().model || {};
    if (cm && cm.model_garden_id && cm.model_garden_id !== sm.model_garden_id) return { model_garden_id: cm.model_garden_id, hf_id: cm.hf_id, name: null };
    return sm;
  }
  function renderChips() {
    const sel = selected();
    $("chipModelText").textContent = modelName(chosenModel());
    const hw = sel.hardware || {};
    $("chipHwText").textContent = hw.accelerator_type ? `${hw.accelerator_count || 1} × ${accName(hw.accelerator_type)} per replica`
      : (S.session && S.session.no_fit) ? "None fits" : "Choose";
    renderPrecision();
    $("chipModel").setAttribute("aria-label", `Model: ${modelName(chosenModel())}. Compare models`);
    $("chipHw").setAttribute("aria-label", `Hardware: ${$("chipHwText").textContent}. Compare hardware`);
  }

  // Precision is its own choice (contract: model.precision bf16 | fp8 | int4). Only precisions that exist for the model
  // get a button: INT4 needs a 4-bit checkpoint (published in INT4, or prequantized AWQ / GPTQ in Model Garden), models
  // published in INT4 / FP8 only have no BF16 to quantize from. FP8 is quantized on the fly unless a prequantized
  // FP8 checkpoint exists.
  function prequantFp8(mgid) {
    const info = modelInfo(mgid);
    return (info && info.prequantized && info.prequantized.fp8) || null;
  }
  const INT4_KEYS = ["int4", "w4a16", "awq", "gptq"];
  const PRECS = ["bf16", "fp8", "int4", "mxfp4"];
  const PREC_LABEL = { bf16: "BF16", fp8: "FP8", int4: "INT4", mxfp4: "MXFP4" };
  // MXFP4 (OpenAI gpt-oss): only for models published in MXFP4 (live Hugging Face quantization_config) or with an
  // MXFP4 checkpoint; for a model published in MXFP4 it is the only button unless a BF16 checkpoint really exists.
  function availablePrecisions(info) {
    if (!info) return ["bf16", "fp8"];
    if (info.native_precision === "int4") return ["int4"];
    if (info.native_precision === "mxfp4") {
      const own = Array.isArray(info.precisions) ? info.precisions.filter((p) => p === "bf16" || p === "mxfp4") : [];
      return own.length ? own : ["mxfp4"];
    }
    const base = Array.isArray(info.precisions) ? info.precisions.filter((p) => PRECS.indexOf(p) >= 0) : (info.fp8_only ? ["fp8"] : ["bf16", "fp8"]);
    const pq = info.prequantized || {};
    if (base.indexOf("int4") < 0 && INT4_KEYS.some((k) => pq[k])) base.push("int4");
    if (base.indexOf("mxfp4") < 0 && pq.mxfp4) base.push("mxfp4");
    return base.length ? base : ["bf16", "fp8"];
  }
  const normPrec = (p) => { p = String(p || "bf16").toLowerCase(); return PRECS.indexOf(p) >= 0 ? p : "bf16"; };
  const precLabel = (p) => PREC_LABEL[String(p || "").toLowerCase()] || String(p || "").toUpperCase();
  // Hardware rows at MXFP4: how well this GPU is known to serve it (verified recipe / documented by vLLM / untested).
  function precSupportHTML(c, tag) {
    if (!c || !c.precision_support || c.precision_support === "verified") return "";
    const untested = c.precision_support === "untested";
    const text = c.precision_support_label || (untested ? "MXFP4 untested on this GPU" : "MXFP4 supported by vLLM");
    const cls = tag === "p" ? (untested ? "warn-text" : "") : `sub note${untested ? " warn-text" : ""}`;
    return `<${tag}${cls ? ` class="${cls}"` : ""} title="${esc(c.precision_support_note || "")}">${untested ? icon("i-alert", "ico-sm") : ""}${esc(text)}</${tag}>`;
  }
  function renderPrecision() {
    const sel = selected();
    const prec = normPrec((S.cfg && S.cfg.model && S.cfg.model.precision) || sel.precision);
    const info = modelInfo(chosenModel().model_garden_id);
    const avail = availablePrecisions(info);
    document.querySelectorAll("#precSeg [data-prec]").forEach((b) => {
      const p = b.dataset.prec, on = p === prec;
      b.hidden = avail.indexOf(p) < 0 && !on;   // the current value stays visible (it's what the numbers use)
      b.setAttribute("aria-checked", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
      b.disabled = !!S.busy;
      b.title = avail.indexOf(p) < 0 ? `Not available for this model` : "";
    });
    const hint = $("precHint");
    const pq = prequantFp8(chosenModel().model_garden_id);
    const native = info && info.native_precision;
    if (avail.indexOf(prec) < 0) {
      hint.innerHTML = `${icon("i-alert", "ico-sm")}<span>${precLabel(prec)} isn't available for this model.</span>`;
      hint.title = "";
    } else if (native && native === prec) {
      hint.textContent = `Published in ${precLabel(prec)}.`;
      hint.title = prec === "mxfp4"
        ? "The model's own checkpoint: 4-bit MXFP4 experts with attention and embeddings in BF16, served as released (no --quantization flag). Throughput estimates for it are rough."
        : "The model's own checkpoint: nothing is quantized.";
    } else if (prec === "mxfp4") {
      hint.innerHTML = `${icon("i-alert", "ico-sm")}<span>MXFP4: check quality on your prompts. Uses the MXFP4 checkpoint.</span>`;
      hint.title = (info && info.prequantized && info.prequantized.mxfp4) ? `Checkpoint: ${info.prequantized.mxfp4}` : "";
    } else if (prec === "fp8") {
      hint.innerHTML = `${icon("i-alert", "ico-sm")}<span>FP8: check quality on your prompts. ${pq || (info && info.fp8_only) ? "Uses the prequantized FP8 checkpoint." : "Quantized on the fly by vLLM."}</span>`;
      hint.title = pq ? `Prequantized checkpoint: ${pq}` : "No prequantized FP8 checkpoint is listed for this model, so vLLM quantizes the BF16 weights when it loads them.";
    } else if (prec === "int4") {
      const ck = info && info.prequantized ? INT4_KEYS.map((k) => info.prequantized[k]).find(Boolean) : null;
      hint.innerHTML = `${icon("i-alert", "ico-sm")}<span>4-bit: check quality on your prompts. Uses the prequantized 4-bit checkpoint.</span>`;
      hint.title = ck ? `Prequantized checkpoint: ${ck}` : "";
    } else {
      hint.textContent = "";
      hint.title = "";
    }
  }
  async function setPrecision(prec) {
    if (!S.cfg || S.busy) return;
    const cur = normPrec(S.cfg.model.precision || selected().precision);
    if (cur === prec && !(S.session && S.session.no_fit)) return;
    if (!(await confirmLoseMeasurement())) { renderPrecision(); return; }
    S.cfg.model = Object.assign({}, S.cfg.model, { precision: prec });
    S.ui.fleetOverride = null;
    S.tpsEdit = undefined;
    S.hw = null;
    renderPrecision();
    if (await reestimate({ clearOverride: true, force: hasRealRun() })) toast(`Now planning ${modelName(selected().model)} at ${prec.toUpperCase()}.`);
  }

  function renderVerdict() {
    renderChips();
    const r = S.r, v = $("verdict");
    const sel = selected();
    const hw = sel.hardware || {};
    const model = esc(modelName(sel.model));
    const acc = esc(accName(hw.accelerator_type));
    v.classList.toggle("stale", !!S.busy);
    v.classList.remove("nofit");
    document.body.classList.remove("is-nofit");
    if (!r || r.total == null || !isFinite(r.total)) {
      v.innerHTML = `No fleet works with these inputs. <span class="muted">${esc(ap((r && r.infeasible) || "Check the traffic and the MaaS rate limit."))}</span>`;
      return;
    }
    const b = baselineOf(r);
    const ratio = b ? b.vs : r.total / r.maasOnly;
    const n = `<span class="num">${r.gpus}</span>`;
    const cost = `<span class="num">${fmtUSD(r.total)}</span>`;
    const share = `<span class="num">${pct(ratio)}</span>`;
    const host = `<span class="ent">${model}</span> on ${n} × <span class="ent">${acc}</span>`;
    const issues = policyIssues(r);
    let html;
    if (issues.length && S.base && S.base.gpus) {
      // Outside the fleet policy: show what it would cost, but keep the answer on the policy's optimum.
      const bb = baselineOf(S.base);
      html = `With your fleet of ${n} × <span class="ent">${acc}</span>, self-hosting <span class="ent">${model}</span> would cost ${cost} a month, but it ${esc(issues.map((x) => x.short).join(" and "))}<span class="muted">, so it's outside the fleet policy</span>. The cheapest fleet within the policy is <span class="num">${S.base.gpus}</span> chips at <span class="num">${fmtUSD(S.base.total)}</span> (${pct(bb ? bb.vs : NaN)} of ${baseShort(bb)}).`;
    } else if (b && !b.feasible) {
      // MaaS alone can't serve the peak: never tell the user to stay on it.
      const planned = b.capRps / Math.max(1, +(S.p && S.p.headroom) || 1);
      const lead = `Open-model MaaS can't serve your peak <span class="muted">(limit ${fmtRps(b.capRps)} req/s, planned at ${planned.toFixed(1)} to absorb bursts; peak ${fmtRps(b.peakRps)} req/s)</span>.`;
      const basis = `<span class="num">${fmtUSD(b.total)}</span> for MaaS up to its limit plus ${esc(targetName(b.target))} overflow`;
      if (r.overridden && S.base && S.base.gpus) {
        html = `${lead} With your fleet, self-hosting ${host} costs ${cost} a month vs ${basis}. <span class="muted">The cheapest fleet is ${S.base.gpus} chips at ${fmtUSD(S.base.total)}.</span>`;
      } else if (ratio <= 1) {
        html = `${lead} Self-host ${host}: ${cost} a month vs ${basis}.`;
      } else {
        // Self-hosting loses: recommend the baseline first, then say what self-hosting would cost.
        html = `Stay on open-model MaaS up to its limit with ${esc(targetName(b.target))} overflow: <span class="num">${fmtUSD(b.total)}</span> a month <span class="muted">(MaaS limit ${fmtRps(b.capRps)} req/s, planned at ${(b.capRps / Math.max(1, +(S.p && S.p.headroom) || 1)).toFixed(1)}; peak ${fmtRps(b.peakRps)} req/s)</span>. Self-hosting ${host} would cost ${cost} (${share} of that).`;
      }
    } else if (r.overridden && S.base && S.base.gpus) {
      html = `With your fleet of ${n} × <span class="ent">${acc}</span>, self-hosting <span class="ent">${model}</span> costs ${cost} a month, ${share} of what MaaS would cost. <span class="muted">The cheapest fleet is ${S.base.gpus} chips at ${fmtUSD(S.base.total)} (${pct(vsBase(S.base))}).</span>`;
    } else if (ratio <= 1) {
      html = `Self-host ${host}: ${cost} a month, ${share} of what MaaS would cost.`;
    } else {
      html = `Stay on MaaS for now. Self-hosting ${host} costs ${cost} a month, ${share} of the <span class="num">${fmtUSD(b ? b.total : r.maasOnly)}</span> MaaS bill.`;
    }
    // Default policy (no Gemini at peaks): say what allowing the failover tier at peaks would save, with a button.
    const hint = S.base && S.base.peakGeminiHint;
    if (hint) {
      const text = M.hintText ? M.hintText(hint.name, hint.gpus, hint.saving, chipWord(hw.accelerator_type, hint.gpus))
        : `Allowing ${hint.name} overflow at peaks would cut it to ${hint.gpus} chips and save about ${fmtUSD(hint.saving)} a month.`;
      html += `<span class="verdict-hint">${esc(text)} <button type="button" class="link-btn" id="btnAllowGemini" data-target="${esc(hint.target)}">Allow Gemini at peaks</button></span>`;
    }
    v.innerHTML = html;
    const ag = $("btnAllowGemini");
    if (ag) ag.onclick = () => allowGeminiAtPeaks(ag.dataset.target);
  }
  // Sets hybrid.peak_overflow_target through the rail select, so it saves and recomputes like a rail edit.
  function allowGeminiAtPeaks(target) {
    const sel = $("f-overflow");
    if (!sel || !target) return;
    S.ui.fleetOverride = null;
    sel.value = target;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    toast(`Peak overflow now goes to ${targetName(target)}.`);
  }

  function renderFigures() {
    const r = S.r, p = S.p;
    const box = $("figures");
    if (!r || r.total == null) { box.innerHTML = ""; return; }
    const tp = throughput();
    const t = tagHTML(tp.source, tp.basis, tp.note);
    const hw = selected().hardware;
    const rep = S.data.replica || 1;
    const replicas = Math.round(r.gpus / rep);
    const words = chipWord(hw.accelerator_type, r.gpus);
    const resNote = p.resilience > 1 ? `, includes ${p.resilience}× redundancy` : "";
    const b = baselineOf(r);
    const over = (r.fleet ? r.fleet.maasCost || 0 : 0) + (r.geminiOverflowCost || 0) + (r.longMaasCost || 0);
    const issues = policyIssues(r, p);
    const items = [
      [`${words.charAt(0).toUpperCase() + words.slice(1)} ${t}`, String(r.gpus), `${replicas} ${replicas === 1 ? "replica" : "replicas"} of ${rep}${resNote}`],
      [`Monthly cost ${t}`, fmtUSD(r.total), issues.length ? `What-if outside the fleet policy: ${issues.map((x) => x.short).join(", ")}`
        : over > 0.5 ? `${TIER_LABELS[p.pricing] || p.pricing}, incl. ${fmtUSD(over)} overflow` : (TIER_LABELS[p.pricing] || p.pricing), issues.length > 0],
      b && !b.feasible
        ? ["Share of MaaS + overflow", pct(b.vs), `MaaS to its limit + ${targetName(b.target)}: ${fmtUSD(b.total)} a month`]
        : ["Share of MaaS cost", pct(b ? b.vs : r.total / r.maasOnly), `MaaS only: ${fmtUSD(b ? b.total : r.maasOnly)} a month`],
      ["Average utilization", pct(r.fleet ? r.fleet.util : NaN), "of the fleet you pay for"],
    ];
    box.innerHTML = items.map(([k, v, d, bad]) => `<div${bad ? ' class="fig-policy"' : ""}><dt>${k}</dt><dd><span class="fig-v">${esc(v)}${bad ? icon("i-alert", "ico-sm") : ""}</span><span class="fig-d">${esc(d)}</span></dd></div>`).join("");
    const pw = $("policyWarn");
    if (pw) {
      if (issues.length) {
        const minN = r.minPolicyGpus, optN = S.base && S.base.fleet ? S.base.fleet.gpus : null;
        const tail = minN != null ? ` Fleets below ${minN} chips (shaded on the chart) are outside the policy` + (optN != null ? `; the cheapest within it is ${optN} chips.` : ".") : "";
        pw.innerHTML = icon("i-alert") + `<span>${esc(issues.map((x) => x.text).join(" ") + tail)}</span>`;
        pw.hidden = false;
      } else pw.hidden = true;
    }
  }

  // The chosen accelerator in the session's region: planner.plan sets chosen.region_note when Compute Engine doesn't
  // offer it there (prices then come from the reference catalog), and prices.source says whether the Billing Catalog
  // had SKUs for it. Shown under the headline so a region change never leaves numbers for hardware you can't get.
  function regionStatus() {
    const s = S.session;
    const ch = s && s.plan && s.plan.chosen;
    if (!ch || !ch.accelerator_type) return null;
    const region = (s.config && s.config.region) || (S.cfg && S.cfg.region) || "us-central1";
    return { region, acc: ch.accelerator_type, cnt: ch.accelerator_count || 1, notOffered: !!ch.region_note,
             priceSource: (ch.prices && ch.prices.source) || null, prices: ch.prices || {} };
  }
  function renderRegionWarn() {
    const el = $("regionWarn");
    if (!el) return;
    const st = S.busy ? null : regionStatus();
    const chip = $("chipHw");
    if (chip) chip.classList.toggle("chip-warn", !!(st && st.notOffered));
    if (!st || (!st.notOffered && st.priceSource !== "reference_catalog" && st.priceSource !== "unavailable")) { el.hidden = true; el.innerHTML = ""; return; }
    const name = esc(accName(st.acc)), reg = esc(st.region);
    const acts = `<button type="button" class="link-btn" data-rw="hw">Choose hardware for ${reg}</button> · <button type="button" class="link-btn" data-rw="region">Change region</button>`;
    let msg;
    if (st.notOffered) msg = `<strong>${name} isn't offered in ${reg}.</strong> Compute Engine lists no ${name} there, so this fleet can't be deployed in this region and its prices are reference list prices, not ${reg}'s. ${acts}`;
    else if (st.priceSource === "unavailable") msg = `<strong>No price for ${name} in ${reg}.</strong> The Cloud Billing Catalog has no SKU for it and there is no reference price. ${acts}`;
    else msg = `No Cloud Billing price for ${name} in ${reg}: the costs use reference list prices adjusted for the region. Check them, or set your own under Advanced → Pricing overrides.`;
    el.innerHTML = icon("i-alert") + `<span>${msg}</span>`;
    el.hidden = false;
    el.querySelectorAll("[data-rw]").forEach((b) => b.onclick = () => {
      if (b.dataset.rw === "hw") { if (S.drawer !== "hw") openDrawer("hw", $("chipHw")); }
      else { $("projectBtn").click(); setTimeout(() => $("regionSelect").focus(), 0); }
    });
  }
  // Region picked in the project popover: applies at once (prices and availability are per region), re-plans and
  // re-prices, then says what changed.
  async function applyRegion(region) {
    if (!S.cfg || !region || S.cfg.region === region) return;
    const before = regionStatus();
    const tier = S.p && S.p.pricing;
    S.cfg.region = region;
    S.hw = null;
    markDirty();
    if (!S.session) return;
    if (!(await reestimate())) return;
    const after = regionStatus();
    if (!after) return;
    const name = accName(after.acc);
    if (after.notOffered) { toast(`${name} isn't offered in ${region}: choose hardware for this region or another region.`, true); return; }
    if (before && before.acc !== after.acc) { toast(`In ${region} the plan is now ${after.cnt} × ${name} per replica (${accName(before.acc)} before).`); return; }
    const t = (tier && after.prices[tier] > 0) ? tier : TIER_ORDER.find((k) => after.prices[k] > 0);
    if (!t) return;
    const b = before && before.prices[t], a2 = after.prices[t];
    const usd = (v) => `$${(+v).toFixed(2)}`;
    toast(`Prices now for ${region}: ${name} ${b > 0 && Math.abs(b - a2) > 0.005 ? `${usd(b)} → ` : ""}${usd(a2)} per chip-hour (${TIER_LABELS[t] || t})${after.priceSource === "reference_catalog" ? ", reference list price" : ""}.`);
  }

  function renderProvenance() {
    const tp = throughput();
    const el = $("provLine");
    if (!tp) { el.textContent = ""; return; }
    const v = fmtInt(tp.per_chip);
    let html;
    const benchLink = `<button type="button" class="link-btn" data-goto-tab="bench">Run a benchmark</button>`;
    if (tp.label && !tp.pending) {
      html = esc(ap(tp.label)).replace(/Run a benchmark/, benchLink);
    } else if (tp.source === "measured") html = `Measured ${v} input tok/s per chip in a load test on Agent Platform.`;
    if (tp.source === "measured" && !tp.pending && measuredServing()) html += " " + servingText(measuredServing());
    else if (tp.source === "calibrated") html = `Assumed ${v} input tok/s per chip, calibrated from a profile run.`;
    else if (tp.source === "override") html = `Using your value of ${v} input tok/s per chip. Clear it in Advanced to go back to the estimate.`;
    else if (tp.source !== "measured") html = `Assumed ${v} input tok/s per chip, estimated from specs. <button type="button" class="link-btn" data-goto-tab="bench">Run a benchmark</button> to measure it.`;
    if (tp.source === "estimated") {
      const base = tp.label && !tp.pending ? esc(ap(tp.label)).replace(/Run a benchmark/, benchLink)
        : `Estimated ${v} input tok/s per chip, not measured. ${benchLink} to measure it.`;
      html = `${tagHTML("estimated", tp.basis, tp.note)} ${base}` + (tp.note ? ` <span class="prov-note">${esc(ap(tp.note))}</span>` : "");
    }
    if (needsCaution(tp.source)) html += " " + cautionHTML("Not measured on your workload");
    if (S.p && S.p.tputMult !== 1) html += ` Scaled ${S.p.tputMult}× for the what-if.`;
    const thr = S.r && S.r.throughput;
    if (thr && thr.mixFactor > 1.001 && !S.r.routing && thr.all > 0) {
      const b = S.r.buckets || {}, ps = b.penaltySource || {};
      const measured = ps.medium === "measured" || ps.large === "measured";
      const slow = measured
        ? ` (${["medium", "large"].filter((k) => b[k] && b[k].req > 0).map((k) => `${k === "medium" ? "32K–128K" : "over 128K"} ${b[k].penalty.toFixed(1)}× slower${ps[k] !== "measured" ? ", assumed" : (S.p && S.p[k + "Tps"]) ? ", your rate" : ", measured"}`).join("; ")})`
        : " (assumed 3× and 4× slower; enter measured rates in Advanced or run a benchmark)";
      html += ` Your prompts over 32K cost ${thr.mixFactor.toFixed(2)}× more GPU time per token${slow}, so with every prompt on the fleet that is ${fmtInt(thr.all)} tok/s per chip.`;
    }
    el.innerHTML = html;
    const warn = $("warnLine");
    const r = S.r;
    let w = null;
    if (S.profileError && !S.cfg.traffic.hourly_csv) w = `${S.profileError} The chart still shows the last valid week; adjust the busy window or weekend load under Traffic.`;
    else if (r && r.fallbackNote) w = `${ap(r.fallbackNote)}. Showing the smallest fleet that serves the peak on its own, with overflow above the rate limit priced at ${targetName(S.p.failoverTarget)}.`;
    else if (S.base && S.base.infeasible && !S.base.capPerGpu) w = ap(S.base.infeasible) + ".";
    if (w) { warn.innerHTML = icon("i-alert") + `<span>${esc(w)}</span>`; warn.hidden = false; } else warn.hidden = true;
  }

  function renderBucketShare() {
    if (!S.cfg) return;
    const r = S.cfg.requests;
    const b = M.buckets({
      avgInput: +r.avg_input_tokens || 1, avgOutput: +r.avg_output_tokens || 0,
      pctOver32: +r.pct_over_32k || 0, pctOver128: Math.min(+r.pct_over_128k || 0, +r.pct_over_32k || 0),
      mediumMean: +r.medium_mean || 69000, largeMean: +r.large_mean || 188000,
      systemPrompt: +r.system_prompt_tokens || 0, systemHit: (+r.system_prompt_cache_hit || 0) * 100,
    });
    const bar = $("bucketShare").querySelector(".share-bar");
    bar.children[0].style.width = (b.small.tokenShare * 100).toFixed(1) + "%";
    bar.children[1].style.width = (b.medium.tokenShare * 100).toFixed(1) + "%";
    bar.children[2].style.width = (b.large.tokenShare * 100).toFixed(1) + "%";
    $("bucketText").textContent = (+r.pct_over_32k > 0)
      ? `Tokens: ${pct(b.small.tokenShare)} up to 32K, ${pct(b.medium.tokenShare)} 32K–128K, ${pct(b.large.tokenShare)} over 128K. Prompts over 32K take ${pct(b.longGpuShare)} of fleet time.`
      : "All prompts are 32K tokens or shorter.";
    // The long tail must fit inside the average prompt (config._mix_check): else short prompts would average < 800.
    const o32 = Math.min(100, +r.pct_over_32k || 0) / 100, o128 = Math.min(+r.pct_over_128k || 0, +r.pct_over_32k || 0) / 100;
    const longTok = (o32 - o128) * (+r.medium_mean || 69000) + o128 * (+r.large_mean || 188000);
    const shortMean = ((+r.avg_input_tokens || 0) - longTok) / Math.max(1e-9, 1 - o32);
    if (o32 > 0 && shortMean < 800) {
      const w = document.createElement("span");
      w.className = "warn-text";
      w.textContent = ` Prompts over 32K alone average ${fmtInt(longTok)} tokens per request, more than fits in the average prompt size (${fmtInt(+r.avg_input_tokens || 0)}): raise the average (it includes long prompts) or lower the long-prompt share.`;
      $("bucketText").appendChild(w);
    }
  }

  // ---------------------------------------------------------------- waterline
  function waterlineModel() {
    const r = S.r, p = S.p;
    const D = r.demand;
    const share = r.routing ? r.buckets.small.tokenShare : 1;
    const Ds = D.map((d) => d * share);
    const C = r.fleet ? r.fleet.gpus * r.capPerGpu * U : 0;
    // Smallest fleet within the policy (capacity), for the shaded "outside the policy" band
    const minN = r.minPolicyGpus != null ? r.minPolicyGpus : (S.base && S.base.minPolicyGpus);
    const Cmin = minN != null && minN > (S.data.replica || 1) ? minN * r.capPerGpu * U : 0;
    return { D: D.slice(), Ds, C, Mc: (r.maasCapTps || 0) / p.headroom, Cmin };
  }
  function lerpModel(a, b, t) {
    const L = (x, y) => x + (y - x) * t;
    return { D: b.D.map((v, i) => L(a.D[i], v)), Ds: b.Ds.map((v, i) => L(a.Ds[i], v)), C: L(a.C, b.C), Mc: L(a.Mc, b.Mc), Cmin: b.Cmin };
  }
  function renderWaterline(animate) {
    if (S.shape.on) { drawShape(); return; }
    if (!S.r || !S.r.demand || !S.r.capPerGpu) {
      $("wlSvg").textContent = ""; $("wlHandle").hidden = true; $("wlHit").hidden = true; $("wlLegend").innerHTML = ""; S.legendHTML = null;
      return;
    }
    $("wlHandle").hidden = false; $("wlHit").hidden = false;
    const target = waterlineModel();
    const from = S.drawn;
    cancelAnimationFrame(S.anim);
    const canAnimate = animate && from && !reduceMotion.matches && from.D.length === target.D.length && !S.drag;
    if (!canAnimate) { drawWaterline(target); }
    else {
      const t0 = performance.now(), dur = 150;
      const step = (now) => {
        const k = Math.min(1, (now - t0) / dur);
        const e = 1 - Math.pow(1 - k, 3);
        drawWaterline(lerpModel(from, target, e));
        if (k < 1) S.anim = requestAnimationFrame(step);
      };
      S.anim = requestAnimationFrame(step);
    }
    S.target = target;
    renderLegend(target);
    renderWeekTable(target);
    updateHandleAria();
    $("btnResetFleet").hidden = !(S.ui.fleetOverride != null && S.r.overridden);
    if (!$("btnResetFleet").hidden && S.base && S.base.gpus) $("btnResetFleet").textContent = `Reset to optimal (${S.base.gpus} chips)`;
  }

  function geom() {
    const plot = $("wlPlot");
    const W = plot.clientWidth, H = plot.clientHeight;
    const narrow = W < 520;
    return { W, H, L: narrow ? 40 : 52, R: 8, T: 12, B: 24 };
  }

  function drawWaterline(m) {
    S.drawn = m;
    const g = geom();
    const { W, H, L, R, T, B } = g;
    const s = $("wlSvg");
    s.setAttribute("viewBox", `0 0 ${W} ${H}`);
    s.textContent = "";
    if (!W || !H) return;
    const n = m.D.length;
    const ymax = S.drag ? S.drag.ymax : niceMax(Math.max(Math.max.apply(null, m.D), m.C) * 1.12);
    g.ymax = ymax;
    S.geom = g;
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (v) => T + (1 - Math.min(v, ymax) / ymax) * (H - T - B);
    const p = S.p;
    // grid + axes
    const grid = svg("g", { class: "grid" }, s);
    const ticks = 4;
    for (let k = 0; k <= ticks; k++) {
      const v = (ymax * k) / ticks;
      svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }, grid);
      svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, s, k === ticks ? fmtTok(v) + (W > 520 ? " tok/s" : "") : fmtTok(v));
    }
    const days = W < 420 ? ["M", "T", "W", "T", "F", "S", "S"] : ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    for (let d = 0; d < 7; d++) {
      if (d) svg("line", { class: "day-sep", x1: x(d * 24), x2: x(d * 24), y1: T, y2: H - B }, s);
      svg("text", { x: x(d * 24 + 11.5), y: H - 6, "text-anchor": "middle" }, s, days[d]);
    }
    // Fleets below the smallest one within the fleet policy: amber hatched band (stronger while the fleet is there)
    const bad = policyIssues(S.r).length > 0;
    if (m.Cmin > 0 && m.Cmin < ymax) {
      svg("rect", { class: "policy-band" + (bad ? " on" : ""), x: L, y: y(m.Cmin), width: Math.max(0, W - L - R), height: Math.max(0, y(0) - y(m.Cmin)) }, s);
      svg("line", { class: "policy-min" + (bad ? " on" : ""), x1: L, x2: W - R, y1: y(m.Cmin), y2: y(m.Cmin) }, s);
    }
    // layers
    const served = m.Ds.map((d) => Math.min(d, m.C));
    const spillM = m.Ds.map((d, i) => Math.min(d - served[i], m.Mc));
    const spillG = m.Ds.map((d, i) => Math.max(0, d - served[i] - m.Mc));
    const area = (lo, hi, style) => {
      let top = "", bot = "";
      for (let i = 0; i < n; i++) top += (i ? "L" : "M") + x(i).toFixed(1) + "," + y(hi(i)).toFixed(1);
      for (let i = n - 1; i >= 0; i--) bot += "L" + x(i).toFixed(1) + "," + y(lo(i)).toFixed(1);
      return svg("path", { d: top + bot + "Z", style }, s);
    };
    const hatch = document.documentElement.dataset.theme === "dark" ? "url(#hatch-dark)" : (document.documentElement.dataset.theme === "light" ? "url(#hatch-light)" : (matchMedia("(prefers-color-scheme: dark)").matches ? "url(#hatch-dark)" : "url(#hatch-light)"));
    area((i) => served[i], () => m.C, `fill:${hatch}`);
    area(() => 0, (i) => served[i], "fill:var(--s-fleet);fill-opacity:0.62");
    // Peak overflow fills open-model MaaS within its limit first; beyond it goes to Gemini (the peak-overflow tier,
    // or the failover tier as a what-if when Gemini isn't allowed at peaks).
    area((i) => served[i], (i) => served[i] + spillM[i], "fill:var(--s-maas);fill-opacity:0.78");
    if (spillG.some((v) => v > 0)) area((i) => served[i] + spillM[i], (i) => m.Ds[i], "fill:var(--s-gem);fill-opacity:0.9");
    if (S.r.routing) area((i) => m.Ds[i], (i) => m.D[i], `fill:${targetColor(p.routeLongTarget)};fill-opacity:0.45`);
    // MaaS limit guide (only when spill happens)
    if (spillM.some((v) => v > 0) && m.C + m.Mc < ymax) {
      svg("line", { class: "maas-limit", x1: L, x2: W - R, y1: y(m.C + m.Mc), y2: y(m.C + m.Mc) }, s);
      svg("text", { x: L + 6, y: y(m.C + m.Mc) - 5 }, s, "MaaS rate limit");
    }
    // Label at the right end, above the policy-minimum line; skipped when the fleet handle sits right there.
    if (m.Cmin > 0 && m.Cmin < ymax && bad && Math.abs(y(m.Cmin) - y(S.drag ? S.drag.lineC : m.C)) > 30) {
      svg("text", { class: "policy-label", x: W - R - 6, y: y(m.Cmin) - 5, "text-anchor": "end" }, s, "Below: outside the fleet policy");
    }
    // demand curve
    let dpath = "";
    for (let i = 0; i < n; i++) dpath += (i ? "L" : "M") + x(i).toFixed(1) + "," + y(m.D[i]).toFixed(1);
    svg("path", { d: dpath, class: "curve" }, s);
    // waterline
    const lineC = S.drag ? S.drag.lineC : m.C;
    svg("line", { class: "waterline", x1: L, x2: W - R, y1: y(lineC), y2: y(lineC) }, s);
    // crosshair layer
    S.cross = svg("line", { class: "cross", x1: 0, x2: 0, y1: T, y2: H - B, visibility: "hidden" }, s);
    // handle position
    const yy = y(lineC);
    $("wlHandle").style.top = yy + "px";
    $("wlHit").style.top = yy + "px";
    $("wlHit").style.left = L + "px";
  }

  function renderLegend(m) {
    const p = S.p, r = S.r;
    // Stable legend: the same items in the same order whatever the fleet size, so it never re-wraps while the
    // waterline is dragged. Layers that are empty at this fleet size are dimmed, not removed.
    const served = m.Ds.map((d) => Math.min(d, m.C));
    const hasSpill = m.Ds.some((d, i) => d - served[i] > 1);
    const hasOver = m.Ds.some((d, i) => d - served[i] - m.Mc > 1);
    const item = (sw, label, on, extra) => `<li${on === false ? ' class="off"' : ""}>${sw}<span>${label}${on === false ? '<span class="visually-hidden"> (none at this fleet size)</span>' : ""}</span>${extra || ""}</li>`;
    const items = [
      item(`<span class="sw" style="background:var(--s-fleet);opacity:.62"></span>`, "Served on the fleet"),
      item(`<span class="sw sw-idle"></span>`, "Idle capacity you pay for"),
      item(`<span class="sw" style="background:var(--s-maas);opacity:.78"></span>`, "Overflow to open-model MaaS", hasSpill),
    ];
    const gemAllowed = r.geminiAllowedAtPeaks != null ? r.geminiAllowedAtPeaks : p.peakOverflowTarget !== "open_maas";
    const spillT = r.spillGeminiTarget || (gemAllowed ? p.peakOverflowTarget : p.failoverTarget);
    items.push(item(`<span class="sw" style="background:var(--s-gem)"></span>`, `Over the MaaS limit, to ${esc(targetName(spillT))}${gemAllowed ? "" : " (what-if)"}`, hasOver));
    if (m.Cmin > 0) items.push(item(`<span class="sw sw-policy"></span>`, "Outside the fleet policy", policyIssues(r).length > 0));
    if (r.routing) items.push(item(`<span class="sw" style="background:${targetColor(p.routeLongTarget)};opacity:.45"></span>`, `Prompts over 32K, to ${esc(targetName(p.routeLongTarget))}`));
    items.push(item(`<span class="sw sw-curve"></span>`, "Demand"));
    items.push(item(`<span class="sw sw-line"></span>`, "Fleet capacity (drag to resize)"));
    const html = items.join("");
    if (S.legendHTML !== html) { $("wlLegend").innerHTML = html; S.legendHTML = html; }
  }

  function renderWeekTable(m) {
    const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    let rows = "";
    for (let d = 0; d < 7; d++) {
      const D = m.D.slice(d * 24, d * 24 + 24), Ds = m.Ds.slice(d * 24, d * 24 + 24);
      const served = Ds.map((v) => Math.min(v, m.C));
      const tot = D.reduce((a, b) => a + b, 0), sv = served.reduce((a, b) => a + b, 0);
      rows += `<tr><td>${days[d]}</td><td class="n">${fmtTok(Math.max.apply(null, D))}</td><td class="n">${fmtTok(tot / 24)}</td><td class="n">${pct(sv / Math.max(1, tot))}</td></tr>`;
    }
    $("wlTable").innerHTML = `<caption>Input tokens per second. Fleet capacity ${fmtTok(m.C)} tok/s.</caption><thead><tr><th>Day</th><th class="n">Peak hour</th><th class="n">Average</th><th class="n">Served on fleet</th></tr></thead><tbody>${rows}</tbody>`;
    const r = S.r;
    $("wlSummary").textContent = `Weekly demand peaks at ${fmtTok(Math.max.apply(null, m.D))} tokens per second. The fleet of ${r.fleet ? r.fleet.gpus : 0} chips serves up to ${fmtTok(m.C)} tokens per second; the rest spills to pay-per-token APIs.`;
  }

  function fleetBounds() {
    const rep = S.data.replica || 1;
    const base = S.base;
    const opt = base && base.fleet ? base.fleet.gpus : (S.r && S.r.fleet ? S.r.fleet.gpus : rep);
    const peakN = S.r && S.r.capPerGpu ? Math.ceil(Math.max.apply(null, S.r.demand) / (S.r.capPerGpu * U) / rep) * rep : opt;
    const max = Math.max(opt * 3, Math.ceil((peakN * 1.5) / rep) * rep, rep * 4);
    return { min: rep, max, opt, rep };
  }
  function updateHandleAria() {
    const h = $("wlHandle"), r = S.r;
    if (!r || !r.fleet) return;
    const b = fleetBounds();
    const n = r.fleet.gpus;
    const hw = selected().hardware;
    h.setAttribute("aria-valuemin", b.min);
    h.setAttribute("aria-valuemax", b.max);
    h.setAttribute("aria-valuenow", n);
    const bl = baselineOf(r);
    const issues = policyIssues(r);
    h.setAttribute("aria-valuetext", `${n} ${chipWord(hw.accelerator_type, n)} serving, ${fmtUSD(r.total)} a month, ${pct(bl ? bl.vs : r.total / r.maasOnly)} of ${baseShort(bl)}` + (issues.length ? `. Outside the fleet policy: ${issues.map((x) => x.short).join(", ")}` : ""));
    $("wlHandleText").textContent = `${n} × ${accName(hw.accelerator_type)}`;
    h.classList.toggle("policy-bad", issues.length > 0);
    h.title = issues.length ? issues.map((x) => x.text).join(" ") : "";
    $("wlHandleWarn").hidden = !issues.length;
  }
  // Keep the chart still while the fleet changes: text above it (the verdict) may re-wrap, so compensate the scroll.
  function keepPlotStill(fn) {
    const plot = $("wlPlot"), top0 = plot.getBoundingClientRect().top;
    fn();
    const d = plot.getBoundingClientRect().top - top0;
    if (Math.abs(d) > 0.5) window.scrollBy(0, d);
  }
  function setFleet(n, animate) {
    const b = fleetBounds();
    n = Math.max(b.min, Math.min(b.max, Math.round(n / b.rep) * b.rep));
    S.ui.fleetOverride = n === b.opt ? null : n;
    markDirty(true);
    keepPlotStill(() => recompute({ animate }));
  }

  function initWaterlineInteraction() {
    const plot = $("wlPlot"), handle = $("wlHandle"), hit = $("wlHit");
    const yToCap = (clientY) => {
      const g = S.geom; const rect = plot.getBoundingClientRect();
      const yy = Math.max(g.T, Math.min(g.H - g.B, clientY - rect.top));
      return { cap: (1 - (yy - g.T) / (g.H - g.T - g.B)) * g.ymax, yy };
    };
    const start = (e) => {
      if (!S.r || !S.r.capPerGpu || e.button > 0) return;
      e.preventDefault();
      hideTip();
      const target = e.currentTarget;
      target.setPointerCapture(e.pointerId);
      S.drag = { ymax: S.geom.ymax, lineC: S.drawn ? S.drawn.C : 0, id: e.pointerId, el: target };
      handle.classList.add("dragging");
      handle.focus({ preventScroll: true });
    };
    const move = (e) => {
      if (!S.drag || e.pointerId !== S.drag.id) return;
      const { cap } = yToCap(e.clientY);
      S.drag.lineC = cap;
      const b = fleetBounds();
      const n = Math.max(b.min, Math.min(b.max, Math.max(b.rep, Math.round(cap / (S.r.capPerGpu * U) / b.rep) * b.rep)));
      const cur = S.r.fleet ? S.r.fleet.gpus : null;
      if (n !== cur) {
        S.ui.fleetOverride = n === b.opt ? null : n;
        markDirty(true);
        keepPlotStill(() => recompute({ animate: false }));
      } else if (S.drawn) {
        drawWaterline(S.drawn);
      }
    };
    const end = (e) => {
      if (!S.drag || e.pointerId !== S.drag.id) return;
      const ymax = S.drag.ymax;
      const prev = Object.assign({}, S.drawn, { C: S.drag.lineC });
      S.drag = null;
      handle.classList.remove("dragging");
      S.drawn = prev;
      void ymax;
      keepPlotStill(() => recompute({ animate: true })); // snaps to whole replicas
    };
    [handle, hit].forEach((el) => {
      el.addEventListener("pointerdown", start);
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", end);
      el.addEventListener("pointercancel", end);
    });
    handle.addEventListener("keydown", (e) => {
      if (!S.r || !S.r.fleet) return;
      const b = fleetBounds();
      const n = S.r.fleet.gpus;
      let next = null;
      if (e.key === "ArrowUp" || e.key === "ArrowRight") next = n + b.rep;
      else if (e.key === "ArrowDown" || e.key === "ArrowLeft") next = n - b.rep;
      else if (e.key === "PageUp") next = n + 5 * b.rep;
      else if (e.key === "PageDown") next = n - 5 * b.rep;
      else if (e.key === "Home") next = b.min;
      else if (e.key === "End") next = b.max;
      else if (e.key === "Escape" || e.key === "r" || e.key === "R") next = b.opt;
      if (next == null) return;
      e.preventDefault();
      setFleet(next, true);
    });
    $("btnResetFleet").addEventListener("click", () => { S.ui.fleetOverride = null; markDirty(true); keepPlotStill(() => recompute({ animate: true })); $("wlHandle").focus({ preventScroll: true }); });

    // hover tooltip
    const svgEl = $("wlSvg");
    svgEl.addEventListener("pointermove", (e) => {
      if (S.drag || !S.drawn || !S.geom) return;
      const g = S.geom, rect = svgEl.getBoundingClientRect();
      const px = e.clientX - rect.left;
      if (px < g.L || px > g.W - g.R) { hideTip(); if (S.cross) S.cross.setAttribute("visibility", "hidden"); return; }
      const m = S.drawn, n = m.D.length;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - g.L) / (g.W - g.L - g.R)) * (n - 1))));
      const xx = g.L + (i / (n - 1)) * (g.W - g.L - g.R);
      if (S.cross) { S.cross.setAttribute("x1", xx); S.cross.setAttribute("x2", xx); S.cross.setAttribute("visibility", "visible"); }
      const day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][Math.floor(i / 24)];
      const served = Math.min(m.Ds[i], m.C), spill = Math.min(m.Ds[i] - served, m.Mc), over = Math.max(0, m.Ds[i] - served - m.Mc);
      const rows = [["var(--curve)", fmtTok(m.D[i]), "demand, tok/s"], ["var(--s-fleet)", fmtTok(served), "on the fleet"]];
      if (spill > 0.5) rows.push(["var(--s-maas)", fmtTok(spill), "to open-model MaaS"]);
      if (over > 0.5) rows.push(["var(--s-gem)", fmtTok(over), "to " + targetName(S.r.spillGeminiTarget || S.p.failoverTarget) + (S.r.geminiAllowedAtPeaks ? "" : " (not allowed at peaks)")]);
      if (S.r.routing) rows.push([targetColor(S.p.routeLongTarget), fmtTok(m.D[i] - m.Ds[i]), "over 32K, to " + TARGETS[S.p.routeLongTarget]]);
      rows.push(["", fmtTok(Math.max(0, m.C - served)), "idle capacity"]);
      showTip(e, `${day} ${String(i % 24).padStart(2, "0")}:00 UTC`, rows);
    });
    svgEl.addEventListener("pointerleave", () => { hideTip(); if (S.cross) S.cross.setAttribute("visibility", "hidden"); });

    if (window.ResizeObserver) {
      let w = 0;
      new ResizeObserver(() => {
        if (plot.clientWidth !== w) { w = plot.clientWidth; if (S.shape.on) drawShape(); else if (S.drawn && S.r) drawWaterline(S.target || S.drawn); }
      }).observe(plot);
    }
  }

  // ---------------------------------------------------------------- edit the week on the chart (traffic.custom_hourly)
  const DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const hourLabel = (i) => `${DAY_NAMES[Math.floor(i / 24)]} ${String(i % 24).padStart(2, "0")}:00`;
  function enterShapeEdit() {
    if (!S.cfg || !S.session) return;
    const prof = currentProfile();
    if (!prof || prof.length !== 168) { toast("Draw the week once the traffic is valid: fix the peak, average and quietest hour first.", true); return; }
    S.shape.on = true;
    S.shape.work = prof.map((v) => +v || 0);
    S.shape.undo = [];
    S.shape.cursor = Math.max(0, S.shape.work.indexOf(Math.max.apply(null, S.shape.work)));
    S.shape.ymax = null;
    hideTip();
    $("wlFigure").classList.add("editing");
    $("shapeBar").hidden = false;
    $("wlShape").hidden = false;
    $("wlHandle").hidden = true; $("wlHit").hidden = true; $("btnResetFleet").hidden = true;
    $("btnEditShape").setAttribute("aria-pressed", "true");
    $("btnEditShape").textContent = "Editing shape";
    $("wlCaption").textContent = "Draw your week of input traffic (tok/s, before the growth what-ifs)";
    syncShapeBar();
    drawShape();
    $("wlShape").focus({ preventScroll: true });
  }
  function exitShapeEdit() {
    if (!S.shape.on) return;
    S.shape.on = false;
    S.shape.stroke = null;
    $("wlFigure").classList.remove("editing");
    $("shapeBar").hidden = true;
    $("wlShape").hidden = true;
    $("btnEditShape").setAttribute("aria-pressed", "false");
    $("btnEditShape").textContent = "Edit shape";
    $("wlCaption").textContent = "A typical week of input traffic against the fleet";
    S.drawn = null;
    recompute({ animate: false });
    $("btnEditShape").focus({ preventScroll: true });
  }
  function shapeToSimple() {
    if (!S.cfg) return;
    S.cfg.traffic.custom_hourly = null;
    if (S.shape.on) { S.shape.work = (currentProfile() || S.shape.work).slice(); S.shape.undo = []; syncShapeBar(); }
    populateRail(); markDirty(); recompute({ animate: true });
    toast("Back to the simple shape: peak, average and quietest hour.");
  }
  function commitShape() {
    const t = S.cfg.traffic;
    if (t.hourly_csv) t.hourly_csv = null; // a drawn week replaces the uploaded curve (precedence: CSV > drawn > preset > simple)
    t.preset = null;
    t.custom_hourly = S.shape.work.map((v) => Math.max(0, Math.round(v)));
    if (!monthlyAnchor(t)) t.average_input_tps = Math.round(meanOf(t.custom_hourly));
    markDirty();
    populateRail();
    scheduleRecompute();
  }
  function syncShapeBar() {
    document.querySelectorAll("[data-shape-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.shapeMode === S.shape.mode)));
    document.querySelectorAll("[data-shape-brush]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.shapeBrush === S.shape.brush)));
    $("btnShapeUndo").disabled = !S.shape.undo.length;
    $("btnShapeSimple").disabled = !isCustom(S.cfg.traffic);
    const t = S.cfg.traffic;
    $("shapeHint").innerHTML = (S.shape.mode === "repeat"
      ? "Drag across the chart to paint the hourly traffic. A stroke on any weekday changes all five; a stroke on Saturday or Sunday changes both."
      : "Drag across the chart to paint the hourly traffic. Each stroke changes only the day you paint.")
      + (monthlyAnchor(t) ? ` The monthly total stays at ${fmtTok(+t.monthly_input_tokens)} tokens, so you set the shape and the scale follows.` : "")
      + '<span class="kbd-hint"> Keyboard: focus the chart, choose the hour with the left and right arrows, change it with up and down.</span>';
  }
  // Indices a stroke at week-hour i touches, with brush weights.
  function brushTargets(i) {
    const out = [];
    const rad = S.shape.brush === "smooth" ? 3 : 0, sigma = 1.4;
    const day = Math.floor(i / 24), h = i % 24;
    const days = S.shape.mode === "repeat" ? (day < 5 ? [0, 1, 2, 3, 4] : [5, 6]) : [day];
    for (let k = -rad; k <= rad; k++) {
      const w = rad ? Math.exp(-(k * k) / (2 * sigma * sigma)) : 1;
      for (const d of days) {
        let hh = h + k, dd = d;
        if (S.shape.mode === "repeat") hh = (hh + 24) % 24; // the daily curve wraps around midnight
        else { const j = (d * 24 + hh + 168) % 168; dd = Math.floor(j / 24); hh = j % 24; }
        out.push([dd * 24 + hh, w]);
      }
    }
    return out;
  }
  function paintAt(i, v) {
    const w = S.shape.work;
    for (const [j, wt] of brushTargets(i)) w[j] = Math.max(0, w[j] + wt * (v - w[j]));
  }
  function pushUndo() {
    S.shape.undo.push(S.shape.work.slice());
    if (S.shape.undo.length > 100) S.shape.undo.shift();
  }
  function shapeUndo() {
    if (!S.shape.undo.length) return;
    S.shape.work = S.shape.undo.pop();
    commitShape();
    syncShapeBar();
    drawShape();
  }
  function shapeYmax() {
    if (S.shape.stroke && S.shape.ymax) return S.shape.ymax;
    const peak = Math.max.apply(null, S.shape.work.concat([1]));
    S.shape.ymax = niceMax(peak * 1.3);
    return S.shape.ymax;
  }
  function drawShape() {
    const g = geom();
    const { W, H, L, R, T, B } = g;
    const s = $("wlSvg");
    s.setAttribute("viewBox", `0 0 ${W} ${H}`);
    s.textContent = "";
    if (!W || !H) return;
    const w = S.shape.work, n = w.length;
    const ymax = shapeYmax();
    g.ymax = ymax;
    S.geom = g;
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (v) => T + (1 - Math.min(v, ymax) / ymax) * (H - T - B);
    svg("rect", { class: "weekend", x: x(120), y: T, width: x(167) - x(120), height: H - T - B }, s);
    const grid = svg("g", { class: "grid" }, s);
    for (let k = 0; k <= 4; k++) {
      const v = (ymax * k) / 4;
      svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }, grid);
      svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, s, k === 4 ? fmtTok(v) + (W > 520 ? " tok/s" : "") : fmtTok(v));
    }
    const days = W < 420 ? ["M", "T", "W", "T", "F", "S", "S"] : ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    for (let d = 0; d < 7; d++) {
      if (d) svg("line", { class: "day-sep", x1: x(d * 24), x2: x(d * 24), y1: T, y2: H - B }, s);
      svg("text", { x: x(d * 24 + 11.5), y: H - 6, "text-anchor": "middle" }, s, days[d]);
    }
    let top = "";
    for (let i = 0; i < n; i++) top += (i ? "L" : "M") + x(i).toFixed(1) + "," + y(w[i]).toFixed(1);
    svg("path", { d: top + `L${x(n - 1).toFixed(1)},${y(0).toFixed(1)}L${x(0).toFixed(1)},${y(0).toFixed(1)}Z`, class: "shape-area" }, s);
    svg("path", { d: top, class: "curve shape-curve" }, s);
    // cursor: the hour the keyboard edits, plus the hours a stroke there would also change
    const c = S.shape.cursor;
    const linked = new Set(brushTargets(c).filter(([, wt]) => wt >= 0.999).map(([j]) => j));
    linked.forEach((j) => { if (j !== c) svg("circle", { cx: x(j), cy: y(w[j]), r: 3, class: "shape-linked" }, s); });
    svg("line", { x1: x(c), x2: x(c), y1: T, y2: H - B, class: "shape-cursor" }, s);
    svg("circle", { cx: x(c), cy: y(w[c]), r: 5, class: "shape-dot" }, s);
    const lx = Math.min(W - R - 4, Math.max(L + 4, x(c) + 8));
    svg("text", { x: lx, y: Math.max(T + 12, y(w[c]) - 10), class: "shape-label", "text-anchor": x(c) > W - 160 ? "end" : "start" }, s, `${hourLabel(c)} · ${fmtTok(w[c])} tok/s`);
    const el = $("wlShape");
    el.setAttribute("aria-valuemax", Math.round(ymax));
    el.setAttribute("aria-valuenow", Math.round(w[c]));
    el.setAttribute("aria-valuetext", `${hourLabel(c)} UTC, ${fmtTok(w[c])} tokens per second${S.shape.mode === "repeat" ? (c < 120 ? ", applies to every weekday" : ", applies to both weekend days") : ""}`);
    $("wlSummary").textContent = `Drawing the week. Peak ${fmtTok(Math.max.apply(null, w))}, average ${fmtTok(meanOf(w))}, quietest hour ${fmtTok(Math.min.apply(null, w))} tokens per second.`;
  }
  function initShapeEditor() {
    const el = $("wlShape");
    const at = (e) => {
      const g = S.geom, rect = el.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const i = Math.max(0, Math.min(167, Math.round(((px - g.L) / (g.W - g.L - g.R)) * 167)));
      const yy = Math.max(g.T, Math.min(g.H - g.B, py));
      return { i, v: Math.max(0, (1 - (yy - g.T) / (g.H - g.T - g.B)) * g.ymax) };
    };
    el.addEventListener("pointerdown", (e) => {
      if (!S.shape.on || e.button > 0) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      el.focus({ preventScroll: true });
      pushUndo();
      shapeYmax();
      const q = at(e);
      S.shape.stroke = { id: e.pointerId, last: q };
      S.shape.cursor = q.i;
      paintAt(q.i, q.v);
      drawShape();
    });
    el.addEventListener("pointermove", (e) => {
      if (!S.shape.on) return;
      const q = at(e);
      const st = S.shape.stroke;
      if (!st || st.id !== e.pointerId) { if (e.pointerType === "mouse") { S.shape.cursor = q.i; drawShape(); } return; }
      // Fill the hours between pointer events so fast strokes leave no gaps.
      const a = st.last, steps = Math.abs(q.i - a.i);
      for (let k = 1; k <= steps; k++) {
        const t = k / steps;
        paintAt(Math.round(a.i + (q.i - a.i) * t), a.v + (q.v - a.v) * t);
      }
      if (!steps) paintAt(q.i, q.v);
      st.last = q;
      S.shape.cursor = q.i;
      drawShape();
    });
    const end = (e) => {
      const st = S.shape.stroke;
      if (!st || st.id !== e.pointerId) return;
      S.shape.stroke = null;
      commitShape();
      if (monthlyAnchor(S.cfg.traffic)) { const prof = currentProfile(); if (prof) S.shape.work = prof.slice(); }
      S.shape.ymax = null;
      syncShapeBar();
      drawShape();
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("keydown", (e) => {
      if (!S.shape.on) return;
      const w = S.shape.work;
      let c = S.shape.cursor, changed = false;
      const stepV = Math.max(1, Math.max.apply(null, w) * (e.shiftKey ? 0.01 : 0.05));
      if (e.key === "ArrowRight") c = Math.min(167, c + (e.shiftKey ? 6 : 1));
      else if (e.key === "ArrowLeft") c = Math.max(0, c - (e.shiftKey ? 6 : 1));
      else if (e.key === "PageDown") c = Math.min(167, c + 24);
      else if (e.key === "PageUp") c = Math.max(0, c - 24);
      else if (e.key === "Home") c = 0;
      else if (e.key === "End") c = 167;
      else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        pushUndo();
        paintAt(c, Math.max(0, w[c] + (e.key === "ArrowUp" ? stepV : -stepV)));
        changed = true;
      } else if ((e.key === "z" || e.key === "Z") && (e.ctrlKey || e.metaKey)) { e.preventDefault(); shapeUndo(); return; }
      else if (e.key === "Escape") { e.preventDefault(); exitShapeEdit(); return; }
      else return;
      e.preventDefault();
      S.shape.cursor = c;
      if (changed) { S.shape.ymax = null; commitShape(); syncShapeBar(); }
      drawShape();
    });
    el.addEventListener("pointerleave", () => { if (!S.shape.stroke) drawShape(); });
    $("btnEditShape").addEventListener("click", () => (S.shape.on ? exitShapeEdit() : enterShapeEdit()));
    $("btnShapeDone").addEventListener("click", exitShapeEdit);
    $("btnShapeUndo").addEventListener("click", shapeUndo);
    $("btnShapeSimple").addEventListener("click", shapeToSimple);
    document.querySelectorAll("[data-shape-mode]").forEach((b) => b.addEventListener("click", () => { S.shape.mode = b.dataset.shapeMode; syncShapeBar(); drawShape(); }));
    document.querySelectorAll("[data-shape-brush]").forEach((b) => b.addEventListener("click", () => { S.shape.brush = b.dataset.shapeBrush; syncShapeBar(); drawShape(); }));
  }

  // ---------------------------------------------------------------- tabs
  const TABS = ["costs", "resilience", "long", "sim", "bench", "config"];
  function selectTab(id, focus) {
    S.tab = id;
    for (const t of TABS) {
      const b = $("tab-" + t), pnl = $("panel-" + t);
      const on = t === id;
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
      pnl.hidden = !on;
    }
    if (focus) $("tab-" + id).focus();
    renderTab();
  }
  function renderTab() {
    try {
      if (S.tab === "bench") return renderBench();
      if (S.tab === "sim") return renderSim();
      if (S.tab === "config") return renderConfig();
      if (!S.data || !S.r || S.r.total == null) return;
      if (S.tab === "costs") renderCosts();
      else if (S.tab === "resilience") renderResilience();
      else if (S.tab === "long") renderLong();
    } catch (e) {
      console.warn("Render failed", e);
    }
  }
  function evalWith(over) {
    const p = Object.assign({}, S.p, over);
    return evaluate(p, S.ui.fleetOverride != null ? +S.ui.fleetOverride : null);
  }

  function renderCosts() {
    const r = S.r, p = S.p;
    const tiers = Object.keys(S.data.prices).sort((a, b) => TIER_ORDER.indexOf(a) - TIER_ORDER.indexOf(b));
    const opts = tiers.map((k) => ({ k, s: evalWith({ pricing: k }) }));
    const bl = baselineOf(r) || { total: r.maasOnly, feasible: true };
    const refName = bl.feasible ? "MaaS only" : "MaaS + overflow";
    const max = niceMax(Math.max.apply(null, opts.map((o) => o.s.total || 0).concat([bl.total])) * 1.04);
    const ot = p.peakOverflowTarget, rt = p.routeLongTarget, ft = p.failoverTarget;
    const gemOver = (s) => (s.fleet && s.fleet.geminiCost) || 0;
    const legend = [`<li><span class="sw fill-fleet"></span>Fleet</li>`, `<li><span class="sw ${targetClass(ot)}"></span>Overflow to ${esc(targetName(ot))}</li>`];
    if (opts.some((o) => gemOver(o.s) > 0.5)) legend.push(`<li><span class="sw fill-gem"></span>Over the MaaS limit, to ${esc(targetName(ft))}</li>`);
    if (r.routing) legend.push(`<li><span class="sw ${targetClass(rt)}" style="opacity:.55"></span>Over 32K to ${esc(targetName(rt))}</li>`);
    $("costLegend").innerHTML = legend.join("");
    const box = $("costBars");
    box.classList.add("has-ref");
    box.innerHTML = "";
    opts.forEach((o) => {
      const s = o.s;
      const row = document.createElement("div");
      row.className = "hbar" + (o.k === p.pricing ? " current" : "");
      const label = TIER_LABELS[o.k] || o.k;
      if (s.total == null) {
        row.innerHTML = `<span class="hbar-label">${esc(label)}</span><span class="hbar-track"><span class="hbar-val">Not feasible</span></span>`;
      } else {
        const segs = [[s.gpuCost, "fill-fleet"], [s.fleet ? s.fleet.maasCost : 0, targetClass(ot)], [gemOver(s), "fill-gem"], [s.longMaasCost || 0, targetClass(rt) + " seg-long"]];
        const segHTML = segs.filter((x) => x[0] > 0).map(([v, cls]) => `<i class="hbar-seg ${cls}" style="width:${(v / max * 100).toFixed(2)}%"></i>`).join("");
        row.innerHTML = `<span class="hbar-label">${esc(label)}</span><span class="hbar-track">${segHTML}<span class="hbar-val">${fmtUSD(s.total)}</span><span class="hbar-ref" style="left:${(bl.total / max * 100).toFixed(2)}%" aria-hidden="true"></span></span>`;
        const tipRows = [["var(--s-fleet)", fmtUSD(s.gpuCost), `${s.gpus} chips`], [targetColor(ot), fmtUSD(s.fleet ? s.fleet.maasCost : 0), "overflow"]];
        if (gemOver(s) > 0.5) tipRows.push(["var(--s-gem)", fmtUSD(gemOver(s)), "over the MaaS limit"]);
        if (s.routing) tipRows.push([targetColor(rt), fmtUSD(s.longMaasCost), "over 32K"]);
        tipRows.push(["", pct(s.total / bl.total), "of " + refName]);
        row.addEventListener("pointermove", (e) => showTip(e, label, tipRows));
        row.addEventListener("pointerleave", hideTip);
      }
      box.appendChild(row);
    });
    const firstRef = box.querySelector(".hbar-ref");
    if (firstRef) firstRef.innerHTML = `<span class="hbar-ref-label">${refName} ${fmtUSD(bl.total)}</span>`;
    renderSensitivity();
    const rows = [
      [`Fleet: ${r.gpus} × ${accName(selected().hardware.accelerator_type)}, ${TIER_LABELS[p.pricing]}`, fmtUSD(r.gpuCost)],
      [`Overflow to ${targetName(ot)}`, fmtUSD(r.fleet ? r.fleet.maasCost : 0)],
    ];
    if (gemOver(r) > 0.5) rows.push([`Over the MaaS limit, to ${targetName(ft)}`, fmtUSD(gemOver(r))]);
    if (r.routing) rows.push([`Prompts over 32K to ${targetName(rt)}`, fmtUSD(r.longMaasCost)]);
    rows.push(["Total a month, overflow included", fmtUSD(r.total)]);
    const baseRows = bl.feasible
      ? `<tr><td>MaaS only, a month</td><td>${fmtUSD(bl.total)}</td></tr>`
      : `<tr><td>MaaS only can't serve the peak: ${fmtRps(bl.peakRps)} req/s vs a limit of ${fmtRps(bl.capRps)} req/s, planned at ${(bl.capRps / Math.max(1, +(S.p && S.p.headroom) || 1)).toFixed(1)} per hour to absorb bursts</td><td></td></tr>
         <tr><td>Baseline: MaaS up to its limit</td><td>${fmtUSD(bl.maasCost)}</td></tr>
         <tr><td>Baseline: overflow to ${esc(targetName(bl.target))}</td><td>${fmtUSD(bl.overflowCost)}</td></tr>
         <tr><td>Baseline, a month</td><td>${fmtUSD(bl.total)}</td></tr>`;
    $("costTable").innerHTML = `<tbody>${rows.map((x, i) => `<tr${i === rows.length - 1 ? ' class="total"' : ""}><td>${esc(x[0])}</td><td>${x[1]}</td></tr>`).join("")}
      <tr><td>Per million input tokens, output included</td><td>${perM(r.perM)}</td></tr>
      ${baseRows}
      <tr><td>Accelerator price per chip-hour</td><td>${fmtUSDc(r.price)}</td></tr>
      <tr><td>Tokens a month, input</td><td>${fmtTok(r.totalTok)}</td></tr></tbody>`;
  }

  function lineChart(container, cfg) {
    container.textContent = "";
    const W = Math.max(280, container.clientWidth || 480), H = cfg.height || 220;
    const L = 48, R = cfg.right || 84, T = 14, B = 26;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.label }, container);
    const xs = cfg.xs;
    const all = [];
    cfg.series.forEach((se) => se.values.forEach((v) => { if (isFinite(v)) all.push(v); }));
    const ymax = niceMax(Math.max.apply(null, all.concat([1])) * 1.1);
    const xmin = cfg.xmin != null ? cfg.xmin : xs[0], xmax = cfg.xmax != null ? cfg.xmax : xs[xs.length - 1];
    const x = (v) => L + ((v - xmin) / (xmax - xmin || 1)) * (W - L - R);
    const y = (v) => T + (1 - v / ymax) * (H - T - B);
    const grid = svg("g", { class: "grid" }, s);
    for (let k = 0; k <= 4; k++) {
      const v = (ymax * k) / 4;
      svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }, grid);
      svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, s, cfg.yFmt(v));
    }
    (cfg.xTicks || xs).forEach((v) => svg("text", { x: x(v), y: H - 6, "text-anchor": "middle" }, s, cfg.xFmt(v)));
    if (cfg.refX != null) {
      svg("line", { x1: x(cfg.refX), x2: x(cfg.refX), y1: T, y2: H - B, stroke: "var(--muted)", "stroke-dasharray": "4 3" }, s);
      if (cfg.refLabel) svg("text", { x: x(cfg.refX) + 4, y: T + 10 }, s, cfg.refLabel);
    }
    const labels = [];
    cfg.series.forEach((se) => {
      const pts = xs.map((xv, i) => [xv, se.values[i], i]).filter((q) => isFinite(q[1]));   // q[2]: original index
      if (!pts.length) return;
      svg("path", { d: pts.map((q, i) => (i ? "L" : "M") + x(q[0]).toFixed(1) + "," + y(q[1]).toFixed(1)).join(""), fill: "none", style: `stroke:${se.color};stroke-width:2` }, s);
      if (se.points) pts.forEach((q, i) => svg("circle", { cx: x(q[0]), cy: y(q[1]), r: se.pointR ? se.pointR(q[2]) : 4, style: `fill:${se.color};stroke:var(--sheet);stroke-width:2` }, s));
      if (se.pointLabels) pts.forEach((q) => svg("text", { x: x(q[0]) + 7, y: y(q[1]) - 6 }, s, se.pointLabels[q[2]]));
      const last = pts[pts.length - 1];
      if (se.name) labels.push({ y: y(last[1]), x: x(last[0]) + 8, text: se.name });
    });
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 14) labels[i].y = labels[i - 1].y + 14;
    labels.forEach((l) => svg("text", { x: l.x, y: l.y + 4, class: "lbl" }, s, l.text));
    if (cfg.marker) {
      const mx = x(cfg.marker.x);
      svg("line", { x1: mx, x2: mx, y1: T, y2: H - B, stroke: "var(--ink)", "stroke-width": 1, "stroke-dasharray": "2 3" }, s);
      svg("text", { x: mx, y: T - 2, "text-anchor": "middle", class: "lbl" }, s, cfg.marker.label);
    }
    // crosshair tooltip
    if (cfg.tip) {
      const cross = svg("line", { x1: 0, x2: 0, y1: T, y2: H - B, stroke: "var(--muted)", "stroke-dasharray": "3 3", visibility: "hidden" }, s);
      s.addEventListener("pointermove", (e) => {
        const rect = s.getBoundingClientRect();
        const px = ((e.clientX - rect.left) / rect.width) * W;
        let bi = 0, bd = Infinity;
        xs.forEach((xv, i) => { const d = Math.abs(x(xv) - px); if (d < bd) { bd = d; bi = i; } });
        cross.setAttribute("x1", x(xs[bi])); cross.setAttribute("x2", x(xs[bi])); cross.setAttribute("visibility", "visible");
        const t = cfg.tip(bi);
        showTip(e, t.title, t.rows);
      });
      s.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
    }
    return s;
  }

  function renderSensitivity() {
    const xmax = S.p.offPeak > 6 ? 10 : 6;
    const xs = [];
    for (let k = 1; k <= xmax + 1e-4; k += xmax > 6 ? 0.5 : 0.25) xs.push(+k.toFixed(2));
    const pts = xs.map((k) => {
      const s = evalWith({ offPeak: k });   // the fleet you set on the chart, else the cheapest one
      const b = baselineOf(s);
      return { fleet: s.total != null ? s.total : NaN, maas: b ? b.total : s.maasOnly, feasible: b ? b.feasible : true };
    });
    const fixed = S.ui.fleetOverride != null && S.r && S.r.overridden;
    const selfName = fixed ? `Self-host, ${S.r.gpus} chips` : "Self-host";
    const bName = pts.every((q) => q.feasible) ? "MaaS only" : "MaaS + overflow";
    const ticks = xmax > 6 ? [1, 2, 4, 6, 8, 10] : [1, 2, 3, 4, 5, 6];
    lineChart($("sensChart"), {
      label: `Monthly cost of self-hosting and of ${bName} as off-peak traffic grows`,
      xs, xTicks: ticks, xFmt: (v) => v + "×", yFmt: fmtUSD,
      series: [
        { name: selfName, color: "var(--s-fleet)", values: pts.map((q) => q.fleet) },
        { name: bName, color: "var(--s-maas)", values: pts.map((q) => q.maas) },
      ],
      marker: S.p.offPeak !== 1 ? { x: Math.min(xmax, S.p.offPeak), label: "Now" } : null,
      tip: (i) => ({ title: `Off-peak ×${xs[i]}`, rows: [["var(--s-fleet)", fmtUSD(pts[i].fleet), selfName.toLowerCase()], ["var(--s-maas)", fmtUSD(pts[i].maas), pts[i].feasible ? "MaaS only" : "MaaS + overflow"]] }),
    });
  }

  function renderResilience() {
    const r = S.r, p = S.p;
    const opts = [[1, "Single zone, Gemini failover"], [1.5, "Zone-redundant, 1.5×"], [2, "Two regions, 2×"]];
    const single = evalWith({ resilience: 1 });
    const rows = opts.map(([m, label]) => {
      const s = evalWith({ resilience: m });
      const cur = m === p.resilience;
      return `<tr${cur ? ' class="current"' : ""}><td>${esc(label)}${cur ? ' <span class="sel-tag">Current</span>' : ""}</td><td class="n">${s.gpus}</td><td class="n">${fmtUSD(s.total)}</td><td class="n">${m === 1 ? "—" : "+" + fmtUSD(s.total - single.total)}</td><td class="n">${pct(vsBase(s))}</td></tr>`;
    }).join("");
    $("resTable").innerHTML = `<thead><tr><th>Option</th><th class="n">Chips</th><th class="n">A month</th><th class="n">Extra</th><th class="n">Share of ${baseShort(baselineOf(r))}</th></tr></thead><tbody>${rows}</tbody>`;
    let inc = null;
    try { inc = M.incidents(S.data, p, r); } catch (_) { inc = null; }
    if (!inc) { $("incTable").innerHTML = ""; $("resLead").textContent = ""; return; }
    // A failover target pinned to a specific model (not one of the role columns) gets its own column.
    const failCol = inc.failoverTarget && GEMINI_KEYS.indexOf(inc.failoverTarget) < 0 && inc.rows[0] && inc.rows[0].fail != null;
    $("incTable").innerHTML = `<thead><tr><th>Incident</th><th class="n">Tokens</th><th class="n">${esc(targetName("gemini_flash_lite"))}</th><th class="n">${esc(targetName("gemini_flash"))}</th>${inc.rows[0] && inc.rows[0].pro != null ? `<th class="n">${esc(targetName("gemini_pro"))}</th>` : ""}${inc.rows[0] && inc.rows[0].g4 != null ? `<th class="n">${esc(targetName("gemini_4"))}</th>` : ""}${failCol ? `<th class="n">${esc(targetName(inc.failoverTarget))}</th>` : ""}<th class="n">Open-model MaaS</th></tr></thead><tbody>` +
      inc.rows.map((x) => `<tr><td>${esc(x.name.replace("Open MaaS", "Open-model MaaS"))}</td><td class="n">${fmtTok(x.tok)}</td><td class="n">${fmtUSDc(x.lite)}</td><td class="n">${fmtUSDc(x.flash)}</td>${x.pro != null ? `<td class="n">${fmtUSDc(x.pro)}</td>` : ""}${x.g4 != null ? `<td class="n">${fmtUSDc(x.g4)}</td>` : ""}${failCol ? `<td class="n">${fmtUSDc(x.fail)}</td>` : ""}<td class="n">${fmtUSDc(x.open)}</td></tr>`).join("") + "</tbody>";
    const hours = inc.breakeven_outage_hours != null ? inc.breakeven_outage_hours : inc.breakevenHours;
    const never = inc.breakeven_never === true || !(hours <= 730);
    const peakHour = inc.rows[0] ? inc.rows[0].lite : 0;
    let lead = `Zone redundancy adds <strong>${inc.zoneGpus} chips</strong> (+${fmtUSD(inc.zoneMonthly)} a month); two regions add <strong>${inc.regGpus}</strong> (+${fmtUSD(inc.regMonthly)}). `;
    lead += `A full peak-hour outage failed over to ${esc(targetName("gemini_flash_lite"))} costs about <strong>${fmtUSDc(peakHour)}</strong>. `;
    lead += never
      ? "Failover is cheaper than idle redundant hardware at any realistic outage rate."
      : `Idle redundant hardware only pays off past <strong>${Math.round(hours)} peak outage hours</strong> a month.`;
    $("resLead").innerHTML = lead;
  }

  function renderLong() {
    const r = S.r, p = S.p, b = r.buckets;
    const legend = [["1", "Up to 32K"], [".6", "32K–128K"], [".32", "Over 128K"]].map(([o, l]) => `<li><span class="sw fill-fleet" style="opacity:${o}"></span>${l}</li>`).join("");
    $("longLegend").innerHTML = legend;
    const tw = b.small.relGpuSec + b.medium.relGpuSec + b.large.relGpuSec;
    const rows = [
      ["Requests", [b.small.req, b.medium.req, b.large.req]],
      ["Tokens", [b.small.tokenShare, b.medium.tokenShare, b.large.tokenShare]],
      ["Fleet time", [b.small.relGpuSec / tw, b.medium.relGpuSec / tw, b.large.relGpuSec / tw]],
    ];
    $("longBars").classList.remove("has-ref");
    $("longBars").innerHTML = rows.map(([label, v]) => {
      const segs = v.map((x, i) => x > 0 ? `<i class="hbar-seg fill-fleet" style="width:${(x * 100).toFixed(2)}%;opacity:${[1, 0.6, 0.32][i]}"></i>` : "").join("");
      return `<div class="hbar current"><span class="hbar-label">${label}: ${pct1(v[1] + v[2])} over 32K</span><span class="hbar-track">${segs}</span></div>`;
    }).join("");
    $("longBars").querySelectorAll(".hbar").forEach((el) => { el.style.gridTemplateColumns = ""; });
    // Rows: the role aliases (newest model of each kind), the routed model if it is pinned, every other offered
    // model after "Show all", then open-model MaaS.
    const gm = gemMap(p);
    const roleIds = new Set(GEMINI_KEYS.map((k) => gm[k] && gm[k].modelId));
    const extra = geminiModels().filter((m) => gm[m.id] && !roleIds.has(m.id) && (!m.deprecated || m.id === p.routeLongTarget));
    const pinned = extra.filter((m) => m.id === p.routeLongTarget);
    const rest = extra.filter((m) => m.id !== p.routeLongTarget);
    const opts = [[null, "Keep them on the fleet"]]
      .concat(GEMINI_KEYS.filter((k) => gm[k]).map((k) => [k, targetName(k)]))
      .concat(pinned.map((m) => [m.id, m.name + stageTag(m)]))
      .concat(S.routeAll ? rest.map((m) => [m.id, m.name + stageTag(m)]) : [])
      .concat([["open_maas", "Open-model MaaS"]]);
    const allOn = evaluate(Object.assign({}, p, { routeLong: false }), null);
    let best = null;
    const trs = opts.map(([t, label]) => {
      const s = evaluate(Object.assign({}, p, t ? { routeLong: true, routeLongTarget: t } : { routeLong: false }), null);
      const cur = t ? (p.routeLong && p.routeLongTarget === t) : !p.routeLong;
      if (t && s.total != null && !s.fallbackNote && (!best || s.total < best.s.total)) best = { t, s, label };
      if (s.total == null || s.fallbackNote) return `<tr${cur ? ' class="current"' : ""}><td>${esc(label)}</td><td class="n" colspan="4">Over the MaaS rate limit</td></tr>`;
      return `<tr${cur ? ' class="current"' : ""}><td>${esc(label)}${cur ? ' <span class="sel-tag">Current</span>' : ""}</td><td class="n">${s.gpus}</td><td class="n">${fmtUSD(s.gpuCost)}</td><td class="n">${fmtUSD(s.total)}</td><td class="n">${pct(vsBase(s))}</td></tr>`;
    }).join("");
    const moreBtn = rest.length ? `<tfoot><tr><td colspan="5"><button type="button" class="link-btn" id="btnRouteAll" aria-expanded="${S.routeAll ? "true" : "false"}">${S.routeAll ? "Show only the newest of each kind" : `Show all Gemini models (${rest.length} more)`}</button></td></tr></tfoot>` : "";
    $("routeTable").innerHTML = `<thead><tr><th>Prompts over 32K go to</th><th class="n">Chips</th><th class="n">Fleet a month</th><th class="n">Total a month</th><th class="n">Share of ${baseShort(baselineOf(r))}</th></tr></thead><tbody>${trs}</tbody>${moreBtn}`;
    const ra = $("btnRouteAll");
    if (ra) ra.onclick = () => { S.routeAll = !S.routeAll; renderLong(); const b = $("btnRouteAll"); if (b) b.focus(); };
    const none = !(p.pctOver32 > 0);
    ["longBars", "longLegend", "routeTable"].forEach((id) => { $(id).hidden = none; });
    $("panel-long").querySelectorAll(".panel-h").forEach((h) => { h.hidden = none; });
    if (none) {
      $("longLead").textContent = "No prompts over 32K in this workload. Set their share under Prompts to see what routing them elsewhere would save.";
      return;
    }
    let lead = `Prompts over 32K are <strong>${pct1(b.longReqShare)}</strong> of requests and <strong>${pct(b.longTokenShare)}</strong> of tokens, but take <strong>${pct(b.longGpuShare)}</strong> of fleet time.`;
    if (best && allOn.total != null && best.s.total < allOn.total) {
      lead += ` Sending them to ${esc(best.label)} saves <strong>${Math.max(0, allOn.gpus - best.s.gpus)} chips</strong> and <strong>${fmtUSD(allOn.total - best.s.total)}</strong> a month.`;
    }
    $("longLead").innerHTML = lead;
  }

  // ---------------------------------------------------------------- simulation
  const SIM_NAMES = {
    gpu: "All prompts on the fleet",
    flat: "Long prompts spread evenly over the day",
    deferred: "Long prompts deferred to off-peak hours",
    routed: "Long prompts routed to MaaS",
  };
  const burstLabel = (b) => (+b === 1 ? "None" : +b === 2 ? "With 10-minute surges (2×)" : `${b}×`);
  function renderSim() {
    if (!S.session) return;
    const sim = S.session.simulation;
    const btn = $("btnRunSim");
    btn.disabled = S.simRunning;
    btn.lastChild.textContent = sim ? "Run it again" : "Run simulation";
    const st = $("simStatus");
    if (S.simRunning) st.innerHTML = `<span class="busy-note">Replaying a week of requests. This takes 5–20 seconds…</span>`;
    else if (S.dirty) st.textContent = "Includes your unsaved changes. The result isn't saved until you save the session.";
    else if (sim && sim.whatIf) st.textContent = "This result includes changes that aren't saved.";
    else st.textContent = "";
    $("simResult").hidden = !sim;
    if (!sim) return;
    const notes = sim.notes || [];
    $("simNotes").hidden = !notes.length;
    $("simNotes").innerHTML = notes.map((n) => `<li>${icon("i-info", "ico-sm")}<span>${esc(n)}</span></li>`).join("");
    const rows = (sim.rows || []).slice();
    let best = null;
    rows.forEach((r) => { if (r.feasible && r.total_month != null && (!best || r.total_month < best.total_month)) best = r; });
    $("simTable").innerHTML = `<thead><tr><th>Approach</th><th>Surges</th><th class="n">GPUs</th><th class="n">Hourly model</th><th class="n">A month</th><th class="n">Share of MaaS only</th><th class="n">Late, short / long</th><th>Passes</th></tr></thead><tbody>` +
      rows.map((r) => {
        const w = r.week || {};
        const isBest = r === best;
        return `<tr${isBest ? ' class="current"' : ""}><td>${esc(SIM_NAMES[r.scenario] || r.scenario)}${isBest ? ' <span class="sel-tag">Cheapest that passes</span>' : ""}</td><td>${esc(burstLabel(r.burst))}</td>
          <td class="n">${r.gpus != null ? r.gpus : "—"}</td><td class="n">${r.hourly_model_gpus != null ? r.hourly_model_gpus : "—"}</td>
          <td class="n">${fmtUSD(r.total_month)}</td><td class="n">${pct(r.vs_maas_only)}</td>
          <td class="n">${pct1(w.late_short)} / ${pct1(w.late_long)}</td>
          <td>${r.feasible ? `<span class="yes">${icon("i-check", "ico-sm")}Yes</span>` : `<span class="no">${icon("i-x", "ico-sm")}No</span>`}</td></tr>`;
      }).join("") + "</tbody>";
    const inp = sim.inputs || {};
    const bits = [];
    if (inp.tier) bits.push(`Prices: ${TIER_LABELS[inp.tier] || inp.tier}.`);
    if (inp.input_tps_per_gpu) bits.push(`Per-chip throughput: ${fmtInt(inp.input_tps_per_gpu)} input tok/s.`);
    if (inp.latency_budget_s) bits.push(`Latency budget: ${fmtInt(inp.latency_budget_s)} s.`);
    bits.push("“Hourly model” is the GPU count the averages-based answer above would buy.");
    $("simFoot").textContent = bits.join(" ");
  }
  async function runSim() {
    if (!S.session || S.simRunning) return;
    S.simRunning = true;
    renderSim();
    try {
      const body = { scenarios: ["gpu", "flat", "deferred", "routed"], bursts: [1, 2], target_late: 0.01 };
      if (S.p && S.p.pricing) body.tier = S.p.pricing;
      if (S.dirty) body.config = collectCfg();   // what-if: simulate unsaved edits without saving them
      const res = await postJSON(`/api/sessions/${enc(S.name)}/simulate`, body);
      if (res.config) applySession(res, { keepCfg: S.dirty });
      else S.session.simulation = res.simulation ? { ...res.simulation, whatIf: !!res.what_if } : null;
    } catch (e) {
      toast(`The simulation failed: ${e.message}. Check the inputs and try again.`, true);
    } finally {
      S.simRunning = false;
      renderSim();
    }
  }

  // ---------------------------------------------------------------- benchmark
  const FORCE_STOP_AFTER_MS = 60000;
  function jobStartedAt() {
    const j = S.job;
    if (j && j.started_at) return j.started_at * 1000;
    const key = "dilemma-job-start-" + S.name;
    let v = +sessionStorage.getItem(key);
    if (!v) { v = Date.now(); sessionStorage.setItem(key, String(v)); }
    return v;
  }
  function endpointRate() {
    const prices = S.data && S.data.prices;
    const rep = (S.data && S.data.replica) || selected().hardware.accelerator_count || 1;
    const ep = S.session && S.session.endpoint;
    const count = (ep && ep.accelerator_count) || rep;
    return prices && prices.vertex ? prices.vertex * count : null;
  }
  function renderRunPill() {
    const pill = $("runPill"), j = S.job, ep = S.session && S.session.endpoint;
    const running = j && j.status === "running";
    $("tab-bench").innerHTML = "Benchmark" + (running ? '<span class="tab-dot" aria-label="(running)"></span>' : "");
    if (running) {
      const st = STEPS.find((x) => x.id === j.stage);
      const lp = j.level_progress;
      let txt = `Benchmark: ${st ? st.name.toLowerCase() : "running"}`;
      if (j.stage === "load" && lp && lp.total) txt += ` ${lp.completed}/${lp.total}`;
      txt += ` · ${fmtElapsed((Date.now() - jobStartedAt()) / 1000)}`;
      $("runPillText").textContent = txt;
      pill.classList.remove("endpoint", "alert");
      pill.hidden = false;
    } else if (j && j.endpoint_warning) {
      $("runPillText").textContent = "Benchmark interrupted · endpoint may be billing";
      pill.classList.remove("endpoint");
      pill.classList.add("alert");
      pill.hidden = false;
    } else if (ep && ep.endpoint) {
      $("runPillText").textContent = `Endpoint live · ${ep.accelerator_count || ""} × ${accName(ep.accelerator_type)}`;
      pill.classList.remove("alert");
      pill.classList.add("endpoint");
      pill.hidden = false;
    } else pill.hidden = true;
  }

  function realLevels() {
    const j = S.job, m = S.session && S.session.measured;
    if (j && j.status === "running" && Array.isArray(j.levels_completed) && j.levels_completed.length) return { levels: j.levels_completed, peak: null };
    if (m && !m.estimated && Array.isArray(m.levels) && m.levels.length) return { levels: m.levels, peak: m.peak ? m.peak.concurrency : null };
    return null;
  }

  // ---- serving image (contract: deployment.image; null = the Model Garden verified recipe's image)
  // Artifact Registry (<region>-docker.pkg.dev/<project>/<repo>/<image>) or Container Registry ([xx.]gcr.io/<project>/<image>),
  // with an optional :tag or @sha256 digest.
  const IMAGE_RE = /^(?:[a-z0-9-]+-docker\.pkg\.dev\/[a-z0-9][a-z0-9.:-]*(?:\/[a-z0-9][a-z0-9._-]*){2,}|(?:[a-z]+\.)?gcr\.io\/[a-z0-9][a-z0-9.:-]*(?:\/[a-z0-9][a-z0-9._-]*)+)(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[a-f0-9]{64})?$/;
  const validImage = (v) => IMAGE_RE.test(String(v || "").trim());
  // "pytorch-vllm-serve:gemma4 (sha256:3fbc0e08…)" for a digest pin with a catalog tag (same as loadtest.short_image).
  function shortImage(img) {
    const name = String(img || "").split("/").pop();
    const at = name.indexOf("@sha256:");
    if (at < 0) return name;
    const entry = ((S.catalog && S.catalog.serving_images) || []).find((x) => x.image === img);
    const repo = name.slice(0, at), digest = name.slice(at + 1);
    return entry && entry.tag ? `${repo}:${entry.tag} (${digest.slice(0, 15)}…)` : `${repo}@${digest.slice(0, 15)}…`;
  }
  function pinnedImage() { return ((S.catalog && S.catalog.serving_images) || []).find((x) => x.key === "pinned") || null; }
  function imageChoice(img) {
    if (!img) return "verified";
    const p = pinnedImage();
    return p && p.image === img ? "pinned" : "custom";
  }
  function pinnedTested() { const p = pinnedImage(); return p && p.tested ? `last tested ${p.tested}` : "not yet tested"; }
  function setImageError(msg) {
    $("imageError").textContent = msg || "";
    $("imageError").hidden = !msg;
    if (msg) $("f-imageCustom").setAttribute("aria-invalid", "true"); else $("f-imageCustom").removeAttribute("aria-invalid");
  }
  function renderImageField() {
    if (!S.cfg) return;
    const img = S.cfg.deployment.image || null;
    const sel = $("f-image"), opt = sel.querySelector('option[value="pinned"]');
    const p = pinnedImage();
    // The pinned image is a GPU (CUDA) build: not offered for TPUs; flag GPUs it wasn't tested on.
    const hw = selected().hardware || {}, kind = accKind(hw.accelerator_type) || "gpu";
    const fits = !p || !p.accelerator_kinds || p.accelerator_kinds.indexOf(kind) >= 0;
    const tested = !p || !p.tested_accelerators || p.tested_accelerators.indexOf(hw.accelerator_type) >= 0;
    opt.textContent = !fits ? `Pinned image (GPUs only)` : tested ? `Pinned image (${pinnedTested()})`
      : `Pinned image (not tested on ${accName(hw.accelerator_type)})`;
    opt.disabled = !p || !fits;
    if (p) opt.title = p.image;
    const hint = $("imageHint");
    if (!hint.dataset.base) hint.dataset.base = hint.textContent;
    // "Custom" stays selected while its URI is still empty or invalid (deployment.image unchanged until it's valid).
    const keep = S.imageChoice && S.imageChoice.name === S.name ? S.imageChoice.v : null;
    const choice = keep === "custom" ? "custom" : imageChoice(img);
    sel.value = choice;
    sel.title = img || "";
    $("imageCustomField").hidden = choice !== "custom";
    if (choice === "custom" && imageChoice(img) === "custom" && document.activeElement !== $("f-imageCustom")) $("f-imageCustom").value = img;
    if (choice !== "custom") setImageError("");
    hint.classList.toggle("warn", choice === "pinned" && !fits);
    hint.textContent = choice === "pinned" && !fits
      ? `The pinned image is a GPU build and won't run on ${accName(hw.accelerator_type)}. Choose the Model Garden verified recipe; the benchmark won't deploy otherwise.`
      : choice === "pinned" && p && p.tested
      ? `${shortImage(p.image)}, tested ${p.tested}${p.tested_on ? " on " + p.tested_on : ""}. ${p.notes || ""}`.trim() +
        (tested ? "" : ` Not tested on ${accName(hw.accelerator_type)}:`) + " The pre-flight check still runs on every benchmark."
      : hint.dataset.base;
  }
  function servingText(sv) {
    const flags = (sv.args || []).filter((a) => /^--/.test(a) && !/^--(host|port)\b/.test(a));
    return `Measured with <code>${esc(shortImage(sv.image))}</code>` +
      (flags.length ? `, vLLM args <code>${esc(flags.join(" "))}</code>` : "") +
      (sv.inference_timeout_s ? `, endpoint inference timeout ${esc(sv.inference_timeout_s)} s` : "") + ".";
  }
  function measuredServing() {
    const m = S.session && S.session.measured;
    if (!m || m.estimated) return null;
    const sv = m.serving || (m.endpoint && m.endpoint.image ? { image: m.endpoint.image, args: m.endpoint.args, inference_timeout_s: m.endpoint.inference_timeout_s } : null);
    return sv && sv.image ? sv : null;
  }
  function renderBenchServing() {
    if (!S.cfg) return;
    const img = S.cfg.deployment.image || null, ch = imageChoice(img);
    const what = ch === "verified" ? "the Model Garden verified recipe" :
      ch === "pinned" ? `the pinned image <code>${esc(shortImage(img))}</code> (${esc(pinnedTested())})` : `a custom image <code>${esc(shortImage(img))}</code>`;
    const pre = S.cfg.loadtest.preflight !== false ? "A pre-flight check (about two minutes) runs before the sweep." : "The pre-flight check is off.";
    let html = `Serving image: ${what}. ${pre} <button type="button" class="link-btn" id="btnEditImage">Change</button>`;
    html += `<br><span id="benchPrompts">Prompts: ${esc(promptSourceText())}.</span> <button type="button" class="link-btn" id="btnEditPrompts">Change</button>`;
    const sv = measuredServing();
    if (sv) html += `<br>Last run: ${servingText(sv)}`;
    const lp = S.session && S.session.measured && !S.session.measured.estimated && S.session.measured.prompt_source;
    if (lp && lp.text) html += ` Prompts: ${esc(lp.text)}.`;
    const pf = S.session && S.session.measured && S.session.measured.preflight;
    if (pf && pf.ran) html += ` Pre-flight ${pf.passed ? "passed" : "failed"}${pf.duration_s ? ` in ${Math.round(pf.duration_s)} s` : ""}.`;
    $("benchServing").innerHTML = html;
    $("benchServing").title = img || "";
  }
  function editServingImage() {
    const sel = $("f-image"), det = sel.closest("details");
    if (getComputedStyle($("btnEditWorkload")).display !== "none" && !$("rail").classList.contains("open")) openRail();
    if (det) det.open = true;
    setTimeout(() => { sel.scrollIntoView({ block: "center", behavior: reduceMotion.matches ? "auto" : "smooth" }); sel.focus(); }, 60);
  }

  function renderBench() {
    if (!S.session) return;
    renderBenchServing();
    renderRunPill();
    const j = S.job || { status: "idle" };
    const running = j.status === "running";
    const failed = j.status === "error" || j.status === "failed" || j.status === "interrupted";
    const stopAt = running && j.stop_requested_at ? j.stop_requested_at * 1000 : 0;
    $("btnRunBench").hidden = running;
    $("btnStopBench").hidden = !running;
    $("btnStopBench").disabled = !!stopAt;
    $("btnStopBench").lastChild.textContent = stopAt ? "Stopping…" : "Stop";
    $("btnForceStop").hidden = !(stopAt && Date.now() - stopAt > FORCE_STOP_AFTER_MS);
    $("chkKeepEndpoint").disabled = running;
    $("stepPicker").hidden = running;
    // meter
    const rate = endpointRate();
    const ep = S.session.endpoint;
    let meter = "";
    if (running) {
      const el = (Date.now() - jobStartedAt()) / 1000;
      meter = `Running for <strong>${fmtElapsed(el)}</strong>` +
        (j.source === "cli" ? ` (started from the command line${j.command ? `: <code>${esc(j.command)}</code>` : ""})` : "") + ".";
      if (stopAt) meter += Date.now() - stopAt > FORCE_STOP_AFTER_MS
        ? " It hasn't stopped yet: it may be waiting for a deployment to finish. Force stop ends it at once, but then check that the endpoint is gone."
        : j.stage === "deploy" ? " Stopping: it stops once the deployment finishes (Agent Platform can't cancel it midway), then tears the endpoint down."
        : " Stopping: it finishes the current request batch, then tears down the endpoint it deployed.";
      if (rate) {
        const billed = ["deploy", "load", "analyze", "report", "teardown"].indexOf(j.stage) >= 0 || (ep && ep.endpoint);
        meter += ` The endpoint bills about <strong>${fmtUSDc(rate)} an hour</strong> while it exists` + (billed ? `, roughly <strong>${fmtUSDc((rate * el) / 3600)}</strong> so far.` : ".");
      }
    } else if (j.status === "completed") meter = "The last run finished. The numbers above now use its throughput.";
    else if (j.status === "error" || j.status === "failed") {
      const errText = String(j.error || "see the run log").replace(/\.\s*$/, "");
      meter = `The run stopped with an error: ${esc(errText)}. Fix the cause, then run it again.`;
      const pin = pinnedImage();
      if (/^Pre-flight failed/.test(errText) && pin && S.cfg.deployment.image !== pin.image) {
        meter += ` <button type="button" class="link-btn" id="btnUsePinned">Use the pinned image</button>`;
      }
    }
    else if (j.status === "interrupted") meter = esc(j.error || "The run stopped unexpectedly.");
    else if (j.status === "cancelled") meter = "The run was stopped.";
    $("benchMeter").innerHTML = meter;
    // a job that died (server or machine restart, force stop) may have left a billing endpoint behind
    const warn = !running && !!j.endpoint_warning;
    const deployWarn = !running && !warn && !!j.deploy_warning;
    $("jobAlert").hidden = !(warn || deployWarn);
    $("btnAlertTeardown").hidden = !warn;
    if (warn) {
      $("jobAlertText").innerHTML = `<strong>A benchmark stopped unexpectedly and its endpoint may still be running (billing).</strong> Tear it down` +
        (rate ? ` to stop paying about ${fmtUSDc(rate)} an hour.` : ".");
    } else if (deployWarn) {
      $("jobAlertText").innerHTML = `<strong>A benchmark stopped while its endpoint was being deployed.</strong> Agent Platform may still finish creating it, and this app has no record of it: check the Agent Platform endpoints in the Google Cloud console for one named <code>${esc(("dilemma-" + S.name).slice(0, 60))}</code> and delete it if it's there.`;
    }
    // steps
    // The job's own step list (it may run a subset); unknown steps get a generic label so a new step never
    // leaves the earlier ones looking unfinished.
    const jobSteps = Array.isArray(j.steps) && j.steps.length ? j.steps : STEPS.map((s) => s.id);
    const shown = jobSteps.map((id) => STEPS.find((s) => s.id === id) || { id, name: id.charAt(0).toUpperCase() + id.slice(1), desc: "" });
    const order = shown.map((s) => s.id);
    const ai = j.stage === "done" ? order.length : order.indexOf(j.stage);
    const allDone = j.status === "completed";
    $("steps").innerHTML = shown.map((s, i) => {
      let cls = "step";
      if (allDone || (running && ai > i) || (!running && j.status !== "idle" && ai > i)) cls += " done";
      if (running && ai === i) cls += " active";
      if (failed && ai === i) cls += " failed";
      let detail = s.desc;
      if (running && ai === i && s.id === "load" && j.preflight && j.preflight.status === "running") {
        detail = "Pre-flight check: a long answer and a long prompt";
      } else if (running && ai === i && s.id === "load" && j.level_progress && j.level_progress.total) {
        const lp = j.level_progress;
        detail = `c${lp.concurrency} · ${runName(lp.run || (j.stage === "calibrate" ? "short" : "mixed"))}: ${lp.completed}/${lp.total}`;
      }
      const mk = cls.indexOf("done") >= 0 ? icon("i-check") : cls.indexOf("failed") >= 0 ? icon("i-x") : "";
      const runBtn = "";
      return `<li class="${cls}"${running && ai === i ? ' aria-current="step"' : ""}><span class="marker">${mk}</span><span class="step-name">${s.name}</span><span class="step-detail">${esc(detail)}</span>${runBtn}</li>`;
    }).join("");
    // endpoint
    const hasEp = !!(ep && ep.endpoint);
    $("endpointBox").hidden = !hasEp;
    if (hasEp) {
      $("endpointText").innerHTML = `Endpoint <code>${esc(String(ep.endpoint).split("/").pop())}</code> is serving ${esc(ep.accelerator_count || "")} × ${esc(accName(ep.accelerator_type))}` +
        (rate ? `. It bills about ${fmtUSDc(rate)} an hour until you tear it down.` : ". It bills until you tear it down.");
      $("btnTeardown").disabled = running;
      $("btnTeardown").hidden = warn;     // the alert above carries the same button
    }
    // sweep
    const lv = realLevels();
    $("sweepWrap").hidden = !lv;
    if (lv) renderSweep(lv.levels, lv.peak);
    // log
    if (Array.isArray(j.logs) && j.logs.length) {
      const pre = $("runLog");
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
      pre.textContent = j.logs.join("\n");
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    }
  }

  // Each level's prompt set: "mixed" (the workload's mix), "short" (calibration, <= 32K), "long". Older runs don't
  // record it: a concurrency seen again later in the same job is the short-only calibration pass.
  function labelLevels(levels) {
    const seen = new Set();
    return (levels || []).map((s) => {
      let run = s.run;
      if (!run) run = seen.has(s.concurrency) ? "short" : "mixed";
      seen.add(s.concurrency);
      return Object.assign({}, s, { run });
    });
  }
  const RUN_STYLE = { mixed: ["var(--s-fleet)", "mixed"], short: ["var(--s-maas)", "short"], long: ["var(--s-gem)", "long"] };
  const runName = (r) => (RUN_STYLE[r] ? RUN_STYLE[r][1] : r);
  const levelLabel = (s) => `c${s.concurrency} · ${runName(s.run)}`;

  // Compact per-level prompt mix (loadtest.level_mix): "97/2/1 · 10% think", details in the tooltip.
  function mixCell(m, note) {
    if (!m || !m.requests) return '<td class="n muted">–</td>';
    const pc = (v) => Math.round(100 * (v || 0));
    const r = m.requests, t = m.tokens || {};
    const tip = `Requests by size (32K or less / 32K-128K / over 128K): ${pc(r.short)}% / ${pc(r.medium)}% / ${pc(r.long)}%. ` +
      `Input tokens: ${pc(t.short)}% / ${pc(t.medium)}% / ${pc(t.long)}%. Thinking on: ${pc(m.thinking)}%. ` +
      `Average ${fmtInt(m.avg_in || 0)} tokens in, ${fmtInt(m.avg_out || 0)} out.` + (m.repeats ? ` ${m.repeats} requests reused a prompt.` : "") + (note ? " " + note : "");
    return `<td class="n" title="${esc(tip)}">${pc(r.short)}/${pc(r.medium)}/${pc(r.long)} · ${pc(m.thinking)}% think${m.repeats ? " *" : ""}</td>`;
  }

  function renderSweep(levels, peakConc) {
    const budget = +S.cfg.constraints.latency_budget_s || null;
    const all = labelLevels(levels);
    const ok = all.filter((s) => s.ok > 0);
    if (ok.length) {
      const xsLat = ok.map((s) => s.lat_p95 || 0);
      const xmax = niceMax(Math.max.apply(null, xsLat.concat([budget && budget < Math.max.apply(null, xsLat) * 3 ? budget : 0])) * 1.1);
      // one line per prompt set, each drawn over the shared latency axis (NaN = not part of this line)
      const runs = [...new Set(ok.map((s) => s.run))];
      const series = runs.map((r) => ({
        color: (RUN_STYLE[r] || ["var(--ink)"])[0],
        values: ok.map((s) => (s.run === r ? s.input_tps || 0 : NaN)),
        points: true,
        pointR: (i) => (ok[i].run === "mixed" && ok[i].concurrency === peakConc ? 6 : 4),
        pointLabels: ok.map((s) => (s.run === r ? levelLabel(s) : "")),
      }));
      lineChart($("sweepChart"), {
        label: "Input throughput against p95 latency for each concurrency level and prompt set",
        xs: xsLat, xmin: 0, xmax, xTicks: [0, xmax / 4, xmax / 2, (3 * xmax) / 4, xmax], xFmt: (v) => Math.round(v) + "s", yFmt: fmtTok, right: 24,
        series,
        refX: budget && budget <= xmax ? budget : null, refLabel: "Latency budget",
        tip: (i) => ({ title: `Concurrency ${ok[i].concurrency}, ${runName(ok[i].run)} prompts`, rows: [[(RUN_STYLE[ok[i].run] || ["var(--ink)"])[0], fmtTok(ok[i].input_tps), "input tok/s"], ["", (ok[i].lat_p95 || 0).toFixed(1) + "s", "p95 latency"]] }),
      });
    } else $("sweepChart").textContent = "";
    levels = all;
    $("sweepTable").innerHTML = `<thead><tr><th>Level</th><th class="n">OK / errors</th><th class="n">Req/s</th><th class="n">Input tok/s</th><th class="n">Output tok/s</th><th class="n">p50</th><th class="n">p95</th><th class="n">TTFT p50</th><th class="n" title="Prompt mix the level saw: share of requests short (32K or less) / medium (32K-128K) / long (over 128K), and with thinking on. Every level gets the same mix.">Mix</th></tr></thead><tbody>` +
      levels.map((s) => { const pk = s.concurrency === peakConc && (s.run || "mixed") === "mixed"; return `<tr${pk ? ' class="current"' : ""}><td>${esc(levelLabel(s))}${pk ? ' <span class="sel-tag">Peak</span>' : ""}</td><td class="n">${s.ok || 0} / ${s.errors || 0}</td><td class="n">${(s.rps || 0).toFixed(2)}</td><td class="n">${fmtInt(s.input_tps || 0)}</td><td class="n">${fmtInt(s.output_tps || 0)}</td><td class="n">${(s.lat_p50 || 0).toFixed(1)}s</td><td class="n">${(s.lat_p95 || 0).toFixed(1)}s</td><td class="n">${(s.ttft_p50 || 0).toFixed(2)}s</td>${mixCell(s.mix, s.sample_note)}</tr>`; }).join("") + "</tbody>";
  }

  async function runBench(mode, steps) {
    if (!S.session) return;
    const sel = selected(), hw = sel.hardware || {};
    const rate = endpointRate();
    const conc = S.cfg.loadtest.concurrency, per = S.cfg.loadtest.requests_per_level;
    const deploys = mode === "auto" || (steps || []).indexOf("deploy") >= 0;
    if (mode === "auto" || deploys) {
      const ok = await dialog({
        title: mode === "auto" ? "Run a benchmark on Agent Platform?" : "Deploy an endpoint?",
        body: `<p>This deploys ${esc(modelName(sel.model))} on ${esc(hw.accelerator_count || 1)} × ${esc(accName(hw.accelerator_type))}` +
          (rate ? `, which bills about ${fmtUSDc(rate)} an hour` : "") + `. Deployment takes 10–30 minutes.</p>` +
          (mode === "auto" ? `<p>It then ${S.cfg.loadtest.preflight !== false ? "runs a two-minute pre-flight check (it stops and tears down if the endpoint cuts long requests), and " : ""}sends ${per} requests at each of ${conc.length} concurrency levels (${esc(conc.join(", "))})` +
            ($("chkKeepEndpoint").checked ? " and keeps the endpoint afterwards." : " and tears the endpoint down at the end.") + " You can stop at any time.</p>" : ""),
        okLabel: mode === "auto" ? "Deploy and run" : "Deploy",
      });
      if (!ok) return;
    }
    if (!(await ensureWritable())) return;
    try {
      await postJSON(`/api/sessions/${enc(S.name)}/run`, {
        config: collectCfg(), mode, steps, keep_endpoint: $("chkKeepEndpoint").checked,
        concurrency: conc, requests_per_level: per,
      });
      sessionStorage.setItem("dilemma-job-start-" + S.name, String(Date.now()));
      S.job = { status: "running", stage: (steps && steps[0]) || "plan", logs: [], levels_completed: [], level_progress: null };
      S.dirty = false; renderSaveState();
      selectTab("bench");
      connectSse();
      startPolling();
    } catch (e) {
      if (e.status === 409 || /already running/i.test(e.message)) {
        toast("A benchmark is already running for this session. Showing it.");
        await refreshSession();
        selectTab("bench");
      } else toast(`Couldn't start the benchmark: ${e.message}`, true);
    }
  }

  function connectSse() {
    if (S.sse) S.sse.close();
    const name = S.name;
    // Resume after a dropped connection (e.g. a server restart) from the last event seen; replay all on a new page.
    const since = S.evPos && S.evPos.name === name && S.evPos.started === (S.job && S.job.started_at) ? S.evPos.pos : 0;
    const src = new EventSource(`/api/sessions/${enc(name)}/events` + (since ? `?since=${since}` : ""));
    S.sse = src;
    src.onmessage = async (e) => {
      if (S.name !== name) { src.close(); return; }
      if (e.lastEventId) S.evPos = { name, started: S.job && S.job.started_at, pos: +e.lastEventId };
      let ev;
      try { ev = JSON.parse(e.data); } catch (_) { return; }
      const j = S.job || (S.job = { status: "running", logs: [], levels_completed: [] });
      j.logs = j.logs || [];
      if (ev.message) {
        const line = `[${new Date((ev.ts || Date.now() / 1000) * 1000).toLocaleTimeString()}] ${ev.message}`;
        if (!j.logs.some((l) => l.indexOf(ev.message) >= 0)) j.logs.push(line);
      }
      if (ev.type === "idle") { src.close(); return; }
      if (ev.type === "stage") j.stage = ev.stage;
      else if (ev.type === "level_start") j.level_progress = { concurrency: ev.concurrency, completed: 0, total: ev.total, run: ev.run };
      else if (ev.type === "request_done") j.level_progress = { concurrency: ev.concurrency, completed: ev.completed, total: ev.total, run: (j.level_progress || {}).run };
      else if (ev.type === "level_done" && ev.summary) {
        j.levels_completed = j.levels_completed || [];
        const sm = Object.assign({}, ev.summary, { run: ev.summary.run || ev.run || (j.stage === "calibrate" ? "short" : undefined) });
        if (!labelLevels(j.levels_completed).some((s) => s.concurrency === sm.concurrency && s.run === (sm.run || "mixed"))) j.levels_completed.push(sm);
      } else if (ev.type === "deploy_done" && ev.endpoint && S.session) S.session.endpoint = ev.endpoint;
      else if (ev.type === "teardown_done" && S.session) S.session.endpoint = null;
      else if (ev.type === "preflight_start") j.preflight = { status: "running", started_at: ev.ts };
      else if (ev.type === "preflight_done") j.preflight = { status: ev.passed ? "passed" : "failed", duration_s: ev.duration_s, durations: ev.durations, message: ev.message };
      else if (ev.type === "cancelling") j.stop_requested_at = j.stop_requested_at || ev.ts;
      else if (ev.type === "complete" || ev.type === "error" || ev.type === "cancelled" || ev.type === "interrupted") {
        src.close();
        S.sse = null;
        await refreshSession();
        if (ev.type === "complete") toast("Benchmark finished. The answer now uses measured throughput.");
        else if (ev.type === "error") toast(`The benchmark failed: ${ev.error || "see the run log"}`, true);
        else if (ev.type === "interrupted") toast("The benchmark process stopped unexpectedly.", true);
        return;
      }
      if (S.tab === "bench") renderBench(); else renderRunPill();
    };
    src.onerror = () => { src.close(); if (S.sse === src) S.sse = null; };
  }

  function startPolling() {
    stopPolling();
    S.tick = setInterval(() => { if (S.tab === "bench") renderBench(); else renderRunPill(); }, 1000);
    S.poll = setInterval(async () => {
      if (!S.name) return;
      try {
        const fresh = await api(`/api/sessions/${enc(S.name)}`);
        if (fresh.name !== S.name) return;
        S.session.endpoint = fresh.endpoint;
        if (fresh.job) {
          const keepLogs = S.job && S.job.logs;
          S.job = Object.assign({}, S.job, fresh.job);
          if ((!fresh.job.logs || !fresh.job.logs.length) && keepLogs) S.job.logs = keepLogs;
          if (fresh.job.status !== "running") { stopPolling(); applySession(fresh, { keepCfg: S.dirty }); return; }
          if (!S.sse) connectSse();
        }
        if (S.tab === "bench") renderBench(); else renderRunPill();
      } catch (_) { /* transient */ }
    }, 3000);
  }
  function stopPolling() {
    clearInterval(S.poll); clearInterval(S.tick);
    S.poll = S.tick = null;
  }
  async function refreshSession() {
    try {
      const fresh = await api(`/api/sessions/${enc(S.name)}`);
      applySession(fresh, { keepCfg: S.dirty });
    } catch (e) { console.warn("Could not refresh the session", e); }
  }

  // ---------------------------------------------------------------- config tab
  function renderConfig() {
    if (!S.session) return;
    const ed = $("yamlEditor");
    if (!S.yamlDirty && ed.value !== (S.session.yaml_text || "")) ed.value = S.session.yaml_text || "";
    ed.classList.toggle("dirty", S.yamlDirty);
    $("btnApplyYaml").disabled = !S.yamlDirty;
    $("btnDiscardYaml").hidden = !S.yamlDirty;
    $("yamlStatus").textContent = S.yamlDirty
      ? "You edited the YAML. Apply it to update the workload, or discard your edits."
      : S.dirty
        ? "This is the last saved version. Save your changes in the rail to update it."
        : isExample(S.name) ? "The example's configuration. It is read-only; applying edits saves a copy." : "The saved configuration for this session. Edit it here or in the rail.";
    const n = enc(S.name);
    $("dlYaml").href = `/api/sessions/${n}/export/workload.yaml`;
    const hasReport = !!S.session.report_md;
    const hasDash = !!S.session.dashboard_data || !!S.session.measured;
    setLink($("dlReport"), hasReport ? `/api/sessions/${n}/export/report.md` : null);
    setLink($("dlDashboard"), hasDash ? `/api/sessions/${n}/export/dashboard.html` : null);
    $("reportDetails").hidden = !hasReport;
    if (hasReport) $("reportPre").textContent = S.session.report_md;
  }
  function setLink(a, href) {
    if (href) { a.href = href; a.removeAttribute("aria-disabled"); a.tabIndex = 0; }
    else { a.removeAttribute("href"); a.setAttribute("aria-disabled", "true"); a.tabIndex = -1; }
  }

  // ---------------------------------------------------------------- sessions
  function isExample(name) {
    if (S.session && S.session.name === name && S.session.example === true) return true;
    const s = S.sessions.find((x) => x.name === name);
    return !!(s && (s.example === true || s.is_demo === true));
  }
  function renderSessionSelect() {
    const sel = $("sessionSelect");
    sel.innerHTML = "";
    S.sessions.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.name;
      o.textContent = s.name + (s.example || s.is_demo ? " (example)" : "") +
        (s.job_status === "running" ? " (benchmark running)" : s.job_endpoint_warning ? " (interrupted: endpoint may be billing)" : s.job_status === "interrupted" ? " (interrupted)" : "");
      sel.appendChild(o);
    });
    if (S.name && !S.sessions.some((s) => s.name === S.name)) {
      const o = document.createElement("option"); o.value = S.name; o.textContent = S.name; sel.appendChild(o);
    }
    if (S.name) sel.value = S.name;
    sel.disabled = !S.sessions.length && !S.name;
  }
  async function loadSessions() {
    try {
      const res = await api("/api/sessions");
      S.sessions = res.sessions || [];
    } catch (e) {
      S.sessions = [];
      toast(`Couldn't list sessions: ${e.message}`, true);
    }
    renderSessionSelect();
  }
  async function loadSession(name) {
    try {
      const s = await api(`/api/sessions/${enc(name)}`);
      applySession(s, { resetUi: true });
      return true;
    } catch (e) {
      toast(`Couldn't open "${name}": ${e.message}`, true);
      return false;
    }
  }

  function applySession(s, opts) {
    opts = opts || {};
    const switching = !S.session || S.session.name !== s.name;
    S.session = s;
    S.name = s.name;
    // A pinned Gemini model the live catalog no longer offers: say so once when the session opens (details under Routing).
    if (switching && (s.gemini_warnings || []).length) setTimeout(() => toast(`${s.gemini_warnings[0]} See Routing.`, true), 400);
    if (!opts.keepCfg || switching) {
      S.cfg = normalizeCfg(clone(s.config || {}));
      S.dirty = false;
    }
    if (switching && S.shape.on) {
      S.shape.on = false; S.shape.stroke = null;
      $("wlFigure").classList.remove("editing"); $("shapeBar").hidden = true; $("wlShape").hidden = true;
      $("btnEditShape").setAttribute("aria-pressed", "false"); $("btnEditShape").textContent = "Edit shape";
      $("wlCaption").textContent = "A typical week of input traffic against the fleet";
    }
    if (opts.resetUi || switching) {
      S.ui = Object.assign({}, DEFAULT_UI);
      const st = s.ui_state || {};
      for (const k in DEFAULT_UI) if (st[k] !== undefined) S.ui[k] = st[k];
      // The saved pricing option (pricing.tier in the config) wins over the older ui_state copy.
      const ct = S.cfg.pricing && S.cfg.pricing.tier;
      if (ct && ct !== "auto") S.ui.pricing = ct;
      S.yamlDirty = false;
      S.models = null; S.hw = null;
      S.drawn = null;
    }
    if (switching || opts.resetUi) S.tpsEdit = undefined;
    S.job = s.job || { status: "idle" };
    const listed = S.sessions.find((x) => x.name === s.name);
    if (listed) { listed.job_status = S.job.status; listed.job_endpoint_warning = !!S.job.endpoint_warning; }
    try { localStorage.setItem("dilemma-last-session", s.name); } catch (_) { /* private mode */ }
    const url = new URL(location.href);
    url.searchParams.set("session", s.name);
    history.replaceState(null, "", url);
    showMain();
    renderSessionSelect();
    populateRail();
    buildData();
    // Fleet overrides are whole replicas of the current hardware.
    if (S.ui.fleetOverride != null && S.data && (+S.ui.fleetOverride % (S.data.replica || 1) !== 0)) S.ui.fleetOverride = null;
    renderSaveState();
    recompute({ animate: false });
    renderConfig();
    renderBench();
    if (S.job.status === "running") {
      if (!S.sse) connectSse();
      if (!S.poll) startPolling();
    } else stopPolling();
    if (!S.data && !isExample(s.name) && !S.autoEstimated[s.name] && S.job.status !== "running") {
      S.autoEstimated[s.name] = true;
      reestimate();
    }
  }

  function markDirty(uiOnly) {
    S.dirty = true;
    void uiOnly;
    renderSaveState();
  }
  function renderSaveState() {
    const st = $("saveState"), btn = $("btnSave");
    const ex = S.name && isExample(S.name);
    btn.textContent = ex ? "Save a copy" : "Save";
    btn.disabled = !S.session || (!S.dirty && !ex);
    if (S.dirty) { st.textContent = "Unsaved changes"; st.classList.add("dirty"); }
    else { st.textContent = ex ? "Example, read-only" : S.session ? "Saved" : ""; st.classList.remove("dirty"); }
    if (S.tab === "config") renderConfig();
  }

  function existingNames() { return new Set(S.sessions.map((s) => s.name)); }
  function suggestName(base) {
    const names = existingNames();
    base = (base || "my-workload").replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "my-workload";
    if (!names.has(base)) return base;
    for (let i = 2; i < 1000; i++) if (!names.has(`${base}-${i}`)) return `${base}-${i}`;
    return base + "-" + Date.now();
  }
  const validName = (v) => {
    const clean = v.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
    if (!clean) return "Use letters, numbers, dashes or underscores.";
    if (existingNames().has(clean)) return `A session named "${clean}" already exists. Pick another name.`;
    return "";
  };

  // Example sessions are read-only: any write first creates a copy.
  async function ensureWritable() {
    if (!S.session) return false;
    if (!isExample(S.name)) return true;
    const name = await dialog({
      title: "Save a copy to make changes",
      body: "<p>The example is read-only. Your changes go into a new session you can keep editing.</p>",
      input: { label: "Name of the copy", value: suggestName("my-workload"), validate: validName },
      okLabel: "Save copy",
    });
    if (!name) return false;
    try {
      const res = await postJSON(`/api/sessions/${enc(name)}`, { config: Object.assign(collectCfg(), { name }), ui_state: uiState(), copy_of: S.name });
      const keepUi = Object.assign({}, S.ui);
      S.autoEstimated[res.name] = true;
      await loadSessions();
      applySession(res, { resetUi: true });
      S.ui = keepUi;
      recompute({ animate: false });
      toast(`Saved a copy as "${res.name}".`);
      return true;
    } catch (e) {
      toast(e.status === 409 ? `A session named "${name}" already exists. Pick another name.` : `Couldn't save a copy: ${e.message}`, true);
      return false;
    }
  }

  async function save() {
    if (!S.session) return;
    if (isExample(S.name)) {
      const ok = await ensureWritable();
      if (ok && !S.data) reestimate();
      return;
    }
    const tpsEdit = S.tpsEdit;
    $("btnSave").disabled = true;
    try {
      const res = await postJSON(`/api/sessions/${enc(S.name)}`, { config: collectCfg(), ui_state: uiState() });
      applySession(res, { resetUi: false });
      toast("Saved.");
      if (tpsEdit !== undefined) {
        const ok = await reestimate(tpsEdit > 0 ? { tps: tpsEdit } : { clearOverride: true });
        if (ok) S.tpsEdit = undefined;
      }
      await loadSessions();
    } catch (e) {
      toast(`Couldn't save: ${e.message}`, true);
      renderSaveState();
    }
  }

  function setBusy(msg) {
    S.busy = msg || "";
    $("busyNote").textContent = S.busy;
    $("verdict").classList.toggle("stale", !!S.busy);
    document.querySelectorAll("[data-needs-idle]").forEach((b) => { b.disabled = !!S.busy; });
    if (S.cfg) renderPrecision();
  }

  // Re-plan on the server (model or hardware changed). Writes workload.yaml + plan + estimate.
  async function reestimate(opts) {
    opts = opts || {};
    if (!(await ensureWritable())) return false;
    setBusy("Updating the estimate…");
    if (!S.data) renderNoData();
    try {
      const body = { config: collectCfg() };
      if (opts.tps > 0) body.tps_per_gpu = +opts.tps;
      if (opts.clearOverride) body.clear_override = true;
      if (opts.force) body.force_estimate = true;
      const res = await postJSON(`/api/sessions/${enc(S.name)}/estimate`, body);
      const keepUi = Object.assign({}, S.ui);
      setBusy("");
      applySession(res, { resetUi: false });
      S.ui = keepUi;
      // persist rail-only state alongside
      postJSON(`/api/sessions/${enc(S.name)}`, { config: collectCfg(), ui_state: uiState() }).then((s2) => { S.session = Object.assign(S.session, { yaml_text: s2.yaml_text, ui_state: s2.ui_state }); renderConfig(); }).catch(() => {});
      recompute({ animate: true });
      return true;
    } catch (e) {
      setBusy("");
      toast(`Couldn't update the estimate: ${e.message}. Check the inputs and try again.`, true);
      if (!S.data) renderNoData(); else recompute();
      return false;
    }
  }

  // ---------------------------------------------------------------- rail binding
  function readInput(el) {
    const kind = el.dataset.kind;
    if (el.type === "checkbox") return el.checked;
    if (kind === "str") return el.value;
    if (kind === "strnull") return el.value.trim() || null;
    if (kind === "numnull") { const n = parseFloat(el.value); return isFinite(n) && n > 0 ? n : null; }
    if (el.tagName === "SELECT" && kind !== "num") return el.value;
    const v = parseFloat(el.value);
    if (!isFinite(v)) return undefined;
    return el.dataset.scale ? v / +el.dataset.scale : v;
  }
  function writeInput(el, v) {
    if (el.type === "checkbox") { el.checked = !!v; return; }
    if (el.type === "radio") { el.checked = normRes(v) === +el.value; return; }
    if (v == null) { el.value = ""; return; }
    if (el.dataset.scale) v = Math.round(v * +el.dataset.scale * 100) / 100;
    el.value = v;
  }
  // "65B", "1.2 T", "800M", "65,000,000,000" -> tokens
  function parseTokens(v) {
    const m = String(v || "").trim().replace(/,/g, "").match(/^(\d*\.?\d+)\s*([kmbt])?/i);
    if (!m) return null;
    const mult = { k: 1e3, m: 1e6, b: 1e9, t: 1e12 }[(m[2] || "").toLowerCase()] || 1;
    const x = parseFloat(m[1]) * mult;
    return x > 0 ? x : null;
  }
  const UI_LIMITS = { offPeak: [1, 10], growth: [0.1, 100] };
  function updateOutputs(except) {
    const set = (id, txt) => { const o = $(id); if (o) o.textContent = txt; };
    const c = S.cfg, t = c.traffic, r = c.requests;
    const pkW = presetOf(t);
    set("o-weekend", Math.round((pkW ? TRAFFIC_PRESETS[pkW].weekend : (+t.weekend_factor || 0)) * 100) + "% of weekdays");
    set("o-sysHit", Math.round((+r.system_prompt_cache_hit || 0) * 100) + "%");
    set("o-tputMult", "×" + (+S.ui.tputMult || 1).toFixed(2));
    // Slider + number twins show the same value; never rewrite the field being typed in.
    ["offPeak", "growth"].forEach((k) => {
      document.querySelectorAll(`#railForm [data-ui="${k}"]`).forEach((el) => {
        if (el === except) return;
        const v = +S.ui[k] || 1;
        el.value = el.type === "range" ? Math.max(+el.min, Math.min(+el.max, v)) : v;
      });
    });
    // Volume anchor: weekly average or monthly input tokens; the other is shown derived.
    const monthly = monthlyAnchor(t);
    $("anchorAvg").setAttribute("aria-pressed", String(!monthly));
    $("anchorMonthly").setAttribute("aria-pressed", String(monthly));
    $("rowAvg").hidden = monthly;
    $("rowMonthly").hidden = !monthly;
    const avg = +t.average_input_tps || 0;
    if (except !== $("f-monthly")) $("f-monthly").value = monthly ? fmtTok(+t.monthly_input_tokens) : "";
    const custom = isCustom(t);
    const avgNow = custom && S.data && S.data.profile ? meanOf(S.data.profile) : avg;
    $("anchorDerived").textContent = monthly
      ? `A weekly average of ${fmtTok(avgNow)} tok/s. Overflow and MaaS costs follow this monthly total (before the growth what-ifs).`
      : `About ${fmtTok(avgNow * SEC_MONTH)} input tokens a month.`;
    // Thinking: answer + share × thinking tokens.
    const ans = +r.avg_output_tokens || 0, sh = Math.max(0, Math.min(1, +r.thinking_share || 0)), th = +r.avg_thinking_tokens || 0;
    $("thinkDerived").textContent = sh > 0 && th > 0
      ? `Effective output: ${fmtInt(ans)} + ${pct(sh)} × ${fmtInt(th)} = ${fmtInt(ans + sh * th)} tokens per request. Output tokens exclude thinking.`
      : `Effective output: ${fmtInt(ans)} tokens per request. Output tokens exclude thinking; add it with the two fields above.`;
    renderPriceSummaries();
  }
  function renderPriceSummaries() {
    if (!S.cfg) return;
    const m = S.cfg.maas;
    const d = (x) => (x == null || x === "" || !isFinite(+x)) ? "—" : "$" + (+x < 1 ? (+x).toFixed(3).replace(/0$/, "") : (+x).toFixed(2).replace(/\.00$/, ""));
    $("maasSum").textContent = `${d(m.input_per_m)} in · ${d(m.output_per_m)} out per M · ${m.max_rps != null ? fmtRps(+m.max_rps) : "—"} req/s limit`;
    const g = M.getGeminiPrices ? M.getGeminiPrices(S.data || { gemini: geminiCatalog() }, S.p || (S.data ? baseParams() : { gemOverrides: S.ui.gem || {} })) : {};
    const short = (k) => ({ gemini_flash_lite: "Lite", gemini_flash: "Flash", gemini_pro: "Pro", gemini_4: "G4" }[k]);
    const parts = GEMINI_KEYS.map((k) => (g[k] ? `${short(k)} ${d(g[k].in)}/${d(g[k].out)}` : null)).filter(Boolean);
    const n = geminiModels().filter((m) => !m.deprecated).length;
    $("gemSum").textContent = parts.length ? parts.join(" · ") + " per M, in/out" + (n > 4 ? ` · ${n} models` : "") : "";
    $("gemSum").title = GEMINI_KEYS.map((k) => targetName(k)).join(", ");
  }
  function populateRail() {
    const c = S.cfg;
    document.querySelectorAll("#railForm [data-path]").forEach((el) => writeInput(el, getPath(c, el.dataset.path)));
    document.querySelectorAll("#railForm [data-ui]").forEach((el) => {
      const k = el.dataset.ui;
      if (k === "pricing") return;
      writeInput(el, S.ui[k]);
    });
    $("f-route").value = c.hybrid.route_long ? (c.hybrid.route_long_target || "gemini_flash_lite") : "fleet";
    $("f-sysPrompts").value = (c.prompts.custom_system_prompts || []).join("\n---\n");
    renderMyPrompts();
    $("f-conc").value = (c.loadtest.concurrency || []).join(", ");
    $("f-extraArgs").value = (c.deployment.extra_args || []).join(", ");
    $("f-preflight").checked = c.loadtest.preflight !== false;
    renderImageField();
    const st = S.session && S.session.throughput;
    $("u-tpsOverride").value = S.tpsEdit !== undefined ? (S.tpsEdit || "") : (st && st.source === "override" ? Math.round(st.per_chip) : "");
    const tpNow = throughput();
    $("u-tpsOverride").placeholder = tpNow ? String(Math.round(tpNow.per_chip)) : "";
    classPlaceholders();
    const csv = !!c.traffic.hourly_csv;
    const custom = isCustom(c.traffic);
    $("csvNotice").hidden = !csv;
    if (csv) $("csvName").textContent = String(c.traffic.hourly_csv).split("/").pop();
    $("customNotice").hidden = !custom;
    ["f-avg", "f-monthly", "anchorAvg", "anchorMonthly"].forEach((id) => { $(id).disabled = csv; });
    ["f-peak", "f-floor", "f-busy0", "f-busy1", "f-weekend"].forEach((id) => { $(id).disabled = csv || custom; });
    // Traffic shape preset: peak and quietest hour are derived (typing one switches back to custom); the busy
    // window and weekend load belong to the preset.
    const pk = presetOf(c.traffic);
    $("f-preset").value = pk || "";
    $("f-preset").disabled = csv;
    ["f-busy0", "f-busy1", "f-weekend"].forEach((id) => { if (pk) $(id).disabled = true; });
    if (pk) {
      syncPresetFields(c.traffic);
      $("f-peak").value = Math.round(c.traffic.peak_input_tps);
      $("f-floor").value = Math.round(c.traffic.floor_input_tps);
      // Busy window and weekend load shown as the preset's (approximately); they only apply to your own numbers.
      const a = presetApprox(pk);
      $("f-busy0").value = a.busy[0]; $("f-busy1").value = a.busy[1];
      writeInput($("f-weekend"), a.weekend);
    }
    renderPresetLine();
    if (custom) {
      const prof = currentProfile() || c.traffic.custom_hourly;
      $("f-peak").value = Math.round(Math.max.apply(null, prof));
      $("f-floor").value = Math.round(Math.min.apply(null, prof));
      if (!monthlyAnchor(c.traffic)) $("f-avg").value = Math.round(meanOf(prof));
      $("customNoticeText").textContent = `Using the week you drew: peak ${fmtTok(Math.max.apply(null, prof))}, average ${fmtTok(meanOf(prof))}, quietest hour ${fmtTok(Math.min.apply(null, prof))} tok/s.`;
    }
    populateTargetSelects();
    $("projectInput").value = c.project || "";
    if (c.region && $("regionSelect").querySelector(`option[value="${c.region}"]`)) $("regionSelect").value = c.region;
    updateProjectLabel();
    populatePricingSelect();
    populateGeminiPlaceholders();
    updateOutputs();
  }
  // Peak / quietest hour in the config follow the preset week (as config.normalize writes them).
  function syncPresetFields(t) {
    const pk = presetOf(t);
    if (!pk) return null;
    const prof = currentProfile();
    if (!prof) return null;
    t.peak_input_tps = Math.round(Math.max.apply(null, prof));
    t.floor_input_tps = Math.round(Math.min.apply(null, prof));
    return prof;
  }
  function renderPresetLine() {
    const t = S.cfg.traffic, pk = presetOf(t), el = $("presetDerived");
    if (!pk) {
      el.textContent = isCustom(t) ? "Using the week you drew. Pick a shape to replace it at the same volume." : "";
      return;
    }
    const sh = presetShape(pk), m = sh.reduce((a, v) => a + v, 0) / sh.length;
    const ratio = Math.max.apply(null, sh) / m;
    el.textContent = `Same monthly volume, ${PRESET_PHRASE[pk]}: peak ${ratio.toFixed(1)}× the average.`;
  }
  // Busy window (UTC) and weekend load that approximate a preset with the simple parametric shape.
  function presetApprox(pk) {
    const P = TRAFFIC_PRESETS[pk], sh = presetShape(pk).slice(0, 24);
    const lo = Math.min.apply(null, sh), hi = Math.max.apply(null, sh), mid = lo + (hi - lo) / 2;
    let best = [13, 20], bestLen = 0;
    for (let s0 = 0; s0 < 24; s0++) {            // longest run of hours above half the peak (wrapping midnight)
      let n = 0;
      while (n < 24 && sh[(s0 + n) % 24] >= mid) n++;
      if (n > bestLen && n < 20) { bestLen = n; best = [s0, s0 + n]; }
    }
    if (best[1] > 24) best = [best[0], 24];   // the busy-window fields don't wrap midnight
    return { busy: best, weekend: P.weekend };
  }
  // Choose a traffic shape: a preset keeps the weekly average (or the monthly total); "" = your own numbers.
  function applyPreset(key) {
    if (!S.cfg) return;
    const t = S.cfg.traffic, prev = presetOf(t);
    if (key && TRAFFIC_PRESETS[key]) {
      if (isCustom(t)) {
        if (!monthlyAnchor(t)) t.average_input_tps = Math.round(meanOf(currentProfile() || t.custom_hourly));
        t.custom_hourly = null;
      }
      t.preset = key;
      syncPresetFields(t);
    } else {
      if (!prev) return;
      leavePreset(t, prev);
    }
    populateRail();
    markDirty();
    scheduleRecompute();
  }
  // Back to the parametric week: keep the peak, the quietest hour and the volume, and pick the busy window and
  // weekend load closest to the preset's shape that the simple week can reach. keepTyped: a peak or quietest hour
  // the user just typed stays as typed.
  function leavePreset(t, pk, keepTyped) {
    if (!keepTyped) syncPresetFields(t);
    const P = TRAFFIC_PRESETS[pk], day = presetShape(pk).slice(0, 24);
    const a = presetApprox(pk);
    t.preset = null;
    const avg = monthlyAnchor(t) ? +t.monthly_input_tokens / SEC_MONTH : +t.average_input_tps;
    const ok = (busy, w) => { try { parametric(+t.peak_input_tps, avg, +t.floor_input_tps, busy, w); return true; } catch (_) { return false; } };
    const peakH = day.indexOf(Math.max.apply(null, day));
    const len0 = a.busy[1] - a.busy[0];
    const lens = [];
    for (let n = 3; n <= 19; n++) lens.push(n);
    lens.sort((x, y) => Math.abs(x - len0) - Math.abs(y - len0) || y - x);
    const tries = [[a.busy, P.weekend]];
    for (const n of lens) {
      const s0 = Math.max(0, Math.min(24 - n, Math.round(peakH + 0.5 - n / 2)));
      for (const w of [P.weekend, 0.15, 0.08, 0.05, 0.02, 0.3, 0.5, 0.7, 0.9]) tries.push([[s0, s0 + n], w]);
    }
    const hit = tries.find(([b, w]) => ok(b, w)) || tries[0];
    t.busy_hours_utc = hit[0].slice();
    t.weekend_factor = hit[1];
  }
  function populatePricingSelect() {
    const sel = $("u-pricing");
    const prices = (S.data && S.data.prices) || pickTiers((S.session && S.session.dashboard_data && S.session.dashboard_data.prices) || (S.session && S.session.plan && S.session.plan.chosen && S.session.plan.chosen.prices));
    const keys = Object.keys(prices || {}).sort((a, b) => TIER_ORDER.indexOf(a) - TIER_ORDER.indexOf(b));
    sel.innerHTML = `<option value="auto">Cheapest option</option>` + keys.map((k) => `<option value="${k}">${esc(TIER_LABELS[k] || k)}</option>`).join("");
    sel.value = S.ui.pricing && keys.indexOf(S.ui.pricing) >= 0 ? S.ui.pricing : "auto";
  }
  // Effective price override for a Gemini model id: per-model (S.ui.gem) first, then the older per-role fields of a
  // role alias that resolves to it (liteIn …), as model.js getGeminiPrices applies them.
  const LEGACY_GEM = { gemini_flash_lite: ["liteIn", "liteOut"], gemini_flash: ["flashIn", "flashOut"], gemini_pro: ["proIn", "proOut"], gemini_4: ["g4In", "g4Out"] };
  function gemOverride(id, f) {
    const o = S.ui.gem && S.ui.gem[id];
    if (o && o[f] != null && isFinite(+o[f])) return +o[f];
    const g = geminiCatalog();
    for (const a of GEMINI_KEYS) {
      if (a !== id && !(g[a] && g[a].model_id === id)) continue;
      const v = S.ui[LEGACY_GEM[a][f === "in" ? 0 : 1]];
      if (v != null && isFinite(+v)) return +v;
    }
    return null;
  }
  function setGemOverride(id, f, v) {
    S.ui.gem = Object.assign({}, S.ui.gem || {});
    const o = Object.assign({}, S.ui.gem[id] || {});
    o[f] = isFinite(v) && v >= 0 ? v : null;
    if (o.in == null && o.out == null) delete S.ui.gem[id]; else S.ui.gem[id] = o;
    // The per-model value replaces an older per-role override of the same model.
    const g = geminiCatalog();
    for (const a of GEMINI_KEYS) if (a === id || (g[a] && g[a].model_id === id)) S.ui[LEGACY_GEM[a][f === "in" ? 0 : 1]] = null;
  }
  // Rail price table: the models the role aliases resolve to and any pinned target first; "Show all" adds every other
  // offered model (deprecated ones only when selected), grouped by family. Every price is an editable override.
  function populateGeminiPlaceholders() {
    const g = geminiCatalog();
    const models = geminiModels();
    const h = (S.cfg && S.cfg.hybrid) || {};
    const pinned = new Set([h.route_long_target, h.peak_overflow_target, h.failover_target].filter((t) => t && GEMINI_KEYS.indexOf(t) < 0));
    const roleOf = {};
    const roleTag = (l) => { const t = l.replace(/^Gemini\s+(\d+\s+)?/, ""); return t.charAt(0).toUpperCase() + t.slice(1); };
    for (const a of geminiAliases()) if (a.model_id) (roleOf[a.model_id] = roleOf[a.model_id] || []).push(roleTag(a.label));
    const featured = (m) => roleOf[m.id] || pinned.has(m.id) || (S.ui.gem && S.ui.gem[m.id]);
    const shown = models.filter((m) => (!m.deprecated || pinned.has(m.id)) && (S.gemAll || featured(m)));
    const hidden = models.filter((m) => !m.deprecated && !featured(m)).length;
    const grid = $("gemGrid");
    const fmtP = (v) => (v == null ? "" : String(+(+v).toFixed(4)));
    let html = '<span class="gem-h" aria-hidden="true"></span><span class="gem-h" aria-hidden="true">Input</span><span class="gem-h" aria-hidden="true">Output</span>';
    let group = null;
    for (const m of shown) {
      if (S.gemAll && m.group !== group) { group = m.group; html += `<span class="gem-group">${esc(group || "")}</span>`; }
      const nid = "gn-" + m.id.replace(/[^a-z0-9-]/gi, "_");
      const tags = (roleOf[m.id] || []).map((r) => `<span class="gem-role" title="Role alias: the newest ${esc(r)}">${esc(r)}</span>`).join("")
        + (m.stage === "preview" ? '<span class="gem-tag">Preview</span>' : "") + (m.deprecated ? '<span class="gem-tag gem-dep">Deprecated</span>' : "")
        + (m.stage === "unlisted" ? '<span class="gem-tag" title="Billed but not listed in Model Garden yet">New</span>' : "")
        + (m.long_context ? `<span class="gem-tag" title="Prompts over ${fmtInt(m.long_context.threshold)} tokens: $${fmtP(m.long_context.input_per_m)} in / $${fmtP(m.long_context.output_per_m)} out per M">&gt;${Math.round(m.long_context.threshold / 1000)}K tier</span>` : "");
      const cell = (f, base, step) => {
        const ov = gemOverride(m.id, f);
        return `<span class="unit-input"><span class="unit pre">$</span><input type="number" data-gem="${esc(m.id)}" data-f="${f}" min="0" step="${step}" inputmode="decimal" value="${fmtP(ov != null ? ov : base)}" placeholder="${fmtP(base)}" class="${ov != null ? "is-over" : ""}" aria-labelledby="${nid} gl-${f}"></span>`;
      };
      html += `<span class="gem-name" id="${nid}">${esc(m.name.replace(/^Gemini\s+/, ""))}${tags ? " " + tags : ""}</span>` + cell("in", m.input_per_m, m.input_per_m < 1 ? 0.01 : 0.05) + cell("out", m.output_per_m, 0.05);
    }
    grid.innerHTML = html;
    const all = $("btnGemAll");
    all.hidden = !hidden && !S.gemAll;
    all.textContent = S.gemAll ? "Show fewer" : `Show all (${hidden} more)`;
    all.setAttribute("aria-expanded", S.gemAll ? "true" : "false");
    const meta = (S.catalog && S.catalog.gemini_meta) || {};
    const src = meta.source || ((g.gemini_flash_lite && g.gemini_flash_lite.source) || "");
    const when = meta.updated_at ? new Date(meta.updated_at * 1000) : null;
    const whenTxt = when ? when.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
    $("geminiSource").textContent = src === "billing-catalog"
      ? `Gemini prices from the Cloud Billing Catalog${whenTxt ? ", updated " + whenTxt : ""}${meta.model_garden === false ? " (launch stages unavailable)" : ""}${meta.refreshing ? " · refreshing…" : ""}.`
      : src ? "Reference prices. Sign in to read the Cloud Billing Catalog." : "";
    renderPriceSummaries();
  }
  // Target selects: role aliases first ("Gemini Flash-Lite (latest: Gemini 3.5 Flash-Lite)"), then every offered model
  // grouped by family; previews labelled; deprecated ones only when selected. A pinned model the catalog no longer
  // lists stays visible as "(no longer offered)".
  function geminiOptionsHTML(current) {
    const models = geminiModels();
    const g = geminiCatalog();
    let html = '<optgroup label="Newest of each kind">' + geminiAliases().filter((a) => g[a.key] || a.key === current)
      .map((a) => `<option value="${a.key}">${esc(a.label)} (latest: ${esc(a.name)}${stageTag(geminiModel(a.model_id))})</option>`).join("") + "</optgroup>";
    const groups = [];
    for (const m of models) {
      if (m.deprecated && m.id !== current) continue;
      let gr = groups.find((x) => x.label === m.group);
      if (!gr) { gr = { label: m.group || "Other", items: [] }; groups.push(gr); }
      gr.items.push(m);
    }
    for (const gr of groups) html += `<optgroup label="${esc(gr.label)}">` + gr.items.map((m) => `<option value="${esc(m.id)}">${esc(m.name + stageTag(m))}</option>`).join("") + "</optgroup>";
    if (isGeminiTarget(current) && GEMINI_KEYS.indexOf(current) < 0 && !models.some((m) => m.id === current)) {
      html += `<option value="${esc(current)}">${esc(current)} (no longer offered: priced as ${esc(targetName(current))})</option>`;
    }
    return html;
  }
  function populateTargetSelects() {
    const h = S.cfg.hybrid;
    const route = h.route_long ? (h.route_long_target || "gemini_flash_lite") : "fleet";
    $("f-route").innerHTML = '<option value="fleet">The fleet, with everything else</option>' + geminiOptionsHTML(h.route_long_target) + '<option value="open_maas">Open-model MaaS</option>';
    $("f-overflow").innerHTML = '<option value="open_maas">Open-model MaaS</option>' + geminiOptionsHTML(h.peak_overflow_target);
    $("f-failover").innerHTML = geminiOptionsHTML(h.failover_target);
    const has = (id, v) => !!v && !!$(id).querySelector(`option[value="${CSS.escape(v)}"]`);
    $("f-route").value = has("f-route", route) ? route : "fleet";
    $("f-overflow").value = has("f-overflow", h.peak_overflow_target) ? h.peak_overflow_target : "open_maas";
    $("f-failover").value = h.failover_target && h.failover_target !== "open_maas" && has("f-failover", h.failover_target) ? h.failover_target : "gemini_flash_lite";
    renderGeminiWarnings();
  }
  // Pinned models in the saved config that the live catalog no longer offers (the server replaced them with their
  // role alias) or that are deprecated, with the replacement.
  function renderGeminiWarnings() {
    const box = $("gemWarn");
    if (!box) return;
    // Server warnings (a pinned model replaced on load) last until that field is changed.
    const saved = (S.session && S.session.config && S.session.config.hybrid) || {};
    const msgs = ((S.session && S.session.gemini_warnings) || []).filter((w) => {
      const m = /\(hybrid\.(\w+)\)/.exec(w);
      return !m || ((S.cfg && S.cfg.hybrid) || {})[m[1]] === saved[m[1]];
    });
    const live = ((S.catalog && S.catalog.gemini_meta) || {}).source === "billing-catalog";
    const h = (S.cfg && S.cfg.hybrid) || {};
    const fields = { route_long_target: "Prompts over 32K", peak_overflow_target: "Peak overflow", failover_target: "Failover" };
    for (const f in fields) {
      const t = h[f];
      if (!live || !isGeminiTarget(t) || GEMINI_KEYS.indexOf(t) >= 0) continue;
      const m = geminiModel(t);
      const a = geminiAliases().find((x) => x.key === (M.aliasForId ? M.aliasForId(t) : "gemini_flash_lite"));
      const repl = a ? `${a.label} (latest: ${a.name})` : "the newest model of its kind";
      if (!m) msgs.push(`${fields[f]}: ${t} is no longer offered; priced as ${repl}.`);
      else if (m.deprecated) msgs.push(`${fields[f]}: ${m.name} is deprecated; consider ${repl}.`);
    }
    const uniq = msgs.filter((x, i) => msgs.indexOf(x) === i);
    box.hidden = !uniq.length;
    box.innerHTML = uniq.map((x) => `<p>${icon("i-info", "ico-sm")} ${esc(x)}</p>`).join("");
  }
  async function refreshGemini() {
    const b = $("btnGemRefresh");
    if (b) { b.disabled = true; b.textContent = "Refreshing…"; }
    try {
      const c = S.cfg || {};
      const res = await postJSON("/api/gemini/refresh", { project: c.project, region: c.region });
      if (S.catalog) Object.assign(S.catalog, { gemini: res.gemini, gemini_models: res.gemini_models, gemini_aliases: res.gemini_aliases, gemini_meta: res.gemini_meta });
      if (S.session) { buildData(); populateTargetSelects(); populateGeminiPlaceholders(); recompute({ animate: false }); }
      toast(res.gemini_meta && res.gemini_meta.source === "billing-catalog" ? `Gemini prices updated: ${res.gemini_models.filter((m) => !m.deprecated).length} models.` : "Couldn't read the Cloud Billing Catalog; showing reference prices.", !(res.gemini_meta && res.gemini_meta.source === "billing-catalog"));
    } catch (e) { toast(`Couldn't refresh Gemini prices: ${e.message}`, true); }
    if (b) { b.disabled = false; b.textContent = "Refresh"; }
  }
  const parseList = (s, num) => s.split(",").map((x) => x.trim()).filter(Boolean).map((x) => (num ? parseInt(x, 10) : x)).filter((x) => (num ? x > 0 : true));

  function onRailInput(e) {
    const el = e.target;
    if (!S.cfg) return;
    if (el.type === "radio" && !el.checked) return;
    if (el.id === "f-monthly" && e.type === "input") return; // applied on change: "65B" passes through "6" while typing
    let modelStale = false;
    if (el.id === "f-preset") { applyPreset(el.value); return; }
    let leftPreset = false;
    if (el.dataset.path) {
      const v = readInput(el);
      if (v === undefined) return;
      const pk0 = presetOf(S.cfg.traffic);
      setPath(S.cfg, el.dataset.path, v);
      if (pk0 && (el.dataset.path === "traffic.peak_input_tps" || el.dataset.path === "traffic.floor_input_tps")) {
        leavePreset(S.cfg.traffic, pk0, true);   // typing your own peak or quietest hour means your own numbers
        leftPreset = true;
      }
      const p = el.dataset.path;
      if (p === "traffic.average_input_tps" && presetOf(S.cfg.traffic)) syncPresetFields(S.cfg.traffic);
      if (p.indexOf("requests.pct_over") === 0) syncBuckets(S.cfg);
      if (p.indexOf("use_case") === 0 || p === "requests.thinking_share" || p === "requests.avg_thinking_tokens" || p === "constraints.latency_budget_s" || p.indexOf("requests.pct_over") === 0) modelStale = true;
      // Typing a weekly average with a drawn week rescales the drawing.
      if (p === "traffic.average_input_tps" && isCustom(S.cfg.traffic) && v > 0) {
        const m = meanOf(S.cfg.traffic.custom_hourly);
        if (m > 0) S.cfg.traffic.custom_hourly = S.cfg.traffic.custom_hourly.map((x) => x * v / m);
      }
    } else if (el.dataset.ui) {
      const k = el.dataset.ui;
      if (k === "pricing") { S.ui.pricing = el.value; S.cfg.pricing = Object.assign({}, S.cfg.pricing, { tier: el.value || "auto" }); }
      else if (UI_LIMITS[k]) {
        const v = parseFloat(el.value);
        if (!isFinite(v)) return;
        const [lo, hi] = UI_LIMITS[k];
        S.ui[k] = Math.max(lo, Math.min(hi, v));
      } else {
        const v = parseFloat(el.value);
        S.ui[k] = isFinite(v) ? v : null;
      }
    } else if (el.id === "f-monthly") {
      const v = parseTokens(el.value);
      if (v == null) return;
      setMonthly(v);
    } else if (el.id === "f-route") {
      S.cfg.hybrid.route_long = el.value !== "fleet";
      if (el.value !== "fleet") S.cfg.hybrid.route_long_target = el.value;
    } else if (el.id === "f-sysPrompts") {
      S.cfg.prompts.custom_system_prompts = el.value.split(/\n---\n/).map((x) => x.trim()).filter(Boolean);
    } else if (el.id === "f-conc") {
      const l = parseList(el.value, true);
      if (l.length) S.cfg.loadtest.concurrency = l;
    } else if (el.id === "f-extraArgs") {
      S.cfg.deployment.extra_args = parseList(el.value, false);
    } else if (el.id === "f-preflight") {
      S.cfg.loadtest.preflight = el.checked;
      renderBenchServing();
    } else if (el.id === "f-image") {
      S.imageChoice = { name: S.name, v: el.value };
      if (el.value === "custom") {
        $("imageCustomField").hidden = false;
        const v = $("f-imageCustom").value.trim();
        if (!validImage(v)) { if (v) setImageError(imageErrorText()); return; }
        S.cfg.deployment.image = v;
      } else {
        const p = pinnedImage();
        S.cfg.deployment.image = el.value === "pinned" && p ? p.image : null;
        renderImageField();
      }
      renderBenchServing();
    } else if (el.id === "f-imageCustom") {
      const v = el.value.trim();
      if (!validImage(v)) {
        // flag it once there is enough to judge (on change, or while typing something that already can't match)
        setImageError(v && (e.type === "change" || /\s|[A-Z]/.test(v.split(/[:@]/)[0])) ? imageErrorText() : "");
        return;
      }
      setImageError("");
      S.cfg.deployment.image = v;
      renderBenchServing();
    } else if (el.id === "u-tpsOverride") {
      const v = parseFloat(el.value);
      S.tpsEdit = v > 0 ? v : null;
      buildData();
    } else if (el.dataset.gem) {
      setGemOverride(el.dataset.gem, el.dataset.f, parseFloat(el.value));
      el.classList.toggle("is-over", gemOverride(el.dataset.gem, el.dataset.f) != null);
      renderPriceSummaries();
    } else return;
    if (modelStale) S.models = null;
    if (el.id === "f-monthly" && presetOf(S.cfg.traffic)) syncPresetFields(S.cfg.traffic);
    if (leftPreset || (presetOf(S.cfg.traffic) && el.id === "f-monthly")) {
      // Keep the field being typed in; refresh the derived ones.
      const keep = el.value;
      populateRail();
      el.value = keep;
    }
    updateOutputs(el);
    // A new Gemini target: the price table lists pinned models, and a replaced-model warning ends.
    if (el.id === "f-route" || String(el.dataset.path || "").indexOf("hybrid.") === 0) { renderGeminiWarnings(); populateGeminiPlaceholders(); }
    markDirty();
    scheduleRecompute();
    if (S.cfg.prompts.source === "file") renderMyPrompts();   // repeats and answer lengths follow the settings
  }
  function imageErrorText() {
    return "Use an Artifact Registry or Container Registry image, e.g. us-docker.pkg.dev/my-project/my-repo/vllm:tag or gcr.io/my-project/vllm:tag. Not saved yet.";
  }
  function setMonthly(v) {
    const t = S.cfg.traffic;
    t.monthly_input_tokens = Math.round(v);
    t.average_input_tps = Math.round(v / SEC_MONTH); // keeps older servers (and the parametric week) consistent
  }
  function setAnchor(kind) {
    if (!S.cfg) return;
    const t = S.cfg.traffic;
    if (kind === "monthly" && !monthlyAnchor(t)) {
      const avg = isCustom(t) ? meanOf(t.custom_hourly) : (+t.average_input_tps || 0);
      if (avg > 0) setMonthly(avg * SEC_MONTH); else t.monthly_input_tokens = null;
      if (!t.monthly_input_tokens) t.monthly_input_tokens = 1e9;
    } else if (kind === "avg" && monthlyAnchor(t)) {
      t.average_input_tps = Math.round(+t.monthly_input_tokens / SEC_MONTH);
      if (isCustom(t)) {
        const prof = currentProfile();
        if (prof) t.custom_hourly = prof.map((x) => Math.round(x));
      }
      t.monthly_input_tokens = null;
    } else return;
    populateRail();
    markDirty();
    scheduleRecompute();
    (kind === "monthly" ? $("f-monthly") : $("f-avg")).focus();
  }

  // ---------------------------------------------------------------- uploads
  function summarizeIngest(ing) {
    const lines = [];
    for (const f of ing.files || []) {
      const bits = [f.detected_type];
      if (f.rows) bits.push(`${fmtInt(f.rows)} rows`);
      if (f.count) bits.push(`${fmtInt(f.count)} items`);
      lines.push(`${f.filename}: ${bits.filter(Boolean).join(", ")}`);
    }
    if (ing.traffic && ing.traffic.peak_input_tps) lines.push(`Traffic: peak ${fmtTok(ing.traffic.peak_input_tps)} tok/s, average ${fmtTok(ing.traffic.average_input_tps)} tok/s`);
    if (ing.requests && ing.requests.avg_input_tokens) lines.push(`Prompts: ${fmtInt(ing.requests.avg_input_tokens)} tokens in, ${fmtInt(ing.requests.avg_output_tokens)} out on average`);
    if (ing.prompts && ing.prompts.num_system_prompts) lines.push(`${ing.prompts.num_system_prompts} system prompts, about ${fmtInt(ing.prompts.avg_system_prompt_tokens)} tokens each`);
    return lines;
  }
  function applyIngest(c, ing) {
    if (ing.traffic && ing.traffic.peak_input_tps) {
      c.traffic.peak_input_tps = ing.traffic.peak_input_tps;
      c.traffic.average_input_tps = ing.traffic.average_input_tps;
      c.traffic.floor_input_tps = ing.traffic.floor_input_tps;
      if (ing.traffic.hourly_csv) c.traffic.hourly_csv = ing.traffic.hourly_csv;
    }
    const r = ing.requests;
    if (r && r.avg_input_tokens) {
      c.requests.avg_input_tokens = r.avg_input_tokens;
      if (r.avg_output_tokens) c.requests.avg_output_tokens = r.avg_output_tokens;
      const lng = +r.pct_long || 0, med = +r.pct_medium || 0;
      c.requests.pct_over_32k = +(med + lng).toFixed(2);
      c.requests.pct_over_128k = lng;
      if (r.medium_mean) c.requests.medium_mean = r.medium_mean;
      if (r.large_mean) c.requests.large_mean = r.large_mean;
      if (r.system_prompt_tokens) c.requests.system_prompt_tokens = r.system_prompt_tokens;
      syncBuckets(c);
    }
    if (ing.prompts) {
      if (ing.prompts.custom_system_prompts && ing.prompts.custom_system_prompts.length) c.prompts.custom_system_prompts = ing.prompts.custom_system_prompts;
      if (ing.prompts.file) { c.prompts.source = "file"; c.prompts.file = ing.prompts.file; }
      if (ing.prompts.system_prompts_file) c.prompts.system_prompts_file = ing.prompts.system_prompts_file;
    }
  }
  async function uploadTo(name, files, role) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    fd.append("role_hint", role || "auto");
    const hf = S.cfg && S.cfg.model && S.cfg.model.hf_id;
    if (hf) fd.append("hf_id", hf);
    const res = await api(`/api/sessions/${enc(name)}/upload`, { method: "POST", body: fd });
    return res.ingested || {};
  }
  // ---------------------------------------------------------------- workload from a document (POST /api/ingest/describe)
  const DOC_EXT = /\.(pdf|docx)$/i, TEXT_EXT = /\.(txt|md)$/i;
  // A .txt/.md file is a workload description when it reads like prose about volumes, not like a prompt.
  function looksLikeDescription(text) {
    const t = String(text || "").slice(0, 20000);
    if (!t.trim() || /^\s*[{[]/.test(t) || /^\s*you are\b/i.test(t)) return false;
    const kw = t.match(/\b(requests?|tokens?|per second|req\/s|rps|peak|traffic|latency|users?|per (?:day|month|hour|week)|monthly|weekly|average|throughput|SLA|rate limit|busiest|quiet(?:est)?)\b/gi) || [];
    const distinct = new Set(kw.map((k) => k.toLowerCase().replace(/s$/, "")));
    return distinct.size >= 4 && !/\n---\n/.test(t);
  }
  async function splitDescribe(files, role) {
    const docs = [], rest = [];
    for (const f of files) {
      if (role === "describe" || DOC_EXT.test(f.name)) docs.push(f);
      else if (role === "auto" && TEXT_EXT.test(f.name)) {
        let text = "";
        try { text = await f.text(); } catch (_) { text = ""; }
        (looksLikeDescription(text) ? docs : rest).push(f);
      } else rest.push(f);
    }
    return { docs, rest };
  }
  async function describeDoc(file) {
    const fd = new FormData();
    fd.append("file", file);
    try {
      return await api("/api/ingest/describe", { method: "POST", body: fd });
    } catch (e) {
      if (e.status === 404 || e.status === 405) throw new Error("this server can't read workload descriptions yet (POST /api/ingest/describe is missing). Update and restart dilemma ui");
      if (e.status === 401 || e.status === 403) throw new Error(S.auth && S.auth.sign_in_available === false
        ? `${e.message}. Check that ${S.auth.adc_account || "the app's service account"} has the Agent Platform User role on this project`
        : `${e.message}. Sign in to Google Cloud from the project menu, then try again`);
      if (e.status === 413) throw new Error(`${e.message}. Try a shorter document`);
      if (e.status === 415) throw new Error(`${e.message}. Use PDF, DOCX, TXT or MD`);
      throw e;
    }
  }
  async function runDescribe(file, out) {
    out = out || $("railUploadResult");
    out.className = "upload-result";
    out.innerHTML = `<span class="busy-note">Reading ${esc(file.name)} with Gemini. This takes 10–45 seconds…</span>`;
    try {
      const res = await describeDoc(file);
      out.textContent = `Read ${file.name}. Review the values before applying them.`;
      await openReview(res, file.name);
    } catch (e) {
      out.className = "upload-result err";
      out.textContent = `Couldn't read ${file.name}: ${e.message}.`;
    }
  }

  const TASKS = { extraction: "Structured extraction", summarization: "Summarization and RAG", reasoning: "Multi-step reasoning", coding: "Code generation", chat: "Low-latency chat" };
  const FIELD_INFO = {
    "traffic.peak_input_tps": ["Peak hour", (v) => `${fmtTok(v)} tok/s`],
    "traffic.average_input_tps": ["Weekly average", (v) => `${fmtTok(v)} tok/s`],
    "traffic.floor_input_tps": ["Quietest hour", (v) => `${fmtTok(v)} tok/s`],
    "traffic.monthly_input_tokens": ["Monthly input tokens", (v) => `${fmtTok(v)} tokens`],
    "traffic.busy_hours_utc": ["Busy window, UTC", (v) => Array.isArray(v) ? `${String(v[0]).padStart(2, "0")}:00 to ${String(v[1]).padStart(2, "0")}:00` : String(v)],
    "traffic.weekend_factor": ["Weekend load", (v) => `${pct(+v)} of weekdays`],
    "traffic.custom_hourly": ["Hour-by-hour week", (v) => Array.isArray(v) && v.length ? `${v.length} hourly values, peak ${fmtTok(Math.max.apply(null, v))} tok/s` : "—"],
    "requests.avg_input_tokens": ["Input tokens per request", (v) => `${fmtInt(v)} tokens`],
    "requests.avg_output_tokens": ["Output tokens (answer)", (v) => `${fmtInt(v)} tokens`],
    "requests.pct_over_32k": ["Requests over 32K", (v) => `${+v}%`],
    "requests.pct_over_128k": ["Requests over 128K", (v) => `${+v}%`],
    "requests.system_prompt_tokens": ["System prompt", (v) => `${fmtInt(v)} tokens`],
    "requests.medium_mean": ["Typical 32K–128K prompt", (v) => `${fmtInt(v)} tokens`],
    "requests.large_mean": ["Typical 128K+ prompt", (v) => `${fmtInt(v)} tokens`],
    "requests.thinking_share": ["Requests with thinking on", (v) => pct(+v)],
    "requests.avg_thinking_tokens": ["Thinking tokens when on", (v) => `${fmtInt(v)} tokens`],
    "maas.max_rps": ["MaaS rate limit", (v) => `${fmtRps(+v)} req/s`],
    "maas.input_per_m": ["MaaS input price", (v) => `${perM(+v)} per M`],
    "maas.output_per_m": ["MaaS output price", (v) => `${perM(+v)} per M`],
    "maas.cached_per_m": ["MaaS cached input price", (v) => `${perM(+v)} per M`],
    "constraints.latency_budget_s": ["Latency budget", (v) => `${fmtInt(v)} s`],
    "use_case.task_type": ["Task", (v) => TASKS[v] || String(v)],
    "use_case.description": ["Description", (v) => String(v)],
  };
  const fieldLabel = (path) => (FIELD_INFO[path] ? FIELD_INFO[path][0] : path.split(".").pop().replace(/_/g, " "));
  const fieldValue = (path, v) => (v == null || v === "") ? "not set" : (FIELD_INFO[path] ? FIELD_INFO[path][1](v) : (typeof v === "object" ? JSON.stringify(v) : String(v)));
  function flattenWorkload(w, prefix, out) {
    out = out || [];
    for (const k in (w || {})) {
      const path = prefix ? prefix + "." + k : k, v = w[k];
      if (v == null) continue;
      if (typeof v === "object" && !Array.isArray(v)) flattenWorkload(v, path, out);
      else out.push({ path, value: v });
    }
    return out;
  }
  function currentValue(path) {
    if (path === "traffic.monthly_input_tokens") return monthlyAnchor(S.cfg.traffic) ? +S.cfg.traffic.monthly_input_tokens : (+S.cfg.traffic.average_input_tps || 0) * SEC_MONTH;
    return getPath(S.cfg, path);
  }
  const sameValue = (a, b) => (Array.isArray(a) || Array.isArray(b)) ? JSON.stringify(a) === JSON.stringify(b)
    : (typeof a === "number" || typeof b === "number") ? Math.abs((+a) - (+b)) <= 1e-6 * Math.max(1, Math.abs(+a)) : String(a) === String(b);

  function openReview(res, filename) {
    if (!S.cfg) return;
    const byPath = {};
    (Array.isArray(res.fields) ? res.fields : []).forEach((f) => { if (f && f.path) byPath[f.path] = Object.assign({}, f); });
    flattenWorkload(res.workload).forEach((f) => { if (!byPath[f.path]) byPath[f.path] = { path: f.path, value: f.value, confidence: null, evidence: "" }; });
    const order = Object.keys(FIELD_INFO);
    const fields = Object.values(byPath).filter((f) => f.value != null)
      .sort((a, b) => (order.indexOf(a.path) < 0 ? 99 : order.indexOf(a.path)) - (order.indexOf(b.path) < 0 ? 99 : order.indexOf(b.path)));
    S.describe = { fields, filename, res };
    $("reviewTitle").textContent = "Review what we read from your document";
    $("reviewSub").textContent = fields.length
      ? `${filename}${res.model ? ` · read by ${res.model}` : ""}. Nothing changes until you apply. Values marked Assumed are not in the document and start unselected.`
      : `${filename}: no workload figures were found. Nothing was changed.`;
    const notes = (res.notes || []).map((n) => `<li>${icon("i-info", "ico-sm")}<span>${esc(ap(n))}</span></li>`).join("");
    $("reviewNotes").innerHTML = `<ul class="tips review-notes"${notes ? "" : " hidden"}>${notes}</ul><p class="warn-line review-conflict" id="reviewConflict" role="status" hidden></p>`;
    $("reviewList").innerHTML = fields.map((f, i) => {
      const conf = ["stated", "derived", "assumed"].indexOf(f.confidence) >= 0 ? f.confidence : null;
      const cur = currentValue(f.path);
      const same = cur != null && sameValue(cur, f.value);
      // The monthly total becomes the volume anchor, so only pre-select it when the document states it.
      const checked = (f.path === "traffic.monthly_input_tokens" ? conf === "stated" : conf !== "assumed") && !same;
      return `<div class="review-row${same ? " same" : ""}">
        <input type="checkbox" id="rv-${i}" data-rv="${i}"${checked ? " checked" : ""} aria-describedby="rv-${i}-v rv-${i}-e">
        <label for="rv-${i}" class="review-name">${esc(fieldLabel(f.path))}</label>
        <span class="review-val" id="rv-${i}-v"><strong>${esc(fieldValue(f.path, f.value))}</strong><span class="review-now">${same ? "Same as now" : "Now: " + esc(fieldValue(f.path, cur))}</span></span>
        <span class="conf conf-${conf || "unknown"}">${conf ? conf.charAt(0).toUpperCase() + conf.slice(1) : "No tag"}</span>
        <span class="review-ev" id="rv-${i}-e">${f.evidence ? `<q>${esc(f.evidence)}</q>` : '<span class="muted">No evidence given.</span>'}</span>
      </div>`;
    }).join("");
    $("reviewApply").disabled = !fields.length;
    const d = $("reviewDlg");
    const sync = () => {
      const boxes = Array.from(d.querySelectorAll("[data-rv]"));
      const n = boxes.filter((b) => b.checked).length;
      $("reviewApply").textContent = n ? `Apply ${n} selected` : "Apply selected";
      $("reviewApply").disabled = !n;
      $("reviewAll").checked = n === boxes.length && n > 0;
      $("reviewAll").indeterminate = n > 0 && n < boxes.length;
      reviewConflicts(boxes.filter((b) => b.checked).map((b) => S.describe.fields[+b.dataset.rv]));
    };
    d.querySelectorAll("[data-rv]").forEach((b) => b.addEventListener("change", sync));
    $("reviewAll").onchange = () => { d.querySelectorAll("[data-rv]").forEach((b) => { b.checked = $("reviewAll").checked; }); sync(); };
    sync();
    $("reviewCancel").onclick = () => d.close();
    $("reviewForm").onsubmit = (e) => {
      e.preventDefault();
      const picked = Array.from(d.querySelectorAll("[data-rv]")).filter((b) => b.checked).map((b) => S.describe.fields[+b.dataset.rv]);
      d.close();
      applyDescribed(picked);
    };
    d.showModal();
    const first = d.querySelector("[data-rv]");
    (first || $("reviewCancel")).focus();
    return new Promise((resolve) => d.addEventListener("close", () => resolve(), { once: true }));
  }
  // Warn, in the sheet, about selections that contradict each other or the current workload.
  function reviewConflicts(picked) {
    const val = (path) => { const f = picked.find((x) => x.path === path); return f ? +f.value : (path === "traffic.monthly_input_tokens" ? null : +getPath(S.cfg, path)); };
    const msgs = [];
    const peak = val("traffic.peak_input_tps"), avg = val("traffic.average_input_tps"), floor = val("traffic.floor_input_tps");
    const pm = picked.find((x) => x.path === "traffic.monthly_input_tokens");
    const pa = picked.find((x) => x.path === "traffic.average_input_tps");
    const avgEff = pm ? +pm.value / SEC_MONTH : avg;
    if (pm && pa && Math.abs(+pa.value * SEC_MONTH - +pm.value) > 0.05 * +pm.value) msgs.push(`The weekly average (${fmtTok(+pa.value)} tok/s) and the monthly total (${fmtTok(+pm.value)} tokens, about ${fmtTok(+pm.value / SEC_MONTH)} tok/s) disagree. The monthly total wins if you apply both.`);
    if (!picked.some((x) => x.path === "traffic.custom_hourly") && peak > 0 && avgEff > peak) msgs.push(`The average (${fmtTok(avgEff)} tok/s) would be above the peak (${fmtTok(peak)} tok/s).`);
    if (floor > 0 && avgEff > 0 && floor > avgEff) msgs.push(`The quietest hour (${fmtTok(floor)} tok/s) would be above the average (${fmtTok(avgEff)} tok/s).`);
    const o32 = val("requests.pct_over_32k"), o128 = val("requests.pct_over_128k");
    if (o128 > o32) msgs.push(`Requests over 128K (${o128}%) can't exceed requests over 32K (${o32}%).`);
    const el = $("reviewConflict");
    if (!el) return;
    el.hidden = !msgs.length;
    el.innerHTML = msgs.length ? icon("i-alert") + `<span>${msgs.map(esc).join(" ")}</span>` : "";
  }
  function applyDescribed(picked) {
    if (!picked.length || !S.cfg) return;
    const c = S.cfg, t = c.traffic;
    const has = (p) => picked.some((f) => f.path === p);
    let simpleTraffic = false;
    for (const f of picked) {
      const v = f.value;
      if (f.path === "traffic.monthly_input_tokens") { if (+v > 0) setMonthly(+v); continue; }
      if (f.path === "traffic.average_input_tps") { if (!has("traffic.monthly_input_tokens")) { t.average_input_tps = +v; t.monthly_input_tokens = null; } continue; }
      if (f.path === "traffic.custom_hourly") { if (Array.isArray(v) && v.length === 168) { t.custom_hourly = v.map((x) => Math.max(0, +x || 0)); t.hourly_csv = null; } continue; }
      if (/^traffic\.(peak_input_tps|floor_input_tps|busy_hours_utc|weekend_factor)$/.test(f.path)) simpleTraffic = true;
      setPath(c, f.path, Array.isArray(v) ? v.slice() : v);
    }
    if (simpleTraffic && !has("traffic.custom_hourly")) { t.custom_hourly = null; t.hourly_csv = null; }
    if (simpleTraffic || has("traffic.custom_hourly")) t.preset = null;
    if (has("traffic.average_input_tps") && !has("traffic.monthly_input_tokens") && isCustom(t)) {
      const m = meanOf(t.custom_hourly); if (m > 0) t.custom_hourly = t.custom_hourly.map((x) => x * t.average_input_tps / m);
    }
    S.cfg = normalizeCfg(c);
    S.models = null;
    populateRail();
    markDirty();
    recompute({ animate: true });
    toast(`Applied ${picked.length} ${picked.length === 1 ? "value" : "values"} from ${S.describe ? S.describe.filename : "the document"}. Save to keep them.`);
  }

  // ---------------------------------------------------------------- uploads, one per rail section
  // Traffic: time series (review sheet), request logs with timestamps, workload descriptions (Gemini).
  // System prompt: system-prompt files. Prompts: your prompt set for the benchmark (POST /api/sessions/{s}/prompts).
  // Each section takes only its kind, points to the right section otherwise, and offers the other parts of a file
  // that holds several kinds (nothing is applied without a click).
  const STRUCT_EXT = /\.(csv|tsv|jsonl|json|parquet)$/i;
  async function classifyFiles(files) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    const res = await api("/api/uploads/classify", { method: "POST", body: fd });
    const by = {};
    (res.files || []).forEach((k, i) => { by[i] = k; });
    return files.map((f, i) => ({ file: f, kind: by[i] || { kinds: [], primary: "unknown" } }));
  }
  const hasKind = (k, kind) => (k.kinds || []).indexOf(kind) >= 0;
  function setResult(out, html, isErr) {
    out.className = "upload-result" + (isErr ? " err" : "");
    out.innerHTML = html;
  }
  // "Also has …" offers below an upload result: [{text, label, run}]
  function addOffers(out, offers) {
    offers.forEach((o) => {
      const p = document.createElement("p");
      p.className = "offer";
      p.innerHTML = `<span>${esc(o.text)}</span> `;
      const b = document.createElement("button");
      b.type = "button"; b.className = "link-btn"; b.textContent = o.label;
      b.addEventListener("click", async () => { p.remove(); await o.run(); });
      p.appendChild(b);
      out.appendChild(p);
    });
    if (offers.length && out.classList.contains("err")) out.classList.remove("err");
  }
  async function applyTrafficPart(ing, name) {
    applyIngest(S.cfg, { traffic: ing.traffic });
    const res = await postJSON(`/api/sessions/${enc(S.name)}`, { config: collectCfg(), ui_state: uiState() });
    applySession(res, { resetUi: false });
    toast(`Traffic shape from ${name} applied.`);
  }
  async function trafficFromFile(file, out, review) {
    // Request logs / hourly CSVs: only the traffic shape (the request sizes and prompts are offered separately).
    // review: read it and return without applying (the Traffic section shows it with an Apply button).
    const ing = await uploadTo(S.name, [file], "traffic");
    if (!review && ing.traffic && ing.traffic.peak_input_tps) await applyTrafficPart(ing, file.name);
    return ing;
  }
  function requestSizeItems(r) {
    if (!r || !r.avg_input_tokens) return [];
    const items = [["requests.avg_input_tokens", r.avg_input_tokens], ["requests.pct_over_32k", +(+(r.pct_medium || 0) + +(r.pct_long || 0)).toFixed(2)], ["requests.pct_over_128k", +r.pct_long || 0]];
    if (r.avg_output_tokens) items.push(["requests.avg_output_tokens", r.avg_output_tokens]);
    if (+r.pct_medium > 0 && r.medium_mean) items.push(["requests.medium_mean", r.medium_mean]);
    if (+r.pct_long > 0 && r.large_mean) items.push(["requests.large_mean", r.large_mean]);
    return items.map(([path, value]) => ({ path, value }));
  }

  async function trafficUpload(files) {
    const out = $("railUploadResult");
    if (!files || !files.length || !S.session) return;
    files = Array.from(files);
    const docs = [], structured = [], wrong = [];
    for (const f of files) {
      if (DOC_EXT.test(f.name)) docs.push(f);
      else if (TEXT_EXT.test(f.name)) {
        let text = "";
        try { text = await f.text(); } catch (_) { text = ""; }
        if (looksLikeDescription(text)) docs.push(f);
        else wrong.push(`${f.name} looks like a prompt or a system prompt, not traffic: use Upload in Prompts (or in System prompt).`);
      } else if (STRUCT_EXT.test(f.name)) structured.push(f);
      else wrong.push(`${f.name}: unsupported file type here. Traffic takes CSV, TSV, JSON, JSONL or Parquet exports, or a PDF, DOCX, TXT or MD description.`);
    }
    const msgs = [], offers = [];
    if (structured.length) {
      if (!(await ensureWritable())) return;
      setResult(out, `<span class="busy-note">Reading ${structured.length} ${structured.length === 1 ? "file" : "files"}…</span>`);
      let classified;
      try { classified = await classifyFiles(structured); } catch (e) { setResult(out, `Couldn't read the files: ${esc(e.message)}.`, true); return; }
      const series = classified.filter((c) => hasKind(c.kind, "timeseries")).map((c) => c.file);
      for (const c of classified) {
        const k = c.kind, f = c.file;
        if (hasKind(k, "timeseries")) continue;
        if (hasKind(k, "hourly_traffic") || hasKind(k, "request_timestamps")) {
          try {
            const ing = await trafficFromFile(f, out, true);
            const t = ing.traffic || {};
            if (t.peak_input_tps) {
              msgs.push(`${f.name}: a week of traffic from its ${hasKind(k, "request_timestamps") ? "request timestamps" : "hourly values"}: peak ${fmtTok(t.peak_input_tps)}, average ${fmtTok(t.average_input_tps)}, quietest hour ${fmtTok(t.floor_input_tps)} tok/s.`);
              offers.unshift({ text: `Use it for the traffic shape?`, label: "Apply traffic shape", run: () => applyTrafficPart(ing, f.name).catch((e) => toast(`Couldn't apply the traffic: ${e.message}`, true)) });
            } else msgs.push(`${f.name}: too few requests over time to derive a traffic shape.`);
            const sizes = requestSizeItems(ing.requests);
            if (sizes.length) offers.push({ text: `${f.name} also has request sizes: update them from the file?`, label: "Review request sizes", run: () => openSizeReview(sizes, f.name) });
          } catch (e) { msgs.push(`${f.name}: ${e.message}.`); }
          if (hasKind(k, "prompts")) offers.push({ text: `${f.name} also has prompts (${fmtInt(k.prompts)}): use them for the benchmark?`, label: "Use as my prompts", run: () => promptUpload([f]) });
        } else if (hasKind(k, "prompts")) {
          wrong.push(`${f.name} looks like a prompt set without timestamps, not traffic: use Upload in Prompts.`);
          offers.push({ text: `Use ${f.name} for the benchmark instead?`, label: "Use as my prompts", run: () => promptUpload([f]) });
        } else if (hasKind(k, "token_counts")) {
          try {
            const ing = await uploadTo(S.name, [f], "traffic");
            const sizes = requestSizeItems(ing.requests);
            msgs.push(`${f.name}: per-request token counts without timestamps, so no traffic shape.`);
            if (sizes.length) offers.push({ text: `Update the request sizes from ${f.name}?`, label: "Review request sizes", run: () => openSizeReview(sizes, f.name) });
          } catch (e) { msgs.push(`${f.name}: ${e.message}.`); }
        } else if (hasKind(k, "system_prompts")) {
          wrong.push(`${f.name} holds system prompts, not traffic: use Upload in System prompt.`);
        } else wrong.push(`${f.name}: no timestamps or traffic columns found. Traffic files need a time column with values (e.g. a Cloud Monitoring CSV) or per-request timestamps.`);
      }
      const lines = msgs.concat(wrong);
      setResult(out, lines.length ? `<ul>${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : "", !msgs.length && wrong.length > 0);
      addOffers(out, offers);
      if (series.length) await runTrafficImport(series, out.querySelector("ul") ? document.createElement("div") : out);
    } else if (wrong.length) {
      setResult(out, `<ul>${wrong.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`, true);
    }
    for (const d of docs) await runDescribe(d, out);
  }

  async function sysUpload(files) {
    const out = $("sysUploadResult");
    if (!files || !files.length || !S.session) return;
    files = Array.from(files);
    const bad = files.filter((f) => DOC_EXT.test(f.name) || !/\.(txt|md|json|jsonl|csv|tsv)$/i.test(f.name));
    if (bad.length) {
      setResult(out, esc(`${bad.map((f) => f.name).join(", ")}: ${DOC_EXT.test(bad[0].name) ? "a workload description goes to Upload in Traffic" : "use TXT, MD, JSON, JSONL, CSV or TSV"}.`), true);
      return;
    }
    setResult(out, `<span class="busy-note">Reading system prompts…</span>`);
    let classified = [];
    try {
      const structured = files.filter((f) => STRUCT_EXT.test(f.name));
      if (structured.length) classified = await classifyFiles(structured);
      const fd = new FormData();
      for (const f of files) fd.append("files", f);
      const res = await api(`/api/sessions/${enc(S.name)}/system-prompts`, { method: "POST", body: fd });
      const n = res.count;
      out.className = "upload-result";
      out.innerHTML = `<p>Found <strong>${fmtInt(n)}</strong> system ${n === 1 ? "prompt" : "prompts"}, about ${fmtInt(res.avg_tokens)} tokens on average${res.token_method === "estimate" ? " (estimated)" : ""}.</p>
        <label class="check"><input type="checkbox" id="sysSetSize"> Also set "System prompt" to ${fmtInt(res.avg_tokens)} tokens (now ${fmtInt(+S.cfg.requests.system_prompt_tokens || 0)})</label>
        <span class="btn-row"><button type="button" class="btn btn-sm btn-primary" id="sysUse">Use ${n === 1 ? "it" : "them"} for benchmarks</button><button type="button" class="btn btn-sm" id="sysCancel">Cancel</button></span>`;
      $("sysCancel").onclick = () => setResult(out, "");
      $("sysUse").onclick = () => {
        S.cfg.prompts.custom_system_prompts = res.system_prompts.slice();
        if ($("sysSetSize").checked) S.cfg.requests.system_prompt_tokens = res.avg_tokens;
        populateRail(); markDirty(); scheduleRecompute();
        setResult(out, esc(`Using ${n} system ${n === 1 ? "prompt" : "prompts"} for benchmarks. Save to keep ${n === 1 ? "it" : "them"}.`));
        renderMyPrompts();
        addOffers(out, sysOffers());
      };
    } catch (e) {
      setResult(out, esc(e.message), true);
    }
    function sysOffers() {
      const offers = [];
      classified.forEach((c) => {
        if (hasKind(c.kind, "prompts")) offers.push({ text: `${c.file.name} also has prompts (${fmtInt(c.kind.prompts)}): use them for the benchmark?`, label: "Use as my prompts", run: () => promptUpload([c.file]) });
        if (hasKind(c.kind, "request_timestamps")) offers.push({ text: `${c.file.name} also has timestamps: use them for the traffic shape?`, label: "Use for traffic", run: () => trafficFromFile(c.file).catch((e) => toast(e.message, true)) });
      });
      return offers;
    }
    if (out.classList.contains("err")) addOffers(out, sysOffers());
  }

  // ---------------------------------------------------------------- my prompts (bring your own prompts)
  const levelRequests = (c) => (c.loadtest.concurrency || []).reduce((a, x) => a + Math.max(+c.loadtest.requests_per_level || 0, 2 * +x), 0);
  const fmtDay = (ts) => ts ? new Date(ts * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "";
  function promptSourceText() {
    const s = S.session && S.session.user_prompts;
    if (S.cfg && S.cfg.prompts.source === "file" && s) return `your ${fmtInt(s.count)} prompts (uploaded ${fmtDay(s.uploaded_at)})`;
    if (S.cfg && S.cfg.prompts.source === "file") return "your prompts";
    return "synthetic, shaped like your request sizes";
  }
  function answerText(s, c) {
    const n = s.count, a = s.answer || {};
    const out = Math.max(16, Math.round(+c.requests.avg_output_tokens || 400));
    const share = +c.requests.thinking_share || 0, think = Math.round(+c.requests.avg_thinking_tokens || 2 * out);
    const def = a.prompts_using_default || 0;
    let t = def === n ? `No answer lengths in the file: every prompt gets your "Output tokens (answer)" setting, ${fmtInt(out)} tokens.`
      : def ? `${fmtInt(n - def)} prompts bring their own answer length; the other ${fmtInt(def)} get your "Output tokens (answer)" setting, ${fmtInt(out)} tokens.`
      : "Every prompt brings its own answer length (max_tokens or output token counts from the file).";
    const free = n - (a.prompts_with_template_kwargs || 0);
    const nThink = share > 0 ? Math.floor(free * share + 1e-9) : 0;
    if (nThink) t += ` ${fmtInt(nThink)} of them (${pct(share)}) run with thinking on, +${fmtInt(think)} tokens.`;
    return t;
  }
  function renderMyPrompts() {
    if (!S.cfg) return;
    const s = S.session && S.session.user_prompts;
    const mine = S.cfg.prompts.source === "file" || S.promptMine === S.name;
    $("promptSrcSyn").setAttribute("aria-pressed", String(!mine));
    $("promptSrcMine").setAttribute("aria-pressed", String(mine));
    $("myPrompts").hidden = !mine;
    const card = $("promptSummary");
    const hasSet = !!(s && S.cfg.prompts.source === "file");
    $("promptUploader").hidden = hasSet && !S.promptReplace;
    card.hidden = !hasSet;
    if (!hasSet) { card.innerHTML = ""; return; }
    const c = S.cfg, inp = s.input || {}, n = s.count;
    const req = levelRequests(c);
    const warns = (s.warnings || []).filter((w) => w.kind !== "repeat").map((w) => w.text);
    if (req > n) warns.unshift(`${fmtInt(n)} prompts for ${fmtInt(req)} load-test requests (concurrency ${(c.loadtest.concurrency || []).join(", ")}, ${fmtInt(c.loadtest.requests_per_level)} per level): prompts repeat across levels, each sent about ${(req / n).toFixed(1)} times. Prefix caching may flatter repeated requests; upload ${fmtInt(req)} or more for fresh prompts at every level.`);
    const tm = s.tokens || {};
    const tokLabel = tm.method === "tokenizer" ? `counted with the model's tokenizer` : tm.method === "estimate" ? "estimated (about 3.5 characters per token)" : "from the file";
    const files = (s.files || []).map((f) => `${f.filename} (${f.format}${f.fields && f.fields.length && f.filename !== "pasted" ? `: ${f.fields.join(", ")}` : ""})`).join("; ");
    const outs = s.answer || {};
    const sys = s.with_system ? `${fmtInt(s.with_system)} of ${fmtInt(n)} bring their own system prompt` : "None bring their own system prompt";
    const sysAdd = s.with_system < n && (c.prompts.custom_system_prompts || []).length ? `; the other ${fmtInt(n - s.with_system)} get one of your ${(c.prompts.custom_system_prompts || []).length} system prompts` : "";
    card.innerHTML = `
      <p class="pc-head"><strong>${fmtInt(n)} ${n === 1 ? "prompt" : "prompts"}</strong> <span class="muted">uploaded ${esc(fmtDay(s.uploaded_at))}</span></p>
      <p class="pc-files muted">${esc(files)}</p>
      <dl class="pc-stats">
        <div><dt>Median</dt><dd>${fmtInt(inp.median)}</dd></div>
        <div><dt>p95</dt><dd>${fmtInt(inp.p95)}</dd></div>
        <div><dt>Max</dt><dd>${fmtInt(inp.max)}</dd></div>
        <div><dt>Over 32K</dt><dd>${pct1((inp.pct_over_32k || 0) / 100)}</dd></div>
        <div><dt>Over 128K</dt><dd>${pct1((inp.pct_over_128k || 0) / 100)}</dd></div>
      </dl>
      <p class="hint">Input tokens per prompt, ${esc(tokLabel)}.</p>
      <ul class="pc-facts">
        <li>${esc(answerText(s, c))}</li>
        <li>${outs.prompts_with_output_tokens ? `${fmtInt(outs.prompts_with_output_tokens)} came with output token counts` : "No output token counts in the file"}; ${outs.prompts_with_max_tokens ? `${fmtInt(outs.prompts_with_max_tokens)} with max_tokens` : "no max_tokens"}.</li>
        <li>${esc(sys + sysAdd)}.</li>
      </ul>
      <p class="pc-sub">First ${Math.min(3, (s.preview || []).length)}</p>
      <ol class="pc-preview">${(s.preview || []).map((p) => `<li><span class="pc-text">${esc(p.text)}</span><span class="muted"> · ${fmtInt(p.tokens)} tokens${p.system ? " · own system prompt" : ""}</span></li>`).join("")}</ol>
      ${warns.length ? `<ul class="pc-warn">${warns.map((w) => `<li>${icon("i-alert", "ico-sm")}<span>${esc(w)}</span></li>`).join("")}</ul>` : ""}
      <div class="pc-actions">
        <button type="button" class="btn btn-sm" id="btnPromptSizes">Update request sizes from my prompts</button>
        <button type="button" class="btn btn-sm" id="btnPromptReplace">Replace</button>
        <button type="button" class="btn btn-sm btn-danger" id="btnPromptRemove">Remove (back to synthetic)</button>
      </div>
      <div id="promptSizeReview"></div>`;
    $("btnPromptSizes").onclick = () => openSizeReview((s.proposal || []).map((p) => ({ path: p.path, value: p.value })), "your prompts", $("promptSizeReview"));
    $("btnPromptReplace").onclick = () => { S.promptReplace = true; renderMyPrompts(); $("promptUploadInput").click(); };
    $("btnPromptRemove").onclick = removePrompts;
  }
  // Review list for request sizes derived from a file (checkboxes, like the traffic importer's derived values).
  function openSizeReview(items, from, host) {
    host = host || $("railUploadResult");
    if (!items.length) { toast("No request sizes found.", true); return; }
    const rows = items.map((it, i) => {
      const cur = getPath(S.cfg, it.path);
      const same = cur != null && sameValue(cur, it.value);
      return `<label class="size-row${same ? " same" : ""}"><input type="checkbox" data-sz="${i}"${same ? "" : " checked"}><span class="size-name">${esc(fieldLabel(it.path))}</span><span class="size-val"><strong>${esc(fieldValue(it.path, it.value))}</strong><span class="muted">${same ? "Same as now" : "Now: " + esc(fieldValue(it.path, cur))}</span></span></label>`;
    }).join("");
    const box = document.createElement("div");
    box.className = "size-review";
    box.innerHTML = `<p class="size-h">Request sizes measured on ${esc(from)}</p>${rows}<span class="btn-row"><button type="button" class="btn btn-sm btn-primary" data-sz-apply>Apply selected</button><button type="button" class="btn btn-sm" data-sz-cancel>Cancel</button></span>`;
    host.querySelectorAll(".size-review").forEach((x) => x.remove());
    host.appendChild(box);
    host.hidden = false;
    const sync = () => { const k = box.querySelectorAll("[data-sz]:checked").length; const b = box.querySelector("[data-sz-apply]"); b.disabled = !k; b.textContent = k ? `Apply ${k} selected` : "Apply selected"; };
    box.querySelectorAll("[data-sz]").forEach((b) => b.addEventListener("change", sync));
    sync();
    box.querySelector("[data-sz-cancel]").onclick = () => box.remove();
    box.querySelector("[data-sz-apply]").onclick = () => {
      const picked = Array.from(box.querySelectorAll("[data-sz]:checked")).map((b) => items[+b.dataset.sz]);
      for (const it of picked) setPath(S.cfg, it.path, +it.value);
      if (picked.some((it) => /pct_over/.test(it.path))) syncBuckets(S.cfg);
      S.cfg = normalizeCfg(S.cfg);
      S.models = null;
      populateRail(); markDirty(); recompute({ animate: true });
      box.remove();
      toast(`Applied ${picked.length} ${picked.length === 1 ? "value" : "values"} from ${from}. Save to keep them.`);
    };
    const first = box.querySelector("[data-sz]");
    if (first) first.focus();
  }
  function afterPromptsSaved(res) {
    S.cfg.prompts.source = res.source;
    S.cfg.prompts.file = res.file || null;
    if (S.session) { S.session.user_prompts = res.summary; S.session.prompt_source = res.prompt_source; }
    S.promptReplace = false;
    renderMyPrompts();
    renderBenchServing();
  }
  async function promptUpload(files, paste) {
    const out = $("promptStatus");
    if (!S.session) return;
    files = Array.from(files || []);
    const wrong = [];
    for (const f of files) {
      if (DOC_EXT.test(f.name)) wrong.push(`${f.name} looks like a workload description: use Upload in Traffic.`);
      else if (!/\.(jsonl|json|csv|tsv|txt|md)$/i.test(f.name)) wrong.push(`${f.name}: unsupported file type. Use JSONL, JSON, CSV, TSV, TXT or MD.`);
    }
    if (wrong.length) { setResult(out, wrong.map(esc).join("<br>"), true); return; }
    if (!(await ensureWritable())) return;
    S.promptMine = S.name;
    renderMyPrompts();
    setResult(out, `<span class="busy-note">Reading your prompts…</span>`);
    let classified = [];
    try {
      const structured = files.filter((f) => STRUCT_EXT.test(f.name));
      if (structured.length) {
        classified = await classifyFiles(structured);
        const traffic = classified.filter((c) => !hasKind(c.kind, "prompts") && (hasKind(c.kind, "timeseries") || hasKind(c.kind, "hourly_traffic") || hasKind(c.kind, "request_timestamps")));
        if (traffic.length) {
          setResult(out, esc(`${traffic.map((c) => c.file.name).join(", ")} ${traffic.length === 1 ? "looks" : "look"} like a traffic time series: use Upload in Traffic.`), true);
          addOffers(out, [{ text: "", label: "Send to Traffic instead", run: () => trafficUpload(traffic.map((c) => c.file)) }]);
          return;
        }
      }
      let res;
      if (paste != null) res = await postJSON(`/api/sessions/${enc(S.name)}/prompts`, { text: paste, mode: $("promptTextMode").value });
      else {
        const fd = new FormData();
        for (const f of files) fd.append("files", f);
        fd.append("text_mode", $("promptTextMode").value);
        res = await api(`/api/sessions/${enc(S.name)}/prompts`, { method: "POST", body: fd });
      }
      afterPromptsSaved(res);
      setResult(out, "");
      if (paste != null) { $("promptPaste").value = ""; $("pasteBox").hidden = true; $("btnPastePrompts").setAttribute("aria-expanded", "false"); }
      toast(`Using your ${fmtInt(res.summary.count)} prompts for the benchmark.`);
      const offers = classified.filter((c) => hasKind(c.kind, "request_timestamps")).map((c) => ({ text: `${c.file.name} also has timestamps: use them for the traffic shape?`, label: "Use for traffic", run: () => trafficFromFile(c.file).then(() => setResult(out, esc(`Traffic shape from ${c.file.name} applied.`))).catch((e) => setResult(out, esc(e.message), true)) }));
      if (offers.length) { setResult(out, ""); addOffers(out, offers); }
      const card = $("promptSummary");
      if (card && !card.hidden) card.scrollIntoView({ block: "nearest", behavior: reduceMotion.matches ? "auto" : "smooth" });
    } catch (e) {
      setResult(out, esc(e.status === 413 ? e.message : `Couldn't use these prompts: ${e.message}`), true);
    }
  }
  async function removePrompts() {
    const s = S.session && S.session.user_prompts;
    const ok = await dialog({ title: "Remove your prompts?", body: `<p>The benchmark goes back to synthetic prompts shaped like your request sizes. ${s ? `Your ${fmtInt(s.count)} uploaded prompts are deleted from this session's folder.` : ""}</p>`, okLabel: "Remove", danger: true });
    if (!ok) return false;
    if (!(await ensureWritable())) return false;
    try {
      const res = await api(`/api/sessions/${enc(S.name)}/prompts`, { method: "DELETE" });
      S.promptMine = null;
      afterPromptsSaved(res);
      setResult($("promptStatus"), "");
      toast("Back to synthetic prompts.");
      return true;
    } catch (e) {
      toast(`Couldn't remove the prompts: ${e.message}`, true);
      return false;
    }
  }
  function initMyPrompts() {
    bindDrop($("trafficDrop"), $("trafficUploadInput"), trafficUpload);
    bindDrop($("sysDrop"), $("sysUploadInput"), sysUpload);
    bindDrop($("promptDrop"), $("promptUploadInput"), (files) => promptUpload(files));
    $("promptSrcMine").addEventListener("click", () => { S.promptMine = S.name; renderMyPrompts(); });
    $("promptSrcSyn").addEventListener("click", async () => {
      if (S.cfg.prompts.source === "file") { await removePrompts(); return; }
      S.promptMine = null; S.promptReplace = false; renderMyPrompts();
    });
    $("btnPastePrompts").addEventListener("click", () => {
      const open = $("pasteBox").hidden;
      $("pasteBox").hidden = !open;
      $("btnPastePrompts").setAttribute("aria-expanded", String(open));
      if (open) $("promptPaste").focus();
    });
    $("btnUsePasted").addEventListener("click", () => {
      const t = $("promptPaste").value;
      if (!t.trim()) { $("promptPaste").focus(); return; }
      promptUpload([], t);
    });
  }
  // ---------------------------------------------------------------- traffic time series (POST /api/sessions/{s}/traffic/import, /apply)
  // Timestamped exports (Cloud Monitoring CSVs, timestamp,value files) and Cloud Monitoring imports open a review
  // sheet: per-series mapping, coverage, the resulting week against the current one, derived values, partial-data
  // options. Nothing is applied until "Apply".
  const TS_ROLE_OPTS = [["requests", "requests/s"], ["input_tps", "input tokens/s"], ["output_tps", "output tokens/s"], ["total_tps", "total tokens/s (input + output)"], ["ignore", "ignore"]];
  const CONF_TAG = { high: "stated", medium: "derived", low: "assumed" };
  function trafficContext() {
    let prof = null;
    try { prof = currentProfile(); } catch (_) { prof = null; }
    return { current_profile: Array.isArray(prof) && prof.length === 168 ? prof : null, requests: clone(S.cfg.requests || {}), weekend_factor: S.cfg.traffic.weekend_factor };
  }
  async function runTrafficImport(files, out) {
    out = out || $("railUploadResult");
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    fd.append("context", JSON.stringify(trafficContext()));
    out.className = "upload-result";
    out.innerHTML = `<span class="busy-note">Reading the traffic in ${esc(files.map((f) => f.name).join(", "))}…</span>`;
    try {
      const res = await api(`/api/sessions/${enc(S.name)}/traffic/import`, { method: "POST", body: fd });
      out.textContent = `Read ${res.series.length} ${res.series.length === 1 ? "series" : "series"}. Review the traffic before applying it.`;
      await openTrafficReview(res, { files: files.map((f) => f.name) });
    } catch (e) {
      out.className = "upload-result err";
      out.textContent = e.status === 404 || e.status === 405 ? "This server can't import traffic series yet. Update and restart dilemma ui." : `Couldn't read the traffic: ${e.message}`;
    }
  }
  function tsSpark(week, cur) {
    const W = 336, H = 64, all = (week || []).concat(cur || []);
    const mx = Math.max.apply(null, all.length ? all : [1]) || 1;
    const pts = (a) => a.map((v, i) => `${(i * W / 167).toFixed(1)},${(H - 2 - (Math.max(0, +v || 0) / mx) * (H - 6)).toFixed(1)}`).join(" ");
    let days = "";
    for (let d = 1; d < 7; d++) days += `<line x1="${d * 48}" x2="${d * 48}" y1="0" y2="${H}" class="ts-grid"/>`;
    const lbl = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((d, i) => `<span>${d}</span>`).join("");
    return `<svg class="ts-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Proposed week compared with the current week">${days}
      ${cur ? `<polyline class="ts-cur" points="${pts(cur)}"/>` : ""}<polyline class="ts-new" points="${pts(week)}"/></svg><div class="ts-days" aria-hidden="true">${lbl}</div>`;
  }
  function openTrafficReview(res, meta) {
    if (!S.cfg) return Promise.resolve();
    const d = $("tsDlg");
    const st = { res, mapping: {}, option: res.option, gaps: !!(res.coverage && res.coverage.gaps_as_zero), accept: {}, busy: false };
    (res.series || []).forEach((s) => { st.mapping[s.id] = s.role; });
    (res.derived || []).forEach((x) => { st.accept[x.path] = x.checked !== false; });
    const src = res.source || {};
    $("tsSub").textContent = src.kind === "monitoring"
      ? `Cloud Monitoring: ${src.model || src.endpoint_id || src.label || "source"}${src.location ? " · " + src.location : ""} · last ${src.days} days. Nothing changes until you apply.`
      : `${((meta && meta.files) || src.files || []).join(", ")}. Nothing changes until you apply.`;
    const render = () => {
      const r = st.res, opts = r.options || [], cov = r.coverage;
      const opt = opts.find((o) => o.key === st.option) || opts[0];
      let cur = null;
      try { cur = currentProfile(); } catch (_) { cur = null; }
      const warn = (r.warnings || []).map((w) => `<li>${icon("i-info", "ico-sm")}<span>${esc(ap(w))}</span></li>`).join("");
      const rows = (r.series || []).map((s) => {
        const conf = CONF_TAG[s.confidence] || "unknown";
        const lab = Object.keys(s.labels || {}).map((k) => `<span class="ts-chip">${esc(k)}: ${esc(s.labels[k])}</span>`).join("");
        const span = s.first && s.last ? `${new Date(s.first).toISOString().slice(5, 16).replace("T", " ")} – ${new Date(s.last).toISOString().slice(5, 16).replace("T", " ")} UTC` : "";
        return `<div class="ts-row">
          <div class="ts-name"><strong>${esc(s.name)}</strong><span class="muted">${esc(s.file || "")}</span>${lab ? `<span class="ts-chips">${lab}</span>` : ""}</div>
          <div class="ts-map"><label for="tsm-${esc(s.id)}" class="visually-hidden">What ${esc(s.name)} measures</label>
            <select id="tsm-${esc(s.id)}" data-ts="${esc(s.id)}">${TS_ROLE_OPTS.map(([k, l]) => `<option value="${k}"${st.mapping[s.id] === k ? " selected" : ""}>${l}</option>`).join("")}</select></div>
          <div class="ts-why"><span class="conf conf-${conf}">Guess: ${esc((r.role_labels && r.role_labels[s.metric_guess]) || s.metric_guess)}</span> <span class="muted">${esc(s.reason || "")}. ${fmtInt(s.points_n)} points every ${s.step_s >= 3600 ? (s.step_s / 3600) + " h" : Math.round(s.step_s / 60) + " min"}, ${span}.</span></div>
        </div>`;
      }).join("");
      const optHTML = opts.length > 1 || (cov && cov.span_hours < 24) ? `<fieldset class="ts-opts"><legend>How to use it</legend>${opts.map((o) => `
          <label class="ts-opt"><input type="radio" name="tsOpt" value="${esc(o.key)}"${o.key === (opt && opt.key) ? " checked" : ""}>
            <span><strong>${esc(o.label)}</strong> <span class="muted">peak ${fmtTok(o.peak_input_tps)}, average ${fmtTok(o.average_input_tps)} tok/s</span><span class="hint">${esc(o.description)}</span></span></label>`).join("")}
          <p class="hint">Or cancel to keep your current traffic.</p></fieldset>` : (opt ? `<p class="hint ts-optline"><strong>${esc(opt.label)}:</strong> ${esc(opt.description)}</p>` : "");
      const der = (r.derived || []).map((x) => `<div class="review-row">
          <input type="checkbox" id="tsd-${esc(x.path)}" data-tsd="${esc(x.path)}"${st.accept[x.path] ? " checked" : ""}>
          <label for="tsd-${esc(x.path)}" class="review-name">${esc(x.label)}</label>
          <span class="review-val"><strong>${fmtInt(x.value)} tokens</strong><span class="review-now">Now: ${fmtInt(x.now)}</span></span>
          <span class="conf conf-${x.derived_from_config ? "assumed" : "derived"}">${x.derived_from_config ? "Uses your settings" : "Derived"}</span>
          <span class="review-ev">${esc(x.reason)}</span></div>`).join("");
      const gapBox = cov && (cov.raw_gap_hours || cov.gaps_as_zero) && src.kind !== "monitoring"
        ? `<label class="check ts-gaps"><input type="checkbox" id="tsGaps"${st.gaps ? " checked" : ""}> Gaps inside the data are zero traffic (exports skip intervals with no requests)</label>` : "";
      $("tsBody").innerHTML = `
        ${cov ? `<p class="ts-cov">${icon("i-info", "ico-sm")}<span><strong>Coverage:</strong> ${esc(cov.label)}.</span></p>` : ""}
        <ul class="tips review-notes"${warn ? "" : " hidden"}>${warn}</ul>
        ${r.error ? `<p class="warn-line" role="alert">${icon("i-alert")}<span>${esc(r.error)}</span></p>` : ""}
        <h3 class="ts-h">Series</h3><div class="ts-series">${rows}</div>${gapBox}
        ${opt ? `<h3 class="ts-h">Resulting week <span class="ts-legend"><i class="ts-key-new"></i>imported <i class="ts-key-cur"></i>current</span></h3>
          <div class="ts-spark-wrap">${tsSpark(opt.custom_hourly, cur)}</div>
          <p class="hint">Peak ${fmtTok(opt.peak_input_tps)} tok/s, average ${fmtTok(opt.average_input_tps)}, quietest ${fmtTok(opt.floor_input_tps)} · ${fmtTok(opt.monthly_input_tokens)} input tokens a month${r.proposal && r.proposal.basis ? ` · from ${esc(r.proposal.basis)}` : ""}${cur ? ` · now: peak ${fmtTok(Math.max.apply(null, cur))}, average ${fmtTok(meanOf(cur))}` : ""}.</p>${optHTML}` : ""}
        ${der ? `<h3 class="ts-h">Derived per-request values</h3><div class="review-list ts-derived">${der}</div>` : ""}
        <p class="ts-busy busy-note" id="tsBusy" role="status">${st.busy ? "Updating…" : ""}</p>`;
      const n = Object.values(st.accept).filter(Boolean).length;
      $("tsApply").disabled = !opt || !!r.error || st.busy;
      $("tsApply").textContent = opt ? `Apply week${n ? ` + ${n} ${n === 1 ? "value" : "values"}` : ""}` : "Apply";
      d.querySelectorAll("[data-ts]").forEach((sel) => sel.addEventListener("change", () => { st.mapping[sel.dataset.ts] = sel.value; refresh(sel.id); }));
      d.querySelectorAll('input[name="tsOpt"]').forEach((rb) => rb.addEventListener("change", () => { st.option = rb.value; render(); const el = d.querySelector(`input[name="tsOpt"][value="${rb.value}"]`); if (el) el.focus(); }));
      d.querySelectorAll("[data-tsd]").forEach((cb) => cb.addEventListener("change", () => { st.accept[cb.dataset.tsd] = cb.checked; const n2 = Object.values(st.accept).filter(Boolean).length; $("tsApply").textContent = `Apply week${n2 ? ` + ${n2} ${n2 === 1 ? "value" : "values"}` : ""}`; }));
      const g = $("tsGaps");
      if (g) g.addEventListener("change", () => { st.gaps = g.checked; refresh("tsGaps"); });
    };
    const refresh = async (focusId) => {
      st.busy = true; $("tsBusy").textContent = "Updating…"; $("tsApply").disabled = true;
      try {
        const r = await postJSON(`/api/sessions/${enc(S.name)}/traffic/import`, { mapping: st.mapping, option: st.option, gaps_as_zero: st.gaps, context: trafficContext() });
        r.source = r.source || st.res.source;
        const prevAccept = st.accept;
        st.res = r; st.option = r.option; st.accept = {};
        (r.derived || []).forEach((x) => { st.accept[x.path] = prevAccept[x.path] != null ? prevAccept[x.path] : x.checked !== false; });
      } catch (e) {
        st.res = Object.assign({}, st.res, { error: `Couldn't update: ${e.message}` });
      }
      st.busy = false;
      render();
      if (focusId && $(focusId)) $(focusId).focus();
    };
    render();
    $("tsCancel").onclick = () => d.close();
    $("tsForm").onsubmit = async (e) => {
      e.preventDefault();
      const accept = Object.keys(st.accept).filter((k) => st.accept[k]);
      $("tsApply").disabled = true;
      try {
        const out = await postJSON(`/api/sessions/${enc(S.name)}/traffic/apply`, { mapping: st.mapping, option: st.option, gaps_as_zero: st.gaps, accept, context: trafficContext(), config: collectCfg() });
        d.close();
        S.models = null;
        applySession(out.session, { resetUi: false });
        const n = accept.length;
        toast(`Applied the imported week${n ? ` and ${n} per-request ${n === 1 ? "value" : "values"}` : ""}. Saved.`);
        const o = $("railUploadResult");
        if (o) { o.className = "upload-result"; o.textContent = `Traffic: imported week applied (peak ${fmtTok(out.session.config.traffic.peak_input_tps)} tok/s).`; }
      } catch (err) {
        $("tsApply").disabled = false;
        st.res = Object.assign({}, st.res, { error: `Couldn't apply: ${err.message}` });
        render();
      }
    };
    if (!d.open) d.showModal();
    const first = d.querySelector("[data-ts]");
    (first || $("tsCancel")).focus();
    return new Promise((resolve) => d.addEventListener("close", () => resolve(), { once: true }));
  }

  // ---------------------------------------------------------------- Cloud Monitoring (GET /api/monitoring/sources, POST /api/monitoring/import)
  function openMonitoring() {
    if (!S.session) return;
    const d = $("monDlg");
    if (!$("monProject").value) $("monProject").value = currentProject() || "";
    $("monStatus").className = "mon-status";
    let picked = null;
    const find = async () => {
      const project = $("monProject").value.trim();
      if (!project) { $("monStatus").className = "mon-status err"; $("monStatus").textContent = "Enter a project ID."; $("monProject").focus(); return; }
      picked = null; $("monReview").disabled = true; $("monList").innerHTML = "";
      $("monStatus").className = "mon-status busy-note";
      $("monStatus").textContent = "Looking for models and endpoints with traffic…";
      try {
        const res = await api(`/api/monitoring/sources?project=${enc(project)}&days=${enc($("monDays").value)}`);
        $("monStatus").className = "mon-status";
        if (!res.sources.length) { $("monStatus").textContent = res.message || "No model or endpoint served requests in this window."; return; }
        $("monStatus").textContent = `${res.sources.length} ${res.sources.length === 1 ? "source" : "sources"} with traffic in the last ${res.days} days.`;
        $("monList").innerHTML = res.sources.map((s, i) => `<label class="mon-src"><input type="radio" name="monSrc" value="${esc(s.id)}"${i === 0 ? "" : ""}>
          <span class="mon-name"><strong>${esc(s.name)}</strong><span class="muted">${s.kind === "publisher" ? esc(s.publisher || "model") + " model" : "dedicated endpoint"} · ${esc(s.location)}${s.has_tokens ? "" : " · requests only"}</span></span>
          <span class="mon-num"><strong>${fmtTok(s.total_requests)}</strong><span class="muted">requests · ${fmtInt(s.active_hours)} active h</span></span></label>`).join("");
        d.querySelectorAll('input[name="monSrc"]').forEach((rb) => rb.addEventListener("change", () => { picked = rb.value; $("monReview").disabled = false; }));
      } catch (e) {
        $("monStatus").className = "mon-status err";
        $("monStatus").textContent = e.status === 401 ? `${e.message}` : e.message;
      }
    };
    $("monFind").onclick = find;
    $("monDays").onchange = () => { if ($("monList").innerHTML) find(); };
    $("monCancel").onclick = () => d.close();
    $("monForm").onsubmit = async (e) => {
      e.preventDefault();
      if (!picked) return;
      if (!(await ensureWritable())) return;
      $("monReview").disabled = true;
      $("monStatus").className = "mon-status busy-note";
      $("monStatus").textContent = "Reading hourly requests and tokens…";
      try {
        const res = await postJSON("/api/monitoring/import", { project: $("monProject").value.trim(), source: picked, days: +$("monDays").value, session: S.name, context: trafficContext() });
        d.close();
        await openTrafficReview(res, null);
      } catch (err) {
        $("monReview").disabled = false;
        $("monStatus").className = "mon-status err";
        $("monStatus").textContent = err.message;
      }
    };
    d.showModal();
    if ($("monProject").value) find(); else $("monProject").focus();
  }

  function bindDrop(zone, input, onFiles) {
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("dragover"); if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files); });
    input.addEventListener("change", () => { if (input.files.length) onFiles(Array.from(input.files)); input.value = ""; });
  }

  // ---------------------------------------------------------------- first run
  function showFirstRun() {
    $("firstRun").hidden = false;
    $("mainView").hidden = true;
    $("btnFrCancel").hidden = !S.session;
    $("frName").value = suggestName("my-workload");
    document.body.classList.add("first-run-open");
    $("frTitle").focus && $("frTitle").setAttribute("tabindex", "-1");
  }
  function showMain() {
    $("firstRun").hidden = true;
    $("mainView").hidden = false;
    document.body.classList.remove("first-run-open");
    setTimeout(() => estimatesNotice("main"), 300);
  }

  // First-use notice about throughput estimates: once on the main page and once on first opening the hardware
  // drawer (separate localStorage flags). After that, the inline cautions next to each estimate remain.
  const NOTICE_KEYS = { main: "dilemma-seen-estimates-notice", hw: "dilemma-seen-hw-estimates-notice" };
  async function estimatesNotice(where) {
    let seen = false;
    try { seen = localStorage.getItem(NOTICE_KEYS[where]) === "1"; } catch (_) { /* private mode: show once per load */ }
    if (seen || S.noticeShown === where || $("dlg").open) return;
    if (where === "main" && $("mainView").hidden) return;
    S.noticeShown = where;
    const lead = where === "hw"
      ? "<p>Each accelerator's throughput per chip is an estimate: from a reference benchmark on that GPU, scaled from one, or a rough figure from its specs.</p>"
      : "<p>Throughput figures here are estimates from reference benchmarks or GPU specs. They may not represent your workload.</p>";
    const res = await dialog({
      title: where === "hw" ? "Throughput per chip is an estimate" : "Throughput figures are estimates",
      body: lead + "<p>Prompt sizes, thinking and output length, long prompts and caching all change throughput, often by several times. In our runs, unbudgeted thinking cut throughput about 4×.</p>" +
        "<p>For a decision, run a benchmark with your own prompts and traffic.</p>",
      okLabel: "I understand", cancelLabel: "Run a benchmark", cancelValue: "bench",
    });
    try { localStorage.setItem(NOTICE_KEYS[where], "1"); } catch (_) { /* ignore */ }
    if (res === "bench") {
      if (S.drawer) closeDrawer();
      selectTab("bench", true);
      $("panel-bench").scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
    }
  }
  function frName(statusEl) {
    const v = $("frName").value.trim();
    const msg = validName(v || "");
    if (!v || msg) { statusEl.className = "fr-status err"; statusEl.textContent = msg || "Name the workload first."; $("frName").focus(); return null; }
    return v;
  }
  async function frCreate(name, cfg) {
    return postJSON(`/api/sessions/${enc(name)}`, { config: Object.assign({ name }, cfg || {}) });
  }
  function defaultsCfg() {
    const d = clone((S.catalog && S.catalog.defaults) || {});
    return normalizeCfg(d);
  }

  function initFirstRun() {
    bindDrop($("frDrop"), $("frUploadInput"), async (files) => {
      const st = $("frUploadStatus");
      const name = frName(st);
      if (!name) return;
      st.className = "fr-status";
      st.textContent = `Reading ${files.length} ${files.length === 1 ? "file" : "files"}…`;
      try {
        const { docs, rest } = await splitDescribe(Array.from(files), "auto");
        const created = await frCreate(name, null);
        const cfg = normalizeCfg(clone(created.config));
        S.cfg = cfg;
        let saved = created;
        let tsFiles = [];
        if (rest.length) {
          const ing = await uploadTo(created.name, rest, "auto");
          tsFiles = rest.filter((f) => (ing.timeseries_files || []).indexOf(f.name) >= 0);
          applyIngest(cfg, ing);
          st.textContent = summarizeIngest(ing).join(". ") + ". Sizing hardware…";
          saved = await postJSON(`/api/sessions/${enc(created.name)}`, { config: Object.assign(cfg, { name: created.name }) });
        }
        await loadSessions();
        S.autoEstimated[saved.name] = false;
        applySession(saved, { resetUi: true });
        st.textContent = "";
        if (tsFiles.length) await runTrafficImport(tsFiles);
        for (const d of docs) await runDescribe(d);
      } catch (e) {
        st.className = "fr-status err";
        st.textContent = `Upload failed: ${e.message}. Check the files and try again.`;
      }
    });
    $("btnFrExample").addEventListener("click", async () => {
      const st = $("frExampleStatus");
      const name = frName(st);
      if (!name) return;
      const ex = S.sessions.find((s) => s.example || s.is_demo) || S.sessions.find((s) => s.name === "blog-example");
      const exName = ex ? ex.name : "blog-example";
      st.className = "fr-status";
      st.textContent = "Copying the example…";
      try {
        const src = await api(`/api/sessions/${enc(exName)}`);
        const res = await postJSON(`/api/sessions/${enc(name)}`, { config: Object.assign(clone(src.config), { name }), ui_state: src.ui_state || {}, copy_of: exName });
        await loadSessions();
        applySession(res, { resetUi: true });
        st.textContent = "";
      } catch (e) {
        st.className = "fr-status err";
        st.textContent = `Couldn't copy the example: ${e.message}.`;
      }
    });
    $("frNumbers").addEventListener("submit", async (e) => {
      e.preventDefault();
      const st = $("frNumbersStatus");
      const name = frName(st);
      if (!name) return;
      const peak = +$("fr-peak").value, avg = +$("fr-avg").value, floor = +$("fr-floor").value;
      // Find a weekday busy window and weekend load that reproduce the three numbers exactly.
      let shape = null, lastErr = null;
      const windows = [[13, 20], [12, 21], [14, 18], [10, 22], [15, 17], [8, 23]];
      const weekends = [0.15, 0.3, 0.05, 0.5, 0.75, 0.0, 1.0];
      for (const w of windows) {
        for (const wk of weekends) {
          try { parametric(peak, avg, floor, w, wk); shape = { w, wk }; break; } catch (err) { lastErr = err; }
        }
        if (shape) break;
      }
      if (!shape) { st.className = "fr-status err"; st.textContent = (lastErr && lastErr.message) || "These numbers don't describe a week."; return; }
      const cfg = defaultsCfg();
      cfg.traffic.busy_hours_utc = shape.w; cfg.traffic.weekend_factor = shape.wk;
      cfg.traffic.peak_input_tps = peak; cfg.traffic.average_input_tps = avg; cfg.traffic.floor_input_tps = floor; cfg.traffic.hourly_csv = null;
      cfg.requests.avg_input_tokens = +$("fr-in").value; cfg.requests.avg_output_tokens = +$("fr-out").value;
      cfg.constraints.latency_budget_s = +$("fr-lat").value;
      st.className = "fr-status";
      st.textContent = "Sizing hardware for this workload. This takes a few seconds…";
      try {
        const created = await frCreate(name, cfg);
        await loadSessions();
        applySession(created, { resetUi: true });
        st.textContent = "";
      } catch (err) {
        st.className = "fr-status err";
        st.textContent = `Couldn't create the workload: ${err.message}.`;
      }
    });
    $("btnFrCancel").addEventListener("click", showMain);
  }

  // ---------------------------------------------------------------- drawers
  function openDrawer(kind, returnEl) {
    S.drawer = kind;
    S.drawerReturn = returnEl || document.activeElement;
    $("drawerTitle").textContent = kind === "model" ? "Compare models" : "Compare hardware";
    $("drawer").hidden = false;
    $("drawerScrim").hidden = false;
    document.body.style.overflow = "hidden";
    syncViewToggle();
    renderDrawer();
    $("drawerClose").focus();
    if (kind === "model" && !S.models && !S.modelsLoading) fetchModels();
    if (kind === "hw" && !S.hw && !S.hwLoading) fetchHardware();
    if (kind === "hw") setTimeout(() => estimatesNotice("hw"), 200);
  }
  function closeDrawer() {
    if (!S.drawer) return;
    S.drawer = null;
    $("drawer").hidden = true;
    $("drawerScrim").hidden = true;
    document.body.style.overflow = "";
    hideTip();
    if (S.drawerReturn && S.drawerReturn.focus) S.drawerReturn.focus();
  }
  function syncViewToggle() {
    const v = S.drawerView[S.drawer] || "table";
    $("viewTable").setAttribute("aria-pressed", v === "table");
    $("viewCards").setAttribute("aria-pressed", v === "cards");
  }
  function renderDrawer() {
    if (S.drawer === "model") renderModelDrawer();
    else if (S.drawer === "hw") renderHwDrawer();
  }
  const skeletonRows = (cols, n) => Array.from({ length: n }, () => `<tr class="skeleton">${Array.from({ length: cols }, () => "<td><span></span></td>").join("")}</tr>`).join("");

  async function fetchModels() {
    if (!S.cfg) return;
    S.modelsLoading = true;
    if (S.drawer === "model") renderModelDrawer();
    try {
      const c = collectCfg();
      S.models = await postJSON("/api/recommend/model", { session: S.name, use_case: c.use_case, requests: c.requests, traffic: c.traffic, constraints: c.constraints, project: c.project, region: c.region });
    } catch (e) {
      S.models = { error: e.message, recommendations: [] };
    }
    S.modelsLoading = false;
    if (S.drawer === "model") renderModelDrawer();
  }

  function longestPrompt() {
    const r = S.cfg.requests;
    if (+r.pct_over_128k > 0) return +r.large_mean || 188000;
    if (+r.pct_over_32k > 0) return +r.medium_mean || 69000;
    return (+r.avg_input_tokens || 0) * 2;
  }
  function fitsCell(m) {
    let fits = m.fits_longest_prompt;
    if (fits === undefined && m.max_context) fits = m.max_context >= longestPrompt();
    if (fits === undefined) return "—";
    return fits ? `<span class="yes">${icon("i-check", "ico-sm")}Yes</span>` : `<span class="no">${icon("i-x", "ico-sm")}No</span>`;
  }
  function precisionCell(m) {
    const p = m.recommended_precision_label || String(m.recommended_precision || "").toUpperCase();
    if (!p) return "—";
    return m.needs_quality_check ? `${p} <span class="sub warn-text">check quality</span>` : p;
  }
  // Live sizes come with Hub precision (684.53B): show 3 significant digits at most.
  const fmtB = (x) => { const v = +x; return v >= 100 ? String(Math.round(v)) : v >= 10 ? String(Math.round(v * 10) / 10) : String(Math.round(v * 100) / 100); };
  function paramsText(m) {
    if (!m.params_b) return "—";
    return m.active_params_b && m.active_params_b < m.params_b * 0.7 ? `${fmtB(m.params_b)}B, ${fmtB(m.active_params_b)}B active` : `${fmtB(m.params_b)}B`;
  }
  const ctxText = (m) => (m.max_context ? Math.round(m.max_context / 1024) + "K" : "—");
  // Provenance of a model's metadata (backend modelinfo.annotate: data_source + field_sources).
  const SRC_LABEL = { hf: "Hugging Face", model_garden: "Model Garden", rules: "built-in estimate" };
  const FIELD_LABEL = [["params_b", "Size"], ["active_params_b", "Active parameters"], ["max_context", "Context"], ["released", "Release date"], ["native_precision", "Native precision"], ["precisions", "Precisions"], ["family", "Family"]];
  function provText(m) {
    const d = m.data_source;
    if (!d) return "";
    if (d.kind === "rules") return "Data: built-in estimate";
    return `Data: Hugging Face, updated ${d.updated_at || "recently"}${(d.estimated_fields || []).length ? " (some values estimated)" : ""}`;
  }
  function provHTML(m) {
    const d = m.data_source;
    if (!d || m.custom) return "";
    const fs = m.field_sources || {};
    const parts = FIELD_LABEL.filter(([k]) => fs[k]).map(([k, l]) => `${l}: ${SRC_LABEL[fs[k]] || fs[k]}`);
    const tip = parts.join("; ") + (m.license ? `. License: ${m.license}` : "") + ((d.notes || []).length ? `. ${d.notes.join("; ")}` : "") + (d.hf_id ? `. Source: huggingface.co/${d.hf_id}` : "");
    return `<span class="sub prov${d.kind === "hf" ? "" : " prov-est"}" title="${esc(tip)}">${esc(provText(m))}</span>`;
  }
  function whyHTML(m, repeated) {
    if (m.match_score == null) return "—";
    let body = "";
    if (m.score_breakdown && typeof m.score_breakdown === "object") {
      body = `<table>${Object.keys(m.score_breakdown).map((k) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td>${esc(m.score_breakdown[k])}</td></tr>`).join("")}</table>`;
    } else {
      const items = (m.reasons || []).concat((m.warnings || []).filter((w) => !repeated.has(w)));
      body = items.length ? `<ul>${items.map((x) => `<li>${esc(ap(x))}</li>`).join("")}</ul>` : "<p>No breakdown available.</p>";
    }
    return `<details class="why"><summary aria-label="Match ${m.match_score}%, why">${m.match_score}%</summary><div class="why-pop">${body}</div></details>`;
  }

  // ---- model catalog: one row per model, precision and prequantized checkpoints folded in, older/specialty hidden
  const QUANT_RE = /[-_.](fp8|mxfp4|fp4|nvfp4|awq|gptq|w8a8|w4a16|w8a16|int4|int8|bnb[-_\w]*|gguf|dynamic|quantized?)(?=$|[-_.])/gi;
  const LEGACY_RE = /(^|[\/@\-_ ])(gemma-?(2b|7b|1\.1|2[-_ ])|gemma 2\b|gemma 1\b|llama-?2|vicuna|falcon|code-?llama|mistral-7b(-instruct)?-v0\.[12]|qwen1\.5|qwen-\d|qwen2(?![\.\d])|dolly|bloom|flan|mpt-|starcoder(?!2)|phi-?[12](\b|[-_])|baichuan|chatglm|stablelm|yi-\d)/i;
  const SPECIALTY_RE = [
    [/flux|paligemma|blip|llava|wan[-_]?2|stable[-_]?diffusion|sdxl|imagen|whisper|clip|siglip|owl|segment|dino|\bvit\b|t2v|i2v|tts|bark|musicgen|vision|image|video|audio/i, "image, video or audio"],
    [/embed|bge|\be5[-_]|gte[-_]|rerank|retriev/i, "embeddings"],
    [/codegemma|code-?llama|starcoder|qwen[\d.]*-?coder|codestral|deepseek-coder|qodo|coder/i, "code"],
    [/medgemma|txgemma|med42|bio|clinical|medical/i, "medical"],
    [/shieldgemma|guard|safety/i, "safety"],
    [/translate|madlad|nllb|seamless/i, "translation"],
  ];
  const PREC_TOKENS = { fp8: "fp8", bf16: "bf16", fp16: "bf16", int8: "fp8", "8bit": "fp8", "8-bit": "fp8", int4: "int4", "4bit": "int4", "4-bit": "int4", w4a16: "int4", awq: "int4", gptq: "int4", mxfp4: "mxfp4" };
  const mdlId = (m) => String(m.model_garden_id || m.hf_id || m.id || "").toLowerCase();
  function baseKey(m) {
    let id = mdlId(m).split("@").pop().split("/").pop();
    id = id.replace(QUANT_RE, "");
    for (let i = 0; i < 3; i++) id = id.replace(/[-_](it|instruct|chat|hf|pt|base|v\d+(\.\d+)?-hf)$/i, "");
    return id;
  }
  const isQuantVariant = (m) => { QUANT_RE.lastIndex = 0; return QUANT_RE.test(mdlId(m).split("@").pop()); };
  const hasInstructMark = (m) => /[-_](it|instruct|chat)([-_]|$)|thinking/i.test(mdlId(m).split("@").pop()) || /instruct|\bchat\b/i.test(m.name || "");
  // Discovered models often carry their id as the name ("qwen3-30b-a3b"); show "Qwen3 30B A3B".
  function prettyName(n) {
    n = shortModelName(n);
    if (!n || /[A-Z]/.test(n) || !/[-_]/.test(n)) return n;
    return n.split(/[-_]/).filter(Boolean).map((t, i) => /^\d+(\.\d+)?[bmk]$/i.test(t) || /^a\d+(\.\d+)?b$/i.test(t) ? t.toUpperCase()
      : /^(it|fp8|awq|gptq|moe|vl)$/i.test(t) ? t.toUpperCase()
      : i === 0 || /^[a-z]{3,}$/.test(t) ? t.charAt(0).toUpperCase() + t.slice(1) : t).join(" ");
  }
  const isBaseCheckpoint = (m) => /[-_](pt|base)$/i.test(mdlId(m).split("@").pop()) || (/\b(pt|base)\b/i.test(m.name || "") && !/instruct|\bit\b|chat/i.test(m.name || ""));
  function familyOf(m, clean) {
    if (clean && m.family) return m.family;
    const s = (m.name || "") + " " + mdlId(m);
    let r;
    if ((r = s.match(/gemma[-_ ]?(\d)(?:\D|$)/i))) return "Gemma " + r[1];
    if (/gemma/i.test(s)) return "Gemma";
    if ((r = s.match(/llama[-_ ]?(\d(?:\.\d)?)/i))) return "Llama " + r[1];
    if ((r = s.match(/qwen[-_ ]?(\d(?:\.\d)?)/i))) return "Qwen " + r[1];
    if (/qwen/i.test(s)) return "Qwen";
    if (/mi[sx]tral|ministral|codestral/i.test(s)) return "Mistral";
    if (/deepseek/i.test(s)) return "DeepSeek";
    if ((r = s.match(/phi[-_ ]?(\d)/i))) return "Phi " + r[1];
    if (/gpt-oss/i.test(s)) return "gpt-oss";
    const f = String(m.family || "Other").replace(/-ai$|ai$/i, "").replace(/[-_]/g, " ").trim();
    return f ? f.charAt(0).toUpperCase() + f.slice(1) : "Other";
  }
  function newestKey(m) {
    if (m.released && /^\d{4}-\d{2}/.test(m.released)) return +m.released.slice(0, 4) * 12 + +m.released.slice(5, 7);
    const g = parseFloat(m.generation);
    const v = isFinite(g) ? g : (() => { const r = familyOf(m, false).match(/(\d+(?:\.\d)?)/); return r ? +r[1] : 0; })();
    return v; // generation within families; mixed families sort by their version number
  }
  // Groups the scored rows; merges catalog fields (legacy, precisions, prequantized...) when the backend sends them.
  function isSelectedModel(m, sel) {
    if (!sel || !m) return false;
    const ids = [sel.model_garden_id, sel.hf_id].filter(Boolean).map((x) => String(x).toLowerCase());
    const mine = [m.model_garden_id, m.hf_id].concat(Array.isArray(m.aliases) ? m.aliases : []).filter(Boolean).map((x) => String(x).toLowerCase());
    return ids.length > 0 && mine.some((x) => ids.indexOf(x) >= 0);
  }
  function modelGroups() {
    const res = S.models || {};
    const recs = res.recommendations || [];
    const cat = {};
    ((S.catalog && S.catalog.models) || []).forEach((m) => { if (m.model_garden_id) cat[m.model_garden_id] = m; });
    const clean = Array.isArray(res.families) || Array.isArray(S.catalog && S.catalog.families) || recs.some((m) => Array.isArray(m.precisions));
    const task = S.cfg.use_case.task_type;
    const groups = new Map();
    recs.forEach((r0) => {
      const m = Object.assign({}, cat[r0.model_garden_id] || {}, r0);
      const key = clean && Array.isArray(m.precisions) ? mdlId(m) : baseKey(m);
      let g = groups.get(key);
      if (!g) { g = { key, members: [] }; groups.set(key, g); }
      g.members.push(m);
    });
    const sel = chosenModel();
    const out = [];
    // Which row is the selection is decided here, from the current selection: a cached response's is_selected was
    // computed for whatever was selected when it was fetched (stale after choosing another model).
    const isSelRow = (m) => isSelectedModel(m, sel);
    if (sel.model_garden_id && !recs.some(isSelRow)) {
      const cm = S.cfg.model;
      groups.set("__selected", { key: "__selected", members: [Object.assign({}, cat[sel.model_garden_id] || {}, { model_garden_id: sel.model_garden_id, hf_id: sel.hf_id, name: modelName(sel), params_b: cm.params_b, active_params_b: cm.active_params_b, custom: !cat[sel.model_garden_id], is_selected: true })] });
    }
    groups.forEach((g) => {
      const rank = (m) => (isQuantVariant(m) ? 2 : 0) + (isBaseCheckpoint(m) ? 1 : 0);
      g.members.sort((a, b) => rank(a) - rank(b) || (b.match_score || 0) - (a.match_score || 0));
      const m = g.members[0];
      const prequant = Object.assign({}, m.prequantized || {});
      const variants = [];
      g.members.slice(1).forEach((v) => {
        if (isQuantVariant(v)) {
          const q = (mdlId(v).match(/fp8|mxfp4|nvfp4|fp4|awq|gptq|w8a8|w4a16|int8|int4/) || ["quantized"])[0];
          if (q === "fp8" && !prequant.fp8) prequant.fp8 = v.model_garden_id;
          variants.push(q.toUpperCase() + " checkpoint");
        } else if (isBaseCheckpoint(v) || (hasInstructMark(m) && !hasInstructMark(v))) variants.push("base checkpoint");
        else variants.push(shortModelName(v.name) || baseKey(v));
      });
      const hay = (m.name || "") + " " + mdlId(m) + " " + (m.hf_id || "");
      let specialty = m.specialty !== undefined ? m.specialty : null;
      if (m.specialty === undefined) for (const [re, label] of SPECIALTY_RE) if (re.test(hay)) { specialty = label; break; }
      const legacy = m.legacy !== undefined ? !!m.legacy : LEGACY_RE.test(hay);
      const textModel = !/image|video|audio|embeddings/.test(specialty || "");
      const isSel = g.members.some(isSelRow);
      out.push(Object.assign({}, m, {
        family: familyOf(m, clean), variants, prequantized: prequant, specialty, legacy, textModel, isSel,
        precisions: Array.isArray(m.precisions) ? m.precisions : ["bf16", "fp8"],
        // Code models are a specialty unless the task is code generation.
        hiddenByDefault: legacy || (!!specialty && !(specialty === "code" && task === "coding")),
        search: (hay + " " + familyOf(m, clean) + " " + (specialty || "") + " " + variants.join(" ")).toLowerCase(),
      }));
    });
    return { groups: out, clean };
  }
  function parseQuery(q) {
    const toks = String(q || "").toLowerCase().split(/\s+/).filter(Boolean);
    let prec = null;
    const words = toks.filter((t) => { if (PREC_TOKENS[t]) { prec = PREC_TOKENS[t]; return false; } return true; });
    const phrase = words.length > 1 ? squash(words.join("")) : null;
    return { words, prec, phrase };
  }
  // Phrase match on word boundaries: "gemma 4" matches "Gemma 4 26B", "gemma-4-31b-it" and "gemma4", not "paligemma2-…-448".
  const reEsc = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const phraseMatch = (m, q) => {
    if (!q.re) {
      const last = q.words[q.words.length - 1];
      q.re = new RegExp("(^|[^a-z0-9])" + q.words.map(reEsc).join("[\\s\\-_.]*") + (/\d$/.test(last) ? "(?![0-9])" : ""), "i");
    }
    return [m.name, mdlId(m), m.hf_id, m.family].some((f) => f && q.re.test(String(f)));
  };
  let phraseCache = { key: null, hits: false };
  function phraseHits(groups, q) {
    const key = q.phrase + "|" + groups.length;
    if (phraseCache.key !== key) phraseCache = { key, hits: groups.some((m) => phraseMatch(m, q)) };
    return phraseCache.hits;
  }
  const squash = (s) => s.replace(/[\s\-_.\/@]+/g, "");
  const SIZES = { all: [0, Infinity, "Any size"], s: [0, 10, "Under 10B"], m: [10, 40, "10–40B"], l: [40, 100, "40–100B"], xl: [100, Infinity, "Over 100B"] };
  function fitsLongest(m) {
    if (m.fits_longest_prompt !== undefined) return !!m.fits_longest_prompt;
    return m.max_context ? m.max_context >= longestPrompt() : true;
  }
  function filterModels(groups, except) {
    const f = S.mdl, q = parseQuery(f.q);
    return groups.filter((m) => {
      if (m.isSel) return true;
      if (!f.older && !m.textModel) return false; // image, video, audio and embedding models: only with the toggle
      if (!f.older && m.hiddenByDefault && !q.words.length) return false;
      if (except !== "family" && f.family !== "all" && m.family !== f.family) return false;
      if (f.size !== "all") { const [lo, hi] = SIZES[f.size]; const p = +m.params_b || 0; if (!(p >= lo && p < hi)) return false; }
      if (f.fits && !fitsLongest(m)) return false;
      if (q.words.length) {
        const hs = m.search, hq = squash(hs);
        if (!q.words.every((w) => hs.indexOf(w) >= 0 || hq.indexOf(squash(w)) >= 0)) return false;
        // "gemma 4" means the phrase: prefer models that contain it as written (gemma-4, Gemma 4, gemma4).
        if (q.phrase && phraseHits(groups, q) && !phraseMatch(m, q)) return false;
        if (!f.older && m.hiddenByDefault && m.legacy) return false; // search still finds specialty models, not older ones
      }
      return true;
    });
  }
  const usdOf = (m) => (m.est_usd_per_m_input != null ? +m.est_usd_per_m_input : m.usd_per_m_input != null ? +m.usd_per_m_input : null);
  function sortModels(list) {
    const k = S.mdl.sort;
    const num = (v, d) => (v == null || !isFinite(v) ? d : v);
    const cmp = {
      match: (a, b) => num(b.match_score, -1) - num(a.match_score, -1),
      cost: (a, b) => num(usdOf(a), Infinity) - num(usdOf(b), Infinity),
      size: (a, b) => num(+a.params_b, Infinity) - num(+b.params_b, Infinity),
      newest: (a, b) => newestKey(b) - newestKey(a) || num(b.match_score, -1) - num(a.match_score, -1),
    }[k];
    return list.slice().sort((a, b) => (a.isSel ? -1 : b.isSel ? 1 : cmp(a, b) || (b.match_score || 0) - (a.match_score || 0)));
  }
  function modelTips(all) {
    const res = S.models || {};
    let tips = Array.isArray(res.tips) ? res.tips.slice() : null;
    let repeated = new Set();
    if (!tips) {
      const count = {};
      all.forEach((m) => (m.warnings || []).forEach((w) => { count[w] = (count[w] || 0) + 1; }));
      tips = Object.keys(count).filter((w) => count[w] >= Math.max(3, all.length / 2));
      repeated = new Set(tips);
    }
    const k = res.calibration_k;
    if (k && Math.abs(k - 1) > 0.01) tips.push(`Estimates for other models are their own estimates × ${k.toFixed(2)}, the ratio your selected model's throughput has to its own estimate on the same hardware.`);
    return { tips: tips.map(ap), repeated };
  }
  // One short item per precision that exists for the model, tagged with how it is obtained; the recommended one in bold
  // with its own quality note (never a note naming another precision).
  function precText(m) {
    const rec = String(m.recommended_precision || "").toLowerCase();
    const opts = Array.isArray(m.precision_options) ? m.precision_options : availablePrecisions(m).map((p) => ({
      precision: p, label: precLabel(p), recommended: p === rec,
      how_label: p === "bf16" || p === m.native_precision ? "native" : p === "int4" || p === "mxfp4" || (m.prequantized && m.prequantized.fp8) || m.fp8_only ? "prequantized checkpoint" : "quantized on the fly by vLLM",
      quality_note: p === rec && p !== "bf16" && p !== m.native_precision && m.needs_quality_check ? (p === "int4" || p === "mxfp4" ? "4-bit: check quality on your prompts." : "FP8: check quality on your prompts.") : null,
    }));
    return `<ul class="prec-list">${opts.map((o) => {
      const main = `${esc(o.label)} <span class="prec-how">${esc(o.how_label)}</span>`;
      const size = o.size_text ? `<span class="prec-fit${o.fits_on ? "" : " warn-text"}"${o.weights_note ? ` title="${esc(o.weights_note)}"` : ""}>${esc(o.size_text)}, ${esc(ap(o.fit_text || ""))}</span>` : "";
      return `<li${o.recommended ? ' class="rec"' : ""}>${o.recommended ? `<strong>${main}</strong><span class="visually-hidden"> (recommended)</span>` : main}${size}${o.quality_note ? `<span class="warn-text">${esc(o.quality_note)}</span>` : ""}</li>`;
    }).join("")}</ul>`;
  }

  function renderModelDrawer() {
    const body = $("drawerBody");
    const taskLabel = $("f-task").selectedOptions[0] ? $("f-task").selectedOptions[0].textContent.toLowerCase() : "";
    const f = S.mdl;
    let html = `<div class="drawer-tools"><p class="drawer-note">Ranked for ${esc(taskLabel)} at the lowest cost. Change the task under Prompts.</p>
      <button type="button" class="btn btn-sm" id="btnRefreshModels" data-needs-idle>${icon("i-refresh", "ico-sm")}Refresh the model list</button></div>`;
    if (S.modelsLoading || !S.models) {
      html += `<p class="drawer-note" role="status">Scoring open models for this workload…</p><div class="table-wrap"><table class="tbl cmp"><tbody>${skeletonRows(7, 6)}</tbody></table></div>`;
      body.innerHTML = html;
      wireModelDrawer();
      return;
    }
    if (S.models.error) html += `<p class="warn-line">${icon("i-alert")}<span>Couldn't score models: ${esc(S.models.error)}. <button type="button" class="link-btn" id="btnRetryModels">Try again</button></span></p>`;
    const sorts = [["match", "Best match"], ["cost", "Lowest est. cost"], ["size", "Smallest first"], ["newest", "Newest first"]];
    html += `<div class="mdl-filters" role="search" aria-label="Find a model">
      <div class="mdl-row">
        <span class="search-box"><label class="visually-hidden" for="mdlSearch">Search models</label>
          <input type="search" id="mdlSearch" placeholder="Search by name or family, e.g. Gemma 4" value="${esc(f.q)}" autocomplete="off" spellcheck="false" aria-describedby="mdlCount mdlQueryNote">
        </span>
        <span class="field row mdl-sort"><label for="mdlSort">Sort</label><select id="mdlSort">${sorts.map(([k, l]) => `<option value="${k}"${k === f.sort ? " selected" : ""}>${l}</option>`).join("")}</select></span>
      </div>
      <div class="facet-chips" id="mdlFamilies" role="group" aria-label="Family"></div>
      <div class="mdl-row mdl-row-wrap">
        <span class="seg seg-wrap" role="group" aria-label="Size">${Object.keys(SIZES).map((k) => `<button type="button" class="seg-btn" data-mdl-size="${k}" aria-pressed="${f.size === k}">${SIZES[k][2]}</button>`).join("")}</span>
        <label class="check chip-check"><input type="checkbox" id="mdlFits"${f.fits ? " checked" : ""}> Fits my longest prompts</label>
        <label class="check chip-check switch"><input type="checkbox" id="mdlOlder" role="switch"${f.older ? " checked" : ""}> <span id="mdlOlderLabel">Show older and specialty models</span></label>
      </div>
      <p class="hint" id="mdlQueryNote" role="note" hidden></p>
    </div>
    <div id="mdlTips"></div>
    <div id="mdlResults"></div>`;
    const cm = S.cfg.model;
    html += `<details class="custom-model"><summary>Use a model that isn't listed</summary>
      <div class="custom-grid">
        <div class="field"><label for="cmMg">Model Garden ID</label><input type="text" id="cmMg" spellcheck="false" value="${esc(cm.model_garden_id || "")}"></div>
        <div class="field"><label for="cmHf">Hugging Face ID</label><input type="text" id="cmHf" spellcheck="false" value="${esc(cm.hf_id || "")}"></div>
        <div class="field"><label for="cmParams">Total parameters, billions</label><input type="number" id="cmParams" min="0" step="0.1" value="${esc(cm.params_b || "")}"></div>
        <div class="field"><label for="cmActive">Active parameters, billions</label><input type="number" id="cmActive" min="0" step="0.1" value="${esc(cm.active_params_b || "")}"></div>
        <div class="field"><label for="cmPrec">Precision</label><select id="cmPrec"><option value="bf16">BF16</option><option value="fp8">FP8</option></select></div>
      </div>
      <p class="hint" id="cmStatus" role="status"></p>
      <div class="pop-actions"><button type="button" class="btn" id="btnInspectHf">Read sizes from Hugging Face</button><button type="button" class="btn btn-primary" id="btnUseCustom" data-needs-idle>Use this model</button></div>
    </details>`;
    body.innerHTML = html;
    const prec = $("cmPrec"); if (prec) prec.value = (cm.precision || "bf16").toLowerCase();
    wireModelDrawer();
    renderModelResults();
  }

  function renderModelResults() {
    const box = $("mdlResults");
    if (!box || !S.models) return;
    const { groups } = modelGroups();
    const f = S.mdl, q = parseQuery(f.q);
    // Family chips: counts honor every filter except the family itself.
    const forFam = filterModels(groups, "family");
    const famCount = {};
    forFam.forEach((m) => { famCount[m.family] = (famCount[m.family] || 0) + 1; });
    const fams = Object.keys(famCount).sort((a, b) => famCount[b] - famCount[a] || a.localeCompare(b));
    if (f.family !== "all" && !famCount[f.family]) fams.unshift(f.family);
    const shown = S.mdlAllFams ? fams : fams.slice(0, 9);
    const focusedFam = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.mdlFam : null;
    $("mdlFamilies").innerHTML = `<button type="button" class="fchip" data-mdl-fam="all" aria-pressed="${f.family === "all"}">All families <span class="fchip-n">${forFam.length}</span></button>` +
      shown.map((x) => `<button type="button" class="fchip" data-mdl-fam="${esc(x)}" aria-pressed="${f.family === x}">${esc(x)} <span class="fchip-n">${famCount[x] || 0}</span></button>`).join("") +
      (fams.length > 9 ? `<button type="button" class="link-btn fchip-more" id="mdlMoreFams" aria-expanded="${!!S.mdlAllFams}">${S.mdlAllFams ? "Fewer families" : `${fams.length - 9} more`}</button>` : "");
    $("mdlFamilies").querySelectorAll("[data-mdl-fam]").forEach((b) => b.onclick = () => { S.mdl.family = b.dataset.mdlFam; S.mdl.limit = 40; renderModelResults(); });
    const more = $("mdlMoreFams"); if (more) more.onclick = () => { S.mdlAllFams = !S.mdlAllFams; renderModelResults(); $("mdlMoreFams") && $("mdlMoreFams").focus(); };
    if (focusedFam) { const b = $("mdlFamilies").querySelector(`[data-mdl-fam="${CSS.escape(focusedFam)}"]`); if (b) b.focus(); }
    const hiddenN = groups.filter((m) => m.hiddenByDefault && !m.isSel).length;
    $("mdlOlderLabel").textContent = `Show older and specialty models${hiddenN ? ` (${hiddenN})` : ""}`;
    const note = $("mdlQueryNote");
    note.hidden = !q.prec;
    if (q.prec) note.innerHTML = `${q.prec.toUpperCase()} isn't a separate model: it's a precision. Pick a model below and it is planned at ${q.prec.toUpperCase()}${q.prec === "fp8" ? " (prequantized when a checkpoint exists, otherwise quantized on the fly)" : ""}. You can switch precision next to the verdict at any time.`;
    const { tips, repeated } = modelTips(groups);
    $("mdlTips").innerHTML = tips.length ? `<ul class="tips">${tips.map((t) => `<li>${icon("i-info", "ico-sm")}<span>${esc(t)}</span></li>`).join("")}</ul>` : "";
    const list = sortModels(filterModels(groups));
    const pinned = list.find((m) => m.isSel);
    const nonSel = list.filter((m) => !m.isSel);
    const visible = (pinned ? [pinned] : []).concat(nonSel.slice(0, f.limit));
    const estN = groups.filter((m) => m.data_source && m.data_source.kind !== "hf" && !m.custom).length;
    const meta = (S.catalog && S.catalog.model_metadata) || {};
    let provNote = "";
    // Gated repos (Llama, some Gemma) whose config.json needed a token: point to Settings.
    const gatedN = groups.filter((m) => !m.custom && m.data_source && m.data_source.kind !== "hf" && (m.data_source.notes || []).some((n) => /gated/i.test(n))).length;
    const hfTok = S.settings && S.settings.hf_token;
    const gatedNote = gatedN ? (hfTok && hfTok.set
      ? ` ${gatedN} gated ${gatedN === 1 ? "model uses" : "models use"} built-in values; if they don't update after the background refresh, accept their licenses on Hugging Face with the token's account.`
      : ` ${gatedN} gated ${gatedN === 1 ? "model uses" : "models (such as Llama) use"} built-in values. <button type="button" class="link-btn" id="btnMdlSettings">Add a Hugging Face token in Settings</button> to load them.`) : "";
    if (estN) provNote = `<p class="hint mdl-prov-note" role="note">${icon("i-info", "ico-sm")} Sizes, context and release dates come from Hugging Face where available. ${estN} of ${groups.length} models still use built-in estimates for some values (see “Data:” under each model; hover for the source of each field).${gatedNote}${meta.refreshing ? " Hugging Face metadata is being refreshed in the background." : ""}</p>`;
    let html = provNote + `<p class="mdl-count" id="mdlCount" role="status" aria-live="polite">${nonSel.length ? `${nonSel.length} ${nonSel.length === 1 ? "model" : "models"}${f.family !== "all" ? " in " + esc(f.family) : ""}${q.words.length ? ` matching “${esc(q.words.join(" "))}”` : ""}` : "No models match. Clear the search or a filter."}${pinned ? " · your selection is pinned on top" : ""}</p>`;
    const top = groups.filter((m) => !m.blocking && !m.hiddenByDefault && m.textModel).sort((a, b) => (b.match_score || 0) - (a.match_score || 0))[0];
    if (top && pinned && !top.isSel && (top.match_score || 0) > (pinned.match_score || 0) && !q.words.length) {
      html += `<div class="switch-prompt"><span>${esc(shortModelName(top.name))} fits better (${top.match_score}%${pinned.match_score != null ? ` vs ${pinned.match_score}%` : ""}). Switch?</span>
        <button type="button" class="btn btn-sm btn-primary" data-select-model="${esc(top.model_garden_id)}" data-needs-idle>Switch to ${esc(shortModelName(top.name))}</button></div>`;
    }
    const selBtn = (m) => {
      if (m.isSel) return "";
      if (m.blocking) return `<span class="sub">Doesn't fit</span>`;
      if (!m.textModel) return `<span class="sub">Not a text model</span>`;
      const lbl = q.prec ? `Select at ${q.prec.toUpperCase()}` : "Select";
      return `<button type="button" class="btn btn-sm" data-select-model="${esc(m.model_garden_id)}" data-needs-idle aria-label="${lbl} ${esc(shortModelName(m.name))}">${lbl}</button>`;
    };
    const tagsHTML = (m) => (m.legacy ? '<span class="mtag">Older</span>' : "") + (m.specialty ? `<span class="mtag">${esc(m.specialty)}</span>` : "") +
      (m.meets_quality_floor === false ? '<span class="mtag mtag-warn" title="Below the quality bar for this task">Below the quality bar</span>' : "");
    const subLine = (m) => `${esc(m.family)} · ${esc(m.model_garden_id || m.hf_id || "")}${m.custom ? " · not in the catalog" : ""}`;
    const variantLine = (m) => m.variants.length ? `<span class="sub">Includes ${esc(Array.from(new Set(m.variants)).join(", "))}</span>` : "";
    if (S.drawerView.model === "cards") {
      html += `<div class="cards">${visible.map((m) => {
        const warns = (m.warnings || []).filter((w) => !repeated.has(w));
        return `<article class="card${m.isSel ? " pinned" : ""}"><h3><span>${esc(prettyName(m.name))}${tagsHTML(m)}</span><span>${m.match_score != null ? m.match_score + "%" : ""}</span></h3>
          <p class="meta">${esc(m.family)} · ${esc(paramsText(m))} · ${ctxText(m)} context</p>
          <p class="meta">${precText(m).replace(/<span class="sub">.*$/, "")}</p>
          ${m.summary ? `<p>${esc(ap(m.summary))}</p>` : ""}
          ${provHTML(m) ? `<p class="meta">${provHTML(m)}</p>` : ""}
          ${warns.length ? `<ul>${warns.map((x) => `<li class="warn-text">${esc(ap(x))}</li>`).join("")}</ul>` : ""}
          <div class="card-foot"><span>Fits longest prompts: ${fitsCell(m)}</span>${m.isSel ? '<span class="sel-tag">Selected</span>' : selBtn(m)}</div></article>`;
      }).join("")}</div>`;
    } else {
      const showChips = groups.some((m) => m.est_chips != null);
      const showUsd = groups.some((m) => usdOf(m) != null);
      const head = `<tr><th>Model</th><th>Parameters</th><th class="n">Context</th><th>Fits your longest prompts</th><th>Precision <span class="th-sub">recommended in bold</span></th>${showChips ? '<th class="n">Est. chips</th>' : ""}${showUsd ? '<th class="n">Est. $/M input</th>' : ""}<th class="n">Match</th><th><span class="visually-hidden">Action</span></th></tr>`;
      html += `<div class="table-wrap"><table class="tbl cmp"><thead>${head}</thead><tbody>${visible.map((m) => {
        const warns = (m.warnings || []).filter((w) => !repeated.has(w));
        return `<tr${m.isSel ? ' class="pinned"' : ""}><td><span class="name">${esc(prettyName(m.name) || modelName(m))}</span>${m.isSel ? '<span class="sel-tag">Selected</span>' : ""}${tagsHTML(m)}<span class="sub">${subLine(m)}</span>${provHTML(m)}${variantLine(m)}${warns.map((w) => `<span class="sub warn-text">${esc(ap(w))}</span>`).join("")}</td>
          <td>${esc(paramsText(m))}</td><td class="n">${ctxText(m)}</td><td>${fitsCell(m)}</td><td class="prec-cell">${precText(m)}</td>
          ${showChips ? `<td class="n">${liveEst(m, m.isSel).chips}</td>` : ""}
          ${showUsd ? `<td class="n">${liveEst(m, m.isSel).usd}</td>` : ""}
          <td class="n">${whyHTML(m, repeated)}</td>
          <td class="n">${selBtn(m)}</td></tr>`;
      }).join("")}</tbody></table></div>`;
    }
    if (nonSel.length > f.limit) html += `<p class="mdl-more"><button type="button" class="btn btn-sm" id="mdlShowMore">Show ${Math.min(40, nonSel.length - f.limit)} more (${nonSel.length - f.limit} not shown)</button></p>`;
    box.innerHTML = html;
    box.querySelectorAll("[data-select-model]").forEach((b) => {
      b.onclick = () => {
        const id = b.dataset.selectModel;
        const m = groups.find((x) => x.model_garden_id === id);
        if (m) selectModel(Object.assign({}, m, q.prec ? { recommended_precision: q.prec } : {}));
      };
    });
    const sm = $("mdlShowMore");
    if (sm) sm.onclick = () => { S.mdl.limit += 40; renderModelResults(); const rows = box.querySelectorAll("tbody tr, .card"); const r = rows[Math.max(0, rows.length - 40)]; const b = r && r.querySelector("button"); if (b) b.focus(); };
    box.querySelectorAll("details.why").forEach((d) => d.addEventListener("toggle", () => {
      if (d.open) box.querySelectorAll("details.why[open]").forEach((o) => { if (o !== d) o.open = false; });
    }));
    document.querySelectorAll("[data-needs-idle]").forEach((b) => { if (S.busy) b.disabled = true; });
  }
  function wireModelDrawer() {
    const r = $("btnRefreshModels");
    if (r) r.onclick = async () => {
      r.disabled = true;
      try {
        const c = collectCfg();
        const res = await postJSON("/api/catalog/refresh", { project: c.project, region: c.region });
        if (S.catalog) { S.catalog.models = res.models || S.catalog.models; if (res.gemini) S.catalog.gemini = res.gemini; if (res.families) S.catalog.families = res.families; }
        S.models = null;
        await fetchModels();
      } catch (e) { toast(`Couldn't refresh the model list: ${e.message}`, true); }
      r.disabled = false;
    };
    const retry = $("btnRetryModels"); if (retry) retry.onclick = fetchModels;
    const search = $("mdlSearch");
    if (search) {
      let t = 0;
      search.oninput = () => { clearTimeout(t); t = setTimeout(() => { S.mdl.q = search.value; S.mdl.limit = 40; renderModelResults(); }, 80); };
      search.onkeydown = (e) => { if (e.key === "Escape" && search.value) { e.stopPropagation(); search.value = ""; S.mdl.q = ""; renderModelResults(); } };
      $("mdlSort").onchange = (e) => { S.mdl.sort = e.target.value; renderModelResults(); };
      $("mdlFits").onchange = (e) => { S.mdl.fits = e.target.checked; S.mdl.limit = 40; renderModelResults(); };
      $("mdlOlder").onchange = (e) => { S.mdl.older = e.target.checked; S.mdl.limit = 40; renderModelResults(); };
      document.querySelectorAll("[data-mdl-size]").forEach((b) => b.onclick = () => {
        S.mdl.size = b.dataset.mdlSize; S.mdl.limit = 40;
        document.querySelectorAll("[data-mdl-size]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        renderModelResults();
      });
    }
    const insp = $("btnInspectHf");
    if (insp) insp.onclick = async () => {
      const hf = $("cmHf").value.trim();
      const st = $("cmStatus");
      if (!hf) { st.textContent = "Enter a Hugging Face ID first."; return; }
      st.textContent = `Reading ${hf} from Hugging Face…`;
      try {
        const info = await postJSON("/api/models/inspect", { hf_id: hf });
        if (info.params_b) $("cmParams").value = info.params_b;
        $("cmActive").value = info.active_params_b || info.params_b || "";
        st.textContent = `${hf}: about ${info.params_b || "?"}B parameters, ${info.max_context ? Math.round(info.max_context / 1024) + "K" : "default"} context.`;
      } catch (e) { st.textContent = `Couldn't read ${hf}: ${e.message}`; }
    };
    const use = $("btnUseCustom");
    if (use) use.onclick = () => {
      const mg = $("cmMg").value.trim(), hf = $("cmHf").value.trim();
      if (!mg && !hf) { $("cmStatus").textContent = "Enter a Model Garden or Hugging Face ID."; return; }
      selectModel({ model_garden_id: mg || null, hf_id: hf || null, params_b: +$("cmParams").value || null, active_params_b: +$("cmActive").value || null, recommended_precision: $("cmPrec").value, name: null, custom: true });
    };
    document.querySelectorAll("[data-needs-idle]").forEach((b) => { if (S.busy) b.disabled = true; });
  }
  const hasRealRun = () => { const m = S.session && S.session.measured; return !!(m && !m.estimated && (m.levels || []).length); };
  // The selected row shows the live numbers behind the verdict (Agent Platform + reservation, 3-yr) so the drawer agrees with the hero.
  function liveEst(m, isSelected) {
    if (isSelected && S.data && S.p && S.data.prices.vertex_cud_3y) {
      const r = evaluate(Object.assign({}, S.p, { pricing: "vertex_cud_3y" }), null);
      if (r && r.total != null) return { chips: String(r.gpus), usd: perM(r.perM) };
    }
    const usd = m.est_usd_per_m_input != null ? m.est_usd_per_m_input : m.usd_per_m_input;
    return { chips: m.est_chips != null ? String(m.est_chips) : "—", usd: perM(usd) };
  }
  async function confirmLoseMeasurement() {
    if (!hasRealRun()) return true;
    return !!(await dialog({
      title: "Replace the measured throughput?",
      body: "<p>Your benchmark measured the current model and hardware. Switching replaces that number with an estimate for the new choice. The run's report stays downloadable until you re-run.</p>",
      okLabel: "Switch anyway",
    }));
  }
  async function selectModel(m) {
    if (!(await confirmLoseMeasurement())) return;
    const c = S.cfg;
    c.model = Object.assign({}, c.model, {
      // a custom model keeps exactly the IDs entered (inheriting the previous model's hf_id would mix two models)
      model_garden_id: m.custom ? (m.model_garden_id || null) : (m.model_garden_id || c.model.model_garden_id),
      hf_id: m.custom ? (m.hf_id || null) : (m.hf_id || c.model.hf_id),
      params_b: m.params_b || null,
      active_params_b: m.active_params_b || m.params_b || null,
      precision: String(m.recommended_precision || c.model.precision || "bf16").toLowerCase(),
    });
    c.use_case.selection_mode = "manual";
    c.deployment.mode = "auto";
    S.ui.fleetOverride = null;
    S.tpsEdit = undefined;
    S.hw = null;
    S.models = null;   // its rows carry the old selection's economics and is_selected: refetch on the next open
    closeDrawer();
    renderChips();     // the chip shows the new model at once, before the estimate returns
    if (await reestimate({ clearOverride: true, force: hasRealRun() })) {
      if (S.session && S.session.no_fit) toast(`${modelName(selected().model)} doesn't fit a single machine at ${normPrec(c.model.precision).toUpperCase()}: see the options above.`, true);
      else toast(`Now planning ${modelName(selected().model)}.`);
    }
  }

  async function fetchHardware() {
    if (!S.cfg) return;
    const req = ++S.hwReq;
    S.hwLoading = true;
    if (S.drawer === "hw") renderHwDrawer();
    try {
      const c = collectCfg();
      const res = await postJSON("/api/recommend/hardware", { session: S.name, config: c, accelerator_filter: S.hwFilter });
      if (req !== S.hwReq) return;
      S.hw = res;
    } catch (e) {
      if (req !== S.hwReq) return;
      S.hw = { ok: false, error: e.message };
    }
    S.hwLoading = false;
    if (S.drawer === "hw") renderHwDrawer();
  }
  function hwSelected(c) {
    if (c.is_selected !== undefined) return !!c.is_selected;
    const h = selected().hardware || {};
    return c.accelerator_type === h.accelerator_type && +c.accelerator_count === +h.accelerator_count;
  }
  // Long-prompt rates belong to the selected hardware; another candidate gets the same slowdown, not the same tok/s.
  function candClassParams(c, sel) {
    if (sel) return {};
    const b = S.r && S.r.buckets, ps = (b && b.penaltySource) || {};
    const short = +c.est_input_tps_per_gpu || 0, out = { mediumTps: null, largeTps: null };
    if (short > 0) ["medium", "large"].forEach((k) => { if (ps[k] === "measured") out[k + "Tps"] = short / b[k].penalty; });
    return out;
  }
  // The row's fleet, $/M and monthly cost, all at one pricing option: the rail's "Fleet pricing" when it's fixed, else
  // the cheapest for this candidate (as in the verdict). byTier keeps every option for the details list.
  function hwDerived(c) {
    const sel = hwSelected(c);
    const byTier = candMonthlyByTier(c, sel);
    if (byTier.length) {
      const fixed = S.ui.pricing && S.ui.pricing !== "auto" ? S.ui.pricing : null;
      const pick = fixed ? byTier.find((m) => m.tier === fixed) : byTier.reduce((a, b) => (b.total < a.total ? b : a));
      if (pick) return { sel, byTier, fleet: pick.gpus, usd: pick.perM, monthly: pick.total, monthlyTier: pick.tier };
      return { sel, byTier, fleet: null, usd: null, monthly: null, monthlyTier: fixed };   // no price for that option
    }
    // No live cost model yet (still loading): the server's figures (Agent Platform + reservation, 3-yr).
    const r = hwDerived0(c);
    const tok = S.r && S.r.totalTok;
    return Object.assign(r, { byTier: [], monthly: r.usd != null && tok ? r.usd * tok / 1e6 : null,
                              monthlyTier: "vertex_cud_3y" });
  }
  // Full monthly bill (fleet + overflow, cheapest policy-OK fleet) for this candidate at each pricing option, computed
  // exactly like the main Monthly column (hwDerived) but with the tier pinned. [{tier, total, gpus}]
  function candMonthlyByTier(c, sel) {
    const out = [];
    if (!S.data || !S.p) return out;
    const prices = pickTiers(c.prices);
    (c.pricing_options || []).forEach((o) => { if (o.usd_per_chip_hour > 0 && TIER_ORDER.indexOf(o.tier) >= 0) prices[o.tier] = o.usd_per_chip_hour; });
    const data2 = Object.assign({}, S.data, { perGpu: sel ? S.data.perGpu : c.est_input_tps_per_gpu,
      mixFactorRef: sel ? S.data.mixFactorRef : 1, classTps: sel ? S.data.classTps : null, replica: c.accelerator_count, prices });
    const saved = S.data;
    try {
      S.data = data2; S.native = undefined;
      const cp = candClassParams(c, sel);
      for (const t of TIER_ORDER) {
        if (!(prices[t] > 0)) continue;
        try {
          const r = evaluate(Object.assign({}, S.p, cp, { pricing: t, fleetOverride: null }), null);
          if (r && isFinite(r.total)) out.push({ tier: t, total: r.total, gpus: r.gpus, perM: r.perM });
        } catch (_) { /* skip this tier */ }
      }
    } finally { S.data = saved; S.native = undefined; }
    return out;
  }
  function hwDerived0(c) {
    const sel = hwSelected(c);
    let fleet = c.fleet_gpus, usd = c.usd_per_m_input;
    if ((fleet == null || usd == null) && S.data && S.p) {
      const prices = pickTiers(c.prices);
      if (Object.keys(prices).length) {
        const data2 = Object.assign({}, S.data, { perGpu: sel ? S.data.perGpu : c.est_input_tps_per_gpu,
          mixFactorRef: sel ? S.data.mixFactorRef : 1, classTps: sel ? S.data.classTps : null, replica: c.accelerator_count, prices });
        const tier = prices.vertex_cud_3y ? "vertex_cud_3y" : Object.keys(prices)[0];
        const saved = S.data;
        try {
          S.data = data2; S.native = undefined;
          const r = evaluate(Object.assign({}, S.p, candClassParams(c, sel), { pricing: tier }), null);
          if (fleet == null) fleet = r.gpus;
          if (usd == null) usd = r.perM;
        } catch (_) { /* leave blank */ } finally { S.data = saved; S.native = undefined; }
      }
    }
    return { fleet, usd, sel };
  }
  // ---- Hardware drawer: compact rows to choose (what, fleet, cost, can I deploy it, throughput); prices per tier,
  // quota numbers, shape and throughput basis on demand (row expands / card disclosure).
  const TIER_SHORT = { vertex: "On demand", vertex_cud_1y: "Reserved 1y", vertex_cud_3y: "Reserved 3y", cud_1y: "GKE 1y", cud_3y: "GKE 3y" };
  const hwKey = (c) => `${c.accelerator_type}|${c.accelerator_count}`;
  const DEPLOY_RANK = { ready: 0, bench: 1, unknown: 2, noquota: 3, offregion: 4 };
  // Can this option be deployed in the region with the project's Agent Platform serving quota (S.hw.quota, Cloud
  // Quotas API)? The benchmark needs one replica's chips, the production fleet all of them. TPU quotas count cores
  // (one per v5e / v6e chip).
  function deployStatus(c, fleet) {
    const region = collectCfg().region;
    const q = S.hw && S.hw.quota;
    const need = +c.accelerator_count || 1;
    if (c.region_note) return { state: "offregion", label: "Not offered", detail: `not in ${region}`, title: `${accName(c.accelerator_type)} isn't offered in ${region}. Choose another region to deploy it.` };
    if (!q || !q.ok) return { state: "unknown", label: "Quota unknown", detail: "", title: (q && q.error) || "Quota not read" };
    const e = (q.by_accelerator || {})[c.accelerator_type];
    if (!e || e.limit == null) return { state: "unknown", label: "Quota unknown", detail: "no serving quota listed", title: "The Cloud Quotas API lists no Agent Platform serving quota for this accelerator." };
    const n = e.limit, u = e.unit === "cores" ? " cores" : "";
    const base = { limit: n, unit: u, need, fleet, entry: e };
    if (n < need) return Object.assign(base, { state: "noquota", label: n ? "Quota too low" : "No quota", detail: `${n}${u} of ${need} for one replica`, title: `Quota ${n}${u}; one replica needs ${need}. Request ${need - n} more to benchmark.` });
    if (fleet != null && n < fleet) return Object.assign(base, { state: "bench", label: "Benchmark only", detail: `fleet of ${fleet} needs +${fleet - n}`, title: `Quota ${n}${u}: enough for a benchmark (${need}); the fleet of ${fleet} needs ${fleet - n} more.` });
    return Object.assign(base, { state: "ready", label: "Ready", detail: `quota ${n}${u}`, title: `Quota ${n}${u} covers ${fleet != null ? `the fleet of ${fleet}` : "a benchmark"}.` });
  }
  function deployPill(st, withDetail) {
    const ico = st.state === "noquota" || st.state === "offregion" ? icon("i-alert", "ico-sm") : `<span class="dot" aria-hidden="true"></span>`;
    return `<span class="dstat dstat-${st.state}" title="${esc(st.title)}">${ico}${esc(st.label)}</span>${withDetail && st.detail ? `<span class="sub">${esc(st.detail)}</span>` : ""}`;
  }
  function candPrices(c) {
    const pr = Object.assign({}, c.prices || {});
    (c.pricing_options || []).forEach((o) => { pr[o.tier] = o.usd_per_chip_hour; });
    return pr;
  }
  function candTput(c, sel) {
    const tp = throughput();
    return {
      src: sel && tp && tp.pending ? tp.source : (c.throughput_source || (sel && tp ? tp.source : "estimated")),
      tps: sel && tp && (tp.pending || c.est_input_tps_per_gpu == null) ? tp.per_chip : c.est_input_tps_per_gpu,
      basis: sel && tp && tp.pending ? tp.basis : c.est_basis,
      note: sel && tp && tp.pending ? tp.note : c.est_basis_note,
    };
  }
  // The details behind one option: prices per chip-hour (every tier), quota in the region, shape and throughput basis.
  function hwDetailsHTML(row, st) {
    const { c, sel } = row;
    const pr = candPrices(c);
    const monthlyView = S.hwView === "monthly";
    const tiers = TIER_ORDER.filter((t) => pr[t] > 0);
    // Every pricing option for this hardware, with the cheapest fleet at that price (it can differ by option: an
    // expensive GPU-hour favours fewer chips and more overflow). The row's option is highlighted.
    const costHead = monthlyView ? "Monthly cost" : "Per chip-hour";
    const byTier = row.byTier && row.byTier.length ? row.byTier
      : tiers.map((t) => ({ tier: t, total: null, gpus: null }));
    const chipsTxt = (n) => (n == null ? "" : ` <span class="sub-inline">· ${n} chip${n === 1 ? "" : "s"}</span>`);
    const costRows = byTier.filter((m) => pr[m.tier] > 0).map((m) => {
      const cur = m.tier === row.monthlyTier ? ' class="cur"' : "";
      const v = monthlyView ? fmtUSD(m.total) : `${fmtUSDc(pr[m.tier])}/h`;
      return `<dt${cur}>${esc(TIER_SHORT[m.tier] || m.tier)}</dt><dd${cur}>${v}${chipsTxt(m.gpus)}</dd>`;
    }).join("");
    const q = S.hw && S.hw.quota;
    let quota;
    if (st.state === "offregion") quota = `<p>${esc(st.title)}</p>`;
    else if (st.limit == null) quota = `<p>${esc(st.title)}</p>`;
    else quota = `<dl class="kv"><dt>Limit</dt><dd>${st.limit}${esc(st.unit)}</dd><dt>One replica (benchmark)</dt><dd>${st.need} ${st.limit >= st.need ? "✓" : `<span class="warn-text">+${st.need - st.limit}</span>`}</dd>${st.fleet != null ? `<dt>Fleet</dt><dd>${st.fleet} ${st.limit >= st.fleet ? "✓" : `<span class="warn-text">+${st.fleet - st.limit}</span>`}</dd>` : ""}</dl>`
      + (st.limit < (st.fleet || st.need) && q && q.console_url ? `<p><a href="${esc(q.console_url)}" target="_blank" rel="noopener">Request an increase</a> <span class="sub">(filter for "custom_model_serving")</span></p>` : "");
    const t = candTput(c, sel);
    const shape = [`${c.accelerator_count} × ${esc((c.gpu || accName(c.accelerator_type)).replace(/^Cloud\s+/, ""))} on ${esc(c.machine_type || "—")}`];
    if (c.vram_gb) shape.push(`${Math.round(c.vram_gb)} GB memory${c.weights_gb ? `, weights ${Math.round(c.weights_gb)} GB` : ""}`);
    if (c.verified) shape.push("Verified on Agent Platform (Model Garden recipe)");
    if (c.shape_note) shape.push(esc(ap(c.shape_note)));
    return `<div class="hw-details">
      <section><h4>${costHead}</h4><dl class="kv">${costRows}</dl></section>
      <section><h4>Quota in ${esc(collectCfg().region)}</h4>${quota}</section>
      <section><h4>Shape and throughput</h4><ul class="plain">${shape.map((x) => `<li>${x}</li>`).join("")}</ul>
        ${precSupportHTML(c, "p")}
        <p>${fmtInt(t.tps)} input tok/s per chip ${tagHTML(t.src, t.basis, t.note)}</p>${t.note ? `<p class="sub">${esc(ap(t.note))}</p>` : ""}
        ${sel ? "" : `<button type="button" class="btn btn-sm btn-primary" data-select-hw="${esc(hwKey(c))}" data-needs-idle>Select this hardware</button>`}</section>
    </div>`;
  }
  function hwAboutHTML(plan) {
    const q = S.hw && S.hw.quota;
    const tot = fmtTok(S.r && S.r.totalTok);
    let quota;
    if (!q) quota = "";
    else if (!q.ok) quota = `<p>Quota: ${esc(q.error || "not available")}</p>`;
    else {
      const day = q.fetched_at ? new Date(q.fetched_at * 1000).toLocaleString() : "";
      quota = `<p><strong>Can I deploy it?</strong> Your project's Agent Platform serving quota ("Custom model serving … per region") for ${esc(q.project)} in ${esc(q.region)}, from the Cloud Quotas API${day ? `, read ${esc(day)}` : ""}. Ready: the quota covers the fleet. Benchmark only: it covers one replica. The quota is a limit, not what's free (running endpoints count against it); a fleet on GKE uses Compute Engine GPU quota instead.</p>`;
    }
    return `<details class="hw-about"><summary>About these numbers</summary>
      <p><strong>Fleet and cost</strong> assume your traffic, routing and redundancy settings. Fleet, $/M and monthly cost (the full bill for ${tot} input tokens, overflow included) are at the pricing named under each figure: the rail's Fleet pricing, or each option's cheapest when it's on auto. Expand a row for every pricing option; the cheapest fleet can differ between them. Prices are per chip-hour from the Cloud Billing Catalog.</p>
      ${quota}${availabilityHTML(plan && plan.availability)}</details>`;
  }
  function renderHwDrawer() {
    const body = $("drawerBody");
    const monthlyView = S.hwView === "monthly";
    const filt = ["all", "gpu", "tpu"].map((f) => `<button type="button" class="seg-btn" data-hw-filter="${f}" aria-pressed="${S.hwFilter === f}">${{ all: "All", gpu: "GPUs", tpu: "TPUs" }[f]}</button>`).join("");
    const views = [["usd", "$/M input"], ["monthly", "Monthly"]].map(([k, l]) => `<button type="button" class="seg-btn" data-hw-view="${k}" aria-pressed="${S.hwView === k}">${l}</button>`).join("");
    const quotaOk = !!(S.hw && S.hw.quota && S.hw.quota.ok);
    const depBtn = `<button type="button" class="seg-btn" data-hw-deployable aria-pressed="${!!S.hwDeployable && quotaOk}"${quotaOk ? "" : ` disabled title="Quotas couldn't be read"`}>Deployable now</button>`;
    let html = `<div class="drawer-tools"><div class="hw-segs"><div class="seg" role="group" aria-label="Accelerator type">${filt}</div><div class="seg" role="group" aria-label="Filter">${depBtn}</div><div class="seg" role="group" aria-label="Cost as">${views}</div></div>
      ${S.cfg.deployment.mode === "manual" ? `<button type="button" class="btn btn-sm" id="btnHwAuto" data-needs-idle>Let the planner choose</button>` : ""}</div>`;
    const costCol = monthlyView ? ["monthly", "Monthly cost"] : ["usd", "$/M input"];
    const head = (sortable) => {
      const th = (k, l) => (sortable ? sortBtn(k, l) : l);
      return `<tr><th>Accelerator</th><th class="n">${th("fleet", "Fleet")}</th><th class="n">${th(costCol[0], costCol[1])}</th><th>${th("deploy", "Can I deploy it?")}<span class="th-sub">in ${esc(collectCfg().region)}</span></th><th class="n">${th("tps", "Throughput per chip")}${!S.hw || (S.hw.plan && (S.hw.plan.candidates || []).some((c) => needsCaution(c.throughput_source || "estimated"))) ? `<span class="th-sub">${cautionHTML("Estimates")}</span>` : ""}</th><th><span class="visually-hidden">Action</span></th></tr>`;
    };
    const costVal = (x) => (monthlyView ? fmtUSD(x.monthly) : perM(x.usd));
    const costSub = (x) => (x.monthlyTier ? TIER_SHORT[x.monthlyTier] || x.monthlyTier : "");
    if (S.hwLoading || !S.hw) {
      html += `<p class="drawer-note" role="status">Sizing the model weights and ranking accelerators. This takes a few seconds…</p><div class="table-wrap"><table class="tbl cmp hw-tbl"><thead>${head(false)}</thead><tbody>${skeletonRows(6, 6)}</tbody></table></div>`;
      body.innerHTML = html;
      wireHwDrawer();
      return;
    }
    if (!S.hw.ok || !S.hw.plan) {
      if (S.hw.no_fit) html += `<div class="nofit-box">${noFitHTML(S.hw.no_fit, "hnf")}</div>`;
      else html += `<p class="warn-line">${icon("i-alert")}<span>${esc(ap(S.hw.error || "No accelerator fits this model."))} <button type="button" class="link-btn" id="btnRetryHw">Try again</button></span></p>`;
      html += availabilityHTML(S.hw.availability);
      body.innerHTML = html;
      wireHwDrawer();
      wireNoFit(body);
      return;
    }
    const plan = S.hw.plan;
    const b = S.r && S.r.buckets;
    const tips = [];
    if (plan.weights_gb) tips.push(`${modelName(selected().model)} needs about ${Math.round(plan.weights_gb)} GB for weights at ${String(selected().precision || "").toUpperCase()}; chips per replica follow from that and the machine shapes.`);
    if (b && b.longGpuShare > 0.3 && !S.cfg.hybrid.route_long) tips.push(`Prompts over 32K take ${pct(b.longGpuShare)} of fleet time. Sending them to Gemini under Routing shrinks every fleet below.`);
    if (Array.isArray(S.hw.tips)) S.hw.tips.forEach((t) => tips.push(ap(t)));
    if (tips.length) html += `<ul class="tips">${tips.map((t) => `<li>${icon("i-info", "ico-sm")}<span>${esc(t)}</span></li>`).join("")}</ul>`;
    // The selected hardware stays visible even when the region doesn't offer it (then it isn't among the candidates).
    const cands = (plan.candidates || []).slice();
    if (plan.chosen && plan.chosen.accelerator_type && !cands.some((c) => hwKey(c) === hwKey(plan.chosen)) && (S.hwFilter === "all" || (plan.chosen.kind || "gpu") === S.hwFilter)) cands.unshift(plan.chosen);
    let rows = cands.map((c) => {
      const d = Object.assign({ c }, hwDerived(c));
      d.st = deployStatus(c, d.fleet);
      d.deploy = DEPLOY_RANK[d.st.state];
      d.tps = candTput(c, d.sel).tps;
      return d;
    });
    const total = rows.length;
    if (S.hwDeployable && quotaOk) rows = rows.filter((r) => r.sel || r.st.state === "ready" || r.st.state === "bench");
    const k = S.hwSort.key, dir = S.hwSort.dir;
    const val = (r) => (r[k] == null ? Infinity : r[k]);
    rows.sort((a, b2) => (a.sel ? -1 : b2.sel ? 1 : (val(a) - val(b2)) * dir || (a.usd || 0) - (b2.usd || 0)));
    const sortName = { usd: "cost per million input tokens", monthly: "monthly cost", fleet: "fleet size", deploy: "deployability, then cost", tps: "throughput per chip" }[k] || "cost";
    html += `<p class="drawer-note hw-count">${rows.length < total ? `${rows.length} of ${total} options you can deploy now` : `${total} options`} · sorted by ${sortName}${k === "tps" && dir < 0 ? " (highest first)" : ""}.</p>`;
    const open = S.hwOpen || (S.hwOpen = new Set());
    if (S.drawerView.hw === "cards") {
      html += `<div class="cards hw-cards">${rows.map((row) => {
        const { c, fleet, sel, st } = row;
        const t = candTput(c, sel);
        const name = esc((c.gpu || accName(c.accelerator_type)).replace(/^Cloud\s+/, ""));
        const kind = (c.kind || accKind(c.accelerator_type)) === "tpu" ? "TPU" : "GPU";
        return `<article class="card hw-card${sel ? " pinned" : ""}">
          <header><h3>${name}</h3>${sel ? '<span class="sel-tag">Selected</span>' : ""}</header>
          <p class="meta">${c.accelerator_count} × ${esc(c.machine_type || "")} · ${kind}${c.shape_note ? ` <span class="info-i" title="${esc(ap(c.shape_note))}">${icon("i-info", "ico-sm")}</span>` : ""}</p>
          <div class="hero"><span class="hero-v">${costVal(row)}</span><span class="hero-u">${monthlyView ? "a month" : "per M input"}</span><span class="hero-fleet">${fleet != null ? fleet : "—"} chips</span></div>
          <p class="hero-sub">${monthlyView ? `${perM(row.usd)} per M input` : `${fmtUSD(row.monthly)} a month`}${costSub(row) ? ` · ${esc(costSub(row))}` : ""}</p>
          <div class="hw-lines"><div>${deployPill(st, true)}</div><div>${tagHTML(t.src, t.basis, t.note)} <span class="sub-inline">${fmtInt(t.tps)} tok/s per chip</span></div>${precSupportHTML(c, "div")}</div>
          <details class="hw-more"><summary>${monthlyView ? "Monthly cost" : "Prices"}, quota and shape</summary>${hwDetailsHTML(row, st)}</details>
          <div class="card-foot">${sel ? '<span class="sel-tag">Selected</span>' : `<button type="button" class="btn btn-sm" data-select-hw="${esc(hwKey(c))}" data-needs-idle>Select</button>`}</div></article>`;
      }).join("")}</div>` + hwAboutHTML(plan);
      body.innerHTML = html;
      wireHwDrawer();
      return;
    }
    html += `<div class="table-wrap"><table class="tbl cmp hw-tbl"><thead>${head(true)}</thead><tbody>${rows.map((row, i) => {
      const { c, fleet, sel, st } = row;
      const t = candTput(c, sel);
      const key = hwKey(c), isOpen = open.has(key), did = `hwd-${i}`;
      const kind = (c.kind || accKind(c.accelerator_type)) === "tpu" ? "TPU" : "GPU";
      return `<tr class="hw-row${sel ? " pinned" : ""}${isOpen ? " open" : ""}" data-hw-row="${esc(key)}">
        <td><button type="button" class="row-toggle" data-hw-toggle="${esc(key)}" aria-expanded="${isOpen}" aria-controls="${did}" aria-label="Details for ${esc(accName(c.accelerator_type))}">${icon("i-chevron", "ico-sm")}</button><span class="name">${esc((c.gpu || accName(c.accelerator_type)).replace(/^Cloud\s+/, ""))}</span>${sel ? '<span class="sel-tag">Selected</span>' : ""}
          <span class="sub">${c.accelerator_count} × ${esc(c.machine_type || "")} · ${kind}${c.verified ? " · verified" : ""}${c.shape_note ? ` <span class="info-i" title="${esc(ap(c.shape_note))}">${icon("i-info", "ico-sm")}</span>` : ""}</span>${precSupportHTML(c, "span")}</td>
        <td class="n">${fleet != null ? fleet : "—"}</td>
        <td class="n"><strong>${costVal(row)}</strong>${costSub(row) ? `<span class="sub">${esc(costSub(row))}</span>` : ""}</td>
        <td>${deployPill(st, true)}</td>
        <td class="n">${fmtInt(t.tps)} ${tagHTML(t.src, t.basis, t.note)}</td>
        <td class="n">${sel ? "" : `<button type="button" class="btn btn-sm" data-select-hw="${esc(key)}" data-needs-idle>Select</button>`}</td></tr>
        <tr class="hw-detail" id="${did}"${isOpen ? "" : " hidden"}><td colspan="6">${hwDetailsHTML(row, st)}</td></tr>`;
    }).join("")}</tbody></table></div>` + hwAboutHTML(plan);
    body.innerHTML = html;
    wireHwDrawer();
  }
  // Region availability (backend hardware.region_shapes via plan.availability): where the accelerator list came from,
  // accelerators the region doesn't offer, and ones it offers that this tool has no performance data for.
  function availabilityHTML(av) {
    if (!av) return "";
    const region = av.region || collectCfg().region;
    const kindOk = (x) => S.hwFilter === "all" || (x.kind || "gpu") === S.hwFilter;
    const day = av.fetched_at ? new Date(av.fetched_at * 1000).toISOString().slice(0, 10) : null;
    const src = av.source === "compute_api"
      ? `Accelerators and machine shapes offered in ${esc(region)}: Compute Engine API${day ? `, updated ${day}` : ""}. Performance figures per chip come from the built-in chip table.`
      : `Accelerators and machine shapes: built-in catalog (the Compute Engine API wasn't reached, so availability in ${esc(region)} isn't checked).`;
    const gone = (av.not_offered || []).filter(kindOk);
    const unrated = (av.unrated || []).filter(kindOk);
    let html = `<div class="hw-avail"><p class="hint">${icon("i-info", "ico-sm")} ${src}</p>`;
    if (gone.length) html += `<p class="hint"><strong>Not offered in ${esc(region)}:</strong> ${gone.map((x) => esc(String(x.name).replace(/^Cloud\s+/, ""))).join(", ")}. Choose another region to plan them.</p>`;
    if (unrated.length) html += `<details class="hw-unrated"><summary>${unrated.length} more offered in ${esc(region)} without performance data (not estimated)</summary><ul>${unrated.map((x) => `<li><strong>${esc(x.name)}</strong> <span class="sub">${esc(x.machines || "")}</span> · no performance data, rough: not estimated</li>`).join("")}</ul></details>`;
    return html + "</div>";
  }
  function sortBtn(key, label) {
    const on = S.hwSort.key === key;
    return `<button type="button" class="th-sort" data-hw-sort="${key}"${on ? ` aria-sort="${S.hwSort.dir > 0 ? "ascending" : "descending"}"` : ""}>${label}${on ? icon("i-chevron", "ico-sm") : ""}</button>`;
  }
  function wireHwDrawer() {
    const body = $("drawerBody");
    body.querySelectorAll("[data-hw-filter]").forEach((b) => b.onclick = () => { S.hwFilter = b.dataset.hwFilter; S.hw = null; fetchHardware(); });
    body.querySelectorAll("[data-hw-view]").forEach((b) => b.onclick = () => {
      S.hwView = b.dataset.hwView;
      if (S.hwSort.key === "usd" || S.hwSort.key === "monthly") S.hwSort = { key: S.hwView, dir: S.hwSort.dir };
      renderHwDrawer();
      const nb = $("drawerBody").querySelector(`[data-hw-view="${S.hwView}"]`); if (nb) nb.focus();
    });
    body.querySelectorAll("[data-hw-deployable]").forEach((b) => b.onclick = () => { S.hwDeployable = !S.hwDeployable; renderHwDrawer(); const nb = $("drawerBody").querySelector("[data-hw-deployable]"); if (nb) nb.focus(); });
    const toggleRow = (key) => {
      const open = S.hwOpen || (S.hwOpen = new Set());
      if (open.has(key)) open.delete(key); else open.add(key);
      const btn = body.querySelector(`[data-hw-toggle="${CSS.escape(key)}"]`);
      if (!btn) return;
      const on = open.has(key);
      btn.setAttribute("aria-expanded", on);
      btn.closest("tr").classList.toggle("open", on);
      const det = $(btn.getAttribute("aria-controls")); if (det) det.hidden = !on;
    };
    body.querySelectorAll("[data-hw-toggle]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); toggleRow(b.dataset.hwToggle); });
    body.querySelectorAll("[data-hw-row]").forEach((tr) => tr.onclick = (e) => {
      if (e.target.closest("button, a, input, select, .tag, .info-i")) return;
      toggleRow(tr.dataset.hwRow);
    });
    body.querySelectorAll("[data-hw-sort]").forEach((b) => b.onclick = () => {
      const k = b.dataset.hwSort;
      // Throughput sorts highest first; everything else lowest first.
      S.hwSort = { key: k, dir: S.hwSort.key === k ? -S.hwSort.dir : (k === "tps" ? -1 : 1) };
      renderHwDrawer();
    });
    body.querySelectorAll("[data-select-hw]").forEach((b) => b.onclick = () => {
      const [type, count] = b.dataset.selectHw.split("|");
      const c = ((S.hw && S.hw.plan && S.hw.plan.candidates) || []).find((x) => x.accelerator_type === type && String(x.accelerator_count) === count);
      if (c) selectHardware(c);
    });
    const auto = $("btnHwAuto");
    if (auto) auto.onclick = async () => { S.cfg.deployment.mode = "auto"; S.ui.fleetOverride = null; closeDrawer(); S.hw = null; await reestimate(); };
    const retry = $("btnRetryHw"); if (retry) retry.onclick = () => { S.hw = null; fetchHardware(); };
    document.querySelectorAll("[data-needs-idle]").forEach((b2) => { if (S.busy) b2.disabled = true; });
  }
  async function selectHardware(c) {
    if (!(await confirmLoseMeasurement())) return;
    Object.assign(S.cfg.deployment, { mode: "manual", accelerator_type: c.accelerator_type, accelerator_count: c.accelerator_count, machine_type: c.machine_type });
    S.ui.fleetOverride = null;
    S.tpsEdit = undefined;
    closeDrawer();
    S.hw = null;
    if (await reestimate({ clearOverride: true, force: hasRealRun() })) toast(`Now planning ${c.accelerator_count} × ${accName(c.accelerator_type)} per replica.`);
  }

  // ---------------------------------------------------------------- catalog, auth, theme
  function updateProjectLabel() {
    const proj = (S.cfg && S.cfg.project) || (S.catalog && S.catalog.auth && S.catalog.auth.project) || "Project";
    $("projectBtnLabel").textContent = proj;
    $("projectBtn").setAttribute("aria-label", `Google Cloud project: ${proj}`);
  }
  function authHeadline(st) {
    const who = st.adc_account || null;
    const t = st.credential_type;
    if (st.adc_ok && st.service_identity && st.sign_in_available === false) {
      const where = st.runtime === "cloud_run" ? "This app's Cloud Run runtime service account" : t === "service_account" ? "Application Default Credentials from the key file in GOOGLE_APPLICATION_CREDENTIALS" : "Application Default Credentials from the metadata server";
      return { head: `Running as ${who || "the service account"}`, detail: `${where}; benchmark endpoints deploy as this identity` };
    }
    if (st.adc_ok) {
      if (t === "service_account") return { head: `Using service account ${who || ""}`.trim(), detail: st.google_application_credentials ? "Application Default Credentials from the key file in GOOGLE_APPLICATION_CREDENTIALS" : "Application Default Credentials (service account)" };
      if (t === "compute_metadata") return { head: `Using this machine's service account${who ? " " + who : ""}`, detail: "Application Default Credentials from the metadata server (Compute Engine or Cloud Workstations). Sign in to use your own account instead." };
      if (t === "external_account") return { head: `Using workload identity federation${who ? " as " + who : ""}`, detail: "Application Default Credentials (external account)" };
      if (t === "impersonated") return { head: `Impersonating ${who || "a service account"}`, detail: "Application Default Credentials (service account impersonation)" };
      return { head: `Signed in as ${who || st.account || "your Google account"}`, detail: "Application Default Credentials" + (st.google_application_credentials ? " from GOOGLE_APPLICATION_CREDENTIALS" : "") };
    }
    if (st.authenticated) return { head: `The gcloud CLI is signed in${st.account ? " as " + st.account : ""}, but Application Default Credentials aren't set up.`, detail: "Sign in so this app can read live prices and list your projects." };
    return { head: "Not signed in. Prices use the bundled reference catalog; benchmarks need sign-in.", detail: null };
  }
  function renderAuth(st) {
    if (!st) return;
    S.auth = st;
    $("authDot").className = "dot" + (st.adc_ok ? " ok" : " warn");
    const { head, detail } = authHeadline(st);
    $("authStatusText").textContent = head;
    const extra = [];
    if (st.adc_ok) extra.push("prices from the Cloud Billing Catalog");
    const d = $("authDetail");
    d.textContent = detail ? [detail].concat(extra).join(" · ") : "";
    d.hidden = !detail;
    // The app's own project (chosen here, else the machine default); it never changes gcloud's configuration.
    const appProj = st.app_project || null;
    const ap_ = $("authProject");
    ap_.innerHTML = appProj ? `Project for this app: <strong>${esc(appProj)}</strong>${st.app_project_source === "default" ? ' <span class="muted">(the gcloud default; pick another below)</span>' : ""}` : "No project chosen for this app yet: pick one below.";
    ap_.hidden = false;
    const q = $("authQuota");
    q.textContent = st.quota_project_differs ? `Your credentials bill API calls to ${st.quota_project} by default; the app uses ${appProj} for its calls.` : "";
    q.hidden = !st.quota_project_differs;
    const cl = st.cloud || null;   // shared Cloud Run deployment: fixed project, no sign-in, IAP user shown
    const canSignIn = st.sign_in_available !== false && !cl;
    $("btnSignIn").hidden = !!st.adc_ok || !canSignIn;
    $("btnSwitchAccount").hidden = !st.adc_ok || !canSignIn;
    if (cl) {
      if (cl.user) { d.textContent = [`You: ${cl.user} (via Identity-Aware Proxy)`, detail].filter(Boolean).concat(extra).join(" · "); d.hidden = false; }
      if (cl.project_locked) ap_.innerHTML = `Project for this app: <strong>${esc(cl.project_locked)}</strong> <span class="muted">(fixed for this deployment)</span>`;
      $("projectInput").disabled = true;
      $("projectToggle").disabled = true;
      $("btnApplyProject").hidden = true;
    }
    $("btnSignOut").hidden = !cl;   // IAP session: only on the cloud deployment
    $("authDot").title = head;
    if (S.catalog) S.catalog.auth = st;
    if (!$("projectInput").value && st.project) $("projectInput").value = currentProject() || st.project;
    updateProjectLabel();
  }
  function currentProject() {
    return (S.cfg && S.cfg.project) || (S.auth && S.auth.project_configured && S.auth.project) || (S.catalog && S.catalog.auth && S.catalog.auth.project) || "";
  }

  // ---- project combobox (type-ahead over id + display name; "Use '<typed>'" for projects not listed)
  const PROJECT_ID_RE = /^([a-z0-9][a-z0-9.-]*[a-z0-9]:)?[a-z][a-z0-9-]{4,28}[a-z0-9]$/;
  const PJ_MAX = 80;
  S.pj = { list: null, loading: null, msg: "", needSignIn: false, open: false, idx: -1, items: [], checks: {} };
  function loadProjects(force) {
    if (S.pj.loading) return S.pj.loading;
    if (S.pj.list && !force) return Promise.resolve(S.pj.list);
    if (S.auth && !S.auth.authenticated) {   // known signed out: skip the 401 round trip
      S.pj.list = []; S.pj.needSignIn = true;
      S.pj.msg = "Sign in to Google Cloud to pick from your projects, or type a project ID.";
      renderProjectList();
      return Promise.resolve(S.pj.list);
    }
    S.pj.msg = "Loading your projects…";
    renderProjectList();
    const cur = currentProject();
    S.pj.loading = api(`/api/projects${cur ? "?current=" + enc(cur) : ""}`).then((r) => {
      S.pj.list = r.projects || [];
      S.pj.needSignIn = false;
      S.pj.msg = r.message || (S.pj.list.length ? "" : "No projects found for this account. Type a project ID.");
      return S.pj.list;
    }).catch((e) => {
      S.pj.list = [];
      S.pj.needSignIn = e.status === 401;
      S.pj.msg = e.status === 401 ? "Sign in to Google Cloud to pick from your projects, or type a project ID." : `Couldn't list your projects (${e.message}). You can still type a project ID.`;
      return S.pj.list;
    }).finally(() => { S.pj.loading = null; renderProjectList(); });
    return S.pj.loading;
  }
  function filteredProjects(q) {
    const all = S.pj.list || [];
    q = (q || "").trim().toLowerCase();
    if (!q) return all;
    const words = q.split(/\s+/);
    return all.filter((p) => { const hay = `${p.project_id} ${p.name || ""} ${p.project_number || ""}`.toLowerCase(); return words.every((w) => hay.includes(w)); });
  }
  function renderProjectList() {
    const input = $("projectInput"), ul = $("projectList"), msg = $("projectListMsg");
    const typed = input.value.trim();
    const cur = currentProject();
    // Show the whole list when the box still holds the current project (just opened), else filter by what's typed.
    const q = typed === cur ? "" : typed;
    const rows = filteredProjects(q);
    const items = rows.slice(0, PJ_MAX).map((p) => ({ id: p.project_id, p }));
    const exact = (S.pj.list || []).some((p) => p.project_id === typed);
    if (typed && !exact && PROJECT_ID_RE.test(typed)) items.push({ id: typed, use: true });
    S.pj.items = items;
    if (S.pj.idx >= items.length) S.pj.idx = items.length - 1;
    ul.innerHTML = items.map((it, i) => {
      const sel = i === S.pj.idx;
      if (it.use) return `<li role="option" id="pj-opt-${i}" class="po po-use" data-i="${i}" aria-selected="${sel}">${icon("i-plus", "ico-sm")}<span class="po-name">Use “${esc(it.id)}”</span></li>`;
      const p = it.p, untitled = /^sys-/.test(p.project_id) || /^untitled project/i.test(p.name || "");
      const showName = p.name && p.name !== p.project_id;
      return `<li role="option" id="pj-opt-${i}" class="po${untitled ? " po-untitled" : ""}" data-i="${i}" aria-selected="${sel}">` +
        `<span class="po-text"><span class="po-name">${esc(showName ? p.name : p.project_id)}</span>${showName ? `<span class="po-id">${esc(p.project_id)}</span>` : ""}</span>` +
        (p.project_id === cur ? `<span class="po-cur">${icon("i-check", "ico-sm")}<span class="visually-hidden">current</span></span>` : "") + `</li>`;
    }).join("");
    let note = S.pj.msg || "";
    if (!note && S.pj.list && rows.length > PJ_MAX) note = `Showing ${PJ_MAX} of ${rows.length}. Keep typing to narrow the list.`;
    else if (!note && S.pj.open && S.pj.list && q && !rows.length) note = PROJECT_ID_RE.test(typed) ? "Not in your list. You can still use it." : "No matching projects.";
    msg.innerHTML = esc(note) + (S.pj.needSignIn && !(S.auth && S.auth.sign_in_available === false) ? ` <button type="button" class="link-btn" data-act="signin">Sign in</button>` : "");
    const show = S.pj.open && items.length > 0;
    ul.hidden = !show;
    input.setAttribute("aria-expanded", String(show));
    if (show && S.pj.idx >= 0) {
      input.setAttribute("aria-activedescendant", `pj-opt-${S.pj.idx}`);
      const el = $(`pj-opt-${S.pj.idx}`);
      if (el) el.scrollIntoView({ block: "nearest" });
    } else input.removeAttribute("aria-activedescendant");
  }
  function openProjectList() {
    S.pj.open = true;
    renderProjectList();
    loadProjects(false);
  }
  function closeProjectList() {
    S.pj.open = false;
    S.pj.idx = -1;
    renderProjectList();
  }
  async function applyProject(proj, opts) {
    const region = $("regionSelect").value;
    try {
      const st = await postJSON("/api/auth/login", { project: proj || null, trigger_oauth: false, mode: "adc" });
      renderAuth(st);
    } catch (e) { toast(`Couldn't use that project: ${e.message}`, true); return false; }
    if (S.cfg) {
      const projChanged = !!(proj && S.cfg.project !== proj);
      S.cfg.project = proj || S.cfg.project;
      if (S.cfg.region !== region) await applyRegion(region);   // re-plans and re-prices for the region
      else if (projChanged) { S.hw = null; markDirty(); }
    }
    $("projectInput").value = proj || currentProject();
    updateProjectLabel();
    if (proj) checkProject(proj);
    if (opts && opts.toast) toast(`This app now uses project ${proj}. gcloud's own default is unchanged.`);
    return true;
  }
  function selectProjectItem(i) {
    const it = S.pj.items[i];
    if (!it) return;
    $("projectInput").value = it.id;
    closeProjectList();
    applyProject(it.id, { toast: true });
  }
  async function checkProject(proj) {
    const box = $("projectCheck");
    if (!proj || !PROJECT_ID_RE.test(proj) || proj === "my-gcp-project") { box.hidden = true; return; }
    let r = S.pj.checks[proj];
    if (!r) {
      try { r = await api(`/api/projects/${enc(proj)}/check`); S.pj.checks[proj] = r; }
      catch (_) { box.hidden = true; return; }
    }
    if (currentProject() !== proj && $("projectInput").value.trim() !== proj) return;
    const off = r.services.filter((s) => s.enabled === false);
    const on = r.services.filter((s) => s.enabled === true);
    let html = "";
    if (off.length) html = off.map((s) => `<p class="api-off">${icon("i-alert", "ico-sm")}<span>${esc(s.title)} is not enabled in ${esc(proj)}. <a href="${esc(s.enable_url)}" target="_blank" rel="noopener noreferrer">Enable it in the console</a></span></p>`).join("");
    else if (on.length === r.services.length && on.length) html = `<p class="api-on">${icon("i-check", "ico-sm")}<span>${on.map((s) => esc(s.title)).join(" and ")} enabled</span></p>`;
    else if (r.message) html = `<p class="api-unknown">${esc(r.message)}</p>`;
    box.innerHTML = html;
    box.hidden = !html;
  }
  // After switching to an account that can't use the app's project: open the picker and say so (never keep it silently).
  function askForProject(pa) {
    S.pj.needProject = pa.project || true;
    $("projectPop").hidden = false;
    $("projectBtn").setAttribute("aria-expanded", "true");
    const box = $("projectCheck");
    box.innerHTML = `<p class="api-off">${icon("i-alert", "ico-sm")}<span>${esc(pa.message || `The signed-in account can't use project ${pa.project}. Choose a project for this app.`)}</span></p>`;
    box.hidden = false;
    $("projectInput").value = "";
    openProjectList();
  }
  function onProjectsChanged() {
    S.pj.list = null; S.pj.checks = {};
    if (!$("projectPop").hidden) { loadProjects(true); checkProject(currentProject()); }
  }

  // ---- browser-completed sign-in (gcloud --no-launch-browser on the server; the code is pasted here)
  S.signin = { id: null, state: "idle" };
  function signinUI(state, statusText, isErr) {
    S.signin.state = state;
    const link = $("signinLink"), fin = $("signinFinish"), cancel = $("signinCancel"), code = $("signinCode");
    const ready = state === "ready" || state === "failed-code";
    link.setAttribute("aria-disabled", String(!(ready || state === "finishing")));
    if (!S.signin.url || state === "starting" || state === "error") link.removeAttribute("href");
    code.disabled = !ready;
    fin.disabled = state === "starting" || state === "finishing";
    fin.textContent = state === "done" ? "Done" : state === "error" || state === "failed" ? "Start again" : state === "finishing" ? "Finishing…" : "Finish sign-in";
    cancel.hidden = state === "done";
    const st = $("signinStatus");
    st.textContent = statusText || "";
    st.className = "signin-status" + (isErr ? " err" : state === "done" ? " ok" : "");
  }
  async function startSignIn(force) {
    S.signin.url = null; S.signin.id = null;
    $("signinCode").value = "";
    signinUI("starting", "Starting sign-in…");
    try {
      const r = await postJSON("/api/auth/login/start", { mode: "adc", force: !!force });
      S.signin.id = r.login_id; S.signin.url = r.url;
      $("signinLink").href = r.url;
      signinUI("ready", "");
      $("signinLink").focus();
    } catch (e) {
      signinUI("error", `Couldn't start sign-in: ${e.message} You can also run "gcloud auth application-default login" in a terminal on the machine running this app.`, true);
    }
  }
  // ---- Settings: Hugging Face token (contract: GET /api/settings, POST / DELETE /api/settings/hf-token; the token
  // itself never comes back, only a hint such as "hf_…abcd")
  function hfStatusText(t) {
    if (!t || !t.set) return { cls: "off", html: `No token set. Gated models use built-in values.` };
    const who = t.user ? ` · user ${esc(t.user)}` : "";
    if (t.invalid) return { cls: "err", html: `${icon("i-alert", "ico-sm")}<span>Hugging Face no longer accepts the saved token (${esc(t.hint || "")}). Save a new one or remove it.</span>` };
    if (t.source === "env") return { cls: "ok", html: `${icon("i-check", "ico-sm")}<span>Set by the ${esc(t.env_var || "HF_TOKEN")} environment variable (read-only here): ${esc(t.hint || "")}${who}</span>` };
    if (t.source === "memory") return { cls: "ok", html: `${icon("i-check", "ico-sm")}<span>Set for this server session only (its config folder is read-only): ${esc(t.hint || "")}${who}</span>` };
    return { cls: "ok", html: `${icon("i-check", "ico-sm")}<span>Set from file: ${esc(t.hint || "")}${who}</span>` };
  }
  function renderSettings() {
    const t = (S.settings && S.settings.hf_token) || null;
    const st = $("hfTokenStatus");
    if (!t) { st.className = "set-status off"; st.textContent = "Checking…"; return; }
    const v = hfStatusText(t);
    st.className = "set-status " + v.cls;
    st.innerHTML = v.html;
    const env = t.source === "env";
    const managed = !!(S.settings && S.settings.managed);   // cloud deployment: the token comes from Secret Manager
    if (managed && !t.set) { st.className = "set-status off"; st.textContent = "No token on this deployment. Gated models use built-in values. The deployment's owner can add one with Secret Manager (--hf-token-secret)."; }
    $("hfTokenField").hidden = env || managed;
    $("hfTokenSave").hidden = env || managed;
    $("hfTokenRemove").hidden = env || managed || !t.set;
    $("hfTokenSave").textContent = t.set ? "Replace token" : "Save token";
  }
  async function loadSettings() {
    try { S.settings = await api("/api/settings"); } catch (_) { S.settings = null; }
    renderSettings();
    if (S.drawer === "model" && S.models) renderModelDrawer();
  }
  function openSettings() {
    const d = $("settingsDlg");
    $("hfTokenErr").textContent = "";
    $("hfTokenInput").value = "";
    $("hfTokenInput").removeAttribute("aria-invalid");
    renderSettings();
    if (!d.open) d.showModal();
    loadSettings();
    setTimeout(() => (($("hfTokenField").hidden ? $("settingsClose") : $("hfTokenInput"))).focus(), 30);
  }
  async function saveHfToken() {
    const inp = $("hfTokenInput"), err = $("hfTokenErr"), btn = $("hfTokenSave");
    const token = inp.value.trim();
    err.textContent = "";
    inp.removeAttribute("aria-invalid");
    if (!token) { err.textContent = "Paste a token first (it starts with hf_)."; inp.setAttribute("aria-invalid", "true"); inp.focus(); return; }
    btn.disabled = true;
    const label = btn.textContent;
    btn.textContent = "Checking…";
    try {
      const r = await postJSON("/api/settings/hf-token", { token });
      inp.value = "";
      S.settings = { hf_token: r.hf_token };
      renderSettings();
      toast(`Hugging Face token saved${r.user ? " for " + r.user : ""}. Gated models are being refreshed.`);
      S.models = null;   // the model list is refetched on the next open, with the refreshed metadata
    } catch (e) {
      err.textContent = e.message || "Couldn't save the token.";
      inp.setAttribute("aria-invalid", "true");
      inp.focus();
    } finally {
      btn.disabled = false;
      if (btn.textContent === "Checking…") btn.textContent = label;
      renderSettings();
    }
  }
  async function removeHfToken() {
    const ok = await dialog({ title: "Remove the Hugging Face token?", body: "<p>Gated models (such as Llama) go back to built-in values for size, context and tokenizer.</p>", okLabel: "Remove token", danger: true });
    if (!ok) { openSettings(); return; }
    try {
      const r = await api("/api/settings/hf-token", { method: "DELETE" });
      S.settings = { hf_token: r.hf_token };
      toast("Hugging Face token removed.");
      S.models = null;
    } catch (e) {
      toast(`Couldn't remove the token: ${e.message}`, true);
    }
    openSettings();
  }

  function openSignIn() {
    const d = $("signinDlg");
    if (!d.open) d.showModal();
    closePopover();
    startSignIn(S.signin.state === "done");
  }
  async function finishSignIn() {
    const st = S.signin.state;
    if (st === "done") { $("signinDlg").close(); return; }
    if (st === "error" || st === "failed") { startSignIn(true); return; }
    const code = $("signinCode").value.trim();
    if (!code) { signinUI(st, "Paste the code Google showed you after you allowed access.", true); $("signinCode").focus(); return; }
    signinUI("finishing", "Finishing sign-in…");
    try {
      const r = await postJSON("/api/auth/login/finish", { login_id: S.signin.id, code });
      S.signin.id = null;
      if (r.login_ok) {
        S.authLive = true;
        renderAuth(r);
        const who = r.adc_account || r.account;
        const pa = r.project_access || {};
        signinUI("done", `Signed in${who ? " as " + who : ""}.${r.catalog_refreshing ? " Refreshing live prices…" : ""}` +
          (r.needs_project ? ` This account can't use project ${pa.project}: choose a project for this app.` : ""));
        onProjectsChanged();
        if (r.account_changed) { S.models = null; S.hw = null; }
        if (r.needs_project) $("signinDlg").addEventListener("close", () => setTimeout(() => askForProject(pa), 0), { once: true });
        if (r.catalog_refreshing) pollCatalogRefresh();
        $("signinFinish").focus();
      } else {
        signinUI("failed", `Sign-in didn't complete: ${r.login_message || "unknown error"}`, true);
      }
    } catch (e) {
      S.signin.id = null;
      signinUI("failed", `Sign-in didn't complete: ${e.message}`, true);
    }
  }
  async function pollCatalogRefresh() {
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 3000));
      try {
        const cat = await api("/api/catalog");
        if (!(cat.cache && cat.cache.refreshing)) { applyCatalog(cat); if (S.auth) renderAuth(S.auth); toast("Live prices updated."); return; }
      } catch (_) { return; }
    }
  }
  function closePopover() {
    $("projectPop").hidden = true;
    $("projectBtn").setAttribute("aria-expanded", "false");
    closeProjectList();
  }
  function applyCatalog(cat) {
    S.catalog = cat;
    if (!S.authLive) renderAuth(cat.auth);
    const rs = $("regionSelect");
    rs.innerHTML = Object.entries(cat.regions || {}).map(([code, city]) => `<option value="${code}">${code} (${esc(city)})</option>`).join("");
    rs.value = (S.cfg && S.cfg.region) || "us-central1";
    const mp = $("maasPreset");
    mp.innerHTML = `<option value="">Keep current prices</option>` + Object.entries(cat.maas_presets || {}).map(([k, v]) => `<option value="${k}">${esc(v.name)}</option>`).join("");
    if (S.session) {
      buildData();
      renderImageField();
      populateTargetSelects();
      populateGeminiPlaceholders();
      recompute({ animate: false });
    }
    // A background Gemini refresh is running: pick up the new list when it lands.
    if (cat.gemini_meta && cat.gemini_meta.refreshing && !S.gemPoll) {
      S.gemPoll = setTimeout(async () => {
        S.gemPoll = null;
        try {
          const fresh = await api("/api/catalog");
          if (fresh && fresh.gemini_meta && S.catalog && fresh.gemini_meta.updated_at !== (S.catalog.gemini_meta || {}).updated_at) applyCatalog(Object.assign({}, S.catalog, { gemini: fresh.gemini, gemini_models: fresh.gemini_models, gemini_aliases: fresh.gemini_aliases, gemini_meta: fresh.gemini_meta, maas_presets: fresh.maas_presets }));
          else if (fresh && fresh.gemini_meta && fresh.gemini_meta.refreshing) applyCatalog(Object.assign({}, S.catalog, { gemini_meta: fresh.gemini_meta }));
        } catch (_) { /* keep the current list */ }
      }, 8000);
    }
  }
  function currentTheme() {
    const t = document.documentElement.dataset.theme;
    if (t) return t;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function syncThemeButton() {
    const dark = currentTheme() === "dark";
    $("btnTheme").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    $("themeIconUse").setAttribute("href", dark ? "#i-sun" : "#i-moon");
  }

  // ---------------------------------------------------------------- wiring
  function openRail() {
    $("rail").classList.add("open");
    $("scrim").hidden = false;
    $("btnEditWorkload").setAttribute("aria-expanded", "true");
    setTimeout(() => { const s = $("rail").querySelector("summary"); if (s) s.focus(); }, 30);
  }
  function closeRail() {
    if (!$("rail").classList.contains("open")) return;
    $("rail").classList.remove("open");
    $("scrim").hidden = true;
    $("btnEditWorkload").setAttribute("aria-expanded", "false");
    $("btnEditWorkload").focus();
  }

  function trapFocus(container, e) {
    const f = Array.from(container.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea, summary, [tabindex]:not([tabindex="-1"])')).filter((x) => x.offsetParent !== null);
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function wire() {
    const form = $("railForm");
    form.addEventListener("input", onRailInput);
    $("btnGemAll").addEventListener("click", () => { S.gemAll = !S.gemAll; populateGeminiPlaceholders(); $("btnGemAll").focus(); });
    $("btnGemRefresh").addEventListener("click", refreshGemini);
    form.addEventListener("change", (e) => { if (e.target.tagName === "SELECT" || e.target.type === "radio" || e.target.type === "checkbox" || e.target.id === "f-monthly" || e.target.id === "f-imageCustom") onRailInput(e); });
    $("f-monthly").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); onRailInput({ target: e.target, type: "change" }); } });
    $("anchorAvg").addEventListener("click", () => setAnchor("avg"));
    $("anchorMonthly").addEventListener("click", () => setAnchor("monthly"));
    $("btnCustomReset").addEventListener("click", () => shapeToSimple());
    $("btnCustomEdit").addEventListener("click", () => { closeRail(); enterShapeEdit(); });
    // Precision: a radio group over the visible options (arrow keys move and select).
    document.querySelectorAll("#precSeg [data-prec]").forEach((b) => {
      b.addEventListener("click", () => setPrecision(b.dataset.prec));
      b.addEventListener("keydown", (e) => {
        if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].indexOf(e.key) < 0) return;
        e.preventDefault();
        const btns = Array.from(document.querySelectorAll("#precSeg [data-prec]")).filter((x) => !x.hidden);
        if (btns.length < 2) return;
        const i = btns.indexOf(b), step = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
        const next = btns[(i + step + btns.length) % btns.length];
        next.focus();
        setPrecision(next.dataset.prec);
      });
    });
    $("maasPreset").addEventListener("change", (e) => {
      const p = S.catalog && S.catalog.maas_presets && S.catalog.maas_presets[e.target.value];
      if (!p || !S.cfg) return;
      Object.assign(S.cfg.maas, { input_per_m: p.input_per_m, output_per_m: p.output_per_m, cached_per_m: p.cached_per_m, max_rps: p.max_rps });
      populateRail(); markDirty(); scheduleRecompute();
      e.target.value = "";
    });
    $("btnUseParametric").addEventListener("click", () => {
      S.cfg.traffic.hourly_csv = null;
      populateRail(); markDirty(); scheduleRecompute();
    });
    initMyPrompts();
    $("btnMonImport").addEventListener("click", openMonitoring);

    // top bar
    const pop = $("projectPop"), pbtn = $("projectBtn"), pin = $("projectInput");
    pbtn.addEventListener("click", () => {
      const open = pop.hidden;
      if (!open) { closePopover(); return; }
      pop.hidden = false;
      pbtn.setAttribute("aria-expanded", "true");
      pin.value = currentProject();
      pin.focus();
      pin.select();
      loadProjects(false).then(() => { if (!pop.hidden) renderProjectList(); });
      checkProject(currentProject());
    });
    document.addEventListener("click", (e) => {
      if (pop.hidden) return;
      if (!pop.contains(e.target) && !pbtn.contains(e.target)) closePopover();
      else if (pop.contains(e.target) && !$("projectCombo").contains(e.target)) closeProjectList();
    });
    pop.addEventListener("keydown", (e) => { if (e.key === "Escape") { closePopover(); pbtn.focus(); } });
    // No open-on-focus: the popover focuses the box when it opens, and the list would cover the region and buttons.
    // Click, typing, ArrowDown or the chevron open it.
    pin.addEventListener("click", () => { if (!S.pj.open) openProjectList(); });
    pin.addEventListener("input", () => {
      S.pj.open = true;
      S.pj.idx = pin.value.trim() ? 0 : -1;
      renderProjectList();
      loadProjects(false);
    });
    pin.addEventListener("keydown", (e) => {
      const n = S.pj.items.length;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!S.pj.open) { openProjectList(); return; }
        if (!n) return;
        const d = e.key === "ArrowDown" ? 1 : -1;
        S.pj.idx = S.pj.idx < 0 ? (d > 0 ? 0 : n - 1) : (S.pj.idx + d + n) % n;
        renderProjectList();
      } else if (e.key === "Home" && S.pj.open && n && e.ctrlKey) { e.preventDefault(); S.pj.idx = 0; renderProjectList(); }
      else if (e.key === "End" && S.pj.open && n && e.ctrlKey) { e.preventDefault(); S.pj.idx = n - 1; renderProjectList(); }
      else if (e.key === "Enter") {
        e.preventDefault();
        if (S.pj.open && S.pj.idx >= 0 && S.pj.items[S.pj.idx]) selectProjectItem(S.pj.idx);
        else if (pin.value.trim()) { closeProjectList(); applyProject(pin.value.trim(), { toast: true }); }
      } else if (e.key === "Escape" && S.pj.open) {
        e.stopPropagation();
        closeProjectList();
      } else if (e.key === "Tab") closeProjectList();
    });
    $("projectList").addEventListener("mousedown", (e) => e.preventDefault());   // keep focus in the input
    $("projectList").addEventListener("click", (e) => {
      const li = e.target.closest("li[data-i]");
      if (li) selectProjectItem(+li.dataset.i);
    });
    $("projectToggle").addEventListener("mousedown", (e) => e.preventDefault());
    $("projectToggle").addEventListener("click", () => { if (S.pj.open) closeProjectList(); else { pin.focus(); openProjectList(); } });
    $("projectListMsg").addEventListener("click", (e) => { if (e.target.closest("[data-act=signin]")) openSignIn(); });
    $("regionSelect").addEventListener("change", () => applyRegion($("regionSelect").value));
    $("btnApplyProject").addEventListener("click", async () => {
      if (await applyProject(pin.value.trim())) closePopover();
    });
    $("btnSignIn").addEventListener("click", openSignIn);
    // IAP handles this query itself: it clears its session cookie and sends the browser through Google sign-in again
    // (pick another account there, or sign out of Google to leave completely).
    $("btnSignOut").addEventListener("click", () => { location.href = "/?gcp-iap-mode=CLEAR_LOGIN_COOKIE"; });
    $("btnSwitchAccount").addEventListener("click", openSignIn);
    $("signinForm").addEventListener("submit", (e) => { e.preventDefault(); finishSignIn(); });
    $("signinLink").addEventListener("click", (e) => {
      if ($("signinLink").getAttribute("aria-disabled") === "true") { e.preventDefault(); return; }
      setTimeout(() => $("signinCode").focus(), 50);
    });
    const cancelSignIn = () => {
      if (S.signin.id && S.signin.state !== "done") postJSON("/api/auth/login/cancel", { login_id: S.signin.id }).catch(() => {});
      S.signin.id = null;
      if (S.signin.state !== "done") S.signin.state = "idle";
    };
    $("signinCancel").addEventListener("click", () => $("signinDlg").close());
    $("signinDlg").addEventListener("close", cancelSignIn);
    $("sessionSelect").addEventListener("change", async (e) => {
      const name = e.target.value;
      if (S.dirty) {
        const ok = await dialog({ title: "Discard unsaved changes?", body: `<p>You have unsaved changes in "${esc(S.name)}". Switching sessions discards them.</p>`, okLabel: "Discard and switch", danger: true });
        if (!ok) { e.target.value = S.name; return; }
      }
      if (S.sse) { S.sse.close(); S.sse = null; }
      stopPolling();
      await loadSession(name);
    });
    $("btnNewWorkload").addEventListener("click", showFirstRun);
    $("btnSave").addEventListener("click", save);
    $("runPill").addEventListener("click", () => { closeRail(); selectTab("bench", true); $("panel-bench").scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" }); });
    $("btnSettings").addEventListener("click", openSettings);
    // the model drawer's data note re-renders with the list: delegate its "Add a Hugging Face token" link
    document.addEventListener("click", (e) => { if (e.target.closest && e.target.closest("#btnMdlSettings")) openSettings(); });
    $("settingsClose").addEventListener("click", () => $("settingsDlg").close());
    $("settingsForm").addEventListener("submit", (e) => { e.preventDefault(); saveHfToken(); });
    $("hfTokenRemove").addEventListener("click", () => { $("settingsDlg").close(); removeHfToken(); });
    $("btnTheme").addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("dilemma-theme", next); } catch (_) { /* ignore */ }
      syncThemeButton();
      if (S.shape.on) drawShape(); else if (S.drawn) drawWaterline(S.target || S.drawn);
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (ev) => {
      let stored = null;
      try { stored = localStorage.getItem("dilemma-theme"); } catch (_) { /* ignore */ }
      if (!stored) document.documentElement.dataset.theme = ev.matches ? "dark" : "light";
      syncThemeButton(); if (S.shape.on) drawShape(); else if (S.drawn) drawWaterline(S.target || S.drawn); });

    // chips + drawers
    $("chipModel").addEventListener("click", (e) => openDrawer("model", e.currentTarget));
    $("chipHw").addEventListener("click", (e) => openDrawer("hw", e.currentTarget));
    $("drawerClose").addEventListener("click", closeDrawer);
    $("drawerScrim").addEventListener("click", closeDrawer);
    $("drawerWide").addEventListener("click", () => {
      S.drawerWide = !S.drawerWide;
      $("drawer").classList.toggle("wide", S.drawerWide);
      const b = $("drawerWide");
      b.setAttribute("aria-pressed", S.drawerWide);
      b.setAttribute("aria-label", S.drawerWide ? "Normal width" : "Full width"); b.title = b.getAttribute("aria-label");
      b.querySelector("use").setAttribute("href", S.drawerWide ? "#i-collapse" : "#i-expand");
    });
    $("viewTable").addEventListener("click", () => { S.drawerView[S.drawer] = "table"; syncViewToggle(); renderDrawer(); });
    $("viewCards").addEventListener("click", () => { S.drawerView[S.drawer] = "cards"; syncViewToggle(); renderDrawer(); });
    $("drawer").addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        const openWhy = $("drawer").querySelector("details.why[open]");
        if (openWhy) { openWhy.open = false; openWhy.querySelector("summary").focus(); return; }
        closeDrawer();
      } else if (e.key === "Tab") trapFocus($("drawer"), e);
    });

    // mobile rail
    $("btnEditWorkload").addEventListener("click", openRail);
    $("btnRailDone").addEventListener("click", closeRail);
    $("scrim").addEventListener("click", closeRail);
    $("rail").addEventListener("keydown", (e) => {
      if (!$("rail").classList.contains("open")) return;
      if (e.key === "Escape") closeRail();
      else if (e.key === "Tab") trapFocus($("rail"), e);
    });

    // tabs
    TABS.forEach((t, i) => {
      const b = $("tab-" + t);
      b.addEventListener("click", () => selectTab(t));
      b.addEventListener("keydown", (e) => {
        let j = null;
        if (e.key === "ArrowRight") j = (i + 1) % TABS.length;
        else if (e.key === "ArrowLeft") j = (i - 1 + TABS.length) % TABS.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = TABS.length - 1;
        if (j == null) return;
        e.preventDefault();
        selectTab(TABS[j], true);
      });
    });
    document.addEventListener("click", (e) => {
      const g = e.target.closest("[data-goto-tab]");
      if (g) { if (S.drawer) closeDrawer(); selectTab(g.dataset.gotoTab, true); $("panel-" + g.dataset.gotoTab).scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" }); }
      const rs = e.target.closest("[data-run-step]");
      if (rs) runBench("step", [rs.dataset.runStep]);
    });

    // benchmark
    $("btnRunBench").addEventListener("click", () => runBench("auto", null));
    $("benchServing").addEventListener("click", (e) => {
      if (e.target.closest("#btnEditImage")) editServingImage();
      if (e.target.closest("#btnEditPrompts")) {
        const sec = $("promptSrcLabel").closest("details");
        if (getComputedStyle($("btnEditWorkload")).display !== "none" && !$("rail").classList.contains("open")) openRail();
        if (sec) sec.open = true;
        setTimeout(() => { $("promptSrcLabel").scrollIntoView({ block: "start", behavior: reduceMotion.matches ? "auto" : "smooth" }); $(S.cfg.prompts.source === "file" ? "promptSrcMine" : "promptSrcSyn").focus(); }, 60);
      }
    });
    $("benchMeter").addEventListener("click", (e) => {
      if (!e.target.closest("#btnUsePinned") || !pinnedImage()) return;
      S.cfg.deployment.image = pinnedImage().image;
      S.imageChoice = null;
      renderImageField();
      markDirty();
      renderBench();
      toast("Serving image set to the pinned image. Run the benchmark again to use it.");
      $("benchServing").querySelector("#btnEditImage").focus();
    });
    $("btnRunSim").addEventListener("click", runSim);
    $("btnRunStep").addEventListener("click", () => runBench("step", [$("stepSelect").value]));
    $("btnStopBench").addEventListener("click", async () => {
      try {
        const r = await postJSON(`/api/sessions/${enc(S.name)}/stop`, { teardown: false });
        if (r.job) S.job = Object.assign({}, S.job, r.job, { logs: (S.job && S.job.logs) || r.job.logs });
        if (S.job && S.job.status === "running" && !S.job.stop_requested_at) S.job.stop_requested_at = Date.now() / 1000;
        toast("Stopping after the current request batch…");
        renderBench();
      } catch (e) { toast(`Couldn't stop the run: ${e.message}`, true); }
    });
    $("btnForceStop").addEventListener("click", async () => {
      const ok = await dialog({
        title: "Force stop the benchmark?",
        body: "<p>This kills the benchmark process at once. It can't tear down its endpoint, so check the Benchmark tab afterwards and tear the endpoint down if it's still there. A deployment that was in progress may finish on Agent Platform without being recorded: check the Agent Platform console too.</p>",
        okLabel: "Force stop", danger: true,
      });
      if (!ok) return;
      try { await postJSON(`/api/sessions/${enc(S.name)}/stop`, { force: true }); }
      catch (e) { toast(`Couldn't force stop the run: ${e.message}`, true); }
      await refreshSession();
    });
    const confirmTeardown = async () => {
      const ep = S.session && S.session.endpoint;
      const ok = await dialog({
        title: "Tear down the endpoint?",
        body: `<p>This undeploys the model and deletes endpoint ${esc(String((ep && ep.endpoint) || "").split("/").pop())} on Agent Platform. Billing stops once it's gone. You can deploy again from the benchmark.</p>`,
        okLabel: "Tear down endpoint", danger: true,
      });
      if (ok) runBench("step", ["teardown"]);
    };
    $("btnTeardown").addEventListener("click", confirmTeardown);
    $("btnAlertTeardown").addEventListener("click", confirmTeardown);

    // config
    $("yamlEditor").addEventListener("input", () => { if (!S.yamlDirty) { S.yamlDirty = true; renderConfig(); } });
    $("btnDiscardYaml").addEventListener("click", () => { S.yamlDirty = false; $("yamlEditor").value = S.session.yaml_text || ""; renderConfig(); });
    $("btnApplyYaml").addEventListener("click", async () => {
      if (S.dirty) {
        const ok = await dialog({ title: "Replace your unsaved changes?", body: "<p>Applying the YAML replaces the unsaved changes you made in the rail.</p>", okLabel: "Apply YAML", danger: true });
        if (!ok) return;
      }
      const text = $("yamlEditor").value;
      if (!(await ensureWritable())) return;
      try {
        const res = await postJSON(`/api/sessions/${enc(S.name)}/yaml`, { yaml_text: text });
        S.yamlDirty = false;
        applySession(res, { resetUi: false });
        toast("Applied workload.yaml.");
      } catch (e) {
        toast(`The YAML wasn't applied: ${e.message}. Fix it and apply again.`, true);
      }
    });
    document.querySelectorAll(".downloads a").forEach((a) => a.addEventListener("click", (e) => { if (a.getAttribute("aria-disabled") === "true") e.preventDefault(); }));

    window.addEventListener("beforeunload", (e) => { if (S.dirty || S.yamlDirty) { e.preventDefault(); e.returnValue = ""; } });
    initWaterlineInteraction();
    initShapeEditor();
    initFirstRun();
  }

  // ---------------------------------------------------------------- boot
  async function init() {
    syncThemeButton();
    wire();
    selectTab("costs");
    const catP = api("/api/catalog").then(applyCatalog).catch((e) => {
      console.warn("Catalog unavailable", e);
      $("authStatusText").textContent = "Couldn't reach the catalog. Prices use the reference values.";
    });
    api("/api/auth/status").then((st) => { S.authLive = true; renderAuth(st); }).catch(() => {});
    loadSettings();
    await loadSessions();
    const q = new URL(location.href).searchParams.get("session");
    let last = null;
    try { last = localStorage.getItem("dilemma-last-session"); } catch (_) { /* ignore */ }
    const names = new Set(S.sessions.map((s) => s.name));
    const own = S.sessions.filter((s) => !(s.example || s.is_demo));
    let pick = null;
    if (q && names.has(q)) pick = q;
    else if (last && names.has(last)) pick = last;
    else if (own.length) pick = own[0].name;
    if (pick && (await loadSession(pick))) { /* loaded */ }
    else showFirstRun();
    await catP;
  }

  init();
})();
