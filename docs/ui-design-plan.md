# aiops-dilemma UI overhaul: design plan and contract

## Subject, audience, job

- **Subject:** capacity planning for LLM inference: a fixed, reserved accelerator fleet vs per-token APIs
  (open-model MaaS, Gemini on Agent Platform).
- **Audience:** platform / ML-infra engineers and the FinOps or engineering manager they report to.
  Comfortable with numbers; impatient with forms.
- **Primary job:** get a defensible answer in under a minute ("self-host on N× GPU for $X/month, Y% of MaaS"),
  see *why* (the shape of the week vs the fleet), then refine and, optionally, validate on real hardware.

## The memorable thing: the capacity waterline

The hero is the week of traffic drawn against the fleet's capacity, a horizontal **waterline** the user can
drag. Below the line and under the curve is traffic served on the fleet. Below the line and above the curve is
paid-for idle capacity. Above the line, the curve spills to MaaS (up to its rate limit) and then to Gemini.
Dragging the line, or changing any input, moves the verdict sentence and the numbers live (< 100 ms, no
server round trip; `model.js` computes it). Releasing snaps to whole replicas. A "Reset to optimal" control
appears when the user has moved it off the cost-optimal fleet. This is the single bold element; everything
else stays quiet.

**Fleet policy** (one rule for the optimum and the waterline, governed by "Peak overflow goes to",
`hybrid.peak_overflow_target`). Peak overflow always fills open-model MaaS within its limit first. Default
`open_maas`: Gemini is not allowed at peaks. A fleet is within the policy when (a) no hour spills beyond what the
peak-overflow target may take (the MaaS cap; with a Gemini tier, MaaS cap + 150 req/s Gemini quota) and (b) it has
at least `(peak − cap) × headroom / per-chip` chips (burst headroom). The optimum is the cheapest policy-OK fleet,
which equals the cheapest policy-OK point on the waterline. Dragging below the smallest policy-OK fleet is still
allowed (a what-if; the spill beyond the MaaS cap is priced at the failover tier), but the UI never presents it as
the better option: the region below the policy minimum is hatched amber (stronger while the fleet is in it, label
"Below: outside the fleet policy"), the handle turns amber with an alert icon, the verdict says what the fleet
would cost and that it's outside the policy, then anchors on the policy optimum, the "Monthly cost" figure says
"What-if outside the fleet policy: …" and a warning line under the chart gives the reasons ("Needs Gemini at peaks:
not allowed while peak overflow goes to open-model MaaS." / "Below the 1.3× burst headroom: …"). With the default
policy, when peak overflow to the failover tier (`hybrid.failover_target`) would be ≥ 3% cheaper, the verdict adds
"Allowing <tier> overflow at peaks would cut it to N GPUs and save about $XK a month." with an "Allow Gemini at
peaks" button that sets `hybrid.peak_overflow_target` (same line in report.md and dashboard.html).

## Tokens

### Color (light theme; dark theme uses its own steps of the same hues, validated against the dark surface)

| Name | Hex | Use |
|---|---|---|
| Fog | `#EEF1F4` | App background |
| Sheet | `#FFFFFF` | Raised surfaces (rail, drawer, tab panels) |
| Slate ink | `#15232E` | Primary text, axis labels |
| Slate muted | `#5B6B78` | Secondary text (check ≥ 4.5:1 on Fog and Sheet) |
| Fleet teal | `#0E7C74` | Served-on-fleet area, primary actions, focus ring |
| Overflow amber | `#D9901F` | Spill to open-model MaaS |
| Gemini violet | `#6D5AE6` | Spill or failover to Gemini |
| Idle | Fleet teal at low opacity + 45° hatch | Paid-for idle capacity |
| Alert red | `#C2402F` | Errors, destructive actions (teardown) only |

Validate the chart palette with the dataviz skill's validator
(`find / -name validate_palette.js -path '*dataviz*' 2>/dev/null`) for both modes; fix anything that FAILs.
Text never wears series colors.

### Type

- **Archivo** (variable: weight 100–900, width 62–125), bundled locally as woff2 under
  `src/aiops_dilemma/assets/fonts/` with a `system-ui` fallback. One family, using the width axis as the
  type treatment: the verdict's numbers are set wide and heavy (`font-stretch: 125%`, weight 750); UI text is
  normal width; dense tables use semi-condensed (`font-stretch: 87.5%`) with tabular figures.
- Scale (1.25 ratio): 12 / 14 (body) / 17.5 / 22 / 27 / 34 / 42 px. Body line-height 1.5; headings 1.2.
  Sentence case everywhere. No all-caps labels, no eyebrows above headings.
- Line length ≤ 75 characters for prose; numbers right-aligned in tables.

### Layout

```
┌ aiops-dilemma   [project ▾]  [session ▾]                     ● run status   ◐ ┐
├───────────────┬──────────────────────────────────────────────────────────────┤
│ Workload      │ Self-host Gemma 4 26B on 9 × RTX PRO 6000 — $15K a month,     │
│ (rail, the    │ 99% of what MaaS would cost.            [model ▾] [GPU ▾] FP8 │
│ only place    │ ┌──────────────────────────────────────────────────────────┐ │
│ inputs live)  │ │  week curve · draggable fleet waterline · spill colors    │ │
│               │ └──────────────────────────────────────────────────────────┘ │
│ ▸ Traffic     │  4 key figures in one quiet row (GPUs · $/month · % of MaaS · │
│ ▸ Prompts     │  avg utilization), each with a provenance tag where relevant  │
│ ▸ Pricing     │ ┌ Costs │ Resilience │ Long prompts │ Benchmark │ Config ┐    │
│ ▸ Routing     │ │ tab content                                               │ │
│ ▸ Advanced    │ └───────────────────────────────────────────────────────────┘ │
└───────────────┴──────────────────────────────────────────────────────────────┘
```

- Left-aligned throughout. Rail 320 px, collapsible sections (details/summary semantics), sticky.
- Below 900 px the rail becomes a bottom sheet / top drawer ("Edit workload"), the verdict and chart come first,
  tabs become a horizontally scrollable segmented control. No horizontal page scroll at 360 px.
- Surfaces: one raised level (Sheet) on Fog. No card grid of identical rounded boxes; separate sections with
  space and a single hairline where needed. Radius 6 px for controls, 10 px for the drawer; no shadow except
  the drawer and popovers.

### First run (empty or example session)

A focused start screen instead of 25 fields: three paths, side by side on desktop, stacked on mobile:
1. **Upload logs and prompts** (drop zone; auto-detects traffic CSV, request logs, prompt JSONL, system prompts).
2. **Start from the example** (loads the blog workload as a new, editable copy).
3. **Describe it in six numbers** (peak, average, quietest hour tok/s, avg input / output tokens, latency budget).
Any path lands on the main view with a verdict already computed.

### Model and hardware

Opened from the chips next to the verdict, in a right-side drawer (sheet). Each has a **comparison table**
(default) with a card view toggle:
- Model: name, total / active params, context, fits your longest prompts?, recommended precision, est. chips,
  est. $/M input tokens, match score (with a "why" popover showing the score breakdown). The current selection
  is pinned on top and marked "Selected". If the recommendation differs from the selection, show a one-line
  prompt: "Gemma 4 26B fits better (99% vs 64%). Switch?"
- Hardware: accelerator, chips per replica (with a note when the machine shape forces it), fleet size,
  $/M input tokens (sort key), $/GPU-hour per pricing option, est. throughput per chip with its provenance tag.
- Long tips ("divert >32K prompts…") appear once above the table, never per row.

### Provenance

Every throughput figure and everything derived from it carries a tag: **Estimated** (roofline from specs),
**Calibrated** (scaled by a profile run), or **Measured** (load test on Agent Platform). An instant estimate never
shows a concurrency sweep table or latency chart; it shows one line: "Assumed 5,700 input tok/s per chip,
estimated from specs. Run a benchmark to measure it."

**Estimate tiers (sub-labels of Estimated).** `planner.reference_throughput(cfg, acc, count)` returns
`{per_chip, basis, label, note, spec_per_chip, reference, model_scale, workload_scale}`; candidates carry
`est_input_tps_per_gpu` (still the short-prompt rate), `est_basis`, `est_basis_label`, `est_basis_note`,
`spec_input_tps_per_gpu`. `resolve_throughput` keeps its order (measured > override > saved override/calibrated >
estimated); an estimated result adds `basis`, `basis_label`, `note` (the session's `throughput`), and an estimate's
measured.json records `throughput_basis` + `throughput_note`.

| basis | tag | label | when |
|---|---|---|---|
| `reference` | Reference est. | Reference benchmark | `catalog.REFERENCE_BENCHMARKS` entry for this accelerator × count, model, precision: its rate × workload adjustment |
| `scaled` | Scaled est. | Scaled from a reference benchmark | entry on this accelerator with another model / precision / count: × spec-formula model ratio × workload adjustment |
| `spec` | Rough estimate (amber, dashed) | Rough spec estimate | no entry for this accelerator: the spec formula |

Tooltip (`note`), e.g. "Measured on 1x RTX PRO 6000 with the reference workload (Gemma 4 26B-A4B FP8, ~5K-token
prompts, 500-token answers, no thinking) on <date>. Adjusted to your output length and prompt size: x0.62. Your
workload may differ: run a benchmark with your prompts." Recommender alternatives use the same basis
(`calibration_k` = session rate / reference-anchored estimate for the session's hardware).

**Data provenance (live vs built-in, 2026-11-09).** Every curated model entry (`/api/catalog` `models`, and the
recommendations built from them) carries:

- `field_sources`: `{field: "hf" | "model_garden" | "rules"}` for `params_b`, `active_params_b`, `max_context`,
  `released`, `family`, `native_precision`, `precisions`, `license`, `gated`, `checkpoint_precision`.
- `data_source`: `{kind: "hf" | "mixed" | "rules", label, hf_id, updated_at (YYYY-MM-DD), estimated_fields: [...],
  notes: [...]}`; `label` is "Hugging Face, updated <date>" (+ "; built-in estimate for context, …" when mixed) or
  "built-in estimate".
- `checkpoint_precision` (bf16 / fp8 / int4 / int8 / nvfp4 / mxfp4, from `quantization_config` / `torch_dtype`),
  `license`, `gated`, `model_type` when the Hub provides them.

The catalog adds `model_metadata: {models, hugging_face, built_in, refreshing, last_refresh}` and
`hardware: {source: "compute_api" | "fallback", region, fetched_at, note, offered: {acc: {name, kind, machines:
{count: machine type}, zones}}, not_offered: [{accelerator_type, name, kind, note: "Not offered in <region>"}],
unrated: [{accelerator_type, name, kind, zones, machines, note}]}`. `/api/recommend/hardware` returns the same
`availability` block in `plan.availability` (and top-level with `no_fit`); a manually chosen accelerator the region
lacks gets `chosen.region_note`.

UI: the model drawer shows a muted "Data: Hugging Face, updated <date>" / "~ Data: built-in estimate" line under
each model (table and cards; tooltip = source per field, license, notes, Hub id) and a note above the list when some
models use built-in estimates. The hardware drawer ends with a provenance line ("Accelerators and machine shapes
offered in <region>: Compute Engine API, updated <date>" or "built-in catalog"), "Not offered in <region>: …" and a
collapsed list of accelerators offered there without performance data ("no performance data, rough: not
estimated"). Changing the region drops the cached hardware list.

**Warnings.** Estimated and Your value / Assumed tags carry a caution icon; the provenance line ends with a
"Not measured on your workload" caution (`.caution`, a button with `data-goto-tab="bench"`; it closes the drawer),
and the hardware drawer's throughput column header has an "Estimates" caution. A first-use dialog (`#dlg`, "I
understand" / "Run a benchmark") appears once on the main page (`localStorage dilemma-seen-estimates-notice`) and
once on first opening the hardware drawer (`dilemma-seen-hw-estimates-notice`). report.md and dashboard.html carry
`report.DISCLAIMER` in their header whenever throughput isn't from a load test.

### Benchmark tab

Real runs only. A step list (plan → prompts → deploy → load sweep → analyze → report → teardown) that reflects
live SSE state, elapsed time, and a running cost meter (endpoint $/hour × time). One Teardown button, shown only
while an endpoint exists, with a confirm step. Sweep table and throughput-vs-latency chart fill in level by level.
Jobs are durable (see "Benchmark jobs" below): reloading the page or restarting the server reattaches to a running
job; jobs started from the CLI show up too ("started from the command line: `dilemma auto …`"). Stop turns into
"Stopping…"; if the job is still running 60 s later, "Force stop" appears (confirm dialog, SIGKILL). A job that
died while its endpoint existed shows a red alert ("A benchmark stopped unexpectedly and its endpoint may still
be running (billing). Tear it down") with the guarded teardown button, the header pill turns red ("Benchmark
interrupted · endpoint may be billing") and the session picker says "(interrupted: endpoint may be billing)".
A job killed mid-deploy (no endpoint recorded yet) gets an alert to check the Agent Platform console instead.

### Motion

Only in answer to the user: the waterline and areas ease to new values (150 ms) when inputs change, the drawer
slides in, a run step checks off. No entrance animations. Respect `prefers-reduced-motion`.

### Copy

Plain verbs, sentence case, buttons say what happens ("Run benchmark", "Tear down endpoint", "Download
workload.yaml"). Errors say what failed and what to do. No emoji icons: inline SVG with `aria-hidden`, text
labels always present.

## Principles

1. One answer, always visible, always current.
2. Inputs live in one place (the rail); outputs everywhere else.
3. Every number knows where it came from.
4. Spend boldness on the waterline; keep the rest quiet.
5. Works at 360 px, keyboard-only, and with reduced motion.

## Backend ↔ frontend contract

Fields the frontend relies on. The backend agent implements them; the frontend agent consumes them and must not
re-derive them differently.

- `GET /api/sessions` → each item gains `"example": true|false`. Example sessions are read-only: saving one
  creates a copy (`POST /api/sessions/{new}` with `"copy_of": "blog-example"`).
- `GET /api/sessions/{name}` (session payload) gains:
  - `"selected": {"model": {"model_garden_id", "hf_id", "name"}, "hardware": {"accelerator_type",
    "accelerator_count", "machine_type"}, "precision"}` — the single source of truth for what is being planned.
    `hardware` is null when nothing fits the chosen model; then `"plan_error": str` and `"no_fit"` (below) explain why.
  - `"no_fit": null | {reason: "too_big"|"native_only"|"no_int4"|"no_mxfp4", message, model, precision, params_b, weights_gb,
    largest: {accelerator_type, accelerator_count, machine_type, gpu, vram_gb, weights_limit_gb},
    available_precisions, options: [{label, action?: {type: "precision", value} | {type: "models"}}]}`
    (`planner.no_fit_explanation`). `POST /api/sessions/{name}/estimate` returns 200 with it (the new model stays
    selected; the previous plan / estimate / report files are removed) instead of a 400; the main page shows the
    message and the options (precision switch, "Pick a smaller model") in place of the verdict, the model chip shows
    the new model and the hardware chip "None fits". `POST /api/recommend/hardware` → `{ok: false, error, no_fit}`.
  - `"throughput": {"per_chip": float, "source": "estimated"|"calibrated"|"measured"|"override",
    "label": str}` — used by every view; the instant estimate never contradicts it.
  - `measured` for estimates is `{"estimated": true, "levels": [], "input_tps_per_gpu": …, …}`; real runs keep
    `levels`.
  - `config.constraints.resilience_multiplier` is always one of 1, 1.5, 2 (normalized on load and save).
- `POST /api/recommend/model` → each model gains `"fits_longest_prompt": bool`, `"warnings": [str]`,
  `"score_breakdown": {component: points}`, `"recommended_precision"` (the published precision for FP8- / INT4-only
  models; BF16 when it fits on one chip, else FP8 with `"needs_quality_check": true`; INT4 only when even FP8 fits no
  single machine and a 4-bit checkpoint exists), `"precision_options": [{precision, label, how: "native"|
  "prequantized"|"on_the_fly", how_label, weights_gb, size_text, fits_on, fits_tight, fit_text, recommended,
  quality_note}]` (only precisions that exist; the quality note sits on the recommended item and names it),
  `"is_selected": bool` (the UI recomputes the selected row from the current selection; a cached response's flag is
  stale after choosing another model, so choosing also drops the cached list); response gains `"tips": [str]`.
- `POST /api/recommend/hardware` → each candidate gains `"fleet_gpus": int`, `"usd_per_m_input": float` (Agent Platform +
  reservation 3-yr, the default sort), `"shape_note": str|null`; pricing tier labels use "Agent Platform + reservation, 3-yr".
- `GET /api/catalog` → served from an on-disk cache (fast after first call). Gemini (live catalog, `gemini.py`):
  `"gemini"` = the engines' map `{role alias or model id: {id, model_id, name, family, version, stage, input_per_m,
  output_per_m, cached_per_m, long_context: {threshold, input_per_m, output_per_m, cached_per_m}|null,
  long_input_per_m, long_output_per_m, source: "billing-catalog"|"fallback", alias?, role?, rule?}}` (aliases
  `gemini_flash_lite`, `gemini_flash`, `gemini_pro`, `gemini_4` first, each a copy of the model it resolves to);
  `"gemini_models"` = every Gemini text model with per-token SKUs `[{id, name, family, group, version, stage
  ("ga"|"preview"|"deprecated"|"unlisted"|null), stage_label, deprecated, input_per_m, output_per_m, cached_per_m,
  long_context, batch, regional, source, first_seen, released, context_window, model_garden_id, roles, skus}]`,
  newest first within family; `"gemini_aliases"` = `{alias: {key, label, rule, model_id, name, source}}`;
  `"gemini_meta"` = `{source, updated_at, model_garden, billing, counts, errors, refreshing}`.
- `POST /api/gemini/refresh {project, region}` → re-reads the Billing Catalog + Model Garden now; returns
  `{gemini, gemini_models, gemini_aliases, gemini_meta}` and updates the cached catalog. Session payloads carry
  `"gemini_warnings": [str]` (a pinned model the live catalog no longer offers, replaced by its role alias on load,
  or a deprecated one).
- UI (Gemini): the three target selects (prompts over 32K, peak overflow, failover) list "Newest of each kind" (role
  aliases, "Gemini Flash-Lite (latest: Gemini 3.5 Flash-Lite)") then every offered model grouped by family
  (Flash-Lite, Flash, Pro, Gemini 4 …), previews "(Preview)", deprecated models only when selected, a pinned model the
  catalog dropped as "(no longer offered: priced as …)". A warning notice under Routing (and a toast when the session
  opens) names a replaced or deprecated pinned model and its replacement. The rail's Gemini section shows "Gemini
  prices from the Cloud Billing Catalog, updated <time>" with Refresh, a compact editable price table (models the
  aliases resolve to, pinned models and overridden ones; role tags, Preview, ">200K tier" with the long prices on
  hover; dashed input = override) and "Show all (N more)" grouped by family. The Long prompts table compares the role
  aliases and the routed model, with "Show all Gemini models"; the incident table adds a column for a pinned failover
  model. Overrides are saved in `ui_state.gem`.
- `window.DilemmaModel` (`assets/model.js`, shared with the exported dashboard): `scenario(DATA, p)` and
  `incidents(DATA, p, r)` guard `p.resilience` (< 1 or missing → 1); incident results include
  `"breakeven_outage_hours"` capped at 730 with `"breakeven_never": true` when failover is cheaper at any
  realistic outage rate. `fleetOverride` (whole replicas) evaluates the waterline's fleet. Fleet-policy fields
  (Python snake_case in `costmodel.scenario`): `policyOk`, `needsGeminiAtPeaks`, `overOverflowQuota`,
  `shortOfHeadroom` (for the evaluated fleet; always OK for the optimum), `minPolicyGpus`, `headroomGpus`,
  `optimalGpus`, `optimalTotal`, `spillGeminiTarget` (tier beyond the MaaS cap), `geminiAllowedAtPeaks`, and with
  the default policy `peakGeminiHint` `{target, name, gpus, total, saving, savingShare, vsBaseline, text}` or null.
  `hintText(name, gpus, saving, chips)` formats the hint line.

## Benchmark jobs (jobs.py + cli.py + server.py)

Benchmarks run out of process and keep their state on disk, so a server restart neither loses nor kills them.

- **Runner.** `POST /api/sessions/{name}/run` (`{"config"?, "mode": "auto"|<step>, "steps"?, "keep_endpoint",
  "concurrency", "requests_per_level"}`) launches `python -m aiops_dilemma.cli run-job -c runs/<s>/workload.yaml
  --run-dir runs/<s> --mode … --source app --steps … [--keep] [--concurrency …] [--requests N]` detached
  (`start_new_session=True`, output appended to `runs/<s>/job.log`) and returns once the runner owns `job.json`:
  `{"ok": true, "job": {"status": "running", "steps": [...], "pid": int}}`. 409 when `job.json` says running
  and the pid is alive; 400 for unknown steps; 500 with the log tail if the process exits at once.
  `mode: "auto"` = plan, prompts, deploy, load, analyze, calibrate (skipped unless the workload has prompts over
  32K), report, teardown; `keep_endpoint` only skips the teardown in auto mode. If the job deployed (or was asked to
  deploy) and was meant to tear down, a failure or cancel still tears the endpoint down.
  The CLI's `deploy`, `load`, `calibrate`, `teardown`, `auto` and `run-job` go through the same runner
  (`jobs.run_job`) and write the same files (`source: "cli"`); one live job per session (a second exits non-zero).
- **`runs/<s>/job.json`** (atomic temp + rename): `status` running | completed | error | cancelled | interrupted,
  `pid`, `pid_start` (/proc start time, guards against pid reuse), `source` app | cli, `command`, `mode`, `steps`,
  `keep_endpoint`, `concurrency`, `requests_per_level`, `started_at`, `finished_at`, `stage` (a step, or `done`),
  `error`, `level_progress` `{concurrency, completed, total}`, `levels_completed` (load summaries), `endpoint`
  (endpoint.json contents once deployed), `deployed_here`, `calibration`, `preflight` (`{status running | passed |
  failed, duration_s, durations {long_output, long_prompt}, message}`), `teardown_failed`, `logs` (last 200).
- **`runs/<s>/job_events.jsonl`** (append-only, truncated when a job starts): `{"ts", "type", …, "message"?}` with
  types `stage`, `log`, `plan_done`, `prompts_done`, `deploy_done` (`endpoint`), `waiting_ready`, `preflight_start` (`target_s`,
  `output_tokens`, `prompt_tokens`, `prompt_note`), `preflight_done` (`passed`, `duration_s`, `durations`, `result`),
  `level_start`,
  `request_done` (≈1/s per level plus the last), `level_done` (`summary`), `analyze_done` (`measured`),
  `calibrate_done` (`calibration`), `report_done`, `teardown_done`, `cancelling`, and one terminal event
  `complete` | `error` (`error`) | `cancelled` | `interrupted`. The terminal event is written before the final
  status, so a reader that sees a terminal status already has it.
- **Liveness.** Whenever the server reads a `running` job whose pid is gone (or a zombie, or reused: start time
  differs) it rewrites it as `interrupted` with an `error` and appends an `interrupted` event. The startup scan does
  this for every `runs/*/job.json`.
- **Session payload `job`**: `status` (idle when there is no job.json), `started_at`, `finished_at`, `stage`, `steps`,
  `mode`, `source`, `command`, `pid`, `error`, `level_progress`, `levels_completed`, `logs` (last 80), `endpoint`,
  `calibration`, `preflight`, `stop_requested_at` (while running, after Stop), `endpoint_warning` (interrupted or teardown failed,
  and `endpoint.json` exists and isn't `attached_existing`), `deploy_warning` (interrupted mid-deploy, nothing
  recorded). `GET /api/sessions` items carry `job_status`, `job_source`, `job_endpoint_warning`.
- **`GET /api/sessions/{name}/events`** (SSE): replays `job_events.jsonl` from byte offset `?since=` (or
  `Last-Event-ID`; each message has `id: <offset>`), follows it while the job runs, ends after the terminal event;
  `{"type": "idle"}` when the session has no job. Keep-alive comments every 15 s.
- **`POST /api/sessions/{name}/stop`** `{"force"?: bool, "teardown"?: bool}` → `{"ok", "signal": "SIGTERM" |
  "SIGKILL" | null, "job"}`. SIGTERM (recorded in `job_stop.json`): the runner cancels at the next safe point
  (between steps, at the next request/level/readiness poll; a deployment in progress finishes first), tears down
  what it deployed, writes `cancelled`. Ctrl-C does the same for CLI runs (twice aborts). `force` SIGKILLs the
  process group; the job then reads as `interrupted`. `teardown: true` additionally runs `deploy.teardown`.
- **Tests and UI checks** set `DILEMMA_TEST_FAKE_BENCH=tests/fake_bench.py` (stubs plan/prompts/deploy/load/
  teardown inside the job process; `FAKE_BENCH_TICK_S`, `FAKE_BENCH_DEPLOY_S`, `FAKE_BENCH_REQS`,
  `FAKE_BENCH_FAIL_AT`) and `DILEMMA_RUNS_DIR` (server sessions directory).

## Round 2 (user feedback): contract additions

Naming: in all **user-facing text** say "Agent Platform" instead of "Vertex AI" (e.g. "Gemini on Agent Platform",
"Agent Platform endpoint", "Agent Platform + reservation, 3-yr"). Keep code identifiers, API/SDK names, tier keys
(`vertex`, `vertex_cud_3y`), URLs and SKU-matching strings unchanged. The lead will run a final rename pass; write
new copy with "Agent Platform" from the start.

### Model catalog (modelinfo.py / recommender.py / catalog.py SEED_MODELS block)
- `/api/catalog` `models`: curated, deployable **text LLMs** only. Drop image/video/audio/vision-only models
  (FLUX, PaliGemma, BLIP, LLaVA, Wan…), base/pretrained checkpoints (`-pt`, no `-it`/`-instruct`/`-chat` when an
  instruct sibling exists), and pre-quantized duplicates (fold `-fp8`, `-awq`, `-gptq`, `-w8a8`, `-nvfp4` into the
  base model's `precisions`). Each model: `{id, name, family, params_b, active_params_b, max_context,
  generation (sortable number/str), released (YYYY-MM if known), legacy: bool, specialty: null|"code"|"medical"|
  "translation"|…, precisions: ["bf16","fp8",…], prequantized: {"fp8": model_garden_id, …}, verified_in_model_garden,
  model_garden_id, hf_id, supports_thinking, supports_json, summary}`. Older generations (Gemma 1/2, Llama 2,
  Vicuna, Falcon, CodeLlama, Mistral 7B v0.1…) are kept but `legacy: true`. Response adds `families: [..]` with counts.
- Precision is a separate choice: `model.precision` (`bf16` | `fp8` | `int4`; `awq` / `gptq` / `w4a16` / `4bit`
  normalize to `int4`). FP8 = vLLM on-the-fly quantization unless the model has a `prequantized.fp8` Model Garden id
  (the planner/deployer uses it then). INT4 exists only with a real 4-bit checkpoint: `native_precision: "int4"`
  (published in INT4, e.g. Kimi K2 Thinking / K2.5 / K2.6 / K2.7-Code, compressed-tensors W4A16) or a `prequantized`
  `awq` / `gptq` / `w4a16` / `int4` checkpoint; NVFP4 is not INT4. Sized at 0.6 bytes/param (0.5 + 20% for scales and
  16-bit layers), GPUs only, deployed from that checkpoint without `--quantization` (vLLM reads its config), estimates
  labelled rough (decode bytes 0.5/param, prefill as BF16). `native_precision: "fp8"` (Kimi K2 Instruct) =
  `fp8_only`. `precisions` lists only what exists: `["int4"]` for native INT4, `["fp8"]` for FP8-only, else
  `["bf16","fp8"]` + `"int4"` when a 4-bit checkpoint exists. The main-page toggle shows BF16 / FP8 / INT4 / MXFP4
  buttons only for those.
- **MXFP4** (`model.precision: "mxfp4"`, aliases `mx-fp4` / `mx_fp4`): only for models whose live HF
  `quantization_config.quant_method` is `mxfp4` (`native_precision: "mxfp4"`, gpt-oss; rule fallback in
  `_FAMILY_RULES`) or with a `prequantized.mxfp4` checkpoint. Natively-MXFP4 models: `precisions: ["mxfp4"]` (plus
  `"bf16"` only if a BF16 checkpoint is listed). Catalog entries carry `mxfp4_weight_gb` + `mxfp4_weight_note` (HF
  per-dtype safetensors counts: U8 expert params × 4.25 bits + BF16 × 2; gpt-oss-120b 65.25 GB, 20b 13.76 GB);
  without them 0.55 byte/param and the precision option's `size_text` ends "(estimated)" (`weights_source`
  `hf` | `estimate`, `weights_note` as tooltip). GPUs only; plan candidates at MXFP4 carry `precision_support`
  `verified` (live Model Garden recipe for this model on that accelerator) | `documented` (vLLM gpt-oss recipe:
  H100, H200, B200, A100 via Triton attention + Marlin, `catalog.MXFP4_VLLM_DOCUMENTED`) | `recipe_seen` (no live
  recipes: `catalog.MXFP4_RECIPES_SEEN`, RTX PRO 6000 2026-11-09) | `untested` (ranked last), with
  `precision_support_label` / `precision_support_note` (hardware drawer sub-line, warning style when untested).
  Deployed with no `--quantization` (`deployment_model.quantization: "native"`); estimates note "MXFP4: rough"
  (decode bytes/param = real weight bytes / params, prefill as BF16). Recommender: `recommended_precision: "mxfp4"`,
  `needs_quality_check: false` for gpt-oss.
- "Optimize for" (`use_case.priority`) is removed from scoring and the UI; the tool always optimizes for cost. Old
  configs with `priority` still load (ignored).

### Cost model (costmodel.py + assets/model.js, parity required)
- Baseline is honest about the MaaS rate limit. `scenario()` returns `maas_only_feasible` / `maasOnlyFeasible`
  (peak requests/s ≤ MaaS limit), `maas_peak_rps` / `maasPeakRps`, `maas_cap_rps` / `maasCapRps`, and a
  `baseline` object `{kind: "maas_only"|"maas_plus_overflow", total, maas_cost, overflow_cost, overflow_target,
  label}`: when MaaS can't serve the peak, the baseline is "open-model MaaS up to its limit + overflow to
  <failover target>" priced per token. `vs_baseline` / `vsBaseline` = total / baseline.total. The self-hosted
  `total` always includes every overflow cost (MaaS within the limit and Gemini beyond it).
- Overflow / failover / long-prompt targets: `open_maas`, a role alias (`gemini_flash_lite`, `gemini_flash`,
  `gemini_pro`, `gemini_4` = newest generation's flagship) or any live Gemini model id (`gemini-4-hydrogen`). Both
  engines price any key of `DATA.gemini`; a model id that isn't there is priced as its role alias
  (`costmodel.gemini_key` / `M.geminiKey`, `gemini.alias_for_id` / `M.aliasForId`). Price overrides in model.js:
  `p.gemOverrides = {id: {in, out}}` (per model; an alias follows the override of the model it resolves to) and the
  older per-role `p.liteIn/liteOut`, `p.flashIn/flashOut`, `p.proIn/proOut`, `p.g4In/g4Out` (also applied to the
  model that alias resolves to). Overriding input keeps the cached and long-tier ratios.
- Gemini long-context tier (both engines, parity-tested): prompts routed to a Gemini model are priced per class
  (32K–128K, > 128K); each class's share above the model's `long_context.threshold` (200K; sizes assumed spread
  evenly from the class floor to 2 × mean − floor: token share for input / cached, request share for output) pays the
  long-tier prices (`costmodel.gemini_long_rate` / `M.geminiLongRate`). With one price it equals the old blended rate.
- Thinking: `requests.avg_output_tokens` = visible answer tokens; `requests.thinking_share` (0–1) = share of
  requests with thinking on; `requests.avg_thinking_tokens` = extra output tokens when thinking is on (default 0).
  Effective output per request = avg_output + thinking_share × avg_thinking (helper
  `config.effective_output_tokens(req)`; JS `M.effectiveOutput(p)` with `p.thinkingShare`, `p.thinkingTokens`).
  Every cost, throughput (output ratio) and MaaS price uses the effective value.
- Traffic: `traffic.monthly_input_tokens` (optional) rescales the weekly profile so its monthly total matches
  (it then overrides `average_input_tps`; the UI shows which one is the anchor). `traffic.custom_hourly` (168
  values, input tok/s, Monday 00:00 UTC first) is a user-drawn or document-inferred week; precedence:
  hourly_csv > custom_hourly > parametric. Off-peak multiplier range is 1–10.

### Workload from a document (ingest.py + server.py)
- `POST /api/ingest/describe` (multipart `file`: .pdf/.txt/.md/.docx, or JSON `{text}`) sends the document to
  Gemini on Agent Platform (google-genai, `vertexai=True`, the auth helper's project, `location="global"`, newest
  Flash model; structured JSON output) and returns
  `{workload: {traffic: {peak_input_tps, average_input_tps, floor_input_tps, busy_hours_utc, weekend_factor,
  monthly_input_tokens, custom_hourly?}, requests: {avg_input_tokens, avg_output_tokens, pct_over_32k,
  pct_over_128k, system_prompt_tokens, thinking_share, avg_thinking_tokens}, maas: {max_rps, input_per_m,
  output_per_m, cached_per_m}, constraints: {latency_budget_s}, use_case: {task_type, description}},
  fields: [{path, value, confidence: "stated"|"derived"|"assumed", evidence: "short quote or reasoning"}],
  notes: [str], model: str}`. Omit fields the document doesn't support. Never persist the document; nothing from
  it goes into tracked files or tests (tests use a synthetic document).

## Google Cloud identity and projects (auth.py + server.py)

Sign-in must work when the app runs on a remote machine (Cloud Workstations): nothing opens a browser server-side.

- `GET /api/auth/status` → `{authenticated, adc_ok, gcloud_token_ok, account (gcloud CLI), credential_type:
  "user"|"service_account"|"compute_metadata"|"external_account"|"impersonated"|null, adc_account (email when
  knowable: tokeninfo for user creds, the SA email, or the metadata server), adc_account_source, quota_project,
  google_application_credentials: bool, adc_source: "GOOGLE_APPLICATION_CREDENTIALS"|"metadata_server"|"gcloud_adc_file"|null,
  gcloud_project (env / gcloud config), app_project (what the app uses), app_project_source: "app"|"default"|null,
  quota_project_differs, project (= app project or the placeholder), project_configured, login_in_progress, error}`.
  Never includes tokens. The UI shows "Signed in as you@example.com", then "Project for this app: X" and, when the
  ADC file's quota project differs, "Your credentials bill API calls to Y by default; the app uses X for its calls."
- **The app's project is the app's own.** Choosing a project sets it in the server's memory (`auth.set_project`) and
  the session config; it never runs `gcloud config set project` or changes the environment (gcloud config is only a
  default when nothing is chosen). Every Google API call uses that project for resource paths **and** as the
  quota / billing project (`auth.credentials(project)` = ADC `with_quota_project`, `auth.auth_headers` adds
  `x-goog-user-project`): Billing Catalog, Model Garden, Cloud Monitoring, google-genai, deploy / teardown, and job
  processes (started with `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_QUOTA_PROJECT` = the session's project). So the quota
  project gcloud stamps into the ADC file at sign-in no longer matters. Listing projects and the API check retry
  without the quota header when that project refuses the call (USER_PROJECT_DENIED, API disabled).
- `POST /api/auth/login/start {mode: "adc"|"gcloud", force?}` → `{login_id, url, mode, resumed}` within ~20 s. Runs
  `gcloud auth application-default login --no-launch-browser` (mode "gcloud": `gcloud auth login --no-launch-browser`)
  with piped stdio and prompts disabled, and returns the accounts.google.com link it prints. One sign-in at a time:
  the same mode resumes the running one, another mode → 409, `force` replaces it. 503 when gcloud isn't installed,
  502 when gcloud exits early, 504 when it prints no link. Abandoned sign-ins are killed after 10 minutes.
- `POST /api/auth/login/finish {login_id, code}` → auth status + `{login_ok, login_message (trimmed gcloud output,
  no URL), login_mode, catalog_refreshing, account_changed, previous_account, project_access: {project, ok, message},
  needs_project}`; 400 for an empty code, 404 for an unknown/expired `login_id`. On success, cached catalogs built
  without credentials (or with fallback Gemini prices) are rebuilt in the background; the UI polls `/api/catalog`
  until `cache.refreshing` is false. When the ADC account changed, per-identity caches (project lists, tokeninfo,
  Model Garden recipes) are cleared, every cached catalog is rebuilt, and the app's project is re-checked against the
  new account's projects; `needs_project: true` makes the UI open the project picker with "The signed-in account
  can't see project X. Choose a project for this app." instead of silently keeping it.
- `POST /api/auth/login/cancel {login_id?}` → `{cancelled: bool}` (kills the gcloud process).
- `POST /api/auth/login {project}` sets the app's project (in memory only; 400 for an invalid ID). `trigger_oauth: true` no longer blocks: it starts the
  flow above and returns status + `{login_started, login_id, url}`.
- `GET /api/projects?q=&current=` → `{projects: [{project_id, name, project_number}], total, current, source:
  "resource_manager"|"gcloud"|"stub", message}`. Resource Manager v3 `projects:search` (ACTIVE only, paged, cap 500)
  with the ADC token, `gcloud projects list` fallback, cached 5 minutes per identity; `q` filters id / name / number.
  Sorted: current first, then by display name, `sys-…` / "Untitled project" last. Not signed in → 401 (UI shows
  the sign-in prompt); permission errors → empty list + `message` (UI still accepts a typed ID).
  `DILEMMA_TEST_PROJECTS_FILE` (tests and UI checks only) serves a JSON list instead.
- `GET /api/projects/{id}/check` → `{project_id, services: [{name, title, enabled: bool|null, enable_url}], message}`
  for `aiplatform.googleapis.com` and `cloudbilling.googleapis.com` (Service Usage API); fails soft (`enabled: null`).
- UI: the header project popover shows the identity, a combobox (`role="combobox"` input + `listbox`, type-ahead over
  id + name, arrow keys / Enter / Escape, "Use '<typed id>'" for unlisted projects; selecting applies it at once), the
  API check note with a console link, and "Sign in" / "Use a different account". The sign-in sheet has two steps: open
  the Google link in a new tab, then paste the code and press "Finish sign-in".

## Serving image and pre-flight check (2026-11-09)

- **Config.** `deployment.image`: null = the Model Garden verified recipe's image (default); otherwise the image URI
  (the verified recipe's args are kept, only the image is swapped; `deploy.valid_image_ref` rejects anything that
  isn't an Artifact Registry / Container Registry reference before anything is created). `loadtest.preflight`
  (default true) runs the pre-flight check after the endpoint is ready, before the first level (untagged runs only).
- **`GET /api/catalog`** adds `serving_images`: `[{key: "pinned", image, label, tested (date or null), tested_on,
  notes}]` (`catalog.SERVING_IMAGES`, never cached, so an edit shows on the next load).
- **Pre-flight** (`loadtest.preflight`): (a) one streamed request with `max_tokens = min_tokens = ceil(target_s ×
  150 tok/s)` (capped by `model.max_model_len`), `ignore_eos: true`, where `target_s` = latency budget / 2 clamped
  to 45–90 s; if it ends in under 40 s it is retried once with up to 4× the tokens, and a still-short run passes
  with an "inconclusive" note; (b) one long prompt with a 16-token answer: ~100K tokens when the workload has
  prompts over 32K, else the largest prompt in `prompts.jsonl` (or 2× the average, ≤ 32K); skipped when it
  exceeds `model.max_model_len` or the server rejects it as too long. Pass = both end with `finish_reason` or
  `[DONE]`. A cut stream, timeout or non-200 raises `SystemExit("Pre-flight failed: … after 31 s (4,906 tokens).
  Agent Platform endpoints served by <image> cut long requests; choose the pinned image (…, tested <date>) …")`,
  so the job's teardown removes the endpoint. Result: `runs/<s>/preflight.json` (`ran, passed, image, target_s,
  output_tokens, prompt_tokens, checks[], message, duration_s`), copied to `measured.json.preflight` by analyze.
  `dilemma preflight -c CONFIG [--no-wait]` runs it alone against `endpoint.json`.
- **Level mix.** Each level's summary (job `level_done` events, `measured.json.levels[]`) has `mix`:
  `{requests: {short, medium, long}, tokens: {short, medium, long}, thinking, avg_in, avg_out, repeats}` (shares
  0–1, from answered requests); `sample_note` when prompts repeat. The Benchmark table shows a Mix column
  ("97/2/1 · 10% think", details in the tooltip). `load/levels_plan.json` records the planned per-level samples.
- **Provenance.** `endpoint.json` gains `image`, `image_source` (deployment.image | verified recipe | fallback, plus
  "(args adapted from the <n>x <GPU> recipe)" when the machine had no recipe of its own),
  `args` (final container args) next to `inference_timeout_s`; analyze copies them to `measured.json.serving`;
  the report's benchmark section and the app's provenance line / Benchmark tab say "Measured with
  `pytorch-vllm-serve:gemma4`, vLLM args `…`, endpoint inference timeout 600 s" (host/port flags omitted).
- **UI.** Rail → Advanced → Endpoint: "Serving image" select (`#f-image`: verified / pinned "Pinned image (last
  tested <date>)" / custom), custom reveals `#f-imageCustom` (validated on input/change with the same pattern;
  invalid values show `#imageError`, `aria-invalid`, and are not written); hint explains why, and shows the pinned
  test notes when pinned is chosen. Load test: "Pre-flight check before the sweep" checkbox (`loadtest.preflight`).
  Benchmark tab: `#benchServing` line (current choice + Change link that opens the rail and focuses the select,
  plus the last run's serving provenance and pre-flight result); the Load sweep step reads "Pre-flight check…"
  while it runs; a failed pre-flight shows a "Use the pinned image" link in the error meter.

## Traffic import: time series and Cloud Monitoring (2026-11-09)

- **Parsing** (`ingest.parse_timeseries(fname, raw)`): Cloud Monitoring Metrics Explorer "Download CSV" (`label,value`
  rows such as `TimeSeries ID`, `project_id`, `location`, `model_user_id` in any order, then `timestamp,value` rows;
  several series = several value columns or `timestamp,value` column blocks), header-row CSVs (`timestamp,value`,
  `Time,<metric name>`), JSON / JSONL records. Timestamps: JavaScript `Date.toString()` (`Sun Nov 08 2026 21:50:00
  GMT+0200 (…)`), ISO 8601 / RFC 3339, `YYYY-MM-DD HH:MM[:SS]` and `YYYY/MM/DD …` (UTC assumed and said so),
  BigQuery `… UTC`, RFC 2822, epoch s / ms / µs / ns. Per-request logs (irregular timestamps or `input_tokens`-style
  columns) are not series and keep the existing log path; files without timestamps (168-row hourly, 24-row daily)
  keep the existing hourly path. Labels naming a project / account / series id are dropped (never stored or shown).
- **Classification** (`classify_series`): label `type` (input / cache_* → input, output → output), then column /
  series name, metric labels, file name: requests/s (QPS, RPS, requests, invocations, predictions), input tok/s
  (input, prompt), output tok/s (output, completion), total tok/s (token count without a split), or unknown; with
  confidence high / medium / low and a reason. Values whose names say "count"/"total" but not "per second" are
  divided by the step.
- **Hourly week** (`build_traffic_proposal`): each point covers the step before its timestamp; hourly means are
  time-weighted; hours without points are missing (an hour covered < 10 min by 5-min data is dropped). Option
  `gaps_as_zero` fills missing hours between the first and last point with 0. ≥ 7 days → `typical_week`
  (coverage-weighted mean per weekday-hour UTC, unseen slots interpolated); 1–6 days → `weekday_weekend` (missing
  weekend = weekdays × weekend factor); < 24 h → `daily_shape` (the day's hours, others interpolated, weekend ×
  factor) with a warning naming the span. `level_only` (when a current week exists) keeps the current shape scaled
  to the measured traffic over the same weekday-hours.
- **Combining**: input tok/s from an input series; else total − output; else total × configured input share
  (avg_in / (avg_in + effective output), flagged); else requests/s × avg_input_tokens (flagged). Derived values:
  `requests.avg_input_tokens` = Σinput ÷ Σrequests over the hours both cover (`derived_from_config` when it used
  the configured split), `requests.avg_output_tokens` = Σoutput ÷ Σrequests (includes thinking), `output_ratio`.
- **Endpoints**: `POST /api/sessions/{name}/traffic/import` multipart `files` (+ optional `context` JSON
  `{current_profile[168], requests, weekend_factor}`, `mapping`, `option`, `gaps_as_zero`), or JSON
  `{mapping, option, gaps_as_zero, context}` to re-map the last import (stored in `runs/<s>/traffic_import.json`) →
  `{series: [{id, file, name, metric_guess, confidence, reason, role, labels, points_n, first, last, step_s,
  coverage_h}], proposal: {custom_hourly, peak_input_tps, average_input_tps, floor_input_tps, monthly_input_tokens,
  avg_input_tokens?, avg_output_tokens?, output_ratio?, input_derived, basis}, coverage: {hours, days, span_hours,
  first, last, gaps, gap_hours, label}, options: [{key, label, description, custom_hourly, peak/average/floor,
  monthly_input_tokens}], option, derived: [{path, label, value, now, reason, derived_from_config, checked}],
  warnings, source, role_labels, other_files}`; 422 when no file is a time series. Nothing is applied.
  `POST /api/sessions/{name}/traffic/apply {mapping, option, gaps_as_zero, accept: [paths], context?, config?}`
  writes `traffic.custom_hourly` (clears hourly_csv / preset / monthly_input_tokens) and the accepted
  `requests.*` values, on top of `config` (the client's unsaved config) when given → `{ok, applied, option,
  session}`. `POST /api/sessions/{name}/upload` reports time-series files as `detected_type: "timeseries"` in
  `timeseries_files` without applying them; the app then opens the review sheet.
- **Cloud Monitoring** (`monitoring.py`, REST + ADC, needs `roles/monitoring.viewer`): metrics verified in a real
  project: publisher models (`aiplatform.googleapis.com/PublisherModel`: publisher, model_user_id, location) use
  `publisher/online_serving/model_invocation_count` and `publisher/online_serving/token_count` (label `type`:
  input, output, cache_read_input, cache_write_input, cache_write_1h_input; input tok/s = all but output);
  dedicated endpoints (`aiplatform.googleapis.com/Endpoint`: endpoint_id, location) use
  `prediction/online/prediction_count` (+ any non-distribution token metric on that resource). Discovery via
  `metricDescriptors.list`; `timeSeries.list` with 3600 s alignment, ALIGN_RATE (DELTA/CUMULATIVE) or ALIGN_MEAN
  (GAUGE), REDUCE_SUM per source. Missing hours in the window are zero (delta counters skip idle hours).
  `GET /api/monitoring/sources?project=&days=7|14|28` → `{project, days, start, end, sources: [{id
  ("publisher:<pub>/<model>@<loc>" | "endpoint:<id>@<loc>"), kind, name, location, total_requests, active_hours,
  has_tokens}], metrics, message}`; `POST /api/monitoring/import {project, source, days, session, context?}` →
  the import shape above (apply with `/traffic/apply`). Errors: 401 not signed in, 403 permission (names
  roles/monitoring.viewer) or API disabled, 404 no data. `DILEMMA_TEST_MONITORING_STUB=1` (tests / UI checks only)
  serves synthetic sources.
- **UI**: Traffic section: the upload control routes timestamped files to the review sheet (`#tsDlg`): per series
  a mapping select (requests/s, input / output / total tok/s, ignore) with the guess, confidence tag and reason,
  labels, points, step and span; coverage line; warnings; "Gaps inside the data are zero traffic" checkbox
  (files only); a sparkline of the resulting week (teal) over the current week (dashed); the options as radio
  cards ("Or cancel to keep your current traffic"); derived values as checkboxes ("Uses your settings" when they
  depend on the configured split). Changing a mapping or the gap choice re-queries the server; Apply posts the
  client's config and saves. "Import from Cloud Monitoring" (`#monDlg`): project (default current), time range
  7/14/28 days, sources with traffic (name, model/endpoint, location, requests, active hours) → the same sheet.


## Settings: Hugging Face token (2026-11-09)

- `GET /api/settings` → `{hf_token: {set, source: "env"|"file"|"memory"|"none", hint: "hf_…abcd"|null, user, orgs,
  env_var (HF_TOKEN | HUGGING_FACE_HUB_TOKEN when source is env), path (file only), invalid?, error?}}`; never the
  token. A token whose user isn't known yet in this process is checked once with whoami-v2 (cached).
- `POST /api/settings/hf-token {token}` → validates with `https://huggingface.co/api/whoami-v2` (Bearer) and saves
  only if valid: 200 `{ok, user, orgs, token_hint, stored: "file"|"memory", active_source, overridden_by_env,
  refresh: {invalidated, refreshing}, hf_token: {…as GET}}`; 400 `{detail}` for an empty / refused token, 502 when
  Hugging Face can't be reached. `DELETE /api/settings/hf-token` → `{ok, removed, refresh, hf_token}`.
- Storage (`hftoken.py`): `<config dir>/hf_token`, 0600 in a 0700 directory, config dir `DILEMMA_CONFIG_DIR` >
  `$XDG_CONFIG_HOME/aiops-dilemma` > `~/.config/aiops-dilemma`; read-only filesystem → memory only (passed to job
  subprocesses as `HF_TOKEN`). Precedence `HF_TOKEN` > `HUGGING_FACE_HUB_TOKEN` > memory > file. Used by every
  huggingface.co call (`hfmeta._get`, `prompts._tokenizer`, `modelinfo._get_json` / safetensors HEAD). Save / delete
  drop cached 401 / 403 HF records (`hfmeta.invalidate_auth_failures`) and start a background refresh
  (`modelinfo.refresh_after_token_change`; `hfmeta.ON_UPDATE` rebuilds catalogs). `DILEMMA_TEST_HF_WHOAMI_STUB=1`
  (tests / UI checks only): `hf_valid…` tokens are accepted as `test-user` and never sent to huggingface.co.
- UI: a gear (`#btnSettings`, "Settings") left of the theme button opens `#settingsDlg`: "Hugging Face token"
  section (what it's for, link to https://huggingface.co/settings/tokens "a read-only token is enough"), status line
  ("Set from file: hf_…abcd · user <name>" / "Set by the HF_TOKEN environment variable (read-only here): …" (input
  hidden) / "No token set…" / invalid), password input, Save (Replace) / Remove (confirm dialog), inline error with
  `aria-invalid`. The model drawer's data note adds "N gated models (such as Llama) use built-in values. Add a
  Hugging Face token in Settings to load them." (link opens Settings) when models carry the gated config note.

## Bring your own prompts + one uploader per section (2026-11-12)

- **Storage** (`userprompts.py`): `runs/<s>/uploaded_prompts.jsonl` (one record per line: `id`, `messages`,
  `input_tokens` (the file's count, else the model's tokenizer, else chars / 3.5), optional `max_tokens`,
  `output_tokens`, `response_format`, `chat_template_kwargs`, `source_id`) + `uploaded_prompts_meta.json`
  (`uploaded_at`, `count`, `files: [{filename, format, items, prompts, fields, skipped}]`, `dropped: {empty,
  duplicates}`, `tokens: {method: tokenizer|estimate, tokenizer, from_file, counted}`). Config: `prompts.source:
  file`, `prompts.file: uploaded_prompts.jsonl` (relative to the session folder). Copies (`copy_of`) carry both
  files. Limits: 50 MB per upload (413), 20,000 prompts (413), no prompt text (422 naming the columns found), a
  traffic series sent here (422 "… looks like a traffic time series: use Upload in Traffic.").
- `POST /api/sessions/{name}/prompts`: multipart `files` (+ `text_mode`: separator | blocks | lines, + optional
  `hf_id`) or JSON `{text, mode}` → stores (replacing the previous set), writes `prompts.source/file` into
  workload.yaml (rest of the saved config untouched) → `{source, file, summary, prompt_source}`. 403 for example
  sessions (the UI runs the copy-on-save flow first). `GET` → same shape (`summary` null for synthetic).
  `DELETE` → removes both files, `prompts.source: synthetic`, `file: null`.
- `summary`: `{source, count, uploaded_at, files, dropped, tokens, input: {median, p95, max, avg, pct_over_32k,
  pct_over_128k, over_32k, over_128k}, with_system, system_added, custom_system_prompts, answer:
  {default_max_tokens, prompts_using_default, prompts_with_max_tokens, prompts_with_output_tokens, thinking_share,
  thinking_tokens, prompts_thinking, prompts_with_template_kwargs, text}, requests, repeat_factor, preview: [{text
  (≤ 240 chars), tokens, system}] (3), warnings: [{kind: repeat|context|dropped|skipped|estimate, text}], proposal:
  [{path, value, current, same}] (requests.avg_input_tokens, pct_over_32k, pct_over_128k, medium_mean /
  large_mean when present, avg_output_tokens when ≥ 50% of prompts brought an output count), context}`. The UI
  recomputes the repeat warning and the answer-length line from the current (unsaved) settings.
- Session payload gains `user_prompts` (the summary or null) and `prompt_source: {kind: file|synthetic, count,
  uploaded_at, text}` ("your N prompts (uploaded YYYY-MM-DD)" / "synthetic, shaped like your request sizes").
  measured.json records `prompt_source` for the run (Benchmark tab "Last run … Prompts: …", report.md
  "Prompts: …" line); job.json / job payload `prompt_source`; the `prompts_done` event names the set and the
  repeat factor.
- Benchmark: `prompts.from_file` applies `userprompts.apply_defaults` (no `max_tokens`: the file's output count,
  else `requests.avg_output_tokens`; `thinking_share` of the prompts without `chat_template_kwargs` get
  `enable_thinking: true` + `avg_thinking_tokens`, deterministic floor((j+1)·share) > floor(j·share)) and custom
  system prompts. `cli.needs_calibration(cfg, rd)` uses the prompt file (any prompt > 32K) for file sets;
  `prompts.short_only` keeps the file's prompts ≤ 32K by their `input_tokens`. Pre-flight unchanged.
- Generic `POST /api/sessions/{name}/upload` stays: prompt files (and `.txt`/`.md` with `role_hint: prompts`) go
  through the same parser and storage, set the same config, and return `ingested.prompts_upload` +
  `prompts_summary`. `role_hint: traffic` on a request log derives only the traffic (and request sizes), never
  the prompt set or system prompts.
- `POST /api/uploads/classify` (multipart `files`, nothing stored) → `{files: [{filename, kinds ⊆ {description,
  text, timeseries, hourly_traffic, request_timestamps, prompts, system_prompts, token_counts}, prompts,
  system_prompts, timestamps, series, primary: traffic|system|prompts|unknown}]}`.
  `POST /api/sessions/{name}/system-prompts` (multipart, nothing stored) → `{system_prompts (≤ 50, most frequent
  first), count, avg_tokens, token_method, files}`; 422 pointing to Traffic / Prompts for the wrong kind.
- **UI**: the generic "Upload logs, prompts or a description" area and its "Treat files as" select are gone.
  Traffic (`#trafficDrop`): time series → review sheet; request logs / hourly CSVs → the derived week shown with
  "Apply traffic shape", plus offers "also has request sizes" (review list) and "also has prompts" (→ My prompts);
  PDF / DOCX / prose TXT → describe review sheet; prompt or system-prompt text → "use Upload in Prompts / System
  prompt". System prompt (`#sysDrop`): found prompts + average size, "Also set System prompt to N tokens", "Use
  them for benchmarks" (fills the textarea), offers for prompts / timestamps in the same file. Prompts: "Prompts
  for the benchmark" segmented control (`#promptSrcSyn` "Synthetic, shaped like the numbers above" | `#promptSrcMine`
  "My prompts"); My prompts: privacy note, drop zone (`#promptDrop`), text-mode select, "Paste prompts instead"
  (textarea + "Use these prompts"), summary card (`#promptSummary`: count, files and understood fields, median /
  p95 / max / over 32K / over 128K, token method, answer-length line, outputs / max_tokens, own system prompts,
  first 3, warnings, "Update request sizes from my prompts" (checkbox review, Apply), Replace, Remove (back to
  synthetic) with a confirm dialog). Choosing Synthetic with a set uploaded asks to remove it. Benchmark tab:
  "Prompts: your N prompts (uploaded <date>)" / "Prompts: synthetic, shaped like your request sizes" + Change
  (opens the rail at the control). The first-run upload keeps auto-detection (`/upload` with `auto`), which
  stores prompt files through the same path.

