"""Markdown report: benchmark results, fleet sizing, cost by deployment option, Gemini hybrid routing, resilience, sensitivities, verdict."""

from pathlib import Path

from aiops_dilemma import costmodel, pricing
from aiops_dilemma.costmodel import TIERS


def money(x: float) -> str:
    return f"${x / 1e6:.2f}M" if x >= 1e6 else (f"${x:,.0f}" if x < 1000 else f"${x / 1e3:,.0f}K")


def table(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def mix_text(m: dict | None) -> str:
    """'97/2/1, 10% thinking' (+ repeats) from loadtest.level_mix; '-' for older runs that didn't record it."""
    if not m or not m.get("requests"):
        return "-"
    r = m["requests"]
    pc = lambda v: f"{100 * (v or 0):.0f}"  # noqa: E731
    return (f"{pc(r.get('short'))}/{pc(r.get('medium'))}/{pc(r.get('long'))}, {pc(m.get('thinking'))}% thinking"
            + (f", {m['repeats']} repeats" if m.get("repeats") else ""))


def is_measured(measured: dict | None) -> bool:
    """True only for a real load test (levels recorded, not an instant estimate)."""
    return bool(measured and measured.get("levels") and not measured.get("estimated"))


def throughput_source(measured: dict | None) -> str:
    if is_measured(measured):
        return "calibrated" if measured.get("throughput_source") == "calibrated" else "measured"
    src = (measured or {}).get("throughput_source")
    return src if src in ("estimated", "calibrated", "override") else "estimated"


def throughput_label(per_chip: float, source: str, basis: str | None = None) -> str:
    """One-line provenance. Estimates name their tier (planner.reference_throughput): reference / scaled / spec."""
    n = f"{per_chip:,.0f} input tok/s per chip"
    if source == "estimated" and basis in ("reference", "scaled"):
        return (f"Estimated {n} from a reference benchmark (not your workload). Run a benchmark to measure it."
                if basis == "reference" else
                f"Estimated {n}, scaled from a reference benchmark on another model (not your workload). "
                "Run a benchmark to measure it.")
    return {
        "measured": f"Measured {n} (load test on Agent Platform).",
        "calibrated": f"Calibrated {n} (spec estimate scaled by a profile run).",
        "override": f"Assumed {n}, set by you (not measured). Run a benchmark to measure it.",
    }.get(source, f"Assumed {n}, a rough estimate from GPU specs. Run a benchmark to measure it.")


DISCLAIMER = ("Throughput figures here are estimates (from reference benchmarks or GPU specs), not measured on your "
              "workload. Prompt sizes, thinking and output length, long prompts and caching change throughput by "
              "several times (unbudgeted thinking cut throughput about 4x in our runs). For a decision, run a "
              "benchmark with your own prompts and traffic.")


def disclaimer(measured: dict | None) -> str | None:
    """The estimate disclaimer for reports and dashboards, or None when throughput was measured by a load test."""
    return None if is_measured(measured) else DISCLAIMER


def measured_label(measured: dict | None, per_chip: float, source: str) -> str:
    """Label saved with an estimate (e.g. the example's) wins; otherwise the standard wording for the source."""
    if source in ("override", "calibrated") and (measured or {}).get("throughput_label"):
        return measured["throughput_label"]
    return throughput_label(per_chip, source, (measured or {}).get("throughput_basis"))


def chosen_tier(cfg: dict, base: dict) -> tuple[str, bool]:
    """Pricing option for the headline: config `pricing.tier` when set (and priced), else the cheapest total.

    Returns (tier, fixed) where fixed is True when the user chose it.
    """
    want = ((cfg.get("pricing") or {}).get("tier") or "auto")
    if want != "auto" and (base.get(want) or {}).get("total"):
        return want, True
    return min((t for t in base if base[t].get("total")), key=lambda t: base[t]["total"]), False


def verdict_lines(b: dict, tier_label: str, gem_cat: dict, fixed: bool = False) -> list[str]:
    """Verdict against the baseline that respects the MaaS rate limit (costmodel.baseline)."""
    bl = b["baseline"]
    util = f"Average GPU utilization {b['fleet'].util:.0%}."
    lead = "Your pricing option" if fixed else "Cheapest option"
    if b["maas_only_feasible"]:
        out = [f"{lead}: **{tier_label}** — **{money(b['total'])}/month** with {b['gpus']} GPUs "
               f"(+ MaaS overflow), **{b['vs_maas']:.0%} of MaaS-only** ({money(b['maas_only'])}/month at your "
               f"traffic; peak {b['maas_peak_rps']:.1f} req/s fits the {b['maas_cap_rps']:g} req/s limit, planned at "
               f"{b.get('maas_planned_rps', b['maas_cap_rps']):.1f} req/s per hour). {util}\n"]
        if b["vs_baseline"] > 1:
            out.append("Self-hosting costs more than MaaS at this traffic shape. It still makes sense for data-control "
                       "reasons, or if your off-peak load grows (see sensitivity).\n")
        return out
    over_name = (gem_cat.get(bl["overflow_target"]) or {}).get("name", bl["overflow_target"])
    planned = b.get("maas_planned_rps", b["maas_cap_rps"])
    out = [f"MaaS can't serve your peak: limit {b['maas_cap_rps']:g} req/s (planned at {planned:.1f} req/s per hour to "
           f"absorb bursts) vs {b['maas_peak_rps']:.1f} req/s at peak; "
           f"MaaS up to its limit plus {over_name} overflow would cost **{money(bl['total'])}/month** "
           f"({money(bl['maas_cost'])} MaaS + {money(bl['overflow_cost'])} {over_name}).\n",
           f"{lead}: **{tier_label}** — **{money(b['total'])}/month** with {b['gpus']} GPUs "
           f"(+ MaaS overflow within its limit), **{b['vs_baseline']:.0%} of that baseline**. "
           f"For reference, MaaS-only would cost {money(b['maas_only'])}/month if the limit didn't apply "
           f"(self-hosting = {b['vs_maas']:.0%} of that). {util}\n"]
    if b["vs_baseline"] > 1:
        out.append(f"Stay on MaaS up to its limit with {over_name} overflow: self-hosting costs more at this traffic "
                   "shape. It still makes sense for data-control reasons, or if your off-peak load grows "
                   "(see sensitivity).\n")
    return out


def serving_line(measured: dict | None) -> str | None:
    """'Measured with pytorch-vllm-serve:gemma4, vLLM args --a --b, inference timeout 600 s' (None if unknown)."""
    m = measured or {}
    sv = m.get("serving") or {k: (m.get("endpoint") or {}).get(k) for k in ("image", "args", "inference_timeout_s")}
    if not sv.get("image"):
        return None
    flags = [a for a in (sv.get("args") or []) if str(a).startswith("--") and not str(a).startswith(("--host", "--port"))]
    from aiops_dilemma.catalog import SERVING_IMAGES
    from aiops_dilemma.loadtest import short_image
    known = next((x for x in SERVING_IMAGES if x["image"] == sv["image"]), None)
    name = short_image(known) if known else str(sv["image"]).rsplit("/", 1)[-1]
    tuned = sv.get("tuned") or {}
    extra = []
    if sv.get("command"):
        extra.append(f"command `{' '.join(sv['command'])}`")
    if sv.get("env"):
        extra.append("env " + ", ".join(f"`{k}`" for k in sorted(sv["env"])))
    if sv.get("patch"):
        extra.append(f"a Python patch ({sv['patch'].get('bytes', 0):,} bytes, sha256 {str(sv['patch'].get('sha256'))[:12]})")
    return (("Measured with the tuned config "
             f"\"{tuned.get('title') or 'untitled'}\" ({tuned.get('source', 'imported')}), served by `{name}`")
            if tuned else f"Measured with `{name}`") \
        + (f", vLLM args `{' '.join(flags)}`" if flags else "") \
        + "".join(f", {x}" for x in extra) \
        + (f", endpoint inference timeout {sv['inference_timeout_s']} s" if sv.get("inference_timeout_s") else "") \
        + f" (image `{sv.get('image_requested') or sv['image']}`)."


def preflight_line(measured: dict | None) -> str | None:
    pf = (measured or {}).get("preflight")
    if not pf:
        return None
    return ("Pre-flight check " + ("passed" if pf.get("passed") else "FAILED") + f": {pf.get('message', '')}"
            + (f" ({pf['duration_s']:.0f} s)." if pf.get("duration_s") else ""))


def prompt_source_text(measured: dict, cfg: dict) -> str:
    """'your 240 prompts (uploaded 2026-11-12)' or 'synthetic, ...' for the run behind measured.json."""
    ps = measured.get("prompt_source")
    if not ps:
        from aiops_dilemma import userprompts
        ps = userprompts.describe_source(cfg, cfg.get("_run_dir"))
    return ps.get("text") or ps.get("kind") or "synthetic"


def _gemini_keys_for(cfg: dict, gem_cat: dict) -> list[str]:
    """The four role aliases, plus Gemini models the config pins as targets (e.g. gemini-4-hydrogen)."""
    keys = ["gemini_flash_lite", "gemini_flash", "gemini_pro", "gemini_4"]
    shown = {gem_cat[k].get("model_id") for k in keys if k in gem_cat}
    for f in ("route_long_target", "peak_overflow_target", "failover_target"):
        t = str((cfg.get("hybrid") or {}).get(f) or "").lower()
        if t in gem_cat and t not in keys and gem_cat[t].get("model_id", t) not in shown:
            keys.append(t)
            shown.add(gem_cat[t].get("model_id", t))
    return keys


def render(cfg: dict, plan: dict, measured: dict, prices: dict) -> str:
    cfg = costmodel.with_measured_classes(cfg, measured)   # a benchmark's per-class rates, if it has them
    ep = measured["endpoint"]
    per_gpu = measured["input_tps_per_gpu"]
    replica = ep["accelerator_count"]
    tiers = [t for t in TIERS if prices.get(t)]
    gem_cat = costmodel._gemini_catalog(pricing.gemini_prices(cfg.get("project"), cfg.get("region", "us-central1")))
    f_ref = costmodel.ref_mix_factor(cfg, measured)
    sc = lambda tier, **kw: costmodel.scenario(cfg, per_gpu, prices, tier, replica, gemini_catalog=gem_cat,  # noqa: E731
                                               mix_factor_ref=f_ref, **kw)
    base = {t: sc(t) for t in tiers}
    best_t, fixed_tier = chosen_tier(cfg, base)
    b = base[best_t]
    out_ratio, cache = costmodel.request_mix(cfg)
    m = cfg["maas"]
    res_summary = costmodel.resilience_and_hybrid_summary(cfg, per_gpu, prices, best_t, replica, gemini_catalog=gem_cat,
                                                          mix_factor_ref=f_ref)
    b_info = b.get("buckets", {})
    lite_name = gem_cat["gemini_flash_lite"]["name"]
    flash_name = gem_cat["gemini_flash"]["name"]
    src = throughput_source(measured)

    o = [f"# Self-hosting vs MaaS — {cfg['name']}\n",
         *([f"> **Estimated throughput, not measured on your workload.** {disclaimer(measured)}\n"]
           if disclaimer(measured) else []),
         f"Model **{cfg['model']['model_garden_id']}** ({cfg['model'].get('precision', 'bf16')}) on "
         f"**{replica}x {ep['accelerator_type']}** ({ep['machine_type']}), {cfg['region']}. "
         f"MaaS reference: ${m['input_per_m']}/${m['output_per_m']}/${m['cached_per_m']} per 1M input/output/cached tokens, "
         f"limit {m['max_rps']} req/s. Gemini on Agent Platform ("
         + ("Cloud Billing Catalog" if gem_cat["gemini_flash_lite"].get("source") == "billing-catalog" else "reference prices")
         + "): " + ", ".join(f"{g['name']} ${g['input_per_m']:g}/${g['output_per_m']:g}/${g['cached_per_m']:g}"
                              for g in (gem_cat[k] for k in _gemini_keys_for(cfg, gem_cat)))
         + " per 1M input/output/cached tokens.\n",
         f"Throughput: {measured_label(measured, per_gpu, src)}\n",
         *([f"Prompts: {prompt_source_text(measured, cfg)}\n"] if is_measured(measured) else []),
         "## Verdict\n"]
    o += verdict_lines(b, TIERS[best_t], gem_cat, fixed_tier)
    if b.get("peak_gemini_hint"):   # default fleet policy: no Gemini at peaks (same line as the app and dashboard)
        o.append(f"{b['peak_gemini_hint']['text']} (Set `hybrid.peak_overflow_target: "
                 f"{b['peak_gemini_hint']['target']}` to allow it.)\n")
    for rec in res_summary.get("recommendations", []):
        o.append(f"- **{rec['title']}:** {rec['detail']}\n")

    if is_measured(measured):
        o.append("\n## Benchmark (Agent Platform endpoint, 1 replica)\n")
        o.append(table(["Concurrency", "OK / errors", "req/s", "input tok/s", "output tok/s", "avg in", "avg out",
                        "p50 latency", "p95 latency", "mix: requests S/M/L %, thinking"],
                       [[s["concurrency"], f"{s['ok']}/{s['errors']}", f"{s['rps']:.2f}", f"{s['input_tps']:,.0f}",
                         f"{s['output_tps']:,.0f}", f"{s['avg_in']:,.0f}", f"{s['avg_out']:,.0f}", f"{s['lat_p50']:.0f}s",
                         f"{s['lat_p95']:.0f}s", mix_text(s.get("mix"))] for s in measured["levels"]]))
        o.append("\nEvery level gets its own prompt sample with the same mix (S = 32K tokens or less, M = 32K-128K, "
                 "L = over 128K; token shares and repeats in measured.json).\n")
        pk = measured.get("peak") or {}
        mo = measured.get("measured_out_ratio")
        o.append(f"\nPeak within the {cfg['constraints']['latency_budget_s']}s latency budget: "
                 f"**{pk.get('input_tps', per_gpu * replica):,.0f} input tok/s per replica = {per_gpu:,.0f} per GPU**"
                 + (f" (measured output/input {mo:.3f}; your profile {out_ratio:.3f})" if mo else "") + ".\n")
        for line in (serving_line(measured), preflight_line(measured)):
            if line:
                o.append(f"\n{line}\n")
    else:
        o.append("\n## Throughput\n")
        o.append(("Throughput estimated from accelerator specs (not measured). "
                  if src == "estimated" and (measured or {}).get("throughput_basis") in (None, "spec") else "")
                 + measured_label(measured, per_gpu, src)
                 + (f" {measured['throughput_note']}" if (measured or {}).get("throughput_note") else "")
                 + " No load test was run, so there is no concurrency sweep or latency data.\n")
    o.append("\n## Monthly cost by deployment option (today)\n")
    feasible = b["maas_only_feasible"]
    rows = []
    for t in tiers:
        r = base[t]
        if not r.get("total"):
            rows.append([TIERS[t], "infeasible", "", "", ""] + ([""] if feasible else ["", ""])); continue
        vs = [f"{r['vs_maas']:.0%}"] if feasible else [f"{r['vs_baseline']:.0%}", f"{r['vs_maas']:.0%}"]
        rows.append([TIERS[t], f"${prices[t]:.2f}", r["gpus"], money(r["total"])] + vs
                    + [f"{costmodel.breakeven_utilization(cfg, per_gpu, prices[t], f_ref):.0%}"])
    vs_head = ["vs MaaS-only"] if feasible else ["vs baseline", "vs MaaS-only (no limit)"]
    o.append(table(["Option", "$/GPU-hour", "GPUs (always on)", "Monthly (+ overflow)"] + vs_head
                   + ["Break-even utilization"], rows))
    bl = b["baseline"]
    o.append((f"\nMaaS-only: {money(b['maas_only'])}/month for {b['monthly_tokens'] / 1e9:,.0f}B input tokens "
              f"({cache:.0%} cached). " if feasible else
              f"\nBaseline: {bl['label']}, {money(bl['total'])}/month for {b['monthly_tokens'] / 1e9:,.0f}B input "
              f"tokens ({cache:.0%} cached); MaaS-only would be {money(b['maas_only'])}/month if the "
              f"{b['maas_cap_rps']:g} req/s limit didn't apply. ")
             + f"Break-even utilization = how busy a GPU must be to beat MaaS on the same tokens; "
             f"a fleet sized for the peak averages {b['fleet'].util:.0%}.\n")

    # Hybrid Long-Context & Peak Routing Section
    o.append("\n## Hybrid Routing: Diverting Long-Context (>32K) & Peak Traffic to MaaS / Google Gemini on Agent Platform\n")
    if b_info:
        o.append(
            f"Prompt distribution: **Short (≤32K)** = {cfg['requests'].get('pct_short', 100 - cfg['requests'].get('pct_over_32k', 0)):.1f}% of requests "
            f"({b_info['small']['token_share']:.0%} of tokens), **Medium (32K–128K)** = {cfg['requests'].get('pct_medium', 0):.1f}% "
            f"({b_info['medium']['token_share']:.0%} of tokens), **Long (>128K)** = {cfg['requests'].get('pct_long', 0):.1f}% "
            f"({b_info['large']['token_share']:.0%} of tokens). If served on self-hosted hardware, prompts >32K consume "
            f"**{b_info['long_gpu_time_share']:.0%} of accelerator time**.\n"
        )
        if b_info.get("mix_factor", 1.0) > 1.0:
            o.append(
                f"This mix costs **{b_info['mix_factor']:.2f}×** the accelerator time per token of short prompts: "
                f"{b['blended_per_gpu']:,.0f} input tok/s per chip with every prompt on the fleet, "
                f"{per_gpu * f_ref:,.0f} with >32K prompts routed away.\n"
            )
    ro = res_summary["routing_options"]
    r_rows = []
    for label, item in (
        ("All prompts on self-hosted fleet (+ peak overflow)", ro["all_on_fleet"]),
        ("Divert >32K prompts to Open-Model MaaS", ro["route_long_open_maas"]),
        (f"Divert >32K prompts to {lite_name} (Agent Platform)", ro["route_long_gemini_flash_lite"]),
        (f"Divert >32K prompts to {flash_name} (Agent Platform)", ro["route_long_gemini_flash"]),
    ):
        vs_key = "vs_maas" if feasible else "vs_baseline"
        if item and item.get("total"):
            api = item["fleet"].maas_cost + item["fleet"].gemini_cost + item.get("long_maas_cost", 0)
            r_rows.append([label, item["gpus"], money(item["gpu_cost"]), money(api),
                           money(item["total"]), f"{item[vs_key]:.0%}"])
        else:
            r_rows.append([label, "infeasible", "—", "—", item.get("infeasible", "infeasible"), "—"])
    o.append(table(["Routing Strategy", "Self-Hosted Chips", "Hardware / mo", "MaaS + Gemini / mo", "Total / mo",
                    "vs MaaS-only" if feasible else "vs baseline"], r_rows))

    # Reliability & Resilience Section
    o.append("\n## Reliability & Resilience: Always-On Redundant Hardware vs. Google Gemini on Agent Platform Failover\n")
    o.append(
        "Instead of provisioning idle redundant GPUs/TPUs across zones (+50% hardware) or regions (+100% hardware) that bill 24/7, "
        "a **3-Layer Google Cloud architecture** runs a 1.0× single-zone fleet (Layer 1), bursts normal peaks to Open-Model MaaS (Layer 2), "
        f"and fails over to **Google Gemini on Agent Platform (`{lite_name}` or `{flash_name}`)** as a pay-per-use Layer 3 ($0 when idle).\n"
    )
    inc_rows = [
        [x["scenario"], f"{x['tokens_m']:,.1f}M", money(x["gemini_flash_lite"]), money(x["gemini_flash"]), money(x["open_maas_ref"])]
        for x in res_summary["incidents"]
    ]
    o.append(table(["Incident / Overflow Scenario", "Spilled Input Tokens", f"{lite_name} cost", f"{flash_name} cost", "Open MaaS Reference"], inc_rows))
    o.append(
        f"\nCompare against always-on hardware redundancy on **{TIERS[best_t]}**:\n"
        f"- **Zone-redundant (1.5× fleet):** +{res_summary['zone_extra_gpus']} always-on chips = **+{money(res_summary['zone_extra_cost'])}/month** every month.\n"
        f"- **Two-region (2.0× fleet):** +{res_summary['reg_extra_gpus']} always-on chips = **+{money(res_summary['reg_extra_cost'])}/month** every month.\n"
        + (
            f"- **Break-even:** failover to **{lite_name} on Agent Platform** is cheaper at any realistic outage rate "
            f"(zone redundancy would only pay off beyond {costmodel.HOURS_PER_MONTH} peak outage hours a month, "
            f"i.e. never).\n"
            if res_summary.get("breakeven_never")
            else f"- **Break-even:** always-on zone redundancy only becomes cheaper than failing over to "
                 f"**{lite_name} on Agent Platform** above ~{res_summary['breakeven_outage_hours']:,.0f} peak outage "
                 f"hours a month.\n"
            if res_summary.get("zone_extra_cost", 0) > 0
            else "- **Break-even:** zone redundancy adds no chips at this fleet size.\n"
        )
    )

    def sens(title, key, values, fmt):
        rows = []
        for v in values:
            r = sc(best_t, **{key: v})
            ok = bool(r.get("total"))
            rows.append([fmt(v), money(r["maas_only"]),
                         money(r["baseline"]["total"]) + ("" if r["maas_only_feasible"] else " (+ overflow)"),
                         f"{money(r['total'])} ({r['gpus']} GPUs)" if ok else "infeasible",
                         f"{r['vs_maas']:.0%}" if ok else "", f"{r['vs_baseline']:.0%}" if ok else "",
                         f"{r['fleet'].util:.0%}" if r.get("fleet") else ""])
        o.append(f"\n### {title}\n")
        o.append(table(["Scenario", "MaaS-only", "Baseline", f"Fleet ({TIERS[best_t]})", "vs MaaS-only", "vs baseline",
                        "Avg GPU util"], rows))

    o.append("\n## Sensitivity\n")
    sens("Off-peak load grows (busiest hour unchanged)", "off_peak", [1, 2, 4], lambda v: f"off-peak ×{v}")
    sens("Overall growth", "growth", [1, 2, 5, 10], lambda v: f"volume ×{v}")
    sens("MaaS rate limit", "maas_rps", sorted({0, m["max_rps"], 2 * m["max_rps"], 4 * m["max_rps"]}),
         lambda v: f"{v} req/s")
    sens("Throughput error (vs benchmark)" if src == "measured" else "Throughput error (vs assumed)", "tput_mult",
         [0.75, 1.0, 1.25], lambda v: f"{v:.2f}× " + ("measured" if src == "measured" else "assumed"))
    sens("Resilience", "resilience", [1.0, 1.5, 2.0], lambda v: {1.0: "single zone", 1.5: "zone-redundant", 2.0: "two regions"}[v])

    o.append("\n## Planner candidates\n")
    o.append(table(["Accelerator", "Type", "Chips/replica", "Machine", "Verified", "Agent Platform $/chip-h"],
                   [[c["gpu"], c.get("kind", "gpu").upper(), c["accelerator_count"], c["machine_type"],
                     "yes" if c["verified"] else "no",
                     f"${c.get('prices', {}).get('vertex', 0):.2f}"] for c in plan.get("candidates", [])]))
    o.append("\n*Estimates for planning, not a quote. Fleet is fixed (no autoscaling); prices from the Cloud Billing "
             "catalog (Agent Platform + reservation = GCE CUD rates + Agent Platform management fee with a Flexible Savings Plan).*\n")
    return "\n".join(o)


def write(cfg, plan, measured, prices, run_dir: Path) -> Path:
    path = run_dir / "report.md"
    path.write_text(render(cfg, plan, measured, prices))
    return path
