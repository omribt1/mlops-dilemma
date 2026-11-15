"""Deterministic open-source model recommendation engine based on workload, prompt buckets, and constraints."""

import math

from aiops_dilemma import modelinfo, planner

SHORT_MAX = 32768
MEDIUM_MAX = 131072
# Precision rule of thumb: BF16 when the BF16 weights fit on one chip of a common GPU (RTX PRO 6000, 96 GB)
# under the planner's <=75% usable-memory rule; otherwise FP8, which should be quality-checked on your prompts.
PRECISION_REFERENCE_GPU = "NVIDIA_RTX_PRO_6000"


def required_context(requests_cfg: dict) -> tuple[int, str]:
    """Context length needed by the longest prompt bucket in use, and that bucket's name."""
    over32 = float(requests_cfg.get("pct_over_32k") or 0.0)
    over128 = float(requests_cfg.get("pct_over_128k") or 0.0)
    if over128 > 0:
        return max(MEDIUM_MAX + 1, int(float(requests_cfg.get("large_mean") or 188000))), "long"
    if over32 > 0:
        return MEDIUM_MAX, "medium"
    return SHORT_MAX, "short"


INT4_QUALITY_NOTE = "4-bit: check quality on your prompts."


def recommend_precision(params_b: float, prequantized: dict | None = None, native: str | None = None,
                        fp8_only: bool = False, entry: dict | None = None) -> dict:
    """BF16 when it fits one 96 GB GPU, else FP8; the model's own precision when it is published in FP8 / INT4 /
    MXFP4 only; INT4 (a 4-bit checkpoint must exist) only when even FP8 fits no single machine but INT4 does."""
    from aiops_dilemma import modelinfo, planner
    if native == "mxfp4":
        mw = modelinfo.mxfp4_weights(entry, params_b)
        size = f"~{mw['gb']:,.0f} GB of weights" + ("" if mw["source"] == "hf" else ", estimated")
        return {"recommended_precision": "mxfp4", "recommended_precision_label": "MXFP4", "needs_quality_check": False,
                "precision_reason": (f"Published in MXFP4 ({size}: 4-bit experts, attention and embeddings in BF16; "
                                     "no BF16 checkpoint): served as released, nothing to quantize.")}
    bf16_bytes = params_b * 2e9
    has_int4 = native == "int4" or any((prequantized or {}).get(k) for k in modelinfo.INT4_KEYS)
    int4_bytes = params_b * 1e9 * modelinfo.bytes_per_param("int4")
    if native == "int4":
        return {"recommended_precision": "int4", "recommended_precision_label": "INT4", "needs_quality_check": False,
                "precision_reason": (f"Published in INT4 (~{int4_bytes / 1e9:,.0f} GB of weights; no BF16 checkpoint): "
                                     "served as released, nothing to quantize.")}
    fp8_anywhere = any(planner.fits(params_b * 1e9, a, n) or planner.fits_verified(params_b * 1e9, a, n)
                       for a, n in planner._machines())
    if has_int4 and not fp8_anywhere and any(planner.fits(int4_bytes, a, n) for a, n in planner._machines(True)):
        return {"recommended_precision": "int4", "recommended_precision_label": "INT4", "needs_quality_check": True,
                "precision_reason": (f"Even FP8 ({params_b:,.0f} GB) fits no single machine; Model Garden's 4-bit "
                                     f"checkpoint (~{int4_bytes / 1e9:,.0f} GB) does. {INT4_QUALITY_NOTE}")}
    if native == "fp8":
        return {"recommended_precision": "fp8", "recommended_precision_label": "FP8", "needs_quality_check": False,
                "precision_reason": f"Published in FP8 ({params_b:,.0f} GB of weights; no BF16 checkpoint)."}
    if fp8_only:
        return {"recommended_precision": "fp8", "recommended_precision_label": "FP8", "needs_quality_check": True,
                "precision_reason": (f"Model Garden has this model as an FP8 checkpoint only ({params_b:,.0f} GB of "
                                     "weights). FP8: check quality on your prompts.")}
    if planner.fits(bf16_bytes, PRECISION_REFERENCE_GPU, 1):
        return {"recommended_precision": "bf16", "recommended_precision_label": "BF16", "needs_quality_check": False,
                "precision_reason": f"BF16 weights ({params_b * 2:.0f} GB) fit on one 96 GB GPU; no quantization needed."}
    how = ("Model Garden's pre-quantized FP8 checkpoint is used." if (prequantized or {}).get("fp8")
           else "vLLM quantizes the BF16 weights to FP8 on the fly.")
    return {"recommended_precision": "fp8", "recommended_precision_label": "FP8", "needs_quality_check": True,
            "precision_reason": (f"BF16 weights ({params_b * 2:.0f} GB) don't fit on one 96 GB GPU; FP8 halves them. "
                                 f"{how} FP8: check quality on your prompts.")}


# Smallest model (total parameters, billions) we'd trust per task before it costs quality points: the score favours
# the cheapest model to serve among those that meet the task, context and quality needs.
QUALITY_FLOOR_B = {"reasoning": 20.0, "coding": 20.0, "chat": 7.0}
DEFAULT_QUALITY_FLOOR_B = 12.0
BASE_SCORE = 30
SPECIALTY_TASK = {"code": "coding"}   # specialty -> the task_type that unhides it (default: same name)


def serve_cost_b(params_b: float, active_b: float) -> float:
    """Billions of parameters effectively read per token (MoE: active experts + shared layers), as in the planner."""
    return max(1.0, active_b * 0.85 + params_b * 0.15)


def visible(m: dict, task: str, selected: bool) -> bool:
    """Legacy and specialist models are hidden unless selected, or (specialists) the task matches the specialty."""
    if selected:
        return True
    if m.get("legacy"):
        return False
    spec = m.get("specialty")
    return not spec or SPECIALTY_TASK.get(spec, spec) == task


def recommend(
    use_case: dict,
    requests_cfg: dict,
    traffic_cfg: dict | None = None,
    constraints_cfg: dict | None = None,
    models: list[dict] | None = None,
    project: str | None = None,
    region: str = "us-central1",
    hybrid_cfg: dict | None = None,
    selected_model: dict | None = None,
) -> dict:
    """Score and rank deployable open models. Returns {"recommendations": [...], "tips": [...]}.

    `selected_model` (the session's model) is marked `is_selected`; if it isn't in the catalog it is added as a
    candidate so the UI can always show what is actually being planned.
    """
    if models is None:
        models = modelinfo.discover_models(project, region).get("models", [])
    models = [dict(m) for m in models]
    use_case = use_case or {}
    requests_cfg = requests_cfg or {}
    traffic_cfg = traffic_cfg or {}
    constraints_cfg = constraints_cfg or {}
    hybrid_cfg = hybrid_cfg or {}

    sel = selected_model or {}
    sel_ids = {str(x).lower() for x in (sel.get("model_garden_id"), sel.get("hf_id")) if x}

    def is_sel(m: dict) -> bool:
        ids = {str(m.get("model_garden_id", "")).lower(), str(m.get("hf_id", "")).lower(),
               *(str(a).lower() for a in m.get("aliases") or [])}
        return bool(sel_ids) and bool(ids & sel_ids)

    if sel_ids and not any(is_sel(m) for m in models):
        models.append(selected_candidate(sel))

    task = (use_case.get("task_type") or "extraction").lower()
    # use_case.priority ("Optimize for") is ignored: the tool always optimizes for cost.
    need_json = bool(use_case.get("json_output", True))
    thinking_share = float(requests_cfg.get("thinking_share") or use_case.get("thinking_share") or 0.0)
    pct_over_32k = float(requests_cfg.get("pct_over_32k") or 0.0)
    pct_over_128k = float(requests_cfg.get("pct_over_128k") or 0.0)
    latency_budget = float(constraints_cfg.get("latency_budget_s") or 600)
    route_long = bool(hybrid_cfg.get("route_long", False))
    need_ctx, longest_bucket = required_context(requests_cfg)
    floor_b = QUALITY_FLOOR_B.get(task, DEFAULT_QUALITY_FLOOR_B)

    models = [m for m in models if visible(m, task, is_sel(m))]
    newest_gen: dict[str, float] = {}
    for m in models:
        if m.get("generation") is not None:
            f = m.get("family") or ""
            newest_gen[f] = max(newest_gen.get(f, 0.0), float(m["generation"]))

    tips = []
    if pct_over_32k >= 0.5:
        tips.append(
            f"{pct_over_32k:g}% of your prompts are over 32K tokens. Diverting them to Gemini on Agent Platform usually "
            "shrinks the self-hosted fleet a lot, because long prompts take a disproportionate share of GPU/TPU time."
        )

    ranked = []
    for m in models:
        bd: dict[str, int] = {"base": BASE_SCORE}
        reasons: list[str] = []
        warnings: list[str] = []

        def add(component: str, pts: float):
            bd[component] = bd.get(component, 0) + int(round(pts))

        max_ctx = int(m.get("max_context") or 131072)
        params_b = float(m.get("params_b") or 27.0)
        active_b = float(m.get("active_params_b") or params_b)
        is_moe = active_b < params_b * 0.6
        tasks = [t.lower() for t in (m.get("tasks") or [])]

        # 1. Context window vs the longest prompt bucket in use
        fits_longest = max_ctx >= need_ctx
        blocking = False
        if fits_longest:
            if max_ctx >= 200000 and (pct_over_32k > 0 or pct_over_128k > 0):
                add("context", 14)
                reasons.append(f"{max_ctx // 1024}K context window serves your >32K / >128K prompts natively.")
            else:
                add("context", 8)
        else:
            limit = "128K" if longest_bucket == "long" else "32K"
            if route_long:
                add("context", 0)
                warnings.append(f"Prompts over {limit} must stay routed to Gemini or MaaS: this model's context is "
                                f"{max_ctx // 1024}K.")
            else:
                blocking = True
                add("context", -35)
                warnings.append(f"Can't serve prompts over {limit} unless they are routed elsewhere")

        # 2. Task & structured output fit
        if task in tasks:
            add("task_fit", 12)
            reasons.append(f"Strong fit for {task} workloads.")
        if need_json and m.get("supports_json", True):
            add("json_output", 4)
        if thinking_share > 0:
            if m.get("supports_thinking"):
                add("thinking", 12)
                reasons.append(f"Supports native thinking tokens ({thinking_share * 100:.0f}% thinking share configured).")
            else:
                add("thinking", -8)
                warnings.append("No native thinking mode; uses standard instruction decoding.")

        # 3. Cost to serve: fewer parameters read per token = more tokens per accelerator dollar
        eff_b = serve_cost_b(params_b, active_b)
        add("cost", max(-18.0, min(18.0, 18.0 - 9.0 * math.log2(eff_b / 6.0))))
        if is_moe:
            reasons.append(f"MoE ({active_b:g}B active of {params_b:g}B): cheap to serve per token.")
        elif eff_b <= 9:
            reasons.append(f"Small {params_b:g}B model: among the cheapest to serve.")
        # FP8, or INT4 for a model published in INT4 (it has no FP8 to speak of)
        small = m.get("native_precision") if m.get("native_precision") in ("int4", "mxfp4") else "fp8"
        if not planner.fits(modelinfo.weight_bytes_at(m, small, params_b), PRECISION_REFERENCE_GPU, 1):
            add("cost", -6)
            warnings.append(f"Needs several GPUs even at {planner.PREC_LABEL[small]} ({params_b:g}B parameters).")

        # 4. Quality needs: big enough for the task, and a current generation (better quality per dollar)
        meets_quality = params_b >= floor_b / 2   # far below the floor: ranked after every model that meets it
        if params_b < floor_b:
            add("quality", -30 if not meets_quality else -12)
            warnings.append(f"Small for {task} ({params_b:g}B): check quality on your prompts.")
        gen, fam = m.get("generation"), m.get("family") or ""
        rel = str(m.get("released") or "")
        if (gen is not None and float(gen) >= newest_gen.get(fam, 0.0)) or rel >= "2025-07":
            add("quality", 6)
        elif rel >= "2025-01":
            add("quality", 3)

        # 5. Latency
        if latency_budget <= 15 and active_b > 35:
            add("latency", -10)
            warnings.append(f"Tight {latency_budget:g}s p95 latency budget favors fewer active parameters or MoE.")

        bd = {k: v for k, v in bd.items() if v != 0 or k in ("base", "context", "cost")}
        raw = sum(bd.values())
        score = max(10, min(99, raw))
        if blocking:
            score = min(score, 49)  # never outrank a model that can serve every prompt
        ranked.append({
            **m,
            "match_score": score,
            "score_breakdown": bd,
            **(prec_rec := recommend_precision(params_b, m.get("prequantized"), m.get("native_precision"),
                                               m.get("fp8_only"), m)),
            "precision_options": planner.precision_options({**m, "params_b": params_b},
                                                           prec_rec["recommended_precision"], region),
            "fits_longest_prompt": fits_longest,
            "required_context": need_ctx,
            "blocking": blocking,
            "meets_quality_floor": meets_quality,
            "reasons": reasons[:3],
            "warnings": warnings,
            "is_selected": is_sel(m),
        })

    ranked.sort(key=lambda x: (x["blocking"], not x["meets_quality_floor"], -x["match_score"], x.get("active_params_b") or 99))
    seen, uniq = set(), []
    for t in tips:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return {"recommendations": ranked, "tips": uniq}


def selected_candidate(model: dict) -> dict:
    """Catalog-shaped entry for a model that isn't in the discovered catalog (e.g. a custom or older selection)."""
    mg = model.get("model_garden_id") or model.get("hf_id") or "custom-model"
    hf = model.get("hf_id") or mg
    try:
        params = float(model.get("params_b") or modelinfo.weight_bytes(model) / 2e9)
    except (Exception, SystemExit):
        params = 27.0
    try:
        active = float(model.get("active_params_b") or modelinfo.active_params_b({**model, "params_b": params}))
    except (Exception, SystemExit):
        active = params
    short = (hf.split("/")[-1] if "/" in hf else mg.split("@")[-1])
    return modelinfo.annotate({
        "id": short.lower(),
        "name": model.get("name") or modelinfo.display_name(short),
        "model_garden_id": mg,
        "hf_id": hf,
        "params_b": params,
        "active_params_b": active,
        "max_context": int(model.get("max_context") or model.get("max_model_len") or 131072),
        "supports_thinking": False,
        "supports_json": True,
        "tasks": [],
        "quality_tier": "frontier" if params >= 30 else ("balanced" if params >= 14 else "compact"),
        "summary": "Your current selection (not in the discovered catalog).",
        "not_in_catalog": True,
    })


def recommend_models(use_case: dict, requests_cfg: dict, traffic_cfg: dict | None = None,
                     constraints_cfg: dict | None = None, models: list[dict] | None = None,
                     project: str | None = None, region: str = "us-central1", **kw) -> list[dict]:
    """Backward-compatible list-only form of recommend()."""
    return recommend(use_case, requests_cfg, traffic_cfg, constraints_cfg, models, project, region, **kw)["recommendations"]
