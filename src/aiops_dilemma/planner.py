"""Auto mode, step 1: which GPU and TPU configurations can serve this model, and which to benchmark first.

A configuration fits if the weights use at most 75% of usable accelerator memory (90% of VRAM/HBM minus
~2 GB/chip runtime), leaving room for the KV cache. Candidates are ranked by an *estimated* cost
per token — Agent Platform on-demand price / a speed proxy blending dense TFLOPS (input processing) and
memory bandwidth (output generation), scaled by tensor-parallel efficiency — because the cheapest
machine-hour is rarely the cheapest token. Model Garden verified configs are
preferred (their containers are known to work); the benchmark then replaces the estimate.
"""

from aiops_dilemma import hardware, modelinfo, pricing
from aiops_dilemma.catalog import GPUS, MXFP4_RECIPES_SEEN, MXFP4_VLLM_DOCUMENTED, TP_EFF


def shapes(region: str | None) -> dict:
    """Accelerators (CHIP_SPECS + machine shapes) the planner may propose in `region`: live Compute Engine listing
    when cached (hardware.region_shapes), else the built-in catalog.GPUS."""
    return hardware.region_shapes(region)["accelerators"]


def availability(region: str | None) -> dict:
    """Where the region's accelerator list came from, and what it lacks: {source, region, fetched_at, note,
    not_offered: [...], unrated: [...]} (hardware.region_shapes without the per-accelerator rows)."""
    rs = hardware.region_shapes(region)
    return {k: v for k, v in rs.items() if k != "accelerators"}

WEIGHTS_MAX_SHARE = 0.75
# A Model Garden verified recipe for the exact machine overrides the 75% rule (e.g. Kimi K2 Thinking INT4, ~600 GB, on
# 8 x H100 = 640 GB, served with --gpu-memory-utilization=0.95) as long as the weights fit in this share of raw memory.
VERIFIED_MAX_SHARE = 0.95
TIGHT_FIT_NOTE = "Model Garden verified; little room for KV cache (few concurrent requests, shorter context)"


def usable_gb(acc: str, count: int) -> float:
    return count * max(1.0, GPUS[acc]["mem_gb"] * 0.9 - 2.0)


def fits(weight_bytes: float, acc: str, count: int) -> bool:
    return weight_bytes <= WEIGHTS_MAX_SHARE * usable_gb(acc, count) * 1e9


def fits_verified(weight_bytes: float, acc: str, count: int) -> bool:
    """The looser bound for a shape Model Garden has a verified recipe for."""
    return weight_bytes <= VERIFIED_MAX_SHARE * count * GPUS[acc]["mem_gb"] * 1e9


def estimate_tps_per_chip(cfg: dict, acc: str, count: int) -> float:
    """Physics-grounded estimate of input tokens/sec per GPU/TPU chip for short (<=32K) prompts.

    A short-prompt rate (mix factor 1.0): the cost model applies the long-context slowdown of the configured
    mix itself (costmodel.buckets "mix_factor"), so it is not folded in here. This is the raw spec formula; the
    primary estimate is `reference_throughput` (anchored on measured reference benchmarks when there are any).
    """
    return round(max(150.0, _spec_rate(cfg, acc, count)), 0)


def _spec_rate(cfg: dict, acc: str, count: int) -> float:
    """Unrounded spec formula: 1 / (prefill s per input token + output/input ratio x decode s per output token),
    times the tensor-parallel efficiency."""
    g = GPUS[acc]
    m = cfg.get("model", {})
    req = cfg.get("requests", {})
    act_b = modelinfo.active_params_b(m)
    tot_b = modelinfo.weight_bytes(m) / 2e9
    prec = m.get("precision") or "bf16"
    fp8 = prec == "fp8"
    # Effective active parameter bytes read per decode step (MoE reads active experts + shared layers)
    eff_b = max(3.5, act_b * 0.85 + tot_b * 0.15)
    # INT4 (W4A16): decode reads half the bytes of FP8; prefill computes in 16 bits, so no faster than BF16, and no
    # bandwidth bonus for the dequantization work (conservative: rough).
    tflops = g["tflops"] * (1.35 if fp8 else 1.0)
    bw = g["bw_gbs"] * (1.15 if fp8 else 1.0)
    decode_bytes = 1.0 if fp8 else modelinfo.INT4_BYTES_PER_PARAM if prec == "int4" else 2.0
    if prec == "mxfp4":
        # MXFP4 (W4A16 on Hopper / Ampere kernels): decode reads the checkpoint's real bytes per parameter (4.25-bit
        # experts + BF16 attention / embeddings: ~0.56 for gpt-oss); prefill no faster than BF16. Rough.
        try:
            decode_bytes = max(0.3, min(2.0, modelinfo.precision_bytes(m) / max(1.0, tot_b * 1e9)))
        except (Exception, SystemExit):
            decode_bytes = modelinfo.MXFP4_BYTES_PER_PARAM
    tp_eff = TP_EFF[g["nvlink"]].get(count, 0.80)

    avg_in = max(500.0, float(req.get("avg_input_tokens") or 6000))
    if float(req.get("pct_over_32k") or 0) > 0:   # short prompts only: their mean, not the mix's
        from aiops_dilemma.costmodel import buckets
        avg_in = max(500.0, buckets({"requests": req})["small"]["mean"])
    from aiops_dilemma.config import effective_output_tokens
    avg_out = max(20.0, float(effective_output_tokens(req) or 400))   # answer + thinking
    from aiops_dilemma.config import cached_input_tokens
    cache_share = min(0.85, cached_input_tokens({**req, "avg_input_tokens": avg_in}) / avg_in)
    out_ratio = avg_out / avg_in

    # Prefill GPU-sec per input token + decode GPU-sec per output token folded into input tok/s
    prefill_s_per_tok = (2.0 * eff_b * 1e9) / (0.38 * tflops * 1e12) * (1.0 - 0.75 * cache_share)
    decode_s_per_out_tok = (eff_b * decode_bytes * 1e9) / (18.0 * 0.55 * bw * 1e9)
    s_per_in_tok = prefill_s_per_tok + out_ratio * decode_s_per_out_tok
    return (1.0 / max(1e-9, s_per_in_tok)) * tp_eff


# Estimate tiers (sub-labels of the "estimated" throughput source; docs/ui-design-plan.md, provenance contract)
BASIS_LABELS = {"reference": "Reference benchmark", "scaled": "Scaled from a reference benchmark",
                "spec": "Rough spec estimate"}
YOUR_WORKLOAD = "Your workload may differ: run a benchmark with your prompts."


def _reference_cfg(entry: dict) -> dict:
    """The reference workload as a config, at the reference run's measured averages when recorded."""
    from aiops_dilemma.catalog import REFERENCE_WORKLOAD as W
    return {"model": {"model_garden_id": entry["model_garden_id"], "precision": entry["precision"]},
            "requests": {"avg_input_tokens": entry.get("avg_input_tokens") or W["avg_input_tokens"],
                         "avg_output_tokens": entry.get("avg_output_tokens") or W["avg_output_tokens"],
                         "system_prompt_tokens": W["system_prompt_tokens"],
                         "system_prompt_cache_hit": W["system_prompt_cache_hit"], "thinking_share": 0.0,
                         "pct_over_32k": 0.0, "pct_over_128k": 0.0}}


def _workload_only(cfg: dict, model_cfg: dict) -> dict:
    return {**cfg, "model": model_cfg}


MAX_MODEL_SCALE = 3.0   # beyond this formula ratio, a scaled estimate is labelled rough


def reference_throughput(cfg: dict, acc: str, count: int, references: list[dict] | None = None) -> dict:
    """The primary throughput estimate (input tok/s per chip, short prompts) and how it was obtained.

    1. "reference": a reference benchmark on this accelerator x count, model and precision -> its measured rate.
    2. "scaled": a reference benchmark on this accelerator with another model / precision / count -> its rate x
       the spec formula's ratio (this model and count over the reference model and count on the same accelerator).
    3. "spec": no reference on this accelerator -> the spec formula alone (rough: can be off by several times).

    Tiers 1 and 2 are also adjusted from the reference workload to yours with the same formula: the ratio of the
    formula at your workload over the formula at the reference workload, both for your model, changes only the
    workload terms, i.e. the decode term (output/input ratio: answer + thinking tokens per input token, the term
    that matters most: the reference has 500-token answers and no thinking) and the prefill term (prompt size and
    system-prompt cache share). Formula ratios are used, never the formula's absolute level.
    Returns {per_chip, basis, label, note, spec_per_chip, reference (entry or None), model_scale, workload_scale}.
    """
    from aiops_dilemma.catalog import REFERENCE_WORKLOAD, reference_benchmarks
    g = GPUS[acc]
    spec = estimate_tps_per_chip(cfg, acc, count)
    refs = [e for e in (reference_benchmarks() if references is None else references) if e["accelerator_type"] == acc]
    m = cfg.get("model") or {}
    mid, prec = m.get("model_garden_id"), m.get("precision") or "bf16"

    def rank(e):   # same model+precision+count first, then same model+precision, same precision, then newest
        return (e["model_garden_id"] != mid or e["precision"] != prec or e["accelerator_count"] != count,
                e["model_garden_id"] != mid or e["precision"] != prec, e["precision"] != prec,
                -int(str(e.get("measured_at") or "0").replace("-", "") or 0))
    ref = min(refs, key=rank) if refs else None
    out = {"spec_per_chip": spec, "reference": ref, "model_scale": 1.0, "workload_scale": 1.0}
    if not ref:
        out.update(per_chip=spec, basis="spec", label=BASIS_LABELS["spec"],
                   note=(f"No reference benchmark for {g['name']} yet: estimated from its specs (compute and memory "
                         "bandwidth). Real throughput can differ by several times. " + YOUR_WORKLOAD + _prec_note(prec, m)))
        return out
    ref_cfg = _reference_cfg(ref)
    # workload: your workload vs the reference workload, both on your model and your count
    work = _spec_rate(cfg, acc, count) / max(1e-9, _spec_rate(_workload_only(ref_cfg, m), acc, count))
    exact = ref["model_garden_id"] == mid and ref["precision"] == prec and ref["accelerator_count"] == count
    # model (and count / precision): your model vs the reference model, both on the reference workload
    model = 1.0 if exact else (_spec_rate(_workload_only(ref_cfg, m), acc, count)
                               / max(1e-9, _spec_rate(ref_cfg, acc, ref["accelerator_count"])))
    per = round(max(150.0, ref["short_prompt_tps_per_chip"] * model * work), 0)
    where = f"{ref['accelerator_count']}x {g['name']}"
    adj = (f" Adjusted to your output length and prompt size: x{work:.2f}." if abs(work - 1) > 0.03 else "")
    if exact:
        note = (f"Measured on {where} with the reference workload ({REFERENCE_WORKLOAD['summary']}) on "
                f"{ref['measured_at']}.{adj} {YOUR_WORKLOAD}")
    else:
        what = ("another model" if ref["model_garden_id"] != mid else "another precision"
                if ref["precision"] != prec else "another GPU count")
        note = (f"Scaled from a reference benchmark on {where} with {what} ({ref['model_garden_id'].split('@')[-1]} "
                f"{ref['precision'].upper()}, reference workload: {REFERENCE_WORKLOAD['summary']}, {ref['measured_at']}) "
                f"by the spec formula's ratio for this model and configuration (x{model:.2f}).{adj} {YOUR_WORKLOAD}")
    basis = "reference" if exact else "scaled"
    if not exact and not (1 / MAX_MODEL_SCALE <= model <= MAX_MODEL_SCALE):
        # Too different from the benchmarked model (e.g. dense vs mixture-of-experts, much larger or smaller) for the
        # formula's ratio to be trusted: keep the anchored number but call it what it is.
        basis = "spec"
        note = (f"Rough: this model is too different from the benchmarked one on {where} to scale reliably (spec "
                f"formula ratio x{model:.2f}). Treat this as an order of magnitude.{adj} {YOUR_WORKLOAD}")
    out.update(per_chip=per, basis=basis, label=BASIS_LABELS[basis], note=note + _prec_note(prec, m),
               model_scale=round(model, 4), workload_scale=round(work, 4))
    return out


INT4_NOTE = " INT4: rough; decode assumed to read half the bytes of FP8, input processing no faster than BF16."
INT4_QUALITY = " 4-bit weights can change answers: check quality on your prompts."


MXFP4_NOTE = (" MXFP4: rough; decode assumed to read the checkpoint's bytes per parameter (4.25-bit experts, the rest in"
              " BF16), input processing no faster than BF16.")


def _prec_note(prec: str | None, model: dict | None = None) -> str:
    if prec == "mxfp4":
        return MXFP4_NOTE   # gpt-oss is released in MXFP4: nothing is quantized, no quality caveat
    if prec != "int4":
        return ""
    native = (model or {}).get("native_precision") or (modelinfo.find_model(model or {}) or {}).get("native_precision")
    return INT4_NOTE + ("" if native == "int4" else INT4_QUALITY)   # a model published in INT4 is the release itself


def _estimate_fields(cfg: dict, acc: str, count: int) -> dict:
    """Candidate fields for the primary estimate (reference_throughput)."""
    r = reference_throughput(cfg, acc, count)
    return {"est_input_tps_per_gpu": r["per_chip"], "est_replica_input_tps": r["per_chip"] * count,
            "est_basis": r["basis"], "est_basis_label": r["label"], "est_basis_note": r["note"],
            "spec_input_tps_per_gpu": r["spec_per_chip"]}


def deployment_model(cfg: dict) -> dict:
    """What actually gets deployed for the configured model and precision.

    Precision is a choice, not a separate model: FP8 is vLLM on-the-fly quantization (`--quantization=fp8`) of the
    BF16 checkpoint, unless Model Garden has a pre-quantized FP8 checkpoint of this model (catalog
    `prequantized.fp8`, or `model.prequantized.fp8` in the config): then that checkpoint is deployed as-is.
    INT4 always deploys a 4-bit checkpoint (modelinfo.int4_checkpoint): the model itself when it is published in
    INT4 (quantization "native", e.g. Kimi K2 Thinking), else a pre-quantized AWQ / GPTQ / W4A16 one; vLLM reads
    its quantization config, so no --quantization flag. MXFP4 likewise deploys the MXFP4 checkpoint (gpt-oss is
    published in it: "native"). Weights are sized by modelinfo.precision_bytes.
    """
    m = cfg.get("model") or {}
    prec = m.get("precision") or "bf16"
    out = {"model_garden_id": m.get("model_garden_id"), "hf_id": m.get("hf_id"), "precision": prec,
           "quantization": None}
    if prec not in ("fp8", "int4", "mxfp4"):
        return out
    entry = None
    if prec == "fp8":
        pq = (m.get("prequantized") or {}).get("fp8")
        if not pq:
            entry = modelinfo.find_model(m) or {}
            pq = (entry.get("prequantized") or {}).get("fp8")
            if not pq and entry.get("native_precision") == "fp8":
                out["quantization"] = "native"
                return out
    else:
        entry = {**(modelinfo.find_model(m) or {}), **{k: m[k] for k in ("prequantized", "native_precision") if m.get(k)}}
        pq = modelinfo.int4_checkpoint(entry) if prec == "int4" else modelinfo.mxfp4_checkpoint(entry)
        if not pq:
            return out   # plan() refuses INT4 / MXFP4 without such a checkpoint
        if entry.get("native_precision") == prec:
            out["quantization"] = "native"
            return out
    if pq and pq != m.get("model_garden_id"):
        hf = m.get("hf_id")   # same HF org, pre-quantized repo name (best effort; Model Garden deploys use the MG id)
        hf = f"{hf.split('/')[0]}/{pq.split('@')[-1]}" if hf and "/" in hf and "@" in pq else hf
        out.update(model_garden_id=pq, hf_id=hf, quantization="prequantized", base_model_garden_id=m.get("model_garden_id"))
    else:
        out["quantization"] = "on_the_fly"
    return out


def candidates(cfg: dict, with_prices: bool = True, accelerator_filter: str | None = None) -> list[dict]:
    w = modelinfo.precision_bytes(cfg["model"])
    dep_id = deployment_model(cfg)["model_garden_id"] or cfg["model"]["model_garden_id"]
    verified = modelinfo.verified_options(cfg.get("project"), cfg.get("region", "us-central1"), dep_id)
    vset = {(o["accelerator_type"], o["accelerator_count"]) for o in verified}
    prec = cfg["model"].get("precision") or "bf16"
    acc_filter = (accelerator_filter or cfg.get("deployment", {}).get("accelerator_filter") or "all").lower()
    out = []
    for acc, g in shapes(cfg.get("region", "us-central1")).items():
        kind = g.get("kind", "gpu")
        if acc_filter in ("gpu", "tpu") and kind != acc_filter:
            continue
        if kind == "tpu" and prec in modelinfo.GPU_ONLY_PRECISIONS:
            continue                                               # 4-bit GPU kernels (Marlin etc.): GPUs only
        for count in sorted(g["machines"]):
            tight = False
            if not fits(w, acc, count):
                if (acc, count) not in vset or not fits_verified(w, acc, count):
                    continue
                tight = True                                       # verified recipe overrides the 75% rule
            u_gb = usable_gb(acc, count)
            w_gb = w / 1e9
            est = _estimate_fields(cfg, acc, count)
            c = {
                "accelerator_type": acc,
                "accelerator_count": count,
                "machine_type": g["machines"][count][0],
                "gpu": g["name"],
                "kind": kind,
                "vram_gb": count * g["mem_gb"],
                "usable_gb": round(u_gb, 1),
                "weights_gb": round(w_gb, 1),
                "vram_util": round(w_gb / u_gb, 3),
                "verified": (acc, count) in vset,
                **est,
            }
            if tight:
                c.update(tight_fit=True, fit_note=TIGHT_FIT_NOTE)
            if prec == "mxfp4":
                sup, sup_note = mxfp4_support(acc, {a for a, _n in vset})
                c.update(precision_support=sup, precision_support_note=sup_note,
                         precision_support_label=MXFP4_SUPPORT_LABEL[sup])
            if with_prices:
                p = pricing.prices_for(cfg, acc, count)
                if not p.get("vertex"):
                    c["note"] = "price not found in this region"
                    continue
                c["prices"] = p
                c["price_source"] = p.get("source", "live_billing")
                # Input processing scales with FLOPs, output generation with memory bandwidth.
                speed = count * TP_EFF[g["nvlink"]][count] / (0.5 / g["tflops"] + 0.5 / (g["bw_gbs"] / 4))
                c["est_cost_index"] = p["vertex"] * count / speed   # relative $ per token, lower is better
            out.append(c)
            break                                                  # smallest count that fits, per accelerator type
    # Machines that only come bigger than the model needs (8 x B200 / H200 for a model that fits one GPU) rank after
    # right-sized ones: a replica of 8 large GPUs makes the fleet coarse even when the spec $/token looks good.
    def oversized(c):
        need = min_chips_for_memory(w, c["accelerator_type"])
        return bool(need and c["accelerator_count"] >= 4 * need)
    out.sort(key=lambda c: (not c["verified"], c.get("precision_support") == "untested",
                            c.get("kind") == "tpu" and acc_filter != "tpu", oversized(c), c.get("est_cost_index", 1e9)))
    return out


class NoFitError(SystemExit):
    """The model can't be planned at this precision on any single machine (or the precision isn't available for it).
    `info` (no_fit_explanation) says why and what to do; str() is its message."""

    def __init__(self, info: dict):
        super().__init__(info["message"])
        self.info = info


PREC_LABEL = {"bf16": "BF16", "fp8": "FP8", "int4": "INT4", "mxfp4": "MXFP4"}


MXFP4_SUPPORT_LABEL = {"verified": "MXFP4: Model Garden verified", "documented": "MXFP4 supported by vLLM",
                       "recipe_seen": "MXFP4: Model Garden recipe (not re-checked)",
                       "untested": "MXFP4 untested on this GPU"}


def mxfp4_support(acc: str, verified_accs: set | None = None) -> tuple[str, str]:
    """How well MXFP4 is known to work on an accelerator: "verified" (a live Model Garden recipe for this model on
    it), "documented" (vLLM's gpt-oss recipe names the architecture, catalog.MXFP4_VLLM_DOCUMENTED), "recipe_seen"
    (fallback without live recipes: Model Garden had one on the date in catalog.MXFP4_RECIPES_SEEN) or "untested"."""
    name = (GPUS.get(acc) or {}).get("name", acc)
    if acc in (verified_accs or set()):
        return "verified", f"MXFP4 on {name}: Model Garden has a verified recipe."
    if acc in MXFP4_VLLM_DOCUMENTED:
        return "documented", f"MXFP4 on {name}: supported by vLLM ({MXFP4_VLLM_DOCUMENTED[acc]})."
    if acc in MXFP4_RECIPES_SEEN and not verified_accs:
        return "recipe_seen", (f"MXFP4 on {name}: Model Garden had a verified gpt-oss recipe on "
                               f"{MXFP4_RECIPES_SEEN[acc]} (built-in fallback: live recipes not read; sign in and "
                               "choose a project to check).")
    return "untested", (f"MXFP4 on {name}: untested (no Model Garden recipe, not in vLLM's documented GPUs); it may "
                        "not load or may run slower kernels.")


def _machines(gpu_only: bool = False, acc_filter: str = "all", region: str | None = None):
    """(accelerator, count) of every machine shape (the region's live shapes when known, else the built-in ones)."""
    for acc, g in (shapes(region) if region else GPUS).items():
        kind = g.get("kind", "gpu")
        if (gpu_only and kind != "gpu") or (acc_filter in ("gpu", "tpu") and kind != acc_filter):
            continue
        for count in sorted(g["machines"]):
            yield acc, count


def largest_machine(gpu_only: bool = False, acc_filter: str = "all", region: str | None = None) -> tuple[str, int]:
    """The single machine with the most usable accelerator memory (8 x B200 in the catalog)."""
    return max(_machines(gpu_only, acc_filter, region), key=lambda ac: usable_gb(*ac))


def available_precisions(entry: dict | None) -> list[str]:
    """Precisions the app offers for a catalog entry: INT4 only with a 4-bit checkpoint (modelinfo.int4_checkpoint)."""
    entry = entry or {}
    if entry.get("native_precision") == "int4":
        return ["int4"]
    if entry.get("native_precision") == "mxfp4":   # gpt-oss: MXFP4 only, unless a BF16 checkpoint really exists
        return [p for p in (entry.get("precisions") or ["mxfp4"]) if p in ("bf16", "mxfp4")] or ["mxfp4"]
    base = list(entry.get("precisions") or (["fp8"] if entry.get("fp8_only") else ["bf16", "fp8"]))
    out = [p for p in base if p in ("bf16", "fp8", "int4", "mxfp4")]
    if modelinfo.int4_checkpoint(entry) and "int4" not in out:
        out.append("int4")
    if modelinfo.mxfp4_checkpoint(entry) and "mxfp4" not in out:
        out.append("mxfp4")
    return out


QUALITY_NOTE = {"fp8": "FP8: check quality on your prompts.", "int4": "4-bit: check quality on your prompts.",
                "mxfp4": "4-bit: check quality on your prompts."}
HOW_LABEL = {"native": "native", "prequantized": "prequantized checkpoint", "on_the_fly": "quantized on the fly by vLLM"}


def _size_text(gb: float) -> str:
    return f"{gb / 1000:.1f} TB" if gb >= 1000 else f"{gb:,.0f} GB"


def precision_options(entry: dict, recommended: str | None = None, region: str | None = None) -> list[dict]:
    """One item per precision that really exists for a catalog entry (available_precisions): how it is obtained
    (native checkpoint, prequantized checkpoint, quantized on the fly by vLLM), weight size, the smallest single
    machine it fits, and a quality note on the recommended item only when quantization is involved."""
    params_b = float(entry.get("params_b") or 0)
    native = entry.get("native_precision")
    pq = entry.get("prequantized") or {}
    out = []
    for p in available_precisions(entry):
        if p == "bf16":
            how = "native"
        elif p == native:
            how = "native"
        elif p == "fp8":
            how = "prequantized" if (pq.get("fp8") or entry.get("fp8_only")) else "on_the_fly"
        else:
            how = "prequantized"
        gb = modelinfo.weight_bytes_at(entry, p, params_b) / 1e9
        gpu_only = p in modelinfo.GPU_ONLY_PRECISIONS
        where = _where(gb * 1e9, gpu_only, "all", region) if params_b else None
        if p == "mxfp4" and params_b:   # name a GPU vLLM documents MXFP4 on, not an untested one (L4)
            where = (_where(gb * 1e9, True, "all", region, only=set(MXFP4_VLLM_DOCUMENTED))
                     or (where and where + " (untested with MXFP4)"))
        tight = None
        if params_b and not where:
            t = [(a, n) for a, n in _machines(gpu_only, region=region) if fits_verified(gb * 1e9, a, n)]
            tight = f"{t[0][1]} × {GPUS[t[0][0]]['name']}" if t else None
        item = {"precision": p, "label": PREC_LABEL[p], "how": how, "how_label": HOW_LABEL[how],
                "weights_gb": round(gb, 1), "size_text": _size_text(gb), "fits_on": where, "fits_tight": tight,
                "recommended": p == recommended,
                "quality_note": QUALITY_NOTE.get(p) if p == recommended and how != "native" else None}
        if p == "mxfp4":
            mw = modelinfo.mxfp4_weights(entry, params_b)
            item.update(weights_source=mw["source"], weights_note=mw["note"])
            if mw["source"] != "hf":
                item["size_text"] = "~" + item["size_text"] + " (estimated)"
        item["fit_text"] = (f"fits {where}" if where else
                            f"only {tight} with a Model Garden verified recipe (little KV-cache room)" if tight else
                            "fits no single machine")
        out.append(item)
    return out


def _where(need_bytes: float, gpu_only: bool, acc_filter: str, region: str | None = None,
           only: set | None = None) -> str | None:
    fits_on = [(acc, n) for acc, n in _machines(gpu_only, acc_filter, region) if fits(need_bytes, acc, n)
               and (only is None or acc in only)]
    if not fits_on:
        return None
    acc, n = min(fits_on, key=lambda ac: (GPUS[ac[0]].get("kind") == "tpu", usable_gb(*ac)))   # GPUs first
    return f"{n} × {GPUS[acc]['name']}"


def no_fit_explanation(cfg: dict, accelerator_filter: str | None = None, reason: str = "too_big") -> dict:
    """Why nothing fits (weights at this precision vs the largest machine) and what to do: switch precision when a
    smaller one exists and fits, a bigger machine, or a smaller model. Multi-host serving isn't supported."""
    m = cfg.get("model") or {}
    prec = m.get("precision") or "bf16"
    acc_filter = (accelerator_filter or (cfg.get("deployment") or {}).get("accelerator_filter") or "all").lower()
    entry = modelinfo.find_model(m) or {}
    entry = {**entry, **{k: m[k] for k in ("prequantized", "native_precision") if m.get(k)}}
    name = entry.get("name") or m.get("name") or str(m.get("model_garden_id") or m.get("hf_id") or "This model").split("@")[-1]
    try:
        params_b = modelinfo.weight_bytes(m) / 2e9
    except (Exception, SystemExit):
        params_b = float(entry.get("params_b") or 0)
    gpu_only = prec in modelinfo.GPU_ONLY_PRECISIONS
    region = cfg.get("region", "us-central1")
    acc, cnt = largest_machine(gpu_only, acc_filter if acc_filter in ("gpu", "tpu") else "all", region)
    g = shapes(region).get(acc) or GPUS[acc]
    limit_gb = WEIGHTS_MAX_SHARE * usable_gb(acc, cnt)
    w_gb = modelinfo.weight_bytes_at(entry, prec, params_b) / 1e9
    largest = {"accelerator_type": acc, "accelerator_count": cnt, "machine_type": g["machines"][cnt][0],
               "gpu": g["name"], "vram_gb": cnt * g["mem_gb"], "weights_limit_gb": round(limit_gb)}
    avail = available_precisions(entry)
    options: list[dict] = []
    for p in ("fp8", "int4", "mxfp4", "bf16"):
        if p == prec or p not in avail:
            continue
        need = modelinfo.weight_bytes_at(entry, p, params_b) / 1e9
        where = _where(need * 1e9, p in modelinfo.GPU_ONLY_PRECISIONS, acc_filter, region)
        if where and (reason != "too_big" or need < w_gb):
            ck = (" (Model Garden's 4-bit checkpoint)" if p == "int4" else
                  " (the model's published checkpoint)" if p == "mxfp4" and entry.get("native_precision") == "mxfp4"
                  else " (the MXFP4 checkpoint)" if p == "mxfp4" else "")
            options.append({"label": f"Switch to {PREC_LABEL[p]}{ck}: about {need:,.0f} GB of weights, which fits "
                                     f"{where}.", "action": {"type": "precision", "value": p}})
        elif need < w_gb:
            options.append({"label": f"{PREC_LABEL[p]} would still need about {need:,.0f} GB: too big as well."})
    if "int4" not in avail and prec != "int4" and reason == "too_big":
        options.append({"label": "Model Garden has no INT4 (4-bit) checkpoint of this model, so INT4 isn't offered."})
    if reason == "too_big":
        for o in modelinfo.verified_options(cfg.get("project"), cfg.get("region", "us-central1"),
                                            deployment_model(cfg)["model_garden_id"] or m.get("model_garden_id") or ""):
            a, n = o.get("accelerator_type"), int(o.get("accelerator_count") or 0)
            if a in GPUS and n and not fits_verified(w_gb * 1e9, a, n):
                util = next((x.split("=", 1)[1] for x in o["args"] if x.startswith("--gpu-memory-utilization=")), None)
                ctx = next((x.split("=", 1)[1] for x in o["args"] if x.startswith("--max-model-len=")), None)
                options.append({"label": (f"Model Garden's recipe serves it on {n} × {GPUS[a]['name']}"
                                          + (f" with {float(util):.0%} of GPU memory for the server" if util else "")
                                          + (f" and a {int(ctx):,}-token context" if ctx else "")
                                          + f", but at {PREC_LABEL.get(prec, prec)} the weights alone would exceed "
                                          f"{VERIFIED_MAX_SHARE:.0%} of that machine's memory.")})
                break
        options.append({"label": "Serving one model across several machines (multi-host) isn't supported by this tool."})
    options.append({"label": "Pick a smaller model.", "action": {"type": "models"}})
    P = PREC_LABEL.get(prec, prec.upper())
    if reason == "native_only":
        message = (f"{name} is published in {PREC_LABEL[avail[0]]} only (no {P} checkpoint), so it can't be planned "
                   f"at {P}.")
    elif reason == "no_int4":
        message = f"{name} has no 4-bit checkpoint in Model Garden, so it can't be planned at INT4."
    elif reason == "no_mxfp4":
        message = (f"{name} has no MXFP4 checkpoint (MXFP4 is offered only for models published in it, such as "
                   "gpt-oss), so it can't be planned at MXFP4.")
    else:
        message = (f"{name} needs about {w_gb:,.0f} GB for its weights at {P}. The largest single machine"
                   f"{' with GPUs' if gpu_only else ''}, {cnt} × {g['name']} ({cnt * g['mem_gb']:,} GB), can hold at "
                   f"most {limit_gb:,.0f} GB of weights while leaving room for the KV cache: models up to about "
                   f"{limit_gb / (w_gb / params_b if params_b and w_gb else modelinfo.bytes_per_param(prec)):,.0f}B "
                   f"parameters at {P}.")
    return {"reason": reason, "message": message, "model": name, "precision": prec, "params_b": round(params_b, 1),
            "weights_gb": round(w_gb, 1), "largest": largest, "available_precisions": avail, "options": options}


def _check_precision(cfg: dict, accelerator_filter: str | None) -> None:
    """Refuse a precision the model has no checkpoint for: INT4 without a 4-bit checkpoint, or BF16 / FP8 for a
    model published in INT4 only."""
    m = cfg.get("model") or {}
    prec = m.get("precision") or "bf16"
    entry = {**(modelinfo.find_model(m) or {}), **{k: m[k] for k in ("prequantized", "native_precision") if m.get(k)}}
    if entry.get("native_precision") in ("int4", "mxfp4") and prec not in available_precisions(entry):
        raise NoFitError(no_fit_explanation(cfg, accelerator_filter, reason="native_only"))
    if prec == "int4" and not modelinfo.int4_checkpoint(entry):
        raise NoFitError(no_fit_explanation(cfg, accelerator_filter, reason="no_int4"))
    if prec == "mxfp4" and not modelinfo.mxfp4_checkpoint(entry):
        raise NoFitError(no_fit_explanation(cfg, accelerator_filter, reason="no_mxfp4"))


def plan(cfg: dict, accelerator_filter: str | None = None) -> dict:
    d = cfg["deployment"]
    _check_precision(cfg, accelerator_filter)
    cands = candidates(cfg, accelerator_filter=accelerator_filter)
    w_gb = modelinfo.precision_bytes(cfg["model"]) / 1e9
    if d.get("mode") == "manual" and d.get("accelerator_type"):
        acc = d["accelerator_type"]
        cnt = int(d.get("accelerator_count") or 1)
        if cfg["model"].get("precision") in modelinfo.GPU_ONLY_PRECISIONS and GPUS.get(acc, {}).get("kind") == "tpu":
            raise SystemExit(f"{PREC_LABEL[cfg['model']['precision']]} checkpoints run on GPUs only: choose a GPU, "
                             "or another precision for TPUs")
        region_accs = shapes(cfg.get("region", "us-central1"))
        g = region_accs.get(acc) or GPUS.get(acc, {})
        mtype = d.get("machine_type") or (g.get("machines", {}).get(cnt, ("custom",))[0])
        u_gb = usable_gb(acc, cnt) if acc in GPUS else cnt * 80
        est = _estimate_fields(cfg, acc, cnt) if acc in GPUS else {
            "est_input_tps_per_gpu": 5000.0, "est_replica_input_tps": 5000.0 * cnt, "est_basis": "spec",
            "est_basis_label": BASIS_LABELS["spec"], "est_basis_note": "Unknown accelerator: a placeholder rate.",
            "spec_input_tps_per_gpu": 5000.0}
        chosen = {
            "accelerator_type": acc,
            "accelerator_count": cnt,
            "machine_type": mtype,
            "gpu": g.get("name", acc),
            "kind": g.get("kind", "gpu"),
            "vram_gb": cnt * g.get("mem_gb", 80),
            "usable_gb": round(u_gb, 1),
            "weights_gb": round(w_gb, 1),
            "vram_util": round(w_gb / max(1.0, u_gb), 3),
            "verified": any(c["accelerator_type"] == acc and c["accelerator_count"] == cnt and c["verified"] for c in cands),
            **est,
        }
        if cfg["model"].get("precision") == "mxfp4" and acc in GPUS:
            vaccs = {o["accelerator_type"] for o in modelinfo.verified_options(
                cfg.get("project"), cfg.get("region", "us-central1"),
                deployment_model(cfg)["model_garden_id"] or cfg["model"].get("model_garden_id") or "")}
            sup, sup_note = mxfp4_support(acc, vaccs)
            chosen.update(precision_support=sup, precision_support_note=sup_note,
                          precision_support_label=MXFP4_SUPPORT_LABEL[sup])
        chosen["prices"] = pricing.prices_for(cfg, acc, cnt) if acc in GPUS else {}
        if acc not in region_accs:
            gone = next((x for x in availability(cfg.get("region", "us-central1")).get("not_offered", [])
                         if x["accelerator_type"] == acc), None)
            if gone:
                chosen["region_note"] = gone["note"]
    else:
        if not cands:
            raise NoFitError(no_fit_explanation(cfg, accelerator_filter))
        chosen = cands[0]
    return {"weights_gb": round(w_gb, 1), "chosen": chosen, "candidates": cands, "deployment_model": deployment_model(cfg),
            "availability": availability(cfg.get("region", "us-central1"))}


def min_chips_for_memory(weight_bytes: float, acc: str) -> int | None:
    """Smallest tensor-parallel degree (1, 2, 4, 8) whose memory fits the weights, ignoring machine shapes."""
    for n in (1, 2, 4, 8):
        if fits(weight_bytes, acc, n):
            return n
    return None


def shape_note(weight_bytes: float, acc: str, count: int, machine_type: str, region: str | None = None) -> str | None:
    """Note when the machine shape forces more chips than memory needs (e.g. H200 only comes as 8 GPUs)."""
    need = min_chips_for_memory(weight_bytes, acc)
    if need is None or need >= count:
        return None
    g = (shapes(region).get(acc) if region else None) or GPUS[acc]
    unit = "TPU chips" if g.get("kind") == "tpu" else "GPUs"
    family = machine_type.rsplit("-", 1)[0] if "-" in machine_type else machine_type
    return (f"{family} only comes with {count} {unit} on Agent Platform"
            if min(g["machines"]) == count else f"{machine_type} is the smallest {g['name']} shape that fits") + \
        f"; the weights need {need}."


def workload_cfg(cfg: dict) -> dict:
    """Config with a default traffic shape when none is set, so fleet sizing always has something to work on."""
    t = cfg.get("traffic") or {}
    has_shape = t.get("hourly_csv") or t.get("custom_hourly") or t.get("preset")
    if not has_shape and not (t.get("peak_input_tps") and t.get("average_input_tps")
                              and t.get("floor_input_tps") is not None):
        cfg = {**cfg, "traffic": {**t, "peak_input_tps": 150000, "average_input_tps": 50000, "floor_input_tps": 6000}}
    return cfg


def fleet_economics(cfg: dict, c: dict, per_chip: float | None = None, tier: str = "vertex_cud_3y",
                    gemini_catalog: dict | None = None, mix_factor_ref: float | None = None) -> dict:
    """Fleet size and all-in $ per 1M input tokens (fleet + MaaS overflow) for one candidate on one pricing tier.

    Without `per_chip` the spec estimate is used, a short-prompt rate (mix factor 1.0). `mix_factor_ref` is the mix
    factor of a passed session throughput (costmodel.ref_mix_factor); default 1.0.
    """
    from aiops_dilemma import costmodel
    prices = c.get("prices") or {}
    if not prices.get(tier):
        return {"fleet_gpus": None, "usd_per_m_input": None, "monthly_usd": None}
    per = float(per_chip or c.get("est_input_tps_per_gpu") or 0)
    f_ref = float(mix_factor_ref or 1.0) if per_chip else 1.0
    if per <= 0:
        return {"fleet_gpus": None, "usd_per_m_input": None, "monthly_usd": None}
    try:
        r = costmodel.scenario(workload_cfg(cfg), per, prices, tier, int(c["accelerator_count"]),
                               gemini_catalog=gemini_catalog, mix_factor_ref=f_ref)
    except Exception:
        r = {}
    if not r.get("total"):
        return {"fleet_gpus": None, "usd_per_m_input": None, "monthly_usd": None}
    return {"fleet_gpus": int(r["gpus"]), "usd_per_m_input": round(r["per_m"], 4), "monthly_usd": round(r["total"], 0),
            "vs_maas": round(r["vs_maas"], 3)}


def enrich_candidates(cfg: dict, cands: list[dict], gemini_catalog: dict | None = None,
                      throughput: dict | None = None, selected: dict | None = None) -> list[dict]:
    """Add fleet_gpus, usd_per_m_input (Agent Platform + reservation, 3-yr), shape_note and pricing options; sort by $/M.

    `throughput` ({per_chip, source, label}) replaces the spec estimate for the `selected` hardware so the table and
    the session agree on one number.
    """
    from aiops_dilemma.costmodel import TIERS, class_penalties, ref_mix_factor, with_measured_classes
    w = modelinfo.precision_bytes(cfg["model"])
    sel_key = (selected or {}).get("accelerator_type"), (selected or {}).get("accelerator_count")
    cfg = with_measured_classes(cfg, throughput)     # a benchmark's per-class rates (selected hardware)
    # Long-prompt rates belong to the selected hardware: other candidates get the same slowdown, not the same tok/s.
    other_cfg = cfg
    tp_cfg = cfg.get("throughput") or {}
    if tp_cfg.get("medium_tps") or tp_cfg.get("large_tps"):
        sel = next((c for c in cands if (c["accelerator_type"], c["accelerator_count"]) == sel_key), None)
        short_sel = (float(throughput["per_chip"]) * ref_mix_factor(cfg, throughput)
                     if throughput and throughput.get("per_chip") else (sel or {}).get("est_input_tps_per_gpu"))
        if short_sel:
            pen = class_penalties(cfg, short_sel)
            other_cfg = {**cfg, "throughput": {k: pen[k.split("_")[0]] for k in ("medium_penalty", "large_penalty")
                                               if pen["source"][k.split("_")[0]] == "measured"}}
    out = []
    for c in cands:
        c = dict(c)
        c["throughput_source"] = "estimated"
        c["is_selected"] = (c["accelerator_type"], c["accelerator_count"]) == sel_key
        if c["is_selected"] and throughput and throughput.get("per_chip"):
            c["est_input_tps_per_gpu"] = round(float(throughput["per_chip"]), 0)
            c["est_replica_input_tps"] = c["est_input_tps_per_gpu"] * c["accelerator_count"]
            c["throughput_source"] = throughput.get("source", "estimated")
            c.update(fleet_economics(cfg, c, per_chip=c["est_input_tps_per_gpu"], gemini_catalog=gemini_catalog,
                                     mix_factor_ref=ref_mix_factor(cfg, throughput)))
        else:
            c.update(fleet_economics(other_cfg if not c["is_selected"] else cfg, c, gemini_catalog=gemini_catalog))
        c["shape_note"] = ("; ".join(x for x in (c.get("fit_note"), shape_note(w, c["accelerator_type"],
                                c["accelerator_count"], c["machine_type"], cfg.get("region"))) if x) or None)
        pr = c.get("prices") or {}
        c["pricing_options"] = [{"tier": t, "label": TIERS[t], "usd_per_chip_hour": round(pr[t], 4)}
                                for t in TIERS if pr.get(t)]
        out.append(c)
    out.sort(key=lambda c: (c["usd_per_m_input"] is None, c["usd_per_m_input"] or 0))
    return out


def format_plan(p: dict) -> str:
    rows = [f"Model weights at chosen precision: {p['weights_gb']:.1f} GB", "",
            "| Accelerator | Type | Chips/replica | Machine | Verified | Agent Platform $/chip-h | Agent Platform + reservation, 3-yr $/chip-h | Est. tok/s/chip | Est. cost index |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in p["candidates"]:
        pr = c.get("prices", {})
        rows.append(
            f"| {c['gpu']} | {c.get('kind', 'gpu').upper()} | {c['accelerator_count']} | {c['machine_type']} | "
            f"{'yes' if c['verified'] else 'no'} | ${pr.get('vertex', 0):.2f} | ${pr.get('vertex_cud_3y', 0):.2f} | "
            f"{c.get('est_input_tps_per_gpu', 0):,.0f} | {c.get('est_cost_index', 0):.4f} |"
        )
    ch = p["chosen"]
    rows += ["", f"Chosen: {ch['accelerator_count']}x {ch['accelerator_type']} on {ch['machine_type']}"
             + (" (Model Garden verified)" if ch.get("verified") else "")]
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------ reference benchmarks
def reference_entry(cfg: dict, run_dir) -> tuple[dict, list[str]]:
    """A catalog.REFERENCE_BENCHMARKS entry from a finished reference run (measured.json, endpoint.json if still
    there). Returns (entry, warnings): warnings when the run's workload isn't the reference workload or the peak sat
    at the highest concurrency tested (the GPU may not have been saturated)."""
    import datetime
    import json
    from pathlib import Path

    from aiops_dilemma.catalog import REFERENCE_WORKLOAD, validate_reference
    rd = Path(run_dir)
    mpath = rd / "measured.json"
    if not mpath.exists():
        raise SystemExit(f"{mpath} not found: run the reference benchmark first (dilemma auto -c ...)")
    m = json.loads(mpath.read_text())
    if m.get("estimated") or not m.get("levels"):
        raise SystemExit(f"{mpath} is an estimate, not a load test")
    ep = m.get("endpoint") or {}
    if (rd / "endpoint.json").exists():
        ep = {**ep, **json.loads((rd / "endpoint.json").read_text())}
    warn = []
    req, mdl = cfg.get("requests") or {}, cfg.get("model") or {}
    checks = {"model": (mdl.get("model_garden_id"), REFERENCE_WORKLOAD["model_garden_id"]),
              "precision": (mdl.get("precision"), REFERENCE_WORKLOAD["precision"]),
              "pct_over_32k": (float(req.get("pct_over_32k") or 0), 0.0),
              "thinking_share": (float(req.get("thinking_share") or 0), 0.0)}
    for k, (got, want) in checks.items():
        if got != want:
            warn.append(f"{k} is {got!r}, the reference workload uses {want!r}")
    if (mdl.get("chat_template_kwargs") or {}).get("enable_thinking"):
        warn.append("thinking is on; the reference workload has it off")
    peak = m.get("peak") or {}
    levels = [s.get("concurrency") for s in m["levels"]]
    if peak.get("concurrency") == max(levels):
        warn.append(f"the peak was at the highest level tested (c{peak.get('concurrency')}): the GPU may not be "
                    "saturated; consider adding a higher concurrency")
    if not m.get("within_budget", True):
        warn.append("no level met the latency budget")
    serving = m.get("serving") or {}
    if any(serving.get(k) for k in ("command", "env", "patch", "tuned")) or serving.get("args_mode") == "replace":
        warn.append("the run used a custom serving container or a tuned config: reference benchmarks use the "
                    "recipe or pinned image with default flags, so this is not comparable across GPUs")
    entry = {
        "accelerator_type": ep.get("accelerator_type"), "accelerator_count": ep.get("accelerator_count"),
        "machine_type": ep.get("machine_type"),
        "model_garden_id": mdl.get("model_garden_id"), "precision": ep.get("precision") or mdl.get("precision"),
        "workload": REFERENCE_WORKLOAD["key"],
        "short_prompt_tps_per_chip": round(float(m["input_tps_per_gpu"])),
        "measured_at": datetime.date.fromtimestamp(mpath.stat().st_mtime).isoformat(),
        "image": ep.get("image") or serving.get("image") or (cfg.get("deployment") or {}).get("image"),
        "notes": (f"peak c{peak.get('concurrency')}, p95 {float(peak.get('lat_p95') or 0):.0f} s, "
                  f"{peak.get('ok', 0)} ok / {peak.get('errors', 0)} errors"
                  + (f"; image args from {serving.get('image_source')}" if serving.get("image_source") else "")),
        "source": "measured",
        "avg_input_tokens": round(float(peak.get("avg_in") or 0)),
        "avg_output_tokens": round(float(peak.get("avg_out") or 0)),
        "peak_concurrency": peak.get("concurrency"),
        "p95_latency_s": round(float(peak.get("lat_p95") or 0), 1),
    }
    validate_reference(entry)
    return entry, warn
