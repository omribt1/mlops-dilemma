"""Build the single-file interactive dashboard for a run (no external dependencies; share as-is)."""

import json
from html import escape
from pathlib import Path

from aiops_dilemma import costmodel, pricing, report, traffic
from aiops_dilemma.config import normalize_resilience
from aiops_dilemma.catalog import GPUS
from aiops_dilemma.costmodel import TIERS

ASSETS = Path(__file__).parent / "assets"


def build_data(cfg: dict, measured: dict, prices: dict) -> dict:
    user_tp = dict(cfg.get("throughput") or {})            # rates the user typed (win over a benchmark's)
    cfg = costmodel.with_measured_classes(cfg, measured)   # a benchmark's per-class rates, if it has them
    ep = measured["endpoint"]
    r, m, c = cfg["requests"], cfg["maas"], cfg["constraints"]
    hyb = cfg.get("hybrid") or {}
    gem_cat = costmodel._gemini_catalog(pricing.gemini_prices(cfg.get("project"), cfg.get("region", "us-central1")))
    acc_info = GPUS.get(ep["accelerator_type"], {"name": ep["accelerator_type"], "kind": "gpu"})
    src = report.throughput_source(measured)
    per_chip = measured["input_tps_per_gpu"]
    f_ref = costmodel.ref_mix_factor(cfg, measured)
    data = {
        "profile": [round(float(x), 2) for x in traffic.weekly_profile(cfg["traffic"])],
        "perGpu": measured["input_tps_per_gpu"],
        "mixFactorRef": f_ref,   # mix factor perGpu was obtained at (1.0 = short prompts); see costmodel.ref_mix_factor
        # Per-chip input tok/s by prompt size measured by a benchmark (bucket_tps), or null
        "classTps": ({k: round(float(measured["bucket_tps"][b]), 1) for k, b in
                      (("short", "small"), ("medium", "medium"), ("large", "large")) if measured["bucket_tps"].get(b)}
                     if (measured.get("bucket_tps") or {}).get("small") else None),
        "replica": ep["accelerator_count"],
        "prices": {t: prices[t] for t in TIERS if prices.get(t)},
        "pricingLabels": {t: v for t, v in TIERS.items() if prices.get(t)},
        "gemini": gem_cat,
        "throughput": {"per_chip": per_chip, "source": src, "label": report.measured_label(measured, per_chip, src)},
        "measuredLevels": measured.get("levels") if report.is_measured(measured) else [],
        "meta": {
            "model": cfg["model"]["model_garden_id"],
            "gpu": acc_info["name"],
            "kind": acc_info.get("kind", "gpu"),
            "region": cfg["region"],
        },
        "defaults": {
            "pricing": None,   # set below: the saved pricing option, else the cheapest total
            "resilience": normalize_resilience(c.get("resilience_multiplier")),
            "offPeak": 1.0,
            "growth": 1.0,
            "tputMult": 1.0,
            "avgInput": r["avg_input_tokens"],
            "avgOutput": r["avg_output_tokens"],              # visible answer tokens
            "thinkingShare": float(r.get("thinking_share") or 0.0),   # 0-1, share of requests with thinking on
            "thinkingTokens": float(r.get("avg_thinking_tokens") or 0.0),  # extra output tokens when thinking is on
            "pctOver32": r.get("pct_over_32k", 0.0),
            "pctOver128": r.get("pct_over_128k", 0.0),
            "mediumMean": r.get("medium_mean", 69000),
            "largeMean": r.get("large_mean", 188000),
            "mediumTps": user_tp.get("medium_tps"),   # user-set long-prompt rates (tok/s per chip), or null
            "largeTps": user_tp.get("large_tps"),
            "systemPrompt": r["system_prompt_tokens"],
            "systemHit": r["system_prompt_cache_hit"] * 100,
            "cachedInput": r.get("cached_input_tokens"),   # null: system prompt x hit rate
            "routeLong": bool(hyb.get("route_long", False)),
            "routeLongTarget": hyb.get("route_long_target", "gemini_flash_lite"),
            "peakOverflowTarget": hyb.get("peak_overflow_target", "open_maas"),
            "maasRps": m["max_rps"],
            "headroom": c["burst_headroom"],
            "maasIn": m["input_per_m"],
            "maasOut": m["output_per_m"],
            "maasCached": m["cached_per_m"],
            "liteIn": gem_cat["gemini_flash_lite"]["input_per_m"],
            "liteOut": gem_cat["gemini_flash_lite"]["output_per_m"],
            "flashIn": gem_cat["gemini_flash"]["input_per_m"],
            "flashOut": gem_cat["gemini_flash"]["output_per_m"],
            "proIn": gem_cat["gemini_pro"]["input_per_m"],
            "proOut": gem_cat["gemini_pro"]["output_per_m"],
            "g4In": gem_cat["gemini_4"]["input_per_m"],
            "g4Out": gem_cat["gemini_4"]["output_per_m"],
            "failoverTarget": hyb.get("failover_target", "gemini_flash_lite"),
        },
    }
    want = (cfg.get("pricing") or {}).get("tier") or "auto"
    best = want if want in data["prices"] else min(
        data["prices"], key=lambda t: (costmodel.scenario(cfg, data["perGpu"], prices, t, data["replica"],
                                                          gemini_catalog=gem_cat, mix_factor_ref=f_ref
                                                          ).get("total") or 1e18))
    data["defaults"]["pricing"] = best
    data["resilienceSummary"] = costmodel.resilience_and_hybrid_summary(
        cfg, data["perGpu"], prices, best, data["replica"], gemini_catalog=gem_cat, mix_factor_ref=f_ref
    )
    return data


def build(cfg: dict, measured: dict, prices: dict, run_dir: Path) -> Path:
    data = build_data(cfg, measured, prices)
    html = ((ASSETS / "dashboard.html").read_text()
            .replace("/*__DATA__*/", "const DATA = " + json.dumps(data, separators=(",", ":"), default=lambda o: vars(o) if hasattr(o, "__dict__") else str(o)) + ";")
            .replace("/*__MODEL__*/", (ASSETS / "model.js").read_text()))
    note = report.disclaimer(measured)
    if note:   # static, so it shows even before the script runs
        tier = measured.get("throughput_note")
        html = html.replace("</header>", '</header>\n<div class="disclaimer" role="note"><strong>Estimated throughput, '
                            "not measured on your workload.</strong> " + escape(note)
                            + (f" <span>{escape(tier)}</span>" if tier else "") + "</div>", 1)
    out = run_dir / "dashboard.html"
    out.write_text(html)
    return out
