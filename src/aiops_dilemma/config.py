"""Load a workload definition and fill defaults, so a minimal (auto-mode) file works."""

import copy
import os
import re
import sys
from pathlib import Path

import yaml

from aiops_dilemma import auth, regions

DEFAULTS = {
    "name": "run",
    "project": None,
    "region": "us-central1",
    "use_case": {
        "task_type": "extraction",
        "priority": "balanced",
        "selection_mode": "recommend",  # recommend | manual
    },
    "model": {
        "model_garden_id": None,
        "hf_id": None,
        "precision": "bf16",
        "max_model_len": None,
        "chat_template_kwargs": {},
        "params_b": None,
        "active_params_b": None,
    },
    "deployment": {
        "mode": "auto",
        "accelerator_filter": "all",  # all | gpu | tpu
        "machine_type": None,
        "accelerator_type": None,
        "accelerator_count": None,
        "extra_args": [],
        "image": None,                  # serving image; None = the Model Garden verified recipe's (catalog.SERVING_IMAGES)
        "existing_endpoint": None,
        "dedicated_dns": None,
    },
    "traffic": {
        "hourly_csv": None,
        "peak_input_tps": None,
        "average_input_tps": None,
        "floor_input_tps": None,
        "busy_hours_utc": [13, 20],
        "weekend_factor": 0.15,
        "monthly_input_tokens": None,   # optional: rescales the week to this monthly total (overrides the average)
        "custom_hourly": None,          # optional: 168 input tok/s values, Monday 00:00 UTC first (user-drawn week)
        "preset": None,                 # optional named shape: us_core_hours | us_europe | follow_the_sun | steady
    },
    "requests": {
        "avg_input_tokens": 6000,
        "avg_output_tokens": 300,
        "pct_short": 100.0,
        "pct_medium": 0.0,
        "pct_long": 0.0,
        "pct_over_32k": 0.0,
        "pct_over_128k": 0.0,
        "medium_mean": 69000,
        "large_mean": 188000,
        "system_prompt_tokens": 1500,
        "system_prompt_cache_hit": 0.9,
        # Optional: average input tokens per request served from a prompt cache (conversation history / shared
        # context that is resent each turn, as in chat and coding agents). Empty = system prompt x cache hit rate.
        "cached_input_tokens": None,
        "thinking_share": 0.0,        # share of requests with thinking on (0-1)
        "avg_thinking_tokens": 0,     # extra output tokens per request when thinking is on
    },
    "maas": {"input_per_m": 0.15, "output_per_m": 0.60, "cached_per_m": 0.015, "max_rps": 10},
    "hybrid": {
        "route_long": False,
        # Targets: open_maas | a role alias (OVERFLOW_TARGETS: gemini_flash_lite | gemini_flash | gemini_pro | gemini_4,
        # resolved to the newest matching model) | any live Gemini model id, e.g. gemini-4-carbon (gemini.py)
        "route_long_target": "gemini_flash_lite",
        "peak_overflow_target": "open_maas",
        "failover_target": "gemini_flash_lite",    # also prices the baseline's overflow beyond the MaaS limit
    },
    "constraints": {"latency_budget_s": 600, "burst_headroom": 1.3, "resilience_multiplier": 1.0},
    "prompts": {
        "source": "synthetic",
        "file": None,
        "system_prompts_file": None,
        "custom_system_prompts": [],
        "synthetic": {
            "domain": "General enterprise assistant answering internal questions.",
            "num_system_prompts": 8,
            "count": 1600,   # >= 4 levels x 300 requests + headroom: every level gets fresh prompts
            "json_output": True,
            # Ask for answers of ~avg_output_tokens in the system prompt, so a benchmark measures the output length
            # you configured. False: short answers (reason under 25 words), which flatters throughput.
            "match_output_length": True,
        },
    },
    # preflight: one long streamed answer + one long prompt before the sweep (loadtest.preflight); off = skip it
    "loadtest": {"concurrency": [16, 32, 64, 128], "requests_per_level": 300, "preflight": True},
    # tier: "auto" = the cheapest option, or one of costmodel.TIERS (vertex, vertex_cud_1y, vertex_cud_3y, cud_1y,
    # cud_3y); the headline and the web app use it.
    "pricing": {"tier": "auto", "overrides": {}},
    # Optional measured per-chip input tok/s for long prompts in mixed traffic (same hardware as the short-prompt
    # rate). Empty = default slowdowns (costmodel.DEFAULT_PENALTIES: 3x for 32K-128K, 4x for >128K). A benchmark's bucket_tps fills them in.
    "throughput": {"medium_tps": None, "large_tps": None},
}


def _default_project() -> str | None:
    """The app's project (auth.app_project: chosen in the web app, else env / gcloud config, memoized)."""
    return auth.app_project()


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def normalize_buckets(req: dict) -> dict:
    """Keep pct_short / pct_medium / pct_long and pct_over_32k / pct_over_128k synchronized."""
    if "pct_medium" in req or "pct_long" in req:
        med = max(0.0, float(req.get("pct_medium") or 0.0))
        lng = max(0.0, float(req.get("pct_long") or 0.0))
        if med > 0 or lng > 0 or req.get("pct_over_32k", 0) == 0:
            req["pct_over_128k"] = round(lng, 3)
            req["pct_over_32k"] = round(min(100.0, med + lng), 3)
    over32 = max(0.0, min(100.0, float(req.get("pct_over_32k") or 0.0)))
    over128 = max(0.0, min(over32, float(req.get("pct_over_128k") or 0.0)))
    req["pct_over_32k"] = round(over32, 3)
    req["pct_over_128k"] = round(over128, 3)
    req["pct_long"] = round(over128, 3)
    req["pct_medium"] = round(max(0.0, over32 - over128), 3)
    req["pct_short"] = round(max(0.0, 100.0 - over32), 3)
    return req


OVERFLOW_TARGETS = ("open_maas", "gemini_flash_lite", "gemini_flash", "gemini_pro", "gemini_4")   # + live model ids
HYBRID_TARGET_FIELDS = ("route_long_target", "peak_overflow_target", "failover_target")


def _normalize_gemini_targets(cfg: dict) -> None:
    """Hybrid targets may pin a live Gemini model id; one the live catalog no longer offers falls back to its role
    alias with a warning (`_gemini_warnings`, shown by the app). Nothing is checked without a live catalog."""
    from aiops_dilemma import gemini
    hyb = cfg.get("hybrid") if isinstance(cfg.get("hybrid"), dict) else {}
    cfg["hybrid"] = hyb
    warns = []
    for f in HYBRID_TARGET_FIELDS:
        if f not in hyb:
            continue
        v, w = gemini.check_target(hyb[f], f"hybrid.{f}")
        hyb[f] = v
        if w:
            warns.append(w)
    if warns:
        cfg["_gemini_warnings"] = warns
    else:
        cfg.pop("_gemini_warnings", None)
OFF_PEAK_RANGE = (1.0, 10.0)
HOURS_PER_WEEK = 168


def _num(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if x == x and x not in (float("inf"), float("-inf")) else default


def effective_output_tokens(req: dict) -> float:
    """Output tokens billed and generated per request: visible answer + thinking_share x avg_thinking_tokens.

    `avg_output_tokens` is the visible answer; `thinking_share` (0-1) the share of requests with thinking on;
    `avg_thinking_tokens` the extra output tokens such a request generates (default 0).
    """
    req = req or {}
    out = max(0.0, _num(req.get("avg_output_tokens"), 0.0))
    share = min(1.0, max(0.0, _num(req.get("thinking_share"), 0.0)))
    think = max(0.0, _num(req.get("avg_thinking_tokens"), 0.0))
    return out + share * think


def clamp_off_peak(value) -> float:
    """Off-peak load multiplier, valid range 1-10 (invalid -> 1)."""
    v = _num(value, 1.0)
    return min(OFF_PEAK_RANGE[1], max(OFF_PEAK_RANGE[0], v))


def clean_custom_hourly(values) -> tuple[list[float] | None, str | None]:
    """Validate traffic.custom_hourly: 168 finite numbers (input tok/s, Monday 00:00 UTC first), negatives -> 0.

    Returns (cleaned list or None, error message or None). Empty / missing -> (None, None).
    """
    if values is None or values == "" or values == []:
        return None, None
    if not isinstance(values, (list, tuple)):
        return None, "traffic.custom_hourly must be a list of 168 hourly input tok/s values"
    out = []
    for v in values:
        x = _num(v, float("nan"))
        if x != x:
            return None, "traffic.custom_hourly holds a non-numeric value"
        out.append(max(0.0, x))
    if len(out) != HOURS_PER_WEEK:
        return None, f"traffic.custom_hourly needs {HOURS_PER_WEEK} hourly values (Monday 00:00 UTC first), got {len(out)}"
    if max(out) <= 0:
        return None, "traffic.custom_hourly is all zeros"
    return out, None


RESILIENCE_LEVELS = (1.0, 1.5, 2.0)


def normalize_resilience(value) -> float:
    """Hardware redundancy multiplier: missing, blank, invalid or < 1 -> 1; else snapped to 1, 1.5 or 2.

    A blank UI <select> once sent +"" = 0, which priced a zero-GPU fleet; never let that through.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if v != v or v < 1:  # NaN or below single-zone
        return 1.0
    return min(RESILIENCE_LEVELS, key=lambda lvl: (abs(lvl - v), lvl))


PRICING_TIERS = ("vertex", "vertex_cud_1y", "vertex_cud_3y", "cud_1y", "cud_3y")   # = costmodel.TIERS keys


def normalize_pricing_tier(value) -> str:
    """'auto' (cheapest option) or a known tier key; anything else -> 'auto'."""
    v = str(value or "auto").strip().lower()
    return v if v in PRICING_TIERS else "auto"


PRECISION_ALIASES = {"int4": "int4", "4bit": "int4", "4-bit": "int4", "w4a16": "int4", "awq": "int4", "gptq": "int4",
                     "mxfp4": "mxfp4", "mx-fp4": "mxfp4", "mx_fp4": "mxfp4", "fp8": "fp8", "bf16": "bf16"}


def normalize_precision(value) -> str:
    """bf16 | fp8 | int4 (4-bit checkpoint aliases w4a16 / awq / gptq / 4bit) | mxfp4 (MXFP4 checkpoints such as
    gpt-oss); anything else -> bf16."""
    return PRECISION_ALIASES.get(str(value or "bf16").strip().lower().replace(" ", ""), "bf16")


def normalize(raw: dict, path: str | None = None) -> dict:
    cfg = _merge(DEFAULTS, raw or {})
    pr = cfg.get("pricing") if isinstance(cfg.get("pricing"), dict) else {}
    pr["tier"] = normalize_pricing_tier(pr.get("tier"))
    if not isinstance(pr.get("overrides"), dict):
        pr["overrides"] = {}
    cfg["pricing"] = pr
    cons = cfg.get("constraints") if isinstance(cfg.get("constraints"), dict) else {}
    cons["resilience_multiplier"] = normalize_resilience(cons.get("resilience_multiplier"))
    cfg["constraints"] = cons
    cfg["model"]["precision"] = normalize_precision(cfg["model"].get("precision"))
    region = cfg.get("region")
    if not regions.is_valid(region):   # goes into API host names: never an arbitrary value
        if region not in (None, ""):
            cfg["_region_warning"] = (f"region {str(region)[:40]!r} isn't a Google Cloud region; using "
                                      f"{regions.DEFAULT} (regions look like us-central1 or europe-west4)")
        cfg["region"] = regions.DEFAULT
    else:
        cfg.pop("_region_warning", None)
    maas = cfg.get("maas") if isinstance(cfg.get("maas"), dict) else {}
    rps = max(0.0, _num(maas.get("max_rps"), 0.0))   # null / missing = no MaaS overflow, never a crash
    maas["max_rps"] = int(rps) if rps == int(rps) else rps
    cfg["maas"] = maas
    if path:
        cfg["_path"] = str(Path(path).resolve())
    if not cfg.get("project"):
        cfg["project"] = _default_project() or "my-gcp-project"
    normalize_buckets(cfg["requests"])
    _normalize_traffic(cfg)
    _normalize_gemini_targets(cfg)
    tp = cfg.get("throughput") if isinstance(cfg.get("throughput"), dict) else {}
    for k in ("medium_tps", "large_tps"):
        v = _num(tp.get(k), 0.0)
        tp[k] = v if v > 0 else None
    for k in ("short_tps", "source", "medium_penalty", "large_penalty"):   # transient (costmodel.with_measured_classes), never saved
        tp.pop(k, None)
    cfg["throughput"] = tp
    return cfg


def _normalize_traffic(cfg: dict) -> None:
    """Off-peak clamp, custom week / preset / monthly total validation, and a solvable parametric week.

    With `custom_hourly`, a `preset` or `monthly_input_tokens` (and no CSV), peak / average / floor are rewritten
    from the effective week so the YAML, the UI and the traffic preview agree with what the cost model uses.
    """
    from aiops_dilemma import traffic as tr_mod
    t = cfg.get("traffic") if isinstance(cfg.get("traffic"), dict) else {}
    cfg["traffic"] = t
    warnings = cfg.setdefault("_warnings", [])
    if "off_peak_multiplier" in t:
        t["off_peak_multiplier"] = clamp_off_peak(t.get("off_peak_multiplier"))
    monthly = _num(t.get("monthly_input_tokens"), 0.0)
    t["monthly_input_tokens"] = monthly if monthly > 0 else None
    custom, err = clean_custom_hourly(t.get("custom_hourly"))
    t["custom_hourly"] = custom
    if err:
        warnings.append(err + "; ignored")
    preset = t.get("preset")
    preset = str(preset).strip().lower() if preset not in (None, "", "custom", "none") else None
    if preset and preset not in tr_mod.PRESETS:
        warnings.append(f"traffic.preset {t.get('preset')!r} is unknown (choose from {', '.join(tr_mod.PRESETS)}); ignored")
        preset = None
    if preset and not (_num(t.get("average_input_tps"), 0.0) > 0 or t.get("monthly_input_tokens")):
        warnings.append("traffic.preset needs average_input_tps or monthly_input_tokens; ignored")
        preset = None
    t["preset"] = preset
    csv = bool(t.get("hourly_csv"))
    # Ensure parametric traffic is solvable when user specifies spiky peak/average/floor
    if not csv and not custom and not preset and t.get("peak_input_tps") and t.get("floor_input_tps") and \
            (t.get("average_input_tps") or t.get("monthly_input_tokens")):
        try:
            tr_mod.weekly_profile(t)
        except ValueError:
            for wk in (0.08, 0.05, 0.02):
                t["weekend_factor"] = wk
                try:
                    tr_mod.weekly_profile(t)
                    break
                except ValueError:
                    t["busy_hours_utc"] = [14, 18]
    if not csv and (custom or preset or t.get("monthly_input_tokens")):
        try:
            prof = tr_mod.weekly_profile(t)
            t["peak_input_tps"] = round(float(prof.max()), 3)
            t["average_input_tps"] = round(float(prof.mean()), 3)
            t["floor_input_tps"] = round(float(prof.min()), 3)
        except (ValueError, KeyError, TypeError) as e:
            warnings.append(f"traffic: {e}")
    if not warnings:
        cfg.pop("_warnings", None)


def load(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = normalize(raw, path=path)
    if cfg.get("_region_warning"):
        print(f"warning: {path}: {cfg['_region_warning']}", file=sys.stderr)
    missing = []
    if not cfg.get("project"):
        missing.append("project")
    if not cfg["model"].get("model_garden_id"):
        missing.append("model.model_garden_id")
    if missing:
        raise SystemExit(f"config {path}: missing {', '.join(missing)}")
    return cfg


def to_yaml(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    return yaml.safe_dump(clean, sort_keys=False, default_flow_style=False)


NAME_RE = re.compile(r"(?!\.{1,2}$)[A-Za-z0-9_.-]{1,100}")   # run folder names: no separators, not . or ..


def run_dir(cfg: dict) -> Path:
    """`_run_dir` (the app, `--run-dir`), else a config already in runs/<name>/, else $DILEMMA_RUNS_DIR/<name>
    when set (so CLI runs land where the app looks), else runs/ next to the config (or the cwd)."""
    env_runs = os.environ.get("DILEMMA_RUNS_DIR", "").strip()
    p = Path(cfg["_path"]) if cfg.get("_path") else None
    if cfg.get("_run_dir"):
        d = Path(cfg["_run_dir"])
    elif p is not None and p.parent.parent.name == "runs" and p.parent.name == cfg["name"]:
        d = p.parent
    elif env_runs:
        d = Path(env_runs) / cfg["name"]
    elif p is not None:
        if p.parent.name == "examples":
            d = p.parent.parent / "runs" / cfg["name"]
        else:
            d = p.parent / "runs" / cfg["name"]
    else:
        d = Path.cwd() / "runs" / cfg["name"]
    if not cfg.get("_run_dir") and not NAME_RE.fullmatch(str(cfg.get("name") or "")):
        raise SystemExit(f"config name {str(cfg.get('name'))[:40]!r}: use letters, digits, '-', '_' or '.' "
                         "(it becomes the run folder's name)")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_input_tokens(req: dict) -> float:
    """Average cached input tokens per request: requests.cached_input_tokens if set, else system prompt x hit rate.

    Capped at the average input. Drives the cached-input price on MaaS / Gemini and the prefill discount on GPUs."""
    avg_in = float(req.get("avg_input_tokens") or 0.0)
    v = req.get("cached_input_tokens")
    try:
        v = float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        v = None
    if v is None or v < 0:
        v = float(req.get("system_prompt_tokens") or 0.0) * float(req.get("system_prompt_cache_hit") or 0.0)
    return min(v, avg_in) if avg_in > 0 else v
