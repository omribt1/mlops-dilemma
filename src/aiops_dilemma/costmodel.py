"""Monthly cost of a fixed (no-autoscaling) GPU/TPU fleet with MaaS / Gemini overflow, vs a pay-per-token baseline.

Accelerator capacity is often scarce, so the fleet is sized for the busiest hour and always on
(min = max replicas). A managed API (Open-Model MaaS or Google Gemini on Agent Platform) absorbs the top
of the peak up to its rate limit, and can also absorb diverted long-context (>32K) requests or serve
as a $0-idle-cost Layer-3 resilience failover tier.

The baseline is honest about the MaaS rate limit: when the peak needs more requests/s than open-model MaaS
allows, "MaaS only" is not an option, and the baseline becomes MaaS up to its limit plus overflow to the
failover target (a Gemini tier), priced per token.
"""

import math
from dataclasses import dataclass

import numpy as np

from aiops_dilemma import traffic
from aiops_dilemma.config import effective_output_tokens, normalize_resilience, cached_input_tokens

HOURS_PER_MONTH = 730
TARGET_UTIL = 0.85
TIERS = {
    "vertex": "Agent Platform, on demand",
    "vertex_cud_1y": "Agent Platform + reservation, 1-yr",
    "vertex_cud_3y": "Agent Platform + reservation, 3-yr",
    "cud_1y": "Self-managed GKE, 1-yr",
    "cud_3y": "Self-managed GKE, 3-yr",
}


GEMINI_OVERFLOW_RPS = 150   # quota assumed for Gemini as the peak-overflow target (as in model.js)


def _gemini_catalog(gemini_catalog: dict | None) -> dict:
    if gemini_catalog:
        from aiops_dilemma.catalog import GEMINI_MODELS
        missing = {k: v for k, v in GEMINI_MODELS.items() if k not in gemini_catalog}
        return {**missing, **gemini_catalog} if missing else gemini_catalog
    from aiops_dilemma import pricing
    return pricing.gemini_prices()


def gemini_key(target: str | None, gem_cat: dict) -> str | None:
    """The gem_cat key that prices `target` (a role alias or a live model id): itself when listed; a model id that is
    no longer offered -> its family's role alias (gemini.alias_for_id); open_maas / unknown -> None."""
    t = (target or "").lower()
    if t in gem_cat and t != "open_maas":
        return t
    if t.startswith("gemini"):
        from aiops_dilemma.gemini import alias_for_id
        a = alias_for_id(t)
        return a if a in gem_cat else None
    return None


def failover_key(target: str | None, gem_cat: dict) -> str:
    """Gemini model that takes traffic beyond the open-model MaaS limit (open_maas can't: falls back to Flash-Lite)."""
    return gemini_key(target, gem_cat) or "gemini_flash_lite"


def breakeven_hours(extra_monthly_cost: float, cost_per_outage_hour: float) -> tuple[float, bool]:
    """Outage hours per month at which always-on redundant hardware costs the same as paying for failover.

    Capped at HOURS_PER_MONTH: above that, failover is cheaper at any realistic outage rate (`never` = True).
    """
    if extra_monthly_cost <= 0:
        return 0.0, False
    if cost_per_outage_hour <= 0:
        return float(HOURS_PER_MONTH), True
    hours = extra_monthly_cost / cost_per_outage_hour
    if hours >= HOURS_PER_MONTH:
        return float(HOURS_PER_MONTH), True
    return round(hours, 1), False


@dataclass
class Fleet:
    gpus: int
    gpu_cost: float
    maas_cost: float
    util: float
    gpu_token_share: float
    gemini_cost: float = 0.0   # spill beyond the MaaS cap (to the Gemini peak-overflow tier, or the failover tier)
    # Fleet policy (one rule for the optimizer and the waterline): see fleet_policy()
    policy_ok: bool = True
    needs_gemini_at_peaks: bool = False   # open_maas target: some hour spills beyond the MaaS cap
    over_overflow_quota: bool = False     # Gemini target: some hour spills beyond MaaS cap + Gemini quota
    short_of_headroom: bool = False       # fewer chips than (peak - overflow cap) x headroom / per-chip rate
    headroom_gpus: int = 0                # that headroom bound, in whole replicas
    min_policy_gpus: int | None = None    # smallest fleet that satisfies the policy (fixed_fleet only)

    @property
    def total(self) -> float:
        return self.gpu_cost + self.maas_cost + self.gemini_cost


def maas_rate(cfg: dict, out_ratio: float, cache: float) -> float:
    """$ per input token on MaaS, matching output folded in."""
    m = cfg["maas"]
    return ((1 - cache) * m["input_per_m"] + cache * m["cached_per_m"] + out_ratio * m["output_per_m"]) / 1e6


def gemini_rate(gem: dict, out_ratio: float, cache: float, is_long: bool = False) -> float:
    """$ per input token on Google Gemini on Agent Platform, with context caching and output folded in.
    `is_long`: every request at the long-context tier (prompts over its threshold)."""
    lc = long_tier(gem) if is_long else None
    inp = lc["input_per_m"] if lc else gem["input_per_m"]
    outp = lc["output_per_m"] if lc else gem["output_per_m"]
    cached = lc["cached_per_m"] if lc else gem.get("cached_per_m", inp * 0.1)
    return ((1 - cache) * inp + cache * cached + out_ratio * outp) / 1e6


def long_tier(gem: dict) -> dict | None:
    """The model's long-context price tier {threshold, input_per_m, output_per_m, cached_per_m}, or None (one price).
    From `long_context` (live catalog), else the older long_input_per_m / long_output_per_m fields."""
    lc = gem.get("long_context")
    if lc and lc.get("input_per_m") is not None:
        inp = float(gem["input_per_m"])
        return {"threshold": float(lc.get("threshold") or 200000), "input_per_m": float(lc["input_per_m"]),
                "output_per_m": float(lc.get("output_per_m", gem["output_per_m"])),
                "cached_per_m": float(lc["cached_per_m"]) if lc.get("cached_per_m") is not None
                else float(gem.get("cached_per_m", inp * 0.1)) * float(lc["input_per_m"]) / max(inp, 1e-12)}
    li, lo = gem.get("long_input_per_m"), gem.get("long_output_per_m")
    if (li is not None and li != gem["input_per_m"]) or (lo is not None and lo != gem["output_per_m"]):
        li = float(li if li is not None else gem["input_per_m"])
        inp = float(gem["input_per_m"])
        return {"threshold": 200000.0, "input_per_m": li, "output_per_m": float(lo if lo is not None else gem["output_per_m"]),
                "cached_per_m": float(gem.get("cached_per_m", inp * 0.1)) * li / max(inp, 1e-12)}
    return None


BUCKET_LOW = {"small": 0.0, "medium": 32000.0, "large": 128000.0}   # lower prompt-size bound of each class


def above_threshold(lo: float, mean: float, threshold: float) -> tuple[float, float]:
    """(token share, request share) of a prompt class above `threshold`, assuming its prompt sizes spread evenly
    from `lo` to 2 x mean - lo (the class mean in the middle). Degenerate range: all or nothing by the mean."""
    hi = 2 * mean - lo
    if hi <= lo + 1e-9:
        return (1.0, 1.0) if mean > threshold else (0.0, 0.0)
    if threshold <= lo:
        return 1.0, 1.0
    if threshold >= hi:
        return 0.0, 0.0
    return (hi * hi - threshold * threshold) / (hi * hi - lo * lo), (hi - threshold) / (hi - lo)


def gemini_long_rate(gem: dict, b: dict, classes=("medium", "large")) -> float:
    """$ per input token for the prompt classes routed to a Gemini model (over 32K by default), with its long-context
    tier applied to the share of each class above the tier's threshold (above_threshold). Equals gemini_rate on the
    blended class when the model has a single price. Mirrors model.js geminiLongRate."""
    lc = long_tier(gem)
    inp, outp = float(gem["input_per_m"]), float(gem["output_per_m"])
    cached = float(gem.get("cached_per_m", inp * 0.1))
    tot = cost = 0.0
    for k in classes:
        v = b[k]
        share = v["token_share"]
        if share <= 0:
            continue
        st, sr = above_threshold(BUCKET_LOW[k], v["mean"], lc["threshold"]) if lc else (0.0, 0.0)
        li, lo_, lcache = (lc["input_per_m"], lc["output_per_m"], lc["cached_per_m"]) if lc else (inp, outp, cached)
        c, r = v["cached"], v["out_ratio"]
        rate = ((1 - c) * ((1 - st) * inp + st * li) + c * ((1 - st) * cached + st * lcache)
                + r * ((1 - sr) * outp + sr * lo_)) / 1e6
        tot += share
        cost += share * rate
    return cost / tot if tot > 0 else 0.0


def request_mix(cfg: dict) -> tuple[float, float]:
    """(output/input ratio with thinking tokens included, cached share of input)."""
    r = cfg["requests"]
    out_ratio = effective_output_tokens(r) / r["avg_input_tokens"]
    cache = min(1.0, cached_input_tokens(r) / r["avg_input_tokens"])
    return out_ratio, cache


# Measured on Gemma 4 26B-A4B FP8, 1x RTX PRO 6000, mixed traffic (2026-11): 3.0x / 4.1x.
DEFAULT_PENALTIES = {"medium": 3.0, "large": 4.0}


def class_penalties(cfg: dict, short_tps: float | None = None) -> dict:
    """Per-token slowdown of 32K-128K ("medium") and >128K ("large") prompts relative to short (<=32K) prompts.

    From measured per-class rates when known: cfg["throughput"]["medium_tps"/"large_tps"] (input tok/s per chip in
    mixed traffic, set by the user or filled from a benchmark's bucket_tps), divided into the short-prompt rate
    (`short_tps`, else cfg["throughput"]["short_tps"]). Otherwise the defaults (3x, 4x; measured on one model and GPU).
    Returns {"medium", "large", "source": {"medium": "measured"|"default", ...}}.
    """
    t = cfg.get("throughput") or {}
    short = short_tps or t.get("short_tps")
    out, src = {}, {}
    for k, dflt in DEFAULT_PENALTIES.items():
        fixed = t.get(f"{k}_penalty")     # transient: a slowdown carried over from the selected hardware
        if fixed:
            out[k], src[k] = min(50.0, max(1.0, float(fixed))), "measured"
            continue
        try:
            rate = float(t.get(f"{k}_tps") or 0)
        except (TypeError, ValueError):
            rate = 0.0
        if short and rate > 0:
            out[k], src[k] = min(50.0, max(1.0, float(short) / rate)), "measured"
        else:
            out[k], src[k] = dflt, "default"
    out["source"] = src
    return out


def with_measured_classes(cfg: dict, measured: dict | None) -> dict:
    """cfg with throughput.{short,medium,large}_tps filled from a benchmark's per-class rates (bucket_tps).

    Values the user set in the config win. Returns cfg unchanged when there is nothing to add."""
    bt = (measured or {}).get("bucket_tps") or {}
    if not bt.get("small") or not (bt.get("medium") or bt.get("large")):
        return cfg
    t = dict(cfg.get("throughput") or {})
    for k in ("medium", "large"):          # per class: a value the user set wins
        if not t.get(f"{k}_tps") and bt.get(k):
            t[f"{k}_tps"] = float(bt[k])
    t.update(short_tps=float(bt["small"]), source="measured")
    return {**cfg, "throughput": t}


def buckets(cfg: dict, short_tps: float | None = None) -> dict:
    """Compute Short (<=32K), Medium (32K-128K), and Long (>128K) request & token shares and GPU time share.

    Speed penalties come from class_penalties(cfg, short_tps): measured per-class rates when known, else defaults."""
    pen = class_penalties(cfg, short_tps)
    r = cfg["requests"]
    over32 = max(0.0, min(1.0, float(r.get("pct_over_32k") or 0.0) / 100.0))
    over128 = max(0.0, min(over32, float(r.get("pct_over_128k") or 0.0) / 100.0))
    med_req = max(0.0, over32 - over128)
    lrg_req = over128
    sml_req = max(1e-6, 1.0 - med_req - lrg_req)

    med_mean = float(r.get("medium_mean") or 69000.0)
    lrg_mean = float(r.get("large_mean") or 188000.0)
    avg_in = float(r.get("avg_input_tokens") or 6000.0)
    avg_out = effective_output_tokens(r) if r.get("avg_output_tokens") is not None else 400.0
    sml_mean = max(800.0, (avg_in - med_req * med_mean - lrg_req * lrg_mean) / sml_req)
    total_mean = sml_req * sml_mean + med_req * med_mean + lrg_req * lrg_mean
    cached_tok = cached_input_tokens(r)

    b = {
        "small": {"req": sml_req, "mean": sml_mean, "speed_penalty": 1.0},
        "medium": {"req": med_req, "mean": med_mean, "speed_penalty": pen["medium"]},
        "large": {"req": lrg_req, "mean": lrg_mean, "speed_penalty": pen["large"]},
    }
    weighted_time = 0.0
    for k, v in b.items():
        v["token_share"] = (v["req"] * v["mean"]) / max(1.0, total_mean)
        v["cached"] = min(1.0, cached_tok / max(1.0, v["mean"]))
        v["out_ratio"] = avg_out / max(1.0, v["mean"])
        v["rel_gpu_sec"] = v["token_share"] * v["speed_penalty"]
        weighted_time += v["rel_gpu_sec"]

    # Mix cost factor: GPU-seconds per token of this mix relative to short (<=32K) prompts (1.0 with none over 32K).
    mix_factor = weighted_time if over32 > 0 else 1.0
    small_time_share = b["small"]["rel_gpu_sec"] / max(1e-9, weighted_time)
    long_gpu_time_share = max(0.0, 1.0 - small_time_share)
    # When >32K prompts are diverted off GPUs/TPUs, per-GPU throughput on <=32K prompts increases
    small_speedup = (b["small"]["token_share"] / max(1e-6, small_time_share)) if over32 > 0 else 1.0
    return {
        "small": b["small"],
        "medium": b["medium"],
        "large": b["large"],
        "long_token_share": b["medium"]["token_share"] + b["large"]["token_share"],
        "long_req_share": med_req + lrg_req,
        "long_gpu_time_share": long_gpu_time_share,
        "small_speedup": max(1.0, small_speedup),
        "mix_factor": max(1.0, mix_factor),
        "penalty_source": pen["source"],
        # Mean request size of the class mix: avg_input_tokens, unless the short class hit its 800-token floor
        # because the long tail is too heavy for that average (short_floor_hit: config warns about it).
        "total_mean": total_mean,
        "short_floor_hit": over32 > 0 and (avg_in - med_req * med_mean - lrg_req * lrg_mean) / sml_req < 800.0,
    }


def mean_request_tokens(cfg: dict) -> float:
    """Average input tokens per request, for turning tok/s into requests/s: the configured avg_input_tokens (the
    input you know). A long tail too heavy for it is reported (config._mix_warning), not silently resized."""
    return float(cfg["requests"].get("avg_input_tokens") or 0.0)


def mix_factor(cfg: dict, short_tps: float | None = None) -> float:
    """Mix cost factor f(mix) = sum of token_share x speed_penalty over the size buckets (1.0 without >32K prompts)."""
    return buckets(cfg, short_tps)["mix_factor"]


def _is_real_run(measured: dict) -> bool:
    return bool(measured.get("levels")) and not measured.get("estimated")


def ref_mix_factor(cfg: dict, measured: dict | None) -> float:
    """Mix factor f_ref of the prompt mix a per-chip throughput figure was obtained at.

    `measured` is a measured.json record or a throughput dict ({per_chip, source, ...}).
    - Spec estimates and user overrides (source "estimated" / "override", or nothing) are short-prompt rates: 1.0.
    - Real load tests and "measured"/"calibrated" sources ran at the config's mix: the recorded "mix_factor"
      (written by analyze.py), else the current mix's factor (older runs, i.e. the figure is taken as blended
      for the current mix).
    The blended throughput for the current mix is per_chip x f_ref / f_now.
    """
    if not measured:
        return 1.0
    bt = measured.get("bucket_tps") or {}
    try:
        if bt.get("small") and float(measured.get("input_tps_per_gpu") or 0) > 0:
            # per-class calibration: the blended figure is the short-prompt rate / f, so f_ref = small / blended
            return max(1.0, float(bt["small"]) / float(measured["input_tps_per_gpu"]))
    except (TypeError, ValueError):
        pass
    src = measured.get("throughput_source") or measured.get("source")
    if _is_real_run(measured) or src in ("measured", "calibrated"):
        f = measured.get("mix_factor")
        try:
            f = float(f) if f is not None else None
        except (TypeError, ValueError):
            f = None
        return f if f and f > 0 else mix_factor(cfg)
    return 1.0


def _round_up(g: float, replica: int) -> int:
    return max(replica, math.ceil(max(g, 0) / replica) * replica)


def headroom_bound(demand, per_gpu, replica, maas_cap_tps, headroom, gem_cap_tps=None) -> int:
    """Burst-headroom rule: the fleet plus the overflow it may use must cover peak x headroom, i.e.
    n >= (peak - cap) x headroom / per_gpu, where cap = (MaaS cap + Gemini quota when Gemini takes peak overflow)
    / headroom. In whole replicas, at least one."""
    cap = (maas_cap_tps + (gem_cap_tps or 0.0)) / headroom
    return _round_up((float(np.max(demand)) - cap) * headroom / per_gpu, replica)


def eval_fleet(demand, n, per_gpu, price, maas_cap_tps, rate, headroom, gem_rate, gem_cap_tps=None,
               replica: int = 1) -> Fleet:
    """A fixed fleet of n chips: it serves up to its capacity, the rest spills to open-model MaaS up to its cap
    (`rate`) and beyond that to Gemini (`gem_rate`). One policy for the optimizer and the waterline (model.js
    evalFleet):

    - `gem_cap_tps` None: peak overflow goes to open-model MaaS, Gemini is NOT allowed at peaks. Spill beyond the
      MaaS cap is still priced (at the failover tier) so a what-if shows its cost, but flags `needs_gemini_at_peaks`.
    - `gem_cap_tps` a number: a Gemini tier takes the peak overflow beyond the MaaS cap, within its quota; spill
      beyond MaaS cap + quota flags `over_overflow_quota`.
    - Both: fewer chips than the burst-headroom bound (headroom_bound) flags `short_of_headroom`.
    """
    scale = HOURS_PER_MONTH / len(demand)
    cap = maas_cap_tps / headroom
    allowed = cap + (gem_cap_tps / headroom if gem_cap_tps is not None else 0.0)
    on_gpu = np.minimum(demand, n * per_gpu * TARGET_UTIL)
    spill = demand - on_gpu
    maas = np.minimum(spill, cap)
    gem = np.maximum(0.0, spill - maas)
    beyond = bool((spill > allowed + 1e-6).any())
    bound = headroom_bound(demand, per_gpu, replica, maas_cap_tps, headroom, gem_cap_tps)
    short = n < bound
    needs_gem = beyond and gem_cap_tps is None
    over_quota = beyond and gem_cap_tps is not None
    return Fleet(n, n * price * HOURS_PER_MONTH, float(maas.sum()) * 3600 * scale * rate,
                 float(on_gpu.mean() / (n * per_gpu)) if n > 0 else 0.0,
                 float(on_gpu.sum() / max(1e-9, demand.sum())), float(gem.sum()) * 3600 * scale * gem_rate,
                 policy_ok=not (beyond or short), needs_gemini_at_peaks=needs_gem, over_overflow_quota=over_quota,
                 short_of_headroom=short, headroom_gpus=bound)


def fixed_fleet(demand, per_gpu, replica, price, maas_cap_tps, rate, headroom, gem_rate=0.0,
                gem_cap_tps=None) -> Fleet | None:
    """Cheapest fleet that satisfies the policy (eval_fleet): no spill beyond what the peak-overflow target may take
    and at least the burst-headroom bound. Policy-OK fleets are those from the smallest one up; above the size
    that holds peak x headroom on its own, extra chips only add cost. So this is the cheapest policy-OK point on
    the waterline. `min_policy_gpus` = the smallest policy-OK fleet."""
    peak = float(np.max(demand))
    for_peak = _round_up(peak * headroom / per_gpu, replica)
    allowed = (maas_cap_tps + (gem_cap_tps or 0.0)) / headroom
    lo = max(_round_up((peak - allowed) / (per_gpu * TARGET_UTIL), replica),
             headroom_bound(demand, per_gpu, replica, maas_cap_tps, headroom, gem_cap_tps))
    best, first = None, None
    for n in range(lo, max(lo, for_peak) + 1, replica):
        f = eval_fleet(demand, n, per_gpu, price, maas_cap_tps, rate, headroom, gem_rate, gem_cap_tps, replica)
        if not f.policy_ok:
            continue
        first = n if first is None else first
        if best is None or f.total < best.total:
            best = f
    if best is not None:
        best.min_policy_gpus = first
    return best


def baseline(demand, cfg: dict, maas_rps: float, rate: float, gem_cat: dict, failover_target: str | None,
             out_ratio: float, cache: float, headroom: float = 1.0) -> dict:
    """Pay-per-token baseline that respects the open-model MaaS rate limit.

    MaaS is planned at limit / headroom per hour, exactly as for a fleet's overflow (hourly averages hide bursts that
    would hit the limit). Feasible (peak requests/s <= limit / headroom): everything on MaaS ("maas_only").
    Otherwise MaaS serves up to that each hour and the rest overflows to the failover target ("maas_plus_overflow").
    """
    scale = HOURS_PER_MONTH / len(demand)
    avg_in = mean_request_tokens(cfg)
    peak_rps = float(demand.max()) / max(1.0, avg_in)
    cap_rps = max(0.0, float(maas_rps or 0.0))
    headroom = max(1.0, float(headroom or 1.0))
    feasible = peak_rps <= cap_rps / headroom + 1e-9
    if feasible:
        total = float(demand.sum()) * 3600 * scale * rate
        base = {"kind": "maas_only", "total": total, "maas_cost": total, "overflow_cost": 0.0,
                "overflow_target": None, "label": "Open-model MaaS only"}
    else:
        key = failover_key(failover_target, gem_cat)
        gem = gem_cat[key]
        cap_tps = cap_rps * avg_in / headroom
        on_maas = np.minimum(demand, cap_tps)
        over = demand - on_maas
        maas_cost = float(on_maas.sum()) * 3600 * scale * rate
        over_cost = float(over.sum()) * 3600 * scale * gemini_rate(gem, out_ratio, cache)
        name = gem.get("name", key)
        base = {"kind": "maas_plus_overflow", "total": maas_cost + over_cost, "maas_cost": maas_cost,
                "overflow_cost": over_cost, "overflow_target": key,
                "label": (f"Open-model MaaS up to {cap_rps:g} req/s + {name} overflow" if cap_rps > 0
                          else f"{name} only (open-model MaaS limit is 0 req/s)")}
    return {"maas_only_feasible": feasible, "maas_peak_rps": peak_rps, "maas_cap_rps": cap_rps,
            "maas_planned_rps": cap_rps / headroom, "baseline": base}


def scenario(cfg: dict, per_gpu: float, prices: dict, tier: str, replica: int, *, off_peak=1.0, growth=1.0,
             maas_rps=None, tput_mult=1.0, resilience=None, route_long: bool | None = None,
             route_long_target: str | None = None, gemini_catalog: dict | None = None,
             mix_factor_ref: float = 1.0, peak_overflow_target: str | None = None,
             failover_target: str | None = None, fleet_override: int | None = None, _hint: bool = True) -> dict:
    """Monthly cost of the cheapest fixed fleet on `tier` for this config.

    `per_gpu` is a per-chip throughput obtained at a prompt mix with factor `mix_factor_ref` (1.0 = short prompts;
    see ref_mix_factor). All prompts on the fleet run at per_gpu x f_ref / f_now; with >32K prompts routed away
    the short-prompt fleet runs at per_gpu x f_ref.

    Fleet policy (`peak_overflow_target`, one rule for the optimum and a `fleet_override` fleet, see eval_fleet):
    peak overflow always fills open-model MaaS within its limit first. "open_maas" (default): Gemini is not allowed
    at peaks, so the optimum never spills beyond the MaaS limit; an overridden fleet that does gets the spill
    priced at the failover tier and the flags `needs_gemini_at_peaks` / `policy_ok` False. A Gemini tier: it takes
    the overflow beyond the MaaS limit, within GEMINI_OVERFLOW_RPS. Both apply the burst-headroom bound
    (`short_of_headroom`). The optimum = the cheapest policy-OK fleet = the cheapest policy-OK waterline point.
    `failover_target`: Gemini tier for traffic beyond the MaaS limit (baseline, and the what-if above).
    With the default policy, `peak_gemini_hint` is the optimum when peak overflow may go to the failover tier,
    when that is at least 3% cheaper. Mirrors model.js scenario().
    """
    demand = traffic.weekly_profile(cfg["traffic"]) * growth
    demand = traffic.raise_off_peak(demand, off_peak) if off_peak != 1 else demand
    out_ratio, cache = request_mix(cfg)
    rate = maas_rate(cfg, out_ratio, cache)
    scale = HOURS_PER_MONTH / len(demand)
    tokens = float(demand.sum()) * 3600 * scale
    rps = cfg["maas"]["max_rps"] if maas_rps is None else maas_rps
    res = normalize_resilience(cfg["constraints"].get("resilience_multiplier") if resilience is None else resilience)
    hyb = cfg.get("hybrid", {})
    do_route_long = hyb.get("route_long", False) if route_long is None else route_long
    target = (route_long_target or hyb.get("route_long_target") or "gemini_flash_lite").lower()
    gem_cat = _gemini_catalog(gemini_catalog)
    if target != "open_maas":
        target = gemini_key(target, gem_cat) or "gemini_flash_lite"
    b = buckets(cfg, short_tps=per_gpu * float(mix_factor_ref or 1.0))   # short-prompt rate, before the what-if
    ov_target = (peak_overflow_target or hyb.get("peak_overflow_target") or "open_maas").lower()
    ov_target = gemini_key(ov_target, gem_cat) or "open_maas"
    ov_gem = gem_cat.get(ov_target) if ov_target != "open_maas" else None
    fail_key = failover_key(failover_target or hyb.get("failover_target"), gem_cat)
    gem_quota_rps = GEMINI_OVERFLOW_RPS if ov_gem else None   # Gemini quota beyond the MaaS limit at peaks

    maas_only = tokens * rate
    base_info = baseline(demand, cfg, rps, rate, gem_cat, fail_key, out_ratio, cache,
                         headroom=cfg["constraints"]["burst_headroom"])
    common = {"tier": tier, "maas_only": maas_only, "monthly_tokens": tokens, "peak": float(demand.max()),
              "avg": float(demand.mean()), "buckets": b, "effective_output_tokens": effective_output_tokens(cfg["requests"]),
              "peak_overflow_target": ov_target, "failover_target": fail_key, **base_info}
    eff_per_gpu = per_gpu * tput_mult
    f_ref = float(mix_factor_ref or 1.0)
    blended_per_gpu = eff_per_gpu * f_ref / b["mix_factor"]   # throughput for the current mix, all on the fleet
    long_maas_cost = 0.0
    long_rps_peak = 0.0

    if do_route_long and b["long_token_share"] > 0:
        long_share = b["long_token_share"]
        long_req = b["long_req_share"]
        long_mean = (
            (b["medium"]["req"] * b["medium"]["mean"] + b["large"]["req"] * b["large"]["mean"]) / max(1e-9, long_req)
        )
        long_cached = (
            (b["medium"]["token_share"] * b["medium"]["cached"] + b["large"]["token_share"] * b["large"]["cached"])
            / max(1e-9, long_share)
        )
        long_out_ratio = effective_output_tokens(cfg["requests"]) / max(1.0, long_mean)
        long_tps = demand * long_share
        long_rps_peak = float(long_tps.max()) / max(1.0, long_mean)

        if target in gem_cat:
            long_rate = gemini_long_rate(gem_cat[target], b)   # long-context tier on the share over its threshold
            spare_rps = float(rps or 0)  # Gemini on Agent Platform handles long prompts on its own quota
        else:
            long_rate = maas_rate(cfg, long_out_ratio, long_cached)
            spare_rps = float(rps or 0) - long_rps_peak

        if spare_rps < 0:
            return {
                **common, "fleet": None,
                "infeasible": "Long-prompt traffic exceeds Open-Model MaaS RPS limit; route >32K prompts to Google Gemini on Agent Platform.",
            }

        small_demand = demand * b["small"]["token_share"]
        small_per_gpu = blended_per_gpu * b["small_speedup"]   # = eff_per_gpu * f_ref (small_speedup = f_now)
        small_rate = maas_rate(cfg, b["small"]["out_ratio"], b["small"]["cached"])
        mean_in = b["small"]["mean"]
        fleet_args = (small_demand, small_per_gpu, prices[tier], spare_rps * mean_in, small_rate,
                      cfg["constraints"]["burst_headroom"])
        fail_out, fail_cache = b["small"]["out_ratio"], b["small"]["cached"]
        long_maas_cost = float(long_tps.sum()) * 3600 * scale * long_rate
    else:
        mean_in = mean_request_tokens(cfg)
        fleet_args = (demand, blended_per_gpu, prices[tier], float(rps or 0) * mean_in, rate,
                      cfg["constraints"]["burst_headroom"])
        fail_out, fail_cache = out_ratio, cache
    d_fleet, chip_tps, price, cap_tps, maas_rate_used, headroom = fleet_args
    spill_key = ov_target if ov_gem else fail_key        # Gemini tier that takes spill beyond the MaaS cap
    gem_rate_used = gemini_rate(gem_cat[spill_key], fail_out, fail_cache)
    gem_cap = gem_quota_rps * mean_in if gem_quota_rps else None
    # Redundant chips are standby capacity on top of the serving fleet: each serving chip really costs res x price,
    # so the optimizer weighs chips against overflow at that price (res = 1: unchanged).
    fleet = fixed_fleet(d_fleet, chip_tps, replica, price * res, cap_tps, maas_rate_used, headroom, gem_rate_used,
                        gem_cap)
    optimal = fleet
    if fleet_override is not None:
        n = max(0, int(round(float(fleet_override)))) * max(1, replica)
        fleet = eval_fleet(d_fleet, n, chip_tps, price, cap_tps, maas_rate_used, headroom, gem_rate_used, gem_cap,
                           max(1, replica))

    out = {
        **common, "fleet": fleet, "route_long": do_route_long, "route_long_target": target,
        "long_maas_cost": long_maas_cost, "long_rps_peak": long_rps_peak,
        "mix_factor": b["mix_factor"], "mix_factor_ref": f_ref, "blended_per_gpu": blended_per_gpu,
        "optimal_gpus": optimal.gpus if optimal else None, "overridden": fleet_override is not None,
        "spill_gemini_target": spill_key, "gemini_allowed_at_peaks": bool(ov_gem),
        "min_policy_gpus": optimal.min_policy_gpus if optimal else None,
        "headroom_gpus": headroom_bound(d_fleet, chip_tps, max(1, replica), cap_tps, headroom, gem_cap),
        "policy_ok": True, "needs_gemini_at_peaks": False, "over_overflow_quota": False, "short_of_headroom": False,
    }
    bt = out["baseline"]["total"]
    rnd_res = lambda g: math.ceil(g * res / max(1, replica)) * max(1, replica)  # noqa: E731  whole replicas
    if optimal:
        out["optimal_total"] = rnd_res(optimal.gpus) * prices[tier] * HOURS_PER_MONTH + optimal.maas_cost \
            + optimal.gemini_cost + long_maas_cost
    if fleet:
        gem_over = fleet.gemini_cost
        out["gpus"] = rnd_res(fleet.gpus)
        out["gpu_cost"] = out["gpus"] * prices[tier] * HOURS_PER_MONTH
        out["maas_overflow_cost"] = fleet.maas_cost          # peak overflow within the MaaS limit
        out["gemini_overflow_cost"] = gem_over               # beyond the MaaS limit, to spill_gemini_target
        out["total"] = out["gpu_cost"] + fleet.maas_cost + gem_over + long_maas_cost
        out["per_m"] = out["total"] / (tokens / 1e6)
        out["vs_maas"] = out["total"] / maas_only
        out["vs_baseline"] = out["total"] / bt if bt > 0 else None
        for k in ("policy_ok", "needs_gemini_at_peaks", "over_overflow_quota", "short_of_headroom"):
            out[k] = bool(getattr(fleet, k))
    if _hint and not ov_gem and optimal:
        alt = scenario(cfg, per_gpu, prices, tier, replica, off_peak=off_peak, growth=growth, maas_rps=maas_rps,
                       tput_mult=tput_mult, resilience=resilience, route_long=route_long,
                       route_long_target=route_long_target, gemini_catalog=gem_cat, mix_factor_ref=mix_factor_ref,
                       peak_overflow_target=fail_key, failover_target=failover_target, _hint=False)
        out["peak_gemini_hint"] = peak_gemini_hint(out["optimal_total"], alt, gem_cat)
    return out


PEAK_HINT_MIN_SAVING = 0.03   # show the "allow Gemini at peaks" hint when it saves at least 3%


def peak_gemini_hint(current_total: float, alt: dict, gem_cat: dict) -> dict | None:
    """The optimum with peak overflow to the failover Gemini tier, when it is >= 3% cheaper than `current_total`."""
    if not alt or alt.get("total") is None or not current_total:
        return None
    saving = current_total - alt["total"]
    if saving < PEAK_HINT_MIN_SAVING * current_total:
        return None
    key = alt["peak_overflow_target"]
    name = (gem_cat.get(key) or {}).get("name", key)
    return {"target": key, "name": name, "gpus": alt["gpus"], "total": alt["total"], "saving": saving,
            "saving_share": saving / current_total, "vs_baseline": alt.get("vs_baseline"),
            "text": hint_text(name, alt["gpus"], saving)}


def hint_text(name: str, gpus: int, saving: float, chips: str = "GPUs") -> str:
    """One verdict line (report.md, dashboard.html and the app say the same)."""
    return (f"Allowing {name} overflow at peaks would cut it to {gpus} {chips} and save about "
            f"${saving / 1000:,.1f}K a month.")


def resilience_and_hybrid_summary(cfg: dict, per_gpu: float, prices: dict, tier: str, replica: int,
                                  gemini_catalog: dict | None = None, mix_factor_ref: float = 1.0) -> dict:
    """Analyze reliability/resilience (Always-On Redundant Hardware vs Layer-3 Gemini on Agent Platform)
    and long-context / peak diversion savings.
    """
    gem_cat = _gemini_catalog(gemini_catalog)
    lite_name = gem_cat["gemini_flash_lite"].get("name", "Gemini Flash-Lite")
    sc = lambda **kw: scenario(cfg, per_gpu, prices, tier, replica, gemini_catalog=gem_cat,  # noqa: E731
                               mix_factor_ref=mix_factor_ref, **kw)
    base = sc(resilience=1.0, route_long=False)
    zone_red = sc(resilience=1.5, route_long=False)
    multi_reg = sc(resilience=2.0, route_long=False)

    route_open = sc(resilience=1.0, route_long=True, route_long_target="open_maas")
    route_gem_lite = sc(resilience=1.0, route_long=True, route_long_target="gemini_flash_lite")
    route_gem_flash = sc(resilience=1.0, route_long=True, route_long_target="gemini_flash")

    demand = traffic.weekly_profile(cfg["traffic"])
    peak_hour = int(np.argmax(demand))                 # the week's busiest hour, whichever day it falls on
    day = demand[(peak_hour // 24) * 24:(peak_hour // 24) * 24 + 24]   # and the day that contains it
    peak_tps = float(demand[peak_hour])
    maas_cap_tps = (float(cfg["maas"].get("max_rps") or 0.0) * mean_request_tokens(cfg)
                    / max(1.0, float(cfg["constraints"].get("burst_headroom") or 1.0)))   # as planned for overflow
    spill_peak_hour_tok = max(0.0, peak_tps - maas_cap_tps) * 3600.0
    spill_full_day_tok = float(np.maximum(day - maas_cap_tps, 0.0).sum()) * 3600.0
    maas_peak_hour_tok = min(peak_tps, maas_cap_tps) * 3600.0

    out_ratio, cache = request_mix(cfg)
    open_rate = maas_rate(cfg, out_ratio, cache)
    lite_rate = gemini_rate(gem_cat["gemini_flash_lite"], out_ratio, cache)
    flash_rate = gemini_rate(gem_cat["gemini_flash"], out_ratio, cache)
    pro_rate = gemini_rate(gem_cat["gemini_pro"], out_ratio, cache)
    g4_rate = gemini_rate(gem_cat["gemini_4"], out_ratio, cache)

    incidents = []
    for label, tok in (
        ("Accelerator fleet down for peak hour (1h)", spill_peak_hour_tok),
        ("Accelerator fleet down for full busy weekday (24h)", spill_full_day_tok),
        ("Open-Model MaaS rate-limited during peak hour (1h)", maas_peak_hour_tok),
    ):
        incidents.append({
            "scenario": label,
            "tokens_m": round(tok / 1e6, 2),
            "gemini_flash_lite": round(tok * lite_rate, 2),
            "gemini_flash": round(tok * flash_rate, 2),
            "gemini_pro": round(tok * pro_rate, 2),
            "gemini_4": round(tok * g4_rate, 2),
            "open_maas_ref": round(tok * open_rate, 2),
        })

    base_gpus = base.get("gpus") or 0
    base_total = base.get("total") or 0.0
    zone_extra_gpus = (zone_red.get("gpus") or 0) - base_gpus
    zone_extra_cost = (zone_red.get("total") or 0.0) - base_total
    reg_extra_gpus = (multi_reg.get("gpus") or 0) - base_gpus
    reg_extra_cost = (multi_reg.get("total") or 0.0) - base_total

    peak_hour_lite_cost = spill_peak_hour_tok * lite_rate
    breakeven_outage_hours, breakeven_never = breakeven_hours(zone_extra_cost, peak_hour_lite_cost)

    recommendations = []
    b = base["buckets"]
    if b["long_token_share"] > 0 and route_gem_lite.get("total") and base_total > 0:
        saved_gpus = base_gpus - (route_gem_lite.get("gpus") or base_gpus)
        saved_usd = base_total - route_gem_lite["total"]
        if saved_usd > 0 or saved_gpus > 0:
            recommendations.append({
                "type": "long_context_gemini",
                "title": "Divert >32K Long-Context Requests to Google Gemini on Agent Platform",
                "detail": (
                    f"Requests >32K tokens are {cfg['requests'].get('pct_over_32k', 0):g}% of requests but consume "
                    f"{b['long_gpu_time_share']:.0%} of accelerator time. Routing >32K prompts to {lite_name} "
                    f"cuts the self-hosted fleet by {saved_gpus} chips and saves ${saved_usd:,.0f}/month."
                ),
                "saved_gpus": saved_gpus,
                "saved_monthly_usd": round(saved_usd, 0),
            })

    if zone_extra_cost > 0:
        verdict = (
            "failover is cheaper at any realistic outage rate"
            if breakeven_never
            else f"idle hardware only pays off above ~{breakeven_outage_hours:,.0f} peak outage hours/month"
        )
        recommendations.append({
            "type": "resilience_gemini_failover",
            "title": "Use Pay-Per-Use Google Gemini on Agent Platform for HA Failover Instead of Idle Redundant GPUs/TPUs",
            "detail": (
                f"Always-on zone redundancy (1.5x) adds {zone_extra_gpus} idle chips (+${zone_extra_cost:,.0f}/month), "
                f"and two-region redundancy (2.0x) adds {reg_extra_gpus} chips (+${reg_extra_cost:,.0f}/month). "
                f"Failing over to {lite_name} on Agent Platform costs $0/month when healthy and "
                f"${incidents[0]['gemini_flash_lite']:,.0f} for a full 1-hour peak outage, so {verdict}."
            ),
            "saved_gpus": zone_extra_gpus,
            "saved_monthly_usd": round(zone_extra_cost, 0),
        })

    return {
        "incidents": incidents,
        "zone_extra_gpus": zone_extra_gpus,
        "zone_extra_cost": zone_extra_cost,
        "reg_extra_gpus": reg_extra_gpus,
        "reg_extra_cost": reg_extra_cost,
        "breakeven_outage_hours": breakeven_outage_hours,
        "breakeven_never": breakeven_never,
        "breakeven_peak_outage_hours": breakeven_outage_hours,  # deprecated alias, capped like breakeven_outage_hours
        "ratio_lite_vs_open": round(lite_rate / max(1e-12, open_rate), 2),
        "ratio_flash_vs_open": round(flash_rate / max(1e-12, open_rate), 2),
        "routing_options": {
            "all_on_fleet": base,
            "route_long_open_maas": route_open,
            "route_long_gemini_flash_lite": route_gem_lite,
            "route_long_gemini_flash": route_gem_flash,
        },
        "recommendations": recommendations,
    }


def breakeven_utilization(cfg: dict, per_gpu: float, price: float, mix_factor_ref: float = 1.0) -> float:
    """GPU/TPU utilization above which an accelerator-hour beats MaaS on the same tokens (at the current mix)."""
    out_ratio, cache = request_mix(cfg)
    f_ref = float(mix_factor_ref or 1.0)
    blended = per_gpu * f_ref / mix_factor(cfg, per_gpu * f_ref)
    return price / (blended * 3600 * maas_rate(cfg, out_ratio, cache))
