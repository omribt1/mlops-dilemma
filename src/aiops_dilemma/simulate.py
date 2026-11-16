"""Request-level simulation of a fixed GPU/TPU fleet + MaaS overflow, to check the hourly cost model.

The cost model sizes the fleet on hourly averages. This simulates individual requests instead:
Poisson arrivals that follow the weekly hourly curve, request sizes drawn from the workload's
Short (<= 32K) / Medium (32K-128K) / Long (> 128K) mix, and a gateway in front of the fleet.

Each replica serves its requests first-come-first-served (like vLLM's scheduler). Throughput already
reflects batching, so a replica is modelled as working through its queue of GPU-seconds in arrival
order: a request costs `tokens / bucket_tps` replica-seconds, where bucket_tps is the input tok/s per
GPU for its size bucket (output at the workload's ratio included). Equal-share processor sharing is
deliberately *not* used: it starves long prompts under load, which vLLM's FCFS scheduler does not.

The gateway picks the less loaded of two random replicas. If the request would wait there (more than
`overflow_after_s` of queued work) and the MaaS rate limit allows, it goes to MaaS instead.
A request is late if it finishes more than `constraints.latency_budget_s` after arrival. A fleet
passes when <= 1% of short AND <= 1% of long requests are late (checked separately: overload lands
on one class first).

Scenarios (where long prompts > 32K go, and when they arrive):
  gpu       all prompts on the fleet, long prompts arrive in proportion to traffic (hourly-model assumption)
  flat      all on the fleet, each day's long prompts spread evenly over that day (same daily count)
  deferred  all on the fleet; long prompts wait in a low-priority queue and run in off-peak hours
            (hourly demand <= weekly average) on nearly idle replicas, each within a deadline
  routed    long prompts go to MaaS (priority on its rate limit; they wait for a slot)
`burst` adds one 10-minute surge per hour at `burst`x the hourly average; the other 50 minutes are
lower so the hourly mean is unchanged.

The minimum fleet is found by binary search on the busiest weekday, then confirmed on a full simulated
week (grown if the week fails, e.g. deferred work piling up across days); the monthly cost comes from
that week.
"""

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from aiops_dilemma import costmodel, traffic
from aiops_dilemma.config import effective_output_tokens

SCENARIOS = {
    "gpu": "All prompts on the fleet",
    "flat": "All on the fleet, long prompts spread evenly per day",
    "deferred": "All on the fleet, long prompts deferred to off-peak",
    "routed": "Long prompts (>32K) routed to MaaS",
}
BUCKETS = ("small", "medium", "large")
SHORT_MAX, MEDIUM_MAX = 32000, 128000
DEFER_SLACK_S = 30.0          # deferred work only starts on a replica with less queued work than this
SURGE_S = 600                 # 10-minute surge window per hour


# ---------------------------------------------------------------------------------------------
# Inputs: throughput per size bucket, request sizes
# ---------------------------------------------------------------------------------------------
def bucket_tps(cfg: dict, per_gpu: float, measured: dict | None = None, mix_factor_ref: float | None = None) -> dict:
    """Input tok/s per GPU for each size bucket.

    Uses `measured["bucket_tps"]` when a per-class calibration exists. Otherwise `per_gpu` is a per-chip figure
    obtained at a mix with factor f_ref (`mix_factor_ref`, default costmodel.ref_mix_factor(cfg, measured): 1.0 for
    spec estimates and overrides). The blended rate for the sampled mix is per_gpu x f_ref / f_now; short prompts run
    at blended x f_now = per_gpu x f_ref and longer buckets at that divided by their per-token slowdown.
    """
    if measured and measured.get("bucket_tps"):
        return {k: float(measured["bucket_tps"][k]) for k in BUCKETS}
    f_ref = costmodel.ref_mix_factor(cfg, measured) if mix_factor_ref is None else float(mix_factor_ref)
    small = per_gpu * f_ref                       # the short-prompt rate
    b = costmodel.buckets(cfg, short_tps=small)   # measured per-class rates (cfg throughput) or default penalties
    return {k: small / b[k]["speed_penalty"] for k in BUCKETS}


def _size_mix(cfg: dict) -> dict:
    b = costmodel.buckets(cfg)
    return {"share": np.array([b[k]["req"] for k in BUCKETS]) / sum(b[k]["req"] for k in BUCKETS),
            "mean": {k: b[k]["mean"] for k in BUCKETS}}


def _surge_segments(rate: float, burst: float, offset: float) -> list[tuple[float, float, float]]:
    if burst <= 1:
        return [(0.0, 3600.0, rate)]
    burst = min(burst, 5.99)    # the rest of the hour can't go below zero
    low = rate * (6 - burst) / 5
    return [(0.0, offset, low), (offset, offset + SURGE_S, rate * burst), (offset + SURGE_S, 3600.0, low)]


def sample_requests(cfg: dict, t0_h: int, t1_h: int, scenario: str = "gpu", burst: float = 1.0, seed: int = 0,
                    demand: np.ndarray | None = None, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrival times (s from Monday 00:00), input tokens and bucket (0 short, 1 medium, 2 long) for hours [t0_h, t1_h).

    Hours past the end of the week wrap around. `scale` multiplies the traffic (for quick runs).
    """
    rng = np.random.default_rng(seed)
    demand = traffic.weekly_profile(cfg["traffic"]) if demand is None else np.asarray(demand, dtype=float)
    mix = _size_mix(cfg)
    share = mix["share"]
    avg_in = float(cfg["requests"]["avg_input_tokens"])
    hours = np.arange(t0_h, t1_h)
    week_rate = demand / avg_in * scale                      # requests/s per hour of the week
    req_rate = week_rate[hours % len(demand)]
    long_share = float(share[1:].sum())
    short_rate = req_rate * share[0]
    long_rate = req_rate * long_share
    if scenario == "flat":   # each day's long prompts spread evenly over that day (same daily count)
        day_mean = (week_rate * long_share).reshape(-1, 24).mean(axis=1)
        long_rate = day_mean[(hours // 24) % len(day_mean)]
    times, kinds = [], []
    for h, rs, rl in zip(hours, short_rate, long_rate):
        offset = float(rng.uniform(0, 3600 - SURGE_S))       # surge position within the hour
        for rate, kind in ((rs, 0), (rl, 1)):
            if rate <= 0:
                continue
            for a, z, r in _surge_segments(rate, burst, offset):
                n = rng.poisson(r * (z - a))
                times.append(h * 3600.0 + a + rng.uniform(0, z - a, n))
                kinds.append(np.full(n, kind, dtype=np.int8))
    if not times:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int8)
    t = np.concatenate(times)
    k = np.concatenate(kinds)
    order = np.argsort(t, kind="stable")
    t, k = t[order], k[order]

    bucket = np.zeros(len(t), dtype=np.int8)
    is_long = k == 1
    p_large = share[2] / long_share if long_share > 0 else 0.0
    bucket[is_long] = np.where(rng.random(int(is_long.sum())) < p_large, 2, 1)
    tokens = np.empty(len(t))
    # Same size model as the synthetic prompt builder (prompts._sample_input).
    sigma = 0.62
    mu = traffic.clipped_lognormal_mu(mix["mean"]["small"], sigma, 400, SHORT_MAX)   # clipped mean on target
    s = bucket == 0
    tokens[s] = np.clip(rng.lognormal(mu, sigma, int(s.sum())), 400, SHORT_MAX)
    m = bucket == 1
    tokens[m] = np.clip(mix["mean"]["medium"] * rng.uniform(0.65, 1.35, int(m.sum())), 32500, MEDIUM_MAX)
    lg = bucket == 2
    tokens[lg] = np.maximum(128500, mix["mean"]["large"] * rng.uniform(0.75, 1.25, int(lg.sum())))
    return t, tokens, bucket


# ---------------------------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------------------------
def simulate_fcfs(n_replicas: int, t: np.ndarray, tokens: np.ndarray, bucket: np.ndarray, tps: dict,
                  scenario: str = "gpu", maas_rps: float = 0.0, budget_s: float = 600.0,
                  overflow_after_s: float = 120.0, defer_deadline_s: float | None = None,
                  offpeak_hours: np.ndarray | None = None, measure_until: float | None = None,
                  gpus_per_replica: int = 1, seed: int = 1) -> dict:
    """Simulate a fixed fleet of `n_replicas` FCFS replicas behind a gateway with MaaS overflow.

    `tps` is input tok/s per *replica* for each bucket. O(1) per request (O(replicas) while deferred
    work is waiting). Statistics cover requests arriving before `measure_until` (default: all).
    """
    n = len(t)
    if n == 0:
        raise ValueError("no requests to simulate")
    rate = np.array([tps[k] for k in BUCKETS])
    work = (tokens / rate[bucket]).tolist()
    tl, bl = t.tolist(), bucket.tolist()
    rng = np.random.default_rng(seed)
    pa = rng.integers(0, n_replicas, n).tolist()
    pb = rng.integers(0, n_replicas, n).tolist()
    free = [0.0] * n_replicas                 # time each replica finishes its queued work
    done = [math.nan] * n
    via_maas = [False] * n
    maas_wait = [0.0] * n
    routed = scenario == "routed" and maas_rps > 0
    defer = scenario == "deferred"
    offpeak = offpeak_hours.tolist() if (defer and offpeak_hours is not None) else None
    deadline = math.inf if defer_deadline_s is None else defer_deadline_s
    cap = max(1.0, float(maas_rps))           # MaaS token bucket: 1 s of burst, at least one request
    tb, tb_t = (cap if maas_rps > 0 else 0.0), tl[0]
    dq: deque = deque()
    busy = 0.0
    replicas = range(n_replicas)
    key = free.__getitem__

    for i in range(n):
        now = tl[i]
        b = bl[i]
        if maas_rps > 0:
            tb = min(cap, tb + (now - tb_t) * maas_rps)
            tb_t = now
        if dq:
            # Deferred long prompts run oldest first on nearly idle replicas during off-peak hours,
            # or regardless once they are close to their deadline.
            quiet = offpeak is None or offpeak[int(now // 3600) % len(offpeak)]
            while dq:
                j = dq[0]
                urgent = now - tl[j] + work[j] + 3600 >= deadline
                if not (quiet or urgent):
                    break
                g = min(replicas, key=key)
                f = free[g]
                if f - now >= DEFER_SLACK_S and not urgent:
                    break
                dq.popleft()
                free[g] = (f if f > now else now) + work[j]
                done[j] = free[g]
                busy += work[j]
        if b and routed:
            tb -= 1.0
            w = -tb / maas_rps if tb < 0 else 0.0
            via_maas[i] = True
            maas_wait[i] = w
            done[i] = now + w
            continue
        if b and defer:
            dq.append(i)
            continue
        a, c = pa[i], pb[i]
        g = a if free[a] <= free[c] else c
        f = free[g]
        if f - now > overflow_after_s and tb >= 1.0:
            tb -= 1.0
            via_maas[i] = True
            done[i] = now
            continue
        free[g] = (f if f > now else now) + work[i]
        done[i] = free[g]
        busy += work[i]
    while dq:                                  # drain deferred work left at the end of the window
        j = dq.popleft()
        g = min(replicas, key=key)
        free[g] = max(free[g], tl[-1]) + work[j]
        done[j] = free[g]
        busy += work[j]

    done_a = np.array(done)
    maas_a = np.array(via_maas)
    lat = done_a - t
    m = np.ones(n, dtype=bool) if measure_until is None else t < measure_until
    is_long = bucket > 0
    is_def = is_long & defer
    # Short and (non-deferred) long requests are judged against the latency budget; MaaS overflow
    # finishes immediately, routed long prompts count their wait for a MaaS slot.
    late = lat > budget_s
    short_m, long_m = m & ~is_long, m & is_long & ~is_def
    live = m & ~maas_a & ~is_def
    out = {
        "replicas": n_replicas, "gpus": n_replicas * gpus_per_replica,
        "requests": int(m.sum()), "fleet_served": int(live.sum()),
        "late_short": float(late[short_m].mean()) if short_m.any() else 0.0,
        "late_long": float(late[long_m].mean()) if long_m.any() else 0.0,
        "p50_s": float(np.percentile(lat[live], 50)) if live.any() else 0.0,
        "p99_s": float(np.percentile(lat[live], 99)) if live.any() else 0.0,
        "max_s": float(lat[live].max()) if live.any() else 0.0,
        "maas_share": float(maas_a[m].mean()),
        "maas_requests": int((maas_a & m).sum()),
        "maas_input_tokens": float(tokens[maas_a & m].sum()),
        "fleet_input_tokens": float(tokens[m & ~maas_a].sum()),
        "busy_replica_s": busy,
        "util": busy / (n_replicas * max(1e-9, float(np.nanmax(done_a)) - tl[0])),
    }
    hour = (t // 3600).astype(np.int64)
    hour -= hour.min()
    out["maas_peak_rps"] = float(np.bincount(hour[maas_a], minlength=1).max() / 3600) if maas_a.any() else 0.0
    if routed and (maas_a & is_long & m).any():
        out["maas_wait_p99_s"] = float(np.percentile(np.array(maas_wait)[maas_a & is_long & m], 99))
    if is_def.any():
        dm = is_def & m
        waits = lat[dm]
        out["deferred_requests"] = int(dm.sum())
        out["deferred_p99_wait_h"] = float(np.percentile(waits, 99) / 3600) if dm.any() else 0.0
        out["deferred_max_wait_h"] = float(waits.max() / 3600) if dm.any() else 0.0
        out["deferred_done_by_deadline"] = float(np.mean(waits <= deadline)) if dm.any() else 1.0
    return out


def passes(r: dict, target_late: float = 0.01) -> bool:
    return (max(r["late_short"], r["late_long"]) <= target_late
            and r.get("deferred_done_by_deadline", 1.0) >= 0.999)


def min_fleet(run, lo: int, hi: int, target_late: float = 0.01, log=None, max_hi: int = 4096) -> tuple[int, dict, bool]:
    """Smallest replica count with <= `target_late` of short and of long requests over budget.

    Binary search on `run(n) -> result dict`; `hi` doubles until it passes (up to `max_hi`).
    Returns (replicas, result, feasible).
    """
    cache: dict = {}

    def ev(k: int) -> dict:
        if k not in cache:
            cache[k] = run(k)
            if log:
                r = cache[k]
                log(f"    {r['gpus']} GPUs: late short {r['late_short']:.2%} / long {r['late_long']:.2%}, "
                    f"p99 {r['p99_s']:.0f}s, MaaS peak {r['maas_peak_rps']:.1f} req/s"
                    + (f", deferred within deadline {r['deferred_done_by_deadline']:.1%}"
                       if "deferred_done_by_deadline" in r else ""))
        return cache[k]

    lo, hi = max(1, lo), max(1, hi, lo)
    while not passes(ev(hi), target_late):
        if hi >= max_hi:
            return hi, ev(hi), False
        lo, hi = hi + 1, min(max_hi, hi * 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if passes(ev(mid), target_late):
            hi = mid
        else:
            lo = mid + 1
    return hi, ev(hi), True


# ---------------------------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------------------------
def _maas_cost(cfg: dict, n_req: float, in_tokens: float) -> float:
    r, m = cfg["requests"], cfg["maas"]
    from aiops_dilemma.config import cached_input_tokens
    cached = min(cached_input_tokens(r), in_tokens / max(1.0, n_req))
    return ((in_tokens - cached * n_req) * m["input_per_m"] + cached * n_req * m["cached_per_m"]
            + effective_output_tokens(r) * n_req * m["output_per_m"]) / 1e6   # thinking tokens bill as output


def monthly_cost(week: dict, gpu_hour: float, cfg: dict, all_requests: int | None = None,
                 all_tokens: float | None = None, baseline_month: float | None = None) -> dict:
    """Monthly cost from a one-week simulation: always-on GPUs plus the requests sent to MaaS.

    `baseline_month`: the pay-per-token baseline that respects the MaaS rate limit (costmodel.baseline).
    """
    per_month = costmodel.HOURS_PER_MONTH / 168
    gpu = week["gpus"] * gpu_hour * costmodel.HOURS_PER_MONTH
    maas = _maas_cost(cfg, week["maas_requests"], week["maas_input_tokens"]) * per_month
    out = {"gpu_month": gpu, "maas_month": maas, "total_month": gpu + maas}
    if all_requests:
        out["maas_only_month"] = _maas_cost(cfg, all_requests, all_tokens) * per_month
        out["vs_maas_only"] = out["total_month"] / max(1e-9, out["maas_only_month"])
    if baseline_month:
        out["baseline_month"] = baseline_month
        out["vs_baseline"] = out["total_month"] / baseline_month
    return out


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
def busiest_day_window(demand: np.ndarray) -> int:
    """First hour of the 24-hour window used for sizing: the quietest hour of the busiest day."""
    day = int(demand.reshape(-1, 24).sum(axis=1).argmax())
    return day * 24 + int(demand[day * 24:(day + 1) * 24].argmin())


def run(cfg: dict, per_gpu: float, prices: dict, *, gpus_per_replica: int = 1, scenarios=None, bursts=(1.0,),
        target_late: float = 0.01, tier: str = "vertex_cud_3y", scale: float = 1.0,
        defer_deadline_h: float = 24.0, overflow_after_s: float | None = None, measured: dict | None = None,
        out_dir: Path | None = None, log=print, seed: int = 0, mix_factor_ref: float | None = None,
        gemini_catalog: dict | None = None) -> dict:
    """Minimum fleet + monthly cost per (burst, scenario). Writes simulation.json/.md to `out_dir` if given.

    `mix_factor_ref`: mix factor `per_gpu` was obtained at (default costmodel.ref_mix_factor(cfg, measured)).
    """
    cfg = costmodel.with_measured_classes(cfg, measured)
    scenarios = list(scenarios or SCENARIOS)
    for sc in scenarios:
        if sc not in SCENARIOS:
            raise ValueError(f"unknown scenario {sc!r}; choose from {', '.join(SCENARIOS)}")
    notes = []
    if _size_mix(cfg)["share"][1:].sum() <= 0 and any(sc != "gpu" for sc in scenarios):
        notes.append("No prompts over 32K in this workload, so the long-prompt scenarios (flat, deferred, routed) "
                     "are identical to 'All prompts on the fleet' and were skipped.")
        log("  " + notes[-1])
        scenarios = ["gpu"]
    demand = traffic.weekly_profile(cfg["traffic"])
    budget = float(cfg["constraints"]["latency_budget_s"])
    after = min(120.0, budget / 5) if overflow_after_s is None else overflow_after_s
    maas_rps = float(cfg["maas"]["max_rps"] or 0)
    f_ref = costmodel.ref_mix_factor(cfg, measured) if mix_factor_ref is None else float(mix_factor_ref)
    f_now = costmodel.mix_factor(cfg, per_gpu * f_ref)
    blended = per_gpu * f_ref / f_now                        # all prompts on the fleet, at the sampled mix
    tps_gpu = bucket_tps(cfg, per_gpu, measured, mix_factor_ref=f_ref)
    tps_rep = {k: v * gpus_per_replica for k, v in tps_gpu.items()}
    offpeak = demand <= demand.mean()
    h0 = busiest_day_window(demand)
    deadline_s = defer_deadline_h * 3600
    if tier not in prices or not prices.get(tier):
        tier = next(t for t in costmodel.TIERS if prices.get(t))
    tiers = {t: prices[t] for t in costmodel.TIERS if prices.get(t)}
    out_ratio, cache = costmodel.request_mix(cfg)
    gem_cat = costmodel._gemini_catalog(gemini_catalog)
    base = costmodel.baseline(demand * scale, cfg, maas_rps, costmodel.maas_rate(cfg, out_ratio, cache), gem_cat,
                              (cfg.get("hybrid") or {}).get("failover_target"), out_ratio, cache)

    hourly = {}
    for sc, route in (("gpu", False), ("routed", True)):
        try:
            h = costmodel.scenario(cfg, per_gpu, prices, tier, gpus_per_replica, route_long=route,
                                   route_long_target="open_maas", mix_factor_ref=f_ref)
            hourly[sc] = h.get("gpus")
        except Exception:   # noqa: BLE001 - the comparison column is optional
            hourly[sc] = None
    hourly["flat"] = hourly["deferred"] = hourly["gpu"]

    rows = []
    for burst in bursts:
        for sc in scenarios:
            extra = int(math.ceil(defer_deadline_h)) if sc == "deferred" else 0
            t, tok, bk = sample_requests(cfg, h0, h0 + 24 + extra, sc, burst, seed=seed, demand=demand, scale=scale)
            peak_rate = float(demand.max()) * scale
            guess = max(1, math.ceil(peak_rate / (blended * gpus_per_replica)))
            log(f"  {sc} (burst {burst:g}x, {len(t):,} requests on the busiest day)")

            def one(nr, t=t, tok=tok, bk=bk, sc=sc):
                return simulate_fcfs(nr, t, tok, bk, tps_rep, sc, maas_rps, budget, after,
                                     defer_deadline_s=deadline_s, offpeak_hours=offpeak,
                                     measure_until=(h0 + 24) * 3600.0, gpus_per_replica=gpus_per_replica)
            nr, day, feasible = min_fleet(one, max(1, guess // 4), guess, target_late, log=log)
            # Confirm on a full simulated week (queues that build up across days, e.g. deferred work, only
            # show there) and grow the fleet if the week fails; the week also gives the monthly cost.
            tw, tokw, bkw = sample_requests(cfg, 0, 168, sc, burst, seed=seed + 7, demand=demand, scale=scale)

            def one_week(k, tw=tw, tokw=tokw, bkw=bkw, sc=sc):
                return simulate_fcfs(k, tw, tokw, bkw, tps_rep, sc, maas_rps, budget, after,
                                     defer_deadline_s=deadline_s, offpeak_hours=offpeak,
                                     gpus_per_replica=gpus_per_replica)
            day_replicas = nr
            week = one_week(nr)
            if feasible and not passes(week, target_late):
                log(f"  {sc}: {nr * gpus_per_replica} GPUs pass the busiest day but not the full week; growing")
                nr, week, feasible = min_fleet(one_week, nr + 1, nr + 1, target_late, log=log)
            cost = monthly_cost(week, tiers[tier], cfg, all_requests=len(tw), all_tokens=float(tokw.sum()),
                                baseline_month=base["baseline"]["total"])
            row = {"scenario": sc, "burst": burst, "gpus": nr * gpus_per_replica, "replicas": nr,
                   "feasible": feasible, "hourly_model_gpus": hourly.get(sc), "day": day,
                   "day_gpus": day_replicas * gpus_per_replica,
                   "week": {k: v for k, v in week.items() if k not in ("replicas",)},
                   "tier": tier, **cost,
                   "gpu_month_by_tier": {k: nr * gpus_per_replica * p * costmodel.HOURS_PER_MONTH
                                         for k, p in tiers.items()}}
            if sc == "routed" and maas_rps <= 0:
                row["note"] = "MaaS limit is 0 req/s: long prompts stay on the fleet"
            rows.append(row)
            log(f"  -> {sc}: {row['gpus']} GPUs{'' if feasible else ' (does not pass)'}, "
                f"${row['total_month']:,.0f}/month")

    result = {
        "name": cfg.get("name"),
        "inputs": {"input_tps_per_gpu": per_gpu, "mix_factor_ref": f_ref, "mix_factor": f_now,
                   "blended_tps_per_gpu": blended, "bucket_tps_per_gpu": tps_gpu, "gpus_per_replica": gpus_per_replica,
                   "latency_budget_s": budget, "overflow_after_s": after, "maas_max_rps": maas_rps,
                   "target_late": target_late, "tier": tier, "gpu_hour": tiers[tier], "traffic_scale": scale,
                   "defer_deadline_h": defer_deadline_h, "sizing_window_start_h": h0,
                   "size_mix": dict(zip(BUCKETS, _size_mix(cfg)["share"].round(5).tolist())),
                   "effective_output_tokens": effective_output_tokens(cfg["requests"]),
                   "maas_only_feasible": base["maas_only_feasible"], "maas_peak_rps": base["maas_peak_rps"],
                   "baseline": base["baseline"]},
        "rows": rows,
        "notes": notes,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        (out_dir / "simulation.json").write_text(json.dumps(result, indent=1))
        write_summary(result, out_dir / "simulation.md")
    return result


def write_summary(result: dict, path: Path) -> Path:
    i = result["inputs"]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    h0 = i["sizing_window_start_h"]
    lines = [
        "# Request-level simulation: GPUs needed and monthly cost", "",
        f"Smallest fixed fleet with ≤ {i['target_late']:.0%} of short (≤ 32K) and ≤ {i['target_late']:.0%} of long "
        f"(> 32K) requests over the {i['latency_budget_s']:.0f} s latency budget on the busiest day "
        f"(24 h from {day_names[h0 // 24 % 7]} {h0 % 24:02d}:00 UTC). Poisson arrivals on the hourly curve, "
        f"request sizes from the workload's mix "
        f"(short {i['size_mix']['small']:.1%} / medium {i['size_mix']['medium']:.1%} / long {i['size_mix']['large']:.1%}), "
        f"first-come-first-served replicas, overflow to MaaS (≤ {i['maas_max_rps']:g} req/s) when a request would wait "
        f"more than {i['overflow_after_s']:.0f} s.", "",
        f"Throughput per GPU: {i.get('blended_tps_per_gpu', i['input_tps_per_gpu']):,.0f} input tok/s at the mix → short "
        f"{i['bucket_tps_per_gpu']['small']:,.0f} / medium {i['bucket_tps_per_gpu']['medium']:,.0f} / long "
        f"{i['bucket_tps_per_gpu']['large']:,.0f}. Monthly cost from a full simulated week at that fleet size: "
        f"GPUs at {costmodel.TIERS.get(i['tier'], i['tier'])} (${i['gpu_hour']:.2f}/GPU-hour) plus MaaS overflow. "
        "Burst = one 10-minute surge per hour at that multiple of the hourly average."
        + (f" Traffic scaled ×{i['traffic_scale']:g}." if i["traffic_scale"] != 1 else ""), "",
        "| Burst | Scenario | GPUs | Hourly model | GPU $/month | MaaS $/month | Total $/month | vs MaaS only "
        "| vs baseline | Peak MaaS req/s | Week util | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    bl = i.get("baseline")
    if bl:
        lines[-2:-2] = [("Baseline: " + bl["label"] + f", ${bl['total']:,.0f}/month"
                         + ("" if i.get("maas_only_feasible") else
                            f" (MaaS can't serve the peak: limit {i['maas_max_rps']:g} req/s vs "
                            f"{i.get('maas_peak_rps', 0):.1f} req/s)") + "."), ""]
    for r in result["rows"]:
        w = r["week"]
        notes = []
        if not r["feasible"]:
            notes.append("does not pass at any fleet size")
        elif r.get("day_gpus") and r["day_gpus"] != r["gpus"]:
            notes.append(f"busiest day alone: {r['day_gpus']} GPUs (backlog builds up over the week)")
        if r["scenario"] == "deferred" and "deferred_p99_wait_h" in w:
            notes.append(f"long prompts wait p99 {w['deferred_p99_wait_h']:.1f} h, "
                         f"{w.get('deferred_done_by_deadline', 1):.1%} within {i['defer_deadline_h']:g} h")
        if r["scenario"] == "routed" and "maas_wait_p99_s" in w:
            notes.append(f"MaaS slot wait p99 {w['maas_wait_p99_s']:.0f} s")
        if r.get("note"):
            notes.append(r["note"])
        notes.append(f"week late short {w['late_short']:.2%} / long {w['late_long']:.2%}")
        hm = r.get("hourly_model_gpus")
        lines.append(
            f"| {r['burst']:.1f}x | {SCENARIOS[r['scenario']]} | {r['gpus']} | {hm if hm is not None else '—'} | "
            f"${r['gpu_month']:,.0f} | ${r['maas_month']:,.0f} | ${r['total_month']:,.0f} | "
            f"{r.get('vs_maas_only', 0):.0%} | {r.get('vs_baseline', 0):.0%} | {w['maas_peak_rps']:.1f} | {w['util']:.0%} | {'; '.join(notes)} |")
    lines += [""] + [f"- {n}" for n in result.get("notes", [])]
    path.write_text("\n".join(lines) + "\n")
    return path
