"""Turn load-test records into the number the cost model needs: input tokens/sec per GPU at your mix.

Peak = the highest-throughput concurrency level whose p95 latency fits your latency budget and
whose error rate is <= 10%. Because the load test used your own (or profile-matched) prompts, this
is measured at your input/output mix and cache behaviour — no extrapolation.
"""

import json
from pathlib import Path

from aiops_dilemma.costmodel import mix_factor
from aiops_dilemma.loadtest import summarize

MAX_ERROR_RATE = 0.10


def serving_provenance(state: dict) -> dict:
    """Serving image, final container args and inference timeout as recorded in endpoint.json (deploy.py)."""
    return {k: state[k] for k in ("image", "image_source", "args", "inference_timeout_s")
            if state.get(k) is not None}


def analyze(cfg: dict, run_dir: Path, tag: str = "") -> dict:
    """Peak throughput within the latency budget from `load/c*.jsonl` (or `load/<tag>/` for a tagged run).

    Untagged results go to measured.json (what reports and the UI use); a tagged run, e.g. long prompts with
    schema-sized outputs, goes to measured_<tag>.json so it never replaces the main measurement.
    """
    state = json.loads((run_dir / "endpoint.json").read_text()) if (run_dir / "endpoint.json").exists() \
        else json.loads((run_dir / "measured.json").read_text())["endpoint"]
    load_dir = run_dir / "load" / tag if tag else run_dir / "load"
    levels = []
    for path in sorted(load_dir.glob("c*.jsonl"), key=lambda p: int(p.stem[1:])):
        recs = [json.loads(line) for line in open(path)]
        s = summarize(recs)
        if s.get("ok"):
            s["concurrency"] = int(path.stem[1:])
            s["error_rate"] = s["errors"] / len(recs)
            levels.append(s)
    if not levels:
        raise SystemExit("no successful load-test levels found — run `dilemma load` first")
    clean = [s for s in levels if s["error_rate"] <= MAX_ERROR_RATE]
    if not clean:
        worst = ", ".join(f"c{s['concurrency']}: {s['error_rate']:.0%} errors {s.get('error_breakdown', {})}" for s in levels)
        raise SystemExit(f"every load-test level had more than {MAX_ERROR_RATE:.0%} failed requests ({worst}) — "
                         "the endpoint was unhealthy; fix the serving issue and rerun (see load/c*.jsonl)")
    ok = [s for s in clean if s["lat_p95"] <= cfg["constraints"]["latency_budget_s"]]
    peak = max(ok or clean, key=lambda s: s["input_tps"])
    gpus = state["accelerator_count"]
    measured = {
        "endpoint": state, "levels": levels, "peak": peak,
        "input_tps_per_gpu": peak["input_tps"] / gpus,
        "measured_out_ratio": peak["avg_out"] / peak["avg_in"],
        "within_budget": bool(ok),
    }
    serving = serving_provenance(state)
    if serving:
        measured["serving"] = serving
    pre = run_dir / "preflight.json"
    if pre.exists():
        try:
            measured["preflight"] = json.loads(pre.read_text())
        except ValueError:
            pass
    try:   # which prompts the run used: your uploaded set or synthetic ones (Benchmark tab, report.md)
        from aiops_dilemma import userprompts
        if not tag:
            measured["prompt_source"] = userprompts.describe_source(cfg, run_dir)
    except Exception:
        pass
    if tag:
        measured["tag"] = tag
    elif cfg.get("requests"):
        # The main load test's synthetic prompts follow the config's size mix, so the peak is a blended figure at
        # this mix factor (costmodel.ref_mix_factor); the cost model rescales it when the mix changes. Tagged runs
        # (e.g. long prompts only) use their own prompt set, so no config mix is recorded for them.
        measured["mix_factor"] = round(mix_factor(cfg), 6)
    (run_dir / (f"measured_{tag}.json" if tag else "measured.json")).write_text(json.dumps(measured, indent=2))
    print(f"peak: concurrency {peak['concurrency']} -> {peak['input_tps']:,.0f} input tok/s "
          f"({measured['input_tps_per_gpu']:,.0f} per GPU), p95 {peak['lat_p95']:.0f}s"
          + ("" if ok else "  [no level met the latency budget — using the best level]"))
    return measured


DEFAULT_SLOWDOWN = {"medium": 3.0, "large": 4.0}   # same shape as costmodel.DEFAULT_PENALTIES


def _bucket(tokens: float) -> str:
    return "large" if tokens > 128000 else "medium" if tokens > 32000 else "small"


def calibrate_classes(cfg: dict, run_dir: Path, short_tag: str = "short") -> dict:
    """Per-class throughput from two runs on the same endpoint: the main (mixed) run and a short-only run.

    Short prompts run at S (the short-only peak). In the mixed run's peak level, the token shares by size
    (t_small, t_medium, t_large, from the per-request prompt_tokens) and its blended rate T give
        1/T = t_small/S + k (3 t_medium + 4 t_large)/S
    so one long-prompt factor k scales the default slowdowns (two runs, two factors: S and k). Writes
    bucket_tps (input tok/s per chip by size) and a `calibration` record into measured.json.
    """
    path = run_dir / "measured.json"
    mixed = json.loads(path.read_text())
    short = json.loads((run_dir / f"measured_{short_tag}.json").read_text())
    if mixed.get("estimated") or not mixed.get("levels"):
        raise SystemExit("measured.json is not a load test — run the main benchmark first")
    gpus = mixed["endpoint"]["accelerator_count"]
    peak_c = mixed["peak"]["concurrency"]
    recs = [json.loads(line) for line in open(run_dir / "load" / f"c{peak_c}.jsonl")]
    tok = {"small": 0.0, "medium": 0.0, "large": 0.0}
    for r in recs:
        if r.get("status") == 200 and r.get("prompt_tokens"):
            tok[_bucket(r["prompt_tokens"])] += r["prompt_tokens"]
    total = sum(tok.values())
    shares = {k: v / total for k, v in tok.items()} if total else tok
    S = float(short["input_tps_per_gpu"])
    T = float(mixed["input_tps_per_gpu"])
    long_w = DEFAULT_SLOWDOWN["medium"] * shares["medium"] + DEFAULT_SLOWDOWN["large"] * shares["large"]
    if long_w <= 0:
        raise SystemExit(f"the mixed run's peak level (concurrency {peak_c}) has no prompts over 32K — "
                         "nothing to calibrate (raise pct_over_32k or requests_per_level)")
    k = (S / T - shares["small"]) / long_w
    pen = {c: max(1.0, k * DEFAULT_SLOWDOWN[c]) for c in DEFAULT_SLOWDOWN}
    bucket_tps = {"small": S, "medium": S / pen["medium"], "large": S / pen["large"]}
    mixed["bucket_tps"] = bucket_tps
    mixed["calibration"] = {
        "method": "two runs: short-only peak + mixed peak, one long-prompt factor",
        "short_tps_per_gpu": S, "mixed_tps_per_gpu": T, "token_shares": shares, "long_factor": k,
        "slowdown": pen, "short_tag": short_tag, "mixed_peak_concurrency": peak_c,
        "short_peak_concurrency": short["peak"]["concurrency"], "gpus": gpus,
        "note": ("the mixed run was faster than short prompts alone would predict; long prompts floored at "
                 "short-prompt speed") if k * DEFAULT_SLOWDOWN["medium"] < 1 else None,
    }
    path.write_text(json.dumps(mixed, indent=2))
    print(f"per-class calibration: short {S:,.0f} tok/s/chip, mixed {T:,.0f}; token shares "
          f"{shares['small']:.0%} / {shares['medium']:.0%} / {shares['large']:.0%}; "
          f"32K-128K {pen['medium']:.2f}x slower ({bucket_tps['medium']:,.0f}), "
          f">128K {pen['large']:.2f}x slower ({bucket_tps['large']:,.0f})")
    return mixed
