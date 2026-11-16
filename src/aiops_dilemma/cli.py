"""dilemma — benchmark an open model on Agent Platform and find when self-hosting pays off vs MaaS.

  dilemma ui       [--port 8080] [--open]        launch the interactive web application
  dilemma init     [--auto] [-o workload.yaml]   write a starter workload definition
  dilemma plan     -c CONFIG                     size the model, rank GPU/TPU configs, pick one
  dilemma prompts  -c CONFIG                     build the prompt set (yours, or synthetic)
  dilemma deploy   -c CONFIG                     deploy a 1-replica Agent Platform endpoint
  dilemma load     -c CONFIG [--concurrency ...] closed-loop load test (starts with the pre-flight check)
  dilemma preflight -c CONFIG                    pre-flight check alone on the deployed endpoint (long stream + long prompt)
  dilemma analyze  -c CONFIG                     peak throughput per GPU/TPU within your latency budget
  dilemma calibrate -c CONFIG                    short-only run on the same endpoint -> per-class throughput (long prompts)
  dilemma report   -c CONFIG                     cost model + report.md + dashboard.html
  dilemma estimate -c CONFIG [--tps-per-gpu N]   report without a benchmark (what-if)
  dilemma simulate -c CONFIG [--scenario ...] [--burst ...] [--target-late 0.01]
                                                 request-level check: min fleet + monthly cost per scenario
  dilemma teardown -c CONFIG                     undeploy and delete the endpoint
  dilemma auto     -c CONFIG [--keep]            plan -> prompts -> deploy -> load -> analyze -> report -> teardown
  dilemma run-job  -c CONFIG --steps ...         run benchmark steps as a durable job (what the web app launches)
  dilemma reference-entry -c CONFIG              print the catalog.REFERENCE_BENCHMARKS entry of a reference run

deploy, load, calibrate, teardown, auto and run-job record their progress in runs/<name>/job.json and
job_events.jsonl, so the web app shows them live. SIGTERM or Ctrl-C cancels at the next safe point and tears down
an endpoint the job deployed.
"""

import argparse
import json
import shutil
import sys
import webbrowser
from pathlib import Path

from aiops_dilemma import config


def _plan(cfg, rd, refresh=False):
    from aiops_dilemma import planner
    path = rd / "plan.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    p = planner.plan(cfg)
    path.write_text(json.dumps(p, indent=2, default=str))
    print(planner.format_plan(p))
    return p


def _prices(cfg, plan):
    from aiops_dilemma import pricing
    ch = plan["chosen"]
    return ch.get("prices") or pricing.prices_for(cfg, ch["accelerator_type"], ch["accelerator_count"])


def cmd_init(a):
    src = Path(__file__).resolve().parents[2] / "examples" / ("auto.yaml" if a.auto else "workload.yaml")
    shutil.copy(src, a.output)
    print(f"wrote {a.output} — edit project, model and traffic, then: dilemma {'auto' if a.auto else 'plan'} -c {a.output}")


def cmd_ui(a):
    import uvicorn
    from aiops_dilemma.server import app
    url = f"http://{a.host}:{a.port}"
    print(f"Starting mlops-dilemma interactive UI at {url}")
    if a.open:
        webbrowser.open(url)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


def cmd_plan(a, cfg, rd):
    _plan(cfg, rd, refresh=True)


def cmd_prompts(a, cfg, rd):
    from aiops_dilemma import prompts
    if getattr(a, "long_short_output", False):
        print(f"wrote {prompts.long_short_output(cfg, rd, a.n_medium, a.n_large, a.max_tokens)}")
        return
    print(f"wrote {prompts.build(cfg, rd)}")


def _job(a, cfg, rd, steps, mode="step", **kw):
    """Run benchmark steps as a durable job (job.json + job_events.jsonl); exit non-zero unless it completed."""
    from aiops_dilemma import jobs
    status = jobs.run_job(cfg, rd, steps, mode=mode, source=getattr(a, "source", "cli"),
                          command="dilemma " + " ".join(sys.argv[1:]), **kw)
    if status != "completed":
        job = jobs.read_job(rd, mark=False) or {}
        raise SystemExit(f"job {status}" + (f": {job['error']}" if job.get("error") else ""))


def cmd_deploy(a, cfg, rd):
    _job(a, cfg, rd, ["deploy"])


def cmd_load(a, cfg, rd):
    _job(a, cfg, rd, ["load"], concurrency=a.concurrency, requests_per_level=a.requests,
         load_kwargs={"prompts_path": getattr(a, "prompts", None), "tag": getattr(a, "tag", "") or "",
                      "max_tokens": getattr(a, "max_tokens", None)})


def cmd_analyze(a, cfg, rd):
    from aiops_dilemma import analyze
    analyze.analyze(cfg, rd, tag=a.tag)


def cmd_report(a, cfg, rd):
    from aiops_dilemma import dashboard, report
    plan = _plan(cfg, rd)
    measured = json.loads((rd / "measured.json").read_text())
    prices = _prices(cfg, plan)
    print(f"wrote {report.write(cfg, plan, measured, prices, rd)}")
    print(f"wrote {dashboard.build(cfg, measured, prices, rd)}")
    print("\n" + (rd / "report.md").read_text().split("## Benchmark")[0])


def cmd_estimate(a, cfg, rd):
    plan = _plan(cfg, rd)
    ch = plan["chosen"]
    from aiops_dilemma import report
    tps = a.tps_per_gpu or ch.get("est_input_tps_per_gpu") or 8000.0
    source = "override" if a.tps_per_gpu else "estimated"
    # Same shape as the UI's instant estimate: no sweep, no peak, explicit provenance.
    measured = {"endpoint": {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
                "levels": [], "peak": None, "input_tps_per_gpu": tps, "measured_out_ratio": None,
                "estimated": True, "throughput_source": source,
                "throughput_label": report.throughput_label(tps, source)}
    (rd / "measured.json").write_text(json.dumps(measured, indent=2))
    print(f"(estimate mode: assuming {tps:,.0f} input tok/s per chip for short prompts — benchmark to replace this)")
    cmd_report(a, cfg, rd)


def _tps_per_gpu(a, cfg, rd, plan) -> tuple[float, dict | None, str]:
    if a.tps_per_gpu:
        return a.tps_per_gpu, None, "--tps-per-gpu"
    path = rd / "measured.json"
    if path.exists():
        m = json.loads(path.read_text())
        if m.get("input_tps_per_gpu"):
            return float(m["input_tps_per_gpu"]), m, "measured.json" + (" (estimate)" if m.get("estimated") else "")
    est = plan["chosen"].get("est_input_tps_per_gpu")
    return (float(est), None, "planner estimate") if est else (8000.0, None, "default")


def cmd_simulate(a, cfg, rd):
    from aiops_dilemma import simulate
    plan = _plan(cfg, rd)
    from aiops_dilemma import costmodel
    tps, measured, src = _tps_per_gpu(a, cfg, rd, plan)
    f_ref = costmodel.ref_mix_factor(cfg, measured)   # 1.0 for spec estimates / --tps-per-gpu (short-prompt rates)
    print(f"simulating with {tps:,.0f} input tok/s per GPU ({src}, mix factor {f_ref:.2f}), "
          f"{plan['chosen']['accelerator_count']} GPU(s) per replica")
    simulate.run(cfg, tps, _prices(cfg, plan), gpus_per_replica=int(plan["chosen"]["accelerator_count"]),
                 scenarios=a.scenario, bursts=a.burst, target_late=a.target_late, tier=a.tier, scale=a.scale,
                 defer_deadline_h=a.defer_deadline_h, measured=measured, out_dir=rd,
                 mix_factor_ref=f_ref)
    print(f"wrote {rd / 'simulation.json'} and {rd / 'simulation.md'}\n")
    print((rd / "simulation.md").read_text())


def cmd_teardown(a, cfg, rd):
    _job(a, cfg, rd, ["teardown"])


def calibration_levels(cfg: dict, measured: dict) -> list[int]:
    """Concurrency levels for the short-only run: the main run's levels from half its peak upwards."""
    peak_c = int(measured["peak"]["concurrency"])
    levels = [int(c) for c in cfg["loadtest"]["concurrency"] if int(c) >= peak_c / 2]
    return levels or [peak_c]


def run_calibration(cfg: dict, rd, concurrency=None, per_level=None, progress_cb=None) -> dict:
    """Short-only load test on the same endpoint + per-class calibration of measured.json."""
    from aiops_dilemma import analyze, loadtest, prompts
    measured = json.loads((rd / "measured.json").read_text())
    path = prompts.short_only(cfg, rd)
    loadtest.run(cfg, rd, concurrency or calibration_levels(cfg, measured), per_level, progress_cb=progress_cb,
                 prompts_path=path, tag="short")
    analyze.analyze(cfg, rd, tag="short")
    return analyze.calibrate_classes(cfg, rd, short_tag="short")


def needs_calibration(cfg: dict, rd=None) -> bool:
    """Per-class calibration needs prompts over 32K: your uploaded prompts (when prompts.source is file), else the
    workload's share over 32K."""
    if rd is not None and (cfg.get("prompts") or {}).get("source") == "file":
        from aiops_dilemma import userprompts
        return bool(userprompts.file_has_long_prompts(cfg, rd))
    return float(cfg["requests"].get("pct_over_32k") or 0) > 0


def cmd_preflight(a, cfg, rd):
    """Pre-flight check alone against the endpoint in endpoint.json (e.g. to re-test a serving image)."""
    from aiops_dilemma import loadtest
    res = loadtest.run_preflight(cfg, rd, wait=not a.no_wait)
    print(json.dumps({k: res[k] for k in ("passed", "message", "duration_s")}, indent=2))


def cmd_calibrate(a, cfg, rd):
    _job(a, cfg, rd, ["calibrate", "report"], requests_per_level=a.requests, calibrate_concurrency=a.concurrency)


def cmd_reference_entry(a, cfg, rd):
    """Print the catalog.REFERENCE_BENCHMARKS entry for a finished reference-workload run."""
    from aiops_dilemma import planner
    entry, warnings = planner.reference_entry(cfg, rd)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps(entry, indent=4))


def cmd_auto(a, cfg, rd):
    from aiops_dilemma.jobs import AUTO_STEPS
    try:
        _job(a, cfg, rd, AUTO_STEPS, mode="auto", keep_endpoint=a.keep, concurrency=a.concurrency,
             requests_per_level=a.requests)
    finally:
        if a.keep and (rd / "endpoint.json").exists():
            print(f"endpoint kept (billing continues) — `dilemma teardown -c {a.config}` when done")


def cmd_run_job(a, cfg, rd):
    _job(a, cfg, rd, a.steps, mode=a.mode, keep_endpoint=a.keep, concurrency=a.concurrency,
         requests_per_level=a.requests)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dilemma", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ui = sub.add_parser("ui", aliases=["serve"])
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8080)
    p_ui.add_argument("--open", action="store_true", help="open browser automatically")
    p = sub.add_parser("init"); p.add_argument("--auto", action="store_true"); p.add_argument("-o", "--output", default="workload.yaml")
    for name in ("plan", "prompts", "deploy", "load", "analyze", "calibrate", "report", "estimate", "simulate", "teardown",
                 "auto", "run-job", "preflight", "reference-entry"):
        p = sub.add_parser(name)
        p.add_argument("-c", "--config", required=True)
        p.add_argument("--run-dir", default=None, help="session directory (default: derived from the config)")
        if name in ("load", "auto", "calibrate", "run-job"):
            p.add_argument("--concurrency", type=int, nargs="+")
            p.add_argument("--requests", type=int, help="requests per concurrency level")
        if name in ("estimate", "simulate"):
            p.add_argument("--tps-per-gpu", type=float, default=None, help="assumed input tokens/sec per GPU/TPU")
        if name == "prompts":
            p.add_argument("--long-short-output", action="store_true",
                           help="write long prompts (>32K) with schema-sized outputs to prompts_long_shortout.jsonl")
            p.add_argument("--n-medium", type=int, default=100, help="32K-128K prompts (with --long-short-output)")
            p.add_argument("--n-large", type=int, default=50, help=">128K prompts (with --long-short-output)")
            p.add_argument("--max-tokens", type=int, default=400, help="output cap (with --long-short-output)")
        if name == "analyze":
            p.add_argument("--tag", default="", help="analyze load/<tag>/ (a tagged `dilemma load` run) into measured_<tag>.json")
        if name == "load":
            p.add_argument("--prompts", default=None, help="prompt JSONL to use instead of runs/<name>/prompts.jsonl")
            p.add_argument("--tag", default="", help="write results to load/<tag>/ so reruns don't overwrite earlier ones")
            p.add_argument("--max-tokens", type=int, default=None, help="override every prompt's max_tokens")
        if name == "simulate":
            from aiops_dilemma.simulate import SCENARIOS
            p.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
            p.add_argument("--burst", type=float, nargs="+", default=[1.0],
                           help="10-minute surge per hour at this multiple of the hourly average (1 = none)")
            p.add_argument("--target-late", type=float, default=0.01,
                           help="max share of short and of long requests over the latency budget")
            p.add_argument("--tier", default="vertex_cud_3y", help="GPU price tier for the monthly cost")
            p.add_argument("--scale", type=float, default=1.0, help="scale traffic (e.g. 0.1 for a quick run)")
            p.add_argument("--defer-deadline-h", type=float, default=24.0,
                           help="deferred scenario: each long prompt must finish within this many hours")
        if name == "preflight":
            p.add_argument("--no-wait", action="store_true", help="don't wait for the endpoint to answer first")
        if name in ("auto", "run-job"):
            p.add_argument("--keep", action="store_true", help="keep the endpoint after the run")
        if name == "run-job":
            from aiops_dilemma.jobs import AUTO_STEPS, RUN_MODES
            p.add_argument("--steps", nargs="+", choices=AUTO_STEPS, required=True)
            p.add_argument("--mode", default="step", choices=sorted(RUN_MODES),
                           help="auto: --keep skips the final teardown")
            p.add_argument("--source", default="cli", choices=["cli", "app"], help="who started the job")
    a = ap.parse_args(argv)
    if a.cmd in ("ui", "serve"):
        return cmd_ui(a)
    if a.cmd == "init":
        return cmd_init(a)
    cfg = config.load(a.config)
    if a.run_dir:
        cfg["_run_dir"] = a.run_dir
    rd = config.run_dir(cfg)
    globals()[f"cmd_{a.cmd.replace('-', '_')}"](a, cfg, rd)


if __name__ == "__main__":
    sys.exit(main())
