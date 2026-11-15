"""Test-only stand-in for Agent Platform: loaded into a job process via DILEMMA_TEST_FAKE_BENCH=<this file>.

Replaces deploy/teardown/load (and the tokenizer-backed prompt build) with quick fakes, so out-of-process jobs run
in seconds without GCP. Knobs (env): FAKE_BENCH_DEPLOY_S (deploy time), FAKE_BENCH_TICK_S (time per progress
tick), FAKE_BENCH_REQS (requests per level), FAKE_BENCH_FAIL_AT (step name that raises), FAKE_BENCH_PREFLIGHT (pass | cut | 503: what the real pre-flight
check sees from the fake endpoint; it runs when `loadtest.preflight` is on, like the real loadtest.run). Every teardown is
appended to <run_dir>/fake_teardown.log so tests can check the endpoint was cleaned up.
"""
import json
import os
import time
from pathlib import Path

from aiops_dilemma import deploy, loadtest, planner, pricing, prompts

DEPLOY_S = float(os.environ.get("FAKE_BENCH_DEPLOY_S", "0.2"))
TICK_S = float(os.environ.get("FAKE_BENCH_TICK_S", "0.05"))
REQS = int(os.environ.get("FAKE_BENCH_REQS", "20"))
FAIL_AT = os.environ.get("FAKE_BENCH_FAIL_AT", "")
PREFLIGHT = os.environ.get("FAKE_BENCH_PREFLIGHT", "pass")
PREFLIGHT_S = float(os.environ.get("FAKE_BENCH_PREFLIGHT_S", "0"))   # wall time per pre-flight request (UI checks)


def fake_preflight_stream(url, headers, body, timeout_s):
    long_output = body.get("min_tokens")
    time.sleep(PREFLIGHT_S)
    base = {"status": 200, "ttft_s": 0.3, "prompt_tokens": 20 if long_output else 90000, "error": None}
    if PREFLIGHT == "cut" and long_output:
        return {**base, "duration_s": 31.0, "tokens": 4906, "finish_reason": None, "done": False,
                "error": "RemoteProtocolError('peer closed connection without sending complete message body')"}
    if PREFLIGHT == "503":
        return {**base, "status": 503, "duration_s": 0.4, "tokens": 0, "finish_reason": None, "done": False,
                "error": "no healthy upstream"}
    return {**base, "duration_s": 61.0 if long_output else 12.0, "tokens": body["max_tokens"],
            "finish_reason": "length" if long_output else "stop", "done": True}


def _maybe_fail(step):
    if FAIL_AT == step:
        raise RuntimeError(f"fake failure in {step}")


def fake_plan(cfg, **kw):
    _maybe_fail("plan")
    ch = {"machine_type": "g4-standard-48", "accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 1,
          "est_input_tps_per_gpu": 9000.0}
    ch["prices"] = pricing.fallback_prices_for(cfg.get("region", "us-central1"), ch["accelerator_type"])
    return {"chosen": ch, "candidates": [], "weights_gb": 26.0}


def fake_build(cfg, run_dir):
    _maybe_fail("prompts")
    if cfg["prompts"].get("source") == "file":   # your uploaded prompts: the real import (no tokenizer needed)
        return _real_build(cfg, run_dir)
    path = Path(run_dir) / "prompts.jsonl"
    path.write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"hi {i}"}], "max_tokens": 400,
                                        "prompt_tokens": 5000}) + "\n" for i in range(50)))
    return path


def fake_deploy(cfg, plan, run_dir, replicas=1):
    _maybe_fail("deploy")
    time.sleep(DEPLOY_S)
    ch = plan["chosen"]
    state = {"endpoint": "projects/my-gcp-project/locations/us-central1/endpoints/1234567890",
             "dedicated_dns": "fake.example", "region": cfg["region"],
             **{k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
             "precision": cfg["model"].get("precision", "bf16"), "replicas": replicas,
             "image": cfg["deployment"].get("image") or "us-docker.pkg.dev/vertex-ai/fake/pytorch-vllm-serve:verified",
             "args": ["--max-model-len=32768", "--gpu-memory-utilization=0.9"], "inference_timeout_s": 600}
    (Path(run_dir) / "endpoint.json").write_text(json.dumps(state, indent=2))
    return state


def fake_teardown(cfg, run_dir):
    _maybe_fail("teardown")
    path = Path(run_dir) / "endpoint.json"
    with open(Path(run_dir) / "fake_teardown.log", "a") as f:
        f.write(f"{time.time()} {'deleted' if path.exists() else 'none'}\n")
    if path.exists():
        path.unlink()


def fake_load(cfg, run_dir, concurrency=None, per_level=None, progress_cb=None, prompts_path=None, tag="",
              max_tokens=None):
    _maybe_fail("load")
    levels = concurrency or cfg["loadtest"]["concurrency"]
    out = Path(run_dir) / "load" / tag if tag else Path(run_dir) / "load"
    out.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb({"event": "waiting_ready", "waited_s": 0, "last": "HTTP 503: no healthy upstream"})
    if loadtest.preflight_enabled(cfg) and not tag:
        state = json.loads((Path(run_dir) / "endpoint.json").read_text())
        loadtest.preflight(loadtest.endpoint_url(state), cfg, Path(run_dir), progress_cb, image=state.get("image"))
    # The real per-level planning on the real prompt set (uploaded or fake), so repeats and size classes are reported
    path = Path(prompts_path) if prompts_path else Path(run_dir) / "prompts.jsonl"
    samples, info = None, {}
    if path.exists():
        pool = loadtest.load_prompts(Path(run_dir), path)
        samples, info = loadtest.plan_levels(pool, [REQS] * len(levels),
                                             cfg["model"].get("chat_template_kwargs") or {})
        if info.get("note"):
            print(f"note: {info['note']}", flush=True)
        (out / "levels_plan.json").write_text(json.dumps({**info, "prompts": str(path)}, indent=2))
    for li, c in enumerate(levels):
        n = REQS
        if progress_cb:
            progress_cb({"event": "level_start", "concurrency": c, "total": n})
        recs = []
        wall = 60.0
        for i in range(n):
            ptok = loadtest.prompt_tokens_estimate(samples[li][i]) if samples else 5000
            recs.append({"status": 200, "prompt_tokens": ptok, "completion_tokens": 400, "latency": 2.0 + c / 10,
                         "ttft": 0.5, "start": i * wall / n, "concurrency": c})
            if progress_cb:
                time.sleep(TICK_S)
                progress_cb({"event": "request_done", "concurrency": c, "completed": i + 1, "total": n,
                             "last_status": 200, "last_latency": 2.0})
        # throughput grows with concurrency, so the peak is the highest level
        for r in recs:
            r["start"] = r["start"] / c
        (out / f"c{c}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        s = loadtest.summarize(recs)
        s["concurrency"] = c
        if info.get("repeats"):
            s["sample_note"] = info["note"]
        if progress_cb:
            progress_cb({"event": "level_done", "concurrency": c, "summary": s})


_real_build = prompts.build
planner.plan = fake_plan
prompts.build = fake_build
deploy.deploy = fake_deploy
deploy.teardown = fake_teardown
loadtest.run = fake_load
loadtest._preflight_stream = fake_preflight_stream
loadtest.Auth = lambda: type("FakeAuth", (), {"token": lambda self: "fake-token"})()
