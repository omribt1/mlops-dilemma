"""Per-class calibration from a mixed run + a short-only run on the same endpoint (synthetic records)."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from aiops_dilemma import analyze, config, costmodel, prompts

ROOT = Path(__file__).resolve().parents[1]
EP = {"machine_type": "g4-standard-48", "accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 1}


def _records(sizes, wall_s, out=400):
    """Requests spread evenly over `wall_s` seconds, so input tok/s = sum(sizes) / wall_s."""
    n = len(sizes)
    return [{"status": 200, "prompt_tokens": s, "completion_tokens": out, "latency": 5.0, "ttft": 1.0,
             "start": i * (wall_s - 5.0) / max(1, n - 1)} for i, s in enumerate(sizes)]


def _write(d: Path, recs):
    d.mkdir(parents=True, exist_ok=True)
    (d / "c64.jsonl").write_text("\n".join(json.dumps(r) for r in recs))


def _cfg():
    return {"constraints": {"latency_budget_s": 600},
            "requests": {"avg_input_tokens": 5000, "avg_output_tokens": 500, "pct_over_32k": 2.0, "pct_over_128k": 0.5,
                         "medium_mean": 60000, "large_mean": 170000, "system_prompt_tokens": 1800,
                         "system_prompt_cache_hit": 0.9}}


def test_two_run_solve_recovers_slowdowns(tmp_path):
    """Construct runs from known class rates (S=10K, 4x / 8x slower) and check the calibration finds them."""
    S, pm, pl = 10000.0, 4.0, 8.0
    sizes = [3000] * 970 + [60000] * 24 + [170000] * 6          # token mix of the mixed run
    gpu_s = sum(s / S * (pm if 32000 < s <= 128000 else pl if s > 128000 else 1) for s in sizes)
    (tmp_path / "endpoint.json").write_text(json.dumps(EP))
    _write(tmp_path / "load", _records(sizes, gpu_s))              # one GPU busy for gpu_s seconds
    _write(tmp_path / "load" / "short", _records([3000] * 500, 3000 * 500 / S))
    cfg = _cfg()
    analyze.analyze(cfg, tmp_path)
    analyze.analyze(cfg, tmp_path, tag="short")
    m = analyze.calibrate_classes(cfg, tmp_path)
    cal = m["calibration"]
    assert cal["short_tps_per_gpu"] == pytest.approx(S, rel=1e-6)
    # one factor k scales the default 3 / 4 ratio, so it reproduces the blended rate exactly
    t = cal["token_shares"]
    T = 1 / (t["small"] / m["bucket_tps"]["small"] + t["medium"] / m["bucket_tps"]["medium"]
             + t["large"] / m["bucket_tps"]["large"])
    assert T == pytest.approx(m["input_tps_per_gpu"], rel=1e-6)
    assert 3.0 < cal["slowdown"]["medium"] < 6.0 and cal["slowdown"]["large"] > cal["slowdown"]["medium"]
    # the cost model then reads the measured file as a blend with per-class rates
    full = config.normalize(yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text()))
    full["requests"].update(_cfg()["requests"])
    assert costmodel.ref_mix_factor(full, m) == pytest.approx(S / m["input_tps_per_gpu"], rel=1e-9)
    c2 = costmodel.with_measured_classes(full, m)
    assert c2["throughput"]["medium_tps"] == pytest.approx(m["bucket_tps"]["medium"])


def test_no_long_prompts_at_peak_is_an_error(tmp_path):
    (tmp_path / "endpoint.json").write_text(json.dumps(EP))
    _write(tmp_path / "load", _records([3000] * 200, 60))
    _write(tmp_path / "load" / "short", _records([3000] * 200, 60))
    cfg = _cfg()
    analyze.analyze(cfg, tmp_path)
    analyze.analyze(cfg, tmp_path, tag="short")
    with pytest.raises(SystemExit, match="no prompts over 32K"):
        analyze.calibrate_classes(cfg, tmp_path)


def test_short_only_prompt_set(tmp_path):
    raw = yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text())
    raw["requests"].update(pct_over_32k=3, pct_over_128k=1)
    raw["prompts"] = {"source": "synthetic", "synthetic": {"count": 40}}
    raw["model"]["hf_id"] = None          # offline: character-based token estimate
    cfg = config.normalize(raw)
    path = prompts.short_only(cfg, tmp_path, n=40)
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 40 and all(r["id"].startswith("s") for r in rows)
    longest = max(sum(len(m["content"]) for m in r["messages"]) for r in rows)
    assert longest / 3.0 < 32000          # well under 32K tokens even at ~3 chars per token


def test_wait_ready_polls_until_200(monkeypatch):
    from aiops_dilemma import loadtest

    class R:
        def __init__(self, code): self.status_code, self.text = code, "no healthy upstream"
    codes = iter([503, 503, 200])
    monkeypatch.setattr(loadtest.httpx, "post", lambda *a, **k: R(next(codes)))
    monkeypatch.setattr(loadtest, "Auth", lambda: type("A", (), {"token": lambda self: "t"})())
    monkeypatch.setattr(loadtest.time, "sleep", lambda s: None)
    seen = []
    loadtest.wait_ready("https://x/v1beta1/e/chat/completions", progress_cb=seen.append)
    assert len(seen) == 2 and seen[0]["event"] == "waiting_ready" and "503" in seen[0]["last"]


def test_wait_ready_gives_up(monkeypatch):
    from aiops_dilemma import loadtest

    class R:
        status_code, text = 503, "no healthy upstream"
    monkeypatch.setattr(loadtest.httpx, "post", lambda *a, **k: R())
    monkeypatch.setattr(loadtest, "Auth", lambda: type("A", (), {"token": lambda self: "t"})())
    monkeypatch.setattr(loadtest.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="not ready"):
        loadtest.wait_ready("https://x", timeout_s=-1)


def test_inference_timeout_defaults():
    from aiops_dilemma import deploy
    assert deploy.inference_timeout_s({"constraints": {"latency_budget_s": 600}}) == 600
    assert deploy.inference_timeout_s({"constraints": {"latency_budget_s": 30}}) == 300      # never below 5 min
    assert deploy.inference_timeout_s({"deployment": {"inference_timeout_s": 99999}}) == 3600
