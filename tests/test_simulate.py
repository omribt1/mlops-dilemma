import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from aiops_dilemma import cli, config, simulate

ROOT = Path(__file__).resolve().parents[1]
PRICES = {"vertex_cud_3y": 2.5, "vertex": 5.85}
QUIET = lambda *a, **k: None  # noqa: E731


def _cfg(pct_medium=0.0, pct_long=0.0):
    raw = {"name": "sim-test", "project": "my-gcp-project",
           "model": {"model_garden_id": "x"},
           "traffic": {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500,
                       "weekend_factor": 0.08},
           "requests": {"avg_input_tokens": 7000, "avg_output_tokens": 950, "system_prompt_tokens": 500,
                        "system_prompt_cache_hit": 0.9, "pct_medium": pct_medium, "pct_long": pct_long},
           "maas": {"input_per_m": 0.15, "output_per_m": 0.60, "cached_per_m": 0.015, "max_rps": 5}}
    return config.normalize(raw)


TPS = {"small": 1000.0, "medium": 1000.0, "large": 1000.0}


def test_fcfs_conserves_work():
    # 10 requests of 10 s work arriving together on one replica: served back to back.
    t = np.arange(10) * 1e-6
    tok = np.full(10, 10_000.0)
    bk = np.zeros(10, dtype=np.int8)
    r = simulate.simulate_fcfs(1, t, tok, bk, TPS, "gpu", maas_rps=0, budget_s=1e9)
    assert r["max_s"] == pytest.approx(100.0, abs=1e-3)
    assert r["util"] == pytest.approx(1.0, rel=1e-6)
    # Same requests spread out: nobody waits.
    r = simulate.simulate_fcfs(1, np.arange(10) * 20.0, tok, bk, TPS, "gpu", maas_rps=0)
    assert r["max_s"] == pytest.approx(10.0)
    assert r["util"] == pytest.approx(0.5, rel=0.06)
    # Random traffic on several replicas: total busy time = total work (no MaaS), whatever the routing.
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 3600, 2000))
    tok = rng.uniform(500, 20000, 2000)
    bk = np.zeros(2000, dtype=np.int8)
    r = simulate.simulate_fcfs(4, t, tok, bk, TPS, "gpu", maas_rps=0)
    assert r["busy_replica_s"] == pytest.approx(tok.sum() / 1000.0)
    assert r["fleet_input_tokens"] == pytest.approx(tok.sum())
    # With MaaS overflow, fleet work + MaaS tokens still add up to everything that arrived.
    r = simulate.simulate_fcfs(1, t, tok, bk, TPS, "gpu", maas_rps=0.5, overflow_after_s=60)
    assert r["maas_requests"] > 0
    assert r["busy_replica_s"] * 1000.0 + r["maas_input_tokens"] == pytest.approx(tok.sum())


def test_overflow_respects_maas_rate_limit():
    t, tok, bk = simulate.sample_requests(_cfg(), 13, 15, seed=1)
    r = simulate.simulate_fcfs(1, t, tok, bk, simulate.bucket_tps(_cfg(), 11368), "gpu", maas_rps=5)
    assert r["maas_requests"] > 0
    assert r["maas_peak_rps"] <= 5 + 1e-9


def test_surge_keeps_hourly_mean():
    cfg = _cfg()
    t1, _, _ = simulate.sample_requests(cfg, 0, 168, burst=1.0, seed=2)
    t3, _, _ = simulate.sample_requests(cfg, 0, 168, burst=3.0, seed=2)
    assert len(t3) == pytest.approx(len(t1), rel=0.02)


@pytest.fixture(scope="module")
def long_heavy():
    cfg = _cfg(pct_medium=3.0, pct_long=1.0)   # ~4% of requests carry over half the tokens
    # Pin the long-prompt rates (1.75x / 2.4x slower) so these scenario properties don't move with the defaults:
    # with much slower long prompts, deferring them all to the nights can need *more* GPUs than keeping them.
    cfg["throughput"].update(medium_tps=11368 / 1.75, large_tps=11368 / 2.4)
    res = simulate.run(cfg, 11368, PRICES, scenarios=["gpu", "routed", "deferred"], scale=0.5, log=QUIET)
    return {r["scenario"]: r for r in res["rows"]}


def test_routed_needs_fewer_gpus(long_heavy):
    assert long_heavy["routed"]["feasible"] and long_heavy["gpu"]["feasible"]
    assert long_heavy["routed"]["gpus"] < long_heavy["gpu"]["gpus"]
    # Routed: every long prompt goes to MaaS, so the fleet carries fewer tokens than with all prompts on it.
    assert long_heavy["routed"]["week"]["fleet_input_tokens"] < long_heavy["gpu"]["week"]["fleet_input_tokens"]


def test_deferred_meets_deadline(long_heavy):
    day = long_heavy["deferred"]["day"]
    assert day["deferred_requests"] > 0
    assert day["deferred_done_by_deadline"] >= 0.999
    assert day["deferred_max_wait_h"] <= 24.0
    assert long_heavy["deferred"]["gpus"] <= long_heavy["gpu"]["gpus"]


def test_surges_need_more_gpus():
    cfg = _cfg()
    res = simulate.run(cfg, 11368, PRICES, scenarios=["gpu"], bursts=(1.0, 3.0), scale=0.5, log=QUIET)
    calm, surge = res["rows"]
    assert surge["gpus"] > calm["gpus"]


def test_cli_simulate_end_to_end(tmp_path, capsys):
    shutil.copy(ROOT / "tests/fixtures/startup-v1.yaml", tmp_path / "workload.yaml")
    rd = tmp_path / "runs" / "blog-example"
    rd.mkdir(parents=True)
    shutil.copy(ROOT / "blog/example/runs/blog-example/plan.json", rd / "plan.json")
    cli.main(["simulate", "-c", str(tmp_path / "workload.yaml"), "--tps-per-gpu", "9000", "--scale", "0.1",
              "--scenario", "gpu", "routed", "--burst", "1", "2", "--target-late", "0.01"])
    data = json.loads((rd / "simulation.json").read_text())
    assert len(data["rows"]) == 2          # no prompts > 32K: only the all-on-fleet scenario, per burst
    assert data["notes"] and {r["burst"] for r in data["rows"]} == {1.0, 2.0}
    assert all(r["gpus"] >= 1 and r["total_month"] > 0 for r in data["rows"])
    md = (rd / "simulation.md").read_text()
    assert "| Burst | Scenario | GPUs |" in md
    assert "simulation.md" in capsys.readouterr().out


def test_long_short_output_prompts(tmp_path):
    cfg = _cfg(pct_medium=3.0, pct_long=1.0)
    cfg["model"]["hf_id"] = None   # no tokenizer download in tests
    from aiops_dilemma import prompts
    main = prompts.synthetic(cfg, tmp_path / "prompts.jsonl")
    path = prompts.long_short_output(cfg, tmp_path, n_medium=3, n_large=2, max_tokens=400)
    rows = [json.loads(x) for x in open(path)]
    assert len(rows) == 5 and all(r["max_tokens"] == 400 for r in rows)
    assert all(r["target_input_tokens"] > 32000 for r in rows)
    main_users = {json.loads(x)["messages"][1]["content"][:200] for x in open(main)}
    assert not main_users & {r["messages"][1]["content"][:200] for r in rows}
