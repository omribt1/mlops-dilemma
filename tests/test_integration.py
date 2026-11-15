"""Integration glue: simulate endpoint, CLI estimate shape, tagged analyze."""

import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from aiops_dilemma import analyze, config, server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    server.build_example_session(runs / "blog-example")
    return TestClient(server.app)


def test_simulate_example_returns_without_saving(client):
    r = client.post("/api/sessions/blog-example/simulate", json={"bursts": [1], "scale": 0.1})
    assert r.status_code == 200
    body = r.json()
    assert body["example"] is True
    rows = body["simulation"]["rows"]
    assert rows and rows[0]["scenario"] == "gpu" and rows[0]["gpus"] > 0
    assert not (server.RUNS_DIR / "blog-example" / "simulation.json").exists()


def test_simulate_copy_saves_and_appears_in_payload(client):
    assert client.post("/api/sessions/sim-copy", json={"copy_of": "blog-example"}).status_code == 200
    r = client.post("/api/sessions/sim-copy/simulate", json={"bursts": [1, 2], "scale": 0.1})
    assert r.status_code == 200
    sim = r.json()["simulation"]
    assert {row["burst"] for row in sim["rows"]} == {1.0, 2.0}
    assert (server.RUNS_DIR / "sim-copy" / "simulation.json").exists()
    assert client.get("/api/sessions/sim-copy").json()["simulation"]["rows"]


def test_simulate_unknown_session_404(client):
    assert client.post("/api/sessions/nope/simulate", json={}).status_code == 404


def test_cli_estimate_writes_provenance(tmp_path):
    src = server.BLOG_EXAMPLE_DIR / "workload.yaml"
    cfg_path = tmp_path / "workload.yaml"
    cfg_path.write_text(src.read_text().replace("name: blog-example", "name: cli-est"))
    subprocess.run([sys.executable, "-m", "aiops_dilemma.cli", "estimate", "-c", str(cfg_path),
                    "--tps-per-gpu", "9000"], check=True, cwd=tmp_path, capture_output=True)
    cfg = config.load(str(cfg_path))
    m = json.loads((config.run_dir(cfg) / "measured.json").read_text())
    assert m["estimated"] is True and m["levels"] == [] and m["peak"] is None
    assert m["throughput_source"] == "override" and "set by you" in m["throughput_label"]


def test_analyze_tag_writes_separate_file(tmp_path):
    rd = tmp_path
    (rd / "endpoint.json").write_text(json.dumps({"machine_type": "g4-standard-48",
                                                   "accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 1}))
    tagged = rd / "load" / "longshort"
    tagged.mkdir(parents=True)
    recs = [{"status": 200, "prompt_tokens": 60000, "completion_tokens": 400, "latency": 30.0, "ttft": 20.0,
             "start": 0.0 + i, "end": 30.0 + i} for i in range(8)]
    (tagged / "c8.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    cfg = {"constraints": {"latency_budget_s": 600}}
    m = analyze.analyze(cfg, rd, tag="longshort")
    assert m["tag"] == "longshort" and m["input_tps_per_gpu"] > 0
    assert (rd / "measured_longshort.json").exists() and not (rd / "measured.json").exists()
