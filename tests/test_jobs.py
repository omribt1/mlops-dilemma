"""Durable out-of-process benchmark jobs: job.json / job_events.jsonl lifecycle, restart, stop, interrupted jobs.

Jobs run as real subprocesses (`dilemma run-job`) with tests/fake_bench.py standing in for Agent Platform, so no GCP
is touched and a whole run takes a couple of seconds.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aiops_dilemma import config, jobs, server

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "tests" / "fake_bench.py")
CFG = {
    "project": "my-gcp-project", "region": "us-central1",
    "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": None, "precision": "fp8"},
    "traffic": {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500},
    "requests": {"avg_input_tokens": 7000, "avg_output_tokens": 950},
    "loadtest": {"concurrency": [1, 2], "requests_per_level": 20},
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    monkeypatch.setenv("DILEMMA_TEST_FAKE_BENCH", FAKE)
    monkeypatch.setenv("FAKE_BENCH_TICK_S", "0.01")
    server._CHILDREN.clear()
    yield TestClient(server.app)
    for proc in list(server._CHILDREN.values()):      # never leave a job process behind
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    server._CHILDREN.clear()


def _start(client, name="j1", **payload):
    r = client.post(f"/api/sessions/{name}/run", json={"config": json.loads(json.dumps(CFG)), "mode": "auto",
                                                       **payload})
    assert r.status_code == 200, r.text
    return r.json()


def _wait(client, name, pred, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/sessions/{name}").json()["job"]
        if pred(job):
            return job
        time.sleep(0.1)
    raise AssertionError(f"timed out; last job state: {job}")


def _events(rd: Path) -> list[dict]:
    return [json.loads(line) for line in (rd / jobs.EVENTS_FILE).read_text().splitlines() if line.strip()]


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_app_job_lifecycle_on_disk(client):
    out = _start(client)
    assert out["job"]["steps"] == jobs.AUTO_STEPS and out["job"]["pid"]
    rd = server.RUNS_DIR / "j1"
    first = client.get("/api/sessions/j1").json()["job"]
    assert first["status"] in ("running", "completed") and first["pid"] == out["job"]["pid"]
    assert first["source"] == "app"
    job = _wait(client, "j1", lambda j: j["status"] != "running")
    assert job["status"] == "completed", job
    assert job["stage"] == "done" and job["finished_at"] >= job["started_at"]
    assert [s["concurrency"] for s in job["levels_completed"]] == [1, 2]
    assert job["endpoint"]["endpoint"].endswith("/1234567890") and not job["endpoint_warning"]
    types = [e["type"] for e in _events(rd)]
    for t in ("stage", "plan_done", "prompts_done", "deploy_done", "waiting_ready", "preflight_start", "preflight_done",
              "level_start", "request_done",
              "level_done", "analyze_done", "report_done", "teardown_done", "complete"):
        assert t in types, t
    assert types[-1] == "complete"
    # request_done is throttled to about one per second (plus the last of each level)
    assert types.count("request_done") <= 6
    assert (rd / "measured.json").exists() and (rd / "report.md").exists() and not (rd / "endpoint.json").exists()
    assert (rd / jobs.LOG_FILE).exists()
    raw = json.loads((rd / jobs.JOB_FILE).read_text())
    assert raw["pid_start"] and raw["status"] == "completed"
    sessions = {s["name"]: s for s in client.get("/api/sessions").json()["sessions"]}
    assert sessions["j1"]["job_status"] == "completed"


def test_concurrent_start_is_409_and_stop_cancels_with_teardown(client, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_TICK_S", "0.2")
    _start(client)
    rd = server.RUNS_DIR / "j1"
    r = client.post("/api/sessions/j1/run", json={"mode": "auto"})
    assert r.status_code == 409
    _wait(client, "j1", lambda j: j["stage"] == "load" and (rd / "endpoint.json").exists())
    r = client.post("/api/sessions/j1/stop", json={})
    assert r.json()["signal"] == "SIGTERM"
    stop = json.loads((rd / jobs.STOP_FILE).read_text())
    assert stop["pid"] == r.json()["job"]["pid"] and stop["requested_at"]
    job = _wait(client, "j1", lambda j: j["status"] != "running", timeout=20)
    assert job["status"] == "cancelled", job
    assert not (rd / "endpoint.json").exists()
    assert "deleted" in (rd / "fake_teardown.log").read_text()
    types = [e["type"] for e in _events(rd)]
    assert "cancelling" in types and types[-1] == "cancelled"
    assert not job["endpoint_warning"]


def test_failed_step_tears_down_the_endpoint_it_deployed(client, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_FAIL_AT", "load")
    _start(client)
    job = _wait(client, "j1", lambda j: j["status"] != "running")
    rd = server.RUNS_DIR / "j1"
    assert job["status"] == "error" and "fake failure in load" in job["error"]
    assert not (rd / "endpoint.json").exists() and "deleted" in (rd / "fake_teardown.log").read_text()


def test_failed_preflight_stops_before_the_sweep_and_tears_down(client, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_PREFLIGHT", "cut")
    _start(client)
    job = _wait(client, "j1", lambda j: j["status"] != "running")
    rd = server.RUNS_DIR / "j1"
    assert job["status"] == "error" and job["stage"] == "load"
    assert job["error"].startswith("Pre-flight failed: the endpoint cut a streaming request after 31 s (4,906 tokens)")
    assert "pytorch-vllm-serve:verified" in job["error"]
    assert job["preflight"]["status"] == "failed" and job["levels_completed"] == []
    assert not (rd / "endpoint.json").exists() and "deleted" in (rd / "fake_teardown.log").read_text()
    evs = _events(rd)
    types = [e["type"] for e in evs]
    assert types.index("preflight_start") < types.index("preflight_done") < types.index("teardown_done")
    assert "level_start" not in types and types[-1] == "error"
    done = next(e for e in evs if e["type"] == "preflight_done")
    assert done["passed"] is False and done["durations"] == {"long_output": 31.0}
    assert json.loads((rd / "preflight.json").read_text())["passed"] is False


def test_preflight_pass_recorded_in_job_and_measured(client):
    _start(client)
    job = _wait(client, "j1", lambda j: j["status"] != "running")
    assert job["status"] == "completed" and job["preflight"]["status"] == "passed"
    assert set(job["preflight"]["durations"]) == {"long_output", "long_prompt"}
    measured = json.loads((server.RUNS_DIR / "j1" / "measured.json").read_text())
    assert measured["preflight"]["passed"] is True
    assert measured["serving"]["image"].endswith("pytorch-vllm-serve:verified") and measured["serving"]["args"]


def test_server_restart_reattaches_and_streams_from_file(client, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_TICK_S", "0.05")
    _start(client)
    _wait(client, "j1", lambda j: j["stage"] in ("deploy", "load"))
    # "Restart": a new app client with no in-memory knowledge of the child process.
    children = dict(server._CHILDREN)
    server._CHILDREN.clear()
    fresh = TestClient(server.app)
    job = fresh.get("/api/sessions/j1").json()["job"]
    assert job["status"] == "running" and job["source"] == "app"
    sessions = {s["name"]: s for s in fresh.get("/api/sessions").json()["sessions"]}
    assert sessions["j1"]["job_status"] == "running"
    with fresh.stream("GET", "/api/sessions/j1/events") as r:
        data = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    types = [e["type"] for e in data]
    assert types[0] == "stage" and types[-1] == "complete" and "level_done" in types
    server._CHILDREN.update(children)


def test_dead_pid_is_interrupted_with_endpoint_warning(client):
    rd = server.RUNS_DIR / "ghost"
    rd.mkdir()
    (rd / "workload.yaml").write_text(config.to_yaml(config.normalize({**CFG, "name": "ghost"})))
    (rd / "endpoint.json").write_text(json.dumps({"endpoint": "projects/my-gcp-project/locations/us-central1/"
                                                              "endpoints/1", "region": "us-central1"}))
    jobs.atomic_write_json(rd / jobs.JOB_FILE, {"status": "running", "pid": _dead_pid(), "stage": "load",
                                                "started_at": time.time() - 60, "steps": jobs.AUTO_STEPS,
                                                "source": "cli", "logs": []})
    (rd / jobs.EVENTS_FILE).write_text(json.dumps({"ts": 1, "type": "stage", "stage": "load"}) + "\n")
    with TestClient(server.app) as c:            # runs the startup scan
        assert json.loads((rd / jobs.JOB_FILE).read_text())["status"] == "interrupted"
        job = c.get("/api/sessions/ghost").json()["job"]
        assert job["status"] == "interrupted" and job["endpoint_warning"] is True
        assert "stopped unexpectedly" in job["error"]
        s = {x["name"]: x for x in c.get("/api/sessions").json()["sessions"]}["ghost"]
        assert s["job_status"] == "interrupted" and s["job_endpoint_warning"] is True
        with c.stream("GET", "/api/sessions/ghost/events") as r:
            types = [json.loads(line[6:])["type"] for line in r.iter_lines() if line.startswith("data: ")]
        assert types == ["stage", "interrupted"]
        # an attached (pre-existing) endpoint doesn't bill because of us: no warning
        (rd / "endpoint.json").write_text(json.dumps({"endpoint": "e", "attached_existing": True}))
        assert c.get("/api/sessions/ghost").json()["job"]["endpoint_warning"] is False


def test_pid_reuse_counts_as_dead(tmp_path):
    jobs.atomic_write_json(tmp_path / jobs.JOB_FILE, {"status": "running", "pid": os.getpid(), "pid_start": "1",
                                                      "stage": "deploy"})
    if jobs.proc_start_time(os.getpid()) is None:
        pytest.skip("no /proc")
    assert jobs.read_job(tmp_path)["status"] == "interrupted"
    jobs.atomic_write_json(tmp_path / jobs.JOB_FILE, {"status": "running", "pid": os.getpid(),
                                                      "pid_start": jobs.proc_start_time(os.getpid())})
    assert jobs.read_job(tmp_path)["status"] == "running"


def test_events_replay_from_offset(client):
    rd = server.RUNS_DIR / "rp"
    rd.mkdir()
    jobs.atomic_write_json(rd / jobs.JOB_FILE, {"status": "completed", "pid": 1, "logs": []})
    evs = [{"ts": i, "type": t, "message": t} for i, t in enumerate(("stage", "level_done", "complete"))]
    (rd / jobs.EVENTS_FILE).write_text("".join(json.dumps(e) + "\n" for e in evs))
    with client.stream("GET", "/api/sessions/rp/events") as r:
        lines = list(r.iter_lines())
    ids = [int(x[4:]) for x in lines if x.startswith("id: ")]
    assert [json.loads(x[6:])["type"] for x in lines if x.startswith("data: ")] == ["stage", "level_done", "complete"]
    with client.stream("GET", f"/api/sessions/rp/events?since={ids[0]}") as r:
        assert [json.loads(x[6:])["type"] for x in r.iter_lines() if x.startswith("data: ")] == ["level_done", "complete"]
    with client.stream("GET", "/api/sessions/nojob/events") as r:
        assert any('"idle"' in x for x in r.iter_lines())


def test_force_stop_marks_interrupted_then_teardown_clears_warning(client, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_TICK_S", "0.3")
    _start(client)
    rd = server.RUNS_DIR / "j1"
    _wait(client, "j1", lambda j: j["stage"] == "load" and (rd / "endpoint.json").exists())
    r = client.post("/api/sessions/j1/stop", json={"force": True}).json()
    assert r["signal"] == "SIGKILL"
    assert r["job"]["status"] == "interrupted" and r["job"]["endpoint_warning"] is True
    # the warning's button runs the teardown step as a job
    monkeypatch.setenv("FAKE_BENCH_TICK_S", "0.01")
    assert client.post("/api/sessions/j1/run", json={"mode": "step", "steps": ["teardown"]}).status_code == 200
    job = _wait(client, "j1", lambda j: j["status"] != "running")
    assert job["status"] == "completed" and job["endpoint_warning"] is False
    assert not (rd / "endpoint.json").exists()


def test_bad_steps_rejected(client):
    r = client.post("/api/sessions/j2/run", json={"config": CFG, "mode": "step", "steps": ["rm -rf"]})
    assert r.status_code == 400


def test_run_arguments_are_allowlisted(client):
    """Everything that reaches the run-job command line is allowlisted or an int (CWE-78 hardening)."""
    for body in ({"mode": "x; rm -rf /"}, {"mode": "--run-dir=/etc"}, {"mode": ["auto"]},
                 {"mode": "step", "steps": "load"}, {"mode": "auto", "concurrency": ["16; ls"]},
                 {"mode": "auto", "concurrency": [0]}, {"mode": "auto", "requests_per_level": "1e9"},
                 {"mode": "auto", "requests_per_level": 10**7}):
        r = client.post("/api/sessions/j3/run", json={"config": CFG, **body})
        assert r.status_code == 400, (body, r.status_code, r.text)
    assert not (client.get("/api/sessions/j3").json().get("job") or {}).get("pid")


def test_cli_run_job_mode_choices():
    from aiops_dilemma import cli
    import pytest as _p
    with _p.raises(SystemExit):
        cli.main(["run-job", "-c", "x.yaml", "--mode", "evil", "--steps", "plan"])


def test_cli_run_writes_job_files(tmp_path):
    rd = tmp_path / "runs" / "cli1"
    rd.mkdir(parents=True)
    cfg = config.normalize({**CFG, "name": "cli1"}, path=str(rd / "workload.yaml"))
    (rd / "workload.yaml").write_text(config.to_yaml(cfg))
    env = {**os.environ, "DILEMMA_TEST_FAKE_BENCH": FAKE, "FAKE_BENCH_TICK_S": "0.01"}
    p = subprocess.run([sys.executable, "-m", "aiops_dilemma.cli", "auto", "-c", str(rd / "workload.yaml")],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    job = json.loads((rd / jobs.JOB_FILE).read_text())
    assert job["status"] == "completed" and job["source"] == "cli" and job["command"].startswith("dilemma auto")
    assert [e["type"] for e in _events(rd)][-1] == "complete"
    # a second job refuses to start while one is live
    jobs.atomic_write_json(rd / jobs.JOB_FILE, {**job, "status": "running", "pid": os.getpid(),
                                                "pid_start": jobs.proc_start_time(os.getpid())})
    p = subprocess.run([sys.executable, "-m", "aiops_dilemma.cli", "teardown", "-c", str(rd / "workload.yaml")],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and "already running" in (p.stdout + p.stderr)


def test_killed_mid_deploy_warns_about_an_unrecorded_endpoint(tmp_path):
    jobs.atomic_write_json(tmp_path / jobs.JOB_FILE, {"status": "running", "pid": _dead_pid(), "stage": "deploy",
                                                      "deployed_here": True, "logs": []})
    p = jobs.job_payload(tmp_path)
    assert p["status"] == "interrupted" and p["deploy_warning"] is True and p["endpoint_warning"] is False
