"""Durable benchmark jobs: state on disk so jobs survive server restarts and CLI runs show up in the app.

A job runs in its own process (`dilemma run-job`, started detached by the web app, or `dilemma auto` /
`deploy` / `load` / `calibrate` / `teardown` run by hand) and owns three files in the session's run directory:

  job.json            current state, rewritten atomically (temp file + rename) on every change
  job_events.jsonl    append-only event log (the shapes the web app's SSE stream has always used)
  job.log             stdout/stderr of app-started jobs

The server only reads them, except to mark a job `interrupted` when its process is gone (dead pid, or the pid was
reused by another process: the process start time from /proc is stored next to the pid).
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

JOB_FILE = "job.json"
EVENTS_FILE = "job_events.jsonl"
LOG_FILE = "job.log"
STOP_FILE = "job_stop.json"

AUTO_STEPS = ["plan", "prompts", "deploy", "load", "analyze", "calibrate", "report", "teardown"]
STEPS = set(AUTO_STEPS)
RUN_MODES = {"auto", "step", *AUTO_STEPS}   # allowed `run-job --mode` values (the app validates against these too)
TERMINAL = {"completed", "error", "cancelled", "interrupted"}
LOG_KEEP = 200
REQUEST_EVENT_INTERVAL_S = 1.0


class JobCancelled(Exception):
    """Raised at a safe point (between steps, between request batches) after SIGTERM / Ctrl-C."""


class JobBusy(SystemExit):
    """Another live job already owns this session's run directory."""


# ------------------------------------------------------------------------------------------------ files + liveness
def atomic_write_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _proc_stat(pid: int) -> tuple[str, str] | None:
    """(state, starttime) from /proc/<pid>/stat, or None when unavailable (no /proc, or no such process)."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2:].split()     # fields after "pid (comm) "; field 3 (state) is index 0
    return (fields[0], fields[19]) if len(fields) > 19 else None


def proc_start_time(pid: int) -> str | None:
    st = _proc_stat(pid)
    return st[1] if st else None


def pid_alive(pid: int | None, start_time: str | None = None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                                          # exists, owned by someone else
    except OSError:
        return False
    st = _proc_stat(int(pid))
    if st:
        if st[0] in ("Z", "X"):                       # zombie: exited, not yet reaped by its parent
            return False
        if start_time and st[1] != str(start_time):  # pid reused by another process
            return False
    return True


def endpoint_at_risk(rd: Path) -> dict | None:
    """The recorded endpoint if this tool deployed it (so it bills until torn down), else None."""
    ep = _load(Path(rd) / "endpoint.json")
    if isinstance(ep, dict) and ep.get("endpoint") and not ep.get("attached_existing"):
        return ep
    return None


def append_event(rd: Path, event: dict) -> None:
    with open(Path(rd) / EVENTS_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def read_job(rd: Path, *, mark: bool = True) -> dict | None:
    """job.json with liveness applied: a `running` job whose process is gone becomes `interrupted` (persisted)."""
    rd = Path(rd)
    job = _load(rd / JOB_FILE)
    if not isinstance(job, dict):
        return None
    if job.get("status") == "running" and not pid_alive(job.get("pid"), job.get("pid_start")):
        job.update(status="interrupted", finished_at=time.time(),
                   error=f"The benchmark process (pid {job.get('pid')}) stopped unexpectedly during "
                         f"'{job.get('stage')}'.")
        msg = job["error"] + (" Its endpoint may still be running (billing)." if endpoint_at_risk(rd) else "")
        job.setdefault("logs", []).append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if mark:
            try:
                append_event(rd, {"ts": time.time(), "type": "interrupted", "message": msg})
                atomic_write_json(rd / JOB_FILE, job)
            except OSError:
                pass
    return job


def live_job(rd: Path) -> dict | None:
    job = read_job(rd)
    return job if job and job.get("status") == "running" else None


def stop_requested(rd: Path, job: dict | None) -> float | None:
    s = _load(Path(rd) / STOP_FILE)
    if job and isinstance(s, dict) and s.get("pid") == job.get("pid") and s.get("started_at") == job.get("started_at"):
        return s.get("requested_at")
    return None


def job_payload(rd: Path) -> dict:
    """The `job` object of the session payload (same keys the UI always read, plus the durable extras)."""
    rd = Path(rd)
    job = read_job(rd)
    if not job:
        return {"status": "idle", "stage": None, "error": None, "level_progress": None, "levels_completed": [],
                "logs": [], "endpoint_warning": False, "deploy_warning": False}
    at_risk = endpoint_at_risk(rd)
    status = job.get("status")
    warn = bool(at_risk) and (status == "interrupted" or bool(job.get("teardown_failed")))
    return {
        "status": status,
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "stage": job.get("stage"),
        "steps": job.get("steps"),
        "mode": job.get("mode"),
        "source": job.get("source"),
        "command": job.get("command"),
        "pid": job.get("pid"),
        "error": job.get("error"),
        "level_progress": job.get("level_progress"),
        "levels_completed": job.get("levels_completed", []),
        "logs": job.get("logs", [])[-80:],
        "endpoint": job.get("endpoint"),
        "calibration": job.get("calibration"),
        "preflight": job.get("preflight"),
        "prompt_source": job.get("prompt_source"),
        "stop_requested_at": stop_requested(rd, job) if status == "running" else None,
        "endpoint_warning": warn,
        # killed mid-deploy: Agent Platform may finish creating an endpoint that was never recorded in endpoint.json
        "deploy_warning": status == "interrupted" and job.get("stage") == "deploy" and bool(job.get("deployed_here"))
                          and not at_risk,
    }


def scan(runs_dir: Path) -> dict[str, str]:
    """Status of every session with a job file (marks dead `running` jobs interrupted). {session: status}."""
    out: dict[str, str] = {}
    if not Path(runs_dir).is_dir():
        return out
    for f in Path(runs_dir).glob(f"*/{JOB_FILE}"):
        job = read_job(f.parent)
        if job:
            out[f.parent.name] = job.get("status", "idle")
    return out


def signal_job(rd: Path, force: bool = False) -> dict | None:
    """SIGTERM (graceful: the runner cancels at the next safe point and tears down) or SIGKILL the live job."""
    job = live_job(rd)
    if not job:
        return None
    pid = int(job["pid"])
    if pid <= 1 or pid in (os.getpid(), os.getppid()):   # job.json is data: never signal init, all processes or us
        raise ValueError(f"refusing to signal pid {pid} from job.json")
    if force:
        try:
            if os.getpgid(pid) == pid:              # app-started jobs lead their own session: kill gcloud children too
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        os.kill(pid, signal.SIGTERM)
        atomic_write_json(Path(rd) / STOP_FILE, {"pid": pid, "started_at": job.get("started_at"),
                                                  "requested_at": time.time()})
    return job


# ------------------------------------------------------------------------------------------------ the runner side
class JobRecorder:
    """Writes job.json + job_events.jsonl for the job running in this process."""

    def __init__(self, rd: Path, *, steps: list[str], mode: str = "auto", source: str = "cli",
                 keep_endpoint: bool = False, concurrency: list[int] | None = None,
                 requests_per_level: int | None = None, command: str | None = None):
        self.rd = Path(rd)
        self.echo = True
        self.lock = threading.RLock()
        self.cancel_requested: float | None = None
        self._cancel_announced = False
        self._last_request_evt = 0.0
        self._old_handlers: dict[int, Any] = {}
        pid = os.getpid()
        self.state: dict[str, Any] = {
            "status": "running", "pid": pid, "pid_start": proc_start_time(pid), "source": source,
            "command": command, "mode": mode, "steps": list(steps), "keep_endpoint": keep_endpoint,
            "concurrency": concurrency, "requests_per_level": requests_per_level,
            "started_at": time.time(), "finished_at": None, "stage": steps[0] if steps else None, "error": None,
            "level_progress": None, "levels_completed": [], "endpoint": None, "deployed_here": False,
            "calibration": None, "preflight": None, "teardown_failed": False, "logs": [],
            "started_by": os.environ.get("DILEMMA_JOB_USER") or None,   # IAP user on the cloud deployment
        }

    # lifecycle ---------------------------------------------------------------------------------------------------
    def start(self) -> "JobRecorder":
        self.rd.mkdir(parents=True, exist_ok=True)
        other = live_job(self.rd)
        if other and other.get("pid") != os.getpid():
            raise JobBusy(f"a benchmark job is already running for this session (pid {other.get('pid')}, "
                          f"stage {other.get('stage')}); stop it first")
        (self.rd / EVENTS_FILE).write_text("")
        try:
            (self.rd / STOP_FILE).unlink()
        except OSError:
            pass
        self.write()
        return self

    def write(self) -> None:
        with self.lock:
            atomic_write_json(self.rd / JOB_FILE, self.state)

    def emit(self, event_type: str, persist: bool = True, **data) -> None:
        evt = {"ts": time.time(), "type": event_type, **data}
        if "message" in data and self.echo:
            print(f"[{event_type}] {data['message']}", flush=True)
        with self.lock:
            if "message" in data:
                self.state["logs"] = (self.state["logs"] + [f"[{time.strftime('%H:%M:%S')}] {data['message']}"])[-LOG_KEEP:]
            append_event(self.rd, evt)
            if persist:
                self.write()

    def finish(self, status: str, error: str | None = None) -> None:
        with self.lock:
            self.state.update(status=status, finished_at=time.time(), error=error)
            if status == "completed":
                self.state["stage"] = "done"
            self.write()

    # cancellation ------------------------------------------------------------------------------------------------
    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            if self.cancel_requested is None:
                self.cancel_requested = time.time()
                if signum == signal.SIGINT:   # a second Ctrl-C aborts at once (teardown may then be skipped)
                    signal.signal(signal.SIGINT, signal.default_int_handler)
                    print("\ncancelling at the next safe point, then tearing down (Ctrl-C again to abort now)",
                          flush=True)
        for s in (signal.SIGTERM, signal.SIGINT):
            try:
                self._old_handlers[s] = signal.signal(s, handler)
            except ValueError:                # not the main thread (tests)
                pass

    def restore_signal_handlers(self) -> None:
        for s, h in self._old_handlers.items():
            try:
                signal.signal(s, h)
            except (ValueError, TypeError):
                pass
        self._old_handlers.clear()

    def check_cancel(self) -> None:
        if self.cancel_requested is None:
            return
        if not self._cancel_announced:
            self._cancel_announced = True
            self.emit("cancelling", message="Cancellation requested: stopping after the current request batch...")
        raise JobCancelled("cancelled by user")

    # loadtest progress callback ----------------------------------------------------------------------------------
    def progress(self, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "waiting_ready":
            self.emit("waiting_ready", waited_s=ev.get("waited_s"), last=ev.get("last"),
                      message=f"Waiting for the endpoint to accept requests ({ev.get('waited_s')}s): {ev.get('last')}")
        elif kind == "preflight_start":
            self.state["preflight"] = {"status": "running", "started_at": time.time()}
            pt = ev.get("prompt_tokens")
            self.emit("preflight_start", target_s=ev.get("target_s"), output_tokens=ev.get("output_tokens"),
                      prompt_tokens=pt, prompt_note=ev.get("prompt_note"),
                      message=f"Pre-flight check: a ~{ev.get('target_s', 0):.0f} s streamed generation "
                              f"({ev.get('output_tokens', 0):,} tokens)"
                              + (f" and a {pt:,}-token prompt" if pt else "") + " before the sweep...")
        elif kind == "preflight_done":
            res = ev.get("result") or {}
            durations = {c["name"]: c.get("duration_s") for c in res.get("checks", []) if c.get("duration_s")}
            self.state["preflight"] = {"status": "passed" if res.get("passed") else "failed",
                                       "duration_s": res.get("duration_s"), "durations": durations,
                                       "message": res.get("message")}
            self.emit("preflight_done", passed=bool(res.get("passed")), duration_s=res.get("duration_s"),
                      durations=durations, result=res,
                      message=("Pre-flight passed: " if res.get("passed") else "") + str(res.get("message")))
        elif kind == "level_start":
            run = ev.get("run", "mixed")
            self.state["level_progress"] = {"concurrency": ev["concurrency"], "completed": 0, "total": ev["total"],
                                            "run": run}
            self.emit("level_start", concurrency=ev["concurrency"], total=ev["total"], run=run,
                      message=f"Concurrency {ev['concurrency']} ({run} prompts): running {ev['total']} requests...")
        elif kind == "request_done":
            self.state["level_progress"] = {"concurrency": ev["concurrency"], "completed": ev["completed"],
                                            "total": ev["total"],
                                            "run": (self.state.get("level_progress") or {}).get("run", "mixed")}
            now = time.monotonic()
            if now - self._last_request_evt >= REQUEST_EVENT_INTERVAL_S or ev["completed"] >= ev["total"]:
                self._last_request_evt = now
                self.emit("request_done", **{k: v for k, v in ev.items() if k != "event"})
        elif kind == "level_done":
            s = ev["summary"]
            self.state["levels_completed"].append(s)
            run = s.get("run") or ev.get("run", "mixed")
            self.emit("level_done", concurrency=ev["concurrency"], summary=s, run=run, message=(
                f"Concurrency {ev['concurrency']} ({run} prompts) done: input={s.get('input_tps', 0):,.0f} tok/s, "
                f"p95={s.get('lat_p95', 0):.1f}s" if s.get("ok")
                else f"Concurrency {ev['concurrency']}: all {s.get('errors', 0)} requests failed"))
        self.check_cancel()


# ------------------------------------------------------------------------------------------------ the step runner
def run_steps(cfg: dict, rd: Path, steps: list[str], rec: JobRecorder, *, mode: str = "auto",
              keep_endpoint: bool = False, concurrency: list[int] | None = None,
              requests_per_level: int | None = None, calibrate_concurrency: list[int] | None = None,
              load_kwargs: dict | None = None) -> str:
    """Run benchmark steps, recording progress. Returns the final status (completed | error | cancelled).

    If this job deployed the endpoint (or was asked to) and was meant to tear it down, the endpoint is torn down
    even when a later step fails or the job is cancelled."""
    from aiops_dilemma import analyze, cli, dashboard, deploy, loadtest, planner, pricing, prompts, report, userprompts

    rd = Path(rd)
    keeps = keep_endpoint and mode == "auto"
    status, error = "running", None
    plan_data = _load(rd / "plan.json")

    def save_plan():
        p = planner.plan(cfg)
        (rd / "plan.json").write_text(json.dumps(p, indent=2, default=str))
        return p

    try:
        for step in steps:
            rec.check_cancel()
            rec.state["stage"] = step
            rec.emit("stage", stage=step, message=f"Starting step: {step.upper()}")
            if step == "plan":
                plan_data = save_plan()
                ch = plan_data["chosen"]
                rec.emit("plan_done", plan=plan_data,
                         message=f"Selected {ch['accelerator_count']}x {ch['accelerator_type']} ({ch['machine_type']}) "
                                 f"— weights {plan_data['weights_gb']} GB")
            elif step == "prompts":
                path = prompts.build(cfg, rd)
                src = userprompts.describe_source(cfg, rd)
                n_req = prompts.level_requests(cfg)
                note = ""
                if src["kind"] == "file" and src.get("count") and n_req > src["count"]:
                    note = (f" {src['count']:,} prompts for {n_req:,} load-test requests: prompts repeat across "
                            f"levels, each sent about {n_req / src['count']:.1f} times.")
                rec.state["prompt_source"] = src
                rec.emit("prompts_done", prompt_source=src,
                         message=f"Prompts: {src['text']}. Built the prompt set at {Path(path).name}.{note}")
            elif step == "deploy":
                plan_data = plan_data or save_plan()
                prior = _load(rd / "endpoint.json") if (rd / "endpoint.json").exists() else None
                if prior and prior.get("partial"):
                    # An earlier deploy stopped after creating the endpoint but before the model was on it: reusing
                    # it would load-test an empty endpoint. Tear it down, then deploy afresh below.
                    rec.emit("log", message=f"An earlier deploy didn't finish (endpoint {prior.get('endpoint')}); "
                                            "deleting it and deploying again.")
                    deploy.teardown(cfg, rd)
                    prior = None
                if prior:
                    ep = prior
                    rec.state["endpoint"] = ep
                    rec.emit("deploy_done", endpoint=ep, message=f"Using existing endpoint {ep.get('endpoint')}")
                else:
                    rec.state["deployed_here"] = True      # persisted by the emit: a kill mid-deploy is then visible
                    rec.emit("log", message="Deploying 1-replica Agent Platform dedicated endpoint (10–30 min)...")
                    ep = deploy.deploy(cfg, plan_data, rd)
                    rec.state["endpoint"] = ep
                    rec.emit("deploy_done", endpoint=ep, message=f"Deployed endpoint: {ep.get('endpoint')}")
            elif step == "load":
                kw = dict(load_kwargs or {})
                if not kw.get("prompts_path") and not (rd / "prompts.jsonl").exists():
                    prompts.build(cfg, rd)
                loadtest.run(cfg, rd, concurrency=concurrency, per_level=requests_per_level,
                             progress_cb=rec.progress, **kw)
            elif step == "analyze":
                meas = analyze.analyze(cfg, rd, tag=(load_kwargs or {}).get("tag", ""))
                rec.emit("analyze_done", measured=meas,
                         message=f"Peak throughput: {meas['input_tps_per_gpu']:,.0f} input tok/s/chip at "
                                 f"concurrency {meas['peak']['concurrency']}")
            elif step == "calibrate":
                if not cli.needs_calibration(cfg, rd):
                    rec.emit("log", message=(
                        "Your prompts have none over 32K tokens: per-class calibration skipped (it compares a "
                        "short-prompt run with the mixed one)." if cfg["prompts"].get("source") == "file" else
                        "No prompts over 32K in this workload: per-class calibration skipped."))
                else:
                    rec.emit("log", message="Calibrating long prompts: a short-prompt-only run on the same endpoint...")
                    meas = cli.run_calibration(cfg, rd, concurrency=calibrate_concurrency,
                                               per_level=requests_per_level, progress_cb=rec.progress)
                    cal = meas["calibration"]
                    rec.state["calibration"] = cal
                    rec.emit("calibrate_done", calibration=cal, message=(
                        f"Short prompts {cal['short_tps_per_gpu']:,.0f} tok/s per chip; 32K-128K "
                        f"{cal['slowdown']['medium']:.1f}x and over 128K {cal['slowdown']['large']:.1f}x slower "
                        "in mixed traffic"))
            elif step == "report":
                plan_data = plan_data or save_plan()
                meas = _load(rd / "measured.json") or analyze.analyze(cfg, rd)
                ch = plan_data["chosen"]
                prices = ch.get("prices") or pricing.prices_for(cfg, ch["accelerator_type"], ch["accelerator_count"])
                report.write(cfg, plan_data, meas, prices, rd)
                dashboard.build(cfg, meas, prices, rd)
                rec.emit("report_done", message="Generated report.md and interactive dashboard.html")
            elif step == "teardown":
                if keeps:
                    rec.emit("log", message="Keeping the Agent Platform endpoint (keep endpoint is on): it bills "
                                            "until you tear it down.")
                else:
                    deploy.teardown(cfg, rd)
                    rec.emit("teardown_done", message="Endpoint undeployed and deleted.")
            else:
                raise SystemExit(f"unknown step '{step}'")
        status = "completed"
    except JobCancelled:
        status = "cancelled"
    except KeyboardInterrupt:
        status, error = "cancelled", "interrupted (Ctrl-C)"
    except (Exception, SystemExit) as e:   # noqa: BLE001
        status, error = "error", str(e) or type(e).__name__
    finally:
        if status == "running":            # an exception we don't catch (e.g. a second Ctrl-C): still clean up
            status, error = "cancelled", "aborted"
        if (rec.state["deployed_here"] or "deploy" in steps) and "teardown" in steps and not keeps \
                and status in ("error", "cancelled") and (rd / "endpoint.json").exists():
            try:
                rec.emit("log", message="Tearing down the endpoint after the failed or cancelled run...")
                deploy.teardown(cfg, rd)
                rec.emit("teardown_done", message="Endpoint deleted.")
            except BaseException as te:    # noqa: BLE001
                rec.state["teardown_failed"] = True
                rec.emit("log", message=f"Teardown failed — delete the endpoint manually: {te}")
        # The terminal event goes first, then the final state: a reader that sees a terminal status has the event.
        if status == "completed":
            rec.emit("complete", persist=False, message="Benchmark workflow finished successfully.")
        elif status == "cancelled":
            rec.emit("cancelled", persist=False,
                     message="Benchmark cancelled" + (f": {error}." if error else " by user."))
        else:
            rec.emit("error", persist=False, error=error, message=f"Error in step '{rec.state.get('stage')}': {error}")
        rec.finish(status, error)
    return status


def run_job(cfg: dict, rd: Path, steps: list[str], *, mode: str = "auto", source: str = "cli",
            keep_endpoint: bool = False, concurrency: list[int] | None = None, requests_per_level: int | None = None,
            calibrate_concurrency: list[int] | None = None, load_kwargs: dict | None = None,
            command: str | None = None, on_start: Callable[[JobRecorder], None] | None = None) -> str:
    """Create the job files, handle SIGTERM/Ctrl-C as a graceful cancel, run the steps."""
    _maybe_load_test_stub()
    rec = JobRecorder(rd, steps=steps, mode=mode, source=source, keep_endpoint=keep_endpoint,
                      concurrency=concurrency, requests_per_level=requests_per_level, command=command).start()
    rec.install_signal_handlers()
    if on_start:
        on_start(rec)
    try:
        return run_steps(cfg, rd, steps, rec, mode=mode, keep_endpoint=keep_endpoint, concurrency=concurrency,
                         requests_per_level=requests_per_level, calibrate_concurrency=calibrate_concurrency,
                         load_kwargs=load_kwargs)
    finally:
        rec.restore_signal_handlers()


def _maybe_load_test_stub() -> None:
    """Test-only: DILEMMA_TEST_FAKE_BENCH=<path to a .py> is executed before the job runs (it stubs deploy/load)."""
    path = os.environ.get("DILEMMA_TEST_FAKE_BENCH")
    if path:
        import runpy
        runpy.run_path(path, run_name="dilemma_fake_bench")
