"""FastAPI backend server for the mlops-dilemma interactive web application (`dilemma ui`)."""

import asyncio
import contextlib
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aiops_dilemma import (
    auth,
    catalog,
    config,
    costmodel,
    dashboard,
    deploy,
    gemini,
    hardware,
    hftoken,
    hfmeta,
    iap,
    ingest,
    jobs,
    modelinfo,
    paths,
    planner,
    pricing,
    quotas,
    recommender,
    regions,
    report,
    serving,
    traffic,
    userprompts,
)

ROOT_DIR = paths.ROOT_DIR
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RUNS_DIR = paths.runs_dir()          # DILEMMA_RUNS_DIR overrides (UI checks; /data/runs on Cloud Run)
BLOG_EXAMPLE_DIR = ROOT_DIR / "blog" / "example"

@contextlib.asynccontextmanager
async def _lifespan(_app):
    _scan_jobs_on_startup()
    yield


CLOUD = paths.cloud_mode()   # shared Cloud Run deployment behind IAP (K_SERVICE set, or DILEMMA_CLOUD=1)
# No interactive API docs on the shared deployment (less surface; the UI is the interface).
app = FastAPI(title="mlops-dilemma Interactive UI", version="0.2.0", lifespan=_lifespan,
              **({"docs_url": None, "redoc_url": None, "openapi_url": None} if CLOUD else {}))
_SESSION_RX = re.compile(r"^/api/sessions/([^/]+)")


def audit(event: str, request: Request | None = None, **fields) -> None:
    """One structured line on stdout per action: Cloud Run sends JSON lines to Cloud Logging as jsonPayload, so
    `jsonPayload.audit.user="alice@example.com"` finds everything someone did. Written in cloud mode only."""
    if not CLOUD:
        return
    user = getattr(getattr(request, "state", None), "user", None) if request is not None else None
    rec = {"event": event, "user": (user or {}).get("email") or None, **fields}
    if request is not None:
        rec.setdefault("method", request.method)
        rec.setdefault("path", request.url.path)
        m = _SESSION_RX.match(request.url.path)
        if m:
            rec.setdefault("session", m.group(1))
    line = {"severity": "WARNING" if event.endswith("denied") else "NOTICE",
            "message": f"{event}: {rec.get('method', '')} {rec.get('path', '')} by {rec['user'] or 'unknown'}"
                       + (f" -> {rec['status']}" if "status" in rec else ""),
            "audit": rec, "logging.googleapis.com/labels": {"app": "mlops-dilemma", "user": rec["user"] or ""}}
    print(json.dumps(line, default=str), flush=True)


@app.middleware("http")
async def _iap_gate_and_audit(request: Request, call_next):
    """Cloud mode: every request (API, page, assets) must carry a valid IAP header; record who did what."""
    request.state.user = None
    if iap.required():
        try:
            request.state.user = await run_in_threadpool(iap.verify, request.headers.get(iap.HEADER))
        except iap.IapError as e:
            audit("access_denied", request, reason=str(e), status=401,
                  remote=(request.client.host if request.client else None))
            return JSONResponse({"detail": "This app is only reachable through Identity-Aware Proxy."}, status_code=401)
    t0 = time.time()
    response = await call_next(request)
    path = request.url.path
    if request.method != "GET" or "/export/" in path:
        audit("action", request, status=response.status_code, ms=round((time.time() - t0) * 1000))
    return response


def _cloud_forbid(what: str) -> None:
    if CLOUD:
        raise HTTPException(status_code=403, detail=f"{what} is turned off on this shared deployment.")


def _locked_project() -> str | None:
    """Cloud mode: the deployment's project (GOOGLE_CLOUD_PROJECT); the picker can't switch away from it."""
    return (os.environ.get("GOOGLE_CLOUD_PROJECT") or None) if CLOUD else None


def _with_cloud(st: dict, request: Request | None = None) -> dict:
    if CLOUD and isinstance(st, dict):
        user = getattr(getattr(request, "state", None), "user", None) if request is not None else None
        st = {**st, "cloud": {"project_locked": _locked_project(), "hf_token_managed": True, "sign_in": False,
                              "user": (user or {}).get("email") or None}}
    return st
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Benchmark jobs run out of process (`dilemma run-job`) and keep their state in runs/<session>/job.json +
# job_events.jsonl (see jobs.py), so they survive server restarts and CLI-started runs show up too.
_LAUNCH_LOCK = threading.Lock()
_CHILDREN: dict[int, subprocess.Popen] = {}     # app-started job processes, reaped so they don't linger as zombies
JOB_START_TIMEOUT_S = 20.0

# Read-only example sessions (shipped with the repo, generated from blog/example/workload.yaml).
EXAMPLE_SESSIONS = {"blog-example"}
DEFAULT_MODEL = {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": "google/gemma-4-26b-a4b-it"}
DEFAULT_TRAFFIC = {"peak_input_tps": 150000, "average_input_tps": 50000, "floor_input_tps": 6000}
CACHE_DIR = paths.cache_dir()        # DILEMMA_CACHE_DIR overrides
CATALOG_TTL_S = 15 * 60          # serve the cached catalog instantly; refresh in the background when older
_CATALOG_MEM: dict[str, dict] = {}
_CATALOG_REFRESHING: set[str] = set()
_CATALOG_LOCK = threading.Lock()


def _default_project() -> str | None:
    """The app's project (chosen in the app's picker, else the machine default; auth.app_project memoizes gcloud)."""
    return auth.app_project()


def is_example(name: str) -> bool:
    return _sanitize_name(name) in EXAMPLE_SESSIONS


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-]+", "-", name.strip()).strip("-")
    return clean or "session"


def _session_dir(name: str, *, for_write: bool = False) -> Path:
    clean = _sanitize_name(name)
    if for_write and clean in EXAMPLE_SESSIONS:
        raise HTTPException(
            status_code=403,
            detail=f"'{clean}' is a read-only example session. Save a copy instead: "
                   f"POST /api/sessions/<new-name> with {{\"copy_of\": \"{clean}\"}}.",
        )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    user_dir = RUNS_DIR / clean
    demo_dir = BLOG_EXAMPLE_DIR / "runs" / "blog-example"
    if clean == "blog-example" and not user_dir.exists() and demo_dir.exists():
        if not for_write:
            return demo_dir
    return user_dir


def _load_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _session_Config(name: str, rd: Path) -> dict:
    yaml_path = rd / "workload.yaml"
    if not yaml_path.exists() and name == "blog-example" and (BLOG_EXAMPLE_DIR / "workload.yaml").exists():
        yaml_path = BLOG_EXAMPLE_DIR / "workload.yaml"
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text()) or {}
    else:
        raw = {"name": name}
    raw["name"] = name
    cfg = config.normalize(raw, path=str(yaml_path))
    cfg["_run_dir"] = str(rd)
    return cfg


def _with_defaults(cfg: dict) -> dict:
    """Fill the model and traffic the estimate path assumes when the user hasn't chosen them yet."""
    if not cfg["model"].get("model_garden_id"):
        cfg["model"].update(DEFAULT_MODEL)
    t = cfg["traffic"]
    # A drawn/imported week or a preset defines the traffic; a quietest hour of 0 is valid (sporadic traffic)
    has_shape = t.get("hourly_csv") or t.get("custom_hourly") or t.get("preset")
    if not has_shape and not (t.get("peak_input_tps") and t.get("average_input_tps")
                              and t.get("floor_input_tps") is not None):
        t.update({k: (t.get(k) if t.get(k) is not None else v) for k, v in DEFAULT_TRAFFIC.items()})
    return cfg


def _model_name(model: dict) -> str:
    ids = {str(x).lower() for x in (model.get("model_garden_id"), model.get("hf_id")) if x}
    try:
        for m in modelinfo.discover_models(None, refresh=False).get("models", []):
            if {str(m.get("model_garden_id", "")).lower(), str(m.get("hf_id", "")).lower()} & ids:
                return m.get("name") or m.get("id")
    except Exception:
        pass
    ref = model.get("hf_id") or model.get("model_garden_id") or "custom model"
    return ref.split("/")[-1].split("@")[-1]


def _hw(d: dict | None) -> dict | None:
    if not d or not d.get("accelerator_type"):
        return None
    return {"accelerator_type": d["accelerator_type"], "accelerator_count": int(d.get("accelerator_count") or 1),
            "machine_type": d.get("machine_type")}


def _same_hw(a: dict | None, b: dict | None) -> bool:
    return bool(a and b and a.get("accelerator_type") == b.get("accelerator_type")
                and int(a.get("accelerator_count") or 0) == int(b.get("accelerator_count") or 0))


def resolve_throughput(measured: dict | None, chosen: dict | None, override: float | None = None) -> dict:
    """The one per-chip throughput every view uses: measured > explicit override > saved override/calibration > spec estimate."""
    if report.is_measured(measured):
        per, src = float(measured["input_tps_per_gpu"]), report.throughput_source(measured)
        out = {"per_chip": round(per, 1), "source": src, "label": report.throughput_label(per, src)}
        if measured.get("mix_factor"):   # mix the load test ran at (costmodel.ref_mix_factor)
            out["mix_factor"] = float(measured["mix_factor"])
        if (measured.get("bucket_tps") or {}).get("small"):   # per-class rates (costmodel.with_measured_classes)
            out["bucket_tps"] = measured["bucket_tps"]
            out["input_tps_per_gpu"] = per
        return out
    elif override:
        per, src = float(override), "override"
    elif (measured and measured.get("throughput_source") in ("override", "calibrated")
          and measured.get("input_tps_per_gpu") and _same_hw(measured.get("endpoint"), chosen)):
        per, src = float(measured["input_tps_per_gpu"]), measured["throughput_source"]
        out = {"per_chip": round(per, 1), "source": src,
               "label": measured.get("throughput_label") or report.throughput_label(per, src)}
        if src == "calibrated" and measured.get("mix_factor"):
            out["mix_factor"] = float(measured["mix_factor"])
        return out
    elif chosen and chosen.get("est_input_tps_per_gpu"):
        # estimate tiers (planner.reference_throughput): basis reference / scaled / spec, with its tooltip note
        per, basis = float(chosen["est_input_tps_per_gpu"]), chosen.get("est_basis") or "spec"
        return {"per_chip": round(per, 1), "source": "estimated", "basis": basis,
                "basis_label": chosen.get("est_basis_label") or planner.BASIS_LABELS.get(basis),
                "note": chosen.get("est_basis_note"),
                "label": report.throughput_label(per, "estimated", basis)}
    else:
        return {"per_chip": None, "source": "estimated", "label": "No accelerator configuration fits this model."}
    return {"per_chip": round(per, 1), "source": src, "label": report.throughput_label(per, src)}


def _selection(cfg: dict, measured: dict | None) -> tuple[dict, dict | None, dict, str | None, dict | None]:
    """(selected, chosen plan entry, throughput, plan error, no-fit explanation) for a normalized config and its
    measured.json. `no_fit` (planner.no_fit_explanation) is set when the model can't be planned at its precision."""
    plan_err = no_fit = None
    chosen = None
    try:
        chosen = planner.plan(cfg)["chosen"]
    except planner.NoFitError as e:
        plan_err, no_fit = str(e), e.info
    except (Exception, SystemExit) as e:
        plan_err = str(e)
    if report.is_measured(measured):
        ep = measured.get("endpoint") or {}
        if not _same_hw(ep, chosen):
            chosen = {**(chosen or {}), **_hw(ep)}
            chosen["prices"] = pricing.prices_for(cfg, ep["accelerator_type"], int(ep["accelerator_count"])) \
                if ep.get("accelerator_type") in catalog.GPUS else (chosen or {}).get("prices", {})
    m = cfg["model"]
    selected = {
        "model": {"model_garden_id": m.get("model_garden_id"), "hf_id": m.get("hf_id"), "name": _model_name(m)},
        "hardware": _hw(chosen),
        "precision": m.get("precision", "bf16"),
    }
    return selected, chosen, resolve_throughput(measured, chosen), plan_err, no_fit


def _effective_measured(measured: dict | None, selected: dict, throughput: dict) -> dict | None:
    """measured.json for real runs; otherwise an estimate record matching `selected` and `throughput`."""
    if report.is_measured(measured):
        return measured
    if not selected.get("hardware") or not throughput.get("per_chip"):
        return None
    return {"estimated": True, "levels": [], "peak": None, "endpoint": selected["hardware"],
            "input_tps_per_gpu": throughput["per_chip"], "throughput_source": throughput["source"],
            "throughput_label": throughput["label"] if throughput["source"] != "estimated" else None,
            "measured_out_ratio": None,
            **({"throughput_basis": throughput["basis"], "throughput_note": throughput.get("note")}
               if throughput.get("basis") else {}),
            **({"mix_factor": throughput["mix_factor"]} if throughput.get("mix_factor") else {})}


def _build_session_payload(name: str) -> dict:
    rd = _session_dir(name)
    if not rd.exists():
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found")
    cfg = _session_Config(name, rd)
    yaml_path = rd / "workload.yaml"
    if not yaml_path.exists() and name == "blog-example" and (BLOG_EXAMPLE_DIR / "workload.yaml").exists():
        yaml_path = BLOG_EXAMPLE_DIR / "workload.yaml"
    if yaml_path.exists():
        yaml_text = yaml_path.read_text()
        try:  # show the normalized value if the file holds an invalid redundancy multiplier (e.g. 0)
            raw_res = ((yaml.safe_load(yaml_text) or {}).get("constraints") or {}).get("resilience_multiplier", 1)
            if raw_res != cfg["constraints"]["resilience_multiplier"] and \
                    config.normalize_resilience(raw_res) != raw_res:
                yaml_text = config.to_yaml(cfg)
        except Exception:
            pass
    else:
        yaml_text = config.to_yaml(cfg)

    plan_data = _load_json(rd / "plan.json")
    endpoint_data = _load_json(rd / "endpoint.json")
    measured_data = _load_json(rd / "measured.json")
    ui_state = _load_json(rd / "session.json") or {}
    report_md = (rd / "report.md").read_text() if (rd / "report.md").exists() else None
    simulation_data = _load_json(rd / "simulation.json")

    work_cfg = _with_defaults(copy.deepcopy(cfg))
    selected, chosen, throughput, plan_err, no_fit = _selection(work_cfg, measured_data)

    dashboard_data = None
    eff = _effective_measured(measured_data, selected, throughput)
    if eff and chosen:
        try:
            prices = chosen.get("prices") or pricing.prices_for(
                work_cfg, eff["endpoint"]["accelerator_type"], eff["endpoint"]["accelerator_count"])
            dashboard_data = dashboard.build_data(work_cfg, eff, prices)
        except Exception:
            dashboard_data = None

    # Traffic preview
    try:
        t_cfg = cfg["traffic"]
        if (t_cfg.get("hourly_csv") or t_cfg.get("custom_hourly") or t_cfg.get("preset")
                or (t_cfg.get("peak_input_tps") and t_cfg.get("average_input_tps")
                    and t_cfg.get("floor_input_tps") is not None)):
            prof = [round(float(x), 1) for x in traffic.weekly_profile(t_cfg)]
        else:
            prof = [round(float(x), 1) for x in traffic.parametric(150000, 50000, 6000)]
    except Exception:
        prof = []

    return {
        "name": name,
        "example": is_example(name),
        "run_dir": str(rd),
        "config": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
        "yaml_text": yaml_text,
        "ui_state": ui_state,
        "selected": selected,
        "throughput": throughput,
        "plan_error": plan_err,
        "no_fit": no_fit,      # why nothing fits + options (planner.no_fit_explanation), with plan_error
        "plan": plan_data,
        "endpoint": endpoint_data,
        "measured": measured_data,
        "report_md": report_md,
        "simulation": simulation_data,
        "dashboard_data": dashboard_data,
        "traffic_profile": prof,
        # pinned Gemini models in the saved config that the live catalog no longer offers (replaced by their alias)
        "gemini_warnings": cfg.get("_gemini_warnings", []),
        "job": _job_payload(rd),
        "user_prompts": _prompts_summary(cfg, rd),   # your uploaded prompt set (None = synthetic prompts)
        "prompt_source": userprompts.describe_source(cfg, rd),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (ASSETS_DIR / "app.html").read_text()


@app.get("/api/auth/status")
def api_auth_status(request: Request):
    return _with_cloud(auth.status(), request)


# ---- settings: the Hugging Face token (hftoken). Stored server-side in a per-user config file (0600), never in
# sessions / runs/ / caches, never returned (only a hint). HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars win.
def _hf_settings(check: bool = True) -> dict:
    return {"hf_token": hftoken.status(check=check), "managed": CLOUD}


def _after_hf_token_change() -> dict:
    """Refetch metadata that failed for want of a token (in the background) so gated models get live values."""
    try:
        return modelinfo.refresh_after_token_change()
    except Exception as e:   # never fail the request over the refresh
        return {"invalidated": 0, "refreshing": False, "error": str(e)[:120]}


@app.get("/api/settings")
def api_settings():
    return _hf_settings()


@app.post("/api/settings/hf-token")
def api_settings_hf_token(payload: dict):
    _cloud_forbid("Saving a Hugging Face token (it comes from Secret Manager: deploy with --hf-token-secret)")
    tok = str((payload or {}).get("token") or "").strip()
    res = hftoken.validate(tok)
    if not res.get("ok"):
        raise HTTPException(status_code=400 if res.get("status") in (400, 401, 403) else 502,
                            detail=res.get("error") or "Invalid token")
    stored = hftoken.save(tok)
    refresh = _after_hf_token_change()
    st = hftoken.status(check=False)
    return {"ok": True, "user": res.get("user"), "orgs": res.get("orgs") or [], "token_hint": hftoken.hint(tok),
            "stored": stored, "active_source": st["source"],
            "overridden_by_env": st["source"] == "env", "refresh": refresh, **_hf_settings(check=False)}


@app.delete("/api/settings/hf-token")
def api_settings_hf_token_delete():
    _cloud_forbid("Removing the Hugging Face token")
    removed = hftoken.delete()
    refresh = _after_hf_token_change() if removed else {"invalidated": 0, "refreshing": False}
    return {"ok": True, "removed": removed, "refresh": refresh, **_hf_settings(check=False)}


def _auth_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


@app.post("/api/auth/login")
def api_auth_login(payload: dict, request: Request):
    """Set the project (``{project}``); ``trigger_oauth: true`` starts the browser-completed sign-in (non-blocking).
    Cloud mode: no sign-in, and the project stays the deployment's (one shared server: a switch would apply to all)."""
    project = payload.get("project")
    trigger_oauth = payload.get("trigger_oauth", False)
    mode = payload.get("mode", "adc")
    if trigger_oauth:
        _cloud_forbid("Signing in")
        return _auth_call(auth.trigger_login, project=project, mode=mode)
    if project:
        locked = _locked_project()
        if CLOUD and locked and project != locked:
            raise HTTPException(status_code=403, detail=f"This deployment always uses project {locked}.")
        return _with_cloud(_auth_call(auth.set_project, project), request)   # gcloud config is never touched
    return _with_cloud(auth.status(), request)


@app.post("/api/auth/login/start")
def api_auth_login_start(payload: dict | None = None):
    """Start ``gcloud auth [application-default] login --no-launch-browser``; returns {login_id, url, mode, resumed}."""
    _cloud_forbid("Signing in")   # on Cloud Run the app runs as its service account
    payload = payload or {}
    return _auth_call(auth.login_start, payload.get("mode", "adc"), force=bool(payload.get("force")))


def _refresh_catalogs_after_login(all_: bool = False) -> bool:
    """Rebuild cached catalogs that were built without credentials (reference prices), or all of them after an
    account switch (`all_`: they were built as another identity). True if any refresh started."""
    with _CATALOG_LOCK:
        items = list(_CATALOG_MEM.items())
    started = False
    for key, data in items:
        st = data.get("auth") or {}
        gem_fallback = any(isinstance(v, dict) and v.get("source") == "fallback" for v in (data.get("gemini") or {}).values())
        if all_ or not st.get("authenticated") or gem_fallback:
            project, _, region = key.rpartition("_")
            _refresh_catalog_bg(key, None if project == "default" else project, region or "us-central1")
            started = True
    return started


@app.post("/api/auth/login/finish")
def api_auth_login_finish(payload: dict):
    """Send the pasted code to gcloud; returns auth status plus {login_ok, login_message, catalog_refreshing}."""
    _cloud_forbid("Signing in")   # on Cloud Run the app runs as its service account
    before = auth.identity()
    res = _auth_call(auth.login_finish, payload.get("login_id") or "", payload.get("code") or "")
    st = auth.status()
    after = st.get("adc_account")
    changed = bool(res["ok"] and after and after != before)
    access = None
    if changed:
        # Another account: caches built with the previous identity go, and the app's project is re-checked so the
        # UI can ask for a project this account can use instead of silently keeping one it can't.
        _forget_identity_caches()
        access = auth.project_access(st.get("app_project"))
    refreshing = (_refresh_catalogs_after_login(all_=changed) if res["ok"] and st.get("authenticated") else False)
    return {**st, "login_ok": res["ok"], "login_message": res["message"], "login_mode": res["mode"],
            "catalog_refreshing": refreshing, "account_changed": changed, "previous_account": before,
            "project_access": access, "needs_project": bool(access and access.get("ok") is False)}


def _forget_identity_caches() -> None:
    """After an account switch: project lists, tokeninfo, Model Garden recipes looked up as the old identity."""
    auth._invalidate_caches()
    modelinfo._VERIFIED_MEM.clear()
    quotas.clear()


@app.post("/api/auth/login/cancel")
def api_auth_login_cancel(payload: dict | None = None):
    _cloud_forbid("Signing in")
    return auth.login_cancel((payload or {}).get("login_id"))


@app.get("/api/projects")
def api_projects(q: str | None = None, current: str | None = None):
    """Projects the signed-in identity can access (Resource Manager, gcloud fallback); 401 when not signed in.
    Cloud mode: just the deployment's project (the picker can't switch)."""
    locked = _locked_project()
    if locked:
        return {"projects": [auth._project_row(locked, locked, None)], "total": 1, "current": locked,
                "source": "deployment", "message": f"This deployment always uses project {locked}.", "locked": True}
    return _auth_call(auth.list_projects, q, current or _default_project())


@app.get("/api/projects/{project_id}/check")
def api_project_check(project_id: str):
    """Whether the Agent Platform and Cloud Billing APIs are enabled on the project (fails soft)."""
    return _auth_call(auth.check_project, project_id)


def _build_catalog(project: str | None, region: str, refresh: bool = False) -> dict:
    st = auth.status()
    proj = project or st.get("app_project") or "my-gcp-project"
    mod_disc = modelinfo.discover_models(proj, region, refresh=refresh)
    gem_cat = _gemini_catalog_safe(proj, region, refresh)
    gem = gemini.engine_map(gem_cat)
    if refresh:
        hardware.refresh(proj, region)
    else:
        hardware.ensure_fresh(proj, region)    # blocking only when there is no listing for this region yet
    hw = hardware.summary(region)
    return {
        "auth": st,
        "accelerators": catalog.GPUS,          # CHIP_SPECS + fallback shapes (names, kinds, memory) for labels
        "hardware": hw,                        # what the region offers (Compute Engine API) + provenance
        "model_metadata": mod_disc.get("metadata"),
        "models": mod_disc.get("models", []),
        "model_sources": mod_disc.get("sources", []),
        "families": mod_disc.get("families", []),
        "models_updated_at": mod_disc.get("updated_at"),
        "gemini": gem,                                   # cost engines: {role alias or model id: prices}
        "gemini_models": gemini.models_list(gem_cat),    # every live Gemini text model, newest first per family
        "gemini_aliases": gem_cat["aliases"],            # role alias -> the model it resolves to + rule
        "gemini_meta": gem_cat["meta"],                  # source, updated_at, Model Garden / Billing status
        "maas_presets": catalog.maas_presets(gem),
        "pricing_labels": costmodel.TIERS,
        "format": CATALOG_FORMAT,
        "regions": catalog.REGION_NAMES,
        "defaults": config.DEFAULTS,
        "cached_at": time.time(),
    }


def _gemini_catalog_safe(project: str | None, region: str, refresh: bool = False) -> dict:
    try:
        return gemini.get(project, region, refresh=refresh)
    except Exception:
        return gemini.build(None, None)


def _gemini_payload(project: str | None, region: str, refresh: bool = False) -> dict:
    cat = _gemini_catalog_safe(project, region, refresh)
    return {"gemini": gemini.engine_map(cat), "gemini_models": gemini.models_list(cat),
            "gemini_aliases": cat["aliases"], "gemini_meta": {**cat["meta"], "refreshing": gemini.refreshing()}}


def _catalog_file(key: str) -> Path:
    return CACHE_DIR / f"catalog_{_sanitize_name(key)}.json"


def _store_catalog(key: str, data: dict) -> None:
    with _CATALOG_LOCK:
        _CATALOG_MEM[key] = data
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _catalog_file(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str))
        tmp.replace(_catalog_file(key))
    except Exception:
        pass


def _refresh_catalog_bg(key: str, project: str | None, region: str) -> None:
    with _CATALOG_LOCK:
        if key in _CATALOG_REFRESHING:
            return
        _CATALOG_REFRESHING.add(key)

    def work():
        try:
            _store_catalog(key, _build_catalog(project, region))
        except Exception:
            pass
        finally:
            with _CATALOG_LOCK:
                _CATALOG_REFRESHING.discard(key)

    threading.Thread(target=work, daemon=True).start()


CATALOG_FORMAT = 6   # bump when the catalog shape changes; older caches are rebuilt (3: INT4, B200; 4: live HF + Compute; 5: MXFP4; 6: live Gemini catalog)


def _on_hf_update() -> None:
    """New Hugging Face metadata landed (hfmeta background refresh): rebuild the cached catalogs in the background
    so the model drawer shows it; the UI keeps the old catalog until then."""
    with _CATALOG_LOCK:
        keys = list(_CATALOG_MEM)
    for key in keys:
        project, _, region = key.rpartition("_")
        _refresh_catalog_bg(key, None if project == "default" else project, region or "us-central1")


hfmeta.ON_UPDATE.append(_on_hf_update)


def _on_gemini_update() -> None:
    """A background Gemini catalog refresh landed: swap the new Gemini fields into the cached catalogs (cheap; no
    model / hardware rebuild)."""
    with _CATALOG_LOCK:
        items = list(_CATALOG_MEM.items())
    for key, data in items:
        project, _, region = key.rpartition("_")
        try:
            out = _gemini_payload(None if project == "default" else project, region or "us-central1")
            _store_catalog(key, {**data, **out, "maas_presets": catalog.maas_presets(out["gemini"])})
        except Exception:
            pass


gemini.ON_UPDATE.append(_on_gemini_update)


def cached_catalog(project: str | None, region: str) -> dict:
    """Catalog from memory or ~/.cache/aiops-dilemma (instant); rebuilt in the background when stale."""
    key = f"{project or 'default'}_{region}"
    with _CATALOG_LOCK:
        data = _CATALOG_MEM.get(key)
    if data is None and _catalog_file(key).exists():
        try:
            data = json.loads(_catalog_file(key).read_text())
            with _CATALOG_LOCK:
                _CATALOG_MEM[key] = data
        except Exception:
            data = None
    if data is not None and data.get("format") != CATALOG_FORMAT:
        data = None
    if data is None:
        data = _build_catalog(project, region)
        _store_catalog(key, data)
    elif time.time() - float(data.get("cached_at") or 0) > CATALOG_TTL_S:
        _refresh_catalog_bg(key, project, region)
    with _CATALOG_LOCK:
        refreshing = key in _CATALOG_REFRESHING
    if gemini.cached() is not None:
        gemini.get(project, region)            # instant; starts a background refresh when the Gemini list is a day old
    gem_meta = {**(data.get("gemini_meta") or {}), "refreshing": gemini.refreshing()}
    return {**data, "gemini_meta": gem_meta, "pricing_labels": costmodel.TIERS, "traffic_presets": traffic.preset_list(),
            "serving_images": catalog.SERVING_IMAGES, "cache": {"age_s": round(time.time() - float(data.get("cached_at") or 0), 1),
                              "refreshing": refreshing}}


def _reject_bad_region(cfg: dict) -> dict:
    """A config submitted from the browser (YAML editor, save, estimate, run) with an invalid region: say so (400)
    instead of silently pricing and deploying in the default region."""
    if cfg.get("_region_warning"):
        raise HTTPException(status_code=400, detail=cfg["_region_warning"][0].upper() + cfg["_region_warning"][1:])
    if cfg.get("_serving_errors"):   # custom serving container fields (serving.normalize)
        raise HTTPException(status_code=400, detail="Serving container: " + "; ".join(cfg["_serving_errors"]))
    return cfg


def _region_param(value) -> str:
    """A region from a request: becomes part of API host names (with credentials), so only real region names pass."""
    region = value or regions.DEFAULT
    if not regions.is_valid(region):
        raise HTTPException(status_code=400, detail=f"Not a Google Cloud region: {str(region)[:40]!r}")
    return region


@app.get("/api/catalog")
def api_catalog(project: str | None = None, region: str = "us-central1"):
    return cached_catalog(project, _region_param(region))


@app.post("/api/catalog/refresh")
def api_catalog_refresh(payload: dict):
    project = payload.get("project")
    region = _region_param(payload.get("region"))
    data = _build_catalog(project, region, refresh=True)
    _store_catalog(f"{project or 'default'}_{region}", data)
    return {**data, "serving_images": catalog.SERVING_IMAGES, "cache": {"age_s": 0.0, "refreshing": False}}


@app.post("/api/gemini/refresh")
def api_gemini_refresh(payload: dict | None = None):
    """Re-read Gemini models and prices now (Cloud Billing Catalog + Model Garden) and update the cached catalogs."""
    payload = payload or {}
    project = payload.get("project")
    region = _region_param(payload.get("region"))
    out = _gemini_payload(project or _default_project(), region, refresh=True)
    key = f"{project or 'default'}_{region}"
    with _CATALOG_LOCK:
        data = _CATALOG_MEM.get(key)
    if data is not None:
        _store_catalog(key, {**data, **out, "maas_presets": catalog.maas_presets(out["gemini"])})
    return out


@app.post("/api/models/inspect")
def api_inspect_model(payload: dict):
    hf_id = (payload.get("hf_id") or "").strip()
    if not hf_id:
        raise HTTPException(status_code=400, detail="hf_id is required")
    return modelinfo.inspect_hf_model(hf_id)


@app.get("/api/traffic/presets")
def api_traffic_presets():
    """Named traffic shapes (traffic.preset): key, name, one-line description, peak / average."""
    return {"presets": traffic.preset_list()}


@app.post("/api/traffic/preview")
def api_traffic_preview(payload: dict):
    t = payload.get("traffic") or payload
    try:
        t = dict(t)
        if t.get("hourly_csv") and not Path(t["hourly_csv"]).exists():
            t.pop("hourly_csv")
        t.setdefault("peak_input_tps", 150000); t.setdefault("average_input_tps", 50000)
        t.setdefault("floor_input_tps", 6000)
        cn = config.normalize({"traffic": t})    # custom_hourly, preset, monthly anchor, clamps
        tn = cn["traffic"]
        arr = traffic.weekly_profile(tn)
        return {
            "ok": True,
            "preset": tn.get("preset"),
            "warnings": cn.get("_warnings", []),
            "profile": [round(float(x), 1) for x in arr],
            "peak_input_tps": round(float(arr.max())),
            "average_input_tps": round(float(arr.mean())),
            "floor_input_tps": round(float(arr.min())),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "profile": []}


def _session_cfg_or_none(name: str | None) -> dict | None:
    if not name:
        return None
    try:
        rd = _session_dir(name)
        return _session_Config(_sanitize_name(name), rd) if rd.exists() else None
    except HTTPException:
        return None


SPEC_BASIS = "spec estimate"
CALIBRATED_BASIS = "spec estimate × k from your selected model"


def _model_economics(base_cfg: dict, rec: dict, gem: dict, k: float = 1.0) -> dict:
    """Est. chips and all-in $/M input tokens for a model at its recommended precision.

    Throughput = spec estimate (a short-prompt rate) × `k`, the session's calibration factor (1.0 when the session
    itself runs on a spec estimate), so alternatives are compared on the same footing as the selected model.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["project"] = "my-gcp-project"  # skip Model Garden lookups; prices still come from the billing cache
    cfg["deployment"] = {**cfg.get("deployment", {}), "mode": "auto"}
    cfg["model"] = {**cfg["model"], "model_garden_id": rec.get("model_garden_id"), "hf_id": rec.get("hf_id"),
                    "params_b": rec.get("params_b"), "active_params_b": rec.get("active_params_b"),
                    "precision": rec.get("recommended_precision", "bf16")}
    try:
        ch = planner.plan(cfg)["chosen"]
    except (Exception, SystemExit):
        return {"est_hardware": None, "est_chips": None, "est_usd_per_m_input": None,
                "est_throughput_basis": SPEC_BASIS if k == 1.0 else CALIBRATED_BASIS}
    spec = float(ch.get("est_input_tps_per_gpu") or 0)
    eco = planner.fleet_economics(cfg, ch, per_chip=spec * k if spec > 0 else None, gemini_catalog=gem,
                                  mix_factor_ref=1.0)
    return {"est_hardware": _hw(ch), "est_chips": eco.get("fleet_gpus"), "est_usd_per_m_input": eco.get("usd_per_m_input"),
            "est_input_tps_per_gpu": round(spec * k, 0) if spec > 0 else None,
            "est_basis": ch.get("est_basis"), "est_basis_note": ch.get("est_basis_note"),   # estimate tier
            "est_throughput_basis": SPEC_BASIS if k == 1.0 else CALIBRATED_BASIS}


@app.post("/api/recommend/model")
def api_recommend_model(payload: dict):
    proj = payload.get("project") or _default_project() or "my-gcp-project"
    region = _region_param(payload.get("region"))
    sess_cfg = _session_cfg_or_none(payload.get("session"))
    selected_model = payload.get("selected_model") or payload.get("model") or (sess_cfg or {}).get("model")
    src = sess_cfg or {}
    res = recommender.recommend(
        use_case=payload.get("use_case") or src.get("use_case") or {},
        requests_cfg=payload.get("requests") or src.get("requests") or {},
        traffic_cfg=payload.get("traffic") or src.get("traffic") or {},
        constraints_cfg=payload.get("constraints") or src.get("constraints") or {},
        hybrid_cfg=payload.get("hybrid") or src.get("hybrid") or {},
        selected_model=selected_model,
        project=proj,
        region=region,
    )
    if payload.get("with_economics", True):
        base = config.normalize({
            "region": region, "project": "my-gcp-project",
            "requests": payload.get("requests") or src.get("requests") or {},
            "traffic": payload.get("traffic") or src.get("traffic") or {},
            "maas": payload.get("maas") or src.get("maas") or {},
            "constraints": payload.get("constraints") or src.get("constraints") or {},
        })
        base = _with_defaults(base)
        gem = pricing.gemini_prices(proj, region)
        selected_eco = _selected_model_economics(payload.get("session"), sess_cfg, gem)
        k = (selected_eco or {}).get("calibration_k") or 1.0
        res["calibration_k"] = round(k, 4)
        for rec in res["recommendations"]:
            if rec.get("is_selected") and selected_eco:
                rec.update(selected_eco)   # same fleet as the verdict: session hardware, precision, throughput
            else:
                rec.update(_model_economics(base, rec, gem, k=k))
    return res


def _selected_model_economics(session: str | None, sess_cfg: dict | None, gem: dict) -> dict | None:
    """Economics for the session's own model, using its selected hardware and single-source throughput."""
    if not session or not sess_cfg:
        return None
    rd = _session_dir(_sanitize_name(session))
    work = _with_defaults(copy.deepcopy(sess_cfg))
    _, chosen, throughput, _, _ = _selection(work, _load_json(rd / "measured.json"))
    if not chosen or not throughput.get("per_chip"):
        return None
    f_ref = costmodel.ref_mix_factor(work, throughput)
    eco = planner.fleet_economics(work, chosen, per_chip=throughput["per_chip"], gemini_catalog=gem,
                                  mix_factor_ref=f_ref)
    # Calibration factor for the other models: the session's short-prompt rate (per_chip x f_ref) over the spec
    # estimate (also a short-prompt rate) for the same hardware. 1.0 when the session runs on the spec estimate.
    k = 1.0
    acc, cnt = chosen.get("accelerator_type"), int(chosen.get("accelerator_count") or 1)
    if throughput.get("source") != "estimated" and acc in catalog.GPUS:
        # same basis the other models' estimates use (reference-anchored when a reference benchmark exists)
        spec = planner.reference_throughput(work, acc, cnt)["per_chip"]
        if spec > 0:
            k = float(throughput["per_chip"]) * f_ref / spec
    return {"est_hardware": _hw(chosen), "est_chips": eco.get("fleet_gpus"), "est_usd_per_m_input": eco.get("usd_per_m_input"),
            "est_input_tps_per_gpu": throughput["per_chip"], "est_throughput_source": throughput.get("source"),
            "est_throughput_basis": "your session throughput", "calibration_k": k}


@app.post("/api/recommend/hardware")
def api_recommend_hardware(payload: dict):
    raw_cfg = copy.deepcopy(payload.get("config") or {k: v for k, v in payload.items() if k not in ("session", "accelerator_filter")})
    acc_filter = payload.get("accelerator_filter") or (raw_cfg.get("deployment") or {}).get("accelerator_filter") or "all"
    cfg = _with_defaults(config.normalize(raw_cfg))
    throughput, selected_hw = None, None
    sess = payload.get("session")
    if sess:
        try:
            sp = _build_session_payload(_sanitize_name(sess))
            same_model = (sp["selected"]["model"]["model_garden_id"] == cfg["model"].get("model_garden_id")
                          and sp["selected"]["precision"] == cfg["model"].get("precision"))
            if same_model:
                throughput, selected_hw = sp["throughput"], sp["selected"]["hardware"]
        except HTTPException:
            pass
    # The region's accelerators / machine shapes (Compute Engine API): fetched once per region (~1-3 s), then cached.
    hardware.ensure_fresh(auth.real_project(cfg.get("project")) or _default_project(), cfg.get("region", "us-central1"))
    try:
        p = planner.plan(cfg, accelerator_filter=acc_filter)
    except planner.NoFitError as e:
        return {"ok": False, "error": str(e), "plan": None, "no_fit": e.info,
                "availability": planner.availability(cfg.get("region", "us-central1"))}
    except (Exception, SystemExit) as e:
        return {"ok": False, "error": str(e), "plan": None}
    selected_hw = selected_hw or _hw(p["chosen"])
    gem = pricing.gemini_prices(cfg.get("project"), cfg.get("region", "us-central1"))
    p["candidates"] = planner.enrich_candidates(cfg, p["candidates"], gemini_catalog=gem,
                                                throughput=throughput, selected=selected_hw)
    match = next((c for c in p["candidates"] if _same_hw(c, p["chosen"])), None)
    if match:
        p["chosen"] = match
    else:
        p["chosen"] = planner.enrich_candidates(cfg, [p["chosen"]], gemini_catalog=gem, throughput=throughput,
                                                selected=selected_hw)[0]
    return {"ok": True, "plan": p, "pricing_labels": costmodel.TIERS, "sort_key": "usd_per_m_input",
            "sort_tier": "vertex_cud_3y", "quota": _region_quota(cfg, p["candidates"] + [p["chosen"]])}


def _region_quota(cfg: dict, cands: list[dict]) -> dict:
    """The project's Agent Platform serving quota in the region for each candidate accelerator (quotas.for_region)."""
    proj = auth.real_project(cfg.get("project")) or _default_project()
    try:
        return quotas.for_region(proj, cfg.get("region", "us-central1"), [c["accelerator_type"] for c in cands])
    except Exception as e:  # never break the hardware drawer over quotas
        return {"ok": False, "error": str(e), "reason": "error"}


@app.get("/api/sessions")
def api_list_sessions():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dirs = []
    if (RUNS_DIR / "blog-example").exists():
        dirs.append(("blog-example", RUNS_DIR / "blog-example", False))
    elif (BLOG_EXAMPLE_DIR / "runs" / "blog-example").exists():
        dirs.append(("blog-example", BLOG_EXAMPLE_DIR / "runs" / "blog-example", True))
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and d.name != "blog-example":
            dirs.append((d.name, d, False))
    out = []
    for name, d, is_demo in dirs:
        try:
            cfg = _session_Config(name, d)
        except Exception:
            continue
        meas = _load_json(d / "measured.json")
        ep = _load_json(d / "endpoint.json")
        job = _job_payload(d)
        out.append({
            "name": name,
            "example": is_example(name),
            "is_demo": is_demo,
            "updated_at": int(d.stat().st_mtime),
            "model": cfg["model"].get("model_garden_id"),
            "precision": cfg["model"].get("precision", "bf16"),
            "accelerator": (meas or {}).get("endpoint", {}).get("accelerator_type") or cfg["deployment"].get("accelerator_type") or "auto",
            "has_plan": (d / "plan.json").exists(),
            "has_endpoint": bool(ep),
            "endpoint_info": ep,
            "has_measured": report.is_measured(meas),
            "has_estimate": bool(meas) and not report.is_measured(meas),
            "has_report": (d / "report.md").exists(),
            "job_status": job["status"],
            "job_source": job.get("source"),
            "job_endpoint_warning": job["endpoint_warning"],
        })
    return {"sessions": out}


@app.get("/api/sessions/{name}")
def api_get_session(name: str):
    return _build_session_payload(name)


COPY_ARTIFACTS = ("workload.yaml", "plan.json", "measured.json", "report.md", "dashboard.html", "session.json",
                  userprompts.STORE, userprompts.META, userprompts.SYSTEM_STORE)


@app.post("/api/sessions/{name}")
def api_save_session(name: str, payload: dict):
    clean_name = _sanitize_name(name)
    rd = _session_dir(clean_name, for_write=True)
    copy_of = payload.get("copy_of")
    if copy_of:
        src_name = _sanitize_name(copy_of)
        src = _session_dir(src_name)
        if not src.exists() and src_name not in EXAMPLE_SESSIONS:
            raise HTTPException(status_code=404, detail=f"Session '{src_name}' not found")
        if rd.exists() and any(rd.iterdir()):
            raise HTTPException(status_code=409, detail=f"Session '{clean_name}' already exists; pick another name")
        rd.mkdir(parents=True, exist_ok=True)
        for f in COPY_ARTIFACTS:
            if (src / f).exists():
                shutil.copy2(src / f, rd / f)
        if not (rd / "workload.yaml").exists():
            src_cfg = _session_Config(src_name, src)
            (rd / "workload.yaml").write_text(config.to_yaml(src_cfg))
        if not payload.get("config"):
            raw = yaml.safe_load((rd / "workload.yaml").read_text()) or {}
            raw["name"] = clean_name
            raw["project"] = raw.get("project") or None
            cfg = config.normalize(raw, path=str(rd / "workload.yaml"))
            (rd / "workload.yaml").write_text(config.to_yaml(cfg))
        ui = dict(payload.get("ui_state") or _load_json(rd / "session.json") or {})
        ui["copied_from"] = src_name
        (rd / "session.json").write_text(json.dumps(ui, indent=2))
        if not payload.get("config"):
            return _build_session_payload(clean_name)
    rd.mkdir(parents=True, exist_ok=True)

    raw_cfg = payload.get("config") or {}
    raw_cfg["name"] = clean_name
    cfg = _reject_bad_region(config.normalize(raw_cfg, path=str(rd / "workload.yaml")))
    yaml_text = config.to_yaml(cfg)
    (rd / "workload.yaml").write_text(yaml_text)

    if "ui_state" in payload and not copy_of:
        (rd / "session.json").write_text(json.dumps(payload["ui_state"], indent=2))
    return _build_session_payload(clean_name)


@app.post("/api/sessions/{name}/yaml")
def api_save_session_yaml(name: str, payload: dict):
    clean_name = _sanitize_name(name)
    rd = _session_dir(clean_name, for_write=True)
    rd.mkdir(parents=True, exist_ok=True)
    yaml_text = payload.get("yaml_text") or ""
    try:
        raw = yaml.safe_load(yaml_text) or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML syntax: {e}")
    raw["name"] = clean_name
    cfg = _reject_bad_region(config.normalize(raw, path=str(rd / "workload.yaml")))
    (rd / "workload.yaml").write_text(config.to_yaml(cfg))
    return _build_session_payload(clean_name)


@app.post("/api/sessions/{name}/upload")
async def api_upload_files(
    name: str,
    files: list[UploadFile] = File(...),
    role_hint: str = Form("auto"),
    hf_id: str | None = Form(None),
):
    clean_name = _sanitize_name(name)
    rd = _session_dir(clean_name, for_write=True)
    rd.mkdir(parents=True, exist_ok=True)
    file_tuples = []
    for uf in files:
        content = await uf.read()
        file_tuples.append((uf.filename or "upload.csv", content, role_hint))
    try:
        res = ingest.ingest_files(file_tuples, rd, hf_id=hf_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if res.get("prompts_upload"):   # prompt files: same storage + config as POST /prompts
        cfg = _set_prompt_source(clean_name, rd, "file")
        res["prompts_summary"] = _prompts_summary(cfg, rd)
    return {"ok": True, "ingested": res}


# ------------------------------------------------------------------ bring your own prompts (userprompts)
def _model_context(cfg: dict) -> int | None:
    m = cfg.get("model") or {}
    v = m.get("max_model_len") or m.get("context_length")
    if v:
        return int(v)
    try:
        hit = modelinfo.find_model(m, modelinfo.discover_models(None, refresh=False).get("models", []))
        return int(hit["max_context"]) if hit and hit.get("max_context") else None
    except Exception:
        return None


def _prompts_summary(cfg: dict, rd: Path) -> dict | None:
    if (cfg.get("prompts") or {}).get("source") != "file":
        return None
    try:
        return userprompts.summary(rd, cfg, _model_context(cfg))
    except Exception:
        return None


def _tokenizer_dir(hf_id: str | None) -> Path:
    return CACHE_DIR / "tokenizers" / re.sub(r"[^A-Za-z0-9_.-]+", "__", hf_id or "none")


def _set_prompt_source(clean: str, rd: Path, source: str) -> dict:
    """prompts.source / prompts.file in workload.yaml (the rest of the saved config untouched)."""
    raw = {k: v for k, v in _session_Config(clean, rd).items() if not str(k).startswith("_")}
    raw["name"] = clean
    p = raw.setdefault("prompts", {})
    if source == "file":
        p.update(source="file", file=userprompts.STORE)
    else:
        p.update(source="synthetic", file=None)
    cfg = config.normalize(raw, path=str(rd / "workload.yaml"))
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "workload.yaml").write_text(config.to_yaml(cfg))
    cfg["_run_dir"] = str(rd)
    return cfg


def _prompts_response(clean: str, rd: Path, cfg: dict) -> dict:
    return {"source": cfg["prompts"].get("source"), "file": cfg["prompts"].get("file"),
            "summary": _prompts_summary(cfg, rd), "prompt_source": userprompts.describe_source(cfg, rd)}


@app.post("/api/sessions/{name}/prompts")
async def api_upload_prompts(name: str, request: Request):
    """Your prompts for the benchmark: multipart `files` (+ `text_mode` separator | blocks | lines for .txt/.md) or
    JSON {text, mode}. Parses, stores runs/<s>/uploaded_prompts.jsonl, sets prompts.source=file + prompts.file
    (relative), returns the summary (incl. the proposed request sizes). Replaces an earlier upload."""
    clean = _sanitize_name(name)
    rd = _session_dir(clean, for_write=True)
    if int(request.headers.get("content-length") or 0) > userprompts.MAX_UPLOAD_BYTES + 1024 * 1024:
        raise HTTPException(status_code=413, detail="Files are too large (limit 50 MB). Upload a representative sample.")
    cfg = _session_Config(clean, rd)
    ctype = request.headers.get("content-type", "")
    try:
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            files = [(getattr(f, "filename", None) or "prompts.txt", await f.read())
                     for f in form.getlist("files") if hasattr(f, "read")]
            if not files:
                raise HTTPException(status_code=400, detail="Attach one or more prompt files as 'files'.")
            hf = form.get("hf_id") or cfg["model"].get("hf_id")
            tc, method = await run_in_threadpool(userprompts.token_counter, hf, _tokenizer_dir(hf))
            await run_in_threadpool(userprompts.store_files, rd, files, str(form.get("text_mode") or "separator"),
                                    hf, tc, method)
        else:
            try:
                payload = await request.json()
            except ValueError:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            text = str(payload.get("text") or "")
            if not text.strip():
                raise HTTPException(status_code=400, detail="Paste at least one prompt, or attach files.")
            hf = payload.get("hf_id") or cfg["model"].get("hf_id")
            tc, method = await run_in_threadpool(userprompts.token_counter, hf, _tokenizer_dir(hf))
            await run_in_threadpool(userprompts.store_text, rd, text, str(payload.get("mode") or "separator"),
                                    hf, tc, method)
    except userprompts.PromptUploadError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    cfg = _set_prompt_source(clean, rd, "file")
    return _prompts_response(clean, rd, cfg)


@app.get("/api/sessions/{name}/prompts")
def api_get_prompts(name: str):
    clean = _sanitize_name(name)
    rd = _session_dir(clean)
    if not rd.exists() and clean not in EXAMPLE_SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{clean}' not found")
    cfg = _session_Config(clean, rd)
    return _prompts_response(clean, rd, cfg)


@app.delete("/api/sessions/{name}/prompts")
def api_delete_prompts(name: str):
    """Remove the uploaded prompt set: back to synthetic prompts shaped like the request sizes."""
    clean = _sanitize_name(name)
    rd = _session_dir(clean, for_write=True)
    userprompts.remove(rd)
    cfg = _set_prompt_source(clean, rd, "synthetic")
    return _prompts_response(clean, rd, cfg)


@app.post("/api/sessions/{name}/serving/import")
def api_import_tuned(name: str, payload: dict):
    """Read a tuned serving config (alpha_inference best_candidate.py, or the same dict as JSON) and return the
    deployment fields it maps to, a summary and warnings against this workload. Parsed, never run; nothing is saved
    (the app applies the fields to the session and saves as usual)."""
    clean_name = _sanitize_name(name)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="No file content")
    if len(text.encode()) > serving.MAX_CANDIDATE:
        raise HTTPException(status_code=413, detail="The file is larger than 1 MB")
    if payload.get("config"):
        raw = dict(payload["config"])
        raw["name"] = clean_name
        cfg = config.normalize(raw)
    else:
        cfg = _session_Config(clean_name, _session_dir(clean_name))
    image = payload.get("image") or None
    if image is not None and not serving.valid_image_ref(str(image)):
        raise HTTPException(status_code=400, detail="Not an Artifact Registry, Container Registry or Docker Hub image")
    try:
        return serving.import_candidate(text, cfg, filename=str(payload.get("filename") or "")[:200], image=image,
                                        cloud=CLOUD)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Can't import this file: {e}")


@app.post("/api/uploads/classify")
async def api_classify_uploads(files: list[UploadFile] = File(...)):
    """What each file holds (time series, request timestamps, prompts, system prompts, token counts, description), so
    each rail section takes its part and points to the right section for the rest. Nothing is stored."""
    out = []
    for uf in files:
        raw = await uf.read(userprompts.MAX_UPLOAD_BYTES + 1)
        if len(raw) > userprompts.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{uf.filename}: too large (limit 50 MB).")
        out.append(await run_in_threadpool(userprompts.classify, uf.filename or "upload", raw))
    return {"files": out}


@app.post("/api/sessions/{name}/system-prompts")
async def api_parse_system_prompts(name: str, files: list[UploadFile] = File(...)):
    """System prompts read from files (text split on `---` lines, a JSON array of strings, or a request log's system
    messages / column), with their average size. Nothing is stored or applied: the System prompt section shows them
    for review."""
    clean = _sanitize_name(name)
    rd = _session_dir(clean)
    cfg = _session_Config(clean, rd) if rd.exists() or clean in EXAMPLE_SESSIONS else config.normalize({"name": clean})
    texts, info = [], []
    for uf in files:
        raw = await uf.read(userprompts.MAX_UPLOAD_BYTES + 1)
        if len(raw) > userprompts.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{uf.filename}: too large (limit 50 MB).")
        fname = uf.filename or "system.txt"
        kind = await run_in_threadpool(userprompts.classify, fname, raw)
        if "description" in kind["kinds"]:
            raise HTTPException(status_code=422, detail=f"{fname} looks like a workload description: use Upload in Traffic.")
        if "timeseries" in kind["kinds"] or "hourly_traffic" in kind["kinds"]:
            raise HTTPException(status_code=422, detail=f"{fname} looks like a traffic time series: use Upload in Traffic.")
        try:
            found = await run_in_threadpool(userprompts.system_prompts_from, fname, raw)
        except userprompts.PromptUploadError as e:
            raise HTTPException(status_code=e.status, detail=str(e))
        if not found:
            hint = " It holds prompts without system prompts: use Upload in Prompts." if kind["prompts"] else ""
            raise HTTPException(status_code=422, detail=f"{fname}: no system prompts found.{hint}")
        texts.extend(t for t in found if t not in texts)
        info.append(kind)
    texts = texts[:50]
    hf = cfg["model"].get("hf_id")
    tc, method = await run_in_threadpool(userprompts.token_counter, hf, _tokenizer_dir(hf))
    sizes = [tc.count(t) for t in texts]
    return {"system_prompts": texts, "count": len(texts), "avg_tokens": round(sum(sizes) / max(1, len(sizes))),
            "token_method": method, "files": info}


# ----------------------------------------------------------------- traffic time-series import (review before applying)
TRAFFIC_IMPORT_FILE = "traffic_import.json"     # parsed series of the last import (runs/<s>/, never tracked)
TRAFFIC_IMPORT_MAX_BYTES = 50 * 1024 * 1024


def _import_context(name: str, ctx: Any) -> dict:
    """Workload context for an import: the client's current (possibly unsaved) week, requests and weekend factor,
    else the saved session's."""
    ctx = dict(ctx) if isinstance(ctx, dict) else {}
    if ctx.get("requests") and ctx.get("current_profile") is not None:
        return ctx
    try:
        cfg = _session_Config(name, _session_dir(name))
        ctx.setdefault("requests", cfg.get("requests") or {})
        ctx.setdefault("weekend_factor", (cfg.get("traffic") or {}).get("weekend_factor"))
        if ctx.get("current_profile") is None:
            try:
                ctx["current_profile"] = [float(x) for x in traffic.weekly_profile(_with_defaults(cfg)["traffic"])]
            except Exception:
                ctx["current_profile"] = None
    except Exception:
        pass
    return ctx


def _json_field(v: Any, default: Any) -> Any:
    if v is None or v == "":
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def _gaps_choice(payload: dict, stored: dict) -> bool:
    """The user's "gaps are zero traffic" choice, else the import's default (True for Metrics Explorer exports)."""
    if "gaps_as_zero" in payload and payload["gaps_as_zero"] is not None:
        return bool(payload["gaps_as_zero"])
    return bool(stored.get("gaps_default"))


def _proposal_from_stored(stored: dict, ctx: dict, mapping: dict | None, option: str | None,
                          gaps_as_zero: bool) -> dict:
    series = ingest.series_from_json(stored.get("series") or [])
    win = stored.get("window")
    window = (ingest.datetime.fromisoformat(win[0]), ingest.datetime.fromisoformat(win[1])) if win else None
    res = ingest.build_traffic_proposal(series, mapping=mapping, context=ctx, option=option,
                                        missing_is_zero=bool(stored.get("missing_is_zero")), window=window,
                                        gaps_as_zero=gaps_as_zero)
    res["source"] = stored.get("source") or {}
    res["role_labels"] = ingest.TS_ROLE_LABELS
    return res


def _store_import(rd: Path, series: list[dict], source: dict, missing_is_zero: bool = False,
                  window: tuple | None = None) -> dict:
    stored = {"created": time.time(), "source": source, "missing_is_zero": missing_is_zero,
              "window": [window[0].isoformat(), window[1].isoformat()] if window else None,
              "series": ingest.series_to_json(series)}
    rd.mkdir(parents=True, exist_ok=True)
    tmp = rd / (TRAFFIC_IMPORT_FILE + ".tmp")
    tmp.write_text(json.dumps(stored))
    tmp.replace(rd / TRAFFIC_IMPORT_FILE)
    return stored


@app.post("/api/sessions/{name}/traffic/import")
async def api_traffic_import(name: str, request: Request):
    """Parse time-series files (multipart `files`, optional `context`/`mapping`/`option`/`gaps_as_zero` fields) or,
    with a JSON body {mapping, option, context, gaps_as_zero}, re-map the last import. Returns the proposal (series
    with guesses, coverage, options, derived values, warnings) without applying anything."""
    clean = _sanitize_name(name)
    rd = _session_dir(clean, for_write=True)
    ctype = request.headers.get("content-type", "")
    if int(request.headers.get("content-length") or 0) > TRAFFIC_IMPORT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Files are too large (limit 50 MB).")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        files = []
        for f in form.getlist("files"):
            if hasattr(f, "read"):
                files.append((getattr(f, "filename", None) or "upload.csv", await f.read()))
        if not files:
            raise HTTPException(status_code=400, detail="Attach one or more time-series files as 'files'.")
        series, other = ingest.import_timeseries_files(files)
        if not series:
            raise HTTPException(status_code=422, detail="No time series found: these files have no timestamp column "
                                                        "with numeric values. Upload them as logs or prompts instead.")
        stored = _store_import(rd, series, {"kind": "files", "files": [f for f, _ in files if f not in other]})
        # Metrics Explorer exports leave out intervals with no calls, so their gaps are zero traffic by default
        # (the review sheet's checkbox overrides it); other files keep gaps as missing unless asked.
        me_export = any(b"TimeSeries ID" in raw[:4000] for f, raw in files if f not in other)
        if me_export:   # remembered so re-maps and the apply step use the same default
            stored["gaps_default"] = True
            (rd / TRAFFIC_IMPORT_FILE).write_text(json.dumps(stored))
        gz = form.get("gaps_as_zero")
        gaps = me_export if gz in (None, "") else str(gz).lower() in ("1", "true", "yes")
        res = _proposal_from_stored(stored, _import_context(clean, _json_field(form.get("context"), {})),
                                    _json_field(form.get("mapping"), {}), form.get("option") or None, gaps)
        if me_export and gz in (None, ""):
            res.setdefault("warnings", []).append(
                "Cloud Monitoring exports leave out intervals with no requests, so gaps are treated as zero traffic. "
                "Untick \"Gaps are zero traffic\" if the gaps were outages or missing data.")
        res["other_files"] = other
        return res
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    stored = _load_json(rd / TRAFFIC_IMPORT_FILE)
    if not stored:
        raise HTTPException(status_code=404, detail="Nothing imported yet: upload time-series files first.")
    return _proposal_from_stored(stored, _import_context(clean, payload.get("context")), payload.get("mapping") or {},
                                 payload.get("option"), _gaps_choice(payload, stored))


@app.post("/api/sessions/{name}/traffic/apply")
def api_traffic_apply(name: str, payload: dict):
    """Apply the reviewed import: {mapping, option, gaps_as_zero, accept: [derived paths], context?, config?}.
    Writes traffic.custom_hourly (+ accepted requests.* values) into workload.yaml; `config` (the client's current,
    possibly unsaved config) is used as the base when given."""
    clean = _sanitize_name(name)
    rd = _session_dir(clean, for_write=True)
    stored = _load_json(rd / TRAFFIC_IMPORT_FILE)
    if not stored:
        raise HTTPException(status_code=404, detail="Nothing imported yet: upload time-series files first.")
    res = _proposal_from_stored(stored, _import_context(clean, payload.get("context")), payload.get("mapping") or {},
                                payload.get("option"), _gaps_choice(payload, stored))
    if res.get("error") or not res.get("options"):
        raise HTTPException(status_code=400, detail=res.get("error") or "Nothing to apply.")
    opt = next(o for o in res["options"] if o["key"] == res["option"])
    if payload.get("config"):
        raw = copy.deepcopy(payload["config"])
    elif (rd / "workload.yaml").exists():
        raw = yaml.safe_load((rd / "workload.yaml").read_text()) or {}
    else:
        raw = {k: v for k, v in _session_Config(clean, rd).items() if not str(k).startswith("_")}
    raw["name"] = clean
    t = raw.setdefault("traffic", {}) or {}
    raw["traffic"] = t
    t.update({"custom_hourly": opt["custom_hourly"], "hourly_csv": None, "preset": None, "monthly_input_tokens": None,
              "peak_input_tps": opt["peak_input_tps"], "average_input_tps": opt["average_input_tps"],
              "floor_input_tps": opt["floor_input_tps"]})
    accept = set(payload.get("accept") or [])
    applied = {"traffic.custom_hourly": f"{res['option']}: peak {opt['peak_input_tps']:,.0f} tok/s"}
    for d in res.get("derived") or []:
        if d["path"] in accept:
            sec, key = d["path"].split(".", 1)
            raw.setdefault(sec, {})[key] = d["value"]
            applied[d["path"]] = d["value"]
    cfg = config.normalize(raw, path=str(rd / "workload.yaml"))
    (rd / "workload.yaml").write_text(config.to_yaml(cfg))
    return {"ok": True, "applied": applied, "option": res["option"], "session": _build_session_payload(clean)}


@app.get("/api/monitoring/sources")
def api_monitoring_sources(project: str | None = None, days: int = 28):
    """Models (publisher / MaaS) and endpoints that served requests in the last `days` (7, 14 or 28)."""
    from aiops_dilemma import monitoring

    project = project or _default_project()
    try:
        return monitoring.list_sources(project, days)
    except monitoring.MonitoringError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


@app.post("/api/monitoring/import")
def api_monitoring_import(payload: dict):
    """{project, source, days, session, context?} -> the same proposal shape as the file import (nothing applied;
    apply with POST /api/sessions/{session}/traffic/apply)."""
    from aiops_dilemma import monitoring

    session = _sanitize_name(str(payload.get("session") or ""))
    if not payload.get("session"):
        raise HTTPException(status_code=400, detail="Missing 'session'.")
    rd = _session_dir(session, for_write=True)
    project = payload.get("project") or _default_project()
    days = int(payload.get("days") or 28)
    try:
        got = monitoring.fetch_series(project, str(payload.get("source") or ""), days)
    except monitoring.MonitoringError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    src = dict(got["source"])
    src.update({"kind": "monitoring", "days": days, "label": payload.get("source")})
    stored = _store_import(rd, got["series"], src, missing_is_zero=True, window=got["window"])
    return _proposal_from_stored(stored, _import_context(session, payload.get("context")), {}, None, False)


def _multipart_in_memory(content_type: str, body: bytes) -> tuple[bytes | None, str | None, str | None, str | None]:
    """Parse a multipart body without spooling to disk: (file bytes, filename, text field, project field)."""
    from email import policy
    from email.parser import BytesParser

    msg = BytesParser(policy=policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    data = filename = text = project = None
    for part in (msg.iter_parts() if msg.is_multipart() else []):
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if name == "file":
            data, filename = payload, part.get_filename()
        elif name == "text":
            text = payload.decode("utf-8", errors="replace")
        elif name == "project":
            project = payload.decode("utf-8", errors="replace").strip() or None
    return data, filename, text, project


@app.post("/api/ingest/describe")
async def api_ingest_describe(request: Request):
    """Read a workload description (.pdf/.docx/.txt/.md upload or JSON {text}) with Gemini; nothing is stored."""
    limit = ingest.DESCRIBE_MAX_BYTES + 1024 * 1024  # file cap + multipart overhead
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(status_code=413, detail="File is too large (limit 20 MB).")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(status_code=413, detail="File is too large (limit 20 MB).")
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        data, filename, text, project = _multipart_in_memory(ctype, bytes(body))
    else:
        try:
            payload = json.loads(bytes(body) or b"{}")
        except ValueError:
            raise HTTPException(status_code=400, detail="Send a multipart 'file' or JSON {\"text\": ...}.")
        payload = payload if isinstance(payload, dict) else {}
        data, filename, text, project = None, None, payload.get("text"), payload.get("project")
    del body
    project = project or _default_project()
    try:
        return await run_in_threadpool(ingest.describe_workload, data, filename, text, project)
    except ingest.DescribeError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


def _drop_stale_estimate(rd: Path, force: bool = False) -> None:
    """Remove plan / estimate / report files that describe a previous selection. A real load test is kept unless
    `force` (then it moves to measured.prev.json, as a forced estimate does)."""
    measured = _load_json(rd / "measured.json")
    if report.is_measured(measured):
        if not force:
            return
        (rd / "measured.json").replace(rd / "measured.prev.json")
    for f in ("plan.json", "measured.json", "report.md", "dashboard.html"):
        try:
            (rd / f).unlink()
        except FileNotFoundError:
            pass


@app.post("/api/sessions/{name}/estimate")
def api_estimate_session(name: str, payload: dict | None = None):
    """Instant estimate: plan, cost model, report and dashboard without deploying anything.

    Never fabricates a load test: measured.json for an estimate is {"estimated": true, "levels": [], "peak": null,
    "input_tps_per_gpu": ..., "throughput_source": ...}. Throughput is the session's single source (see
    resolve_throughput) unless `tps_per_gpu` is passed (source "override"). A real load test is kept unless
    `force_estimate` is set (then it is moved to measured.prev.json). `clear_override: true` drops a saved override.
    """
    payload = payload or {}
    clean_name = _sanitize_name(name)
    rd = _session_dir(clean_name, for_write=True)
    rd.mkdir(parents=True, exist_ok=True)

    if payload.get("config"):
        raw_cfg = payload["config"]
        raw_cfg["name"] = clean_name
        cfg = _reject_bad_region(config.normalize(raw_cfg, path=str(rd / "workload.yaml")))
        (rd / "workload.yaml").write_text(config.to_yaml(cfg))
    else:
        cfg = _session_Config(clean_name, rd)
    cfg = _with_defaults(cfg)

    try:
        p = planner.plan(cfg)
    except planner.NoFitError:
        # The new model (or precision) fits no single machine: keep the choice (workload.yaml is written), drop the
        # previous model's plan and estimate so nothing on the page describes it any more, and return the session
        # with `plan_error` + `no_fit` (why, the largest machine, options) for the page to explain.
        _drop_stale_estimate(rd, force=bool(payload.get("force_estimate")))
        return _build_session_payload(clean_name)
    except (Exception, SystemExit) as e:
        raise HTTPException(status_code=400, detail=str(e))
    (rd / "plan.json").write_text(json.dumps(p, indent=2, default=str))
    ch = p["chosen"]

    existing = _load_json(rd / "measured.json")
    try:
        override = float(payload.get("tps_per_gpu") or 0) or None
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="tps_per_gpu must be a number")
    if report.is_measured(existing) and not payload.get("force_estimate"):
        measured = existing
        ep = existing["endpoint"]
        prices = (pricing.prices_for(cfg, ep["accelerator_type"], int(ep["accelerator_count"]))
                  if not _same_hw(ep, ch) else ch.get("prices"))
    else:
        if report.is_measured(existing):
            (rd / "measured.json").replace(rd / "measured.prev.json")
            existing = None
        prior = None if payload.get("clear_override") else existing
        tp = resolve_throughput(prior, ch, override=override)
        measured = _effective_measured(None, {"hardware": _hw(ch)}, tp)
        (rd / "measured.json").write_text(json.dumps(measured, indent=2))
        prices = ch.get("prices") or pricing.prices_for(cfg, ch["accelerator_type"], ch["accelerator_count"])

    report.write(cfg, p, measured, prices, rd)
    dashboard.build(cfg, measured, prices, rd)
    return _build_session_payload(clean_name)


@app.post("/api/sessions/{name}/simulate")
def api_simulate_session(name: str, payload: dict | None = None):
    """Request-level simulation (simulate.run) for the session's selected model, hardware and throughput.

    Body (all optional): {"scenarios": ["gpu", "flat", "deferred", "routed"], "bursts": [1.0, 2.0],
    "target_late": 0.01, "tier": "vertex_cud_3y", "scale": 1.0, "config": {...}}. With `config` (unsaved edits)
    or for a read-only example session the result is returned without being saved ({"simulation", "what_if",
    "example", "throughput"}); otherwise simulation.json/.md are written and the full session payload returned.
    """
    from aiops_dilemma import simulate

    payload = payload or {}
    clean_name = _sanitize_name(name)
    rd = _session_dir(clean_name)
    if not rd.exists():
        raise HTTPException(status_code=404, detail=f"Session '{clean_name}' not found")
    what_if = bool(payload.get("config"))
    if what_if:   # unsaved edits from the UI: simulate them without writing anything
        raw = copy.deepcopy(payload["config"])
        raw["name"] = clean_name
        cfg = config.normalize(raw, path=str(rd / "workload.yaml"))
        cfg["_run_dir"] = str(rd)
        cfg = _with_defaults(cfg)
    else:
        cfg = _with_defaults(_session_Config(clean_name, rd))
    measured = _load_json(rd / "measured.json")
    selected, chosen, throughput, plan_err, _ = _selection(cfg, measured)
    if not chosen or not throughput.get("per_chip"):
        raise HTTPException(status_code=400, detail=plan_err or "No accelerator configuration fits this model.")
    prices = chosen.get("prices") or pricing.prices_for(cfg, chosen["accelerator_type"], chosen["accelerator_count"])
    try:
        bursts = tuple(float(b) for b in (payload.get("bursts") or [1.0, 2.0]))
        result = simulate.run(
            cfg, float(throughput["per_chip"]), prices, mix_factor_ref=costmodel.ref_mix_factor(cfg, throughput),
            gpus_per_replica=int(chosen.get("accelerator_count") or 1),
            scenarios=payload.get("scenarios"), bursts=bursts,
            target_late=float(payload.get("target_late") or 0.01),
            tier=payload.get("tier") or "vertex_cud_3y", scale=float(payload.get("scale") or 1.0),
            measured=measured if report.is_measured(measured) else None,
            out_dir=None if (is_example(clean_name) or what_if) else rd, log=lambda *_: None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if is_example(clean_name) or what_if:
        return {"name": clean_name, "example": is_example(clean_name), "what_if": what_if,
                "simulation": result, "throughput": throughput}
    return _build_session_payload(clean_name)


# Assumed FP8 RTX PRO 6000 input tok/s per GPU for the blog's illustrative example (a thinking-heavy workload,
# 5,000 in / 500 + 30% x 1,500 thinking out per request). An assumption for the example, not a measurement.
BLOG_EXAMPLE_TPS = 17946.0   # measured short-prompt rate of the blog workload (blog-mix-g, 1x RTX PRO 6000)


def build_example_session(dest: Path | None = None) -> Path:
    """(Re)generate the read-only example session from blog/example/workload.yaml (no customer data).

    Writes workload.yaml (verbatim), plan.json, an estimate-only measured.json at the blog's throughput figure,
    report.md and dashboard.html into `dest` (default runs/blog-example).
    """
    dest = dest or RUNS_DIR / "blog-example"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    src_yaml = BLOG_EXAMPLE_DIR / "workload.yaml"
    (dest / "workload.yaml").write_text(src_yaml.read_text())
    cfg = config.normalize(yaml.safe_load(src_yaml.read_text()) or {}, path=str(dest / "workload.yaml"))
    cfg["_run_dir"] = str(dest)
    p = planner.plan(cfg)
    (dest / "plan.json").write_text(json.dumps(p, indent=2, default=str))
    ch = p["chosen"]
    label = (f"{BLOG_EXAMPLE_TPS:,.0f} input tok/s per chip on short prompts, measured in a benchmark of this illustrative "
             "example's workload on 1 × RTX PRO 6000 (2026-11-16); long prompts use its measured per-class rates. "
             "Your workload will differ: run a benchmark to measure it.")
    measured = {"estimated": True, "levels": [], "peak": None, "endpoint": _hw(ch),
                "input_tps_per_gpu": BLOG_EXAMPLE_TPS, "throughput_source": "override",
                "throughput_label": label, "measured_out_ratio": None}
    (dest / "measured.json").write_text(json.dumps(measured, indent=2))
    report.write(cfg, p, measured, ch["prices"], dest)
    dashboard.build(cfg, measured, ch["prices"], dest)
    return dest


def _reap_children() -> None:
    for pid, proc in list(_CHILDREN.items()):
        if proc.poll() is not None:
            _CHILDREN.pop(pid, None)


def _job_payload(rd: Path) -> dict:
    _reap_children()
    return jobs.job_payload(rd)


def _scan_jobs_on_startup() -> None:
    """Mark jobs whose process died while no server was watching as interrupted, so the session list badges them."""
    try:
        found = jobs.scan(RUNS_DIR)
    except Exception as e:   # noqa: BLE001
        print(f"job scan failed: {e}")
        return
    for name, status in sorted(found.items()):
        if status == "running":
            print(f"benchmark job running for session '{name}'")
        elif status == "interrupted" and jobs.endpoint_at_risk(RUNS_DIR / name):
            print(f"WARNING: the benchmark for session '{name}' stopped unexpectedly and its endpoint may still be "
                  f"running (billing): tear it down from the app or `dilemma teardown -c runs/{name}/workload.yaml`")


def _log_tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _ensure_model(rd: Path) -> None:
    """Save the default model into a session that has none (the model its estimate already shows)."""
    path = rd / "workload.yaml"
    raw = yaml.safe_load(path.read_text()) or {}     # not config.load(): it rejects a config without a model
    if not (raw.get("model") or {}).get("model_garden_id"):
        raw.setdefault("model", {}).update(DEFAULT_MODEL)
        path.write_text(config.to_yaml(config.normalize(raw, path=str(path))))


@app.post("/api/sessions/{name}/run")
def api_run_session(name: str, payload: dict, request: Request):
    clean_name = _sanitize_name(name)
    rd = _session_dir(clean_name, for_write=True)
    rd.mkdir(parents=True, exist_ok=True)

    # Every value that reaches the job's command line is allowlisted or converted to an int (no shell either way).
    mode = payload.get("mode", "auto")
    if not isinstance(mode, str) or mode not in jobs.RUN_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {str(mode)[:40]!r}")
    steps = list(jobs.AUTO_STEPS) if mode == "auto" else (payload.get("steps") or [mode])
    if not isinstance(steps, list):
        raise HTTPException(status_code=400, detail="steps must be a list")
    bad = [s for s in steps if not isinstance(s, str) or s not in jobs.STEPS]
    if bad or not steps:
        raise HTTPException(status_code=400, detail=f"Unknown step(s): {', '.join(map(str, bad)) or 'none given'}")
    keep_endpoint = bool(payload.get("keep_endpoint", False))
    try:
        concurrency = [int(c) for c in payload.get("concurrency") or []]
        requests_per_level = int(payload["requests_per_level"]) if payload.get("requests_per_level") else None
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="concurrency and requests_per_level must be whole numbers")
    if any(c < 1 or c > 4096 for c in concurrency) or (requests_per_level is not None and not 1 <= requests_per_level <= 100_000):
        raise HTTPException(status_code=400, detail="concurrency must be 1-4096 and requests_per_level 1-100000")

    with _LAUNCH_LOCK:
        live = jobs.live_job(rd)
        if live:
            raise HTTPException(status_code=409, detail=f"A benchmark job is already running for this session "
                                                        f"(pid {live.get('pid')}, started from the {live.get('source')})")
        if payload.get("config"):
            raw_cfg = payload["config"]
            raw_cfg["name"] = clean_name
            cfg = _reject_bad_region(config.normalize(raw_cfg, path=str(rd / "workload.yaml")))
            (rd / "workload.yaml").write_text(config.to_yaml(cfg))
        elif not (rd / "workload.yaml").exists():
            (rd / "workload.yaml").write_text(config.to_yaml(_session_Config(clean_name, rd)))
        # A session started from "six numbers" has no saved model: the estimate shows the default model, so the
        # benchmark must deploy that same model (save it, so the session and the job agree).
        _ensure_model(rd)
        # The job uses the session's project (else the app's) for every call, as resource and quota project.
        try:
            job_proj = auth.real_project(config.load(str(rd / "workload.yaml")).get("project"))
        except (Exception, SystemExit):
            job_proj = None
        job_proj = job_proj or auth.real_project(auth.app_project())
        job_env = {**os.environ, "PYTHONUNBUFFERED": "1", **hftoken.job_env()}   # memory-only HF token
        user = getattr(request.state, "user", None) or {}
        if user.get("email"):
            job_env["DILEMMA_JOB_USER"] = user["email"]     # recorded as job.json "started_by"
        if job_proj:
            job_env.update(GOOGLE_CLOUD_PROJECT=job_proj, GOOGLE_CLOUD_QUOTA_PROJECT=job_proj)

        cmd = [sys.executable, "-m", "aiops_dilemma.cli", "run-job", "-c", str(rd / "workload.yaml"),
               "--run-dir", str(rd), "--mode", mode, "--source", "app", "--steps", *steps]
        if keep_endpoint:
            cmd.append("--keep")
        if concurrency:
            cmd += ["--concurrency", *map(str, concurrency)]
        if requests_per_level:
            cmd += ["--requests", str(requests_per_level)]
        log_path = rd / jobs.LOG_FILE
        with open(log_path, "ab") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd[3:])}\n".encode())
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    cwd=str(ROOT_DIR), start_new_session=True,
                                    env=job_env)
        _CHILDREN[proc.pid] = proc
        # Wait until the runner owns job.json, so the next GET already shows the job (and failures surface here).
        deadline = time.time() + JOB_START_TIMEOUT_S
        while time.time() < deadline:
            job = jobs.read_job(rd, mark=False)
            if job and job.get("pid") == proc.pid:
                break
            if proc.poll() is not None:
                _CHILDREN.pop(proc.pid, None)
                raise HTTPException(status_code=500, detail="The benchmark process exited at once: "
                                                            + (_log_tail(log_path) or f"exit code {proc.returncode}"))
            time.sleep(0.05)
        else:
            raise HTTPException(status_code=500, detail="The benchmark process did not start in time; see "
                                                        f"{log_path}")
    return {"ok": True, "job": {"status": "running", "steps": steps, "pid": proc.pid}}


@app.post("/api/sessions/{name}/stop")
def api_stop_session(name: str, payload: dict | None = None):
    """Graceful stop (SIGTERM: cancel at the next safe point, tear down what the job deployed) or force=true
    (SIGKILL: immediate; the job is then marked interrupted and the endpoint may need a teardown)."""
    clean_name = _sanitize_name(name)
    payload = payload or {}
    rd = _session_dir(clean_name)
    force = bool(payload.get("force"))
    signalled = None
    try:
        job = jobs.signal_job(rd, force=force)
    except ProcessLookupError:
        job = None
    except ValueError as e:   # job.json names a pid we must never signal
        raise HTTPException(status_code=409, detail=str(e))
    if job:
        signalled = "SIGKILL" if force else "SIGTERM"
        if force:
            proc = _CHILDREN.pop(int(job["pid"]), None)
            if proc:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            else:
                for _ in range(50):          # not our child: wait for it to disappear
                    if not jobs.pid_alive(job["pid"], job.get("pid_start")):
                        break
                    time.sleep(0.1)
    if payload.get("teardown"):
        cfg = _session_Config(clean_name, rd)
        try:
            deploy.teardown(cfg, rd)
        except Exception as e:
            return {"ok": False, "error": str(e), "signal": signalled, "job": _job_payload(rd)}
    return {"ok": True, "signal": signalled, "job": _job_payload(rd)}


def _read_new_events(path: Path, pos: int) -> tuple[list[tuple[int, str]], int]:
    """Complete lines appended since byte offset `pos`: [(offset after the line, line)], new offset."""
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    if size < pos:                 # a new job truncated the file
        pos = 0
    if size == pos:
        return [], pos
    out = []
    with open(path, "rb") as f:
        f.seek(pos)
        chunk = f.read(size - pos)
    for line in chunk.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break                  # partial write: pick it up next time
        pos += len(line)
        text = line.decode(errors="replace").strip()
        if text:
            out.append((pos, text))
    return out, pos


@app.get("/api/sessions/{name}/events")
async def api_session_events(name: str, request: Request, since: int | None = None):
    """SSE: replays runs/<session>/job_events.jsonl (from `since` / Last-Event-ID, a byte offset) and follows it
    while the job runs. Works after a server restart and for CLI-started jobs."""
    clean_name = _sanitize_name(name)
    rd = _session_dir(clean_name)
    path = rd / jobs.EVENTS_FILE
    start = since
    if start is None:
        try:
            start = int(request.headers.get("last-event-id") or 0)
        except ValueError:
            start = 0

    async def event_generator():
        pos = max(0, int(start or 0))
        if not (rd / jobs.JOB_FILE).exists():
            yield f"data: {json.dumps({'type': 'idle'})}\n\n"
            return
        last_beat = time.time()
        while True:
            job = await run_in_threadpool(jobs.read_job, rd) or {}
            lines, pos = _read_new_events(path, pos)
            for off, line in lines:
                yield f"id: {off}\ndata: {line}\n\n"
            if job.get("status") != "running":
                lines, pos = _read_new_events(path, pos)      # the terminal event is written before the status
                for off, line in lines:
                    yield f"id: {off}\ndata: {line}\n\n"
                break
            if await request.is_disconnected():
                break
            if time.time() - last_beat > 15:
                last_beat = time.time()
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sessions/{name}/export/{artifact}")
def api_export_artifact(name: str, artifact: str):
    rd = _session_dir(name)
    if artifact == "workload.yaml":
        p = rd / "workload.yaml"
        if not p.exists() and name == "blog-example":
            p = BLOG_EXAMPLE_DIR / "workload.yaml"
        if not p.exists():
            raise HTTPException(status_code=404, detail="workload.yaml not found")
        return FileResponse(str(p), filename=f"{name}-workload.yaml", media_type="text/yaml")
    if artifact == "report.md":
        p = rd / "report.md"
        if not p.exists():
            raise HTTPException(status_code=404, detail="report.md not found")
        return FileResponse(str(p), filename=f"{name}-report.md", media_type="text/markdown")
    if artifact == "dashboard.html":
        p = rd / "dashboard.html"
        if not p.exists():
            raise HTTPException(status_code=404, detail="dashboard.html not found")
        return FileResponse(str(p), filename=f"{name}-dashboard.html", media_type="text/html")
    raise HTTPException(status_code=400, detail=f"Unsupported artifact: {artifact}")
