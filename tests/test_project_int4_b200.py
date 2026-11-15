"""The app's own project (never gcloud config) + quota project on every call, account switching, INT4 checkpoints,
models too big for one machine (explanation instead of a bare error), B200, and the model drawer's selection state.

No network: google.auth.default is patched, HTTP goes through patched helpers, gcloud is tests/fake_gcloud.py."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aiops_dilemma import (auth, catalog, config, deploy, ingest, modelinfo, monitoring, planner, pricing, recommender,
                           server)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "aiops_dilemma" / "assets"
FAKE = Path(__file__).resolve().parent / "fake_gcloud.py"

GARDEN_IDS = [
    "moonshotai/kimi-k2@kimi-k2-thinking", "moonshotai/kimi-k2@kimi-k2-instruct", "moonshotai/kimi-k2-6@kimi-k2.6",
    "qwen/qwen2@qwen2.5-32b-instruct", "qwen/qwen2@qwen2.5-32b-instruct-awq",
    "zai-org/glm-4.7@glm-4.7", "zai-org/glm-4.7@glm-4.7-nvfp4",
    "google/gemma4@gemma-4-26b-a4b-it",
]
KIMI_T = "moonshotai/kimi-k2@kimi-k2-thinking"
KIMI_I = "moonshotai/kimi-k2@kimi-k2-instruct"
QWEN = "qwen/qwen2@qwen2.5-32b-instruct"


@pytest.fixture()
def catalog_models(monkeypatch):
    """A fixed curated catalog (find_model / planner / recommender read it), no verified recipes, no live calls."""
    models = modelinfo.curate_models(GARDEN_IDS)
    monkeypatch.setattr(modelinfo, "discover_models", lambda *a, **k: {"models": models, "sources": [], "families": []})
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a, **k: [])
    return {m["model_garden_id"]: m for m in models}


@pytest.fixture()
def fake_gcloud(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "gcloud"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    logf = tmp_path / "gcloud.log"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GCLOUD_LOG", str(logf))
    for k in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    auth._invalidate_caches()
    return lambda: [json.loads(x) for x in logf.read_text().splitlines()] if logf.exists() else []


def _user_creds(quota="sys-wrong-project"):
    from google.oauth2 import credentials
    c = credentials.Credentials(token="user-token", refresh_token="r", quota_project_id=quota)
    c.expiry = None
    return c


@pytest.fixture()
def user_adc(monkeypatch):
    """ADC = a user credential whose file has a wrong stamped quota project (the user's bug)."""
    import google.auth
    creds = _user_creds()
    monkeypatch.setattr(google.auth, "default", lambda *a, **k: (creds, None))
    monkeypatch.setattr(type(creds), "valid", property(lambda self: True))
    return creds


# --------------------------------------------------------------------------------------- Bug 1: app project only
def test_picker_never_writes_gcloud_config(fake_gcloud, monkeypatch):
    monkeypatch.setattr(auth, "_adc", lambda: (_ for _ in ()).throw(RuntimeError("no adc")))
    before = dict(os.environ)
    st = auth.set_project("my-chosen-project")
    assert st["app_project"] == "my-chosen-project" and st["project"] == "my-chosen-project"
    assert st["app_project_source"] == "app"
    assert st["gcloud_project"] == "my-gcp-project"            # fake gcloud's default, reported separately
    assert auth.app_project() == "my-chosen-project"
    assert not any(c["argv"][:2] == ["config", "set"] for c in fake_gcloud())
    assert os.environ.get("GOOGLE_CLOUD_PROJECT") == before.get("GOOGLE_CLOUD_PROJECT")
    # the HTTP endpoint behaves the same; a bad ID is a 400, not a gcloud call
    c = TestClient(server.app)
    r = c.post("/api/auth/login", json={"project": "other-project-1"}).json()
    assert r["app_project"] == "other-project-1"
    assert c.post("/api/auth/login", json={"project": "Not A Project!"}).status_code == 400
    assert not any(x["argv"][:2] == ["config", "set"] for x in fake_gcloud())
    # back to the machine default
    assert auth.set_project("my-gcp-project")["app_project_source"] == "default"


def test_credentials_carry_the_app_project_as_quota_project(user_adc):
    auth._APP.update(project="app-project-1")
    creds = auth.credentials()
    assert creds.quota_project_id == "app-project-1"            # not the ADC file's sys-... stamp
    assert auth.credentials("explicit-proj").quota_project_id == "explicit-proj"
    h = auth.auth_headers("explicit-proj")
    assert h["x-goog-user-project"] == "explicit-proj" and h["Authorization"] == "Bearer user-token"


def test_every_client_uses_the_app_project(user_adc, monkeypatch):
    auth._APP.update(project="app-project-1")
    # Cloud Billing Catalog (pricing)
    h = pricing._headers("app-project-1")
    assert h["x-goog-user-project"] == "app-project-1" and h["Authorization"] == "Bearer user-token"
    assert pricing._quota_project(None) == "app-project-1"
    # Cloud Monitoring
    with monitoring._client("app-project-1") as cl:
        assert cl.headers["x-goog-user-project"] == "app-project-1"
    # Model Garden (vertexai.init gets credentials with the quota project)
    import vertexai
    seen = {}
    monkeypatch.setattr(vertexai, "init", lambda **kw: seen.update(kw))
    from vertexai import model_garden
    monkeypatch.setattr(model_garden, "OpenModel", lambda *_: (_ for _ in ()).throw(RuntimeError("stop")))
    assert modelinfo._verified_options("app-project-1", "us-central1", KIMI_T) == []
    assert seen["project"] == "app-project-1" and seen["credentials"].quota_project_id == "app-project-1"
    # google-genai (describe a workload)
    from google import genai
    got = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: got.update(kw) or object())
    ingest._genai_client("app-project-1", "global")
    assert got["project"] == "app-project-1" and got["credentials"].quota_project_id == "app-project-1"
    # deploy / teardown in a job use the session's project, else the app's
    assert deploy._project({"project": "sess-project-1"}) == "sess-project-1"
    assert deploy._project({"project": "my-gcp-project"}) == "app-project-1"


def test_identity_wide_calls_retry_without_a_refusing_quota_project(monkeypatch):
    calls = []

    def fake_get(url, token=None, params=None, headers=None, timeout=10):
        calls.append(dict(headers or {}))
        if headers and headers.get("x-goog-user-project"):
            return 403, {"error": {"status": "PERMISSION_DENIED", "details": [{"reason": "USER_PROJECT_DENIED"}]}}
        return 200, {"projects": [{"projectId": "new-acct-proj", "displayName": "New", "name": "projects/1"}]}

    monkeypatch.setattr(auth, "_http_get", fake_get)
    rows, code, err = auth._search_projects("tok", "old-project-1")
    assert code == 200 and rows[0]["project_id"] == "new-acct-proj"
    assert calls[0] == {"x-goog-user-project": "old-project-1"} and calls[1] == {}


def test_status_reports_three_projects_and_the_quota_mismatch(monkeypatch, user_adc):
    monkeypatch.setattr(auth, "_run", lambda cmd, timeout=15: (1, "ERROR: offline") if "print-access-token" in cmd
                        else (0, "you@example.com") if "account" in cmd else (0, "gcloud-default-1"))
    monkeypatch.setattr(auth, "_http_get", lambda *a, **k: (200, {"email": "you@example.com"}))
    for k in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    auth._APP.update(project="app-project-1")
    st = auth.status()
    assert (st["gcloud_project"], st["quota_project"], st["app_project"]) == \
        ("gcloud-default-1", "sys-wrong-project", "app-project-1")
    assert st["quota_project_differs"] is True and st["project"] == "app-project-1"


def test_account_change_rechecks_project_and_clears_caches(monkeypatch):
    accounts = iter(["old@example.com"])
    monkeypatch.setattr(server.auth, "identity", lambda: next(accounts))
    monkeypatch.setattr(server.auth, "login_finish", lambda lid, code: {"ok": True, "mode": "adc", "message": "ok"})
    monkeypatch.setattr(server.auth, "status", lambda: {"authenticated": True, "adc_account": "new@example.com",
                                                         "app_project": "old-project-1"})
    checked = []
    monkeypatch.setattr(server.auth, "project_access", lambda p: checked.append(p) or
                        {"project": p, "ok": False, "message": "The signed-in account can't see project old-project-1."})
    refreshed = []
    monkeypatch.setattr(server, "_refresh_catalogs_after_login", lambda all_=False: refreshed.append(all_) or True)
    auth._PROJECTS_MEM["k"] = (0, {})
    modelinfo._VERIFIED_MEM[("p", "r", "m")] = (0, [])
    r = TestClient(server.app).post("/api/auth/login/finish", json={"login_id": "x", "code": "c"}).json()
    assert r["account_changed"] is True and r["previous_account"] == "old@example.com"
    assert r["needs_project"] is True and checked == ["old-project-1"]
    assert not auth._PROJECTS_MEM and not modelinfo._VERIFIED_MEM and refreshed == [True]


def test_same_account_sign_in_keeps_project(monkeypatch):
    monkeypatch.setattr(server.auth, "identity", lambda: "same@example.com")
    monkeypatch.setattr(server.auth, "login_finish", lambda lid, code: {"ok": True, "mode": "adc", "message": "ok"})
    monkeypatch.setattr(server.auth, "status", lambda: {"authenticated": True, "adc_account": "same@example.com"})
    monkeypatch.setattr(server.auth, "project_access", lambda p: pytest.fail("no re-check for the same account"))
    monkeypatch.setattr(server, "_refresh_catalogs_after_login", lambda all_=False: False)
    r = TestClient(server.app).post("/api/auth/login/finish", json={"login_id": "x", "code": "c"}).json()
    assert r["account_changed"] is False and r["needs_project"] is False


def test_project_access_uses_the_identity_project_list(tmp_path, monkeypatch):
    f = tmp_path / "projects.json"
    f.write_text(json.dumps([{"project_id": "visible-proj", "name": "Visible"}]))
    monkeypatch.setenv(auth.PROJECTS_STUB_ENV, str(f))
    assert auth.project_access("visible-proj")["ok"] is True
    assert auth.project_access("hidden-project")["ok"] is False
    assert auth.project_access("my-gcp-project")["ok"] is None


def test_ui_shows_app_project_and_quota_mismatch_note():
    js = (ASSETS / "app.js").read_text()
    html = (ASSETS / "app.html").read_text()
    assert 'id="authProject"' in html and 'id="authQuota"' in html
    assert "Project for this app:" in js
    assert "Your credentials bill API calls to ${st.quota_project} by default; the app uses ${appProj} for its calls." in js
    assert "askForProject" in js and "needs_project" in js


# --------------------------------------------------------------------------------------- Bug 2: drawer selection
def test_drawer_selection_is_recomputed_client_side():
    js = (ASSETS / "app.js").read_text()
    body = js[js.index("async function selectModel(m)"):js.index("async function selectModel(m)") + 1500]
    assert "S.models = null" in body and "renderChips()" in body
    groups = js[js.index("function modelGroups()"):js.index("function parseQuery")]
    assert "m.is_selected !== undefined" not in groups and "isSelectedModel" in groups


# --------------------------------------------------------------------------------------- INT4
def test_int4_availability(catalog_models):
    kt, ki, qw = catalog_models[KIMI_T], catalog_models[KIMI_I], catalog_models[QWEN]
    assert kt["native_precision"] == "int4" and kt["precisions"] == ["int4"]
    assert ki["native_precision"] == "fp8" and ki["fp8_only"] and ki["precisions"] == ["fp8"]
    assert qw["precisions"] == ["bf16", "fp8", "int4"] and qw["prequantized"]["awq"].endswith("-awq")
    glm = catalog_models["zai-org/glm-4.7@glm-4.7"]
    assert "int4" not in glm["precisions"]                     # NVFP4 is not an INT4 checkpoint
    gemma = catalog_models["google/gemma4@gemma-4-26b-a4b-it"]
    assert "int4" not in gemma["precisions"]
    assert modelinfo.int4_checkpoint(kt) == KIMI_T and modelinfo.int4_checkpoint(qw).endswith("-awq")
    assert modelinfo.int4_checkpoint(gemma) is None


@pytest.mark.parametrize("raw,want", [("int4", "int4"), ("INT4", "int4"), ("awq", "int4"), ("w4a16", "int4"),
                                      ("4bit", "int4"), ("fp8", "fp8"), ("bogus", "bf16"), (None, "bf16")])
def test_precision_normalization(raw, want):
    assert config.normalize({"model": {"model_garden_id": "x/y@z-8b", "precision": raw}})["model"]["precision"] == want


def test_int4_sizing_fits_kimi_thinking_on_8_h200(catalog_models):
    cfg = config.normalize({"project": "my-gcp-project", "model": {"model_garden_id": KIMI_T, "precision": "int4"}})
    assert modelinfo.precision_bytes(cfg["model"]) == pytest.approx(1000e9 * 0.6)
    p = planner.plan(cfg)
    shapes = {(c["accelerator_type"], c["accelerator_count"]) for c in p["candidates"]}
    assert ("NVIDIA_H200_141GB", 8) in shapes and ("NVIDIA_B200", 8) in shapes
    assert not any(c["kind"] == "tpu" for c in p["candidates"])      # INT4 is GPU only
    assert p["deployment_model"]["quantization"] == "native" and p["deployment_model"]["model_garden_id"] == KIMI_T
    assert "INT4: rough" in p["chosen"]["est_basis_note"]


def test_verified_recipe_overrides_the_75_percent_rule(catalog_models, monkeypatch):
    rec = {"machine_type": "a3-highgpu-8g", "accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 8,
           "image": "pytorch-vllm-serve", "args": ["--gpu-memory-utilization=0.95"], "env": {}}
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a, **k: [rec])
    cfg = config.normalize({"project": "real-project-1", "model": {"model_garden_id": KIMI_T, "precision": "int4"}})
    h100 = next(c for c in planner.plan(cfg)["candidates"] if c["accelerator_type"] == "NVIDIA_H100_80GB")
    assert h100["verified"] and h100["tight_fit"] and "little room for KV cache" in h100["fit_note"]
    assert h100["vram_util"] > planner.WEIGHTS_MAX_SHARE


def test_int4_deploys_the_4bit_checkpoint_without_quantization_flag(catalog_models):
    cfg = config.normalize({"project": "my-gcp-project", "model": {"model_garden_id": QWEN, "precision": "int4",
                                                                   "hf_id": "Qwen/qwen2.5-32b-instruct"}})
    dm = planner.deployment_model(cfg)
    assert dm["quantization"] == "prequantized" and dm["model_garden_id"].endswith("-awq")
    ch = {"accelerator_count": 1}
    _, args, _, _ = deploy._args(cfg, ch, None)
    assert not any(a.startswith("--quantization") for a in args)
    kimi = config.normalize({"project": "my-gcp-project", "model": {"model_garden_id": KIMI_I, "precision": "fp8"}})
    _, args, _, _ = deploy._args(kimi, ch, None)
    assert not any(a.startswith("--quantization") for a in args)       # published in FP8: vLLM reads its config


def test_int4_spec_rate_reads_fewer_bytes(catalog_models):
    req = {"avg_input_tokens": 2000, "avg_output_tokens": 1500}
    base = {"model": {"model_garden_id": QWEN, "params_b": 32, "active_params_b": 32}, "requests": req}
    rate = {p: planner._spec_rate({**base, "model": {**base["model"], "precision": p}}, "NVIDIA_H100_80GB", 1)
            for p in ("bf16", "fp8", "int4")}
    assert rate["bf16"] < rate["fp8"] < rate["int4"]


def test_precision_options_only_real_ones(catalog_models):
    res = recommender.recommend({"task_type": "reasoning"}, {"avg_input_tokens": 6000, "avg_output_tokens": 500},
                                {"peak_input_tps": 50000}, models=list(catalog_models.values()))
    by = {m["model_garden_id"]: m for m in res["recommendations"]}
    kt = by[KIMI_T]
    assert kt["recommended_precision"] == "int4" and [o["precision"] for o in kt["precision_options"]] == ["int4"]
    assert kt["precision_options"][0]["how"] == "native" and kt["precision_options"][0]["quality_note"] is None
    assert "INT4" in kt["precision_reason"] and "FP8" not in kt["precision_reason"]
    ki = by[KIMI_I]
    assert [o["precision"] for o in ki["precision_options"]] == ["fp8"] and ki["precision_options"][0]["how"] == "native"
    qw = by[QWEN]
    opts = {o["precision"]: o for o in qw["precision_options"]}
    assert set(opts) == {"bf16", "fp8", "int4"}
    assert opts["fp8"]["how"] == "on_the_fly" and opts["int4"]["how"] == "prequantized"
    for m in res["recommendations"]:
        for o in m["precision_options"]:
            # a quality note sits only on the recommended item and names that item's precision
            if o["quality_note"]:
                assert o["recommended"] and o["quality_note"].startswith("4-bit" if o["precision"] == "int4" else "FP8")
        assert m["recommended_precision"] in [o["precision"] for o in m["precision_options"]]


# --------------------------------------------------------------------------------------- too big for one machine
def test_no_fit_explains_weights_largest_machine_and_options(catalog_models):
    big = {"model_garden_id": "acme/huge@huge-2000b", "params_b": 2000, "active_params_b": 40, "precision": "fp8"}
    cfg = config.normalize({"project": "my-gcp-project", "model": big})
    with pytest.raises(planner.NoFitError) as e:
        planner.plan(cfg)
    info = e.value.info
    assert info["weights_gb"] == 2000 and info["largest"]["accelerator_type"] == "NVIDIA_B200"
    assert "2,000 GB" in info["message"] and "8 × B200 180GB" in info["message"]
    labels = " ".join(o["label"] for o in info["options"])
    assert "multi-host" in labels and "no INT4" in labels
    assert any((o.get("action") or {}).get("type") == "models" for o in info["options"])


def test_no_fit_offers_int4_when_a_checkpoint_exists(catalog_models, monkeypatch):
    monkeypatch.setattr(modelinfo, "find_model", lambda m, models=None: {
        "name": "Huge", "params_b": 1400, "precisions": ["bf16", "fp8"], "prequantized": {"awq": "acme/huge@huge-awq"}})
    cfg = config.normalize({"project": "my-gcp-project",
                            "model": {"model_garden_id": "acme/huge@huge", "params_b": 1400, "precision": "fp8"}})
    with pytest.raises(planner.NoFitError) as e:
        planner.plan(cfg)
    acts = [o["action"] for o in e.value.info["options"] if o.get("action", {}).get("type") == "precision"]
    assert acts == [{"type": "precision", "value": "int4"}]


def test_precision_without_checkpoint_is_refused(catalog_models):
    fp8 = config.normalize({"project": "my-gcp-project", "model": {"model_garden_id": KIMI_T, "precision": "fp8"}})
    with pytest.raises(planner.NoFitError) as e:
        planner.plan(fp8)
    assert e.value.info["reason"] == "native_only"
    assert {"type": "precision", "value": "int4"} in [o.get("action") for o in e.value.info["options"]]
    gem = config.normalize({"project": "my-gcp-project",
                            "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "precision": "int4"}})
    with pytest.raises(planner.NoFitError) as e:
        planner.plan(gem)
    assert e.value.info["reason"] == "no_int4"


def test_estimate_keeps_the_new_model_when_nothing_fits(tmp_path, monkeypatch, catalog_models):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    c = TestClient(server.app)
    gemma = {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": "google/gemma-4-26b-a4b-it",
             "precision": "fp8"}
    base = {"project": "my-gcp-project", "model": gemma,
            "traffic": {"peak_input_tps": 50000, "average_input_tps": 20000, "floor_input_tps": 2000}}
    assert c.post("/api/sessions/nf/estimate", json={"config": base}).status_code == 200
    assert (runs / "nf" / "plan.json").exists()
    big = {"model_garden_id": "acme/huge@huge-2000b", "params_b": 2000, "active_params_b": 40, "precision": "fp8"}
    r = c.post("/api/sessions/nf/estimate", json={"config": {**base, "model": big}})
    assert r.status_code == 200
    d = r.json()
    assert d["selected"]["model"]["model_garden_id"] == big["model_garden_id"] and d["selected"]["hardware"] is None
    assert d["no_fit"]["largest"]["accelerator_type"] == "NVIDIA_B200" and d["plan_error"]
    assert d["dashboard_data"] is None and not (runs / "nf" / "plan.json").exists()
    hw = c.post("/api/recommend/hardware", json={"config": {**base, "model": big}}).json()
    assert hw["ok"] is False and hw["no_fit"]["options"]


def test_ui_no_fit_panel_and_int4_toggle():
    js = (ASSETS / "app.js").read_text()
    html = (ASSETS / "app.html").read_text()
    assert 'data-prec="int4"' in html
    assert "function noFitHTML" in js and "S.session.no_fit" in js and "S.hw.no_fit" in js
    assert "availablePrecisions" in js


# --------------------------------------------------------------------------------------- B200
def test_b200_catalog_and_pricing(monkeypatch):
    g = catalog.GPUS["NVIDIA_B200"]
    assert g["mem_gb"] == 180 and g["machines"] == {8: ("a4-highgpu-8g", 224, 3968)} and g["kind"] == "gpu"
    assert catalog.FALLBACK_PRICES["NVIDIA_B200"]["vertex"] > catalog.FALLBACK_PRICES["NVIDIA_H200_141GB"]["vertex"] * 0.9

    def sku(desc, regions, price):
        return {"description": desc, "serviceRegions": regions, "pricingInfo": [{"pricingExpression": {
            "tieredRates": [{"unitPrice": {"units": str(int(price)), "nanos": int(round((price % 1) * 1e9))}}]}}]}
    v = [sku("Vertex AI: Online/Batch Prediction Nvidia B200 GPU with lssd running in Iowa", ["us-central1"], 18.5),
         sku("Vertex AI: Online/Batch Prediction Management fee on NVIDIA B200 with lssd in Iowa", ["us-central1"], 2.5),
         sku("Vertex AI: Online/Batch Prediction Nvidia GB200 GPU with lssd running in Iowa", ["us-central1"], 99.0)]
    c = [sku("A4 Nvidia B200 (1 gpu slice) running in Americas", ["us-central1"], 16.0),
         sku("A4 Nvidia B200 (1 gpu slice) attached to DWS Defined Duration VMs running in Americas", ["us-central1"], 8.0),
         sku("Commitment v1: A4 Nvidia B200 (1 gpu slice) in Americas for 1 Year", ["us-central1"], 11.0),
         sku("Commitment v1: A4 Nvidia B200 (1 gpu slice) in Americas for 3 Year", ["us-central1"], 7.0)]
    monkeypatch.setattr(pricing, "skus", lambda service, project: v if service == "vertex" else c)
    p = pricing.gpu_hour_prices("my-gcp-project", "us-central1", "NVIDIA_B200", 8)
    assert p["missing_skus"] == []
    assert p["vertex"] == pytest.approx(21.0) and p["on_demand"] == pytest.approx(16.0)
    assert p["cud_1y"] == pytest.approx(11.0) and p["vertex_cud_3y"] == pytest.approx(7.0 + 2.5 * 0.8)


def test_short_image_matches_backend():
    from aiops_dilemma import loadtest
    pinned = catalog.SERVING_IMAGES[0]
    assert loadtest.short_image(pinned) == "pytorch-vllm-serve:gemma4 (sha256:3fbc0e08…)"
    js = (ASSETS / "app.js").read_text()
    assert "entry && entry.tag ? `${repo}:${entry.tag} (${digest.slice(0, 15)}…)`" in js
