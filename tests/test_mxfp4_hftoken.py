"""MXFP4 precision (OpenAI gpt-oss) and the Hugging Face token setting.

No network: Hugging Face records are synthetic (shaped like the real gpt-oss Hub metadata, 2026-11-09), whoami is
the test stub (DILEMMA_TEST_HF_WHOAMI_STUB) or a patched urlopen, and no real token is used anywhere."""

import io
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aiops_dilemma import config, deploy, hfmeta, hftoken, modelinfo, planner, prompts, recommender, server

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "aiops_dilemma" / "assets"

OSS20, OSS120 = "openai/gpt-oss@gpt-oss-20b", "openai/gpt-oss@gpt-oss-120b"
GEMMA = "google/gemma4@gemma-4-26b-a4b-it"
QWEN = "qwen/qwen2@qwen2.5-32b-instruct"
GARDEN = [OSS20, OSS120, GEMMA, QWEN, "qwen/qwen2@qwen2.5-32b-instruct-awq"]


def _oss_record(n_layers=36, experts=128, total=116829156672, bf16=2167371072, u8=114661785600, hf_id="openai/gpt-oss-120b"):
    """A raw hfmeta cache record shaped like gpt-oss's (quantization_config mxfp4, U8 expert tensors)."""
    cfg = {"model_type": "gpt_oss", "hidden_size": 2880, "intermediate_size": 2880, "num_hidden_layers": n_layers,
           "num_local_experts": experts, "experts_per_token": 4, "num_attention_heads": 64, "num_key_value_heads": 8,
           "head_dim": 64, "vocab_size": 201088, "max_position_embeddings": 131072,
           "quantization_config": {"quant_method": "mxfp4",
                                   "modules_to_not_convert": ["model.layers.*.self_attn", "model.embed_tokens"]}}
    return {"format": hfmeta.FORMAT, "hf_id": hf_id, "fetched_at": 1762650000, "status": "ok", "config_status": 200,
            "api": {"id": hf_id, "createdAt": "2025-08-04T00:00:00Z", "gated": False,
                    "safetensors": {"parameters": {"BF16": bf16, "U8": u8}, "total": total},
                    "cardData": {"license": "apache-2.0"}, "config": {}},
            "config": cfg}


REC120 = _oss_record()
REC20 = _oss_record(n_layers=24, experts=32, total=20914757184, bf16=1804459584, u8=19110297600,
                    hf_id="openai/gpt-oss-20b")


@pytest.fixture()
def catalog_models(monkeypatch):
    """Curated catalog without live metadata (gpt-oss from the rules), no verified recipes."""
    models = modelinfo.curate_models(GARDEN)
    monkeypatch.setattr(modelinfo, "discover_models", lambda *a, **k: {"models": models, "sources": [], "families": []})
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a, **k: [])
    return {m["model_garden_id"]: m for m in models}


@pytest.fixture()
def live_models(monkeypatch):
    """Curated catalog with live Hugging Face metadata for gpt-oss (synthetic records, parsed by hfmeta.parse)."""
    recs = {"openai/gpt-oss-120b": hfmeta.parse(REC120), "openai/gpt-oss-20b": hfmeta.parse(REC20)}
    monkeypatch.setattr(hfmeta, "cached", lambda hf_id: recs.get(str(hf_id or "")))
    models = modelinfo.curate_models(GARDEN, hf_map={OSS20: "openai/gpt-oss-20b", OSS120: "openai/gpt-oss-120b"})
    monkeypatch.setattr(modelinfo, "discover_models", lambda *a, **k: {"models": models, "sources": [], "families": []})
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a, **k: [])
    return {m["model_garden_id"]: m for m in models}


# --------------------------------------------------------------------------------------- MXFP4: metadata + sizing
def test_hf_parse_detects_mxfp4_and_sizes_from_dtype_counts():
    f = hfmeta.parse(REC120)["fields"]
    assert f["native_precision"] == "mxfp4" and f["checkpoint_precision"] == "mxfp4"
    # 114.66B expert params x 4.25 bits + 2.17B BF16 x 2 bytes = ~65.3 GB (the real checkpoint is ~65 GB)
    assert f["mxfp4_weight_gb"] == pytest.approx(114.661785600 * 4.25 / 8 + 2.167371072 * 2, abs=0.01)
    assert "safetensors" in f["mxfp4_weight_note"]
    assert hfmeta.parse(REC20)["fields"]["mxfp4_weight_gb"] == pytest.approx(13.76, abs=0.01)


def test_hf_mxfp4_packed_u8_counts_are_taken_as_bytes():
    tc = REC120["config"]
    experts = 36 * 128 * 3 * 2880 * 2880
    b, note = hfmeta.mxfp4_weight_bytes({"U8": experts * 4.25 / 8, "BF16": 2.167e9}, tc)
    assert b == pytest.approx(experts * 4.25 / 8 + 2 * 2.167e9) and "bytes" in note
    assert hfmeta.mxfp4_weight_bytes({"BF16": 1e9}, tc) is None


def test_mxfp4_offered_only_for_gpt_oss(catalog_models, live_models):
    for cat in (catalog_models, live_models):
        for mid in (OSS20, OSS120):
            m = cat[mid]
            assert m["native_precision"] == "mxfp4" and m["precisions"] == ["mxfp4"], m["precisions"]
            assert planner.available_precisions(m) == ["mxfp4"]
            assert modelinfo.mxfp4_checkpoint(m) == mid
        for mid in (GEMMA, QWEN):
            assert "mxfp4" not in cat[mid]["precisions"] and modelinfo.mxfp4_checkpoint(cat[mid]) is None
    assert live_models[OSS120]["field_sources"]["native_precision"] == "hf"
    assert catalog_models[OSS120]["field_sources"]["native_precision"] == "rules"
    # a BF16 checkpoint next to an MXFP4 release would be offered too (only when it really exists)
    both = {**live_models[OSS20], "prequantized": {"bf16": "openai/gpt-oss@gpt-oss-20b-bf16"}, "precisions": ["bf16", "mxfp4"]}
    assert planner.available_precisions(both) == ["bf16", "mxfp4"]
    # a prequantized -mxfp4 checkpoint of another model folds into its precisions
    cur = {m["id"]: m for m in modelinfo.curate_models(["acme/foo@foo-30b-instruct", "acme/foo@foo-30b-instruct-mxfp4"])}
    foo = cur["foo-30b-instruct"]
    assert foo["prequantized"] == {"mxfp4": "acme/foo@foo-30b-instruct-mxfp4"} and foo["precisions"] == ["bf16", "fp8", "mxfp4"]


@pytest.mark.parametrize("raw,want", [("mxfp4", "mxfp4"), ("MXFP4", "mxfp4"), ("mx-fp4", "mxfp4"), ("nvfp4", "bf16")])
def test_mxfp4_precision_normalization(raw, want):
    assert config.normalize({"model": {"model_garden_id": "x/y@z-8b", "precision": raw}})["model"]["precision"] == want


def test_mxfp4_sizing_live_vs_estimate(catalog_models, live_models):
    live = config.normalize({"model": {"model_garden_id": OSS120, "hf_id": "openai/gpt-oss-120b", "precision": "mxfp4"}})
    assert modelinfo.precision_bytes(live["model"]) == pytest.approx(65.25e9, rel=0.002)
    opts = planner.precision_options(live_models[OSS120], "mxfp4")
    assert [o["precision"] for o in opts] == ["mxfp4"]
    assert opts[0]["weights_source"] == "hf" and opts[0]["size_text"] == "65 GB" and opts[0]["how"] == "native"
    assert opts[0]["quality_note"] is None
    # without live metadata: ~0.55 byte per parameter, labelled as an estimate
    est = planner.precision_options(catalog_models[OSS120], "mxfp4")[0]
    assert est["weights_source"] == "estimate" and "(estimated)" in est["size_text"]
    assert est["weights_gb"] == pytest.approx(117.0 * 0.55, abs=0.1)


def test_mxfp4_estimate_when_params_override(live_models):
    """A config whose params_b differs from the catalog's isn't sized with the catalog's checkpoint bytes."""
    m = {"model_garden_id": OSS120, "hf_id": "openai/gpt-oss-120b", "precision": "mxfp4", "params_b": 50}
    assert modelinfo.precision_bytes(m) == pytest.approx(50e9 * 0.55)


# --------------------------------------------------------------------------------------- MXFP4: GPUs + plan + deploy
def test_mxfp4_candidates_gpu_only_with_support_labels(live_models):
    cfg = config.normalize({"project": "my-gcp-project",
                            "model": {"model_garden_id": OSS20, "hf_id": "openai/gpt-oss-20b", "precision": "mxfp4"}})
    p = planner.plan(cfg)
    assert p["weights_gb"] == pytest.approx(13.8, abs=0.1)
    by = {c["accelerator_type"]: c for c in p["candidates"]}
    assert not any(c["kind"] == "tpu" for c in p["candidates"])
    assert by["NVIDIA_H100_80GB"]["precision_support"] == "documented"
    assert by["NVIDIA_A100_80GB"]["precision_support"] == "documented"
    assert by["NVIDIA_L4"]["precision_support"] == "untested" and "untested" in by["NVIDIA_L4"]["precision_support_note"]
    # no live recipes (no project): RTX PRO 6000 falls back to the recipe seen on 2026-11-09, labelled as such
    assert by["NVIDIA_RTX_PRO_6000"]["precision_support"] == "recipe_seen"
    assert "not re-checked" in by["NVIDIA_RTX_PRO_6000"]["precision_support_label"]
    assert p["candidates"][-1]["precision_support"] == "untested"            # untested GPUs rank last
    assert p["deployment_model"]["quantization"] == "native" and p["deployment_model"]["model_garden_id"] == OSS20
    assert "MXFP4: rough" in p["chosen"]["est_basis_note"]
    with pytest.raises(SystemExit):
        planner.plan(config.normalize({"project": "my-gcp-project", "deployment": {
            "mode": "manual", "accelerator_type": "TPU_V6E", "accelerator_count": 4},
            "model": {"model_garden_id": OSS20, "precision": "mxfp4"}}))


def test_mxfp4_verified_recipe_marks_the_gpu(live_models, monkeypatch):
    rec = {"machine_type": "g4-standard-96", "accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 2,
           "image": "us-docker.pkg.dev/vertex-ai/x/pytorch-vllm-serve:tag", "env": {"MODEL_ID": "openai/gpt-oss-120b"},
           "args": ["python3", "-m", "vllm.entrypoints.api_server", "--model=gs://bucket/openai/gpt-oss-120b",
                    "--tensor-parallel-size=2"], "predict_route": "/generate", "health_route": "/ping", "port": 7080}
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a, **k: [rec])
    cfg = config.normalize({"project": "real-project-1",
                            "model": {"model_garden_id": OSS120, "hf_id": "openai/gpt-oss-120b", "precision": "mxfp4"}})
    p = planner.plan(cfg)
    rtx = p["candidates"][0]
    assert rtx["accelerator_type"] == "NVIDIA_RTX_PRO_6000" and rtx["verified"] and rtx["precision_support"] == "verified"
    _, args, env, _ = deploy._args(cfg, rtx, rec)
    assert not any(a.startswith("--quantization") for a in args) and "--model=gs://bucket/openai/gpt-oss-120b" in args
    _, args, _, _ = deploy._args(cfg, {"accelerator_count": 1}, None)          # generic fallback: no flag either
    assert not any(a.startswith("--quantization") for a in args)


def test_adapted_recipe_prefers_the_same_gpu_family():
    a100 = {"accelerator_type": "NVIDIA_A100_80GB", "accelerator_count": 1, "image": "pytorch-vllm-serve:x",
            "args": ["--tensor-parallel-size=1"], "env": {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}}
    rtx = {**a100, "accelerator_type": "NVIDIA_RTX_PRO_6000", "env": {}}
    sgl = {**a100, "accelerator_type": "NVIDIA_TESLA_A100", "image": "sglang-serve", "args": ["--tp=8"]}
    assert deploy.base_recipe([rtx, sgl, a100], "NVIDIA_TESLA_A100") is a100      # Ampere keeps TRITON_ATTN
    assert deploy.base_recipe([rtx, a100], "NVIDIA_L4") is rtx
    assert deploy.base_recipe([sgl], "NVIDIA_L4") is None


def test_mxfp4_refusals(live_models):
    bf16 = config.normalize({"project": "my-gcp-project", "model": {"model_garden_id": OSS120, "precision": "bf16"}})
    with pytest.raises(planner.NoFitError) as e:
        planner.plan(bf16)
    assert e.value.info["reason"] == "native_only" and "MXFP4 only" in e.value.info["message"]
    assert {"type": "precision", "value": "mxfp4"} in [o.get("action") for o in e.value.info["options"]]
    gem = config.normalize({"project": "my-gcp-project", "model": {"model_garden_id": GEMMA, "precision": "mxfp4"}})
    with pytest.raises(planner.NoFitError) as e:
        planner.plan(gem)
    assert e.value.info["reason"] == "no_mxfp4"


def test_mxfp4_decode_bytes_follow_the_weights(live_models):
    req = {"avg_input_tokens": 2000, "avg_output_tokens": 1500}
    m = {"model_garden_id": OSS120, "hf_id": "openai/gpt-oss-120b", "params_b": 116.83, "active_params_b": 5.1}
    rate = {p: planner._spec_rate({"model": {**m, "precision": p}, "requests": req}, "NVIDIA_H100_80GB", 2)
            for p in ("bf16", "mxfp4")}
    assert rate["mxfp4"] > rate["bf16"]
    heavier = planner._spec_rate({"model": {**m, "precision": "mxfp4", "mxfp4_weight_gb": 116.83},
                                  "requests": req}, "NVIDIA_H100_80GB", 2)    # 1 byte/param: slower decode
    assert heavier < rate["mxfp4"]


def test_recommender_picks_mxfp4_for_gpt_oss(live_models):
    res = recommender.recommend({"task_type": "reasoning"}, {"avg_input_tokens": 6000, "avg_output_tokens": 500},
                                {"peak_input_tps": 50000}, models=list(live_models.values()))
    by = {m["model_garden_id"]: m for m in res["recommendations"]}
    for mid in (OSS20, OSS120):
        o = by[mid]
        assert o["recommended_precision"] == "mxfp4" and not o["needs_quality_check"]
        assert [x["precision"] for x in o["precision_options"]] == ["mxfp4"]
        assert "MXFP4" in o["precision_reason"] and "BF16 checkpoint" in o["precision_reason"]
    assert "65 GB" in by[OSS120]["precision_reason"]
    assert all(by[m]["recommended_precision"] != "mxfp4" for m in (GEMMA, QWEN))


def test_ui_mxfp4_toggle_only_for_gpt_oss(live_models):
    html = (ASSETS / "app.html").read_text()
    assert 'data-prec="mxfp4"' in html and 'id="precMxfp4"' in html
    btn = html[html.index('id="precMxfp4"'):]
    assert btn[:btn.index(">")].rstrip().endswith("hidden") or " hidden" in btn[:btn.index(">")]
    js = (ASSETS / "app.js").read_text()
    fn = js[js.index("  const INT4_KEYS"):js.index("  const normPrec")]
    models = {k: {kk: v.get(kk) for kk in ("precisions", "native_precision", "prequantized", "fp8_only")}
              for k, v in live_models.items()}
    script = fn + f"\nconst M = {json.dumps(models)};\nconst out = {{}};\nfor (const k in M) out[k] = availablePrecisions(M[k]);\n" \
                  "console.log(JSON.stringify(out));"
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out[OSS20] == ["mxfp4"] and out[OSS120] == ["mxfp4"]
    assert "mxfp4" not in out[GEMMA] and "mxfp4" not in out[QWEN] and out[QWEN] == ["bf16", "fp8", "int4"]
    assert "function precSupportHTML" in js and "MXFP4 untested on this GPU" in js


# --------------------------------------------------------------------------------------- Hugging Face token
@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    monkeypatch.setenv("DILEMMA_TEST_HF_WHOAMI_STUB", "1")
    monkeypatch.setenv("DILEMMA_CACHE_DIR", str(tmp_path / "cache"))
    return TestClient(server.app)


FAKE = "hf_validFAKEtoken0000abcd"      # accepted by the stub; not a real token


def test_save_validates_and_writes_a_private_file_outside_runs(client):
    r = client.post("/api/settings/hf-token", json={"token": FAKE})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] and d["user"] == "test-user" and d["orgs"] == ["test-org"] and d["token_hint"] == "hf_…abcd"
    assert FAKE not in r.text
    p = hftoken.token_path()
    assert p.read_text().strip() == FAKE
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    assert Path(os.environ["DILEMMA_CONFIG_DIR"]) == p.parent
    runs = server.RUNS_DIR
    assert not str(p).startswith(str(runs)) and not str(p).startswith(str(ROOT))
    assert not any(FAKE in f.read_text(errors="ignore") for f in runs.rglob("*") if f.is_file())
    g = client.get("/api/settings")
    assert FAKE not in g.text
    assert g.json()["hf_token"] == {**g.json()["hf_token"], "set": True, "source": "file", "hint": "hf_…abcd",
                                    "user": "test-user"}


def test_invalid_token_is_refused_and_not_stored(client):
    r = client.post("/api/settings/hf-token", json={"token": "hf_wrongtoken1234"})
    assert r.status_code == 400 and "didn't accept" in r.json()["detail"]
    assert not hftoken.token_path().exists()
    assert client.post("/api/settings/hf-token", json={"token": ""}).status_code == 400
    assert client.get("/api/settings").json()["hf_token"]["source"] == "none"


def test_env_var_wins_over_the_file(client, monkeypatch):
    client.post("/api/settings/hf-token", json={"token": FAKE})
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_validENVsecond9999")
    assert hftoken.current() == ("hf_validENVsecond9999", "env")
    monkeypatch.setenv("HF_TOKEN", "hf_validENVfirst00001")
    t = client.get("/api/settings").json()["hf_token"]
    assert t["source"] == "env" and t["env_var"] == "HF_TOKEN" and t["hint"] == "hf_…0001"
    assert "Authorization" not in hftoken.headers()          # stub tokens are never sent to huggingface.co
    monkeypatch.delenv(hftoken.STUB_ENV)
    assert hftoken.headers()["Authorization"] == "Bearer hf_validENVfirst00001"
    monkeypatch.setenv(hftoken.STUB_ENV, "1")
    r = client.post("/api/settings/hf-token", json={"token": FAKE})
    assert r.json()["overridden_by_env"] is True and r.json()["active_source"] == "env"


def test_delete_removes_the_token(client):
    client.post("/api/settings/hf-token", json={"token": FAKE})
    r = client.delete("/api/settings/hf-token")
    assert r.status_code == 200 and r.json()["removed"] is True
    assert not hftoken.token_path().exists() and hftoken.token() is None
    assert client.get("/api/settings").json()["hf_token"]["set"] is False
    assert "Authorization" not in hftoken.headers()


def test_read_only_config_dir_keeps_the_token_in_memory(client, monkeypatch):
    def ro(_tok):
        raise OSError(30, "Read-only file system")
    monkeypatch.setattr(hftoken, "_write_file", ro)
    r = client.post("/api/settings/hf-token", json={"token": FAKE})
    assert r.json()["stored"] == "memory" and r.json()["hf_token"]["source"] == "memory"
    assert hftoken.job_env() == {"HF_TOKEN": FAKE}            # benchmark jobs (subprocesses) still get it
    client.delete("/api/settings/hf-token")
    assert hftoken.token() is None and hftoken.job_env() == {}


def test_save_invalidates_cached_401_metadata(client, tmp_path):
    d = tmp_path / "cache" / "hf_meta"
    d.mkdir(parents=True)
    gated = {**_oss_record(hf_id="meta-llama/Llama-3.3-70B-Instruct"), "config": None, "config_status": 401}
    denied = {"format": 1, "hf_id": "acme/private", "status": "not_found", "http_status": 401, "fetched_at": 1}
    ok = _oss_record()
    for name, rec in (("meta-llama__llama-3.3-70b-instruct", gated), ("acme__private", denied),
                      ("openai__gpt-oss-120b", ok)):
        (d / f"{name}.json").write_text(json.dumps(rec))
    assert "gated: add a Hugging Face token in Settings" in " ".join(hfmeta.parse(gated)["notes"])
    r = client.post("/api/settings/hf-token", json={"token": FAKE})
    assert r.json()["refresh"]["invalidated"] == 2
    assert sorted(p.name for p in d.iterdir()) == ["openai__gpt-oss-120b.json"]


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_token_is_sent_as_bearer_to_hugging_face(monkeypatch, tmp_path):
    hftoken.save(FAKE)
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append((req.full_url, dict(req.header_items())))
        if req.full_url.endswith("tokenizer.json"):
            return _Resp(b"{}")
        return _Resp(json.dumps({"id": "meta-llama/x"}).encode())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    code, body = hfmeta._get("https://huggingface.co/api/models/meta-llama/x")
    assert code == 200 and body["id"] == "meta-llama/x"
    prompts._tokenizer("meta-llama/x", tmp_path / "tok")
    modelinfo._get_json("https://huggingface.co/api/models?filter=text-generation")
    assert len(seen) == 3 and all(h.get("Authorization") == f"Bearer {FAKE}" for _, h in seen)
    assert seen[1][0] == "https://huggingface.co/meta-llama/x/resolve/main/tokenizer.json"
    hftoken.delete()
    seen.clear()
    hfmeta._get("https://huggingface.co/api/models/meta-llama/x")
    assert "Authorization" not in seen[0][1]


def test_validate_calls_whoami_with_the_token(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen.update(url=req.full_url, auth=req.get_header("Authorization"))
        return _Resp(json.dumps({"name": "someone", "orgs": [{"name": "org-a"}, {"name": "org-b"}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    res = hftoken.validate("hf_anyFAKEvalue1234")
    assert res == {"ok": True, "user": "someone", "orgs": ["org-a", "org-b"]}
    assert seen == {"url": hftoken.WHOAMI_URL, "auth": "Bearer hf_anyFAKEvalue1234"}


def test_ui_settings_dialog_and_gated_note():
    html = (ASSETS / "app.html").read_text()
    js = (ASSETS / "app.js").read_text()
    assert 'id="btnSettings"' in html and 'id="settingsDlg"' in html and 'type="password" id="hfTokenInput"' in html
    assert "https://huggingface.co/settings/tokens" in html and "A read-only token is enough" in html
    assert "Add a Hugging Face token in Settings" in js and "/api/settings/hf-token" in js
    assert "Set by the ${esc(t.env_var || \"HF_TOKEN\")} environment variable (read-only here)" in js
