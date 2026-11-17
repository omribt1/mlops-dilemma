"""Custom serving containers and tuned-config import (serving.py): validation, argument merging, the container spec,
the sitecustomize patch, Docker Hub images via Artifact Registry, deploy wiring, provenance, server and CLI."""
import base64
import json
import os
import subprocess
import sys
import types

import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import analyze, config, deploy, modelinfo, report, server, serving

CANDIDATE = '''"""An alpha_inference candidate (same shape as examples/email_triage/results/best_candidate.py)."""
import os
os.system("touch {marker}")     # module-level code: must never run

# EVOLVE-BLOCK-START
def get_candidate_config():
    return {{
        "plain_title": "2-token n-gram spec + sparse MoE tiles",
        "hypothesis": "Fewer speculative tokens waste less verification work.",
        "category": "vllm_flags",
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--quantization=fp8",
            '--speculative-config={{"method":"ngram_gpu","num_speculative_tokens":2}}',
            '--compilation-config={{"cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}}',
        ],
        "env_vars": {{"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}},
        "moe_tuned_config": {{
            1: {{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}},
            512: {{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4}},
        }},
        "custom_kernel_code": "",
    }}
# EVOLVE-BLOCK-END
'''


def _cfg(**dep):
    return config.normalize({"name": "byoc", "project": "my-gcp-project",
                             "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it",
                                       "hf_id": "google/gemma-4-26b-a4b-it", "precision": "fp8"},
                             "traffic": {"average_input_tps": 30000, "preset": "us_core_hours"},
                             "requests": {"avg_input_tokens": 5000, "avg_output_tokens": 500, "pct_over_32k": 1.25,
                                          "pct_over_128k": 0.3, "medium_mean": 60000, "large_mean": 170000},
                             "deployment": dep})


def _candidate(tmp_path):
    marker = tmp_path / "executed"
    return CANDIDATE.format(marker=marker), marker


# ---------------------------------------------------------------- arguments

def test_merge_args_replaces_the_recipes_flag():
    base = ["python", "-m", "vllm.entrypoints.api_server", "--max-num-seqs=128", "--gpu-memory-utilization", "0.9",
            "--enable-prefix-caching"]
    out = serving.merge_args(base, ["--max-num-seqs=256", "--gpu-memory-utilization=0.95", "--async-scheduling"])
    assert out == ["python", "-m", "vllm.entrypoints.api_server", "--enable-prefix-caching", "--max-num-seqs=256",
                   "--gpu-memory-utilization=0.95", "--async-scheduling"]
    assert serving.merge_args(base, []) == base
    assert serving.arg_value(out, "--max-num-seqs") == "256" and serving.arg_value(base, "--gpu-memory-utilization") == "0.9"


def test_extra_args_merge_into_the_recipe_in_append_mode():
    cfg = _cfg(extra_args=["--max-model-len=262144", "--max-num-seqs=256"])
    cfg["model"]["max_model_len"] = 131072
    verified = {"image": "us-docker.pkg.dev/v/r/vllm:1", "args": ["python", "-m", "x", "--max-num-seqs=64"],
                "env": {"MODEL_ID": "m"}, "predict_route": "/generate", "health_route": "/ping", "port": 8080}
    _, args, _, _ = deploy._args(cfg, {"accelerator_count": 1}, verified)
    assert args.count("--max-num-seqs=256") == 1 and "--max-num-seqs=64" not in args
    assert [a for a in args if a.startswith("--max-model-len")] == ["--max-model-len=262144"]   # ours replaces the session's


# ---------------------------------------------------------------- validation

def test_normalize_accepts_a_full_custom_container():
    d = {"image": "docker.io/vllm/vllm-openai:v0.31.0", "command": "vllm\nserve", "args_mode": "replace",
         "extra_args": "google/gemma-4-26B-A4B-it\n--max-num-seqs=256\n", "env": ["A=1", "B=x=y"],
         "routes": {"predict": "/v1/chat/completions", "health": "/health", "port": "8080"}, "shared_memory_mb": "81920",
         "sitecustomize": "x = 1\n", "tuned": {"title": "t"}}
    assert serving.normalize(d) == []
    assert d["command"] == ["vllm", "serve"] and d["extra_args"] == ["google/gemma-4-26B-A4B-it", "--max-num-seqs=256"]
    assert d["env"] == {"A": "1", "B": "x=y"} and d["routes"]["port"] == 8080 and d["shared_memory_mb"] == 81920
    assert serving.is_custom(d) and not serving.is_custom({"image": "x", "extra_args": ["--a"]})


@pytest.mark.parametrize("bad, msg", [
    ({"env": {"AIP_HTTP_PORT": "1"}}, "reserved"),
    ({"env": {"HF_TOKEN": "hf_x"}}, "pass_hf_token"),
    ({"env": {"1BAD": "x"}}, "valid variable name"),
    ({"env": {serving.PATCH_ENV: "x"}}, "sitecustomize"),
    ({"args_mode": "replace", "extra_args": []}, "full argument list"),
    ({"args_mode": "sideways"}, "args_mode"),
    ({"sitecustomize": "x = 1"}, "needs deployment.command"),
    ({"sitecustomize": "def (:", "command": ["vllm"]}, "not valid Python"),
    ({"routes": {"predict": "no-slash"}}, "path"),
    ({"routes": {"port": 99999}}, "port"),
    ({"shared_memory_mb": 10}, "shared_memory_mb"),
    ({"image": "vllm/vllm-openai:latest"}, "Docker Hub"),
])
def test_normalize_rejects(bad, msg):
    errs = serving.normalize(dict(bad))
    assert errs and any(msg in e for e in errs), errs


def test_cloud_mode_refuses_to_pass_the_hf_token():
    assert any("turned off" in e for e in serving.normalize({"pass_hf_token": True}, cloud=True))
    assert serving.normalize({"pass_hf_token": True}) == []


def test_config_keeps_serving_errors_out_of_the_yaml():
    cfg = _cfg(env={"AIP_X": "1"})
    assert cfg["_serving_errors"] and "_serving_errors" not in config.to_yaml(cfg)
    assert "_serving_errors" not in _cfg(env={"OK": "1"})


# ---------------------------------------------------------------- import

def test_import_reads_the_candidate_without_running_it(tmp_path):
    text, marker = _candidate(tmp_path)
    res = serving.import_candidate(text, _cfg(accelerator_count=1), filename="best_candidate.py")
    assert not marker.exists()                                   # module-level code never ran
    d = res["deployment"]
    assert d["args_mode"] == "replace" and d["extra_args"][0] == "google/gemma-4-26B-A4B-it"
    assert d["image"] == serving.UPSTREAM_VLLM_IMAGE and d["command"] == ["vllm", "serve"]
    assert d["routes"] == {"predict": "/v1/chat/completions", "health": "/health", "port": 8080}
    assert d["env"]["HF_HOME"] == "/dev/shm/hf_cache" and d["env"]["PYTHONUNBUFFERED"] == "1"
    assert d["pass_hf_token"] is True and d["tuned"]["title"].startswith("2-token")
    assert "_DL_MOE_TUNED_CONFIG" in d["sitecustomize"] and "512" in d["sitecustomize"]
    compile(d["sitecustomize"], "sitecustomize.py", "exec")
    assert res["summary"]["moe_tile_sizes"] == 2 and res["summary"]["max_model_len"] == 8192
    w = " ".join(res["warnings"])
    assert "8,192" in w and "will fail" in w                      # 8K context vs this workload's 170K prompts
    assert "Docker Hub" in w and "model is" not in w              # same model (case-insensitive), so no mismatch warning
    assert not any(a.startswith("--speculative-config") for a in d["extra_args"]) and "left out" in w
    assert res["summary"]["dropped_args"][0].startswith("--speculative-config")
    assert serving.normalize(dict(d)) == []


def test_import_warnings_for_a_different_model_precision_and_gpu_count(tmp_path):
    text, _ = _candidate(tmp_path)
    cfg = _cfg(accelerator_count=2)
    cfg["model"].update(hf_id="google/gemma-4-31b-it", precision="bf16")
    cfg["requests"].update(pct_over_32k=0, pct_over_128k=0)
    w = " ".join(serving.import_candidate(text, cfg)["warnings"])
    assert "this workload's model is google/gemma-4-31b-it" in w and "serves FP8" in w and "has 2" in w


def test_import_flags_short_context_even_without_long_prompts(tmp_path):
    text, _ = _candidate(tmp_path)
    cfg = _cfg()
    cfg["requests"].update(pct_over_32k=0, pct_over_128k=0)
    assert serving.needed_context(cfg["requests"]) == 32000 + 1000 + 512
    assert any("will fail" in x for x in serving.import_candidate(text, cfg)["warnings"])


def test_import_in_cloud_mode_does_not_pass_the_token(tmp_path):
    text, _ = _candidate(tmp_path)
    res = serving.import_candidate(text, _cfg(), cloud=True)
    assert res["deployment"]["pass_hf_token"] is False and any("can't pass a Hugging Face token" in w for w in res["warnings"])


def test_import_flags_kernel_code_and_accepts_json():
    data = {"plain_title": "k", "vllm_args": ["m", "--port=9000"], "env_vars": {},
            "custom_kernel_code": "register_vllm_patch('x', lambda m: None)\n"}
    res = serving.import_candidate(json.dumps(data), _cfg())
    assert res["deployment"]["routes"]["port"] == 9000
    assert "# ---- custom kernel code" in res["deployment"]["sitecustomize"]
    assert any("custom kernel code" in w for w in res["warnings"])


@pytest.mark.parametrize("text, msg", [
    ("x = 1\n", "get_candidate_config"),
    ("def get_candidate_config():\n    return build()\n", "literal"),
    ("def get_candidate_config():\n    return {'vllm_args': []}\n", "vllm_args"),
    ("def get_candidate_config(:\n", "not valid Python"),
    ('{"vllm_args": ["m"], "moe_tuned_config": {"1": {"BLOCK_SIZE_M": "big"}}}', "whole numbers"),
    ("{not json", "not valid JSON"),
])
def test_parse_candidate_rejects(text, msg):
    with pytest.raises(ValueError, match=msg):
        serving.parse_candidate(text)


# ---------------------------------------------------------------- patch

def test_patch_hooks_run_after_import_and_moe_schedule_applies(tmp_path, monkeypatch):
    """Execute the generated sitecustomize against fake modules: a kernel-code hook and the fused-MoE schedule."""
    (tmp_path / "dl_fake_mod.py").write_text("value = 1\n")
    pkg = tmp_path / "vllm" / "model_executor" / "layers" / "fused_moe"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "vllm", tmp_path / "vllm" / "model_executor", tmp_path / "vllm" / "model_executor" / "layers", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "fused_moe.py").write_text("def get_moe_configs(E, N, dtype, block_n=None, block_k=None):\n    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    src = serving.compose_patch({1: {"BLOCK_SIZE_M": 16}, "512": {"BLOCK_SIZE_M": 64}},
                                "register_vllm_patch('dl_fake_mod', lambda m: setattr(m, 'patched', True))")
    before = list(sys.meta_path)
    try:
        exec(compile(src, "sitecustomize.py", "exec"), {"__name__": "sitecustomize"})
        import dl_fake_mod
        from vllm.model_executor.layers.fused_moe import fused_moe
        assert dl_fake_mod.patched is True
        assert fused_moe.get_moe_configs(128, 704, None) == {1: {"BLOCK_SIZE_M": 16}, 512: {"BLOCK_SIZE_M": 64}}
    finally:
        sys.meta_path[:] = before
        for m in [k for k in sys.modules if k == "dl_fake_mod" or k == "vllm" or k.startswith("vllm.")]:
            del sys.modules[m]
    assert serving.compose_patch(None, "") is None


def test_wrapped_command_writes_the_patch_and_execs_command_with_args(tmp_path):
    env = {**os.environ, serving.PATCH_ENV: base64.b64encode(b"MARK = 42\n").decode(), "PYTHONPATH": ""}
    cmd = serving.wrap_command([sys.executable, "-c", "import sys, sitecustomize; print(sitecustomize.MARK, sys.argv[1:])"])
    out = subprocess.run(cmd + ["--max-num-seqs=256", "{\"a\": 1, \"b\": 2}"], env=env, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "42 ['--max-num-seqs=256', '{\"a\": 1, \"b\": 2}']"


# ---------------------------------------------------------------- container spec

BASE = {"image": "us-docker.pkg.dev/v/r/vllm:1", "args": ["python", "-m", "x", "--max-num-seqs=64"],
        "env": {"MODEL_ID": "m"}, "routes": {"predict": "/generate", "health": "/ping", "port": 8080}}


def test_container_spec_replace_mode_with_patch_and_token():
    d = {"image": "docker.io/vllm/vllm-openai:v0.31.0", "command": ["vllm", "serve"], "args_mode": "replace",
         "extra_args": ["m", "--port=8000"], "env": {"HF_HOME": "/dev/shm/hf_cache"}, "routes": None,
         "shared_memory_mb": None, "pass_hf_token": True, "sitecustomize": "x = 1\n", "tuned": {"title": "t"}}
    spec = serving.container_spec(d, BASE, "g4-standard-48", hf_token="hf_secret")
    assert spec["args"] == ["m", "--port=8000"] and spec["routes"] == {"predict": "/v1/chat/completions",
                                                                       "health": "/health", "port": 8000}
    assert spec["command"][:2] == ["bash", "-lc"] and spec["command"][-3:] == ["--", "vllm", "serve"]
    assert "MODEL_ID" not in spec["env"] and spec["env"]["HF_TOKEN"] == "hf_secret"
    assert base64.b64decode(spec["env"][serving.PATCH_ENV]) == b"x = 1\n"
    assert spec["shared_memory_mb"] == serving.recommended_shm_mb("g4-standard-48") >= 65536
    rec = json.dumps(spec["record"])
    assert "hf_secret" not in rec and serving.PATCH_ENV not in rec and "x = 1" not in rec
    assert spec["record"]["patch"]["bytes"] == 6 and spec["record"]["hf_token_passed"] is True
    assert spec["record"]["command"] == ["vllm", "serve"]
    assert spec["record"]["api"] == "raw" and spec["record"]["served_model"] == "m"
    d2 = dict(d, extra_args=["m", "--served-model-name", "gemma tuned"])
    assert serving.container_spec(d2, BASE, "g4-standard-48")["record"]["served_model"] == "gemma"


def test_container_spec_append_mode_keeps_the_recipe():
    d = {"extra_args": [], "args_mode": "append", "env": {"X": "1"}, "routes": {"health": "/healthz"}}
    spec = serving.container_spec(d, BASE, "g4-standard-48")
    assert spec["image"] == BASE["image"] and spec["args"] == BASE["args"] and spec["command"] is None
    assert spec["env"] == {"MODEL_ID": "m", "X": "1"} and spec["routes"]["predict"] == "/generate"
    assert spec["routes"]["health"] == "/healthz" and spec["shared_memory_mb"] == serving.DEFAULT_SHM_MB
    assert "api" not in spec["record"] and "served_model" not in spec["record"]     # Model Garden route, "default"


# ---------------------------------------------------------------- Docker Hub via Artifact Registry

def test_dockerhub_uri_mapping():
    assert serving.dockerhub_ar_uri("docker.io/vllm/vllm-openai:v0.31.0", "my-gcp-project", "us-central1") == \
        "us-central1-docker.pkg.dev/my-gcp-project/dilemma-dockerhub/vllm/vllm-openai:v0.31.0"
    assert serving.dockerhub_ar_uri("docker.io/ubuntu", "p", "europe-west4").endswith("/library/ubuntu:latest")
    assert serving.valid_image_ref(serving.dockerhub_ar_uri("docker.io/vllm/vllm-openai@sha256:" + "a" * 64, "p", "us-east4"))


class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._b = status, body or {}
        self.text = json.dumps(self._b)

    def json(self):
        return self._b


class _Session:
    def __init__(self, get_status):
        self.get_status, self.posts = get_status, []

    def get(self, url, **k):
        if "/operations/" in url:
            return _Resp(200, {"done": True})
        return _Resp(self.get_status)

    def post(self, url, params=None, json=None, **k):
        self.posts.append((url, params, json))
        return _Resp(200, {"name": "projects/p/locations/r/operations/1", "done": False})


def test_ensure_dockerhub_repo(monkeypatch):
    monkeypatch.setattr(serving.time, "sleep", lambda s: None)
    s = _Session(200)
    assert serving.ensure_dockerhub_repo("p", "us-central1", session=s).endswith("/repositories/dilemma-dockerhub")
    assert s.posts == []
    s = _Session(404)
    serving.ensure_dockerhub_repo("p", "us-central1", session=s)
    (url, params, body), = s.posts
    assert params == {"repositoryId": "dilemma-dockerhub"} and body["mode"] == "REMOTE_REPOSITORY"
    assert body["remoteRepositoryConfig"]["dockerRepository"]["publicRepository"] == "DOCKER_HUB"
    with pytest.raises(SystemExit, match="gcloud artifacts repositories create dilemma-dockerhub"):
        serving.ensure_dockerhub_repo("p", "us-central1", session=_Session(403))


# ---------------------------------------------------------------- deploy, provenance, report

def test_deploy_a_tuned_config_end_to_end(monkeypatch, tmp_path):
    fake_vx = types.ModuleType("vertexai")
    fake_vx.init = lambda **k: None
    fake_vx.model_garden = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vertexai", fake_vx)
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a: [])
    repo = []
    monkeypatch.setattr(serving, "ensure_dockerhub_repo", lambda p, r: repo.append((p, r)))
    monkeypatch.setattr(deploy, "_project", lambda cfg: "my-gcp-project")
    warmed = []
    monkeypatch.setattr(serving, "warm_remote_image", lambda uri, p: warmed.append(uri))
    monkeypatch.setenv("HF_TOKEN", "hf_from_job")
    seen = {}

    def fake_create(cfg, run_dir, name, secs, common, container, ch):
        seen.update(container)
        return {"endpoint": "projects/my-gcp-project/locations/us-central1/endpoints/1", "dedicated_dns": "x",
                "inference_timeout_s": 600}
    monkeypatch.setattr(deploy, "_deploy_with_timeout", fake_create)
    text, _ = _candidate(tmp_path)
    cfg = _cfg()
    cfg = serving.apply_import(cfg, serving.import_candidate(text, cfg)["deployment"])
    plan = {"chosen": {"machine_type": "g4-standard-48", "accelerator_type": "NVIDIA_RTX_PRO_6000",
                       "accelerator_count": 1}}
    deploy.deploy(cfg, plan, tmp_path)
    saved = json.loads((tmp_path / "endpoint.json").read_text())
    ar = "us-central1-docker.pkg.dev/my-gcp-project/dilemma-dockerhub/vllm/vllm-openai:v0.31.0"
    assert repo == [("my-gcp-project", "us-central1")] and seen["serving_container_image_uri"] == ar and warmed == [ar]
    assert seen["serving_container_args"][0] == "google/gemma-4-26B-A4B-it"
    assert seen["serving_container_command"][-3:] == ["--", "vllm", "serve"]
    assert seen["serving_container_environment_variables"]["HF_TOKEN"] == "hf_from_job"
    assert seen["serving_container_predict_route"] == "/v1/chat/completions"
    assert seen["serving_container_shared_memory_size_mb"] >= 65536
    assert "hf_from_job" not in json.dumps(saved)
    assert saved["image"] == ar and saved["image_requested"] == serving.UPSTREAM_VLLM_IMAGE
    assert saved["image_source"] == "tuned config" and saved["tuned"]["title"].startswith("2-token")
    prov = analyze.serving_provenance(saved)
    assert prov["patch"]["sha256"] and prov["command"] == ["vllm", "serve"]
    line = report.serving_line({"serving": prov})
    assert 'tuned config "2-token' in line and "vllm-openai:v0.31.0" in line and "a Python patch" in line


def test_deploy_refuses_an_invalid_custom_container(monkeypatch, tmp_path):
    fake_vx = types.ModuleType("vertexai")
    fake_vx.model_garden = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vertexai", fake_vx)
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a: [])
    cfg = _cfg()
    cfg["deployment"]["sitecustomize"] = "x = 1"       # no command
    plan = {"chosen": {"machine_type": "g4-standard-48", "accelerator_type": "NVIDIA_RTX_PRO_6000",
                       "accelerator_count": 1}}
    with pytest.raises(SystemExit, match="needs deployment.command"):
        deploy.deploy(cfg, plan, tmp_path)


def test_reference_entry_warns_about_custom_serving(tmp_path):
    from aiops_dilemma import planner
    (tmp_path / "measured.json").write_text(json.dumps({
        "input_tps_per_gpu": 17000, "levels": [{"concurrency": 64}, {"concurrency": 128}],
        "peak": {"concurrency": 64}, "endpoint": {"accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 1,
                     "machine_type": "g4-standard-48", "precision": "fp8"},
        "serving": {"image": "x", "tuned": {"title": "t"}}}))
    _, warn = planner.reference_entry(_cfg(), tmp_path)
    assert any("custom serving container" in w for w in warn)


# ---------------------------------------------------------------- server and CLI

def test_api_import_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(server.app)
    text, marker = _candidate(tmp_path)
    cfg = config.to_yaml(_cfg())
    raw = yaml.safe_load(cfg)
    r = c.post("/api/sessions/byoc/serving/import", json={"text": text, "filename": "best_candidate.py", "config": raw})
    assert r.status_code == 200, r.text
    dep = r.json()["deployment"]
    assert dep["args_mode"] == "replace" and not marker.exists()
    assert c.post("/api/sessions/byoc/serving/import", json={"text": "x = 1", "config": raw}).status_code == 400
    assert c.post("/api/sessions/byoc/serving/import", json={"text": text, "config": raw,
                                                             "image": "evil image"}).status_code == 400
    raw["deployment"].update(dep)
    s = c.post("/api/sessions/byoc", json={"config": raw})
    assert s.status_code == 200, s.text
    saved = yaml.safe_load((tmp_path / "runs" / "byoc" / "workload.yaml").read_text())
    assert saved["deployment"]["tuned"]["title"].startswith("2-token") and saved["deployment"]["sitecustomize"]
    raw["deployment"]["env"] = {"AIP_STORAGE_URI": "x"}
    bad = c.post("/api/sessions/byoc", json={"config": raw})
    assert bad.status_code == 400 and "Serving container" in bad.json()["detail"]


def test_cli_import_tuned(tmp_path):
    from aiops_dilemma import cli
    text, marker = _candidate(tmp_path)
    (tmp_path / "best_candidate.py").write_text(text)
    wl = tmp_path / "workload.yaml"
    wl.write_text(config.to_yaml(_cfg()))
    cli.main(["import-tuned", "-c", str(wl), str(tmp_path / "best_candidate.py"), "--dry-run"])
    assert "args_mode: replace" not in wl.read_text()
    cli.main(["import-tuned", "-c", str(wl), str(tmp_path / "best_candidate.py"),
              "--image", "us-docker.pkg.dev/my-gcp-project/mirror/vllm-openai:v0.31.0"])
    cfg = config.load(str(wl))
    assert cfg["deployment"]["args_mode"] == "replace" and not cfg.get("_serving_errors") and not marker.exists()
    assert cfg["deployment"]["image"] == "us-docker.pkg.dev/my-gcp-project/mirror/vllm-openai:v0.31.0"


def test_warm_remote_image_requests_the_manifest(monkeypatch):
    monkeypatch.setattr(serving.time, "sleep", lambda s: None)
    calls = []

    class S:
        def __init__(self, codes):
            self.codes = list(codes)

        def get(self, url, headers=None, **k):
            calls.append((url, headers["Accept"]))
            return _Resp(self.codes.pop(0))
    serving.warm_remote_image("us-central1-docker.pkg.dev/p/dilemma-dockerhub/vllm/vllm-openai:v0.31.0", "p", S([200]))
    assert calls[-1][0] == "https://us-central1-docker.pkg.dev/v2/p/dilemma-dockerhub/vllm/vllm-openai/manifests/v0.31.0"
    assert "manifest.list.v2" in calls[-1][1]
    serving.warm_remote_image("us-central1-docker.pkg.dev/p/r/library/ubuntu@sha256:" + "a" * 64, "p", S([200]))
    assert calls[-1][0].endswith("/v2/p/r/library/ubuntu/manifests/sha256:" + "a" * 64)
    serving.warm_remote_image("us-central1-docker.pkg.dev/p/r/img", "p", S([200]))
    assert calls[-1][0].endswith("/v2/p/r/img/manifests/latest")
    with pytest.raises(SystemExit, match="Couldn't pull"):
        serving.warm_remote_image("us-central1-docker.pkg.dev/p/r/vllm/x:nope", "p", S([404, 404, 404]))


def test_load_test_uses_stream_raw_predict_and_the_served_model():
    from aiops_dilemma import loadtest
    st = {"dedicated_dns": "1.us-central1-2.prediction.vertexai.goog", "endpoint": "projects/p/locations/us-central1/endpoints/9"}
    url = loadtest.endpoint_url(st)
    assert url.endswith("/v1beta1/projects/p/locations/us-central1/endpoints/9/chat/completions")
    assert loadtest._model(url) == "default"
    url2 = loadtest.endpoint_url({**st, "api": "raw", "served_model": "google/gemma-4-26B-A4B-it"})
    assert url2 == "https://1.us-central1-2.prediction.vertexai.goog/v1/projects/p/locations/us-central1/endpoints/9:streamRawPredict"
    assert loadtest._model(url2) == "google/gemma-4-26B-A4B-it"


def test_mid_stream_server_errors_are_named():
    from aiops_dilemma import loadtest
    recs = [{"status": 200, "error": "server error mid-stream: internal error during generation"},
            {"status": 200}, {"status": 200, "prompt_tokens": 10}]
    assert loadtest._error_breakdown(recs) == {"server error mid-stream": 1, "stream ended without an answer": 1}
