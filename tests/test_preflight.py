"""Pre-flight check (loadtest.preflight), serving-image provenance (endpoint.json -> measured.json -> report) and the
serving image catalog. No GCP: httpx is mocked with MockTransport, or the low-level `_preflight_stream` is stubbed."""
import json
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import analyze, catalog, config, loadtest, planner, report, server

ROOT = Path(__file__).resolve().parents[1]
URL = "https://fake.example/v1beta1/projects/my-gcp-project/locations/us-central1/endpoints/1/chat/completions"
IMG = ("us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve"
       "@sha256:3fbc0e08c46e7736145c9effbcc8839682c80b15b413335146f4428fc3523c8a")


def _cfg(**req):
    return config.normalize({"name": "pf", "project": "my-gcp-project",
                             "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "precision": "fp8"},
                             "traffic": {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500},
                             "requests": {"avg_input_tokens": 7000, "avg_output_tokens": 950, **req}})


@pytest.fixture(autouse=True)
def fake_auth(monkeypatch):
    monkeypatch.setattr(loadtest, "Auth", lambda: type("A", (), {"token": lambda self: "t"})())


def _sse(n_tokens: int, finish: str = "length", prompt_tokens: int = 20) -> list[bytes]:
    lines = [b'data: {"choices":[{"delta":{"content":"1"},"finish_reason":null}]}\n\n'] * n_tokens
    lines.append(('data: {"choices":[{"delta":{},"finish_reason":"%s"}]}\n\n' % finish).encode())
    lines.append(('data: {"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d}}\n\n'
                  % (prompt_tokens, n_tokens)).encode())
    lines.append(b"data: [DONE]\n\n")
    return lines


class _Stream(httpx.SyncByteStream):
    def __init__(self, chunks, cut=False):
        self.chunks, self.cut = chunks, cut

    def __iter__(self):
        yield from self.chunks
        if self.cut:
            raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def _transport(monkeypatch, handler):
    seen = []

    def h(request):
        body = json.loads(request.content)
        seen.append(body)
        return handler(body)
    monkeypatch.setattr(loadtest, "_http_client", lambda t: httpx.Client(transport=httpx.MockTransport(h)))
    return seen


def test_sizing_from_budget_and_workload(tmp_path):
    cfg = _cfg()
    assert loadtest.preflight_target_s(cfg) == 90            # 600 s budget -> capped at 90 s
    cfg["constraints"]["latency_budget_s"] = 60
    assert loadtest.preflight_target_s(cfg) == 45            # floor 45 s
    cfg["constraints"]["latency_budget_s"] = 120
    assert loadtest.preflight_target_s(cfg) == 60
    assert loadtest.preflight_output_tokens(cfg) == 60 * loadtest.PREFLIGHT_DECODE_TPS
    cfg["model"]["max_model_len"] = 4096                       # output capped by the context
    assert loadtest.preflight_output_tokens(cfg) == 4096 - loadtest.CONTEXT_MARGIN
    # short-only workload: the largest prompt in prompts.jsonl (or 2x the average)
    short = _cfg()
    assert loadtest.preflight_prompt_tokens(short)[0] == 14000
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps({"prompt_tokens": n}) + "\n" for n in (3000, 21000)))
    assert loadtest.preflight_prompt_tokens(short, tmp_path)[0] == 21000
    # prompts over 32K -> ~100K; skipped when the context is smaller
    long_cfg = _cfg(pct_over_32k=5)
    assert loadtest.preflight_prompt_tokens(long_cfg)[0] == 100_000
    long_cfg["model"]["max_model_len"] = 65536
    n, why = loadtest.preflight_prompt_tokens(long_cfg)
    assert n is None and "context" in why


def test_preflight_passes_with_real_stream_parsing(monkeypatch, tmp_path):
    cfg = _cfg()
    seen = _transport(monkeypatch, lambda b: httpx.Response(
        200, stream=_Stream(_sse(50 if b.get("min_tokens") else 1, "length" if b.get("min_tokens") else "stop",
                                 20 if b.get("min_tokens") else 13000))))
    events = []
    res = loadtest.preflight(URL, cfg, tmp_path, events.append, image=IMG)
    assert res["passed"] and res["image"] == IMG
    long_out = [b for b in seen if b.get("min_tokens")]
    assert long_out[0]["ignore_eos"] is True and long_out[0]["max_tokens"] == long_out[0]["min_tokens"] == 13500
    assert len(long_out) == 2 and long_out[1]["max_tokens"] == 4 * 13500   # too fast: one longer retry
    lp = [b for b in seen if not b.get("min_tokens")][0]
    assert lp["max_tokens"] == 16 and len(lp["messages"][0]["content"]) > 40000
    names = [c["name"] for c in res["checks"]]
    assert names == ["long_output", "long_output", "long_prompt"] and res["checks"][-1]["prompt_tokens"] == 13000
    assert "too fast" in res["message"]          # inconclusive is reported, not hidden
    assert [e["event"] for e in events] == ["preflight_start", "preflight_done"]
    assert json.loads((tmp_path / "preflight.json").read_text())["passed"] is True


def test_preflight_conclusive_pass_no_retry(monkeypatch, tmp_path):
    calls = []

    def fake(url, headers, body, timeout_s):
        calls.append(body)
        lo = bool(body.get("min_tokens"))
        return {"status": 200, "duration_s": 62.0 if lo else 9.0, "ttft_s": 0.2, "tokens": body["max_tokens"],
                "prompt_tokens": 20 if lo else 99000, "finish_reason": "length" if lo else "stop", "done": True,
                "error": None}
    monkeypatch.setattr(loadtest, "_preflight_stream", fake)
    res = loadtest.preflight(URL, _cfg(pct_over_32k=3), tmp_path)
    assert res["passed"] and len(calls) == 2 and res["checks"][0]["conclusive"]
    assert res["prompt_tokens"] == 100_000 and "99,000-token prompt completed" in res["message"]


def test_preflight_cut_mid_stream_fails_with_precise_message(monkeypatch, tmp_path):
    _transport(monkeypatch, lambda b: httpx.Response(200, stream=_Stream(_sse(30)[:30], cut=True)))
    events = []
    with pytest.raises(SystemExit) as ei:
        loadtest.preflight(URL, _cfg(), tmp_path, events.append, image=IMG)
    msg = str(ei.value)
    assert msg.startswith("Pre-flight failed: the endpoint cut a streaming request after ")
    assert "(30 tokens)" in msg and IMG in msg and "pinned image" in msg and "deployment.image" in msg
    saved = json.loads((tmp_path / "preflight.json").read_text())
    assert saved["passed"] is False and saved["checks"][0]["passed"] is False
    assert events[-1]["event"] == "preflight_done" and events[-1]["result"]["passed"] is False


def test_preflight_stream_without_end_counts_as_cut(monkeypatch, tmp_path):
    _transport(monkeypatch, lambda b: httpx.Response(200, stream=_Stream(_sse(10)[:10])))   # no finish, no [DONE]
    with pytest.raises(SystemExit, match="cut a streaming request"):
        loadtest.preflight(URL, _cfg(), tmp_path)


def test_preflight_503_fails(monkeypatch, tmp_path):
    _transport(monkeypatch, lambda b: httpx.Response(503, text="no healthy upstream"))
    with pytest.raises(SystemExit, match="503 no healthy upstream") as ei:
        loadtest.preflight(URL, _cfg(), tmp_path)
    assert "the Model Garden verified recipe image" in str(ei.value)   # no image recorded


def test_preflight_long_prompt_cut_fails(monkeypatch, tmp_path):
    def h(b):
        if b.get("min_tokens"):
            return httpx.Response(200, stream=_Stream(_sse(5)))
        return httpx.Response(200, stream=_Stream([], cut=True))
    _transport(monkeypatch, h)
    with pytest.raises(SystemExit, match="cut a long-prompt request"):
        loadtest.preflight(URL, _cfg(pct_over_32k=2), tmp_path)


def test_run_skips_preflight_when_off_or_tagged(monkeypatch, tmp_path):
    (tmp_path / "endpoint.json").write_text(json.dumps({"endpoint": "projects/p/locations/l/endpoints/1",
                                                        "dedicated_dns": "fake.example", "image": IMG}))
    (tmp_path / "prompts.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    called = []
    monkeypatch.setattr(loadtest, "wait_ready", lambda *a, **k: 0)
    monkeypatch.setattr(loadtest, "preflight", lambda *a, **k: called.append(k.get("image")))

    async def fake_level(url, prompts, start, n, c, kw, cb):
        return [{"status": 200, "prompt_tokens": 10, "completion_tokens": 5, "latency": 1.0, "start": 0.0}]
    monkeypatch.setattr(loadtest, "_level", fake_level)
    cfg = _cfg()
    cfg["loadtest"].update(concurrency=[1], requests_per_level=1)
    loadtest.run(cfg, tmp_path)
    assert called == [IMG]                       # on by default, told which image served
    loadtest.run(cfg, tmp_path, tag="short")
    assert len(called) == 1                      # tagged runs reuse the main run's check
    cfg["loadtest"]["preflight"] = False
    loadtest.run(cfg, tmp_path)
    assert len(called) == 1


def test_provenance_endpoint_to_measured_to_report(tmp_path):
    cfg = config.normalize(yaml.safe_load((ROOT / "tests" / "fixtures" / "startup-v1.yaml").read_text()))
    ch = planner.plan(cfg)["chosen"]
    ep = {"endpoint": "projects/my-gcp-project/locations/us-central1/endpoints/1", "dedicated_dns": "x",
          **{k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
          "image": IMG, "args": ["python", "-m", "vllm.entrypoints.openai.api_server", "--host=0.0.0.0", "--port=8080",
                                 "--max-model-len=32768", "--enable-prefix-caching"], "inference_timeout_s": 600}
    (tmp_path / "endpoint.json").write_text(json.dumps(ep))
    (tmp_path / "load").mkdir()
    recs = [{"status": 200, "prompt_tokens": 5000, "completion_tokens": 500, "latency": 3.0, "start": i * 0.5}
            for i in range(20)]
    (tmp_path / "load" / "c8.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (tmp_path / "preflight.json").write_text(json.dumps({"ran": True, "passed": True, "duration_s": 75.0,
                                                         "message": "a 61 s stream (9,000 tokens) completed"}))
    meas = analyze.analyze(cfg, tmp_path)
    assert meas["serving"] == {"image": IMG, "args": ep["args"], "inference_timeout_s": 600}
    assert meas["preflight"]["passed"] is True
    saved = json.loads((tmp_path / "measured.json").read_text())
    assert saved["serving"]["image"] == IMG
    md = report.render(cfg, {"candidates": []}, meas, ch["prices"])
    assert "Measured with `pytorch-vllm-serve:gemma4 (sha256:3fbc0e08…)`" in md
    assert "vLLM args `--max-model-len=32768 --enable-prefix-caching`" in md and "--host" not in md.split("Measured with")[1].split("\n")[0]
    assert "inference timeout 600 s" in md and "Pre-flight check passed" in md


def test_catalog_exposes_serving_images(monkeypatch, tmp_path):
    pinned = next(x for x in catalog.SERVING_IMAGES if x["key"] == "pinned")
    assert pinned["image"] == IMG and {"label", "tested", "notes"} <= set(pinned)
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(server, "_build_catalog", lambda p, r, refresh=False: {"format": server.CATALOG_FORMAT,
                                                                               "cached_at": 9e12})
    server._CATALOG_MEM.clear()
    try:
        data = TestClient(server.app).get("/api/catalog?project=pf-test").json()
    finally:
        server._CATALOG_MEM.clear()
    assert data["serving_images"] == catalog.SERVING_IMAGES


def test_cli_preflight_reuses_endpoint_json(monkeypatch, tmp_path, capsys):
    from aiops_dilemma import cli
    rd = tmp_path / "rd"
    rd.mkdir()
    cfg_path = tmp_path / "w.yaml"
    cfg_path.write_text(yaml.safe_dump({"name": "pf", "project": "my-gcp-project",
                                        "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it"}}))
    with pytest.raises(SystemExit, match="endpoint.json not found"):
        cli.main(["preflight", "-c", str(cfg_path), "--run-dir", str(rd), "--no-wait"])
    (rd / "endpoint.json").write_text(json.dumps({"endpoint": "projects/my-gcp-project/locations/us-central1/endpoints/1",
                                                  "dedicated_dns": "fake.example", "image": IMG}))
    urls = []

    def fake(url, headers, body, timeout_s):
        urls.append(url)
        return {"status": 200, "duration_s": 50.0, "ttft_s": 0.2, "tokens": body["max_tokens"], "prompt_tokens": 9,
                "finish_reason": "stop", "done": True, "error": None}
    monkeypatch.setattr(loadtest, "_preflight_stream", fake)
    cli.main(["preflight", "-c", str(cfg_path), "--run-dir", str(rd), "--no-wait"])
    assert urls[0] == "https://fake.example/v1beta1/projects/my-gcp-project/locations/us-central1/endpoints/1/chat/completions"
    assert '"passed": true' in capsys.readouterr().out and (rd / "preflight.json").exists()


VERIFIED = {"image": "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20260713_0916_RC01",
            "args": ["python", "-m", "vllm.entrypoints.openai.api_server", "--host=0.0.0.0", "--port=8080",
                     "--max-model-len=131072", "--enable-prefix-caching"],
            "env": {"MODEL_ID": "x"}, "predict_route": "/generate", "health_route": "/ping", "port": 8080,
            "accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 1}


def test_image_override_keeps_the_verified_recipe_args():
    from aiops_dilemma import deploy
    cfg = _cfg()
    ch = {"accelerator_count": 1}
    image, args, _, routes = deploy._args(cfg, ch, VERIFIED)
    assert image == VERIFIED["image"] and "--enable-prefix-caching" in args and routes["health"] == "/ping"
    assert not any(a.startswith("--max-model-len=131072") for a in args)     # we control the context length
    cfg["deployment"]["image"] = IMG
    image2, args2, _, _ = deploy._args(cfg, ch, VERIFIED)
    assert image2 == IMG and args2 == args
    assert deploy.valid_image_ref(IMG) and deploy.valid_image_ref("gcr.io/my-project/vllm@sha256:" + "a" * 64)
    assert deploy.valid_image_ref("docker.io/vllm/vllm-openai:latest")     # Docker Hub: pulled through Artifact Registry
    for bad in ("pytorch-vllm-serve:gemma4", "vllm/vllm-openai:latest", "docker.io/Vllm/x", "us-docker.pkg.dev/p/r", ""):
        assert not deploy.valid_image_ref(bad), bad


def test_deploy_records_image_and_args_in_endpoint_json(monkeypatch, tmp_path):
    import sys
    import types
    from aiops_dilemma import deploy, modelinfo
    fake_vx = types.ModuleType("vertexai")
    fake_vx.init = lambda **k: None
    fake_vx.model_garden = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vertexai", fake_vx)
    monkeypatch.setattr(modelinfo, "verified_options", lambda *a: [VERIFIED])
    seen = {}

    def fake_create(cfg, run_dir, name, secs, common, container, ch):
        seen.update(container)
        return {"endpoint": "projects/my-gcp-project/locations/us-central1/endpoints/1", "dedicated_dns": "x",
                "inference_timeout_s": 600}
    monkeypatch.setattr(deploy, "_deploy_with_timeout", fake_create)
    cfg = _cfg()
    cfg["deployment"]["image"] = IMG
    plan = {"chosen": {"machine_type": "g4-standard-48", "accelerator_type": "NVIDIA_RTX_PRO_6000",
                       "accelerator_count": 1}}
    state = deploy.deploy(cfg, plan, tmp_path)
    saved = json.loads((tmp_path / "endpoint.json").read_text())
    assert saved["image"] == IMG == seen["serving_container_image_uri"] and saved["image_source"] == "deployment.image"
    assert saved["args"] == seen["serving_container_args"] and "--enable-prefix-caching" in saved["args"]
    assert saved["inference_timeout_s"] == 600 and state["image"] == IMG
    assert analyze.serving_provenance(saved)["image"] == IMG
    cfg["deployment"]["image"] = "not an image"
    with pytest.raises(SystemExit, match="doesn't look like"):
        deploy.deploy(cfg, plan, tmp_path)


def test_failure_message_names_the_pinned_image(monkeypatch, tmp_path):
    monkeypatch.setattr(loadtest, "_preflight_stream", lambda *a: {"status": 503, "duration_s": 0.3, "tokens": 0,
                                                                  "error": "no healthy upstream"})
    with pytest.raises(SystemExit) as ei:
        loadtest.preflight(URL, _cfg(), tmp_path, image=VERIFIED["image"])
    pinned = next(x for x in catalog.SERVING_IMAGES if x["key"] == "pinned")
    assert "choose the pinned image (pytorch-vllm-serve:gemma4 (sha256:3fbc0e08" in str(ei.value)
    assert (f"tested {pinned['tested']}" in str(ei.value)) == bool(pinned["tested"])
    with pytest.raises(SystemExit) as ei:     # the pinned image itself failing is a different problem
        loadtest.preflight(URL, _cfg(), tmp_path, image=IMG)
    assert "not the known image problem" in str(ei.value)


def test_pinned_image_is_gpu_only_and_deploy_refuses_it_on_tpu(tmp_path):
    from aiops_dilemma import catalog, deploy
    pinned = next(x for x in catalog.SERVING_IMAGES if x["key"] == "pinned")
    assert pinned["accelerator_kinds"] == ["gpu"] and "NVIDIA_RTX_PRO_6000" in pinned["tested_accelerators"]
    assert deploy.image_supports(pinned["image"], "NVIDIA_L4") is None          # a GPU, even if untested
    assert "can't run on TPU_V6E" in deploy.image_supports(pinned["image"], "TPU_V6E")
    assert deploy.image_supports(None, "TPU_V6E") is None                         # verified recipe: fine
    cfg = {"name": "x", "region": "us-central1", "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it"},
           "deployment": {"image": pinned["image"]}}
    plan = {"chosen": {"accelerator_type": "TPU_V6E", "accelerator_count": 4, "machine_type": "ct6e-standard-4t"}}
    with pytest.raises(SystemExit, match="GPU build"):
        deploy.deploy(cfg, plan, tmp_path)
    assert not (tmp_path / "endpoint.json").exists()                               # nothing was created
