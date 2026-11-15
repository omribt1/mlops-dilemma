"""Reference workload + reference benchmarks: the example YAML, catalog schema/loader, `dilemma reference-entry`,
recipe adaptation for unverified shapes."""
import json
from pathlib import Path

import pytest

from aiops_dilemma import catalog, cli, config, deploy, planner

ROOT = Path(__file__).resolve().parents[1]
REF_YAML = ROOT / "examples" / "reference-workload.yaml"


def _entry(**kw):
    e = {"accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 1, "machine_type": "a3-highgpu-1g",
         "model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "precision": "fp8", "workload": "reference-v1",
         "short_prompt_tps_per_chip": 20000, "measured_at": "2026-11-10", "image": "img@sha256:abc", "notes": "",
         "source": "measured", "avg_input_tokens": 5100, "avg_output_tokens": 480}
    e.update(kw)
    return e


def test_reference_yaml_is_the_fixed_workload():
    cfg = config.load(str(REF_YAML))
    pinned = next(x for x in catalog.SERVING_IMAGES if x["key"] == "pinned")
    assert cfg["deployment"]["image"] == pinned["image"] and "@sha256:" in pinned["image"]
    r, ref = cfg["requests"], catalog.REFERENCE_WORKLOAD
    assert (r["avg_input_tokens"], r["avg_output_tokens"], r["system_prompt_tokens"]) == (5000, 500, 1800)
    assert r["pct_over_32k"] == 0 and r["thinking_share"] == 0 and r["system_prompt_cache_hit"] == 0.9
    assert cfg["model"]["precision"] == ref["precision"] and cfg["model"]["model_garden_id"] == ref["model_garden_id"]
    assert cfg["model"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert cfg["loadtest"]["concurrency"] == [16, 32, 64] and cfg["loadtest"]["requests_per_level"] == 300
    assert cfg["constraints"]["latency_budget_s"] == 600 and cfg["prompts"]["synthetic"]["json_output"]
    assert "not a promise" in REF_YAML.read_text().lower() or "NOT a promise" in REF_YAML.read_text()


def test_reference_loader_validates(monkeypatch, tmp_path, capsys):
    assert catalog.reference_benchmarks([_entry()])[0]["short_prompt_tps_per_chip"] == 20000.0
    assert catalog.reference_benchmarks([_entry(source="estimated"), _entry(short_prompt_tps_per_chip=None),
                                         _entry(accelerator_type="NOPE")]) == []
    assert "skipped" in capsys.readouterr().out
    f = tmp_path / "refs.json"
    f.write_text(json.dumps([_entry(accelerator_type="NVIDIA_L4", accelerator_count=2)]))
    monkeypatch.setenv("DILEMMA_REFERENCE_BENCHMARKS", str(f))
    assert [e["accelerator_type"] for e in catalog.reference_benchmarks()][-1] == "NVIDIA_L4"


def _fake_run(rd: Path, peak_c=32):
    levels = [{"concurrency": c, "ok": 300, "errors": 0, "input_tps": 10000.0 * c / 16, "avg_in": 5100.0,
               "avg_out": 470.0, "lat_p95": 40.0 + c} for c in (16, 32, 64)]
    peak = next(s for s in levels if s["concurrency"] == peak_c)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "measured.json").write_text(json.dumps({
        "endpoint": {"accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 1, "machine_type": "g4-standard-48",
                     "precision": "fp8", "image": "pinned@sha256:3fbc"},
        "levels": levels, "peak": peak, "input_tps_per_gpu": peak["input_tps"], "within_budget": True,
        "serving": {"image_source": "deployment.image"}}))


def test_reference_entry_from_run(tmp_path, capsys):
    raw = REF_YAML.read_text().replace("name: ref-rtx-pro-6000", "name: ref-rtx")
    rd = tmp_path / "runs" / "ref-rtx"
    _fake_run(rd)
    (rd / "workload.yaml").write_text(raw)
    cli.main(["reference-entry", "-c", str(rd / "workload.yaml")])
    out = capsys.readouterr()
    e = json.loads(out.out)
    assert e["short_prompt_tps_per_chip"] == 20000 and e["workload"] == "reference-v1" and e["source"] == "measured"
    assert e["accelerator_type"] == "NVIDIA_RTX_PRO_6000" and e["image"] == "pinned@sha256:3fbc"
    assert e["avg_output_tokens"] == 470 and e["peak_concurrency"] == 32
    assert "warning" not in out.err
    _fake_run(rd, peak_c=64)
    cfg = config.load(str(rd / "workload.yaml"))
    cfg["model"]["precision"] = "bf16"
    _, warns = planner.reference_entry(cfg, rd)
    assert any("highest level" in w for w in warns) and any("precision" in w for w in warns)


def test_adapt_recipe_sets_tensor_parallel_degree():
    rec = {"image": "x/pytorch-vllm-serve:1", "args": ["python", "-m", "vllm.entrypoints.api_server",
                                                     "--model=gs://bucket/gemma", "--tensor-parallel-size=1"],
           "env": {}, "predict_route": "/generate", "health_route": "/ping", "port": 8080}
    out = deploy.adapt_recipe(rec, 2)
    assert "--tensor-parallel-size=2" in out["args"] and "--model=gs://bucket/gemma" in out["args"]
    assert rec["args"][-1] == "--tensor-parallel-size=1"          # original untouched
    assert "--tensor-parallel-size=4" in deploy.adapt_recipe({**rec, "args": ["serve"]}, 4)["args"]


def test_single_system_prompt_uses_the_full_size(tmp_path):
    from aiops_dilemma import prompts
    cfg = config.load(str(REF_YAML))
    cfg["prompts"]["synthetic"].update(num_system_prompts=1, count=3)
    cfg["loadtest"]["concurrency"] = []
    prompts.synthetic(cfg, tmp_path / "p.jsonl")
    rows = [json.loads(x) for x in open(tmp_path / "p.jsonl")]
    assert {r["system_tokens"] for r in rows} == {1800}
    assert "about 350 words" in rows[0]["messages"][0]["content"]


# ---- estimate tiers (planner.reference_throughput) -----------------------------------------------------------------
def _ref_cfg(**model):
    cfg = config.load(str(REF_YAML))
    cfg["model"].update(model)
    return cfg


def test_tier_reference_exact_match_uses_measured_rate():
    cfg = _ref_cfg()
    # the run's measured averages = the reference workload -> no workload adjustment
    r = planner.reference_throughput(cfg, "NVIDIA_H100_80GB", 1, [_entry(avg_input_tokens=5000, avg_output_tokens=500)])
    assert r["basis"] == "reference" and r["label"] == "Reference benchmark" and r["per_chip"] == 20000
    assert "Measured on 1x H100 80GB with the reference workload" in r["note"] and "2026-11-10" in r["note"]
    assert "run a benchmark with your prompts" in r["note"] and "Adjusted" not in r["note"]


def test_tier_reference_adjusts_for_longer_outputs():
    cfg = _ref_cfg()
    cfg["requests"].update(avg_output_tokens=500, thinking_share=0.3, avg_thinking_tokens=1500)   # +450 out/request
    r = planner.reference_throughput(cfg, "NVIDIA_H100_80GB", 1, [_entry(avg_input_tokens=5000, avg_output_tokens=500)])
    assert r["basis"] == "reference" and r["per_chip"] < 20000 * 0.9 and r["workload_scale"] < 0.9
    assert "Adjusted to your output length" in r["note"]


def test_tier_scaled_from_another_model_on_same_gpu():
    # a similar mixture-of-experts model: within the scaling guardrail
    cfg = _ref_cfg(model_garden_id="qwen/qwen3@qwen3-30b-a3b", hf_id="Qwen/qwen3-30b-a3b",
                   params_b=30, active_params_b=3)
    r = planner.reference_throughput(cfg, "NVIDIA_H100_80GB", 1, [_entry(avg_input_tokens=5000, avg_output_tokens=500)])
    assert r["basis"] == "scaled" and r["label"] == "Scaled from a reference benchmark"
    assert 1 / 3 <= r["model_scale"] <= 3 and abs(r["per_chip"] - 20000 * r["model_scale"] * r["workload_scale"]) <= 3
    assert "another model" in r["note"]
    # another count of the same model is also scaled (tensor-parallel efficiency from the formula)
    r2 = planner.reference_throughput(_ref_cfg(), "NVIDIA_H100_80GB", 2, [_entry()])
    assert r2["basis"] == "scaled" and "another GPU count" in r2["note"]


def test_tier_spec_when_no_reference_for_the_gpu():
    cfg = _ref_cfg()
    r = planner.reference_throughput(cfg, "NVIDIA_L4", 2, [_entry()])
    assert r["basis"] == "spec" and r["label"] == "Rough spec estimate"
    assert r["per_chip"] == planner.estimate_tps_per_chip(cfg, "NVIDIA_L4", 2) and "No reference benchmark for L4" in r["note"]


def test_planner_candidates_and_session_carry_the_tier(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from aiops_dilemma import server
    monkeypatch.setattr(catalog, "REFERENCE_BENCHMARKS", [_entry(accelerator_type="NVIDIA_RTX_PRO_6000",
                                                                 machine_type="g4-standard-48")])
    cfg = _ref_cfg()
    cands = {c["accelerator_type"]: c for c in planner.candidates(cfg, with_prices=False)}
    assert cands["NVIDIA_RTX_PRO_6000"]["est_basis"] == "reference"
    assert cands["NVIDIA_L4"]["est_basis"] == "spec" and cands["NVIDIA_L4"]["est_basis_label"] == "Rough spec estimate"
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    raw = {k: v for k, v in cfg.items() if not k.startswith("_")}
    d = TestClient(server.app).post("/api/sessions/refsess/estimate", json={"config": raw}).json()
    tp = d["throughput"]
    assert tp["source"] == "estimated" and tp["basis"] == "reference" and "reference benchmark" in tp["label"]
    assert "Estimated throughput, not measured" in d["report_md"] and "Measured on 1x RTX PRO 6000" in d["report_md"]
    html = (tmp_path / "runs" / "refsess" / "dashboard.html").read_text()
    assert 'class="disclaimer"' in html and "run a benchmark with your own prompts" in html


def test_measured_report_has_no_disclaimer(tmp_path):
    from aiops_dilemma import report
    assert report.disclaimer({"levels": [{"concurrency": 1}], "input_tps_per_gpu": 1}) is None
    assert "several times" in report.disclaimer({"estimated": True, "levels": []})


def test_scaling_guardrail_marks_very_different_models_rough():
    """A dense 27B model vs the reference MoE (4B active): the formula ratio is beyond 3x, so 'Rough', not 'Scaled'."""
    cfg = _ref_cfg(model_garden_id="google/gemma3@gemma-3-27b-it", hf_id="google/gemma-3-27b-it",
                   params_b=27, active_params_b=27)
    r = planner.reference_throughput(cfg, "NVIDIA_H100_80GB", 1, [_entry(avg_input_tokens=5000, avg_output_tokens=500)])
    assert r["model_scale"] < 1 / planner.MAX_MODEL_SCALE
    assert r["basis"] == "spec" and "too different" in r["note"]
