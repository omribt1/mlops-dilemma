"""Tests for smart ingestion, dynamic model recommendation, GPU+TPU planner, Gemini resilience, and FastAPI UI server."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from aiops_dilemma import config, costmodel, ingest, planner, prompts, recommender, report
from aiops_dilemma.server import app


def test_ingest_traffic_and_prompts_and_system_prompts(tmp_path: Path):
    # 1. Hourly traffic CSV (24 rows -> auto-tiled to 168)
    csv_lines = ["input_tokens_per_sec"] + [str(10000 + i * 2000) for i in range(24)]
    traffic_bytes = "\n".join(csv_lines).encode("utf-8")

    # 2. Specific system prompts (.txt separated by ---)
    sys_bytes = (
        "You are an invoice extraction assistant. Return strict JSON.\n---\n"
        "You are a purchase order auditor. Flag anomalies."
    ).encode("utf-8")

    # 3. Specific user prompts with Short (<=32K), Medium (32K-128K), and Long (>128K) token counts
    jsonl_rows = [
        {"id": "1", "messages": [{"role": "user", "content": "short 1"}], "input_tokens": 4000, "output_tokens": 300},
        {"id": "2", "messages": [{"role": "user", "content": "short 2"}], "input_tokens": 6000, "output_tokens": 400},
        {"id": "3", "messages": [{"role": "user", "content": "med 1"}], "input_tokens": 64000, "output_tokens": 500},
        {"id": "4", "messages": [{"role": "user", "content": "long 1"}], "input_tokens": 160000, "output_tokens": 600},
    ]
    prompts_bytes = "\n".join(json.dumps(r) for r in jsonl_rows).encode("utf-8")

    res = ingest.ingest_files(
        [
            ("traffic.csv", traffic_bytes, "traffic"),
            ("sys.txt", sys_bytes, "system_prompts"),
            ("prod_logs.jsonl", prompts_bytes, "prompts"),
        ],
        tmp_path,
    )

    assert len(res["traffic"]["profile"]) == 168
    assert res["traffic"]["peak_input_tps"] == 56000
    assert res["prompts"]["num_system_prompts"] == 2
    assert res["requests"]["pct_short"] == 50.0
    assert res["requests"]["pct_medium"] == 25.0
    assert res["requests"]["pct_long"] == 25.0
    assert res["requests"]["pct_over_32k"] == 50.0
    assert res["requests"]["pct_over_128k"] == 25.0


def test_recommender_and_gpu_tpu_planner():
    res = recommender.recommend(
        use_case={"task_type": "extraction", "priority": "cost_efficiency", "json_output": True},
        requests_cfg={"avg_input_tokens": 6000, "avg_output_tokens": 400, "pct_over_32k": 2.0, "pct_over_128k": 0.5, "thinking_share": 0.1},
        traffic_cfg={"peak_input_tps": 120000},
    )
    recs = res["recommendations"]
    assert len(recs) >= 5
    assert recs[0]["match_score"] >= recs[-1]["match_score"]
    # The long-prompt tip is shown once at response level, never repeated per model.
    assert len(res["tips"]) == 1 and "Gemini" in res["tips"][0]
    assert not any("Gemini" in " ".join(r["reasons"]) for r in recs)

    cfg = config.normalize({
        "name": "test-plan",
        "project": "my-gcp-project",
        "region": "us-central1",
        "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": "google/gemma-4-26b-a4b-it", "precision": "fp8"},
    })
    p_all = planner.plan(cfg, accelerator_filter="all")
    kinds = {c["kind"] for c in p_all["candidates"]}
    assert "gpu" in kinds
    assert "tpu" in kinds
    assert p_all["chosen"]["est_input_tps_per_gpu"] > 1000

    p_tpu = planner.plan(cfg, accelerator_filter="tpu")
    assert all(c["kind"] == "tpu" for c in p_tpu["candidates"])


def test_gemini_hybrid_routing_and_resilience(tmp_path: Path):
    cfg = config.normalize({
        "name": "test-hybrid",
        "project": "my-gcp-project",
        "region": "us-central1",
        "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": "google/gemma-4-26b-a4b-it", "precision": "fp8"},
        "traffic": {"peak_input_tps": 120000, "average_input_tps": 35000, "floor_input_tps": 3500},
        "requests": {"avg_input_tokens": 7000, "avg_output_tokens": 500, "pct_medium": 1.5, "pct_long": 0.5},
        "maas": {"input_per_m": 0.15, "output_per_m": 0.60, "cached_per_m": 0.015, "max_rps": 8},
    })
    p = planner.plan(cfg)
    ch = p["chosen"]
    prices = ch["prices"]
    summary = costmodel.resilience_and_hybrid_summary(cfg, 11000.0, prices, "cud_3y", ch["accelerator_count"])
    assert len(summary["incidents"]) == 3
    assert summary["zone_extra_cost"] > 0
    assert summary["breakeven_peak_outage_hours"] > 0

    ro = summary["routing_options"]
    assert ro["route_long_gemini_flash_lite"]["gpus"] <= ro["all_on_fleet"]["gpus"]

    # Also test custom system prompts in synthetic prompt builder
    cfg["prompts"]["custom_system_prompts"] = ["Custom system prompt alpha for testing."]
    cfg["prompts"]["synthetic"]["count"] = 12
    built = prompts.build(cfg, tmp_path)
    first = json.loads(built.read_text().splitlines()[0])
    assert first["messages"][0]["content"] == "Custom system prompt alpha for testing."


def test_fastapi_server_endpoints(tmp_path, monkeypatch):
    from aiops_dilemma import server
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    server.build_example_session(tmp_path / "runs" / "blog-example")
    client = TestClient(app)

    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "mlops-dilemma" in r_idx.text

    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat_data = r_cat.json()
    assert "TPU_V6E" in cat_data["accelerators"]
    assert "NVIDIA_RTX_PRO_6000" in cat_data["accelerators"]
    assert "gemini_flash_lite" in cat_data["gemini"]

    r_sess = client.get("/api/sessions")
    assert r_sess.status_code == 200
    names = [s["name"] for s in r_sess.json()["sessions"]]
    assert "blog-example" in names

    r_blog = client.get("/api/sessions/blog-example")
    assert r_blog.status_code == 200
    blog_payload = r_blog.json()
    assert blog_payload["dashboard_data"] is not None
    assert "resilienceSummary" in blog_payload["dashboard_data"]

    # Estimate endpoint on a test session
    r_est = client.post("/api/sessions/pytest-ui-session/estimate", json={
        "config": {
            "name": "pytest-ui-session",
            "project": "my-gcp-project",
            "region": "us-central1",
            "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": "google/gemma-4-26b-a4b-it", "precision": "fp8"},
            "traffic": {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500},
            "requests": {"avg_input_tokens": 6000, "avg_output_tokens": 500, "pct_medium": 1.2, "pct_long": 0.3},
        },
        "tps_per_gpu": 10500,
    })
    assert r_est.status_code == 200
    est_data = r_est.json()
    assert est_data["measured"]["input_tps_per_gpu"] == 10500
    assert est_data["measured"]["levels"] == [] and est_data["throughput"]["source"] == "override"
    assert "Google Gemini on Agent Platform" in est_data["report_md"]
