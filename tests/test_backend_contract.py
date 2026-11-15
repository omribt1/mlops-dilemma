"""Backend <-> frontend contract (docs/ui-design-plan.md) and regression tests for the UI review fixes."""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import config, costmodel, planner, pricing, recommender, report, server

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "aiops_dilemma" / "assets"
NODE = shutil.which("node")

GEMMA = {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": "google/gemma-4-26b-a4b-it", "precision": "fp8"}
BASE_CFG = {
    "project": "my-gcp-project",
    "region": "us-central1",
    "model": GEMMA,
    "traffic": {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500, "weekend_factor": 0.08},
    "requests": {"avg_input_tokens": 7000, "avg_output_tokens": 950, "system_prompt_tokens": 500,
                 "system_prompt_cache_hit": 0.9},
    "maas": {"input_per_m": 0.15, "output_per_m": 0.60, "cached_per_m": 0.015, "max_rps": 5},
}


def _cfg(**over) -> dict:
    raw = json.loads(json.dumps(BASE_CFG))
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) and isinstance(raw.get(k), dict) else v
    return raw


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with sessions in a temp runs/ dir and a freshly generated example session."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    server.build_example_session(runs / "blog-example")
    return TestClient(server.app)


# 1. Redundancy multiplier ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (0, 1.0), ("", 1.0), (None, 1.0), ("abc", 1.0), (float("nan"), 1.0), (-2, 1.0), (0.99, 1.0),
    (1, 1.0), ("1.5", 1.5), (1.4, 1.5), (1.8, 2.0), (2, 2.0), (3, 2.0),
])
def test_normalize_resilience(raw, expected):
    assert config.normalize_resilience(raw) == expected
    cfg = config.normalize(_cfg(constraints={"resilience_multiplier": raw}))
    assert cfg["constraints"]["resilience_multiplier"] == expected


@pytest.mark.parametrize("bad", [0, ""])
def test_resilience_round_trip_save_and_load(client, bad):
    r = client.post("/api/sessions/res-test", json={"config": _cfg(constraints={"resilience_multiplier": bad})})
    assert r.status_code == 200
    assert r.json()["config"]["constraints"]["resilience_multiplier"] == 1.0
    on_disk = yaml.safe_load((server.RUNS_DIR / "res-test" / "workload.yaml").read_text())
    assert on_disk["constraints"]["resilience_multiplier"] == 1.0

    # A file that already holds 0 (as the old UI wrote) loads as 1 and still plans GPUs.
    raw = dict(on_disk, constraints={**on_disk["constraints"], "resilience_multiplier": 0})
    (server.RUNS_DIR / "res-test" / "workload.yaml").write_text(yaml.safe_dump(raw))
    d = client.get("/api/sessions/res-test").json()
    assert d["config"]["constraints"]["resilience_multiplier"] == 1.0
    assert "resilience_multiplier: 1.0" in d["yaml_text"]
    assert d["dashboard_data"]["defaults"]["resilience"] == 1.0
    fleet = d["dashboard_data"]["resilienceSummary"]["routing_options"]["all_on_fleet"]
    assert fleet["gpus"] > 0
    tiers = {t: v for t, v in d["dashboard_data"]["prices"].items()}
    assert len(set(tiers.values())) > 1


def test_costmodel_guards_resilience():
    cfg = config.normalize(_cfg())
    cfg["constraints"]["resilience_multiplier"] = 0  # bypass normalize on purpose
    prices = {"vertex_cud_3y": 2.52}
    r = costmodel.scenario(cfg, 9000, prices, "vertex_cud_3y", 1)
    assert r["gpus"] > 0
    assert r["total"] > r["fleet"].maas_cost


# 2. Instant estimate never fabricates a load test ------------------------------------------------------------------

def test_estimate_writes_no_fake_levels(client):
    r = client.post("/api/sessions/est/estimate", json={"config": _cfg()})
    assert r.status_code == 200
    d = r.json()
    m = d["measured"]
    assert m["estimated"] is True and m["levels"] == [] and m["peak"] is None
    assert m["throughput_source"] == "estimated"
    assert m["input_tps_per_gpu"] == d["throughput"]["per_chip"]
    assert "Throughput estimated from accelerator specs (not measured)" in d["report_md"]
    assert "| Concurrency |" not in d["report_md"]
    html = (server.RUNS_DIR / "est" / "dashboard.html").read_text()
    assert '"measuredLevels":[]' in html


def test_estimate_keeps_real_load_test(client):
    client.post("/api/sessions/real", json={"config": _cfg()})
    ch = planner.plan(config.normalize(_cfg()))["chosen"]
    level = {"concurrency": 32, "ok": 300, "errors": 0, "error_rate": 0.0, "rps": 1.5, "input_tps": 10000.0,
             "output_tps": 1300.0, "avg_in": 7000, "avg_out": 950, "lat_p50": 20.0, "lat_p95": 30.0}
    real = {"endpoint": {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
            "levels": [level], "peak": level, "input_tps_per_gpu": 10000.0, "measured_out_ratio": 0.13,
            "within_budget": True}
    (server.RUNS_DIR / "real" / "measured.json").write_text(json.dumps(real))
    d = client.post("/api/sessions/real/estimate", json={}).json()
    assert d["measured"]["levels"] == [level]
    assert d["throughput"] == {"per_chip": 10000.0, "source": "measured", "label": report.throughput_label(10000.0, "measured")}
    assert "| Concurrency |" in d["report_md"]


# 3. One throughput source ------------------------------------------------------------------------------------------

def test_throughput_single_source(client):
    d = client.post("/api/sessions/tp/estimate", json={"config": _cfg()}).json()
    tp = d["throughput"]
    assert set(tp) == {"per_chip", "source", "label", "basis", "basis_label", "note"} and tp["source"] == "estimated"
    assert tp["basis"] == "spec" and tp["basis_label"] == "Rough spec estimate"   # no reference benchmarks yet
    ch = planner.plan(config.normalize(_cfg()))["chosen"]
    assert tp["per_chip"] == ch["est_input_tps_per_gpu"]
    assert d["dashboard_data"]["perGpu"] == tp["per_chip"]
    assert f"{tp['per_chip']:,.0f}" in tp["label"] and "estimate from GPU specs" in tp["label"]

    hw = client.post("/api/recommend/hardware", json={"config": _cfg(), "session": "tp"}).json()
    sel = [c for c in hw["plan"]["candidates"] if c["is_selected"]]
    assert len(sel) == 1 and sel[0]["est_input_tps_per_gpu"] == tp["per_chip"]

    # Explicit tps_per_gpu -> override, used everywhere and kept on the next estimate.
    d2 = client.post("/api/sessions/tp/estimate", json={"tps_per_gpu": 10500}).json()
    assert d2["throughput"]["per_chip"] == 10500 and d2["throughput"]["source"] == "override"
    assert d2["measured"]["input_tps_per_gpu"] == 10500
    assert d2["dashboard_data"]["perGpu"] == 10500
    hw2 = client.post("/api/recommend/hardware", json={"config": _cfg(), "session": "tp"}).json()
    sel2 = next(c for c in hw2["plan"]["candidates"] if c["is_selected"])
    assert sel2["est_input_tps_per_gpu"] == 10500 and sel2["throughput_source"] == "override"
    d3 = client.post("/api/sessions/tp/estimate", json={}).json()
    assert d3["throughput"]["source"] == "override"
    d4 = client.post("/api/sessions/tp/estimate", json={"clear_override": True}).json()
    assert d4["throughput"]["source"] == "estimated"


# 4. Selected model is never hidden ---------------------------------------------------------------------------------

def test_selected_model_and_stale_selection(client):
    stale = {"model_garden_id": "acme/bigmodel@bigmodel-120b-a12b", "hf_id": "acme/bigmodel-120b-a12b",
             "precision": "fp8", "params_b": 120, "active_params_b": 12}
    d = client.post("/api/sessions/stale", json={"config": _cfg(model=stale)}).json()
    sel = d["selected"]
    assert set(sel) == {"model", "hardware", "precision"}
    assert sel["model"]["model_garden_id"] == stale["model_garden_id"]
    assert sel["model"]["hf_id"] == stale["hf_id"] and sel["model"]["name"]
    assert set(sel["hardware"]) == {"accelerator_type", "accelerator_count", "machine_type"}
    assert sel["precision"] == "fp8"

    r = client.post("/api/recommend/model", json={"session": "stale", "with_economics": False}).json()
    picked = [m for m in r["recommendations"] if m["is_selected"]]
    assert len(picked) == 1 and picked[0]["model_garden_id"] == stale["model_garden_id"]
    assert picked[0].get("not_in_catalog") is True
    assert "tips" in r


def test_selected_model_in_catalog_is_marked(client):
    r = client.post("/api/recommend/model", json={"model": GEMMA, "requests": {"avg_input_tokens": 7000}}).json()
    marked = [m for m in r["recommendations"] if m["is_selected"]]
    assert [m["model_garden_id"] for m in marked] == [GEMMA["model_garden_id"]]
    assert all("est_usd_per_m_input" in m for m in r["recommendations"])


# 5. Recommender ----------------------------------------------------------------------------------------------------

def test_recommender_contract_fields():
    req = {"avg_input_tokens": 9000, "avg_output_tokens": 400, "pct_over_32k": 3.0, "pct_over_128k": 1.0}
    res = recommender.recommend({"task_type": "extraction"}, req, {"peak_input_tps": 120000})
    recs = res["recommendations"]
    for m in recs:
        assert isinstance(m["fits_longest_prompt"], bool)
        assert isinstance(m["warnings"], list)
        assert isinstance(m["score_breakdown"], dict) and "base" in m["score_breakdown"]
        assert m["recommended_precision"] in ("bf16", "fp8", "int4", "mxfp4")
        # quantized by us or pre-quantized -> check quality; published in that precision (Kimi K2: FP8 / INT4) -> no
        assert m["needs_quality_check"] is (m["recommended_precision"] != "bf16" and not m.get("native_precision"))
        assert "is_selected" in m
        assert not any("Tip:" in r for r in m["reasons"])
    by_id = {m["id"]: m for m in recs}
    # 128K-context model with >128K prompts: can't serve them, blocking warning, ranked below every model that can.
    g3 = by_id["gemma-3-27b-it"]
    assert g3["fits_longest_prompt"] is False
    assert "Can't serve prompts over 128K unless they are routed elsewhere" in g3["warnings"]
    last_fit = max(i for i, m in enumerate(recs) if m["fits_longest_prompt"])
    first_blocked = min(i for i, m in enumerate(recs) if not m["fits_longest_prompt"])
    assert last_fit < first_blocked
    assert by_id["gemma-4-26b-a4b-it"]["fits_longest_prompt"] is True
    # Precision: BF16 when BF16 weights fit one RTX PRO 6000 under the 75% rule, else FP8 + quality check.
    assert by_id["gemma-4-26b-a4b-it"]["recommended_precision"] == "bf16"
    assert by_id["llama-3.3-70b-instruct"]["recommended_precision"] == "fp8"
    assert by_id["llama-3.3-70b-instruct"]["needs_quality_check"] is True
    # Unclamped score == sum of the breakdown.
    for m in recs:
        total = sum(m["score_breakdown"].values())
        if 10 <= total <= 99 and not m["blocking"]:
            assert m["match_score"] == total
    # Tips once, at response level.
    assert len(res["tips"]) == len(set(res["tips"])) == 1


def test_recommender_long_context_routing_lifts_block():
    req = {"avg_input_tokens": 9000, "pct_over_32k": 3.0, "pct_over_128k": 1.0}
    res = recommender.recommend({"task_type": "extraction"}, req, hybrid_cfg={"route_long": True})
    g3 = next(m for m in res["recommendations"] if m["id"] == "gemma-3-27b-it")
    assert g3["fits_longest_prompt"] is False and g3["blocking"] is False
    assert not any("Can't serve" in w for w in g3["warnings"])


# 6. Hardware recommendations ---------------------------------------------------------------------------------------

def test_hardware_recommendations_contract(client):
    r = client.post("/api/recommend/hardware", json={"config": _cfg()}).json()
    assert r["ok"]
    cands = r["plan"]["candidates"]
    assert cands
    for c in cands:
        assert "fleet_gpus" in c and "usd_per_m_input" in c and "shape_note" in c
        if c["usd_per_m_input"] is not None:
            assert isinstance(c["fleet_gpus"], int) and c["fleet_gpus"] >= c["accelerator_count"]
    costs = [c["usd_per_m_input"] for c in cands if c["usd_per_m_input"] is not None]
    assert costs == sorted(costs)
    h200 = next(c for c in cands if c["accelerator_type"] == "NVIDIA_H200_141GB")
    assert "a3-ultragpu only comes with 8 GPUs" in h200["shape_note"]
    rtx = next(c for c in cands if c["accelerator_type"] == "NVIDIA_RTX_PRO_6000")
    assert rtx["shape_note"] is None
    assert r["pricing_labels"]["vertex_cud_3y"] == "Agent Platform + reservation, 3-yr"
    assert r["pricing_labels"]["vertex_cud_1y"] == "Agent Platform + reservation, 1-yr"
    labels = {o["label"] for c in cands for o in c["pricing_options"]}
    assert "Agent Platform + reservation, 3-yr" in labels and not any("3-Yr CUD" in x for x in labels)


# 7. Break-even text ------------------------------------------------------------------------------------------------

def test_breakeven_hours_capped():
    assert costmodel.breakeven_hours(1e6, 1.0) == (730.0, True)
    assert costmodel.breakeven_hours(1000.0, 0.0) == (730.0, True)
    assert costmodel.breakeven_hours(1000.0, 100.0) == (10.0, False)
    assert costmodel.breakeven_hours(0.0, 100.0) == (0.0, False)


def test_breakeven_never_prose(tmp_path):
    # MaaS limit covers the whole peak: a fleet outage spills nothing past MaaS, so failover is ~free.
    cfg = config.normalize(_cfg(maas={"max_rps": 40}))
    ch = planner.plan(cfg)["chosen"]
    s = costmodel.resilience_and_hybrid_summary(cfg, 9000, ch["prices"], "vertex_cud_3y", 1)
    assert s["breakeven_never"] is True and s["breakeven_outage_hours"] == 730
    md = report.render(cfg, {"candidates": []}, {"endpoint": ch, "levels": [], "estimated": True,
                                                 "input_tps_per_gpu": 9000}, ch["prices"])
    assert "cheaper at any realistic outage rate" in md
    assert not re.search(r">\s?[\d,]{4,} peak", md)
    rec = next(x for x in s["recommendations"] if x["type"] == "resilience_gemini_failover")
    assert "failover is cheaper at any realistic outage rate" in rec["detail"]


# 8. Catalog cache --------------------------------------------------------------------------------------------------

def test_catalog_fast_and_gemini_contract(client):
    client.get("/api/catalog")  # warm (may build once)
    t = time.time()
    r = client.get("/api/catalog")
    assert time.time() - t < 0.3
    gem = r.json()["gemini"]
    for key in ("gemini_flash_lite", "gemini_flash", "gemini_pro"):
        g = gem[key]
        assert {"id", "name", "input_per_m", "output_per_m", "cached_per_m", "source"} <= set(g)
        assert g["id"] == key and g["source"] in ("billing-catalog", "fallback")
    assert r.json()["pricing_labels"]["vertex_cud_3y"] == "Agent Platform + reservation, 3-yr"
    assert "cache" in r.json()


# 9. Gemini prices from the Cloud Billing Catalog -------------------------------------------------------------------

def _sku(desc, nanos, unit="count"):
    return {"description": desc, "category": {"usageType": "OnDemand"}, "serviceRegions": ["global"],
            "pricingInfo": [{"pricingExpression": {"usageUnit": unit, "tieredRates": [
                {"unitPrice": {"units": "0", "nanos": nanos}}]}}]}


def test_parse_gemini_skus_latest_standard_versions():
    skus = [
        _sku("Gemini 2.5 Flash Lite Text Input - Predictions", 100),
        _sku("Gemini 2.5 Flash Lite Text Output - Predictions", 400),
        _sku("Gemini 3.5 Flash Lite Global Text Input - Predictions", 300),
        _sku("Gemini 3.5 Flash Lite Global Text Output - Predictions", 2500),
        _sku("Gemini 3.5 Flash Lite Global Text Input Caching", 30),
        _sku("Gemini 3.5 Flash Lite Regional Text Input - Predictions", 330),
        _sku("Gemini 3.5 Flash Lite Global Text Input - Deferred Predictions", 150),
        _sku("Gemini 3.8 Flash Global Text Input - Predictions", 1500),
        _sku("Gemini 3.8 Flash Global Text Output - Predictions", 7500),
        _sku("Gemini 3.8 Flash Global Text Input Caching", 150),
        _sku("Gemini 3.9 Flash Cyber Global Text Input - Predictions", 9),
        _sku("Gemini 3.9 Flash Cyber Global Text Output - Predictions", 9),
        _sku("Gemini 4.0 Flash Global Text Input - Batch Predictions", 1),
        _sku("Gemini 4.0 Flash Global Text Output - Batch Predictions", 1),
        _sku("Gemini 2.5 Pro Text Input - Predictions", 1250),
        _sku("Gemini 2.5 Pro Text Output - Predictions", 10000),
        _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input - Predictions", 2000),
        _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Output - Predictions", 12000),
        _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input Caching", 200),
        _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input Priority - Predictions", 3600),
        _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input (Long) - Predictions", 4000),
    ]
    out = pricing.parse_gemini_skus(skus)
    assert out["gemini_flash_lite"]["version"] == "3.5"
    assert (out["gemini_flash_lite"]["input_per_m"], out["gemini_flash_lite"]["output_per_m"],
            out["gemini_flash_lite"]["cached_per_m"]) == (0.3, 2.5, 0.03)
    assert out["gemini_flash"]["version"] == "3.8" and out["gemini_flash"]["output_per_m"] == 7.5
    assert out["gemini_pro"]["version"] == "3.1"
    assert (out["gemini_pro"]["input_per_m"], out["gemini_pro"]["output_per_m"]) == (2.0, 12.0)


def test_gemini_prices_fallback_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "CACHE", tmp_path)
    monkeypatch.setattr(pricing, "_GEM_MEM", {})

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(pricing, "skus", boom)
    g = pricing.gemini_prices("my-gcp-project")
    assert all(g[k]["source"] == "fallback" for k in ("gemini_flash_lite", "gemini_flash", "gemini_pro"))
    assert g["gemini_pro"]["input_per_m"] > g["gemini_flash"]["input_per_m"] > g["gemini_flash_lite"]["input_per_m"]


def test_costmodel_and_report_use_catalog_gemini_prices():
    cfg = config.normalize(_cfg(requests={"pct_medium": 3.0, "pct_long": 1.0}))
    ch = planner.plan(cfg)["chosen"]
    cheap = {k: dict(v) for k, v in pricing.gemini_prices().items()}
    price_keys = ("input_per_m", "output_per_m", "cached_per_m", "long_input_per_m", "long_output_per_m")
    pricey = {k: {**v, **{f: v[f] * 10 for f in price_keys if f in v}} for k, v in cheap.items()}
    a = costmodel.scenario(cfg, 9000, ch["prices"], "vertex_cud_3y", 1, route_long=True,
                           route_long_target="gemini_flash_lite", gemini_catalog=cheap)
    b = costmodel.scenario(cfg, 9000, ch["prices"], "vertex_cud_3y", 1, route_long=True,
                           route_long_target="gemini_flash_lite", gemini_catalog=pricey)
    assert b["long_maas_cost"] == pytest.approx(a["long_maas_cost"] * 10, rel=1e-6)
    md = report.render(cfg, {"candidates": []}, {"endpoint": ch, "levels": [], "estimated": True,
                                                 "input_tps_per_gpu": 9000}, ch["prices"])
    for g in cheap.values():
        assert g["name"] in md
    assert "2.5 Flash-Lite" not in md or cheap["gemini_flash_lite"]["name"].startswith("Gemini 2.5")


# 10. Example session -----------------------------------------------------------------------------------------------

def test_example_session_read_only_and_copy(client):
    sessions = {s["name"]: s for s in client.get("/api/sessions").json()["sessions"]}
    assert sessions["blog-example"]["example"] is True
    d = client.get("/api/sessions/blog-example").json()
    assert d["example"] is True
    c = d["config"]
    assert c["maas"]["max_rps"] == 15 and c["requests"]["system_prompt_tokens"] == 1800
    assert c["traffic"]["preset"] == "us_core_hours" and c["traffic"]["average_input_tps"] == 30000
    assert round(c["traffic"]["peak_input_tps"], -3) == 140000 and c["project"] == "my-gcp-project"
    assert d["throughput"]["per_chip"] == server.BLOG_EXAMPLE_TPS and d["measured"]["levels"] == []

    for path, body in (("/api/sessions/blog-example", {"config": c}),
                       ("/api/sessions/blog-example/yaml", {"yaml_text": "name: x"}),
                       ("/api/sessions/blog-example/estimate", {}),
                       ("/api/sessions/blog-example/run", {"mode": "plan"})):
        assert client.post(path, json=body).status_code == 403

    r = client.post("/api/sessions/my-copy", json={"copy_of": "blog-example"})
    assert r.status_code == 200
    cp = r.json()
    assert cp["example"] is False and cp["ui_state"]["copied_from"] == "blog-example"
    assert cp["config"]["maas"]["max_rps"] == 15
    assert cp["config"]["pricing"]["tier"] == "vertex_cud_1y" and cp["config"]["traffic"]["preset"] == "us_core_hours"
    assert cp["throughput"] == d["throughput"]
    assert (server.RUNS_DIR / "my-copy" / "report.md").exists()
    sessions = {s["name"]: s for s in client.get("/api/sessions").json()["sessions"]}
    assert sessions["my-copy"]["example"] is False
    # Copy is editable; copying onto an existing session is refused.
    assert client.post("/api/sessions/my-copy/estimate", json={}).status_code == 200
    assert client.post("/api/sessions/my-copy", json={"copy_of": "blog-example"}).status_code == 409


# The patterns themselves are private (they describe what must stay out of this repo), so they live outside it:
# $DILEMMA_FORBIDDEN_PATTERNS, else ../aiops-dilemma-private/forbidden-patterns.txt next to the checkout. One regex per
# line, # comments. Without the file the check is skipped (e.g. in a fresh public clone).
def _forbidden_patterns() -> list[str]:
    path = Path(os.environ.get("DILEMMA_FORBIDDEN_PATTERNS") or ROOT.parent / "aiops-dilemma-private" / "forbidden-patterns.txt")
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_no_customer_specific_strings_in_repo():
    FORBIDDEN = _forbidden_patterns()
    if not FORBIDDEN:
        pytest.skip("no private forbidden-patterns file")
    skip_dirs = {".venv", ".git", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
    skip_files = {"uv.lock", Path(__file__).name}
    # Exactly what git would commit (tracked + untracked, not ignored), else every file outside runs/.
    git = shutil.which("git")
    out = subprocess.run([git, "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True) if git else None
    if out is not None and out.returncode == 0:
        files = [ROOT / ln for ln in out.stdout.splitlines() if ln and (ROOT / ln).is_file()]
    else:
        files = [p for p in ROOT.rglob("*") if p.is_file() and "runs" not in p.relative_to(ROOT).parts[:1]]
    hits = []
    for f in files:
        if skip_dirs & set(f.parts) or f.name in skip_files or f.stat().st_size > 5_000_000:
            continue
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        hits += [f"{f.relative_to(ROOT)}: {pat}" for pat in FORBIDDEN if re.search(pat, text)]
    assert not hits, hits


# 11. model.js (shared with the exported dashboard) -----------------------------------------------------------------

def _node(script: str) -> dict:
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_guards_override_and_breakeven(client):
    data = client.get("/api/sessions/blog-example").json()["dashboard_data"]
    script = f"""
const M = require({json.dumps(str(ASSETS / 'model.js'))});
const DATA = {json.dumps(data, default=str)};
const p = Object.assign({{}}, DATA.defaults);
const res = {{}};
for (const v of [0, "", null, "abc", 0.5]) {{ p.resilience = v; res[String(v)] = M.scenario(DATA, p).gpus; }}
p.resilience = 1;
const base = M.scenario(DATA, p);
const inc = M.incidents(DATA, p, base);
const o0 = M.scenario(DATA, Object.assign({{}}, p, {{ fleetOverride: 0 }}));
const o3 = M.scenario(DATA, Object.assign({{}}, p, {{ fleetOverride: 3 }}));
const big = M.incidents(DATA, Object.assign({{}}, p, {{ liteIn: 0.00001, liteOut: 0.00001 }}), base);
console.log(JSON.stringify({{ res, baseGpus: base.gpus, optimal: base.optimalGpus, inc,
  o0: {{ gpus: o0.gpus, gem: o0.fleet.geminiCost, over: o0.overridden, total: o0.total }},
  o3: {{ gpus: o3.gpus, reps: o3.fleetReplicas, series: o3.fleet.series.fleet.length, cap: o3.capacityTps }},
  big: {{ h: big.breakeven_outage_hours, never: big.breakeven_never }} }}));
"""
    r = _node(script)
    assert all(v == r["baseGpus"] and v > 0 for v in r["res"].values())
    assert r["inc"]["breakeven_outage_hours"] <= 730 and isinstance(r["inc"]["breakeven_never"], bool)
    assert r["big"] == {"h": 730, "never": True}
    assert r["o0"]["gpus"] == 0 and r["o0"]["over"] is True and r["o0"]["gem"] > 0
    assert r["o3"]["gpus"] == 3 * data["replica"] and r["o3"]["reps"] == 3 and r["o3"]["series"] == 168
    assert r["o3"]["cap"] > 0 and r["optimal"] == r["baseGpus"]


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_exported_dashboard_script_parses(client, tmp_path):
    html = (server.RUNS_DIR / "blog-example" / "dashboard.html").read_text()
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert scripts
    for i, js in enumerate(scripts):
        f = tmp_path / f"s{i}.js"
        f.write_text(js)
        out = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
    assert "Gemini 2.5 Flash-Lite" not in html or "Gemini 2.5" in json.dumps(pricing.gemini_prices())
