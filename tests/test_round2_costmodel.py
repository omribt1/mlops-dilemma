"""Round 2 contract: rate-limit-aware baseline, Gemini 4, thinking tokens, custom / monthly traffic (Python + model.js)."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import catalog, config, costmodel, dashboard, planner, pricing, report, server, traffic

ROOT = Path(__file__).resolve().parents[1]
MODEL_JS = ROOT / "src" / "aiops_dilemma" / "assets" / "model.js"
NODE = shutil.which("node")
TPS = 9000.0
GEM = copy.deepcopy(catalog.GEMINI_MODELS)   # fixed prices: tests don't depend on the live catalog


@pytest.fixture(scope="module")
def blog():
    cfg = config.normalize(yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text()))
    ch = planner.plan(cfg)["chosen"]
    return cfg, ch


def _sc(cfg, ch, tier="cud_3y", **kw):
    return costmodel.scenario(cfg, TPS, ch["prices"], tier, 1, gemini_catalog=GEM, **kw)


# Baseline ------------------------------------------------------------------------------------------------------------

def test_blog_baseline_is_maas_plus_flash_lite_overflow(blog):
    cfg, ch = blog
    r = _sc(cfg, ch, "vertex_cud_1y")
    assert r["gpus"] == 12 and round(r["total"] / 1e2) == 358 and round(r["vs_baseline"], 2) == 1.12
    assert r["maas_only_feasible"] is False
    assert r["maas_cap_rps"] == 15 and r["maas_peak_rps"] == pytest.approx(r["peak"] / 5000)
    b = r["baseline"]
    assert set(b) == {"kind", "total", "maas_cost", "overflow_cost", "overflow_target", "label"}
    assert b["kind"] == "maas_plus_overflow" and b["overflow_target"] == "gemini_flash_lite"
    assert b["total"] == pytest.approx(b["maas_cost"] + b["overflow_cost"])
    assert b["total"] > r["maas_only"] and b["maas_cost"] < r["maas_only"]
    assert r["vs_baseline"] == pytest.approx(r["total"] / b["total"])
    assert r["total"] == pytest.approx(r["gpu_cost"] + r["maas_overflow_cost"] + r["gemini_overflow_cost"]
                                       + r["long_maas_cost"])


def test_baseline_feasible_when_limit_covers_peak(blog):
    cfg, ch = blog
    r = _sc(cfg, ch, maas_rps=40)   # planned at 40 / 1.3 = 30.8 req/s > the 28 req/s peak
    assert r["maas_only_feasible"] is True
    assert r["baseline"]["kind"] == "maas_only" and r["baseline"]["overflow_target"] is None
    assert r["baseline"]["total"] == pytest.approx(r["maas_only"]) and r["vs_baseline"] == pytest.approx(r["vs_maas"])


def test_baseline_priced_at_failover_target(blog):
    cfg, ch = blog
    lite = _sc(cfg, ch)["baseline"]
    g4 = _sc(cfg, ch, failover_target="gemini_4")["baseline"]
    assert g4["overflow_target"] == "gemini_4" and g4["overflow_cost"] > lite["overflow_cost"]
    assert g4["maas_cost"] == pytest.approx(lite["maas_cost"])
    # Open-model MaaS can't take traffic beyond its own limit: falls back to Flash-Lite.
    assert _sc(cfg, ch, failover_target="open_maas")["baseline"]["overflow_target"] == "gemini_flash_lite"
    c = copy.deepcopy(cfg)
    c["hybrid"]["failover_target"] = "gemini_flash"
    assert _sc(c, ch)["baseline"]["overflow_target"] == "gemini_flash"


def test_fleet_override_spills_beyond_the_limit_to_gemini(blog):
    cfg, ch = blog
    opt = _sc(cfg, ch)
    small = _sc(cfg, ch, fleet_override=5)
    assert small["gpus"] == 5 and small["optimal_gpus"] == opt["gpus"] and small["overridden"]
    assert small["gemini_overflow_cost"] > 0
    assert small["total"] == pytest.approx(small["gpu_cost"] + small["maas_overflow_cost"] + small["gemini_overflow_cost"])
    assert _sc(cfg, ch, fleet_override=opt["gpus"])["total"] == pytest.approx(opt["total"])


def test_peak_overflow_to_gemini(blog):
    cfg, ch = blog
    r = _sc(cfg, ch, peak_overflow_target="gemini_4")
    # Overflow fills MaaS within its limit first, Gemini 4 takes the rest: allowing it at peaks can only widen the
    # choice, so the optimum is never dearer than with the default policy (open-model MaaS only at peaks).
    base = _sc(cfg, ch)
    assert r["peak_overflow_target"] == "gemini_4" and r["gemini_allowed_at_peaks"] and r["total"] <= base["total"]
    assert r["spill_gemini_target"] == "gemini_4" and base["spill_gemini_target"] == "gemini_flash_lite"


# Thinking ------------------------------------------------------------------------------------------------------------

def test_effective_output_tokens():
    assert config.effective_output_tokens({"avg_output_tokens": 950}) == 950
    req = {"avg_output_tokens": 950, "thinking_share": 0.5, "avg_thinking_tokens": 2000}
    assert config.effective_output_tokens(req) == 1950
    assert config.effective_output_tokens({**req, "thinking_share": 3}) == 2950       # share clamped to 1
    assert config.effective_output_tokens({**req, "avg_thinking_tokens": None}) == 950
    assert config.DEFAULTS["requests"]["avg_thinking_tokens"] == 0


def test_thinking_raises_output_ratio_and_costs(blog):
    cfg, ch = blog
    c = copy.deepcopy(cfg)
    c["requests"].update(thinking_share=0.5, avg_thinking_tokens=2000)
    assert costmodel.request_mix(c)[0] == pytest.approx(1500 / 5000)
    assert costmodel.buckets(c)["small"]["out_ratio"] == pytest.approx(1500 / 5000)
    a, b = _sc(cfg, ch), _sc(c, ch)
    assert a["effective_output_tokens"] == pytest.approx(950) and b["effective_output_tokens"] == 1500
    assert b["maas_only"] > a["maas_only"] and b["baseline"]["total"] > a["baseline"]["total"]


# Traffic -------------------------------------------------------------------------------------------------------------

def _week():
    return [1000.0 + 500 * np.sin(i / 24 * 2 * np.pi) ** 2 for i in range(168)]


def test_custom_hourly_precedence_and_monthly(tmp_path):
    week = _week()
    t = {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500, "custom_hourly": week}
    assert np.allclose(traffic.weekly_profile(t), week)
    csv = tmp_path / "w.csv"
    csv.write_text("input_tokens_per_sec\n" + "\n".join("7" for _ in range(168)))
    assert np.allclose(traffic.weekly_profile({**t, "hourly_csv": str(csv)}), 7)       # CSV wins
    m = 5e9
    scaled = traffic.weekly_profile({**t, "monthly_input_tokens": m})
    assert traffic.monthly_tokens(scaled) == pytest.approx(m)
    assert np.allclose(scaled / scaled.max(), np.array(week) / max(week))              # same shape
    par = traffic.weekly_profile({"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500,
                                  "weekend_factor": 0.08, "monthly_input_tokens": 30000 * 3600 * 730})
    assert par.max() == pytest.approx(100000) and par.min() == pytest.approx(2500)     # peak / floor kept
    assert par.mean() == pytest.approx(30000, rel=1e-3)                               # overrides the average


def test_normalize_traffic_fields():
    week = _week()
    cfg = config.normalize({"traffic": {"custom_hourly": week, "monthly_input_tokens": 0}})
    t = cfg["traffic"]
    assert t["custom_hourly"] == pytest.approx(week) and t["monthly_input_tokens"] is None
    assert t["peak_input_tps"] == pytest.approx(max(week), abs=1e-3)
    assert t["average_input_tps"] == pytest.approx(np.mean(week), abs=1e-3)
    assert t["floor_input_tps"] == pytest.approx(min(week), abs=1e-3)
    bad = config.normalize({"traffic": {"custom_hourly": [1, 2, 3], "peak_input_tps": 100, "average_input_tps": 50,
                                        "floor_input_tps": 10}})
    assert bad["traffic"]["custom_hourly"] is None and "168" in bad["_warnings"][0]
    assert "_warnings" not in config.to_yaml(bad)
    assert config.clean_custom_hourly([-1] + [1] * 167)[0][0] == 0
    assert config.normalize({"traffic": {"off_peak_multiplier": 25}})["traffic"]["off_peak_multiplier"] == 10
    assert config.normalize({"traffic": {"off_peak_multiplier": 0.2}})["traffic"]["off_peak_multiplier"] == 1
    assert config.clamp_off_peak("x") == 1
    mon = config.normalize({"traffic": {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500,
                                        "weekend_factor": 0.08, "monthly_input_tokens": 30000 * 3600 * 730}})
    assert mon["traffic"]["average_input_tps"] == pytest.approx(30000, rel=1e-3)
    assert mon["traffic"]["peak_input_tps"] == pytest.approx(100000)


def test_off_peak_clamped_in_scenario(blog):
    cfg, ch = blog
    assert _sc(cfg, ch, off_peak=25)["total"] == pytest.approx(_sc(cfg, ch, off_peak=10)["total"])


def test_session_traffic_preview_uses_custom_week(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    client = TestClient(server.app)
    week = [round(x, 1) for x in _week()]
    d = client.post("/api/sessions/custom-week", json={"config": {
        "project": "my-gcp-project", "traffic": {"custom_hourly": week, "monthly_input_tokens": None}}}).json()
    assert d["traffic_profile"] == pytest.approx(week, abs=0.11)


# Gemini 4 ------------------------------------------------------------------------------------------------------------

def _sku(desc, nanos, regions=("global",), unit="count"):
    return {"description": desc, "serviceRegions": list(regions), "category": {"usageType": "OnDemand"},
            "pricingInfo": [{"pricingExpression": {"usageUnit": unit,
                                                   "tieredRates": [{"unitPrice": {"units": "0", "nanos": nanos}}]}}]}


def test_parse_gemini4_skus():
    skus = [
        _sku("Gemini 4 Argon Global Text Input - Online Predictions", 4000),
        _sku("Gemini 4 Argon Global Text Output - Online Predictions", 20000),
        _sku("Gemini 4 Argon Global Text Input Caching Online", 200),
        _sku("Gemini 4 Argon Cyber Permissive Global Text Input - Online Predictions", 1),
        _sku("CodeMender Gemini 4 Argon Global Text Input", 2),
        _sku("Gemini 4 Argon Global Text Input - Batch Predictions", 3),
        _sku("Gemini 4 Argon Global Text Output - Priority Predictions", 4),
        _sku("Gemini 4 Argon Global Text Input - Offpeak Predictions", 5),
        _sku("Gemini 4 Argon Regional Text Input - Online Predictions", 6),
        _sku("Gemini 4 Argon Global Image Input - Online Predictions", 7),
        _sku("Gemini 4 Argon Global Text Input Caching Flex", 8),
    ]
    g = pricing.parse_gemini4_skus(skus)
    assert g["version"] == "4"
    assert (g["input_per_m"], g["output_per_m"], g["cached_per_m"]) == (4.0, 20.0, 0.2)
    assert pricing.parse_gemini4_skus(skus[3:]) is None
    assert "gemini_4" not in pricing.parse_gemini_skus(skus)   # the tiered parser leaves it alone


def test_gemini_4_in_catalog_and_presets():
    g = catalog.GEMINI_MODELS["gemini_4"]
    assert g["name"] == "Gemini 4 Carbon" and (g["input_per_m"], g["output_per_m"]) == (4.0, 20.0)
    presets = catalog.maas_presets({k: v for k, v in catalog.GEMINI_MODELS.items() if k != "gemini_4"})
    assert "gemini_4" in presets and "Agent Platform" in presets["gemini_4"]["name"]
    assert config.OVERFLOW_TARGETS == ("open_maas", "gemini_flash_lite", "gemini_flash", "gemini_pro", "gemini_4")
    # An older cached catalog without gemini_4 still works.
    old = {k: v for k, v in GEM.items() if k != "gemini_4"}
    assert "gemini_4" in costmodel._gemini_catalog(old)


# Report ----------------------------------------------------------------------------------------------------------------

def test_report_verdict_wording(blog):
    cfg, ch = blog
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    est = {"endpoint": ep, "levels": [], "estimated": True, "input_tps_per_gpu": TPS, "throughput_source": "override"}
    md = report.render(cfg, {"candidates": []}, est, ch["prices"])
    assert ("MaaS can't serve your peak: limit 15 req/s (planned at 11.5 req/s per hour to absorb bursts) vs 28.0 "
            "req/s at peak; MaaS up to its limit plus") in md
    assert "Vertex AI" not in md.split("## Planner candidates")[0]
    c = copy.deepcopy(cfg)
    c["maas"]["max_rps"] = 40   # planned at 40 / 1.3 = 30.8 req/s, above the 28 req/s peak
    md2 = report.render(c, {"candidates": []}, est, ch["prices"])
    assert "can't serve" not in md2 and "of MaaS-only** (" in md2 and "fits the 40 req/s limit, planned at 30.8 req/s per hour" in md2


# Python <-> model.js parity for the new fields ---------------------------------------------------------------------------

@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_parity_round2(blog):
    cfg, ch = blog
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    variants = [
        ({}, {}, {}),
        ({"thinking_share": 0.4, "avg_thinking_tokens": 3000}, {}, {}),
        ({}, {"maas_rps": 20}, {"maasRps": 20}),
        ({}, {"maas_rps": 0}, {"maasRps": 0}),
        ({}, {"failover_target": "gemini_4"}, {"failoverTarget": "gemini_4"}),
        ({}, {"failover_target": "open_maas"}, {"failoverTarget": "open_maas"}),
        ({}, {"peak_overflow_target": "gemini_4"}, {"peakOverflowTarget": "gemini_4"}),
        ({}, {"fleet_override": 5}, {"fleetOverride": 5}),
        ({}, {"fleet_override": 5, "failover_target": "gemini_pro"}, {"fleetOverride": 5, "failoverTarget": "gemini_pro"}),
        ({}, {"off_peak": 3}, {"offPeak": 3}),
        ({}, {"off_peak": 25}, {"offPeak": 25}),
        ({"pct_over_32k": 3, "pct_over_128k": 0}, {"route_long": True, "route_long_target": "open_maas"},
         {"routeLong": True, "routeLongTarget": "open_maas"}),
        ({"thinking_share": 1, "avg_thinking_tokens": 1500, "pct_over_32k": 3, "pct_over_128k": 0},
         {"route_long": True, "route_long_target": "gemini_4"}, {"routeLong": True, "routeLongTarget": "gemini_4"}),
    ]
    cases, expected = [], []
    for req, kw, jp in variants:
        c = copy.deepcopy(cfg)
        c["requests"].update(req)
        data = dashboard.build_data(c, {"endpoint": ep, "input_tps_per_gpu": TPS, "levels": [], "estimated": True,
                                        "throughput_source": "override"}, ch["prices"])
        data["gemini"] = GEM
        for k, (i, o) in {"liteIn": ("gemini_flash_lite", "input_per_m"), "liteOut": ("gemini_flash_lite", "output_per_m"),
                          "flashIn": ("gemini_flash", "input_per_m"), "flashOut": ("gemini_flash", "output_per_m"),
                          "proIn": ("gemini_pro", "input_per_m"), "proOut": ("gemini_pro", "output_per_m"),
                          "g4In": ("gemini_4", "input_per_m"), "g4Out": ("gemini_4", "output_per_m")}.items():
            data["defaults"][k] = GEM[i][o]
        py = costmodel.scenario(c, TPS, ch["prices"], "cud_3y", data["replica"], gemini_catalog=GEM, resilience=1.0,
                                **{"route_long": False, **kw})
        cases.append({"data": data, "p": {**data["defaults"], "pricing": "cud_3y", "resilience": 1,
                                          "routeLong": False, "peakOverflowTarget": "open_maas",
                                          "failoverTarget": "gemini_flash_lite", **jp}})
        expected.append(py)
    script = f"""
const M = require({json.dumps(str(MODEL_JS))});
const cases = {json.dumps(cases, default=str)};
console.log(JSON.stringify(cases.map(c => {{ const r = M.scenario(c.data, c.p);
  return {{gpus: r.gpus ?? null, total: r.total ?? null, vsBaseline: r.vsBaseline ?? null, base: r.baseline,
          feasible: r.maasOnlyFeasible, peakRps: r.maasPeakRps, capRps: r.maasCapRps, effOut: r.effectiveOutput,
          gem: r.geminiOverflowCost ?? null, maasOnly: r.maasOnly, infeasible: r.infeasible ?? null}}; }})));
"""
    out = subprocess.run([NODE, "-"], input=script, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    for py, js, (req, kw, _) in zip(expected, got, variants):
        tag = f"{req} {kw}"
        assert js["gpus"] == py.get("gpus"), tag
        assert js["feasible"] == py["maas_only_feasible"], tag
        assert js["peakRps"] == pytest.approx(py["maas_peak_rps"], rel=1e-6), tag
        assert js["capRps"] == py["maas_cap_rps"], tag
        assert js["effOut"] == pytest.approx(py["effective_output_tokens"]), tag
        assert js["maasOnly"] == pytest.approx(py["maas_only"], rel=1e-6), tag
        for k in ("kind", "overflow_target", "label"):
            assert js["base"][k] == py["baseline"][k], tag
        for k in ("total", "maas_cost", "overflow_cost"):
            assert js["base"][k] == pytest.approx(py["baseline"][k], rel=1e-6, abs=1e-6), tag
        if py.get("total") is None:
            assert js["total"] is None, tag
            continue
        assert js["total"] == pytest.approx(py["total"], rel=1e-6), tag
        assert js["vsBaseline"] == pytest.approx(py["vs_baseline"], rel=1e-6), tag
        assert js["gem"] == pytest.approx(py["gemini_overflow_cost"], rel=1e-6, abs=1e-6), tag
