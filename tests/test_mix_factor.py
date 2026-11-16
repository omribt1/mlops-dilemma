"""Prompt-length mix factor: every per-chip throughput belongs to a reference mix (f_ref); the all-on-GPU fleet runs
at per_chip x f_ref / f_now, the routed short-prompt fleet at per_chip x f_ref. Python and model.js must agree."""

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import analyze, config, costmodel, dashboard, planner, report, server, simulate

ROOT = Path(__file__).resolve().parents[1]
MODEL_JS = ROOT / "src" / "aiops_dilemma" / "assets" / "model.js"
NODE = shutil.which("node")
TPS = 9000.0
MIXES = ((0, 0), (3, 1), (10, 3))


@pytest.fixture(scope="module")
def blog():
    cfg = config.normalize(yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text()))
    ch = planner.plan(cfg)["chosen"]
    return cfg, ch


def _mix(cfg, o32, o128):
    c = copy.deepcopy(cfg)
    c["requests"]["pct_over_32k"], c["requests"]["pct_over_128k"] = o32, o128
    return c


def _level(input_tps=10000.0):
    return {"concurrency": 32, "ok": 300, "errors": 0, "error_rate": 0.0, "rps": 1.5, "input_tps": input_tps,
            "output_tps": 1300.0, "avg_in": 7000, "avg_out": 950, "lat_p50": 20.0, "lat_p95": 30.0}


# Mix factor ---------------------------------------------------------------------------------------------------------

def test_mix_factor_definition(blog):
    cfg, _ = blog
    assert costmodel.buckets(cfg)["mix_factor"] == 1.0
    for o32, o128 in MIXES[1:]:
        b = costmodel.buckets(_mix(cfg, o32, o128))
        f = sum(b[k]["token_share"] * b[k]["speed_penalty"] for k in ("small", "medium", "large"))
        assert b["mix_factor"] == pytest.approx(f) and f > 1.0
        assert b["small_speedup"] == pytest.approx(f)      # routed short-prompt rate = blended x f_now = per_chip x f_ref
    assert costmodel.mix_factor(_mix(cfg, 10, 3)) > costmodel.mix_factor(_mix(cfg, 3, 1))


def test_ref_mix_factor_rules(blog):
    cfg = _mix(blog[0], 3, 1)
    f_now = costmodel.mix_factor(cfg)
    real = {"levels": [_level()], "input_tps_per_gpu": 10000.0}
    assert costmodel.ref_mix_factor(cfg, None) == 1.0
    assert costmodel.ref_mix_factor(cfg, {"estimated": True, "levels": [], "throughput_source": "estimated"}) == 1.0
    assert costmodel.ref_mix_factor(cfg, {"estimated": True, "levels": [], "throughput_source": "override"}) == 1.0
    assert costmodel.ref_mix_factor(cfg, {"per_chip": 9000, "source": "override"}) == 1.0
    assert costmodel.ref_mix_factor(cfg, {"per_chip": 9000, "source": "estimated"}) == 1.0
    # Real run: recorded mix factor wins; older runs without it are taken as blended for the current mix.
    assert costmodel.ref_mix_factor(cfg, {**real, "mix_factor": 1.3}) == 1.3
    assert costmodel.ref_mix_factor(cfg, real) == pytest.approx(f_now)
    # "calibrated" (and a throughput dict from a measured run) behave like measured.
    assert costmodel.ref_mix_factor(cfg, {"estimated": True, "levels": [], "throughput_source": "calibrated"}) \
        == pytest.approx(f_now)
    assert costmodel.ref_mix_factor(cfg, {"estimated": True, "levels": [], "throughput_source": "calibrated",
                                          "mix_factor": 1.2}) == 1.2
    assert costmodel.ref_mix_factor(cfg, {"per_chip": 9000, "source": "measured", "mix_factor": 1.1}) == 1.1
    assert costmodel.ref_mix_factor(cfg, {"per_chip": 9000, "source": "measured"}) == pytest.approx(f_now)


# Cost model ---------------------------------------------------------------------------------------------------------

def test_blog_numbers_unchanged_without_long_prompts(blog):
    cfg, ch = blog
    gke = costmodel.scenario(cfg, TPS, ch["prices"], "cud_3y", 1)
    assert gke["gpus"] == 12 and round(gke["total"] / 1e3) == 21 and round(gke["vs_baseline"] * 100) == 65
    vtx = costmodel.scenario(cfg, TPS, ch["prices"], "vertex_cud_3y", 1)
    assert vtx["gpus"] == 12 and round(vtx["total"] / 1e3) == 25 and round(vtx["vs_baseline"] * 100) == 79
    # No prompts over 32K: routing changes nothing (f_now = 1, nothing to route).
    assert costmodel.scenario(cfg, TPS, ch["prices"], "cud_3y", 1, route_long=True)["gpus"] == 12


def test_long_prompts_grow_the_all_on_gpu_fleet(blog):
    cfg, ch = blog
    gpus = [costmodel.scenario(_mix(cfg, a, b), TPS, ch["prices"], "cud_3y", 1)["gpus"] for a, b in MIXES]
    assert gpus[0] == 12 and gpus[0] < gpus[1] < gpus[2], gpus      # once 12, 12, 12 (issue repro)


def test_routing_beats_all_on_gpu_and_short_rate_is_independent_of_long_share(blog):
    cfg, ch = blog
    for o32, o128 in MIXES[1:]:
        c = _mix(cfg, o32, o128)
        for f_ref in (1.0, 1.4):
            allg = costmodel.scenario(c, TPS, ch["prices"], "cud_3y", 1, route_long=False, mix_factor_ref=f_ref)
            rout = costmodel.scenario(c, TPS, ch["prices"], "cud_3y", 1, route_long=True,
                                      route_long_target="gemini_flash_lite", mix_factor_ref=f_ref)
            assert rout["gpus"] < allg["gpus"]
            assert allg["blended_per_gpu"] == pytest.approx(TPS * f_ref / costmodel.mix_factor(c))
            # Routed short-prompt rate = per_chip x f_ref, whatever the long-prompt share.
            assert rout["blended_per_gpu"] * rout["buckets"]["small_speedup"] == pytest.approx(TPS * f_ref)
            # The routed fleet equals a short-only fleet on the remaining token volume at per_chip x f_ref.
            demand = costmodel.traffic.weekly_profile(c["traffic"]) * rout["buckets"]["small"]["token_share"]
            b = rout["buckets"]
            small_rate = costmodel.maas_rate(c, b["small"]["out_ratio"], b["small"]["cached"])
            ref = costmodel.fixed_fleet(demand, TPS * f_ref, 1, ch["prices"]["cud_3y"],
                                        c["maas"]["max_rps"] * b["small"]["mean"], small_rate,
                                        c["constraints"]["burst_headroom"])
            assert rout["fleet"].gpus == ref.gpus


def test_f_ref_equal_to_f_now_keeps_old_blended_behaviour(blog):
    cfg, ch = blog
    c = _mix(cfg, 3, 1)
    old = costmodel.scenario(c, TPS, ch["prices"], "cud_3y", 1, mix_factor_ref=costmodel.mix_factor(c))
    assert old["gpus"] == 12 and old["blended_per_gpu"] == pytest.approx(TPS)


def test_breakeven_utilization_uses_blended_rate(blog):
    cfg, ch = blog
    c = _mix(cfg, 3, 1)
    p = ch["prices"]["cud_3y"]
    assert costmodel.breakeven_utilization(c, TPS, p) == pytest.approx(
        costmodel.breakeven_utilization(c, TPS, p, costmodel.mix_factor(c)) * costmodel.mix_factor(c))


# Planner ------------------------------------------------------------------------------------------------------------

def test_spec_estimate_is_a_short_prompt_rate(blog):
    cfg, ch = blog
    acc, n = ch["accelerator_type"], ch["accelerator_count"]
    c = _mix(cfg, 3, 1)
    short_only = copy.deepcopy(cfg)
    short_only["requests"]["avg_input_tokens"] = costmodel.buckets(c)["small"]["mean"]
    assert planner.estimate_tps_per_chip(c, acc, n) == planner.estimate_tps_per_chip(short_only, acc, n)
    # Spec estimate (f_ref 1) with long prompts: more chips than without.
    est = planner.estimate_tps_per_chip(cfg, acc, n)
    e0 = planner.fleet_economics(cfg, {**ch, "est_input_tps_per_gpu": est}, tier="cud_3y")
    e1 = planner.fleet_economics(c, {**ch, "est_input_tps_per_gpu": est}, tier="cud_3y")
    assert e1["fleet_gpus"] > e0["fleet_gpus"]


def test_fleet_economics_session_throughput_uses_f_ref(blog):
    cfg, ch = blog
    c = _mix(cfg, 3, 1)
    f_now = costmodel.mix_factor(c)
    over = planner.fleet_economics(c, ch, per_chip=TPS, tier="cud_3y", mix_factor_ref=1.0)
    meas = planner.fleet_economics(c, ch, per_chip=TPS, tier="cud_3y", mix_factor_ref=f_now)
    assert over["fleet_gpus"] == costmodel.scenario(c, TPS, ch["prices"], "cud_3y", 1)["gpus"]
    assert meas["fleet_gpus"] == 12 and over["fleet_gpus"] > meas["fleet_gpus"]


def test_enrich_candidates_selected_hardware_f_ref(blog):
    cfg, ch = blog
    c = _mix(cfg, 3, 1)
    sel = {"accelerator_type": ch["accelerator_type"], "accelerator_count": ch["accelerator_count"]}
    o = planner.enrich_candidates(c, [ch], throughput={"per_chip": TPS, "source": "override"}, selected=sel)[0]
    m = planner.enrich_candidates(c, [ch], throughput={"per_chip": TPS, "source": "measured",
                                                       "mix_factor": costmodel.mix_factor(c)}, selected=sel)[0]
    assert o["est_input_tps_per_gpu"] == m["est_input_tps_per_gpu"] == TPS
    assert m["fleet_gpus"] == 12 and o["fleet_gpus"] > 12


# Simulation ---------------------------------------------------------------------------------------------------------

def test_simulate_bucket_tps(blog):
    cfg = _mix(blog[0], 3, 1)
    f_now = costmodel.mix_factor(cfg)
    t = simulate.bucket_tps(cfg, TPS)                                  # spec/override: short-prompt rate
    assert t["small"] == pytest.approx(TPS)
    assert t["medium"] == pytest.approx(TPS / 3.0) and t["large"] == pytest.approx(TPS / 4.0)
    t = simulate.bucket_tps(cfg, TPS, mix_factor_ref=1.3)
    assert t["small"] == pytest.approx(TPS * 1.3)
    old = simulate.bucket_tps(cfg, TPS, {"levels": [_level()]})        # older real run: blended at this mix
    assert old["small"] == pytest.approx(TPS * f_now)
    b = costmodel.buckets(cfg)
    blended = 1 / sum(b[k]["token_share"] / old[k] for k in ("small", "medium", "large"))
    assert blended == pytest.approx(TPS)
    # No long prompts: unchanged.
    assert simulate.bucket_tps(blog[0], TPS)["small"] == pytest.approx(TPS)


# analyze.py records the mix -------------------------------------------------------------------------------------------

def test_analyze_records_mix_factor(tmp_path, blog):
    cfg = _mix(blog[0], 3, 1)
    (tmp_path / "endpoint.json").write_text(json.dumps({"machine_type": "g4-standard-48",
                                                        "accelerator_type": "NVIDIA_RTX_PRO_6000",
                                                        "accelerator_count": 1}))
    (tmp_path / "load").mkdir()
    recs = [{"status": 200, "prompt_tokens": 9000, "completion_tokens": 950, "latency": 30.0, "ttft": 2.0,
             "start": 0.0 + i, "end": 30.0 + i} for i in range(8)]
    (tmp_path / "load" / "c8.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    m = analyze.analyze(cfg, tmp_path)
    assert m["mix_factor"] == pytest.approx(costmodel.mix_factor(cfg), rel=1e-6)
    assert json.loads((tmp_path / "measured.json").read_text())["mix_factor"] == m["mix_factor"]
    assert costmodel.ref_mix_factor(blog[0], m) == m["mix_factor"]   # later evaluated at a short-only mix
    # Measured at the 3/1 mix, evaluated without long prompts: fewer chips than at the measured mix.
    per = m["input_tps_per_gpu"]
    prices = blog[1]["prices"]
    at_mix = costmodel.scenario(cfg, per, prices, "cud_3y", 1, mix_factor_ref=m["mix_factor"])
    short = costmodel.scenario(blog[0], per, prices, "cud_3y", 1, mix_factor_ref=m["mix_factor"])
    assert short["gpus"] < at_mix["gpus"]


# Report / dashboard / server ------------------------------------------------------------------------------------------

def test_report_and_dashboard_use_ref_mix_factor(blog):
    cfg, ch = blog
    c = _mix(cfg, 3, 1)
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    est = {"endpoint": ep, "levels": [], "estimated": True, "input_tps_per_gpu": TPS, "throughput_source": "override"}
    real = {"endpoint": ep, "levels": [_level()], "peak": _level(), "input_tps_per_gpu": TPS,
            "mix_factor": costmodel.mix_factor(c)}
    d_est = dashboard.build_data(c, est, ch["prices"])
    d_real = dashboard.build_data(c, real, ch["prices"])
    assert d_est["mixFactorRef"] == 1.0
    assert d_real["mixFactorRef"] == pytest.approx(costmodel.mix_factor(c))
    g_est = d_est["resilienceSummary"]["routing_options"]["all_on_fleet"]["gpus"]
    g_real = d_real["resilienceSummary"]["routing_options"]["all_on_fleet"]["gpus"]
    assert g_real == 12 and g_est > g_real
    md = report.render(c, {"candidates": []}, est, ch["prices"])
    assert re.search(r"This mix costs \*\*\d+\.\d\d×\*\*", md)
    assert "This mix costs" not in report.render(cfg, {"candidates": []}, est, ch["prices"])


@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    server.build_example_session(runs / "blog-example")
    return TestClient(server.app)


def test_example_session_numbers(client):
    d = client.get("/api/sessions/blog-example").json()
    assert d["dashboard_data"]["mixFactorRef"] == 1.0
    assert d["config"]["pricing"]["tier"] == "vertex_cud_1y" and d["config"]["traffic"]["preset"] == "us_core_hours"
    assert d["dashboard_data"]["defaults"]["pricing"] == "vertex_cud_1y"
    md = d["report_md"]
    # MaaS (15 req/s) can't serve the 28 req/s peak: the verdict compares against MaaS up to its limit + Flash-Lite.
    assert "MaaS can't serve your peak: limit 15 req/s (planned at 11.5 req/s per hour to absorb bursts) vs 28.0 req/s at peak" in md
    # The saved pricing option (1-yr reservation) drives the headline, and self-hosting loses.
    assert "Your pricing option: **Agent Platform + reservation, 1-yr** — **$36K/month** with 12 GPUs" in md
    assert "**112% of that baseline**" in md and "Stay on MaaS up to its limit" in md
    row = next(line for line in md.splitlines() if line.startswith("| Agent Platform + reservation, 1-yr |"))
    assert "| 12 | $36K | 112% |" in row
    assert "assumption for this illustrative example" in d["dashboard_data"]["throughput"]["label"]


def test_measured_mix_factor_flows_through_server(client):
    cfg = config.normalize(json.loads(json.dumps({**yaml.safe_load((ROOT / "blog/example/workload.yaml").read_text())})))
    cfg["requests"]["pct_over_32k"], cfg["requests"]["pct_over_128k"] = 3, 1
    raw = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    client.post("/api/sessions/mx", json={"config": raw})
    ch = planner.plan(config.normalize(raw))["chosen"]
    f = costmodel.mix_factor(config.normalize(raw))
    real = {"endpoint": {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
            "levels": [_level(TPS)], "peak": _level(TPS), "input_tps_per_gpu": TPS, "measured_out_ratio": 0.13,
            "within_budget": True, "mix_factor": f}
    (server.RUNS_DIR / "mx" / "measured.json").write_text(json.dumps(real))
    d = client.post("/api/sessions/mx/estimate", json={}).json()
    assert d["throughput"]["mix_factor"] == pytest.approx(f)
    assert d["dashboard_data"]["mixFactorRef"] == pytest.approx(f)
    assert d["dashboard_data"]["resilienceSummary"]["routing_options"]["all_on_fleet"]["gpus"] == 12


def test_model_comparison_calibration(client):
    # Example session: override 9,000 vs the RTX PRO 6000 spec estimate -> k applied to the other models.
    r = client.post("/api/recommend/model", json={"session": "blog-example"}).json()
    ex = server._with_defaults(server._session_Config("blog-example", server.RUNS_DIR / "blog-example"))
    ch = planner.plan(ex)["chosen"]
    k = TPS / planner.estimate_tps_per_chip(ex, ch["accelerator_type"], ch["accelerator_count"])
    assert r["calibration_k"] == pytest.approx(k, rel=1e-3) and k != 1.0
    others = [m for m in r["recommendations"] if not m.get("is_selected")]
    assert others and all(m["est_throughput_basis"] == server.CALIBRATED_BASIS for m in others)
    # A session on the spec estimate: k = 1 and raw spec estimates.
    raw = yaml.safe_load((ROOT / "blog/example/workload.yaml").read_text())
    client.post("/api/sessions/spec/estimate", json={"config": raw})
    r2 = client.post("/api/recommend/model", json={"session": "spec"}).json()
    assert r2["calibration_k"] == 1.0
    others2 = [m for m in r2["recommendations"] if not m.get("is_selected")]
    assert all(m["est_throughput_basis"] == server.SPEC_BASIS for m in others2)
    # Calibrated alternatives use spec x k.
    by_id = {m["id"]: m for m in r["recommendations"]}
    for m in others2:
        if m.get("est_input_tps_per_gpu") and by_id[m["id"]].get("est_input_tps_per_gpu"):
            assert by_id[m["id"]]["est_input_tps_per_gpu"] == pytest.approx(m["est_input_tps_per_gpu"] * k, abs=1)


# Python <-> model.js parity -------------------------------------------------------------------------------------------

@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_parity(blog):
    cfg, ch = blog
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    cases, expected = [], []
    for o32, o128 in MIXES:
        c = _mix(cfg, o32, o128)
        for f_ref in (1.0, 1.35):
            src = ({"levels": [_level()], "peak": _level(), "mix_factor": f_ref} if f_ref != 1.0
                   else {"levels": [], "estimated": True, "throughput_source": "override"})
            data = dashboard.build_data(c, {"endpoint": ep, "input_tps_per_gpu": TPS, **src}, ch["prices"])
            assert data["mixFactorRef"] == pytest.approx(f_ref)
            for tier in ("cud_3y", "vertex_cud_3y"):
                for route, target in ((False, None), (True, "open_maas"), (True, "gemini_flash_lite")):
                    py = costmodel.scenario(c, TPS, ch["prices"], tier, data["replica"], route_long=route,
                                            route_long_target=target, gemini_catalog=data["gemini"],
                                            mix_factor_ref=f_ref)
                    cases.append({"data": data, "p": {**data["defaults"], "pricing": tier, "routeLong": route,
                                                      "routeLongTarget": target or "gemini_flash_lite",
                                                      "peakOverflowTarget": "open_maas", "resilience": 1}})
                    expected.append((py.get("gpus"), py.get("total"), route and target != "open_maas",
                                     py["baseline"]["total"], py["maas_only_feasible"], py["maas_peak_rps"]))
    script = f"""
const M = require({json.dumps(str(MODEL_JS))});
const cases = {json.dumps(cases, default=str)};
console.log(JSON.stringify(cases.map(c => {{ const r = M.scenario(c.data, c.p);
  return [r.gpus ?? null, r.total ?? null, r.baseline.total, r.maasOnlyFeasible, r.maasPeakRps]; }})));
"""
    out = subprocess.run([NODE, "-"], input=script, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert len(got) == len(expected) == 36
    for (gpus, total, gemini_route, base, feasible, peak_rps), (jg, jt, jb, jf, jp) in zip(expected, got):
        assert jg == gpus
        assert jb == pytest.approx(base, rel=1e-6) and jf == feasible and jp == pytest.approx(peak_rps, rel=1e-6)
        assert jt == pytest.approx(total, rel=1e-6)   # Gemini routes too: model.js has the long-context tier
    # All-on-GPU fleet in model.js grows with the long-prompt share at f_ref = 1.
    all_gpu = [g for (g, *_), c in zip(got, cases)
               if not c["p"]["routeLong"] and c["p"]["pricing"] == "cud_3y" and c["data"]["mixFactorRef"] == 1.0]
    assert all_gpu[0] == 12 and all_gpu[0] < all_gpu[1] < all_gpu[2]
