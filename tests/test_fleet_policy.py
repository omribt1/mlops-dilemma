"""Fleet policy: one rule for the optimizer and the waterline (costmodel.eval_fleet / fixed_fleet, model.js evalFleet /
fixedFleet), governed by hybrid.peak_overflow_target, plus the "allow Gemini at peaks" verdict hint (Python + model.js)."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from aiops_dilemma import catalog, config, costmodel, dashboard, planner, report

ROOT = Path(__file__).resolve().parents[1]
MODEL_JS = ROOT / "src" / "aiops_dilemma" / "assets" / "model.js"
NODE = shutil.which("node")
TPS = 9000.0
GEM = copy.deepcopy(catalog.GEMINI_MODELS)   # fixed prices
TIER = "vertex_cud_1y"
FLAGS = ("policy_ok", "needs_gemini_at_peaks", "over_overflow_quota", "short_of_headroom")


@pytest.fixture(scope="module")
def blog():
    cfg = config.normalize(yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text()))
    return cfg, planner.plan(cfg)["chosen"]


def _routed(cfg):
    """Blog workload with a few long prompts routed to open-model MaaS and a 10 req/s limit (rtx6000-sample shape)."""
    c = copy.deepcopy(cfg)
    c["requests"].update({"pct_over_32k": 2.0, "pct_over_128k": 0.5, "medium_mean": 60000, "large_mean": 170000})
    c["maas"]["max_rps"] = 10
    c["hybrid"].update({"route_long": True, "route_long_target": "open_maas"})
    return c


def _sc(cfg, ch, **kw):
    return costmodel.scenario(cfg, TPS, ch["prices"], TIER, 1, gemini_catalog=GEM, **kw)


def test_default_policy_flags_on_the_blog_example(blog):
    cfg, ch = blog
    opt = _sc(cfg, ch)
    assert opt["gpus"] == 12 and round(opt["total"] / 1e2) == 358 and round(opt["vs_baseline"], 2) == 1.28   # unchanged
    assert opt["min_policy_gpus"] == 12 and opt["headroom_gpus"] == 12 and opt["policy_ok"]
    assert not opt["gemini_allowed_at_peaks"] and opt["spill_gemini_target"] == "gemini_flash_lite"
    # 11 GPUs: no spill beyond the MaaS cap, but below the 1.3x headroom bound
    r11 = _sc(cfg, ch, fleet_override=11)
    assert {k: r11[k] for k in FLAGS} == {"policy_ok": False, "needs_gemini_at_peaks": False,
                                          "over_overflow_quota": False, "short_of_headroom": True}
    # 5 GPUs: cheaper on paper because the spill beyond the MaaS cap is priced at Flash-Lite, but not allowed
    r5 = _sc(cfg, ch, fleet_override=5)
    assert r5["total"] < opt["total"] and r5["gemini_overflow_cost"] > 0
    assert r5["needs_gemini_at_peaks"] and r5["short_of_headroom"] and not r5["policy_ok"]
    assert r5["optimal_gpus"] == 12 and r5["optimal_total"] == pytest.approx(opt["total"])
    assert _sc(cfg, ch, fleet_override=12)["policy_ok"]


def test_verdict_hint_on_the_blog_example(blog):
    cfg, ch = blog
    h = _sc(cfg, ch)["peak_gemini_hint"]
    assert h["target"] == "gemini_flash_lite" and h["gpus"] == 5 and round(h["vs_baseline"], 2) == 1.04
    alt = _sc(cfg, ch, peak_overflow_target="gemini_flash_lite")
    assert alt["gpus"] == 5 and round(alt["gemini_overflow_cost"] / 1e2) == 89 and "peak_gemini_hint" not in alt
    assert h["total"] == pytest.approx(alt["total"]) and h["saving"] == pytest.approx(_sc(cfg, ch)["total"] - alt["total"])
    assert h["text"] == ("Allowing Gemini 3.5 Flash-Lite overflow at peaks would cut it to 5 GPUs and save about "
                         f"${h['saving'] / 1000:,.1f}K a month.")
    # No hint when allowing Gemini saves under 3% (a fleet that never spills: plenty of MaaS headroom)
    c = copy.deepcopy(cfg)
    c["maas"]["max_rps"] = 1000
    assert _sc(c, ch).get("peak_gemini_hint") is None


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("target", config.OVERFLOW_TARGETS)
def test_optimum_is_the_cheapest_policy_ok_slider_point(blog, target, routed):
    cfg, ch = blog
    c = _routed(cfg) if routed else cfg
    opt = _sc(c, ch, peak_overflow_target=target)
    pts = [_sc(c, ch, peak_overflow_target=target, fleet_override=n) for n in range(1, 2 * opt["gpus"] + 8)]
    ok = [r for r in pts if r["policy_ok"]]
    best = min(ok, key=lambda r: r["total"])
    assert best["gpus"] == opt["gpus"] and best["total"] == pytest.approx(opt["total"])
    assert min(r["gpus"] for r in ok) == opt["min_policy_gpus"]
    assert all(r["policy_ok"] == (r["gpus"] >= opt["min_policy_gpus"]) for r in pts)   # monotone
    if target != "open_maas":   # Gemini allowed at peaks: the slider's cheapest point is the optimum
        assert min(pts, key=lambda r: r["total"])["gpus"] == opt["gpus"]
        assert not any(r["needs_gemini_at_peaks"] for r in pts)


def test_report_has_the_hint_line(blog):
    cfg, ch = blog
    est = {"endpoint": {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
           "input_tps_per_gpu": TPS, "levels": [], "estimated": True, "throughput_source": "override"}
    c = copy.deepcopy(cfg)
    c["pricing"] = {"tier": TIER, "overrides": {}}
    import aiops_dilemma.pricing as pricing_mod
    orig = pricing_mod.gemini_prices
    pricing_mod.gemini_prices = lambda *a, **k: GEM
    try:
        md = report.render(c, {"candidates": []}, est, ch["prices"])
        c["hybrid"]["peak_overflow_target"] = "gemini_flash_lite"
        md2 = report.render(c, {"candidates": []}, est, ch["prices"])
    finally:
        pricing_mod.gemini_prices = orig
    assert "Allowing Gemini 3.5 Flash-Lite overflow at peaks would cut it to 5 GPUs" in md
    assert "peak_overflow_target: gemini_flash_lite" in md
    assert "Allowing" not in md2


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_parity_fleet_policy(blog):
    cfg, ch = blog
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    variants = [   # (routed, python kwargs, model.js params)
        (False, {}, {}),
        (False, {"fleet_override": 11}, {"fleetOverride": 11}),             # below the headroom bound only
        (False, {"fleet_override": 5}, {"fleetOverride": 5}),               # needs Gemini at peaks
        (False, {"fleet_override": 13}, {"fleetOverride": 13}),
        (False, {"peak_overflow_target": "gemini_flash_lite"}, {"peakOverflowTarget": "gemini_flash_lite"}),
        (False, {"peak_overflow_target": "gemini_flash_lite", "fleet_override": 3},
         {"peakOverflowTarget": "gemini_flash_lite", "fleetOverride": 3}),
        (False, {"peak_overflow_target": "gemini_pro", "fleet_override": 1},
         {"peakOverflowTarget": "gemini_pro", "fleetOverride": 1}),
        (False, {"failover_target": "gemini_flash"}, {"failoverTarget": "gemini_flash"}),   # hint names Flash
        (True, {}, {}),
        (True, {"fleet_override": 6}, {"fleetOverride": 6}),
        (True, {"fleet_override": 9}, {"fleetOverride": 9}),
        (True, {"peak_overflow_target": "gemini_flash_lite"}, {"peakOverflowTarget": "gemini_flash_lite"}),
        (True, {"peak_overflow_target": "gemini_flash_lite", "fleet_override": 4},
         {"peakOverflowTarget": "gemini_flash_lite", "fleetOverride": 4}),
    ]
    cases, expected = [], []
    for routed, kw, jp in variants:
        c = _routed(cfg) if routed else copy.deepcopy(cfg)
        data = dashboard.build_data(c, {"endpoint": ep, "input_tps_per_gpu": TPS, "levels": [], "estimated": True,
                                        "throughput_source": "override"}, ch["prices"])
        data["gemini"] = GEM
        for k, (i, o) in {"liteIn": ("gemini_flash_lite", "input_per_m"), "liteOut": ("gemini_flash_lite", "output_per_m"),
                          "flashIn": ("gemini_flash", "input_per_m"), "flashOut": ("gemini_flash", "output_per_m"),
                          "proIn": ("gemini_pro", "input_per_m"), "proOut": ("gemini_pro", "output_per_m"),
                          "g4In": ("gemini_4", "input_per_m"), "g4Out": ("gemini_4", "output_per_m")}.items():
            data["defaults"][k] = GEM[i][o]
        expected.append(_sc(c, ch, resilience=1.0, **kw))
        cases.append({"data": data, "p": {**data["defaults"], "pricing": TIER, "resilience": 1, **jp}})
    script = f"""
const M = require({json.dumps(str(MODEL_JS))});
const cases = {json.dumps(cases, default=str)};
console.log(JSON.stringify(cases.map(c => {{ const r = M.scenario(c.data, c.p);
  return {{gpus: r.gpus ?? null, total: r.total ?? null, optimalGpus: r.optimalGpus, optimalTotal: r.optimalTotal ?? null,
          minPolicy: r.minPolicyGpus ?? null, headroom: r.headroomGpus ?? null, gem: r.geminiOverflowCost ?? null,
          maas: r.maasOverflowCost ?? null, spill: r.spillGeminiTarget, allowed: r.geminiAllowedAtPeaks,
          flags: [r.policyOk, r.needsGeminiAtPeaks, r.overOverflowQuota, r.shortOfHeadroom], hint: r.peakGeminiHint ?? null}}; }})));
"""
    out = subprocess.run([NODE, "-"], input=script, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for py, js, (routed, kw, _) in zip(expected, json.loads(out.stdout), variants):
        tag = f"routed={routed} {kw}"
        assert js["gpus"] == py["gpus"] and js["optimalGpus"] == py["optimal_gpus"], tag
        assert js["total"] == pytest.approx(py["total"], rel=1e-6), tag
        assert js["optimalTotal"] == pytest.approx(py["optimal_total"], rel=1e-6), tag
        assert js["gem"] == pytest.approx(py["gemini_overflow_cost"], rel=1e-6, abs=1e-6), tag
        assert js["maas"] == pytest.approx(py["maas_overflow_cost"], rel=1e-6, abs=1e-6), tag
        assert js["minPolicy"] == py["min_policy_gpus"] and js["headroom"] == py["headroom_gpus"], tag
        assert js["spill"] == py["spill_gemini_target"] and js["allowed"] == py["gemini_allowed_at_peaks"], tag
        assert js["flags"] == [py[k] for k in FLAGS], tag
        ph = py.get("peak_gemini_hint")
        if ph is None:
            assert js["hint"] is None, tag
        else:
            h = js["hint"]
            assert (h["target"], h["name"], h["gpus"], h["text"]) == (ph["target"], ph["name"], ph["gpus"], ph["text"]), tag
            assert h["total"] == pytest.approx(ph["total"], rel=1e-6) and h["saving"] == pytest.approx(ph["saving"], rel=1e-6), tag
