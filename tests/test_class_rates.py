"""Measured per-class throughput (prompts 32K-128K and > 128K): costmodel, simulate, measured.json, model.js parity.

Neutral numbers only (blog example workload with an illustrative long-prompt tail)."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from aiops_dilemma import catalog, config, costmodel, dashboard, planner, simulate

ROOT = Path(__file__).resolve().parents[1]
MODEL_JS = ROOT / "src" / "aiops_dilemma" / "assets" / "model.js"
NODE = shutil.which("node")
SHORT = 12000.0
GEM = copy.deepcopy(catalog.GEMINI_MODELS)
TAIL = {"pct_over_32k": 3.0, "pct_over_128k": 1.0, "medium_mean": 60000, "large_mean": 170000}


@pytest.fixture(scope="module")
def blog():
    raw = yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text())
    raw["requests"].update(TAIL)
    cfg = config.normalize(raw)
    ch = planner.plan(cfg)["chosen"]
    return cfg, ch


def _with(cfg, **tp):
    c = copy.deepcopy(cfg)
    c["throughput"].update(tp)
    return c


def test_defaults_unchanged_without_rates(blog):
    cfg, _ = blog
    pen = costmodel.class_penalties(cfg, SHORT)
    assert (pen["medium"], pen["large"]) == (3.0, 4.0)
    assert pen["source"] == {"medium": "default", "large": "default"}


def test_user_rates_set_penalties_and_grow_the_fleet(blog):
    cfg, ch = blog
    slow = _with(cfg, medium_tps=3000, large_tps=1500)          # 4x and 8x slower than 12K
    pen = costmodel.class_penalties(slow, SHORT)
    assert pen["medium"] == pytest.approx(4.0) and pen["large"] == pytest.approx(8.0)
    a = costmodel.scenario(cfg, SHORT, ch["prices"], "cud_3y", 1, gemini_catalog=GEM, route_long=False)
    b = costmodel.scenario(slow, SHORT, ch["prices"], "cud_3y", 1, gemini_catalog=GEM, route_long=False)
    assert b["mix_factor"] > a["mix_factor"] and b["gpus"] > a["gpus"]
    # Routing the long prompts away makes their speed irrelevant
    ra = costmodel.scenario(cfg, SHORT, ch["prices"], "cud_3y", 1, gemini_catalog=GEM, route_long=True,
                            route_long_target="open_maas")
    rb = costmodel.scenario(slow, SHORT, ch["prices"], "cud_3y", 1, gemini_catalog=GEM, route_long=True,
                            route_long_target="open_maas")
    assert ra["gpus"] == rb["gpus"]


def test_penalty_floor_and_normalize(blog):
    cfg, _ = blog
    fast = _with(cfg, medium_tps=50000)                          # faster than short prompts: floored at 1x
    assert costmodel.class_penalties(fast, SHORT)["medium"] == 1.0
    c = config.normalize({**copy.deepcopy(cfg), "throughput": {"medium_tps": "-5", "large_tps": "2000",
                                                                "short_tps": 1, "source": "measured"}})
    assert c["throughput"] == {"medium_tps": None, "large_tps": 2000.0}
    assert "short_tps" not in config.to_yaml(c)


def test_measured_bucket_tps_is_read_as_blend(blog):
    """A benchmark with per-class rates: its blended figure is not mistaken for a short-prompt rate."""
    cfg, ch = blog
    bt = {"small": SHORT, "medium": 3000.0, "large": 1500.0}
    c2 = costmodel.with_measured_classes(cfg, {"bucket_tps": bt})
    f = costmodel.mix_factor(c2, SHORT)
    measured = {"input_tps_per_gpu": SHORT / f, "bucket_tps": bt, "source": "measured"}
    f_ref = costmodel.ref_mix_factor(cfg, measured)
    assert f_ref == pytest.approx(f)
    via_measured = costmodel.scenario(c2, measured["input_tps_per_gpu"], ch["prices"], "cud_3y", 1,
                                      gemini_catalog=GEM, route_long=False, mix_factor_ref=f_ref)
    via_user = costmodel.scenario(_with(cfg, medium_tps=3000, large_tps=1500), SHORT, ch["prices"], "cud_3y", 1,
                                  gemini_catalog=GEM, route_long=False)
    assert via_measured["gpus"] == via_user["gpus"]
    assert via_measured["total"] == pytest.approx(via_user["total"], rel=1e-9)
    # A user-set rate wins over the benchmark's, per class
    c3 = costmodel.with_measured_classes(_with(cfg, medium_tps=6000), {"bucket_tps": bt})
    assert c3["throughput"]["medium_tps"] == 6000 and c3["throughput"]["large_tps"] == 1500.0


def test_simulate_bucket_rates_follow_user_rates(blog):
    cfg, _ = blog
    tps = simulate.bucket_tps(_with(cfg, medium_tps=3000, large_tps=1500), SHORT)
    assert tps == pytest.approx({"small": SHORT, "medium": 3000.0, "large": 1500.0})
    dflt = simulate.bucket_tps(cfg, SHORT)
    assert dflt["medium"] == pytest.approx(SHORT / 3.0)


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_parity_class_rates(blog):
    cfg, ch = blog
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    bt = {"small": SHORT, "medium": 3000.0, "large": 1500.0}
    variants = [  # (user throughput, measured extras, routed)
        ({}, None, False),
        ({"medium_tps": 3000, "large_tps": 1500}, None, False),
        ({"medium_tps": 5000}, None, False),
        ({}, bt, False),
        ({"medium_tps": 6000}, bt, False),
        ({"medium_tps": 3000, "large_tps": 1500}, None, True),
    ]
    cases, expected = [], []
    for tp, bucket, routed in variants:
        c = _with(cfg, **tp)
        if bucket:
            f = costmodel.mix_factor(costmodel.with_measured_classes(c, {"bucket_tps": bucket}), SHORT)
            measured = {"endpoint": ep, "input_tps_per_gpu": SHORT / f, "bucket_tps": bucket, "levels": [],
                        "estimated": True, "throughput_source": "calibrated"}
        else:
            measured = {"endpoint": ep, "input_tps_per_gpu": SHORT, "levels": [], "estimated": True,
                        "throughput_source": "override"}
        data = dashboard.build_data(c, measured, ch["prices"])
        data["gemini"] = GEM
        for k, (i, o) in {"liteIn": ("gemini_flash_lite", "input_per_m"), "liteOut": ("gemini_flash_lite", "output_per_m"),
                          "flashIn": ("gemini_flash", "input_per_m"), "flashOut": ("gemini_flash", "output_per_m"),
                          "proIn": ("gemini_pro", "input_per_m"), "proOut": ("gemini_pro", "output_per_m"),
                          "g4In": ("gemini_4", "input_per_m"), "g4Out": ("gemini_4", "output_per_m")}.items():
            data["defaults"][k] = GEM[i][o]
        c_eff = costmodel.with_measured_classes(c, measured)
        py = costmodel.scenario(c_eff, measured["input_tps_per_gpu"], ch["prices"], "cud_3y", data["replica"],
                                gemini_catalog=GEM, resilience=1.0, route_long=routed, route_long_target="open_maas",
                                mix_factor_ref=costmodel.ref_mix_factor(c, measured))
        cases.append({"data": data, "p": {**data["defaults"], "pricing": "cud_3y", "resilience": 1,
                                          "routeLong": routed, "routeLongTarget": "open_maas",
                                          "peakOverflowTarget": "open_maas", "failoverTarget": "gemini_flash_lite"}})
        expected.append(py)
    script = f"""
const M = require({json.dumps(str(MODEL_JS))});
const cases = {json.dumps(cases, default=str)};
console.log(JSON.stringify(cases.map(c => {{ const r = M.scenario(c.data, c.p);
  return {{gpus: r.gpus ?? null, total: r.total ?? null, mix: r.buckets.mixFactor, src: r.buckets.penaltySource}}; }})));
"""
    out = subprocess.run([NODE, "-"], input=script, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for py, js, v in zip(expected, json.loads(out.stdout), variants):
        assert js["mix"] == pytest.approx(py["mix_factor"], rel=1e-9), v
        assert js["src"] == py["buckets"]["penalty_source"], v
        assert js["gpus"] == py.get("gpus"), v
        assert js["total"] == pytest.approx(py["total"], rel=1e-6), v


def test_other_hardware_gets_the_slowdown_not_the_rate(blog):
    """Long-prompt rates typed for the selected hardware become slowdowns for the other candidates."""
    cfg, _ = blog
    slow = _with(cfg, medium_tps=3000, large_tps=1500)
    plan = planner.plan(slow)
    sel = {k: plan["chosen"][k] for k in ("accelerator_type", "accelerator_count")}
    tp = {"per_chip": SHORT, "source": "override"}
    out = planner.enrich_candidates(slow, plan["candidates"], gemini_catalog=GEM, throughput=tp, selected=sel)
    base = planner.enrich_candidates(cfg, plan["candidates"], gemini_catalog=GEM, throughput=tp, selected=sel)
    other = next(c for c in out if not c["is_selected"] and c.get("est_input_tps_per_gpu"))
    other_base = next(c for c in base if c["accelerator_type"] == other["accelerator_type"]
                      and c["accelerator_count"] == other["accelerator_count"])
    pen = costmodel.class_penalties({**slow, "throughput": {"medium_penalty": 4.0, "large_penalty": 8.0}})
    assert (pen["medium"], pen["large"]) == (4.0, 8.0)
    assert other["fleet_gpus"] >= other_base["fleet_gpus"]
    expect = planner.fleet_economics({**slow, "throughput": {"medium_penalty": 4.0, "large_penalty": 8.0}},
                                     other_base, gemini_catalog=GEM)
    assert other["fleet_gpus"] == expect["fleet_gpus"]


def test_cached_input_tokens_override(blog):
    """requests.cached_input_tokens (chat / agent workloads) replaces system prompt x hit rate everywhere."""
    from aiops_dilemma.config import cached_input_tokens
    cfg, ch = blog
    r = cfg["requests"]
    assert cached_input_tokens(r) == pytest.approx(r["system_prompt_tokens"] * r["system_prompt_cache_hit"])
    heavy = copy.deepcopy(cfg)
    heavy["requests"]["cached_input_tokens"] = 0.9 * r["avg_input_tokens"]
    assert cached_input_tokens({**heavy["requests"], "cached_input_tokens": 10 ** 9}) == r["avg_input_tokens"]
    a = costmodel.scenario(cfg, SHORT, ch["prices"], "cud_3y", 1, gemini_catalog=GEM)
    b = costmodel.scenario(heavy, SHORT, ch["prices"], "cud_3y", 1, gemini_catalog=GEM)
    assert b["maas_only"] < a["maas_only"]          # cache reads priced at the cached rate
    assert planner.estimate_tps_per_chip(heavy, "NVIDIA_RTX_PRO_6000", 1) > planner.estimate_tps_per_chip(cfg, "NVIDIA_RTX_PRO_6000", 1)


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_parity_cached_input(blog):
    cfg, ch = blog
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    c = copy.deepcopy(cfg)
    c["requests"]["cached_input_tokens"] = 0.8 * c["requests"]["avg_input_tokens"]
    data = dashboard.build_data(c, {"endpoint": ep, "input_tps_per_gpu": SHORT, "levels": [], "estimated": True,
                                    "throughput_source": "override"}, ch["prices"])
    data["gemini"] = GEM
    py = costmodel.scenario(c, SHORT, ch["prices"], "cud_3y", data["replica"], gemini_catalog=GEM, resilience=1.0,
                            route_long=False)
    p = {**data["defaults"], "pricing": "cud_3y", "resilience": 1, "routeLong": False,
         "peakOverflowTarget": "open_maas", "failoverTarget": "gemini_flash_lite"}
    script = f"""
const M = require({json.dumps(str(MODEL_JS))});
const r = M.scenario({json.dumps(data, default=str)}, {json.dumps(p)});
console.log(JSON.stringify({{maasOnly: r.maasOnly, total: r.total ?? null}}));
"""
    out = subprocess.run([NODE, "-"], input=script, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    js = json.loads(out.stdout)
    assert js["maasOnly"] == pytest.approx(py["maas_only"], rel=1e-6)
    assert js["total"] == pytest.approx(py["total"], rel=1e-6)
