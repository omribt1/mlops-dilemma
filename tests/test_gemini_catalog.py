"""Live Gemini catalog (gemini.py): Billing Catalog parsing, Model Garden join, role aliases, config validation, cost
engines with a pinned (non-alias) model and the long-context tier, Python <-> model.js parity, server payloads.

All Billing / Model Garden responses are mocks: made-up prices, two new codename models (Gemini 4 Carbon and Hydrogen
next to Argon) and deprecated ones (Gemini 2.0 Flash no longer listed, Gemini 3.6 Flash listed as DEPRECATED)."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import catalog, config, costmodel, dashboard, gemini, planner, pricing, report, server

ROOT = Path(__file__).resolve().parents[1]
MODEL_JS = ROOT / "src" / "aiops_dilemma" / "assets" / "model.js"
NODE = shutil.which("node")
TPS = 9000.0


def _sku(desc, nanos, usage="OnDemand", unit="count"):
    return {"description": desc, "category": {"usageType": usage}, "serviceRegions": ["global"],
            "pricingInfo": [{"pricingExpression": {"usageUnit": unit, "tieredRates": [
                {"unitPrice": {"units": str(int(nanos // 1e9)), "nanos": int(nanos % 1e9)}}]}}]}


def _text(prefix, i, o, c=None, suffix="Predictions", online_word=""):
    w = f" {online_word}" if online_word else ""
    out = [_sku(f"{prefix} Text Input -{w} {suffix}", i), _sku(f"{prefix} Text Output -{w} {suffix}", o)]
    if c is not None:
        out.append(_sku(f"{prefix} Text Input Caching{w}", c))
    return out


SKUS = [
    *_text("Gemini 3.5 Flash Lite Global", 300, 2500, 30),
    _sku("Gemini 3.5 Flash Lite Global Text Input - Batch Predictions", 150),
    _sku("Gemini 3.5 Flash Lite Global Text Output - Batch Predictions", 1250),
    _sku("Gemini 3.5 Flash Lite Regional Text Input - Predictions", 330),
    _sku("Gemini 3.5 Flash Lite Regional Text Output - Predictions", 2750),
    _sku("Gemini 3.5 Flash Lite Global Text Input - Deferred Predictions", 1),
    *_text("Gemini 3.1 Flash Lite Global", 250, 1500, 25),
    *_text("Gemini 3.1 Flash Lite Image Global", 9, 9),            # another product
    *_text("Gemini 3.8 Flash Global", 1500, 7500, 150),
    *_text("Gemini 3.8 Flash Cyber Global", 1, 1),                  # variant
    *_text("Gemini 3.6 Flash Global", 1500, 7500, 150),             # Model Garden: DEPRECATED
    *_text("Gemini 3.8 Live", 750, 4500),                           # another product
    *_text("Gemini 2.0 Flash", 150, 600),                           # no longer listed -> retired
    *_text("Gemini 2.5 Pro", 1250, 10000, 125),
    _sku("Gemini 2.5 Pro Text Input (Long) - Predictions", 2500),
    _sku("Gemini 2.5 Pro Text Output (Long) - Predictions", 15000),
    _sku("Gemini 2.5 Pro Thinking Text Output - Predictions", 1),
    *_text("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image -", 2000, 12000, 200),
    _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input (Long) - Predictions", 4000),
    _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Output (Long) - Predictions", 18000),
    _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input Caching (Long)", 400),
    _sku("Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input Priority - Predictions", 3600),
    *_text("Gemini 4 Argon Global", 4000, 20000, 200, suffix="Online Predictions"),
    *_text("Gemini 4 Argon Cyber Permissive Global", 1, 1, suffix="Online Predictions"),
    _sku("CodeMender Gemini 4 Argon Global Text Input", 2),
    *_text("Gemini 4 Carbon Global", 4000, 20000, 200, suffix="Online Predictions"),       # new
    *_text("Gemini 4 Hydrogen Global", 1500, 7500, 75, suffix="Online Predictions"),       # new, cheaper
    _sku("Gemini 4 Hydrogen Global Text Input - Offpeak Predictions", 1),
    _sku("Gemini 4 Hydrogen Regional Text Input - Online Predictions", 1650),
    _sku("Gemini 4 Hydrogen Regional Text Output - Online Predictions", 8250),
    _sku("Gemini 3.9 Flash Global Text Input - Predictions", 1, usage="Commit1Yr"),          # not on demand
]

GARDEN = [{"name": f"publishers/google/models/{mid}", "versionId": "default", "launchStage": st} for mid, st in [
    ("gemini-2.5-pro", "GA"), ("gemini-3.1-pro-preview", "PUBLIC_PREVIEW"), ("gemini-3-pro-preview", "PUBLIC_PREVIEW"),
    ("gemini-3.1-flash-lite", "GA"), ("gemini-3.1-flash-lite-image", "GA"), ("gemini-3.5-flash-lite", "GA"),
    ("gemini-3.6-flash", "DEPRECATED"), ("gemini-3.8-flash", "GA"), ("gemini-3.8-live", "GA"),
    ("gemini-4-argon", "GA"), ("gemma-4-26b-a4b-it", "GA"), ("gemini-4-carbon", "GA"), ("gemini-4-hydrogen", "GA"),
]]


@pytest.fixture(scope="module")
def cat():
    return gemini.build(SKUS, GARDEN, {})


def _by_id(c):
    return {m["id"]: m for m in c["models"]}


# Discovery ------------------------------------------------------------------------------------------------------------

def test_parse_billing_models_prices_and_tiers():
    by = {m["id"]: m for m in gemini.parse_billing(SKUS)}
    assert set(by) == {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash",
                       "gemini-2.0-flash", "gemini-2.5-pro", "gemini-3-pro", "gemini-3.1-pro", "gemini-4-argon",
                       "gemini-4-carbon", "gemini-4-hydrogen"}
    fl = by["gemini-3.5-flash-lite"]
    assert (fl["name"], fl["family"], fl["version"]) == ("Gemini 3.5 Flash-Lite", "flash-lite", "3.5")
    assert (fl["input_per_m"], fl["output_per_m"], fl["cached_per_m"]) == (0.3, 2.5, 0.03)
    assert fl["batch"] == {"input_per_m": 0.15, "output_per_m": 1.25, "scope": "global"}
    assert fl["regional"] == {"input_per_m": 0.33, "output_per_m": 2.75} and fl["long_context"] is None
    pro = by["gemini-3.1-pro"]
    assert pro["long_context"] == {"threshold": 200000, "input_per_m": 4.0, "output_per_m": 18.0, "cached_per_m": 0.4}
    assert by["gemini-3-pro"]["input_per_m"] == 2.0          # "3.0 Pro" -> Gemini 3 Pro, sharing the SKU
    assert by["gemini-2.5-pro"]["long_context"]["input_per_m"] == 2.5
    assert by["gemini-2.5-pro"]["long_context"]["cached_per_m"] == pytest.approx(0.25)   # 0.125 x 2.5 / 1.25
    assert by["gemini-2.0-flash"]["name"] == "Gemini 2.0 Flash" and by["gemini-2.0-flash"]["cached_estimated"]
    h = by["gemini-4-hydrogen"]
    assert (h["name"], h["family"], h["input_per_m"], h["output_per_m"]) == ("Gemini 4 Hydrogen", "hydrogen", 1.5, 7.5)
    assert h["regional"] == {"input_per_m": 1.65, "output_per_m": 8.25}


def test_join_stages_and_deprecations(cat):
    by = _by_id(cat)
    assert by["gemini-3.5-flash-lite"]["stage"] == "ga" and by["gemini-3.5-flash-lite"]["model_garden_id"] == "gemini-3.5-flash-lite"
    assert by["gemini-3.1-pro"]["stage"] == "preview" and by["gemini-3.1-pro"]["model_garden_id"] == "gemini-3.1-pro-preview"
    assert by["gemini-3.6-flash"]["deprecated"] and by["gemini-3.6-flash"]["stage"] == "deprecated"   # listed DEPRECATED
    assert by["gemini-2.0-flash"]["deprecated"]          # priced, not listed, a newer Flash is listed -> retired
    assert by["gemini-4-carbon"]["stage"] == "ga" and by["gemini-4-carbon"]["group"] == "Gemini 4"
    assert cat["meta"]["source"] == "billing-catalog" and cat["meta"]["model_garden"] is True
    # newest first within family; tier families before codename families
    ids = [m["id"] for m in cat["models"]]
    assert ids.index("gemini-3.5-flash-lite") < ids.index("gemini-3.1-flash-lite") < ids.index("gemini-3.8-flash")
    assert ids.index("gemini-3.8-flash") < ids.index("gemini-3.6-flash") < ids.index("gemini-2.0-flash")
    assert ids.index("gemini-2.5-pro") < ids.index("gemini-4-argon")


def test_unlisted_new_model_and_no_garden():
    skus = SKUS + _text("Gemini 4 Neon Global", 3000, 15000, suffix="Online Predictions")
    by = _by_id(gemini.build(skus, GARDEN, {}))
    assert by["gemini-4-neon"]["stage"] == "unlisted" and not by["gemini-4-neon"]["deprecated"]
    c = gemini.build(SKUS, None, {})                      # Model Garden unreachable: no stages, nothing hidden
    assert all(m["stage"] is None and not m["deprecated"] for m in c["models"])
    assert c["meta"]["model_garden"] is False


# Aliases ----------------------------------------------------------------------------------------------------------------

def test_role_aliases_resolve_dynamically(cat):
    a = {k: v["model_id"] for k, v in cat["aliases"].items()}
    assert a == {"gemini_flash_lite": "gemini-3.5-flash-lite",   # newest GA Flash-Lite
                 "gemini_flash": "gemini-3.8-flash",             # newest GA Flash
                 "gemini_pro": "gemini-3.1-pro",                 # preview of a newer generation beats GA 2.5 Pro
                 "gemini_4": "gemini-4-carbon"}                  # flagship price tier, listed after Argon
    assert cat["aliases"]["gemini_4"]["label"] == "Gemini 4 flagship"
    # first-seen dates outrank listing order; a cheaper model never becomes the flagship
    c2 = gemini.build(SKUS, GARDEN, {"gemini-4-argon": "2026-12-01", "gemini-4-carbon": "2026-11-01",
                                     "gemini-4-hydrogen": "2026-12-05"})
    assert c2["aliases"]["gemini_4"]["model_id"] == "gemini-4-argon"
    # a deprecated newest Flash never counts; without GA Pro of the newest generation the preview wins only if newer
    skus = SKUS + _text("Gemini 3.9 Flash Global", 1, 1, 1)
    garden = GARDEN + [{"name": "publishers/google/models/gemini-3.9-flash", "launchStage": "DEPRECATED"}]
    assert gemini.build(skus, garden, {})["aliases"]["gemini_flash"]["model_id"] == "gemini-3.8-flash"
    garden2 = GARDEN + [{"name": "publishers/google/models/gemini-3.9-flash-preview", "launchStage": "PUBLIC_PREVIEW"}]
    assert gemini.build(skus, garden2, {})["aliases"]["gemini_flash"]["model_id"] == "gemini-3.8-flash"
    # a Gemini 5 generation takes over the flagship alias
    c5 = gemini.build(SKUS + _text("Gemini 5 Boron Global", 5000, 25000, suffix="Online Predictions"), GARDEN, {})
    assert c5["aliases"]["gemini_4"]["model_id"] == "gemini-5-boron" and c5["aliases"]["gemini_4"]["label"] == "Gemini 5 flagship"


def test_fallback_catalog_keeps_blog_prices():
    c = gemini.build(None, None)
    m = gemini.engine_map(c)
    assert c["meta"]["source"] == "fallback"
    lite = m["gemini_flash_lite"]
    assert (lite["input_per_m"], lite["output_per_m"], lite["cached_per_m"]) == (0.30, 2.50, 0.03)
    assert lite["model_id"] == "gemini-3.5-flash-lite" and lite["source"] == "fallback" and lite["alias"]
    assert m["gemini_4"]["name"] == "Gemini 4 Carbon" and m["gemini-4-carbon"]["input_per_m"] == 4.0
    assert m["gemini_pro"]["long_context"]["threshold"] == 200000
    # tests never see live data
    assert pricing.gemini_prices()["gemini_flash_lite"]["source"] == "fallback"


def test_engine_map_and_models_list(cat):
    m = gemini.engine_map(cat)
    assert list(m)[:4] == ["gemini_flash_lite", "gemini_flash", "gemini_pro", "gemini_4"]
    assert m["gemini_4"]["model_id"] == "gemini-4-carbon" and m["gemini_4"]["name"] == "Gemini 4 Carbon"
    assert m["gemini-4-hydrogen"]["input_per_m"] == 1.5 and not m["gemini-4-hydrogen"].get("alias")
    assert m["gemini-3.1-pro"]["long_input_per_m"] == 4.0 and m["gemini-3.8-flash"]["long_input_per_m"] == 1.5
    lst = gemini.models_list(cat)
    carbon = next(x for x in lst if x["id"] == "gemini-4-carbon")
    assert carbon["roles"] == ["gemini_4"] and carbon["stage"] == "ga" and "input_per_m" in carbon
    g4 = pricing.parse_gemini4_skus(SKUS)     # no listing order without Model Garden: either flagship-priced model
    assert g4["name"] in ("Gemini 4 Argon", "Gemini 4 Carbon") and g4["output_per_m"] == 20.0
    assert pricing.parse_gemini_skus(SKUS)["gemini_flash"]["version"] == "3.8"


def test_alias_for_id():
    assert gemini.alias_for_id("gemini-3.6-flash") == "gemini_flash"
    assert gemini.alias_for_id("gemini-3.1-flash-lite") == "gemini_flash_lite"
    assert gemini.alias_for_id("gemini-3-pro") == "gemini_pro"
    assert gemini.alias_for_id("gemini-4-hydrogen") == "gemini_4"
    assert gemini.alias_for_id("gemini-2.0-nano") == "gemini_flash_lite"
    assert gemini.alias_for_id("gemini_pro") == "gemini_pro"


# Cache ----------------------------------------------------------------------------------------------------------------

def test_get_caches_records_first_seen_and_refreshes_in_background(tmp_path, monkeypatch):
    monkeypatch.setenv("DILEMMA_NO_LIVE_METADATA", "0")
    monkeypatch.setenv("DILEMMA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(gemini, "_MEM", {})
    calls = {"skus": 0, "garden": 0, "bg": 0}

    def fake_skus(service, project, max_age=None):
        calls["skus"] += 1
        assert service == "vertex" and max_age == gemini.TTL_S
        return SKUS

    def fake_garden(project, region="us-central1", timeout=60):
        calls["garden"] += 1
        return GARDEN

    monkeypatch.setattr(pricing, "skus", fake_skus)
    monkeypatch.setattr(pricing, "_quota_project", lambda p: "my-gcp-project")
    monkeypatch.setattr(gemini, "fetch_garden", fake_garden)
    c = gemini.get("my-gcp-project")
    assert c["meta"]["source"] == "billing-catalog" and calls == {"skus": 1, "garden": 1, "bg": 0}
    assert (tmp_path / "gemini_catalog.json").exists()
    seen = json.loads((tmp_path / "gemini_first_seen.json").read_text())
    assert "gemini-4-carbon" in seen and _by_id(c)["gemini-4-carbon"]["first_seen"] == seen["gemini-4-carbon"]
    gemini.get("my-gcp-project")                                  # memory
    assert calls["skus"] == 1
    monkeypatch.setattr(gemini, "_MEM", {})                       # disk
    assert gemini.get("my-gcp-project")["aliases"]["gemini_4"]["model_id"] == "gemini-4-carbon" and calls["skus"] == 1
    stale = copy.deepcopy(gemini._MEM["cat"])
    stale["meta"]["updated_at"] -= 2 * gemini.TTL_S
    gemini._MEM["cat"] = stale
    monkeypatch.setattr(gemini, "refresh_bg", lambda *a, **k: calls.__setitem__("bg", calls["bg"] + 1))
    assert gemini.get("my-gcp-project") is stale and calls["bg"] == 1   # stale: served, refreshed in the background
    assert set(gemini.offered()) >= {"gemini-4-hydrogen", "gemini-2.0-flash"}


# Config validation ----------------------------------------------------------------------------------------------------

def test_normalize_validates_pinned_models(cat, monkeypatch):
    raw = {"hybrid": {"failover_target": "gemini-4-hydrogen", "peak_overflow_target": "gemini-3.7-flash",
                      "route_long_target": "gemini_pro"}}
    c0 = config.normalize(copy.deepcopy(raw))                     # no live catalog: nothing to check against
    assert c0["hybrid"]["peak_overflow_target"] == "gemini-3.7-flash" and "_gemini_warnings" not in c0
    monkeypatch.setattr(gemini, "cached", lambda: cat)
    c = config.normalize(copy.deepcopy(raw))
    assert c["hybrid"]["failover_target"] == "gemini-4-hydrogen"
    assert c["hybrid"]["peak_overflow_target"] == "gemini_flash" and c["hybrid"]["route_long_target"] == "gemini_pro"
    assert c["_gemini_warnings"] == ["Gemini model 'gemini-3.7-flash' (hybrid.peak_overflow_target) is no longer offered "
                                     "in the Cloud Billing Catalog; using Gemini Flash (latest: Gemini 3.8 Flash)."]
    c2 = config.normalize({"hybrid": {"failover_target": "gemini-2.0-flash"}})
    assert c2["hybrid"]["failover_target"] == "gemini-2.0-flash" and "deprecated" in c2["_gemini_warnings"][0]
    assert "_gemini_warnings" not in config.to_yaml(c2)


# Cost engines -----------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blog():
    cfg = config.normalize(yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text()))
    return cfg, planner.plan(cfg)["chosen"]


def test_costmodel_prices_a_pinned_model(blog, cat):
    cfg, ch = blog
    gem = gemini.engine_map(cat)
    sc = lambda **kw: costmodel.scenario(cfg, TPS, ch["prices"], "vertex_cud_1y", 1, gemini_catalog=gem, **kw)  # noqa: E731
    lite = sc()
    hyd = sc(failover_target="gemini-4-hydrogen")
    assert hyd["failover_target"] == "gemini-4-hydrogen" and hyd["baseline"]["overflow_target"] == "gemini-4-hydrogen"
    assert "Gemini 4 Hydrogen overflow" in hyd["baseline"]["label"]
    assert hyd["baseline"]["total"] > lite["baseline"]["total"]
    peak = sc(peak_overflow_target="gemini-4-hydrogen")
    assert peak["spill_gemini_target"] == "gemini-4-hydrogen" and peak["gemini_allowed_at_peaks"]
    gone = sc(failover_target="gemini-3.7-flash")                 # not in the catalog: priced as its role alias
    assert gone["failover_target"] == "gemini_flash"
    assert gone["baseline"]["total"] == pytest.approx(sc(failover_target="gemini_flash")["baseline"]["total"])


def test_long_context_tier_math():
    assert costmodel.above_threshold(128000, 188000, 200000) == pytest.approx(((248e3**2 - 200e3**2) / (248e3**2 - 128e3**2), 0.4))
    assert costmodel.above_threshold(32000, 69000, 200000) == (0.0, 0.0)
    assert costmodel.above_threshold(128000, 400000, 128000) == (1.0, 1.0)
    assert costmodel.above_threshold(128000, 100000, 200000) == (0.0, 0.0)   # degenerate range
    flat = {"input_per_m": 1.0, "output_per_m": 4.0, "cached_per_m": 0.1}
    tier = {**flat, "long_context": {"threshold": 200000, "input_per_m": 2.0, "output_per_m": 6.0, "cached_per_m": 0.2}}
    b = {"medium": {"token_share": 0.2, "mean": 69000, "cached": 0.02, "out_ratio": 0.01},
         "large": {"token_share": 0.3, "mean": 300000, "cached": 0.005, "out_ratio": 0.003}}
    # one price: equals gemini_rate on each class
    want = (0.2 * costmodel.gemini_rate(flat, 0.01, 0.02) + 0.3 * costmodel.gemini_rate(flat, 0.003, 0.005)) / 0.5
    assert costmodel.gemini_long_rate(flat, b) == pytest.approx(want)
    assert costmodel.gemini_long_rate(tier, b) > costmodel.gemini_long_rate(flat, b)
    assert costmodel.gemini_long_rate(tier, b, classes=("medium",)) == pytest.approx(costmodel.gemini_rate(flat, 0.01, 0.02))
    assert costmodel.gemini_rate(tier, 0.01, 0.0, is_long=True) == pytest.approx((2.0 + 0.01 * 6.0) / 1e6)


def _node(script):
    out = subprocess.run([NODE, "-"], input=script, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_parity_pinned_models_and_long_tier(blog, cat):
    cfg, ch = blog
    gem = gemini.engine_map(cat)
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    variants = [
        ({}, {"failover_target": "gemini-4-hydrogen"}),
        ({}, {"peak_overflow_target": "gemini-4-hydrogen"}),
        ({}, {"peak_overflow_target": "gemini-4-argon", "failover_target": "gemini-3.1-flash-lite"}),
        ({}, {"failover_target": "gemini-3.7-flash"}),                       # gone: its role alias
        ({}, {"fleet_override": 5, "failover_target": "gemini-4-hydrogen"}),
        ({"pct_over_32k": 4, "pct_over_128k": 2}, {"route_long": True, "route_long_target": "gemini-2.5-pro"}),
        ({"pct_over_32k": 4, "pct_over_128k": 2, "large_mean": 300000},
         {"route_long": True, "route_long_target": "gemini-3.1-pro"}),          # long tier on most >128K prompts
        ({"pct_over_32k": 4, "pct_over_128k": 2, "large_mean": 300000},
         {"route_long": True, "route_long_target": "gemini_pro"}),
        ({"pct_over_32k": 4, "pct_over_128k": 2, "large_mean": 300000},
         {"route_long": True, "route_long_target": "gemini-4-hydrogen"}),      # single price
    ]
    cases, expected = [], []
    for req, kw in variants:
        c = copy.deepcopy(cfg)
        c["requests"].update(req)
        c = config.normalize({k: v for k, v in c.items() if not str(k).startswith("_")})
        data = dashboard.build_data(c, {"endpoint": ep, "input_tps_per_gpu": TPS, "levels": [], "estimated": True,
                                        "throughput_source": "override"}, ch["prices"])
        data["gemini"] = gem
        for k in ("liteIn", "liteOut", "flashIn", "flashOut", "proIn", "proOut", "g4In", "g4Out"):
            data["defaults"][k] = None
        py = costmodel.scenario(c, TPS, ch["prices"], "cud_3y", data["replica"], gemini_catalog=gem, resilience=1.0,
                                **{"route_long": False, **kw})
        jp = {"routeLong": kw.get("route_long", False), "routeLongTarget": kw.get("route_long_target", "gemini_flash_lite"),
              "peakOverflowTarget": kw.get("peak_overflow_target", "open_maas"),
              "failoverTarget": kw.get("failover_target", "gemini_flash_lite"), "fleetOverride": kw.get("fleet_override")}
        cases.append({"data": data, "p": {**data["defaults"], "pricing": "cud_3y", "resilience": 1, **jp}})
        expected.append(py)
    got = _node(f"""
const M = require({json.dumps(str(MODEL_JS))});
const cases = {json.dumps(cases, default=str)};
console.log(JSON.stringify(cases.map(c => {{ const r = M.scenario(c.data, c.p);
  return {{gpus: r.gpus ?? null, total: r.total ?? null, base: r.baseline.total, fail: r.failoverTarget,
          spill: r.spillGeminiTarget, long: r.longMaasCost, gem: r.geminiOverflowCost ?? null}}; }})));
""")
    assert len(got) == len(variants)
    for py, js, (req, kw) in zip(expected, got, variants):
        tag = f"{req} {kw}"
        assert js["gpus"] == py["gpus"], tag
        assert js["total"] == pytest.approx(py["total"], rel=1e-6), tag
        assert js["base"] == pytest.approx(py["baseline"]["total"], rel=1e-6), tag
        assert js["fail"] == py["failover_target"] and js["spill"] == py["spill_gemini_target"], tag
        assert js["long"] == pytest.approx(py["long_maas_cost"], rel=1e-6, abs=1e-6), tag
        assert js["gem"] == pytest.approx(py["gemini_overflow_cost"], rel=1e-6, abs=1e-6), tag
    # the long-context tier moves the >32K bill: Pro on 300K prompts costs more than at its base price
    flat = {k: dict(v) for k, v in gem.items()}
    flat["gemini-3.1-pro"]["long_context"] = None
    flat["gemini-3.1-pro"]["long_input_per_m"], flat["gemini-3.1-pro"]["long_output_per_m"] = 2.0, 12.0
    c = copy.deepcopy(cfg)
    c["requests"].update(variants[6][0])
    c = config.normalize({k: v for k, v in c.items() if not str(k).startswith("_")})
    base = costmodel.scenario(c, TPS, ch["prices"], "cud_3y", 1, gemini_catalog=flat, route_long=True,
                              route_long_target="gemini-3.1-pro")
    assert expected[6]["long_maas_cost"] > base["long_maas_cost"] * 1.3


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_model_js_overrides_per_model_and_legacy(cat):
    gem = gemini.engine_map(cat)
    got = _node(f"""
const M = require({json.dumps(str(MODEL_JS))});
const D = {{ gemini: {json.dumps(gem)} }};
const a = M.getGeminiPrices(D, {{}});
const b = M.getGeminiPrices(D, {{ gemOverrides: {{ "gemini-4-hydrogen": {{ in: 3, out: 9 }} }}, liteIn: 0.5 }});
const c = M.getGeminiPrices(D, {{ gemOverrides: {{ "gemini-3.1-pro": {{ in: 4 }} }} }});
console.log(JSON.stringify({{ a: [a["gemini-4-hydrogen"].in, a.gemini_4.modelId, a.gemini_4.name],
  b: [b["gemini-4-hydrogen"].in, b["gemini-4-hydrogen"].out, b["gemini-4-hydrogen"].cached, b.gemini_flash_lite.in,
      b["gemini-3.5-flash-lite"].in, b["gemini-3.1-flash-lite"].in],
  c: [c.gemini_pro.in, c.gemini_pro.long.in, c.gemini_pro.cached],
  keys: [M.geminiKey("gemini-3.7-flash", a), M.geminiKey("open_maas", a), M.aliasForId("gemini-4-neon")] }}));
""")
    assert got["a"] == [1.5, "gemini-4-carbon", "Gemini 4 Carbon"]
    assert got["b"][:3] == [3, 9, pytest.approx(0.15)]            # cached keeps the 5% ratio of the catalog price
    assert got["b"][3:] == [0.5, 0.5, 0.25]                       # liteIn: the alias and the model it resolves to
    assert got["c"] == [4, 8, pytest.approx(0.4)]                 # alias follows its model's override; long tier scales
    assert got["keys"] == ["gemini_flash", None, "gemini_4"]


# Server -----------------------------------------------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    server.build_example_session(runs / "blog-example")
    return TestClient(server.app)


def test_catalog_payload_has_gemini_list_aliases_meta(cat, monkeypatch):
    monkeypatch.setattr(server.modelinfo, "discover_models", lambda *a, **k: {"models": [], "sources": [], "updated_at": 1})
    monkeypatch.setattr(server.auth, "status", lambda: {"project": "my-gcp-project"})
    monkeypatch.setattr(server.hardware, "ensure_fresh", lambda *a, **k: None)
    monkeypatch.setattr(server.gemini, "get", lambda *a, **k: cat)
    c = server._build_catalog("my-gcp-project", "us-central1")
    assert c["gemini_aliases"]["gemini_4"]["model_id"] == "gemini-4-carbon"
    assert {"gemini_flash_lite", "gemini-4-hydrogen"} <= set(c["gemini"])
    ids = [m["id"] for m in c["gemini_models"]]
    assert "gemini-4-hydrogen" in ids and c["gemini_meta"]["source"] == "billing-catalog"
    assert c["maas_presets"]["gemini_4"]["name"] == "Google Gemini 4 Carbon on Agent Platform"
    assert c["format"] == server.CATALOG_FORMAT == 6


def test_gemini_refresh_endpoint_and_session_warning(client, cat, monkeypatch):
    seen = {}

    def fake_get(project=None, region="us-central1", refresh=False):
        seen["refresh"] = refresh
        return cat

    monkeypatch.setattr(server.gemini, "get", fake_get)
    r = client.post("/api/gemini/refresh", json={"region": "us-central1"}).json()
    assert seen["refresh"] is True and r["gemini_meta"]["source"] == "billing-catalog"
    assert any(m["id"] == "gemini-4-carbon" for m in r["gemini_models"])
    # a saved session pinned to a model the live catalog no longer offers: replaced by its alias, with a warning
    raw = yaml.safe_load((ROOT / "blog" / "example" / "workload.yaml").read_text())
    client.post("/api/sessions/pinned", json={"config": raw})
    raw["hybrid"] = {"failover_target": "gemini-3.7-flash"}
    (server.RUNS_DIR / "pinned" / "workload.yaml").write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(gemini, "cached", lambda: cat)
    s = client.get("/api/sessions/pinned").json()
    assert s["config"]["hybrid"]["failover_target"] == "gemini_flash"
    assert s["gemini_warnings"] and "gemini-3.7-flash" in s["gemini_warnings"][0]


def test_report_lists_a_pinned_model(blog, cat, monkeypatch):
    cfg, ch = blog
    gem = gemini.engine_map(cat)
    monkeypatch.setattr(pricing, "gemini_prices", lambda *a, **k: gem)
    c = copy.deepcopy(cfg)
    c["hybrid"]["failover_target"] = "gemini-4-hydrogen"
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    md = report.render(c, {"candidates": []}, {"endpoint": ep, "levels": [], "estimated": True,
                                               "input_tps_per_gpu": TPS, "throughput_source": "override"}, ch["prices"])
    assert "Gemini 4 Hydrogen $1.5/$7.5/$0.075" in md and "Gemini 4 Carbon $4/$20/$0.2" in md
    assert "Gemini 4 Hydrogen overflow" in md


def test_catalog_fallbacks_unchanged_for_tests():
    g = catalog.GEMINI_MODELS
    assert (g["gemini_flash_lite"]["input_per_m"], g["gemini_flash_lite"]["output_per_m"],
            g["gemini_flash_lite"]["cached_per_m"]) == (0.30, 2.50, 0.03)
