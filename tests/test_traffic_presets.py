"""Traffic-shape presets (traffic.preset), the saved pricing option (pricing.tier) and the blog example's numbers."""

import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import catalog, config, costmodel, dashboard, planner, report, server, traffic

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "aiops_dilemma" / "assets" / "app.js"
NODE = shutil.which("node")
TPS = 9000.0
GEM = copy.deepcopy(catalog.GEMINI_MODELS)   # fixed prices: tests don't depend on the live catalog


def _reference_week(name: str) -> np.ndarray:
    """The generator the blog numbers were computed with (spec, reproduced verbatim)."""
    reg = {"us_east": -5, "us_west": -8, "eu": 1, "apac": 8}

    def bump(start, end, h, ramp=1.5):
        def s(x):
            return 1 / (1 + math.exp(-x / (ramp / 3)))
        h = h % 24
        return max(s(h - start) * s(end - h), s(h + 24 - start) * s(end - h - 24), s(h - 24 - start) * s(end - h + 24))

    def week(mix, floor, weekend, hours):
        day = np.array([sum(w * bump(hours[0], hours[1], h + reg[r]) for r, w in mix.items()) for h in range(24)])
        day = day / day.max()
        wkd = floor + (1 - floor) * day
        wke = floor + (1 - floor) * day * weekend
        return np.concatenate([wkd] * 5 + [wke] * 2)

    presets = {
        "us_core_hours": dict(mix={"us_east": .8, "us_west": .2}, floor=.015, weekend=.04, hours=(10, 16.5)),
        "us_europe": dict(mix={"us_east": .35, "us_west": .2, "eu": .45}, floor=.08, weekend=.2, hours=(8.5, 18)),
        "follow_the_sun": dict(mix={"us_east": .3, "us_west": .1, "eu": .3, "apac": .3}, floor=.15, weekend=.35,
                               hours=(8.5, 18)),
        "steady": dict(mix={"us_east": .25, "us_west": .25, "eu": .25, "apac": .25}, floor=.7, weekend=.9,
                       hours=(8.5, 18)),
    }
    return week(**presets[name])


@pytest.fixture(scope="module")
def blog():
    cfg = config.normalize(yaml.safe_load((ROOT / "tests" / "fixtures" / "startup-v1.yaml").read_text()))
    return cfg, planner.plan(cfg)["chosen"]


def _sc(cfg, ch, tier="vertex_cud_1y", **kw):
    return costmodel.scenario(cfg, TPS, ch["prices"], tier, 1, gemini_catalog=GEM, **kw)


# Presets -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(traffic.PRESETS))
def test_preset_week_is_deterministic_and_scaled(name):
    w = traffic.preset_week(name, 30000)
    ref = _reference_week(name)
    assert w.shape == (168,)
    assert np.array_equal(w, traffic.preset_week(name, 30000))
    assert w.mean() == pytest.approx(30000, rel=1e-12)
    assert np.allclose(w, ref * 30000 / ref.mean(), rtol=1e-12, atol=0)
    assert np.allclose(traffic.preset_week(name, 1000) * 30, w, rtol=1e-12)


def test_preset_shapes():
    w = traffic.preset_week("us_core_hours", 30000)
    assert round(w.max() / w.mean(), 1) == 4.7 and round(w.max(), -3) == 140000 and round(w.min(), -2) == 2100
    assert w.argmax() % 24 in range(15, 23)        # US afternoon, UTC
    ratios = [traffic.preset_week(k, 1).max() for k in ("us_core_hours", "us_europe", "follow_the_sun", "steady")]
    assert ratios == sorted(ratios, reverse=True)  # flatter shapes: lower peak for the same volume
    with pytest.raises(ValueError):
        traffic.preset_week("nope", 1000)
    lst = traffic.preset_list()
    assert [p["key"] for p in lst] == list(traffic.PRESETS)
    assert lst[0]["name"] == "US business hours" and lst[0]["peak_to_average"] == pytest.approx(4.66, abs=0.01)
    assert all(p["description"] for p in lst)


def test_preset_precedence_and_monthly_anchor():
    t = {"preset": "us_europe", "average_input_tps": 20000, "peak_input_tps": 90000, "floor_input_tps": 100}
    assert np.allclose(traffic.weekly_profile(t), traffic.preset_week("us_europe", 20000))     # preset > parametric
    custom = list(np.linspace(1, 168, 168))
    assert np.allclose(traffic.weekly_profile({**t, "custom_hourly": custom}), custom)          # custom > preset
    m = traffic.preset_week("us_europe", 20000).mean() * 3600 * 730 * 2
    w = traffic.weekly_profile({**t, "monthly_input_tokens": m})                                 # monthly anchor wins
    assert traffic.monthly_tokens(w) == pytest.approx(m) and w.mean() == pytest.approx(40000)


def test_hourly_csv_beats_preset(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("input_tokens_per_sec\n" + "\n".join(str(1000 + i) for i in range(168)))
    w = traffic.weekly_profile({"hourly_csv": str(p), "preset": "steady", "average_input_tps": 5})
    assert w[0] == 1000 and w[-1] == 1167


def test_normalize_rewrites_peak_average_floor():
    cfg = config.normalize({"traffic": {"preset": "us_core_hours", "average_input_tps": 30000,
                                        "peak_input_tps": 1, "floor_input_tps": 1}})
    t = cfg["traffic"]
    w = traffic.preset_week("us_core_hours", 30000)
    assert t["preset"] == "us_core_hours" and "_warnings" not in cfg
    assert t["peak_input_tps"] == pytest.approx(w.max(), abs=1e-3)
    assert t["floor_input_tps"] == pytest.approx(w.min(), abs=1e-3)
    assert t["average_input_tps"] == pytest.approx(30000)
    again = config.normalize({k: v for k, v in cfg.items() if not k.startswith("_")})      # stable on save/load
    assert again["traffic"]["peak_input_tps"] == t["peak_input_tps"]
    # Monthly anchor: the average follows the total.
    cm = config.normalize({"traffic": {"preset": "steady", "monthly_input_tokens": 26_280_000_000}})
    assert cm["traffic"]["average_input_tps"] == pytest.approx(10000)


@pytest.mark.parametrize("bad", ["weekends_only", 42])
def test_unknown_preset_dropped_with_warning(bad):
    cfg = config.normalize({"traffic": {"preset": bad, "peak_input_tps": 100000, "average_input_tps": 25000,
                                        "floor_input_tps": 2500}})
    assert cfg["traffic"]["preset"] is None
    assert any("preset" in w and "unknown" in w for w in cfg["_warnings"])
    assert cfg["traffic"]["peak_input_tps"] == 100000          # parametric week untouched
    for empty in (None, "", "custom"):
        c = config.normalize({"traffic": {"preset": empty}})
        assert c["traffic"]["preset"] is None and "_warnings" not in c


@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    server.build_example_session(runs / "blog-example")
    return TestClient(server.app)


def test_preview_and_presets_endpoints(client):
    r = client.post("/api/traffic/preview", json={"traffic": {"preset": "follow_the_sun", "average_input_tps": 30000}})
    d = r.json()
    w = traffic.preset_week("follow_the_sun", 30000)
    assert d["ok"] and d["preset"] == "follow_the_sun" and len(d["profile"]) == 168
    assert d["peak_input_tps"] == round(w.max()) and d["floor_input_tps"] == round(w.min())
    assert d["average_input_tps"] == 30000
    bad = client.post("/api/traffic/preview", json={"traffic": {"preset": "bogus", "peak_input_tps": 100000,
                                                                  "average_input_tps": 25000,
                                                                  "floor_input_tps": 2500}}).json()
    assert bad["ok"] and bad["preset"] is None and bad["peak_input_tps"] == 100000 and bad["warnings"]
    lst = client.get("/api/traffic/presets").json()["presets"]
    assert [p["key"] for p in lst] == list(traffic.PRESETS)


def test_session_round_trip_preset_and_pricing_tier(client):
    raw = yaml.safe_load((ROOT / "tests" / "fixtures" / "startup-v1.yaml").read_text())
    raw["traffic"]["preset"] = "steady"
    raw["pricing"]["tier"] = "cud_3y"
    r = client.post("/api/sessions/rt", json={"config": raw})
    assert r.status_code == 200
    on_disk = yaml.safe_load((server.RUNS_DIR / "rt" / "workload.yaml").read_text())
    assert on_disk["traffic"]["preset"] == "steady" and on_disk["pricing"]["tier"] == "cud_3y"
    d = client.get("/api/sessions/rt").json()
    assert d["config"]["pricing"]["tier"] == "cud_3y" and d["dashboard_data"]["defaults"]["pricing"] == "cud_3y"
    assert max(d["dashboard_data"]["profile"]) == pytest.approx(traffic.preset_week("steady", 30000).max(), abs=0.01)
    # Unknown tiers fall back to auto (cheapest).
    assert config.normalize({"pricing": {"tier": "five_year"}})["pricing"]["tier"] == "auto"
    assert config.DEFAULTS["pricing"]["tier"] == "auto"


# Pricing tier in the report -------------------------------------------------------------------------------------------

def test_report_headline_uses_pricing_tier(blog):
    cfg, ch = blog
    ep = {k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")}
    est = {"endpoint": ep, "levels": [], "estimated": True, "input_tps_per_gpu": TPS, "throughput_source": "override"}
    md = report.render(cfg, {"candidates": []}, est, ch["prices"])
    assert "Your pricing option: **Agent Platform + reservation, 1-yr**" in md
    auto = copy.deepcopy(cfg)
    auto["pricing"]["tier"] = "auto"
    md2 = report.render(auto, {"candidates": []}, est, ch["prices"])
    assert "Cheapest option: **Self-managed GKE, 3-yr**" in md2
    # The per-tier table is kept either way.
    for m in (md, md2):
        assert "| Agent Platform, on demand |" in m and "| Self-managed GKE, 3-yr |" in m
    d = dashboard.build_data(auto, est, ch["prices"])
    assert d["defaults"]["pricing"] == "cud_3y"


# The blog example ----------------------------------------------------------------------------------------------------

def test_blog_example_numbers(blog):
    cfg, ch = blog
    w = traffic.weekly_profile(cfg["traffic"])
    assert round(w.max(), -3) == 140000 and round(w.max() / w.mean(), 1) == 4.7 and round(w.min(), -2) == 2100
    assert round(traffic.monthly_tokens(w) / 1e9) == 79
    r = _sc(cfg, ch)
    assert r["gpus"] == 12 and round(r["total"] / 1e2) == 358
    assert r["baseline"]["kind"] == "maas_plus_overflow" and round(r["baseline"]["total"] / 1e2) == 319
    assert round(r["vs_baseline"], 2) == 1.12 and round(r["maas_only"] / 1e2) == 174
    for op, want in ((1.5, 0.90), (2, 0.82), (4, 0.69)):
        assert round(_sc(cfg, ch, off_peak=op)["vs_baseline"], 2) == want, op
    for tier, want in (("vertex_cud_3y", 0.79), ("cud_3y", 0.65), ("cud_1y", 0.95), ("vertex", 1.71)):
        assert round(_sc(cfg, ch, tier)["vs_baseline"], 2) == want, tier
    for preset, want in (("us_europe", 0.82), ("follow_the_sun", 0.75), ("steady", 0.64)):
        c = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
        c["traffic"] = {"preset": preset, "average_input_tps": 30000}
        assert round(_sc(config.normalize(c), ch)["vs_baseline"], 2) == want, preset


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_app_js_preset_week_matches_python():
    src = APP_JS.read_text()
    start = src.index("  const REGION_UTC")
    end = src.index("  // Active preset:")
    script = src[start:end] + """
const out = {};
for (const k of Object.keys(TRAFFIC_PRESETS)) out[k] = presetWeek(k, 30000);
console.log(JSON.stringify(out));
"""
    res = subprocess.run([NODE, "-"], input=script, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    for k in traffic.PRESETS:
        assert np.allclose(got[k], traffic.preset_week(k, 30000), rtol=1e-12, atol=0), k
