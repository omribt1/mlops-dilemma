"""Third review round (independent model review of the methodology): the baseline plans MaaS at limit / headroom like
the fleet does, standby redundancy is priced into fleet sizing, a long tail that doesn't fit the average is reported,
the pre-flight finds the largest prompt, and the synthetic size distribution hits its mean after clipping."""
import json
import math

import numpy as np
import pytest
import yaml

from aiops_dilemma import catalog, config, costmodel, loadtest, planner, traffic

from test_traffic_presets import GEM, ROOT, TPS


@pytest.fixture(scope="module")
def blog():
    cfg = config.normalize(yaml.safe_load((ROOT / "tests" / "fixtures" / "startup-v1.yaml").read_text()))
    return cfg, planner.plan(cfg)["chosen"]


def _sc(cfg, ch, tier="vertex_cud_1y", **kw):
    return costmodel.scenario(cfg, TPS, ch["prices"], tier, 1, gemini_catalog=GEM, **kw)


def test_baseline_plans_maas_at_limit_over_headroom(blog):
    cfg, ch = blog
    r = _sc(cfg, ch)
    assert r["maas_cap_rps"] == 15 and r["maas_planned_rps"] == pytest.approx(15 / 1.3)
    w = traffic.weekly_profile(cfg["traffic"])
    planned_tps = 15 / 1.3 * cfg["requests"]["avg_input_tokens"]
    on_maas = np.minimum(w, planned_tps).sum() * 3600 * 730 / 168
    rate = costmodel.maas_rate(cfg, *costmodel.request_mix(cfg))
    assert r["baseline"]["maas_cost"] == pytest.approx(on_maas * rate, rel=1e-9)
    # feasibility uses the planned rate: 30 req/s (23 planned) can't carry a 28 req/s peak, 40 (30.8) can
    assert not _sc(cfg, ch, maas_rps=30)["maas_only_feasible"]
    assert _sc(cfg, ch, maas_rps=40)["maas_only_feasible"]
    # headroom 1.0 = the nominal limit (the old rule)
    c1 = config.normalize({**{k: v for k, v in cfg.items() if not k.startswith("_")},
                           "constraints": {**cfg["constraints"], "burst_headroom": 1.0}})
    assert _sc(c1, ch)["maas_planned_rps"] == 15


def test_resilience_sizes_the_fleet_at_its_real_chip_cost(blog):
    cfg, ch = blog
    one = _sc(cfg, ch, resilience=1.0)
    zone = _sc(cfg, ch, resilience=1.5)
    # standby chips on top of the serving fleet; the serving fleet is the cheapest at 1.5 x price
    serving = zone["optimal_gpus"]
    assert zone["gpus"] == math.ceil(serving * 1.5)
    assert serving <= one["gpus"]          # dearer chips never favour a bigger serving fleet
    d = traffic.weekly_profile(cfg["traffic"])
    price = ch["prices"]["vertex_cud_1y"]
    rate = costmodel.maas_rate(cfg, *costmodel.request_mix(cfg))
    cap = 15 * cfg["requests"]["avg_input_tokens"]
    best = costmodel.fixed_fleet(d, TPS, 1, price * 1.5, cap, rate, 1.3,
                                 costmodel.gemini_rate(GEM["gemini_flash_lite"], *costmodel.request_mix(cfg)))
    assert best.gpus == serving


def test_long_tail_that_does_not_fit_the_average_is_reported(capsys, tmp_path):
    ok = config.normalize({"requests": {"avg_input_tokens": 5000, "pct_over_32k": 2, "pct_over_128k": 0.5}})
    assert "_mix_warning" not in ok
    bad = config.normalize({"requests": {"avg_input_tokens": 5000, "pct_over_32k": 10, "pct_over_128k": 3}})
    assert "more than fits in avg_input_tokens" in bad["_mix_warning"]
    p = tmp_path / "w.yaml"
    p.write_text("name: w\nmodel: {model_garden_id: x}\nrequests: {avg_input_tokens: 5000, pct_over_32k: 10}\n")
    config.load(str(p))
    assert "more than fits" in capsys.readouterr().err
    # the engine keeps the configured average for requests/s (it doesn't resize requests silently)
    assert costmodel.mean_request_tokens(bad) == 5000


def test_preflight_finds_the_largest_prompt(tmp_path):
    rd = tmp_path
    (rd / "prompts.jsonl").write_text("\n".join(json.dumps({"input_tokens": n, "messages": []})
                                                for n in (3000, 21000, 9000)) + "\n")
    n, why = loadtest.preflight_prompt_tokens({"requests": {"avg_input_tokens": 5000, "pct_over_32k": 0}}, rd)
    assert n == 21000, (n, why)


@pytest.mark.parametrize("target", [2000, 8000, 15000, 24000])
def test_clipped_lognormal_hits_its_mean(target):
    mu = traffic.clipped_lognormal_mu(target, 0.62, 400, 32000)
    x = np.clip(np.random.default_rng(7).lognormal(mu, 0.62, 300_000), 400, 32000)
    assert x.mean() == pytest.approx(target, rel=0.01)


def test_gemini_cached_fallback_is_ten_percent():
    gem = {"input_per_m": 1.0, "output_per_m": 2.0}       # no cached_per_m
    assert costmodel.gemini_rate(gem, 0.0, 1.0) == pytest.approx(0.1 / 1e6)
