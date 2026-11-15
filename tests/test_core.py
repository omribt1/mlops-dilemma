import numpy as np
import pytest

from aiops_dilemma import costmodel, planner, traffic

CFG = {
    "traffic": {"peak_input_tps": 150e3, "average_input_tps": 50e3, "floor_input_tps": 6e3,
                "busy_hours_utc": [13, 20], "weekend_factor": 0.15, "hourly_csv": None},
    "requests": {"avg_input_tokens": 5000, "avg_output_tokens": 600, "system_prompt_tokens": 1500,
                 "system_prompt_cache_hit": 0.9},
    "maas": {"input_per_m": 0.15, "output_per_m": 0.60, "cached_per_m": 0.015, "max_rps": 10},
    "constraints": {"burst_headroom": 1.3, "resilience_multiplier": 1.0},
}
PRICES = {"vertex": 5.85, "vertex_cud_3y": 2.52}


def test_parametric_week_hits_targets():
    w = traffic.parametric(150e3, 50e3, 6e3)
    assert len(w) == 168
    assert w.max() == pytest.approx(150e3)
    assert w.min() == pytest.approx(6e3)
    assert w.mean() == pytest.approx(50e3, rel=0.02)


def test_off_peak_growth_keeps_peak():
    w = traffic.parametric(150e3, 50e3, 6e3)
    x = traffic.raise_off_peak(w, 4)
    assert x.max() == pytest.approx(w.max())
    assert x.mean() > 1.3 * w.mean()


def test_fleet_covers_peak_with_maas():
    r = costmodel.scenario(CFG, 12000, PRICES, "vertex_cud_3y", 1)
    f = r["fleet"]
    cap = 10 * 5000 / 1.3
    assert f.gpus * 12000 * 0.85 + cap >= r["peak"] - 1
    assert 0 < f.util < 1


def test_more_off_peak_load_improves_vs_maas():
    a = costmodel.scenario(CFG, 12000, PRICES, "vertex_cud_3y", 1)
    b = costmodel.scenario(CFG, 12000, PRICES, "vertex_cud_3y", 1, off_peak=4)
    assert b["vs_maas"] < a["vs_maas"]


def test_fit_rule():
    w = 51.6e9  # Gemma 4 26B in BF16
    assert planner.fits(w, "NVIDIA_RTX_PRO_6000", 1)
    assert not planner.fits(w, "NVIDIA_L4", 2)
    assert planner.fits(w, "NVIDIA_L4", 4)
    assert planner.fits(w / 2, "NVIDIA_L4", 2)
