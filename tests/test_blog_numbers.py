"""The blog's numbers, from its measured example (blog/example/workload.yaml: reasoning off, 1.25% / 0.3% long tail,
per-class rates and the short-prompt rate measured on 1x RTX PRO 6000) at the figure script's fixed prices."""
import importlib.util
from pathlib import Path

import pytest

from aiops_dilemma import costmodel, server

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mf():
    spec = importlib.util.spec_from_file_location("mf", ROOT / "blog/figures/make_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pct(x):
    return round(x * 100)


def test_measured_rate_is_the_example_constant(mf):
    assert mf.PER_GPU == server.BLOG_EXAMPLE_TPS == 17946
    c = mf.cfg()
    assert c["requests"]["thinking_share"] == 0 and c["throughput"] == {"medium_tps": 5335, "large_tps": 4001}
    assert round(mf.sc(c)["blended_per_gpu"], -2) == 11000


def test_blog_headline_and_arc(mf):
    c = mf.cfg()
    r = mf.sc(c)
    assert r["gpus"] == 10 and round(r["total"] / 1e3) == 29 and round(r["baseline"]["total"] / 1e3) == 22
    assert pct(r["vs_baseline"]) == 132
    # loses at the base case and when nights grow by half, wins once they double
    assert [pct(mf.sc(c, off_peak=o)["vs_baseline"]) for o in (1.5, 2, 4)] == [106, 97, 81]


def test_blog_hybrids_and_tiers(mf):
    c = mf.cfg()
    routed = mf.sc(c, route_long=True, route_long_target="open_maas")
    assert routed["gpus"] == 5 and pct(routed["vs_baseline"]) == 79
    both = mf.sc(c, route_long=True, route_long_target="open_maas", peak_overflow_target="gemini_flash_lite")
    assert both["gpus"] == 3 and pct(both["vs_baseline"]) == 72
    assert {t: pct(mf.sc(c, t)["vs_baseline"]) for t in ("vertex", "vertex_cud_3y", "cud_3y")} == \
        {"vertex": 201, "vertex_cud_3y": 93, "cud_3y": 75}
    assert pct(costmodel.breakeven_utilization(c, mf.PER_GPU, mf.PRICES["vertex_cud_1y"])) == 56
