"""Level comparability: every load-test level gets its own prompt sample with the same (size class, thinking) mix
(loadtest.plan_levels), deterministic, without repeats while the pool allows; the summary records the actual mix."""
import json
from collections import Counter

import pytest

from aiops_dilemma import config, loadtest, report


def _pool(n_short=900, n_short_think=100, n_med=40, n_long=10):
    out = []
    for kind, n, tok, think in (("s", n_short, 5000, False), ("t", n_short_think, 6000, True),
                                ("m", n_med, 60000, False), ("l", n_long, 190000, False)):
        for i in range(n):
            p = {"id": f"{kind}{i}", "input_tokens": tok, "messages": [{"role": "user", "content": "x"}]}
            if think:
                p["chat_template_kwargs"] = {"enable_thinking": True}
            out.append(p)
    return out


def _mix(sample):
    return Counter((p["_size_class"], p["_thinking"]) for p in sample)


def test_levels_get_equal_mixes_and_distinct_prompts():
    pool = _pool()
    samples, info = loadtest.plan_levels(pool, [250, 250, 250, 250])
    mixes = [_mix(s) for s in samples]
    assert all(m == mixes[0] for m in mixes)                        # identical counts per bucket
    assert mixes[0][("short", False)] == 215 and mixes[0][("long", False)] == 2 and mixes[0][("medium", False)] == 9
    ids = [p["id"] for s in samples for p in s]
    assert len(ids) == len(set(ids)) == 1000 and info["repeats"] == 0 and "note" not in info


def test_unequal_level_sizes_keep_shares_within_rounding():
    samples, _ = loadtest.plan_levels(_pool(), [300, 300, 300, 256])
    for s in samples:
        n = len(s)
        long_share = sum(p["_size_class"] != "short" for p in s) / n
        think = sum(p["_thinking"] for p in s) / n
        assert abs(long_share - 50 / 1050) <= 1 / n + 1e-9
        assert abs(think - 100 / 1050) <= 1 / n + 1e-9


def test_plan_is_deterministic_and_shuffled():
    a, _ = loadtest.plan_levels(_pool(), [200, 200], seed="x")
    b, _ = loadtest.plan_levels(_pool(), [200, 200], seed="x")
    assert [p["id"] for p in a[0]] == [p["id"] for p in b[0]]
    assert [p["id"] for p in a[1]] == [p["id"] for p in b[1]]
    assert [p["id"] for p in a[0][:20]] != sorted(p["id"] for p in a[0][:20])   # long prompts spread through


def test_small_pool_repeats_are_noted_and_marked():
    samples, info = loadtest.plan_levels(_pool(90, 10, 4, 1), [100, 100])
    assert info["repeats"] == 95 and "reuse a prompt" in info["note"]
    assert _mix(samples[0]) == _mix(samples[1])
    assert sum(p.get("repeat", False) for p in samples[1]) == 95


def test_file_prompts_classified_by_characters_and_default_thinking():
    long_p = {"messages": [{"role": "user", "content": "w" * 350_000}]}         # ~100K tokens at 3.5 chars/token
    assert loadtest.size_class(loadtest.prompt_tokens_estimate(long_p)) == "medium"
    assert loadtest.prompt_thinking({}, {"enable_thinking": True})
    assert not loadtest.prompt_thinking({"chat_template_kwargs": {"enable_thinking": False}}, {"enable_thinking": True})


def test_summary_records_mix():
    recs = ([{"status": 200, "prompt_tokens": 5000, "completion_tokens": 500, "latency": 2, "start": 0,
              "thinking": i < 2} for i in range(8)]
            + [{"status": 200, "prompt_tokens": 60000, "completion_tokens": 300, "latency": 9, "start": 0, "repeat": True},
               {"status": 200, "prompt_tokens": 150000, "completion_tokens": 300, "latency": 20, "start": 0},
               {"status": -1, "error": "boom", "latency": 1, "start": 0}])
    m = loadtest.summarize(recs)["mix"]
    assert m["requests"] == {"short": 0.8, "medium": 0.1, "long": 0.1}
    assert m["tokens"]["long"] == pytest.approx(150000 / 250000, abs=1e-4)
    assert m["thinking"] == 0.2 and m["repeats"] == 1 and m["avg_in"] == 25000
    assert report.mix_text(m) == "80/10/10, 20% thinking, 1 repeats"
    assert report.mix_text(None) == "-"


def test_run_sends_each_level_its_own_sample(monkeypatch, tmp_path):
    (tmp_path / "endpoint.json").write_text(json.dumps({"endpoint": "projects/p/locations/l/endpoints/1",
                                                        "dedicated_dns": "fake.example"}))
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in _pool(180, 20, 8, 2)))
    monkeypatch.setattr(loadtest, "wait_ready", lambda *a, **k: 0)
    seen = {}

    async def fake_level(url, prompts, start, n, c, kw, cb):
        seen[c] = [p["id"] for p in prompts]
        return [{"status": 200, "prompt_tokens": p["input_tokens"], "completion_tokens": 5, "latency": 1.0,
                 "start": 0.0, "thinking": loadtest.prompt_thinking(p)} for p in prompts[:n]]
    monkeypatch.setattr(loadtest, "_level", fake_level)
    cfg = config.normalize({"name": "mix", "model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it"},
                            "loadtest": {"concurrency": [4, 8], "requests_per_level": 100, "preflight": False}})
    loadtest.run(cfg, tmp_path)
    assert len(seen[4]) == len(seen[8]) == 100 and not set(seen[4]) & set(seen[8])
    plan = json.loads((tmp_path / "load" / "levels_plan.json").read_text())
    assert plan["levels"][0]["mix"] == plan["levels"][1]["mix"] == {"short": 86, "short+thinking": 9,
                                                                    "medium": 4, "long": 1}
