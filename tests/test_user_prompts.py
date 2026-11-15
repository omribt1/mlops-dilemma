"""Bring your own prompts (userprompts + /api/sessions/{name}/prompts): formats, separators, token counting, answer
length defaults, size proposal, endpoints, the benchmark job using the uploaded set (fake bench), level planning with
a small pool, short-only calibration from file prompts. All prompts here are made up."""
import json
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from aiops_dilemma import cli, config, jobs, loadtest, prompts, server, userprompts

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "tests" / "fake_bench.py")

MADE_UP = [
    "Summarize the plot of a novel about a lighthouse keeper who adopts a seagull.",
    "List three ways to keep a sourdough starter alive during a long holiday.",
    "Translate 'the purple giraffe sings at dawn' into formal Latin.",
    "Write a limerick about a robot learning to bake bread.",
]


class _WordCounter:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    def __init__(self):
        self.tok = object()

    def count(self, text):
        return len(text.split())


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    monkeypatch.setattr(userprompts, "token_counter", lambda hf_id, cache: (prompts.TokenCounter(None), "estimate"))


def _cfg(**req):
    c = config.normalize({"name": "t", "requests": {"avg_input_tokens": 3000, "avg_output_tokens": 300, **req},
                          "loadtest": {"concurrency": [2, 4], "requests_per_level": 10}})
    return c


# ------------------------------------------------------------------------------------------------------ formats
def test_jsonl_chat_records_and_objects():
    lines = [
        {"messages": [{"role": "system", "content": "You are a cheerful gardener."},
                      {"role": "user", "content": MADE_UP[0]}], "max_tokens": 200,
         "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}},
        {"prompt": MADE_UP[1], "system": "Answer briefly.", "output_tokens": 120, "input_tokens": 900},
        {"question": MADE_UP[2]},
        {"body": {"messages": [{"role": "user", "content": [{"type": "text", "text": MADE_UP[3]}]}]}},
        {"unrelated": 1},
    ]
    raw = ("\n".join(json.dumps(x) for x in lines) + "\nnot json\n").encode()
    recs, info = userprompts.parse_file("p.jsonl", raw)
    assert len(recs) == 4 and info["skipped"] == 2 and info["format"].startswith("JSONL")
    assert recs[0]["max_tokens"] == 200 and recs[0]["response_format"] == {"type": "json_object"}
    assert recs[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert recs[1]["messages"][0] == {"role": "system", "content": "Answer briefly."}
    assert recs[1]["output_tokens"] == 120 and recs[1]["file_input_tokens"] == 900
    assert recs[2]["messages"] == [{"role": "user", "content": MADE_UP[2]}]
    assert recs[3]["messages"][0]["content"] == MADE_UP[3]
    assert {"messages", "max_tokens", "prompt", "question", "output token counts"} <= set(info["fields"])


def test_json_shapes():
    for data in (MADE_UP, [{"text": t} for t in MADE_UP], {"prompts": MADE_UP}, {"prompts": [{"input": t} for t in MADE_UP]}):
        recs, info = userprompts.parse_file("p.json", json.dumps(data).encode())
        assert [r["messages"][-1]["content"] for r in recs] == MADE_UP, data
    with pytest.raises(userprompts.PromptUploadError):
        userprompts.parse_file("p.json", b"{not json")


def test_csv_and_tsv():
    csv_raw = ("id,text,system,max_tokens\n1,\"" + MADE_UP[0] + "\",Be kind.,150\n2,\"" + MADE_UP[1] + "\",,\n").encode()
    recs, info = userprompts.parse_file("p.csv", csv_raw)
    assert len(recs) == 2 and recs[0]["max_tokens"] == 150 and recs[0]["messages"][0]["role"] == "system"
    assert recs[1]["messages"] == [{"role": "user", "content": MADE_UP[1]}] and info["format"] == "CSV"
    tsv_raw = ("prompt\tmax_tokens\n" + MADE_UP[2] + "\t90\n").encode()
    recs, info = userprompts.parse_file("p.tsv", tsv_raw)
    assert recs[0]["max_tokens"] == 90 and info["format"] == "TSV"
    with pytest.raises(userprompts.PromptUploadError) as e:
        userprompts.parse_file("p.csv", b"colour,size\nblue,3\n")
    assert "no prompt text" in str(e.value) and "colour" in str(e.value)


def test_text_separators():
    text = f"{MADE_UP[0]}\n---\n{MADE_UP[1]}\nsecond line\n  ---  \n\n{MADE_UP[2]}\n---\n"
    assert userprompts.split_text(text, "separator") == [MADE_UP[0], MADE_UP[1] + "\nsecond line", MADE_UP[2]]
    assert userprompts.split_text("a b\n\nc\n \nd", "blocks") == ["a b", "c", "d"]
    assert userprompts.split_text("one\n\ntwo\nthree\n", "lines") == ["one", "two", "three"]
    assert userprompts.split_text("no separators here\nat all", "separator") == ["no separators here\nat all"]
    recs, info = userprompts.parse_file("p.md", text.encode(), "separator")
    assert len(recs) == 3 and "---" in info["format"]


def test_wrong_kind_points_to_the_right_section():
    ts = "timestamp,value\n" + "".join(f"2026-11-0{1 + h // 24}T{h % 24:02d}:00:00Z,{100 + h}\n" for h in range(72))
    with pytest.raises(userprompts.PromptUploadError) as e:
        userprompts.parse_file("traffic.csv", ts.encode())
    assert "use Upload in Traffic" in str(e.value) and e.value.section == "traffic"
    with pytest.raises(userprompts.PromptUploadError) as e:
        userprompts.parse_file("brief.pdf", b"%PDF-1.4")
    assert "Traffic" in str(e.value)
    assert userprompts.classify("traffic.csv", ts.encode())["primary"] == "traffic"


def test_classify_mixed_request_log():
    rows = [{"timestamp": f"2026-11-02T{h % 24:02d}:{h // 24:02d}:00Z", "prompt": f"{MADE_UP[h % 4]} #{h}",
             "system": "Be terse.", "input_tokens": 40 + h, "output_tokens": 20} for h in range(30)]
    raw = "\n".join(json.dumps(r) for r in rows).encode()
    k = userprompts.classify("log.jsonl", raw)
    assert {"prompts", "request_timestamps", "system_prompts", "token_counts"} <= set(k["kinds"])
    assert k["primary"] == "prompts" and k["prompts"] == 30
    assert userprompts.system_prompts_from("log.jsonl", raw) == ["Be terse."]


# --------------------------------------------------------------------------------------- tokens, dedupe, defaults
def test_token_counting_tokenizer_vs_estimate_and_dedupe():
    recs, info = userprompts.parse_file("p.json", json.dumps(MADE_UP + [MADE_UP[0], "   "]).encode())
    out, meta = userprompts.build([(recs, info)], _WordCounter(), "tokenizer", "org/model")
    assert len(out) == 4 and meta["dropped"] == {"empty": 1, "duplicates": 1}
    assert out[0]["input_tokens"] == len(MADE_UP[0].split())
    assert meta["tokens"]["method"] == "tokenizer" and meta["tokens"]["tokenizer"] == "org/model"
    out2, meta2 = userprompts.build([(recs, info)])
    assert out2[0]["input_tokens"] == int(len(MADE_UP[0]) / 3.5) and meta2["tokens"]["method"] == "estimate"
    # counts given in the file win over counting
    recs3, info3 = userprompts.parse_file("p.jsonl", json.dumps({"prompt": "hi there", "input_tokens": 777}).encode())
    out3, meta3 = userprompts.build([(recs3, info3)])
    assert out3[0]["input_tokens"] == 777 and meta3["tokens"]["from_file"] == 1


def test_answer_length_and_thinking_defaults():
    recs = [{"id": f"u{i}", "messages": [{"role": "user", "content": f"q{i}"}]} for i in range(10)]
    recs[0]["max_tokens"] = 77
    recs[1]["output_tokens"] = 55
    recs[2]["chat_template_kwargs"] = {"enable_thinking": True}
    cfg = _cfg(thinking_share=0.3, avg_thinking_tokens=1000)
    out = userprompts.apply_defaults(recs, cfg)
    assert out[0]["max_tokens"] == 77
    assert out[1]["max_tokens"] in (55, 1055)
    assert out[2]["chat_template_kwargs"] == {"enable_thinking": True} and out[2]["max_tokens"] == 300
    thinking = [r for r in out if r is not out[2] and r["chat_template_kwargs"]["enable_thinking"]]
    assert len(thinking) == round(0.3 * 9) - 0 or len(thinking) == int(0.3 * 9)
    assert all(r["max_tokens"] == (r.get("output_tokens") or 300) + 1000 for r in thinking if "max_tokens" not in recs[out.index(r)])
    assert out == userprompts.apply_defaults(recs, cfg)   # deterministic
    assert userprompts.thinking_flags(10, 0.3).count(True) == 3
    no_think = userprompts.apply_defaults(recs, _cfg(thinking_share=0))
    assert "chat_template_kwargs" not in no_think[5] and no_think[5]["max_tokens"] == 300


def test_summary_and_size_proposal(tmp_path):
    long_body = "word " * 140000           # > 32K and > 128K by the estimate? 700K chars / 3.5 = 200K tokens
    texts = MADE_UP + ["context " * 20000 + "question?", long_body]
    userprompts.store_text(tmp_path, "\n---\n".join(texts), "separator")
    cfg = _cfg()
    s = userprompts.summary(tmp_path, cfg, context=131072)
    assert s["count"] == 6 and s["input"]["over_32k"] == 2 and s["input"]["over_128k"] == 1
    assert s["input"]["max"] >= 128000 and s["input"]["median"] > 0
    assert s["answer"]["prompts_using_default"] == 6 and "300 tokens" in s["answer"]["text"]
    assert len(s["preview"]) == 3 and s["preview"][0]["text"] == MADE_UP[0]
    w = " ".join(x["text"] for x in s["warnings"])
    assert "repeat" in w and f"about {20 / 6:.1f} times" in w          # 2 levels x 10 requests, 6 prompts
    assert "longer than the model's 131,072-token context" in w
    assert "estimates" in w
    prop = {p["path"]: p["value"] for p in s["proposal"]}
    assert prop["requests.pct_over_32k"] == pytest.approx(100 * 2 / 6, abs=0.01)
    assert prop["requests.pct_over_128k"] == pytest.approx(100 / 6, abs=0.01)
    assert "requests.medium_mean" in prop and "requests.large_mean" in prop
    assert "requests.avg_output_tokens" not in prop       # no outputs in the upload
    # with outputs given, the proposal includes the average answer
    (tmp_path / "o").mkdir()
    userprompts.store_files(tmp_path / "o", [("p.jsonl", "\n".join(json.dumps({"prompt": t, "output_tokens": 100 + i})
                                                                    for i, t in enumerate(MADE_UP)).encode())])
    s2 = userprompts.summary(tmp_path / "o", cfg)
    assert {p["path"]: p["value"] for p in s2["proposal"]}["requests.avg_output_tokens"] == 102
    assert "bring their own" in s2["answer"]["text"] or "Every prompt" in s2["answer"]["text"]


def test_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(userprompts, "MAX_PROMPTS", 3)
    with pytest.raises(userprompts.PromptUploadError) as e:
        userprompts.store_text(tmp_path, "\n".join(MADE_UP), "lines")
    assert e.value.status == 413 and "limit of 3" in str(e.value)
    monkeypatch.setattr(userprompts, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(userprompts.PromptUploadError) as e:
        userprompts.store_files(tmp_path, [("p.txt", b"x" * 20)])
    assert e.value.status == 413


# ---------------------------------------------------------------------------------------------------- endpoints
@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    monkeypatch.setenv("DILEMMA_TEST_FAKE_BENCH", FAKE)
    monkeypatch.setenv("FAKE_BENCH_TICK_S", "0.0")
    server._CHILDREN.clear()
    yield TestClient(server.app)
    for proc in list(server._CHILDREN.values()):
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    server._CHILDREN.clear()


def _new(client, name="byo", **cfg):
    body = {"config": {"name": name, "loadtest": {"concurrency": [2, 4], "requests_per_level": 10},
                       "requests": {"avg_input_tokens": 3000, "avg_output_tokens": 300},
                       "traffic": {"peak_input_tps": 100000, "average_input_tps": 25000, "floor_input_tps": 2500},
                       **cfg}}
    assert client.post(f"/api/sessions/{name}", json=body).status_code == 200
    return server.RUNS_DIR / name


def test_upload_replace_delete_roundtrip(client):
    rd = _new(client)
    r = client.post("/api/sessions/byo/prompts", files=[("files", ("p.txt", "\n---\n".join(MADE_UP).encode(), "text/plain"))])
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["source"] == "file" and j["file"] == "uploaded_prompts.jsonl" and j["summary"]["count"] == 4
    assert (rd / "uploaded_prompts.jsonl").exists()
    saved = yaml.safe_load((rd / "workload.yaml").read_text())
    assert saved["prompts"]["source"] == "file" and saved["prompts"]["file"] == "uploaded_prompts.jsonl"
    sess = client.get("/api/sessions/byo").json()
    assert sess["user_prompts"]["count"] == 4 and sess["prompt_source"]["kind"] == "file"
    assert "your 4 prompts (uploaded" in sess["prompt_source"]["text"]
    # replace: JSONL file
    jl = "\n".join(json.dumps({"messages": [{"role": "user", "content": t}], "max_tokens": 64}) for t in MADE_UP[:2])
    r = client.post("/api/sessions/byo/prompts", files=[("files", ("p.jsonl", jl.encode(), "application/json"))])
    assert r.json()["summary"]["count"] == 2 and r.json()["summary"]["answer"]["prompts_with_max_tokens"] == 2
    # paste, one per line
    r = client.post("/api/sessions/byo/prompts", json={"text": "\n".join(MADE_UP), "mode": "lines"})
    assert r.status_code == 200 and r.json()["summary"]["count"] == 4
    assert client.get("/api/sessions/byo/prompts").json()["summary"]["count"] == 4
    r = client.delete("/api/sessions/byo/prompts")
    assert r.status_code == 200 and r.json()["source"] == "synthetic" and r.json()["summary"] is None
    assert not (rd / "uploaded_prompts.jsonl").exists()
    assert yaml.safe_load((rd / "workload.yaml").read_text())["prompts"]["source"] == "synthetic"


def test_upload_errors_and_limits(client, monkeypatch):
    _new(client)
    r = client.post("/api/sessions/byo/prompts", files=[("files", ("x.csv", b"colour,size\nblue,3\n", "text/csv"))])
    assert r.status_code == 422 and "no prompt text" in r.json()["detail"]
    r = client.post("/api/sessions/byo/prompts", json={"text": "  "})
    assert r.status_code == 400
    monkeypatch.setattr(userprompts, "MAX_PROMPTS", 2)
    r = client.post("/api/sessions/byo/prompts", json={"text": "\n".join(MADE_UP), "mode": "lines"})
    assert r.status_code == 413 and "limit of 2" in r.json()["detail"]
    monkeypatch.setattr(userprompts, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/sessions/byo/prompts", files=[("files", ("p.txt", b"y " * 200, "text/plain"))])
    assert r.status_code == 413


def test_example_session_is_read_only_copy_then_upload(client):
    r = client.post("/api/sessions/blog-example/prompts", json={"text": MADE_UP[0]})
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
    assert client.delete("/api/sessions/blog-example/prompts").status_code == 403
    r = client.post("/api/sessions/my-copy", json={"copy_of": "blog-example"})
    assert r.status_code == 200
    r = client.post("/api/sessions/my-copy/prompts", json={"text": "\n---\n".join(MADE_UP)})
    assert r.status_code == 200 and r.json()["summary"]["count"] == 4
    # a copy of the copy carries the uploaded set
    r = client.post("/api/sessions/copy2", json={"copy_of": "my-copy"})
    assert r.json()["user_prompts"]["count"] == 4


def test_generic_upload_routes_prompt_files_the_same_way(client):
    rd = _new(client)
    jl = "\n".join(json.dumps({"prompt": t, "max_tokens": 50}) for t in MADE_UP)
    r = client.post("/api/sessions/byo/upload", files=[("files", ("p.jsonl", jl.encode(), "application/json"))],
                    data={"role_hint": "prompts"})
    assert r.status_code == 200, r.text
    ing = r.json()["ingested"]
    assert ing["prompts"]["file"] == "uploaded_prompts.jsonl" and ing["prompts_summary"]["count"] == 4
    via_generic = (rd / "uploaded_prompts.jsonl").read_text()
    client.post("/api/sessions/byo/prompts", files=[("files", ("p.jsonl", jl.encode(), "application/json"))])
    assert (rd / "uploaded_prompts.jsonl").read_text() == via_generic
    # plain text with the prompts role is a prompt set too
    r = client.post("/api/sessions/byo/upload", files=[("files", ("p.txt", "\n---\n".join(MADE_UP).encode(), "text/plain"))],
                    data={"role_hint": "prompts"})
    assert r.json()["ingested"]["prompts_summary"]["count"] == 4


def test_classify_and_system_prompt_endpoints(client):
    _new(client)
    files = [("files", ("sys.txt", b"You are a cheerful gardener.\n---\nYou are a strict librarian.", "text/plain")),
             ("files", ("log.jsonl", json.dumps({"prompt": MADE_UP[0], "system": "Be terse."}).encode(), "application/json"))]
    k = client.post("/api/uploads/classify", files=files).json()["files"]
    assert k[0]["kinds"] == ["text"] and "prompts" in k[1]["kinds"] and "system_prompts" in k[1]["kinds"]
    r = client.post("/api/sessions/byo/system-prompts", files=files)
    assert r.status_code == 200
    assert r.json()["system_prompts"] == ["You are a cheerful gardener.", "You are a strict librarian.", "Be terse."]
    ts = "timestamp,value\n" + "".join(f"2026-11-0{1 + h // 24}T{h % 24:02d}:00:00Z,{100 + h}\n" for h in range(72))
    r = client.post("/api/sessions/byo/system-prompts", files=[("files", ("t.csv", ts.encode(), "text/csv"))])
    assert r.status_code == 422 and "use Upload in Traffic" in r.json()["detail"]


# ----------------------------------------------------------------------------------- the benchmark uses the prompts
def test_from_file_keeps_uploaded_set_and_applies_defaults(tmp_path):
    userprompts.store_text(tmp_path, "\n---\n".join(MADE_UP))
    (tmp_path / "workload.yaml").write_text(config.to_yaml(config.normalize({
        "name": tmp_path.name, "requests": {"avg_output_tokens": 222, "thinking_share": 0.5, "avg_thinking_tokens": 100},
        "prompts": {"source": "file", "file": "uploaded_prompts.jsonl", "custom_system_prompts": ["Be playful."]}})))
    cfg = config.normalize(yaml.safe_load((tmp_path / "workload.yaml").read_text()), path=str(tmp_path / "workload.yaml"))
    out = prompts.from_file(cfg, tmp_path / "prompts.jsonl")
    rows = [json.loads(line) for line in open(out)]
    assert [r["messages"][-1]["content"] for r in rows] == MADE_UP
    assert all(r["messages"][0] == {"role": "system", "content": "Be playful."} for r in rows)
    assert sorted(r["max_tokens"] for r in rows) == [222, 222, 322, 322]
    assert sum(r["chat_template_kwargs"]["enable_thinking"] for r in rows) == 2


def test_level_planning_with_a_small_pool_repeats():
    pool = [{"id": f"u{i}", "messages": [{"role": "user", "content": "q"}], "input_tokens": 500 + i} for i in range(6)]
    pool += [{"id": "big", "messages": [{"role": "user", "content": "q"}], "input_tokens": 50000}]
    samples, info = loadtest.plan_levels(pool, [10, 10])
    assert info["repeats"] == 20 - 7 and "reuse a prompt" in info["note"]
    assert all(any(p["_size_class"] == "medium" for p in s) for s in samples)


def test_short_only_calibration_uses_file_prompts(tmp_path):
    recs = [{"id": f"u{i}", "messages": [{"role": "user", "content": MADE_UP[i % 4]}], "input_tokens": t}
            for i, t in enumerate([800, 1200, 40000, 150000, 900])]
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    cfg = config.normalize({"name": "t", "prompts": {"source": "file", "file": "uploaded_prompts.jsonl"}})
    out = prompts.short_only(cfg, tmp_path)
    kept = [json.loads(line)["input_tokens"] for line in open(out)]
    assert kept == [800, 1200, 900]
    assert cli.needs_calibration(cfg, tmp_path) is True
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs if r["input_tokens"] < 32000))
    assert cli.needs_calibration(cfg, tmp_path) is False
    cfg["requests"]["pct_over_32k"] = 5.0        # the file decides, not the configured share
    assert cli.needs_calibration(cfg, tmp_path) is False


def _wait_done(client, name, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/sessions/{name}").json()["job"]
        if job["status"] not in ("running", "idle"):
            return job
        time.sleep(0.1)
    raise AssertionError(job)


def test_benchmark_job_uses_the_uploaded_prompts(client):
    rd = _new(client, model={"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": None, "precision": "fp8"},
              project="my-gcp-project")
    jl = "\n".join(json.dumps({"prompt": f"{MADE_UP[i % 4]} (variant {i})", "input_tokens": 600 + i}) for i in range(8))
    jl += "\n" + json.dumps({"prompt": "A very long made-up contract to review.", "input_tokens": 60000})
    assert client.post("/api/sessions/byo/prompts", files=[("files", ("p.jsonl", jl.encode(), "application/json"))]).status_code == 200
    cfg = client.get("/api/sessions/byo").json()["config"]
    r = client.post("/api/sessions/byo/run", json={"config": cfg, "mode": "auto"})
    assert r.status_code == 200, r.text
    job = _wait_done(client, "byo")
    assert job["status"] == "completed", job
    rows = [json.loads(line) for line in open(rd / "prompts.jsonl")]
    assert len(rows) == 9 and rows[0]["messages"][-1]["content"].startswith(MADE_UP[0])
    assert all(r["max_tokens"] == 300 for r in rows)        # the configured answer length
    plan = json.loads((rd / "load" / "levels_plan.json").read_text())
    assert plan["pool"] == 9 and plan["repeats"] > 0
    assert "medium" in plan["pool_mix"]
    ev = [json.loads(line) for line in (rd / jobs.EVENTS_FILE).read_text().splitlines()]
    done = next(e for e in ev if e.get("type") == "prompts_done")
    assert "your 9 prompts" in done["message"] and "repeat" in done["message"]
    # calibration ran on the file's short prompts
    short = [json.loads(line) for line in open(rd / "prompts_short.jsonl")]
    assert len(short) == 8 and all(r["input_tokens"] < 32000 for r in short)
    meas = json.loads((rd / "measured.json").read_text())
    assert meas["prompt_source"]["kind"] == "file" and meas["prompt_source"]["count"] == 9
    assert "Prompts: your 9 prompts (uploaded" in (rd / "report.md").read_text()
    assert client.get("/api/sessions/byo").json()["job"]["prompt_source"]["count"] == 9


def test_benchmark_job_skips_calibration_without_long_file_prompts(client):
    rd = _new(client, model={"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": None, "precision": "fp8"},
              project="my-gcp-project", requests={"avg_input_tokens": 3000, "avg_output_tokens": 300, "pct_over_32k": 4})
    client.post("/api/sessions/byo/prompts", json={"text": "\n---\n".join(MADE_UP)})
    cfg = client.get("/api/sessions/byo").json()["config"]
    client.post("/api/sessions/byo/run", json={"config": cfg, "mode": "auto"})
    job = _wait_done(client, "byo")
    assert job["status"] == "completed", job
    ev = [json.loads(line) for line in (rd / jobs.EVENTS_FILE).read_text().splitlines()]
    assert any("Your prompts have none over 32K" in (e.get("message") or "") for e in ev)
    assert not (rd / "prompts_short.jsonl").exists()


def test_synthetic_report_line(client):
    rd = _new(client, model={"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "hf_id": None, "precision": "fp8"},
              project="my-gcp-project")
    cfg = client.get("/api/sessions/byo").json()["config"]
    client.post("/api/sessions/byo/run", json={"config": cfg, "mode": "auto"})
    assert _wait_done(client, "byo")["status"] == "completed"
    assert "Prompts: synthetic" in (rd / "report.md").read_text()


def test_run_saves_default_model_for_six_number_sessions(client):
    """A first-run "six numbers" session has no model; starting a benchmark saves the default the estimate shows."""
    import yaml
    from aiops_dilemma import server
    client.post("/api/sessions/six-numbers", json={"config": {"name": "six-numbers", "traffic": {
        "peak_input_tps": 1000, "average_input_tps": 400, "floor_input_tps": 50}}})
    rd = server._session_dir("six-numbers")
    assert not (yaml.safe_load((rd / "workload.yaml").read_text()).get("model") or {}).get("model_garden_id")
    server._ensure_model(rd)
    assert yaml.safe_load((rd / "workload.yaml").read_text())["model"]["model_garden_id"] == server.DEFAULT_MODEL["model_garden_id"]
