"""POST /api/ingest/describe: workload from a document. Gemini is mocked; the document is synthetic.

Set DILEMMA_LIVE_GEMINI=1 to also run one live call against Gemini on Agent Platform.
"""

import io
import json
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from aiops_dilemma import ingest, server

SYNTHETIC_DOC = """Example workload description (synthetic).
Our assistant answers internal questions. The busiest weekday hour averages 100,000 input tokens per second,
the weekly average is 25,000 and the quietest hour 2,500. Requests average 7,000 input and 950 output tokens,
with a 500-token shared system prompt. Weekdays 13:00-20:00 UTC are busiest; weekends run at 15% of weekdays.
The open-model API we use allows 5 requests per second."""

CANNED = {
    "fields": [
        {"path": "traffic.peak_input_tps", "number": 100000, "confidence": "stated", "evidence": "100,000 input tok/s"},
        {"path": "traffic.average_input_tps", "number": 25000, "confidence": "stated", "evidence": "weekly 25,000"},
        {"path": "traffic.floor_input_tps", "number": 2500, "confidence": "stated", "evidence": "quietest 2,500"},
        {"path": "traffic.busy_hours_utc", "numbers": [13, 20], "confidence": "stated", "evidence": "13:00-20:00"},
        {"path": "traffic.weekend_factor", "number": 15, "confidence": "stated", "evidence": "15% of weekdays"},
        {"path": "traffic.custom_hourly", "numbers": [1000.0 + 100 * h for h in range(24)],
         "confidence": "derived", "evidence": "daily pattern"},
        {"path": "requests.avg_input_tokens", "number": 7000.4, "confidence": "stated", "evidence": "7,000 input"},
        {"path": "requests.avg_output_tokens", "number": 950, "confidence": "stated", "evidence": "950 output"},
        {"path": "requests.system_prompt_tokens", "number": 500, "confidence": "stated", "evidence": "500-token"},
        {"path": "requests.pct_over_32k", "number": 2, "confidence": "assumed", "evidence": "guess"},
        {"path": "requests.pct_over_128k", "number": 5, "confidence": "assumed", "evidence": "inconsistent"},
        {"path": "requests.thinking_share", "number": 25, "confidence": "assumed", "evidence": "percent, not fraction"},
        {"path": "maas.max_rps", "number": 5, "confidence": "stated", "evidence": "5 requests per second"},
        {"path": "maas.input_per_m", "number": -1, "confidence": "assumed", "evidence": "negative: dropped"},
        {"path": "constraints.latency_budget_s", "number": 0, "confidence": "assumed", "evidence": "zero: dropped"},
        {"path": "use_case.task_type", "text": "Chat", "confidence": "assumed", "evidence": "answers questions"},
        {"path": "use_case.description", "text": "  Internal Q&A   assistant. ", "confidence": "stated",
         "evidence": "answers internal questions"},
        {"path": "traffic.peak_input_tps", "number": 1, "confidence": "stated", "evidence": "duplicate: ignored"},
        {"path": "secret.unknown", "number": 42, "confidence": "stated", "evidence": "unknown key: dropped"},
    ],
    "notes": ["Synthetic note."],
}


class FakeAPIError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"{code} {message}")
        self.code = code


class FakeResponse:
    def __init__(self, payload: dict):
        self.parsed = None
        self.text = json.dumps(payload)


class FakeClient:
    """Stands in for google.genai.Client; `behaviour` maps model id -> payload dict or exception."""

    def __init__(self, behaviour: dict, default=CANNED):
        self.calls: list[dict] = []
        self.behaviour = behaviour
        self.default = default
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.behaviour.get(model, self.default)
        if isinstance(out, Exception):
            raise out
        return FakeResponse(out)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    monkeypatch.setattr(server, "_default_project", lambda: "test-project")
    monkeypatch.delenv("DILEMMA_DESCRIBE_MODEL", raising=False)
    holder = {"client": FakeClient({})}
    created = []

    def factory(project, location):
        created.append((project, location))
        return holder["client"]

    monkeypatch.setattr(ingest, "_genai_client", factory)
    holder["created"] = created
    holder["runs"] = runs
    return holder


def _parts_text(call) -> str:
    return "\n".join(p.text or "" for c in call["contents"] for p in c.parts)


def test_describe_text_validates_and_clamps(fake):
    r = TestClient(server.app).post("/api/ingest/describe", json={"text": SYNTHETIC_DOC})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == ingest.DESCRIBE_MODELS[0]
    assert fake["created"] == [("test-project", "global")]
    call = fake["client"].calls[0]
    assert "100,000 input tokens per second" in _parts_text(call)
    assert call["config"].temperature <= 0.2 and call["config"].response_mime_type == "application/json"

    w = body["workload"]
    assert w["traffic"]["peak_input_tps"] == 100000          # duplicate path ignored
    assert w["traffic"]["busy_hours_utc"] == [13, 20]
    assert w["traffic"]["weekend_factor"] == 0.15             # 15 (percent) -> 0.15 (config fraction)
    assert len(w["traffic"]["custom_hourly"]) == 168          # one day repeated for the week
    assert w["requests"]["avg_input_tokens"] == 7000          # token counts are integers
    assert w["requests"]["pct_over_32k"] == 2 and w["requests"]["pct_over_128k"] == 2  # capped to over-32K
    assert w["requests"]["thinking_share"] == 0.25            # thinking_share is 0-1
    assert w["maas"] == {"max_rps": 5}                        # negative price dropped
    assert "constraints" not in w                             # zero latency budget dropped
    assert w["use_case"] == {"task_type": "chat", "description": "Internal Q&A assistant."}
    assert "secret" not in w

    paths = [f["path"] for f in body["fields"]]
    assert len(paths) == len(set(paths)) and "secret.unknown" not in paths
    f0 = body["fields"][0]
    assert set(f0) == {"path", "value", "confidence", "evidence"} and f0["confidence"] == "stated"
    assert "Synthetic note." in body["notes"]
    assert any("128K" in n for n in body["notes"]) and any("repeated" in n for n in body["notes"])
    assert not fake["runs"].exists()                          # nothing persisted


def test_describe_pdf_upload_is_sent_inline(fake):
    pdf = b"%PDF-1.4\n% synthetic test document\n%%EOF\n"
    r = TestClient(server.app).post("/api/ingest/describe", files={"file": ("workload.pdf", pdf, "application/pdf")})
    assert r.status_code == 200, r.text
    parts = fake["client"].calls[0]["contents"][0].parts
    inline = [p.inline_data for p in parts if p.inline_data]
    assert inline and inline[0].mime_type == "application/pdf" and inline[0].data == pdf
    assert not fake["runs"].exists()


def test_describe_docx_and_markdown(fake):
    buf = io.BytesIO()
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    paras = "".join(f"<w:p><w:r><w:t>{ln}</w:t></w:r></w:p>" for ln in SYNTHETIC_DOC.splitlines())
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="{ns}"><w:body>{paras}</w:body></w:document>')
    client = TestClient(server.app)
    r = client.post("/api/ingest/describe", files={"file": ("w.docx", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 200, r.text
    assert "7,000 input and 950 output tokens" in _parts_text(fake["client"].calls[-1])
    r = client.post("/api/ingest/describe", files={"file": ("w.md", SYNTHETIC_DOC.encode(), "text/markdown")})
    assert r.status_code == 200
    assert "quietest hour 2,500" in _parts_text(fake["client"].calls[-1])


def test_describe_falls_back_to_next_model(fake):
    first, second = ingest.DESCRIBE_MODELS[:2]
    fake["client"] = FakeClient({first: FakeAPIError(404, "Publisher model not found")})
    r = TestClient(server.app).post("/api/ingest/describe", json={"text": SYNTHETIC_DOC})
    assert r.status_code == 200 and r.json()["model"] == second
    assert [c["model"] for c in fake["client"].calls] == [first, second]


def test_describe_errors(fake, monkeypatch):
    client = TestClient(server.app)
    fake["client"] = FakeClient({m: FakeAPIError(404, "model not found") for m in ingest.DESCRIBE_MODELS})
    r = client.post("/api/ingest/describe", json={"text": SYNTHETIC_DOC})
    assert r.status_code == 503 and "Agent Platform" in r.json()["detail"]

    fake["client"] = FakeClient({}, default=FakeAPIError(401, "UNAUTHENTICATED"))
    r = client.post("/api/ingest/describe", json={"text": SYNTHETIC_DOC})
    assert r.status_code == 401 and "Sign in" in r.json()["detail"]

    monkeypatch.setattr(server, "_default_project", lambda: None)
    monkeypatch.setattr(server.auth, "default_project", lambda: None)
    r = client.post("/api/ingest/describe", json={"text": SYNTHETIC_DOC})
    assert r.status_code == 401
    monkeypatch.setattr(server, "_default_project", lambda: "test-project")

    r = client.post("/api/ingest/describe", files={"file": ("w.xlsx", b"PK\x03\x04", "application/octet-stream")})
    assert r.status_code == 415 and "Unsupported" in r.json()["detail"]
    r = client.post("/api/ingest/describe", files={"file": ("w.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 415
    r = client.post("/api/ingest/describe", json={"text": "   "})
    assert r.status_code == 400

    monkeypatch.setattr(ingest, "DESCRIBE_MAX_BYTES", 1000)
    r = client.post("/api/ingest/describe", files={"file": ("w.txt", b"x" * 5000, "text/plain")})
    assert r.status_code == 413 and "too large" in r.json()["detail"]
    r = client.post("/api/ingest/describe", files={"file": ("w.txt", b"x" * 1500, "text/plain")})
    assert r.status_code == 413
    assert not fake["runs"].exists()


@pytest.mark.skipif(not os.environ.get("DILEMMA_LIVE_GEMINI"), reason="set DILEMMA_LIVE_GEMINI=1 for a live call")
def test_describe_live_gemini():
    out = ingest.describe_workload(text=SYNTHETIC_DOC)
    w = out["workload"]
    assert out["model"]
    assert abs(w["traffic"]["peak_input_tps"] - 100000) / 100000 < 0.05
    assert w["requests"]["avg_input_tokens"] == 7000 and w["requests"]["avg_output_tokens"] == 950
    assert w["maas"]["max_rps"] == 5
