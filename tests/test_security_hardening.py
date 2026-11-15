"""Security hardening: regions are validated wherever they reach an API host (a crafted region
such as `evil.example#` would send credentials elsewhere), and .docx uploads can't decompress without limit."""
import io
import urllib.request
import zipfile

import pytest
from fastapi.testclient import TestClient

from aiops_dilemma import config, deploy, gemini, ingest, regions, server

EVIL = ["evil.example#", "evil.example/x?", "evil.example:443/", "us-central1.evil.example", "../../x", "",
        "US-CENTRAL1", "us-central1 ", "global/x", None, 12, ["us-central1"]]


def test_region_rule():
    for ok in ("us-central1", "europe-west4", "northamerica-northeast1", "me-central2", "europe-west10",
               "africa-south1", "southamerica-west1", "asia-northeast3"):
        assert regions.is_valid(ok) and regions.require(ok) == ok
    for bad in EVIL:
        assert not regions.is_valid(bad)
        with pytest.raises(regions.InvalidRegion):
            regions.require(bad)
    assert regions.is_valid("global", allow_global=True) and not regions.is_valid("global")
    assert regions.or_default("evil.example#") == "us-central1"


def test_config_never_keeps_a_crafted_region():
    assert config.normalize({"region": "evil.example#"})["region"] == "us-central1"
    assert config.normalize({"region": "europe-west4"})["region"] == "europe-west4"


def test_api_rejects_crafted_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(server.app)
    assert c.get("/api/catalog", params={"region": "evil.example#"}).status_code == 400
    assert c.post("/api/catalog/refresh", json={"region": "evil.example/x?"}).status_code == 400
    assert c.post("/api/gemini/refresh", json={"region": "evil.example:443/"}).status_code == 400
    assert c.post("/api/recommend/model", json={"region": "evil.example#"}).status_code == 400


def test_host_building_code_refuses_crafted_regions(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("a request was made with a crafted region")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(regions.InvalidRegion):
        gemini.fetch_garden("my-gcp-project", "evil.example#")
    with pytest.raises(regions.InvalidRegion):
        deploy.set_inference_timeout({"project": "my-gcp-project"},
                                     "projects/1/locations/evil.example#/endpoints/2", retries=1, wait_s=0)


def _docx(document_xml: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", document_xml)
    return buf.getvalue()


W = b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'


def test_docx_text_still_read():
    xml = W + b"<w:p><w:r><w:t>Peak 140K tok/s on weekdays</w:t></w:r></w:p></w:body></w:document>"
    assert "Peak 140K tok/s" in ingest._docx_text(_docx(xml))


def test_docx_zip_bomb_refused(monkeypatch):
    monkeypatch.setattr(ingest, "DOCX_XML_MAX_BYTES", 1_000_000)
    bomb = _docx(W + b"<w:p><w:r><w:t>" + b"A" * 5_000_000 + b"</w:t></w:r></w:p></w:body></w:document>")
    assert len(bomb) < 50_000                     # tiny upload, ~5 MB once decompressed
    with pytest.raises(ingest.DescribeError) as e:
        ingest._docx_text(bomb)
    assert e.value.status == 413
    with pytest.raises(ingest.DescribeError) as e:   # the same through the upload path, not masked as "unreadable"
        ingest._document_part(bomb, "plan.docx", None)
    assert e.value.status == 413


# ---- second review round ------------------------------------------------------------------------------------------
def test_resilience_peak_uses_the_whole_week():
    from aiops_dilemma import costmodel, traffic
    cfg = config.normalize({"name": "t", "model": {"model_garden_id": "x"}, "requests": {"avg_input_tokens": 1000},
                            "maas": {"max_rps": 1}})
    week = [1000.0] * 168
    week[24 * 3 + 15] = 50_000.0                     # Thursday 15:00 is the busiest hour of the week
    cfg["traffic"] = {"custom_hourly": week}
    s = costmodel.resilience_and_hybrid_summary(cfg, 9000.0, {"vertex": 5.0}, "vertex", 1)
    peak = s["incidents"][0]
    assert peak["scenario"].endswith("peak hour (1h)")
    assert peak["tokens_m"] >= (50_000 - 1000) * 3600 / 1e6 * 0.99, peak   # Monday-only would see ~0


def test_null_max_rps_is_no_overflow_not_a_crash():
    from aiops_dilemma import costmodel
    cfg = config.normalize({"name": "t", "model": {"model_garden_id": "x"}, "maas": {"max_rps": None},
                            "traffic": {"custom_hourly": [5000.0] * 168}})
    assert cfg["maas"]["max_rps"] == 0
    assert config.normalize({"maas": {"max_rps": 10}})["maas"]["max_rps"] == 10      # stays an int in YAML
    costmodel.resilience_and_hybrid_summary(cfg, 9000.0, {"vertex": 5.0}, "vertex", 1)


def test_invalid_region_is_reported_not_silently_replaced(tmp_path, monkeypatch, capsys):
    cfg = config.normalize({"region": "us-central-1"})
    assert cfg["region"] == "us-central1" and "us-central-1" in cfg["_region_warning"]
    assert "_region_warning" not in config.normalize({"region": "europe-west4"})
    assert "_region_warning" not in config.normalize({})                     # missing = default, no warning
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(server.app)
    r = c.post("/api/sessions/s1/yaml", json={"yaml_text": "region: us-central-1\n"})
    assert r.status_code == 400 and "isn't a Google Cloud region" in r.json()["detail"]
    p = tmp_path / "w.yaml"
    p.write_text("name: w\nregion: US-CENTRAL1\nmodel: {model_garden_id: x}\n")
    config.load(str(p))
    assert "isn't a Google Cloud region" in capsys.readouterr().err


def test_run_dir_name_cannot_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("DILEMMA_RUNS_DIR", str(tmp_path / "runs"))
    for bad in ("../../etc", "a/b", "..", ".", "", "x" * 101):
        with pytest.raises(SystemExit):
            config.run_dir({"name": bad})
    assert config.run_dir({"name": "ok-run_1.v2"}) == tmp_path / "runs" / "ok-run_1.v2"


def test_signal_job_refuses_dangerous_pids(tmp_path, monkeypatch):
    import os
    from aiops_dilemma import jobs
    for pid in (0, 1, -1, os.getpid()):
        monkeypatch.setattr(jobs, "live_job", lambda rd, p=pid: {"pid": p, "status": "running"})
        with pytest.raises(ValueError):
            jobs.signal_job(tmp_path)


def test_partial_endpoint_is_torn_down_and_redeployed(tmp_path, monkeypatch):
    import json
    from aiops_dilemma import jobs
    rd = tmp_path / "runs" / "p1"
    rd.mkdir(parents=True)
    (rd / "endpoint.json").write_text(json.dumps({"endpoint": "projects/1/locations/us-central1/endpoints/9",
                                                  "region": "us-central1", "partial": True}))
    calls = []
    monkeypatch.setattr(deploy, "teardown", lambda cfg, d: (calls.append("teardown"), (d / "endpoint.json").unlink()))
    monkeypatch.setattr(deploy, "deploy", lambda cfg, plan, d: calls.append("deploy") or {"endpoint": "new"})
    (rd / "plan.json").write_text(json.dumps({"chosen": {}}))
    cfg = config.normalize({"name": "p1", "model": {"model_garden_id": "google/gemma"}})
    cfg["_run_dir"] = str(rd)
    rec = jobs.JobRecorder(rd, steps=["deploy"], mode="step", source="cli", command="test").start()
    jobs.run_steps(cfg, rd, ["deploy"], rec, mode="step", keep_endpoint=True)
    assert calls == ["teardown", "deploy"]


def test_docx_tables_read_once_and_size_checked_first(monkeypatch):
    xml = (W + b"<w:p><w:r><w:t>Intro</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Peak</w:t></w:r></w:p></w:tc>"
           b"<w:tc><w:p><w:r><w:t>140K</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>")
    text = ingest._docx_text(_docx(xml))
    assert text.splitlines() == ["Intro", "Peak\t140K"]
    # the size check runs before any parser, python-docx included
    import sys
    import types
    monkeypatch.setitem(sys.modules, "docx", types.SimpleNamespace(Document=lambda *a: (_ for _ in ()).throw(
        AssertionError("python-docx parsed an oversized file"))))
    monkeypatch.setattr(ingest, "DOCX_XML_MAX_BYTES", 1_000)
    with pytest.raises(ingest.DescribeError):
        ingest._docx_text(_docx(W + b"<w:p><w:r><w:t>" + b"A" * 5_000 + b"</w:t></w:r></w:p></w:body></w:document>"))
