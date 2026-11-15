"""Time-series traffic import (ingest.parse_timeseries / build_traffic_proposal, /traffic/import + /traffic/apply)
and the Cloud Monitoring connector (monitoring.py, HTTP mocked). All data is synthetic (tests/traffic_fixtures.py).
"""

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

import traffic_fixtures as fx
from aiops_dilemma import ingest, monitoring, server

UTC = timezone.utc
CTX = {"requests": {"avg_input_tokens": 6000, "avg_output_tokens": 300}, "weekend_factor": 0.15}


def _series(files: dict) -> list[dict]:
    series, other = ingest.import_timeseries_files([(k, v.encode()) for k, v in files.items()])
    assert not other
    return series


# ---------------------------------------------------------------------------------------------------- timestamps
@pytest.mark.parametrize("text,expected,assumed", [
    ("Sun Nov 08 2026 21:50:00 GMT+0200 (Example Standard Time)", datetime(2026, 11, 8, 19, 50, tzinfo=UTC), False),
    ("Mon Nov 09 2026 01:05:00 GMT-0500 (Eastern Standard Time)", datetime(2026, 11, 9, 6, 5, tzinfo=UTC), False),
    ("Sun Nov 08 2026 21:50:00 GMT+05:30", datetime(2026, 11, 8, 16, 20, tzinfo=UTC), False),
    ("2026-11-08T19:50:00Z", datetime(2026, 11, 8, 19, 50, tzinfo=UTC), False),
    ("2026-11-08T21:50:00.123+02:00", datetime(2026, 11, 8, 19, 50, 0, 123000, tzinfo=UTC), False),
    ("2026-11-08 19:50", datetime(2026, 11, 8, 19, 50, tzinfo=UTC), True),
    ("2026-11-08 19:50:30", datetime(2026, 11, 8, 19, 50, 30, tzinfo=UTC), True),
    ("2026/11/08 19:50:00", datetime(2026, 11, 8, 19, 50, tzinfo=UTC), True),
    ("2026-11-08 19:50:00.123456789 UTC", datetime(2026, 11, 8, 19, 50, 0, 123456, tzinfo=UTC), False),
    ("Sun, 08 Nov 2026 19:50:00 GMT", datetime(2026, 11, 8, 19, 50, tzinfo=UTC), False),
    ("1794167400", datetime.fromtimestamp(1794167400, UTC), False),
    ("1794167400000", datetime.fromtimestamp(1794167400, UTC), False),
])
def test_parse_time_formats(text, expected, assumed):
    dt, a = ingest.parse_time(text)
    assert dt == expected and a == assumed


@pytest.mark.parametrize("text", ["", "hello", "12", "0.5", "global", "TimeSeries ID"])
def test_parse_time_rejects(text):
    assert ingest.parse_time(text)[0] is None


# ---------------------------------------------------------------------------------------------------- parsing
def test_metrics_explorer_csv_parsed_with_safe_labels():
    pts = fx.points(fx.START_PARTIAL, 16, gaps=[(5, 7)])
    text = fx.metrics_explorer_csv("qps", pts)
    [s] = ingest.parse_timeseries("Model_requests_per_second_(QPS).csv", text.encode())
    assert s["labels"] == {"location": "global", "model_user_id": "example-model"}     # no project / series id
    assert "my-gcp-project" not in json.dumps(ingest.series_to_json([s]))
    assert s["step_s"] == 300 and len(s["points"]) == len(pts)
    assert s["points"][0][0] == fx.START_PARTIAL                                         # GMT+0200 -> UTC
    assert s["points"][0][1] == pytest.approx(pts[0][1])
    assert ingest.classify_series(s)[0] == "requests"


def test_label_order_and_token_series_is_total():
    files = fx.partial_files()
    series = _series(files)
    guesses = {s["file"]: (s["metric_guess"], s["confidence"]) for s in series}
    assert guesses["Model_requests_per_second_(QPS)_example.csv"][0] == "requests"
    assert guesses["Token_count_by_model_(per_second)_example.csv"] == ("total_tps", "medium")
    tok = next(s for s in series if s["metric_guess"] == "total_tps")
    assert "input + output" in tok["reason"] and not tok["per_interval"]


def test_multi_series_columns_are_summed():
    pts = fx.points(fx.START_PARTIAL, 6)
    series = _series({"QPS multi.csv": fx.multi_series_csv(pts)})
    assert len(series) == 2 and {s["labels"]["location"] for s in series} == {"us-central1", "europe-west4"}
    res = ingest.build_traffic_proposal(series, context=CTX)
    # Both are requests/s: input = (0.75 + 0.25) x rps x configured 6,000 tokens per request.
    single = ingest.build_traffic_proposal(_series({"QPS.csv": fx.metrics_explorer_csv("qps", pts)}), context=CTX)
    assert res["options"][0]["custom_hourly"] == pytest.approx(single["options"][0]["custom_hourly"], rel=1e-6)


def test_column_blocks_with_type_labels():
    pts = fx.points(fx.START_WEEKS, 30)
    series = _series({"tokens.csv": fx.block_series_csv(pts)})
    assert [s["metric_guess"] for s in series] == ["input_tps", "output_tps"]
    assert all(s["confidence"] == "high" for s in series)
    res = ingest.build_traffic_proposal(series, context=CTX)
    assert res["proposal"]["output_ratio"] == pytest.approx(200 / fx.INPUT_PER_REQUEST, rel=1e-3)


def test_iso_header_csv_and_column_name_classification():
    pts = fx.points(fx.START_PARTIAL, 3, step_min=1)
    [s] = _series({"export.csv": fx.iso_csv(pts, "Input tokens per second")})
    assert (s["metric_guess"], s["confidence"]) == ("input_tps", "high") and s["step_s"] == 60
    [s] = _series({"input_tokens_per_sec.csv": fx.iso_csv(pts)})      # column "value": the file name decides
    assert (s["metric_guess"], s["confidence"]) == ("input_tps", "medium")


@pytest.mark.parametrize("name,labels,expected", [
    ("Model requests per second (QPS)", {}, "requests"),
    ("prediction_count", {}, "requests"),
    ("model invocation count", {}, "requests"),
    ("Token count by model (per second)", {}, "total_tps"),
    ("prompt_tokens_per_second", {}, "input_tps"),
    ("completion tokens/s", {}, "output_tps"),
    ("token_count", {"type": "input"}, "input_tps"),
    ("token_count", {"type": "cache_read_input"}, "input_tps"),
    ("token_count", {"type": "output"}, "output_tps"),
    ("tps", {}, "input_tps"),
    ("latency_ms", {}, "unknown"),
])
def test_classification(name, labels, expected):
    s = {"name": name, "column": name, "labels": labels, "file": "export.csv"}
    assert ingest.classify_series(s)[0] == expected


def test_request_logs_and_hourly_rows_keep_their_paths(tmp_path):
    # Per-request logs: irregular timestamps + input_tokens columns -> not a series; Case 2 still derives traffic.
    rng = np.random.default_rng(1)
    t0 = datetime(2026, 11, 2, tzinfo=UTC)
    times = sorted(t0 + timedelta(seconds=float(x)) for x in rng.uniform(0, 3 * 86400, 400))
    logs = "timestamp,input_tokens,output_tokens\n" + "\n".join(
        f"{t.isoformat()},{int(rng.integers(500, 9000))},{int(rng.integers(50, 500))}" for t in times)
    assert ingest.parse_timeseries("logs.csv", logs.encode()) == []
    hourly = "input_tokens_per_sec\n" + "\n".join(str(1000 + 10 * h) for h in range(168))
    assert ingest.parse_timeseries("hourly.csv", hourly.encode()) == []
    daily = "tps\n" + "\n".join(str(100 + h) for h in range(24))
    qps = fx.partial_files()
    res = ingest.ingest_files([("logs.csv", logs.encode(), "auto"), ("hourly.csv", hourly.encode(), "auto"),
                               (next(iter(qps)), next(iter(qps.values())).encode(), "auto")], tmp_path)
    kinds = {f["filename"]: f["detected_type"] for f in res["files"]}
    assert kinds == {"logs.csv": "token_logs", "hourly.csv": "hourly_traffic", next(iter(qps)): "timeseries"}
    assert res["timeseries_files"] == [next(iter(qps))]
    assert res["traffic"]["profile"][:2] == [1000.0, 1010.0] and res["requests"]["sample_count"] == 400
    res = ingest.ingest_files([("daily.csv", daily.encode(), "auto")], tmp_path)
    assert len(res["traffic"]["profile"]) == 168 and res["traffic"]["profile"][24] == 100.0


# ---------------------------------------------------------------------------------------------------- resampling
def test_hourly_means_are_time_weighted_and_gaps_missing():
    t = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    # 45-minute points labelled by interval end: 00:45 covers 00:00-00:45, 01:30 covers 00:45-01:30.
    hm = ingest.hourly_means([(t + timedelta(minutes=45), 10.0), (t + timedelta(minutes=90), 30.0)], 2700)
    assert hm[t] == pytest.approx(((10 * 45 + 30 * 15) / 60, 3600))
    assert hm[t + timedelta(hours=1)] == pytest.approx((30.0, 1800))            # half the hour covered
    pts = [(t + timedelta(minutes=5 * i), 1.0) for i in range(1, 13)] + \
          [(t + timedelta(hours=3, minutes=5 * i), 3.0) for i in range(1, 13)]
    hm = ingest.hourly_means(pts, 300)
    assert set(hm) == {t, t + timedelta(hours=3)}        # hours 1-2 missing, not zero


def test_typical_week_from_two_weeks():
    series = _series(fx.weeks_files(14))
    res = ingest.build_traffic_proposal(series, context=CTX)
    assert res["option"] == "typical_week" and res["coverage"]["span_hours"] == 336
    wk = np.array(res["proposal"]["custom_hourly"])
    expected = np.array([np.mean([fx.rps_at(fx.START_WEEKS + timedelta(days=d, hours=h, minutes=m))
                                  for m in range(5, 65, 5)]) for d in range(7) for h in range(24)]) * 6000
    # Monday 00:00 UTC first; each 5-minute point stands for the interval that ends at its timestamp.
    assert abs(int(np.argmax(wk)) - int(np.argmax(expected))) <= 1 and wk[:120].max() > 4 * wk[120:].max()
    assert np.allclose(wk[:-1], expected[:-1], rtol=0.02)   # Sun 23h: the export ends 5 minutes early
    assert res["coverage"]["gap_hours"] >= 2


def test_partial_coverage_options():
    series = _series(fx.partial_files())
    cur = [1000.0] * 120 + [200.0] * 48
    res = ingest.build_traffic_proposal(series, context=dict(CTX, current_profile=cur))
    assert res["warnings"][0].startswith("Only 16 h of data (Sun 19:50 – Mon 11:40 UTC)")
    assert [o["key"] for o in res["options"]] == ["daily_shape", "level_only"] and res["option"] == "daily_shape"
    a = np.array(res["options"][0]["custom_hourly"])
    assert np.allclose(a[:24], a[24:48]) and np.allclose(a[120:144], a[:24] * 0.15, rtol=1e-3)   # weekend = weekday x factor
    assert not np.isnan(a).any() and a.min() > 0                                       # missing hours interpolated
    b = np.array(res["options"][1]["custom_hourly"])
    assert np.allclose(b / b[0], np.array(cur) / cur[0], rtol=1e-3)                             # current shape, new level
    assert res["coverage"]["span_hours"] == 16 and res["coverage"]["hours"] == 15
    res_b = ingest.build_traffic_proposal(series, context=dict(CTX, current_profile=cur), option="level_only")
    assert res_b["proposal"]["peak_input_tps"] == res["options"][1]["peak_input_tps"]
    res_z = ingest.build_traffic_proposal(series, context=CTX, gaps_as_zero=True)
    assert min(res_z["proposal"]["custom_hourly"][:24]) == 0.0


def test_days_without_weekend_use_weekend_factor():
    pts = fx.points(datetime(2026, 11, 3, tzinfo=UTC), 72)            # Tue-Thu
    res = ingest.build_traffic_proposal(_series({"qps.csv": fx.metrics_explorer_csv("qps", pts)}), context=CTX)
    assert res["option"] == "weekday_weekend" and "no weekend data" in res["options"][0]["description"]
    wk = np.array(res["proposal"]["custom_hourly"])
    assert np.allclose(wk[120:168], np.tile(wk[:24], 2) * 0.15, rtol=1e-3)


# ---------------------------------------------------------------------------------------------------- combining
def test_qps_and_total_tokens_give_tokens_per_request():
    res = ingest.build_traffic_proposal(_series(fx.partial_files()), context=CTX)
    r_in = 6000 / 6300
    [d] = res["derived"]
    assert d["path"] == "requests.avg_input_tokens" and d["derived_from_config"]
    assert d["value"] == pytest.approx(fx.TOKENS_PER_REQUEST * r_in, rel=1e-3)
    assert res["proposal"]["input_derived"] and any("input + output" in w for w in res["warnings"])
    # Input week = total x configured input share (not requests x 6,000).
    qps_only = ingest.build_traffic_proposal([s for s in _series(fx.partial_files()) if s["metric_guess"] == "requests"],
                                             context=CTX)
    assert res["proposal"]["average_input_tps"] == pytest.approx(
        qps_only["proposal"]["average_input_tps"] * fx.TOKENS_PER_REQUEST * r_in / 6000, rel=1e-3)
    assert any("Only requests/s" in w for w in qps_only["warnings"])


def test_qps_with_input_and_output_series():
    pts = fx.points(fx.START_WEEKS, 30)
    series = _series({"tokens.csv": fx.block_series_csv(pts), "QPS.csv": fx.metrics_explorer_csv("qps", pts)})
    res = ingest.build_traffic_proposal(series, context=CTX)
    got = {d["path"]: d for d in res["derived"]}
    assert got["requests.avg_input_tokens"]["value"] == fx.INPUT_PER_REQUEST
    assert not got["requests.avg_input_tokens"]["derived_from_config"]
    assert got["requests.avg_output_tokens"]["value"] == 200
    # Remapping: treat the input series as total tokens and the output as output -> input = total - output.
    ids = {s["metric_guess"]: s["id"] for s in series}
    res2 = ingest.build_traffic_proposal(series, context=CTX, mapping={ids["input_tps"]: "total_tps"})
    assert res2["proposal"]["basis"].startswith("total − output")
    assert res2["proposal"]["average_input_tps"] == pytest.approx(
        res["proposal"]["average_input_tps"] * (fx.INPUT_PER_REQUEST - 200) / fx.INPUT_PER_REQUEST, rel=1e-3)
    res3 = ingest.build_traffic_proposal(series, context=CTX, mapping={s["id"]: "ignore" for s in series})
    assert res3["error"] and res3["proposal"] is None


# ---------------------------------------------------------------------------------------------------- API round trip
@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", runs)
    c = TestClient(server.app)
    assert c.post("/api/sessions/ts-check", json={"config": {}}).status_code == 200
    return c


def test_import_then_apply_writes_custom_hourly(client, tmp_path):
    files = [("files", (k, v.encode(), "text/csv")) for k, v in fx.partial_files().items()]
    r = client.post("/api/sessions/ts-check/traffic/import", files=files, data={"context": json.dumps(CTX)})
    assert r.status_code == 200, r.text
    res = r.json()
    assert {s["metric_guess"] for s in res["series"]} == {"requests", "total_tps"}
    assert all({"id", "file", "confidence", "reason", "labels", "points_n", "first", "last", "step_s",
                "coverage_h"} <= set(s) for s in res["series"])
    assert res["coverage"]["label"].startswith("16 h of data")
    assert [o["key"] for o in res["options"]][0] == "daily_shape" and len(res["proposal"]["custom_hourly"]) == 168
    stored = (tmp_path / "runs" / "ts-check" / "traffic_import.json").read_text()
    assert "my-gcp-project" not in stored
    # Nothing applied yet.
    cfg = yaml.safe_load((tmp_path / "runs" / "ts-check" / "workload.yaml").read_text())
    assert not cfg["traffic"].get("custom_hourly")
    # Re-map without re-uploading.
    tok = next(s["id"] for s in res["series"] if s["metric_guess"] == "total_tps")
    r = client.post("/api/sessions/ts-check/traffic/import", json={"mapping": {tok: "ignore"}, "context": CTX})
    assert r.status_code == 200 and not r.json()["derived"]
    # Apply option (a) with the derived tokens per request.
    r = client.post("/api/sessions/ts-check/traffic/apply", json={
        "option": "daily_shape", "accept": ["requests.avg_input_tokens"], "context": CTX})
    assert r.status_code == 200, r.text
    out = r.json()
    cfg = out["session"]["config"]
    assert cfg["traffic"]["custom_hourly"] == pytest.approx(res["proposal"]["custom_hourly"], rel=1e-6)
    assert cfg["traffic"]["peak_input_tps"] == pytest.approx(res["proposal"]["peak_input_tps"], rel=1e-3)
    assert cfg["requests"]["avg_input_tokens"] == res["derived"][0]["value"]
    assert out["session"]["traffic_profile"][:3] == pytest.approx(res["proposal"]["custom_hourly"][:3], rel=1e-3)


def test_import_rejects_non_timeseries_and_generic_upload_routes(client):
    r = client.post("/api/sessions/ts-check/traffic/import",
                    files=[("files", ("hourly.csv", b"tps\n1\n2\n3\n", "text/csv"))])
    assert r.status_code == 422
    k, v = next(iter(fx.partial_files().items()))
    r = client.post("/api/sessions/ts-check/upload", files=[("files", (k, v.encode(), "text/csv"))])
    assert r.status_code == 200 and r.json()["ingested"]["timeseries_files"] == [k]
    assert not r.json()["ingested"]["traffic"]
    assert client.post("/api/sessions/blog-example/traffic/apply", json={}).status_code in (403, 404)


# ---------------------------------------------------------------------------------------------------- monitoring
DESCRIPTORS = {"metricDescriptors": [
    {"type": monitoring.PUB_REQUESTS, "metricKind": "DELTA", "valueType": "INT64",
     "labels": [{"key": "response_code"}], "monitoredResourceTypes": [monitoring.PUB_RESOURCE]},
    {"type": monitoring.PUB_TOKENS, "metricKind": "DELTA", "valueType": "INT64",
     "labels": [{"key": "type"}], "monitoredResourceTypes": [monitoring.PUB_RESOURCE]},
    {"type": monitoring.EP_REQUESTS, "metricKind": "DELTA", "valueType": "INT64",
     "labels": [{"key": "endpoint_display_name"}], "monitoredResourceTypes": [monitoring.EP_RESOURCE]},
    {"type": "aiplatform.googleapis.com/prediction/gdc/input_tokens_per_minute", "metricKind": "DELTA",
     "valueType": "DISTRIBUTION", "monitoredResourceTypes": [monitoring.EP_RESOURCE]},
    {"type": "aiplatform.googleapis.com/quota/x/usage", "metricKind": "DELTA", "valueType": "INT64"},
]}


def _ts(resource: dict, metric: dict, values: list[tuple[datetime, float]]) -> dict:
    return {"resource": {"labels": resource}, "metric": {"labels": metric}, "valueType": "DOUBLE",
            "points": [{"interval": {"endTime": t.strftime("%Y-%m-%dT%H:%M:%SZ")}, "value": {"doubleValue": v}}
                       for t, v in reversed(values)]}


def _mock(calls: list, status: int = 200, body: dict | None = None):
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    hrs = [end - timedelta(hours=h) for h in (3, 2, 1)]

    def handler(req: httpx.Request):
        q = parse_qs(urlparse(str(req.url)).query)
        calls.append((req.url.path, q))
        if status != 200:
            return httpx.Response(status, json=body or {})
        if req.url.path.endswith("/metricDescriptors"):
            return httpx.Response(200, json=DESCRIPTORS)
        flt = q["filter"][0]
        if monitoring.PUB_REQUESTS in flt:
            out = [_ts({"publisher": "google", "model_user_id": "example-model", "location": "global"}, {},
                       [(t, 2.0) for t in hrs])]
            if "model_user_id" not in flt:
                out.append(_ts({"publisher": "google", "model_user_id": "idle-model", "location": "global"}, {},
                               [(hrs[0], 0.0)]))
            return httpx.Response(200, json={"timeSeries": out})
        if monitoring.PUB_TOKENS in flt:
            return httpx.Response(200, json={"timeSeries": [
                _ts({}, {"type": "input"}, [(t, 1000.0) for t in hrs]),
                _ts({}, {"type": "cache_read_input"}, [(t, 3000.0) for t in hrs]),
                _ts({}, {"type": "output"}, [(t, 400.0) for t in hrs])]})
        if monitoring.EP_REQUESTS in flt:
            return httpx.Response(200, json={"timeSeries": [
                _ts({"endpoint_id": "1234567890", "location": "us-central1"}, {"endpoint_display_name": "my-endpoint"},
                    [(hrs[1], 0.5)]),
                _ts({"endpoint_id": "gemini-x", "location": "global"}, {}, [(hrs[1], 9.0)])]})
        return httpx.Response(200, json={})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_monitoring_discovery_and_sources():
    calls = []
    res = monitoring.list_sources("my-gcp-project", 7, client=_mock(calls))
    assert [s["id"] for s in res["sources"]] == ["publisher:google/example-model@global",
                                                 "endpoint:1234567890@us-central1"]
    pub, ep = res["sources"]
    assert pub["total_requests"] == 3 * 2 * 3600 and pub["active_hours"] == 3
    assert ep["name"] == "my-endpoint" and ep["total_requests"] == 1800
    assert res["days"] == 7 and monitoring.PUB_TOKENS in res["metrics"]
    path, q = calls[0]
    assert path.endswith("/projects/my-gcp-project/metricDescriptors")
    assert q["filter"] == ['metric.type = starts_with("aiplatform.googleapis.com/")']
    _, q = calls[1]
    assert q["aggregation.alignmentPeriod"] == ["3600s"] and q["aggregation.perSeriesAligner"] == ["ALIGN_RATE"]
    assert q["aggregation.crossSeriesReducer"] == ["REDUCE_SUM"]
    assert set(q["aggregation.groupByFields"]) == {"resource.label.publisher", "resource.label.model_user_id",
                                                   "resource.label.location"}
    start = datetime.fromisoformat(q["interval.startTime"][0].replace("Z", "+00:00"))
    end = datetime.fromisoformat(q["interval.endTime"][0].replace("Z", "+00:00"))
    assert end - start == timedelta(days=7)


def test_monitoring_query_building():
    t = datetime(2026, 11, 1, tzinfo=UTC)
    q = monitoring.build_query("m/x", t, t + timedelta(days=28), "ALIGN_MEAN", ["metric.label.type"],
                               'resource.label.location = "global"')
    assert q["filter"] == 'metric.type = "m/x" AND resource.label.location = "global"'
    assert q["interval.startTime"] == "2026-11-01T00:00:00Z" and q["interval.endTime"] == "2026-11-29T00:00:00Z"
    assert q["aggregation.perSeriesAligner"] == "ALIGN_MEAN" and q["aggregation.groupByFields"] == ["metric.label.type"]
    assert monitoring._aligner({"kind": "GAUGE"}) == "ALIGN_MEAN" and monitoring._aligner({"kind": "DELTA"}) == "ALIGN_RATE"


def test_monitoring_fetch_series_and_proposal():
    calls = []
    got = monitoring.fetch_series("my-gcp-project", "publisher:google/example-model@global", 7, client=_mock(calls))
    by = {s["metric_guess"]: s for s in got["series"]}
    assert set(by) == {"requests", "input_tps", "output_tps"}
    assert [v for _, v in by["input_tps"]["points"]] == [4000.0] * 3          # input + cache reads
    assert by["input_tps"]["cached_share"] == 0.75 and "cache_read_input + input" in by["input_tps"]["reason"]
    tok_q = next(q for _, q in calls if monitoring.PUB_TOKENS in q.get("filter", [""])[0])
    assert 'resource.label.model_user_id = "example-model"' in tok_q["filter"][0]
    assert tok_q["aggregation.groupByFields"] == ["metric.label.type"]
    res = ingest.build_traffic_proposal(got["series"], context=CTX, missing_is_zero=True, window=got["window"])
    d = {x["path"]: x["value"] for x in res["derived"]}
    assert d == {"requests.avg_input_tokens": 2000, "requests.avg_output_tokens": 200,
                 "requests.cached_input_tokens": 1500}   # 75% of input tokens were cache reads
    assert res["option"] == "typical_week" and res["coverage"]["hours"] == 7 * 24       # missing hours = zero
    assert any("cache reads" in w for w in res["warnings"])


@pytest.mark.parametrize("status,body,code,text", [
    (403, {"error": {"message": "Permission monitoring.metricDescriptors.list denied"}}, "permission_denied",
     "roles/monitoring.viewer"),
    (403, {"error": {"message": "Cloud Monitoring API has not been used in project", "status": "PERMISSION_DENIED",
                     "details": [{"reason": "SERVICE_DISABLED"}]}}, "api_disabled", "monitoring.googleapis.com"),
    (401, {}, "not_signed_in", "Sign in"),
])
def test_monitoring_errors(status, body, code, text, monkeypatch):
    monkeypatch.setattr(monitoring, "_identity", lambda: "someone@example.com")
    with pytest.raises(monitoring.MonitoringError) as e:
        monitoring.list_sources("my-gcp-project", 28, client=_mock([], status, body))
    assert e.value.code == code and text in str(e.value) and e.value.status == status


def test_monitoring_no_data(monkeypatch):
    def handler(req):
        if req.url.path.endswith("/metricDescriptors"):
            return httpx.Response(200, json=DESCRIPTORS)
        return httpx.Response(200, json={})
    with pytest.raises(monitoring.MonitoringError) as e:
        monitoring.fetch_series("my-gcp-project", "publisher:google/none@global", 7,
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert e.value.code == "no_data" and e.value.status == 404
    with pytest.raises(monitoring.MonitoringError):
        monitoring.parse_source("bogus")


def test_monitoring_endpoints_with_stub(client, monkeypatch):
    monkeypatch.setenv("DILEMMA_TEST_MONITORING_STUB", "1")
    r = client.get("/api/monitoring/sources", params={"project": "my-gcp-project", "days": 14})
    assert r.status_code == 200 and r.json()["days"] == 14 and len(r.json()["sources"]) == 2
    src = r.json()["sources"][0]["id"]
    r = client.post("/api/monitoring/import", json={"project": "my-gcp-project", "source": src, "days": 28,
                                                    "session": "ts-check", "context": CTX})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["option"] == "typical_week" and res["source"]["kind"] == "monitoring"
    assert {d["path"]: d["value"] for d in res["derived"]} == {"requests.avg_input_tokens": 2000,
                                                               "requests.avg_output_tokens": 150}
    r = client.post("/api/sessions/ts-check/traffic/apply", json={"accept": ["requests.avg_output_tokens"]})
    cfg = r.json()["session"]["config"]
    assert len(cfg["traffic"]["custom_hourly"]) == 168 and cfg["requests"]["avg_output_tokens"] == 150
    assert cfg["requests"]["avg_input_tokens"] == 6000
    assert client.post("/api/monitoring/import", json={"source": src}).status_code == 400


def test_custom_week_with_zero_hours_is_kept(client):
    """A quietest hour of 0 (sporadic traffic, gaps as zero) must not be replaced by the default week."""
    week = [0.0] * 168
    for d in range(5):
        for h in range(9, 17):
            week[d * 24 + h] = 1000.0
    r = client.post("/api/sessions/zero-floor", json={"config": {"name": "zero-floor", "traffic": {"custom_hourly": week}}})
    assert r.status_code == 200, r.text
    s = client.get("/api/sessions/zero-floor").json()
    assert s["traffic_profile"] == pytest.approx(week)
    assert s["config"]["traffic"]["floor_input_tps"] == 0
