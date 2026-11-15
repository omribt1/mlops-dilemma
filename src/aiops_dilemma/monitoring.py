"""Import traffic from Cloud Monitoring (API v3, REST via httpx, Application Default Credentials).

Sources are the models and endpoints that served requests in the window:

- Model Garden / MaaS publisher models (Gemini, Claude, open models as a service), monitored resource
  `aiplatform.googleapis.com/PublisherModel` (labels publisher, model_user_id, location):
  `publisher/online_serving/model_invocation_count` (requests) and `publisher/online_serving/token_count`
  (metric label `type`: input, output, cache_read_input, cache_write_input, cache_write_1h_input, ...). Every
  type except output is prompt tokens the model had to read, so input tok/s = sum of the non-output types.
  These are the metrics behind Metrics Explorer's "Model requests per second (QPS)" and "Token count by model".
- Dedicated / custom endpoints, resource `aiplatform.googleapis.com/Endpoint` (endpoint_id, location):
  `prediction/online/prediction_count`, plus any non-distribution token metric on that resource the project
  exposes (input/prompt vs output/generation by name).

Metric names are discovered with `metricDescriptors.list` (filtered to `aiplatform.googleapis.com/`), and series are
read with `timeSeries.list` aligned to 1 hour (ALIGN_RATE for DELTA/CUMULATIVE, ALIGN_MEAN for GAUGE), summed per
source (REDUCE_SUM). Delta counters write no points for hours without calls, so missing hours are zero traffic.
Needs `roles/monitoring.viewer` on the project. Project IDs are never logged or stored by this module.
`DILEMMA_TEST_MONITORING_STUB=1` (tests and UI checks only) serves synthetic sources and a synthetic week.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timedelta, timezone

import httpx

API = "https://monitoring.googleapis.com/v3"
PREFIX = "aiplatform.googleapis.com/"
PUB_REQUESTS = PREFIX + "publisher/online_serving/model_invocation_count"
PUB_TOKENS = PREFIX + "publisher/online_serving/token_count"
EP_REQUESTS = PREFIX + "prediction/online/prediction_count"
PUB_RESOURCE = PREFIX + "PublisherModel"
EP_RESOURCE = PREFIX + "Endpoint"
DAYS_CHOICES = (7, 14, 28)
ROLE = "roles/monitoring.viewer"


class MonitoringError(Exception):
    """User-facing failure; `status` is the HTTP status the server returns."""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def _stub() -> bool:
    return os.environ.get("DILEMMA_TEST_MONITORING_STUB", "") not in ("", "0")


def _client(project: str | None = None) -> httpx.Client:
    """An httpx client with an ADC bearer token and `project` as quota project (tests monkeypatch this with a
    MockTransport client)."""
    from aiops_dilemma import auth

    try:
        headers = auth.auth_headers(project)
    except Exception as e:
        raise MonitoringError("Not signed in to Google Cloud. Sign in from the project menu to read Cloud Monitoring "
                              f"({str(e)[:120]}).", 401, "not_signed_in")
    return httpx.Client(headers=headers, timeout=60)


def _identity() -> str:
    try:
        from aiops_dilemma import auth

        st = auth.status() if hasattr(auth, "status") else {}
        return st.get("adc_account") or "the signed-in identity"
    except Exception:
        return "the signed-in identity"


def _raise_for(r: httpx.Response, project: str) -> None:
    if r.status_code < 400:
        return
    try:
        err = r.json().get("error", {})
    except ValueError:
        err = {}
    msg = str(err.get("message") or r.text or r.reason_phrase)[:300]
    body = r.text
    if r.status_code == 401:
        raise MonitoringError("Not signed in to Google Cloud (or the credentials expired). Sign in to read Cloud "
                              "Monitoring.", 401, "not_signed_in")
    if r.status_code == 403:
        if "SERVICE_DISABLED" in body or "has not been used" in body or "is disabled" in body:
            raise MonitoringError(f"The Cloud Monitoring API is disabled in project {project}. Enable "
                                  "monitoring.googleapis.com, then try again.", 403, "api_disabled")
        raise MonitoringError(f"Permission denied reading Cloud Monitoring in project {project}: {_identity()} needs "
                              f"the Monitoring Viewer role ({ROLE}) on the project.", 403, "permission_denied")
    if r.status_code == 404:
        raise MonitoringError(f"Project {project} was not found, or Cloud Monitoring has no workspace for it.", 404,
                              "not_found")
    if r.status_code == 400:
        raise MonitoringError(f"Cloud Monitoring rejected the query: {msg}", 400, "bad_query")
    raise MonitoringError(f"Cloud Monitoring error {r.status_code}: {msg}", 502, "upstream")


def _get(client: httpx.Client, project: str, path: str, params) -> dict:
    try:
        r = client.get(f"{API}/projects/{project}/{path}", params=params)
    except httpx.HTTPError as e:
        raise MonitoringError(f"Could not reach Cloud Monitoring: {str(e)[:160]}", 502, "network")
    _raise_for(r, project)
    return r.json()


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def window(days: int, now: datetime | None = None) -> tuple[datetime, datetime]:
    """[start, end) on whole hours, ending at the start of the current hour."""
    days = int(days) if int(days or 0) in DAYS_CHOICES else 28
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return end - timedelta(days=days), end


# ------------------------------------------------------------------------------------------------------- discovery
def _token_role(metric_type: str) -> str | None:
    t = metric_type.lower()
    if re.search(r"input|prompt", t):
        return "input_tps"
    if re.search(r"output|generation|completion", t):
        return "output_tps"
    return None


def discover_metrics(project: str, client: httpx.Client) -> dict:
    """Which serving metrics exist in this project: {type: {kind, value_type, resources, labels}} for the ones used."""
    descs, token = [], None
    for _ in range(20):
        params = {"filter": f'metric.type = starts_with("{PREFIX}")', "pageSize": 1000}
        if token:
            params["pageToken"] = token
        d = _get(client, project, "metricDescriptors", params)
        descs.extend(d.get("metricDescriptors") or [])
        token = d.get("nextPageToken")
        if not token:
            break
    found: dict = {"publisher_requests": None, "publisher_tokens": None, "endpoint_requests": None,
                   "endpoint_tokens": []}
    for m in descs:
        typ = m.get("type") or ""
        info = {"type": typ, "kind": m.get("metricKind"), "value_type": m.get("valueType"),
                "labels": [lb.get("key") for lb in m.get("labels") or []],
                "resources": m.get("monitoredResourceTypes") or []}
        if typ == PUB_REQUESTS:
            found["publisher_requests"] = info
        elif typ == PUB_TOKENS:
            found["publisher_tokens"] = info
        elif typ == EP_REQUESTS:
            found["endpoint_requests"] = info
        elif (EP_RESOURCE in info["resources"] and "token" in typ.lower() and info["value_type"] in ("INT64", "DOUBLE")
              and "/quota/" not in typ and _token_role(typ)):
            info["role"] = _token_role(typ)
            found["endpoint_tokens"].append(info)
    return found


def _aligner(info: dict | None) -> str:
    return "ALIGN_MEAN" if (info or {}).get("kind") == "GAUGE" else "ALIGN_RATE"


def build_query(metric: str, start: datetime, end: datetime, aligner: str, group_by: list[str],
                extra_filter: str = "", period_s: int = 3600) -> dict:
    """timeSeries.list query parameters: hourly alignment, summed across series within each group."""
    flt = f'metric.type = "{metric}"' + (f" AND {extra_filter}" if extra_filter else "")
    params = {"filter": flt, "interval.startTime": _ts(start), "interval.endTime": _ts(end),
              "aggregation.alignmentPeriod": f"{period_s}s", "aggregation.perSeriesAligner": aligner,
              "aggregation.crossSeriesReducer": "REDUCE_SUM", "pageSize": 1000}
    if group_by:
        params["aggregation.groupByFields"] = list(group_by)
    return params


def _time_series(client: httpx.Client, project: str, params: dict) -> list[dict]:
    out, token = [], None
    for _ in range(50):
        p = dict(params)
        if token:
            p["pageToken"] = token
        d = _get(client, project, "timeSeries", p)
        out.extend(d.get("timeSeries") or [])
        token = d.get("nextPageToken")
        if not token:
            break
    return out


def _val(v: dict) -> float:
    for k in ("doubleValue", "int64Value"):
        if k in v:
            try:
                f = float(v[k])
                return f if math.isfinite(f) else 0.0
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def parse_points(ts: dict) -> list[tuple[datetime, float]]:
    """[(interval end, value)] oldest first."""
    pts = []
    for p in ts.get("points") or []:
        t = (p.get("interval") or {}).get("endTime")
        if not t:
            continue
        dt = datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc)
        pts.append((dt, _val(p.get("value") or {})))
    return sorted(pts)


def _is_publisher_alias(endpoint_id: str, location: str) -> bool:
    """prediction_count also counts publisher-model calls under pseudo endpoint ids (e.g. gemini-3p5-flash in
    `global`); those are listed as publisher models instead."""
    return location == "global" or bool(re.match(r"^(gemini|claude|imagen|veo|chirp|text-|publishers)", endpoint_id))


# ------------------------------------------------------------------------------------------------------- public API
def list_sources(project: str, days: int = 28, client: httpx.Client | None = None) -> dict:
    """Models and endpoints with requests in the last `days`: {project, days, start, end, sources, metrics}."""
    if not project:
        raise MonitoringError("Choose a Google Cloud project first.", 400, "no_project")
    if _stub():
        return _stub_sources(project, days)
    start, end = window(days)
    own = client is None
    client = client or _client(project)
    try:
        metrics = discover_metrics(project, client)
        sources = []
        if metrics["publisher_requests"]:
            q = build_query(PUB_REQUESTS, start, end, _aligner(metrics["publisher_requests"]),
                            ["resource.label.publisher", "resource.label.model_user_id", "resource.label.location"])
            for ts in _time_series(client, project, q):
                rl = (ts.get("resource") or {}).get("labels") or {}
                pts = parse_points(ts)
                total = sum(v for _, v in pts) * 3600
                if total <= 0:
                    continue
                model, loc, pub = rl.get("model_user_id", "?"), rl.get("location", "?"), rl.get("publisher", "?")
                sources.append({"id": f"publisher:{pub}/{model}@{loc}", "kind": "publisher", "name": model,
                                "publisher": pub, "location": loc, "total_requests": round(total),
                                "active_hours": sum(1 for _, v in pts if v > 0), "has_tokens": bool(metrics["publisher_tokens"])})
        if metrics["endpoint_requests"]:
            group = ["resource.label.endpoint_id", "resource.label.location"]
            if "endpoint_display_name" in (metrics["endpoint_requests"].get("labels") or []):
                group.append("metric.label.endpoint_display_name")
            q = build_query(EP_REQUESTS, start, end, _aligner(metrics["endpoint_requests"]), group)
            eps: dict[str, dict] = {}
            for ts in _time_series(client, project, q):
                rl = (ts.get("resource") or {}).get("labels") or {}
                ml = (ts.get("metric") or {}).get("labels") or {}
                eid, loc = rl.get("endpoint_id", "?"), rl.get("location", "?")
                if _is_publisher_alias(eid, loc):
                    continue
                pts = parse_points(ts)
                sid = f"endpoint:{eid}@{loc}"
                e = eps.setdefault(sid, {"id": sid, "kind": "endpoint", "name": f"Endpoint {eid}", "endpoint_id": eid,
                                         "location": loc, "total_requests": 0.0, "hours": set(),
                                         "has_tokens": bool(metrics["endpoint_tokens"])})
                if ml.get("endpoint_display_name"):
                    e["name"] = ml["endpoint_display_name"]
                e["total_requests"] += sum(v for _, v in pts) * 3600
                e["hours"].update(dt for dt, v in pts if v > 0)
            for e in eps.values():
                if e["total_requests"] > 0:
                    e["total_requests"] = round(e["total_requests"])
                    e["active_hours"] = len(e.pop("hours"))
                    sources.append(e)
    finally:
        if own:
            client.close()
    sources.sort(key=lambda s: -s["total_requests"])
    found = [k for k in ("publisher_requests", "publisher_tokens", "endpoint_requests") if metrics.get(k)]
    return {"project": project, "days": int((end - start).days), "start": start.isoformat(), "end": end.isoformat(),
            "sources": sources, "metrics": [metrics[k]["type"] for k in found]
            + [m["type"] for m in metrics.get("endpoint_tokens") or []],
            "message": None if sources else f"No model or endpoint served requests in the last {(end - start).days} days."}


def parse_source(source: str) -> dict:
    m = re.match(r"^publisher:([^/]+)/(.+)@([^@]+)$", source or "")
    if m:
        return {"kind": "publisher", "publisher": m.group(1), "model": m.group(2), "location": m.group(3)}
    m = re.match(r"^endpoint:(.+)@([^@]+)$", source or "")
    if m:
        return {"kind": "endpoint", "endpoint_id": m.group(1), "location": m.group(2)}
    raise MonitoringError(f"Unknown source '{source}'. Pick one from the list.", 400, "bad_source")


def _q(v: str) -> str:
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def fetch_series(project: str, source: str, days: int = 28, client: httpx.Client | None = None) -> dict:
    """Hourly requests/s and input / output tokens/s of one source, as importer series (see ingest.parse_timeseries):
    {series, window: (start, end), source}."""
    if not project:
        raise MonitoringError("Choose a Google Cloud project first.", 400, "no_project")
    src = parse_source(source)
    start, end = window(days)
    if _stub():
        return _stub_series(src, start, end)
    own = client is None
    client = client or _client(project)
    series: list[dict] = []

    def add(name, role, pts, why):
        if pts:
            series.append({"id": f"m{len(series) + 1}", "file": "Cloud Monitoring", "name": name, "column": name,
                           "labels": {k: v for k, v in src.items() if k in ("publisher", "model", "location", "endpoint_id")},
                           "points": pts, "step_s": 3600, "assumed_utc": False, "metric_guess": role,
                           "confidence": "high", "reason": why, "per_interval": False})

    try:
        metrics = discover_metrics(project, client)
        if src["kind"] == "publisher":
            flt = (f"resource.label.model_user_id = {_q(src['model'])} AND resource.label.location = "
                   f"{_q(src['location'])} AND resource.label.publisher = {_q(src['publisher'])}")
            if metrics["publisher_requests"]:
                ts = _time_series(client, project, build_query(PUB_REQUESTS, start, end,
                                                               _aligner(metrics["publisher_requests"]), [], flt))
                add("Model requests per second", "requests", _sum_points(ts), f"metric {PUB_REQUESTS.split('/', 1)[1]}")
            if metrics["publisher_tokens"]:
                ts = _time_series(client, project, build_query(PUB_TOKENS, start, end,
                                                               _aligner(metrics["publisher_tokens"]),
                                                               ["metric.label.type"], flt))
                ins = [t for t in ts if ((t.get("metric") or {}).get("labels") or {}).get("type", "") != "output"]
                outs = [t for t in ts if ((t.get("metric") or {}).get("labels") or {}).get("type", "") == "output"]
                types = sorted({((t.get("metric") or {}).get("labels") or {}).get("type", "?") for t in ins})
                add("Input tokens per second", "input_tps", _sum_points(ins),
                    f"metric {PUB_TOKENS.split('/', 1)[1]}, type = {' + '.join(types)}")
                add("Output tokens per second", "output_tps", _sum_points(outs),
                    f"metric {PUB_TOKENS.split('/', 1)[1]}, type = output")
                # cache reads only: cache writes are billed at or above the input price, not as cheap reads
                cached = [t for t in ins if ((t.get("metric") or {}).get("labels") or {}).get("type", "") == "cache_read_input"]
                tot_in = sum(v for _, v in _sum_points(ins))
                if tot_in > 0 and cached:
                    for s in series:
                        if s["metric_guess"] == "input_tps":
                            s["cached_share"] = round(sum(v for _, v in _sum_points(cached)) / tot_in, 4)
        else:
            flt = f"resource.label.endpoint_id = {_q(src['endpoint_id'])} AND resource.label.location = {_q(src['location'])}"
            if metrics["endpoint_requests"]:
                ts = _time_series(client, project, build_query(EP_REQUESTS, start, end,
                                                               _aligner(metrics["endpoint_requests"]), [], flt))
                add("Predictions per second", "requests", _sum_points(ts), f"metric {EP_REQUESTS.split('/', 1)[1]}")
            for m in metrics["endpoint_tokens"]:
                ts = _time_series(client, project, build_query(m["type"], start, end, _aligner(m), [], flt))
                add(m["type"].split("/")[-1].replace("_", " "), m["role"], _sum_points(ts),
                    f"metric {m['type'].split('/', 1)[1]}")
    finally:
        if own:
            client.close()
    if not series or not any(v > 0 for s in series for _, v in s["points"]):
        raise MonitoringError(f"No traffic recorded for {source.split(':', 1)[1]} in the last {days} days.", 404,
                              "no_data")
    return {"series": series, "window": (start, end), "source": src}


def _sum_points(ts_list: list[dict]) -> list[tuple[datetime, float]]:
    acc: dict[datetime, float] = {}
    for ts in ts_list:
        for dt, v in parse_points(ts):
            acc[dt] = acc.get(dt, 0.0) + v
    return sorted(acc.items())


# ------------------------------------------------------------------------------------------------------- test stub
def _stub_sources(project: str, days: int) -> dict:
    start, end = window(days)
    return {"project": project, "days": (end - start).days, "start": start.isoformat(), "end": end.isoformat(),
            "sources": [
                {"id": "publisher:google/gemini-example-flash@global", "kind": "publisher", "name": "gemini-example-flash",
                 "publisher": "google", "location": "global", "total_requests": 1_234_000, "active_hours": 640,
                 "has_tokens": True},
                {"id": "endpoint:1234567890@us-central1", "kind": "endpoint", "name": "Endpoint 1234567890",
                 "location": "us-central1", "total_requests": 56_000, "active_hours": 300, "has_tokens": False},
            ], "metrics": [PUB_REQUESTS, PUB_TOKENS, EP_REQUESTS], "message": None}


def _stub_series(src: dict, start: datetime, end: datetime) -> dict:
    """A synthetic business-hours week (UTC 13-21 busy, quiet weekends): 2,000 input + 150 output tokens/request."""
    req, inp, out = [], [], []
    h = start + timedelta(hours=1)
    while h <= end:
        x = (h.hour + 0.5 - 17) / 3.5
        day = 0.05 + 0.95 * math.exp(-x * x)
        if h.weekday() >= 5:
            day *= 0.2
        r = 4.0 * day
        req.append((h, r))
        inp.append((h, r * 2000))
        out.append((h, r * 150))
        h += timedelta(hours=1)
    lab = {k: v for k, v in src.items() if k in ("publisher", "model", "location", "endpoint_id")}
    mk = lambda i, n, role, pts: {"id": f"m{i}", "file": "Cloud Monitoring", "name": n, "column": n, "labels": lab,  # noqa: E731
                                  "points": pts, "step_s": 3600, "assumed_utc": False, "metric_guess": role,
                                  "confidence": "high", "reason": "synthetic test data", "per_interval": False}
    series = [mk(1, "Model requests per second", "requests", req)]
    if src["kind"] == "publisher":
        series += [mk(2, "Input tokens per second", "input_tps", inp), mk(3, "Output tokens per second", "output_tps", out)]
    return {"series": series, "window": (start, end), "source": src}
