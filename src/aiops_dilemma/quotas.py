"""Agent Platform serving quotas per accelerator and region, read live from the Cloud Quotas API.

Benchmark endpoints (and a production fleet on Agent Platform) draw on the project's per-region "Custom model serving"
quotas, e.g. `aiplatform.googleapis.com/custom_model_serving_nvidia_rtx_pro_6000_gpus` in us-central1. Source:

    GET https://cloudquotas.googleapis.com/v1/projects/{p}/locations/global/services/aiplatform.googleapis.com/quotaInfos

Each QuotaInfo has `dimensionsInfos`: one entry per region with a non-default value, plus one entry (no
`dimensions`) whose `applicableLocations` lists the regions on the default. A missing `details.value` means 0 (the API
omits zero values). The API returns limits, not usage. Preemptible (Spot) quotas are ignored.

Accelerator -> metric is derived, not listed: `NVIDIA_RTX_PRO_6000` -> `nvidia_rtx_pro_6000_gpus`, `NVIDIA_TESLA_A100` ->
`nvidia_a100_gpus` (drop "tesla_"), `NVIDIA_H100_80GB` -> `nvidia_h100_gpus` (memory suffix dropped when the exact name
has no quota, so `NVIDIA_A100_80GB` keeps `nvidia_a100_80gb_gpus`), `TPU_V6E` -> `tpu_v6e` (TPU quotas count cores).

Needs the Cloud Quotas API enabled and `cloudquotas.quotas.get` (e.g. Viewer or Cloud Quotas Viewer). Errors are
returned as {"ok": False, "error", "reason"}; the app shows them instead of quota figures.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

URL = ("https://cloudquotas.googleapis.com/v1/projects/{project}/locations/global/services/"
       "aiplatform.googleapis.com/quotaInfos")
PREFIX = "aiplatform.googleapis.com/custom_model_serving_"
TTL = 1800
_MEM: dict = {}
_LOCK = threading.Lock()


def _get(url: str, headers: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse(infos: list[dict]) -> dict:
    """{metric suffix: {"quota_id", "display", "unit_cores", "regions": {region: limit}}} for on-demand serving quotas."""
    out = {}
    for q in infos:
        metric = q.get("metric") or ""
        if not metric.startswith(PREFIX) or "preemptible" in metric or q.get("dimensions") not in (["region"], None):
            continue
        regions = {}
        for di in q.get("dimensionsInfos") or []:
            val = int((di.get("details") or {}).get("value") or 0)
            reg = (di.get("dimensions") or {}).get("region")
            for r in ([reg] if reg else di.get("applicableLocations") or []):
                regions[r] = val
        suffix = metric[len(PREFIX):]
        out[suffix] = {"quota_id": q.get("quotaId"), "display": q.get("metricDisplayName") or suffix,
                       "unit_cores": "tpu" in suffix, "regions": regions,
                       "increase_eligible": bool((q.get("quotaIncreaseEligibility") or {}).get("isEligible", True))}
    return out


def metric_for(accelerator_type: str, metrics: dict) -> str | None:
    """The serving-quota metric suffix for an accelerator type, derived from its name (see module doc)."""
    k = accelerator_type.lower().replace("tesla_", "")
    if k.startswith("tpu"):
        tries = [k, re.sub(r"_(lite|pod)$", "", k)]
    else:
        tries = [f"{k}_gpus", re.sub(r"_\d+gb$", "", k) + "_gpus"]
    return next((t for t in tries if t in metrics), None)


def fetch(project: str, headers: dict | None = None) -> dict:
    if headers is None:
        from aiops_dilemma import auth
        headers = auth.auth_headers(project)
    infos, token = [], None
    for _ in range(20):
        q = {"pageSize": 500, **({"pageToken": token} if token else {})}
        d = _get(URL.format(project=project) + "?" + urllib.parse.urlencode(q), headers)
        infos += d.get("quotaInfos") or []
        token = d.get("nextPageToken")
        if not token:
            break
    return {"ok": True, "project": project, "fetched_at": int(time.time()), "metrics": parse(infos)}


def _error(project: str, e: Exception) -> dict:
    reason, msg = "error", str(e)
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = json.loads(e.read().decode())
            err = body.get("error") or {}
            msg = err.get("message") or msg
            reasons = [d.get("reason") for d in err.get("details") or [] if d.get("reason")]
            if "SERVICE_DISABLED" in reasons or "has not been used" in msg or "is disabled" in msg:
                reason, msg = "api_disabled", (f"Enable the Cloud Quotas API (cloudquotas.googleapis.com) in {project} "
                                               "to see GPU quotas here.")
            elif e.code == 403:
                reason, msg = "forbidden", (f"The signed-in identity can't read quotas in {project} (needs "
                                            "cloudquotas.quotas.get, e.g. Viewer or Cloud Quotas Viewer).")
        except Exception:
            pass
    return {"ok": False, "project": project, "error": msg, "reason": reason, "fetched_at": int(time.time())}


def get(project: str | None, refresh: bool = False) -> dict:
    """Cached quota listing for a project (30 min; errors cached 2 min)."""
    if not project or project == "my-gcp-project":
        return {"ok": False, "error": "No Google Cloud project chosen.", "reason": "no_project"}
    with _LOCK:
        hit = _MEM.get(project)
    if hit and not refresh and time.time() - hit["fetched_at"] < (TTL if hit.get("ok") else 120):
        return hit
    try:
        res = fetch(project)
    except Exception as e:  # network, auth, API disabled
        res = _error(project, e)
    with _LOCK:
        _MEM[project] = res
    return res


def clear() -> None:
    with _LOCK:
        _MEM.clear()


def for_region(project: str | None, region: str, accelerator_types: list[str]) -> dict:
    """{ok, region, fetched_at, source, by_accelerator: {acc: {limit, quota_id, display, unit}}} or the error."""
    data = get(project)
    if not data.get("ok"):
        return {k: v for k, v in data.items() if k != "metrics"} | {"region": region}
    metrics = data["metrics"]
    by = {}
    for acc in dict.fromkeys(accelerator_types):
        m = metric_for(acc, metrics)
        if not m:
            by[acc] = {"limit": None, "quota_id": None, "unit": None}
            continue
        info = metrics[m]
        by[acc] = {"limit": info["regions"].get(region, 0) if info["regions"] else None, "quota_id": info["quota_id"],
                   "display": info["display"], "unit": "cores" if info["unit_cores"] else "GPUs",
                   "increase_eligible": info["increase_eligible"]}
    return {"ok": True, "project": data["project"], "region": region, "fetched_at": data["fetched_at"],
            "source": "cloud_quotas_api", "by_accelerator": by,
            "console_url": f"https://console.cloud.google.com/iam-admin/quotas?project={data['project']}"}
