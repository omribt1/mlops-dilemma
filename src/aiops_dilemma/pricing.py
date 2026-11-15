"""GPU, TPU, and Gemini prices from the Cloud Billing Catalog API, with automatic fallback.

Tiers:
  vertex          Agent Platform online prediction, on demand (GPU/TPU + vCPU + RAM + management fees)
  vertex_cud_1y/3y Agent Platform endpoint on a Compute Engine reservation with a CUD: GCE SKUs at CUD
                  rates + Agent Platform management fee discounted by a Flexible Savings Plan (10% / 20%)
  on_demand       self-managed GCE/GKE, list price
  cud_1y / cud_3y self-managed GCE/GKE with resource CUDs
Requires cloudbilling.googleapis.com enabled on the quota project. Prices are cached for 7 days.
"""

import collections
import copy
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from aiops_dilemma import paths
from aiops_dilemma.catalog import FALLBACK_PRICES, GPUS

SERVICES = {"compute": "6F81-5844-456A", "vertex": "C7E2-9256-1C43"}
CACHE = paths.cache_dir()   # DILEMMA_CACHE_DIR overrides
FSP = {"1y": 0.90, "3y": 0.80}
EXCLUDE = re.compile(r"Spot|Preemptible|DWS|Sole Tenancy|Custom|Mega|Commitment|Training", re.I)

# Regional multiplier approximations when falling back to reference catalog prices
REGION_MULT = {
    "us-central1": 1.00, "us-east1": 1.00, "us-east4": 1.05, "us-east5": 1.00,
    "us-west1": 1.00, "us-west4": 1.05, "us-south1": 1.03,
    "europe-west1": 1.08, "europe-west2": 1.14, "europe-west3": 1.15,
    "europe-west4": 1.08, "europe-west9": 1.12, "europe-north1": 1.05,
    "asia-east1": 1.10, "asia-northeast1": 1.15, "asia-northeast3": 1.15,
    "asia-south1": 1.10, "asia-southeast1": 1.12, "australia-southeast1": 1.18,
    "me-west1": 1.12, "northamerica-northeast2": 1.08,
}


SKU_TTL = 7 * 86400
_SKU_MEM: dict[str, tuple[float, list[dict]]] = {}
_PRICE_MEM: dict[tuple, dict] = {}
_LOCK = threading.Lock()


def _headers(project: str) -> dict:
    """Bearer token + quota project for the Cloud Billing Catalog: Application Default Credentials with the app's
    project as quota project (auth.credentials), else the gcloud CLI's account."""
    from aiops_dilemma import auth
    try:
        h = auth.auth_headers(project)
    except Exception:
        token = subprocess.check_output(["gcloud", "auth", "print-access-token"], timeout=10).decode().strip()
        h = {"Authorization": f"Bearer {token}"}
    h["x-goog-user-project"] = project
    return h


def _quota_project(project: str | None) -> str | None:
    """A real project to bill Cloud Billing Catalog API quota to: the given one, else the app's project."""
    if project and project != "my-gcp-project":
        return project
    try:
        from aiops_dilemma import auth
        return auth.real_project(auth.app_project())
    except Exception:
        return None


def skus(service: str, project: str | None, max_age: float | None = None) -> list[dict]:
    """All SKUs of a billing service. Cached on disk for 7 days (`max_age` seconds if given: the Gemini catalog
    wants SKUs at most a day old) and in memory for the process lifetime."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"skus_{service}.json"
    ttl = SKU_TTL if max_age is None else min(SKU_TTL, max_age)
    with _LOCK:
        if path.exists() and time.time() - path.stat().st_mtime < ttl:
            mtime = path.stat().st_mtime
            mem = _SKU_MEM.get(service)
            if mem and mem[0] == mtime:
                return mem[1]
            data = json.loads(path.read_text())
            _SKU_MEM[service] = (mtime, data)
            return data
    project = _quota_project(project)
    if not project:
        raise RuntimeError("Set a real GCP project and authenticate to fetch live Cloud Billing SKUs")
    headers, out, page = _headers(project), [], ""
    while True:
        url = (f"https://cloudbilling.googleapis.com/v1/services/{SERVICES[service]}/skus?pageSize=5000"
               f"&currencyCode=USD" + (f"&pageToken={page}" if page else ""))
        req = urllib.request.Request(url, headers=headers)
        data = json.load(urllib.request.urlopen(req, timeout=180))
        out += data.get("skus", [])
        page = data.get("nextPageToken")
        if not page:
            break
    path.write_text(json.dumps(out))
    with _LOCK:
        _SKU_MEM[service] = (path.stat().st_mtime, out)
    return out


def _price(sku: dict) -> float:
    r = sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"]
    return float(r.get("units", 0)) + r.get("nanos", 0) / 1e9


def _find(catalog, region, pattern, allow=None) -> float | None:
    rx = re.compile(pattern, re.I)
    hits = [s for s in catalog if (region in s.get("serviceRegions", []) or "global" in s.get("serviceRegions", []))
            and rx.search(s["description"])
            and (allow is not None and allow.search(s["description"]) or not EXCLUDE.search(s["description"]))]
    if not hits:
        return None
    return _price(min(hits, key=lambda s: len(s["description"])))


def gpu_hour_prices(project: str, region: str, accelerator_type: str, count: int) -> dict:
    g = GPUS[accelerator_type]
    from aiops_dilemma import hardware   # the region's live machine shape (host vCPUs / RAM) when known
    live = (hardware.region_shapes(region)["accelerators"].get(accelerator_type) or {}).get("machines") or {}
    _, vcpu, ram = live.get(count) or g["machines"][count]
    v, c = skus("vertex", project), skus("compute", project)
    commit = re.compile(r"Commitment", re.I)

    if g.get("kind") == "tpu":
        tname = re.escape(g["sku"]["gce"])
        chip = _find(c, region, rf"{tname}.*chip", None) or _find(v, region, rf"{tname}", None)
        chip_1y = _find(c, region, rf"Commitment.*{tname}.*1 Year", commit)
        chip_3y = _find(c, region, rf"Commitment.*{tname}.*3 Year", commit)
        if not chip:
            raise RuntimeError(f"Live TPU SKU not found for {accelerator_type} in {region}")
        fee = chip * 0.22
        c1 = chip_1y or (chip * 0.63)
        c3 = chip_3y or (chip * 0.45)
        return {
            "vertex": chip + fee,
            "on_demand": chip,
            "management_fee": fee,
            "cud_1y": c1,
            "cud_3y": c3,
            "vertex_cud_1y": c1 + fee * FSP["1y"],
            "vertex_cud_3y": c3 + fee * FSP["3y"],
            "missing_skus": [],
            "source": "live_billing",
        }

    fam = "A3 Ultra" if accelerator_type == "NVIDIA_H200_141GB" else g["family"]
    vname, gname = re.escape(g["sku"]["vertex"]), re.escape(g["sku"]["gce"])
    rx = g.get("sku_rx") or {}   # per-accelerator SKU description patterns where the generic ones don't match
    found = {
        "v_gpu": _find(v, region, rx.get("v_gpu") or rf"Online/Batch Prediction Nvidia (Tesla )?{vname}( GPU)?( with lssd)? running"),
        "v_fee_gpu": _find(v, region, rf"Online/Batch Prediction Management fee on NVIDIA (Tesla )?{vname}"),
        "c_gpu": _find(c, region, rx.get("c_gpu") or rf"^{gname} running in"),
    }
    host = not g.get("host_bundled")   # A3 Ultra / A4 GPU SKUs include the host: no core / RAM SKUs to add
    if host:
        found.update({
            "v_core": _find(v, region, rf"Online/Batch Prediction {fam}( Predefined)? Instance Core running"),
            "v_ram": _find(v, region, rf"Online/Batch Prediction {fam}( Predefined)? Instance Ram running"),
            "v_fee_core": _find(v, region, rf"Online/Batch Prediction Management fee on {fam}( Predefined)? Instance Core"),
            "v_fee_ram": _find(v, region, rf"Online/Batch Prediction Management fee on {fam}( Predefined)? Instance RAM"),
            "c_core": _find(c, region, rf"^{fam} Instance Core running in"),
            "c_ram": _find(c, region, rf"^{fam} Instance Ram running in"),
        })
    for term in ("1", "3"):
        commit_rx = (rx["c_gpu_commit"].format(term=term) if rx.get("c_gpu_commit")
                     else rf"^Commitment v1: {gname} running in .* for {term} Year")
        found[f"c_gpu_{term}y"] = _find(c, region, commit_rx, commit)
        if host:
            found[f"c_core_{term}y"] = _find(c, region, rf"^Commitment v1: {fam} Cpu in .* for {term} Year", commit)
            found[f"c_ram_{term}y"] = _find(c, region, rf"^Commitment v1: {fam} Ram in .* for {term} Year", commit)
    missing = [k for k, x in found.items() if x is None]
    f = collections.defaultdict(float, {k: (x or 0.0) for k, x in found.items()})   # bundled host: core/RAM = 0
    per_gpu_host = lambda core, ram_: (vcpu * core + ram * ram_) / count  # noqa: E731
    fee = f["v_fee_gpu"] + per_gpu_host(f["v_fee_core"], f["v_fee_ram"])
    out = {
        "vertex": f["v_gpu"] + per_gpu_host(f["v_core"], f["v_ram"]) + fee,
        "on_demand": f["c_gpu"] + per_gpu_host(f["c_core"], f["c_ram"]),
        "management_fee": fee,
    }
    for term in ("1y", "3y"):
        gce = f[f"c_gpu_{term}"] + per_gpu_host(f[f"c_core_{term}"], f[f"c_ram_{term}"])
        out[f"cud_{term}"] = gce
        out[f"vertex_cud_{term}"] = gce + fee * FSP[term]
    out["missing_skus"] = missing
    out["source"] = "live_billing"
    return out


def fallback_prices_for(region: str, accelerator_type: str) -> dict:
    base = FALLBACK_PRICES.get(accelerator_type)
    if not base:
        return {"missing_skus": ["all"], "source": "unavailable"}
    mult = REGION_MULT.get(region, 1.0)
    out = {k: round(v * mult, 4) for k, v in base.items()}
    out["missing_skus"] = []
    out["source"] = "reference_catalog"
    return out


PRICE_CACHE_VERSION = "v2"   # bump when SKU matching changes (v2: A3 Ultra / A4 per-GPU SKUs, B200)


def _price_disk_cache() -> Path:
    return CACHE / "gpu_prices.json"


def _cached_gpu_hour_prices(project: str, region: str, accelerator_type: str, count: int) -> dict:
    """gpu_hour_prices() memoized in memory and on disk (same 7-day TTL as the SKU files it is derived from).

    Scanning ~50K SKUs with regexes takes ~0.5 s per accelerator; the planner needs nine of them per call.
    """
    key = (region, accelerator_type, int(count))
    def _ret(p: dict) -> dict:
        if p.get("_error"):
            raise RuntimeError(p["_error"])
        return copy.deepcopy(p)

    with _LOCK:
        if key in _PRICE_MEM:
            return _ret(_PRICE_MEM[key])
    disk_key = PRICE_CACHE_VERSION + "|" + "|".join(str(k) for k in key)
    path = _price_disk_cache()
    disk = {}
    try:
        if path.exists():
            disk = json.loads(path.read_text())
            hit = disk.get(disk_key)
            if hit and time.time() - hit.get("_ts", 0) < SKU_TTL:
                p = {k: v for k, v in hit.items() if k != "_ts"}
                with _LOCK:
                    _PRICE_MEM[key] = p
                return _ret(p)
    except RuntimeError:
        raise
    except Exception:
        disk = {}
    try:
        p = gpu_hour_prices(project, region, accelerator_type, count)
    except Exception as e:  # remember misses too (e.g. a TPU SKU absent in this region), so we don't rescan
        p = {"_error": str(e)[:300]}
    with _LOCK:
        _PRICE_MEM[key] = p
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        disk[disk_key] = {**p, "_ts": time.time()}
        path.write_text(json.dumps(disk))
    except Exception:
        pass
    return _ret(p)


def prices_for(cfg: dict, accelerator_type: str, count: int) -> dict:
    over = cfg.get("pricing", {}).get("overrides") or {}
    region = cfg.get("region", "us-central1")
    try:
        p = _cached_gpu_hour_prices(cfg.get("project") or "my-gcp-project", region, accelerator_type, count)
        if not p.get("vertex") or {"v_gpu", "c_gpu"} & set(p.get("missing_skus", [])):
            p = fallback_prices_for(region, accelerator_type)
    except Exception:
        p = fallback_prices_for(region, accelerator_type)
    p.update(over)
    return p


# Gemini prices: gemini.py builds the live Gemini catalog (every text model with per-token SKUs, Model Garden launch
# stage, role aliases). The two parsers below are the pre-catalog API, kept as thin views over gemini.parse_billing.
GEMINI_TTL = 86400
_GEM_MEM: dict = {}   # kept for callers that reset it; the catalog cache lives in gemini._MEM


def parse_gemini_skus(sku_list: list[dict]) -> dict:
    """Newest Gemini Flash-Lite / Flash / Pro text prices by version ($ per 1M tokens), keyed by role alias."""
    from aiops_dilemma import gemini
    out = {}
    for m in gemini.parse_billing(sku_list):
        key = gemini.ALIAS_FOR_FAMILY.get(m["family"])
        if not key or (key in out and gemini.version_tuple(out[key]["version"]) >= gemini.version_tuple(m["version"])):
            continue
        out[key] = {"version": m["version"], "input_per_m": round(m["input_per_m"], 4),
                    "output_per_m": round(m["output_per_m"], 4), "cached_per_m": round(m["cached_per_m"], 4),
                    "skus": m["skus"], "model_id": m["id"], "long_context": m["long_context"]}
    return out


def parse_gemini4_skus(sku_list: list[dict]) -> dict | None:
    """The newest generation's (>= 4) flagship text prices, e.g. "Gemini 4 Carbon", or None."""
    from aiops_dilemma import gemini
    m = gemini.resolve_alias("gemini_4", gemini.parse_billing(sku_list))
    if not m:
        return None
    return {"version": m["version"], "name": m["name"], "model_id": m["id"], "input_per_m": round(m["input_per_m"], 4),
            "output_per_m": round(m["output_per_m"], 4), "cached_per_m": round(m["cached_per_m"], 4), "skus": m["skus"]}


def gemini_prices(project: str | None = None, region: str = "us-central1", refresh: bool = False) -> dict:
    """Gemini on Agent Platform token prices ($ per 1M tokens) for the cost engines: {role alias or model id: entry}.

    Role aliases (gemini_flash_lite, gemini_flash, gemini_pro, gemini_4) resolve to live models; every live Gemini
    text model is there under its id ("gemini-4-carbon"). Live from the Cloud Billing Catalog + Model Garden (cached a
    day, refreshed in the background), else catalog.GEMINI_MODELS (source "fallback"). See gemini.py.
    """
    from aiops_dilemma import gemini
    try:
        cat = gemini.get(project, region, refresh=refresh)
    except Exception:
        cat = gemini.build(None, None)
    return gemini.engine_map(cat)
