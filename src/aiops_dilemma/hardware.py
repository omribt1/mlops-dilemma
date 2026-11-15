"""Which accelerators and machine shapes exist in a region: live from the Compute Engine API.

Two read-only calls per region (`acceleratorTypes.aggregatedList`: GPU / TPU types per zone with
maximumCardsPerInstance; `machineTypes.aggregatedList` for the accelerator-optimised families a2 / a3 / a4 / g2 / g4 /
ct5lp / ct6e / ct5p …, whose `accelerators` field gives the GPU or TPU type and count, with vCPUs and memory). TPU
machine types (ct5lp-hightpu-4t etc.) are Compute Engine machine types too, so the same call covers TPUs.

Merge (`region_shapes`): the API decides which accelerators, machine shapes and counts exist in the region; the
per-chip performance table `catalog.CHIP_SPECS` (no API publishes it) adds memory / TFLOPS / bandwidth. Accelerators
the API lists but CHIP_SPECS lacks are reported as `unrated` ("no performance data"), CHIP_SPECS accelerators the
region doesn't offer as `not_offered`. Without live data (offline, no project, tests) the hand-made
`catalog.GPUS` is used and labelled as such.

Cached per region on disk (`compute_hw_<region>.json`, TTL `TTL_S` = 1 day). `cached()` / `region_shapes()` never
touch the network; `refresh()` fetches (the catalog build calls it, in the background when the cache is stale).
Compute Engine availability is the best public signal for Agent Platform; Model Garden verified recipes stay the
authority for which shapes are tested for a model.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from aiops_dilemma import hfmeta, paths
from aiops_dilemma.catalog import CHIP_SPECS, FALLBACK_MACHINES, GPUS

log = logging.getLogger(__name__)

TTL_S = 86400
FORMAT = 1
COMPUTE = "https://compute.googleapis.com/compute/v1/projects/{project}/aggregated/{what}"
ZONE_MT = "https://compute.googleapis.com/compute/v1/projects/{project}/zones/{zone}/machineTypes"
# Accelerator-optimised machine families (GPU a* / g*, TPU ct* / tpu*): matched by name so new families appear.
FAMILY_FILTER = r'name eq "(a[0-9][a-z]*|g[0-9][a-z]*|ct[0-9][a-z]*|tpu[0-9][a-z]*)-.*"'
TP_COUNTS = (1, 2, 4, 8)          # tensor-parallel degrees the planner uses on one machine
# Compute Engine accelerator names that don't follow nvidia-x-y -> NVIDIA_X_Y (TPU chips).
TPU_KEYS = {"ct5lp": "TPU_V5E", "ct6e": "TPU_V6E", "ct5p": "TPU_V5P"}
_SKIP = re.compile(r"-vws$")      # virtual-workstation variants of the same GPU


def app_key(compute_name: str) -> str | None:
    """Agent Platform accelerator key for a Compute Engine accelerator type (nvidia-h100-80gb -> NVIDIA_H100_80GB,
    ct6e -> TPU_V6E); None for workstation (-vws) variants."""
    n = (compute_name or "").lower()
    if not n or _SKIP.search(n):
        return None
    if n in TPU_KEYS:
        return TPU_KEYS[n]
    if n.startswith(("ct", "tpu")):
        return "TPU_" + re.sub(r"^(ct|tpu)", "", n).upper().replace("-", "_")
    return n.upper().replace("-", "_")


def _display(key: str, compute_name: str, description: str | None = None) -> str:
    if key in CHIP_SPECS:
        return CHIP_SPECS[key]["name"]
    return description or compute_name


def _cache_path(region: str):
    return paths.cache_dir() / f"compute_hw_{re.sub(r'[^a-z0-9-]', '', region.lower())}.json"


# ------------------------------------------------------------------------------------------------------ fetch
def _get_json(url: str, headers: dict, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.load(r)


def _paged(project: str, what: str, params: dict, headers: dict) -> dict:
    items: dict = {}
    token = None
    for _ in range(50):
        q = {**params, "maxResults": 500, "returnPartialSuccess": "true", **({"pageToken": token} if token else {})}
        d = _get_json(COMPUTE.format(project=project, what=what) + "?" + urllib.parse.urlencode(q), headers)
        for scope, v in (d.get("items") or {}).items():
            items.setdefault(scope, []).extend(v.get(what) or [])
        token = d.get("nextPageToken")
        if not token:
            break
    return items


def fetch(project: str, region: str, headers: dict | None = None) -> dict:
    """Raw per-region listing (network): {region, fetched_at, accelerator_types, machine_types}."""
    if headers is None:
        from aiops_dilemma import auth
        headers = auth.auth_headers(project)
    zone_rx = f'zone eq ".*/zones/{region}-.*"'
    accs = _paged(project, "acceleratorTypes", {"filter": zone_rx}, headers)
    zones = sorted({scope.split("/")[-1] for scope in accs if scope.split("/")[-1].startswith(region + "-")})

    def zone_types(zone):
        out, token = [], None
        for _ in range(20):
            q = {"filter": FAMILY_FILTER, "maxResults": 500, **({"pageToken": token} if token else {})}
            d = _get_json(ZONE_MT.format(project=project, zone=zone) + "?" + urllib.parse.urlencode(q), headers)
            out += d.get("items") or []
            token = d.get("nextPageToken")
            if not token:
                break
        return out

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(zones)))) as ex:
        mts = {f"zones/{z}": lst for z, lst in zip(zones, ex.map(zone_types, zones))}
    out_acc, out_mt = [], []
    for scope, lst in accs.items():
        zone = scope.split("/")[-1]
        if not zone.startswith(region + "-"):
            continue
        for a in lst:
            out_acc.append({"name": a.get("name"), "zone": zone, "max_cards": a.get("maximumCardsPerInstance"),
                            "description": a.get("description")})
    for scope, lst in mts.items():
        zone = scope.split("/")[-1]
        if not zone.startswith(region + "-"):
            continue
        for m in lst:
            for ga in m.get("accelerators") or []:
                out_mt.append({"name": m.get("name"), "zone": zone, "accelerator": ga.get("guestAcceleratorType"),
                               "count": ga.get("guestAcceleratorCount"), "vcpus": m.get("guestCpus"),
                               "memory_gb": round(float(m.get("memoryMb") or 0) / 1024)})
    return {"format": FORMAT, "region": region, "fetched_at": int(time.time()), "accelerator_types": out_acc,
            "machine_types": out_mt}


# ------------------------------------------------------------------------------------------------------ parse
def _pick(shapes: list[dict], fallback_name: str | None) -> dict:
    """One machine per (accelerator, count): the fallback catalog's choice when it exists here, else a standard shape
    (no -nolssd / -metal / edge / mega variants) with the most host memory."""
    by_name = {s["name"]: s for s in shapes}
    if fallback_name in by_name:
        return by_name[fallback_name]
    odd = re.compile(r"(nolssd|metal|edge|mega)")
    return sorted(shapes, key=lambda s: (bool(odd.search(s["name"])), -float(s.get("memory_gb") or 0), s["name"]))[0]


def parse(raw: dict) -> dict:
    """{source, region, fetched_at, accelerators: {key: {...}}} from a raw listing.

    Per accelerator: compute_name, name, kind, zones, max_cards, has_specs (in CHIP_SPECS), machines {count:
    (machine type, vCPUs, RAM GB)} for counts 1 / 2 / 4 / 8, and all_shapes (every machine type with its zones)."""
    accs: dict[str, dict] = {}
    for a in raw.get("accelerator_types") or []:
        key = app_key(a["name"])
        if not key:
            continue
        e = accs.setdefault(key, {"compute_name": a["name"], "zones": set(), "max_cards": 0,
                                  "description": a.get("description")})
        e["zones"].add(a["zone"])
        e["max_cards"] = max(e["max_cards"], int(a.get("max_cards") or 0))
    shapes: dict[str, dict[str, dict]] = {}
    for m in raw.get("machine_types") or []:
        key = app_key(m.get("accelerator") or "")
        if not key:
            continue
        s = shapes.setdefault(key, {}).setdefault(m["name"], {**{k: m[k] for k in ("name", "count", "vcpus",
                                                                                     "memory_gb")}, "zones": set()})
        s["zones"].add(m["zone"])
        accs.setdefault(key, {"compute_name": m["accelerator"], "zones": set(), "max_cards": 0, "description": None})
        accs[key]["zones"].add(m["zone"])
    out = {}
    for key, e in accs.items():
        all_shapes = sorted(shapes.get(key, {}).values(), key=lambda s: (int(s["count"] or 0), s["name"]))
        machines = {}
        for n in TP_COUNTS:
            cand = [s for s in all_shapes if int(s["count"] or 0) == n]
            if cand:
                fb = (FALLBACK_MACHINES.get(key) or {}).get(n)
                s = _pick(cand, fb[0] if fb else None)
                machines[n] = (s["name"], int(s["vcpus"] or 0), int(s["memory_gb"] or 0))
        spec = CHIP_SPECS.get(key)
        out[key] = {"compute_name": e["compute_name"], "name": _display(key, e["compute_name"], e.get("description")),
                    "kind": (spec or {}).get("kind") or ("tpu" if key.startswith("TPU_") else "gpu"),
                    "zones": sorted(e["zones"]), "max_cards": e["max_cards"] or None, "has_specs": bool(spec),
                    "machines": machines,
                    "all_shapes": [{**s, "zones": sorted(s["zones"])} for s in all_shapes]}
    return {"source": "compute_api", "region": raw.get("region"), "fetched_at": raw.get("fetched_at"),
            "accelerators": out}


# ------------------------------------------------------------------------------------------------------ cache
_MEM: dict[str, tuple[float, dict]] = {}


def cached(region: str) -> dict | None:
    """Parsed live listing for the region from the disk cache (any age), or None. Never touches the network."""
    if hfmeta.disabled():
        return None
    p = _cache_path(region)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    hit = _MEM.get(region)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        raw = json.loads(p.read_text())
        if raw.get("format") != FORMAT:
            return None
        parsed = parse(raw)
    except Exception:
        return None
    _MEM[region] = (mtime, parsed)
    return parsed


def is_stale(region: str) -> bool:
    try:
        return time.time() - _cache_path(region).stat().st_mtime > TTL_S
    except OSError:
        return True


def refresh(project: str | None, region: str) -> dict | None:
    """Fetch the region's listing now and cache it; returns the parsed listing (None on failure: fallback stays)."""
    from aiops_dilemma import auth
    proj = auth.real_project(project)
    if hfmeta.disabled() or not proj:
        return None
    try:
        raw = fetch(proj, region)
    except Exception as e:
        log.warning("compute hardware listing failed for %s: %s", region, str(e)[:200])
        return None
    if not raw["accelerator_types"] and not raw["machine_types"]:
        return None
    try:
        p = _cache_path(region)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw))
        tmp.replace(p)
    except Exception:
        pass
    _MEM.pop(region, None)
    return parse(raw)


_BG: dict[str, threading.Thread] = {}


def ensure_fresh(project: str | None, region: str, block_if_missing: bool = True) -> None:
    """Fetch when the cache is missing (blocking, ~1 s) or stale (background thread)."""
    if hfmeta.disabled():
        return
    if cached(region) is None and block_if_missing:
        refresh(project, region)
        return
    if is_stale(region):
        t = _BG.get(region)
        if not (t and t.is_alive()):
            _BG[region] = threading.Thread(target=refresh, args=(project, region), daemon=True)
            _BG[region].start()


# ------------------------------------------------------------------------------------------------------ merge
def region_shapes(region: str | None) -> dict:
    """Accelerators the planner may propose in `region`, merged with CHIP_SPECS.

    Returns {source: "compute_api" | "fallback", region, fetched_at, accelerators: {key: CHIP_SPECS row + machines
    (live shapes) + zones}, not_offered: [{accelerator_type, name, note}], unrated: [{accelerator_type, name,
    machines, zones, note}]}. Fallback (no live listing): catalog.GPUS as is, nothing marked missing."""
    live = cached(region) if region else None
    if not live:
        return {"source": "fallback", "region": region, "fetched_at": None,
                "accelerators": {k: dict(v) for k, v in GPUS.items()}, "not_offered": [], "unrated": [],
                "note": "Accelerators and machine shapes from the built-in catalog (Compute Engine API not reached)."}
    accs, unrated = {}, []
    for key, e in live["accelerators"].items():
        if e["has_specs"]:
            if e["machines"]:
                accs[key] = {**CHIP_SPECS[key], "machines": dict(e["machines"]), "zones": e["zones"],
                             "max_cards": e["max_cards"]}
            continue
        shapes = ", ".join(sorted({s["name"] for s in e["all_shapes"]})) or "attachable to general-purpose machines"
        unrated.append({"accelerator_type": key, "name": e["name"], "kind": e["kind"], "zones": e["zones"],
                        "machines": shapes,
                        "note": f"Offered in {region} ({shapes}) but there is no performance data for it in this "
                                "tool (catalog.CHIP_SPECS): not estimated."})
    not_offered = [{"accelerator_type": k, "name": v["name"], "kind": v["kind"],
                    "note": f"Not offered in {region}"} for k, v in CHIP_SPECS.items() if k not in accs]
    return {"source": "compute_api", "region": region, "fetched_at": live.get("fetched_at"), "accelerators": accs,
            "not_offered": not_offered, "unrated": sorted(unrated, key=lambda u: u["accelerator_type"]),
            "note": f"Accelerators and machine shapes offered in {region}, from the Compute Engine API."}


def summary(region: str) -> dict:
    """Compact view for the catalog payload / UI (no per-shape zone lists)."""
    rs = region_shapes(region)
    return {"source": rs["source"], "region": region, "fetched_at": rs["fetched_at"], "note": rs.get("note"),
            "offered": {k: {"name": v["name"], "kind": v["kind"],
                            "machines": {str(n): m[0] for n, m in sorted(v["machines"].items())},
                            "zones": v.get("zones")} for k, v in rs["accelerators"].items()},
            "not_offered": rs["not_offered"], "unrated": rs["unrated"]}
