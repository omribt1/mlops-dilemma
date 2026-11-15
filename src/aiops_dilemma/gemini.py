"""Live Gemini catalog: every Gemini text model Agent Platform bills per token, with prices and launch stage.

Sources (both read-only, cached on disk for a day, refreshed in the background):

- **Cloud Billing Catalog** (Agent Platform service C7E2-9256-1C43, the SKUs pricing.py already reads): every
  "Gemini <version> <family>" text SKU, e.g. "Gemini 3.5 Flash Lite Global Text Input - Predictions",
  "Gemini 4 Carbon Global Text Output - Online Predictions", "Gemini 3.0 Pro / 3.1 Pro / 3 Pro Image - Text Input
  (Long) - Predictions". Kept: on-demand per-token text input / output / cached input, Global (else Regional),
  online; "(Long)" SKUs give the long-context tier, "Batch" SKUs the batch price, "Regional" SKUs the regional one.
  Dropped: variants and other products sharing the name (Cyber / Permissive, Image, Live, TTS, Transcribe, Omni, Nano
  Banana, Robotics, embeddings), priority / flex / deferred / off-peak / tuning / provisioned / storage / thinking
  SKUs. A model exists here when it has an online input and output price: pricing is what the cost model needs.
- **Model Garden publisher models** (`GET https://{region}-aiplatform.googleapis.com/v1beta1/publishers/google/models`,
  paged): official model ids and launch stage (GA, PUBLIC_PREVIEW, EXPERIMENTAL, DEPRECATED). The listing exposes no
  display names, context windows or retirement dates (checked 2026-11-14), so those stay empty.

Join: a priced model takes the Model Garden stage of its id ("gemini-3.5-flash-lite") or of a preview of it
("gemini-3.1-pro-preview"). A priced model Model Garden doesn't list is `deprecated` (retired) when the listing has a
newer model of the same family, else `unlisted` (e.g. billed before it is listed). Without the listing, stage is None.

Role aliases (the config keys from before the live catalog) resolve dynamically:

- `gemini_flash_lite` / `gemini_flash` / `gemini_pro`: the newest GA model of that family; a preview of a newer major
  generation than the newest GA wins (Gemini 3.1 Pro Preview over Gemini 2.5 Pro). Deprecated models never count.
- `gemini_4`: the newest generation's flagship: the highest major version (>= 4) present, GA first, the models at the
  top output price of that generation (Gemini 4 Argon / Carbon over the cheaper Hydrogen), the newest of them.
"Newest" = version, then the date this app first saw the model (`gemini_first_seen.json`), then Model Garden listing
order (new models are appended). Without live data the aliases use `catalog.GEMINI_MODELS` (source "fallback"), which
is also what tests see (`DILEMMA_NO_LIVE_METADATA=1`), so their numbers are fixed.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone

from aiops_dilemma import paths, regions
from aiops_dilemma.catalog import GEMINI_MODELS

log = logging.getLogger(__name__)

TTL_S = 86400
FORMAT = 1
ON_UPDATE: list = []          # callbacks after a background refresh landed (server rebuilds its catalogs)
GARDEN_URL = "https://{region}-aiplatform.googleapis.com/v1beta1/publishers/google/models"

# Billing SKUs say "(Long)" without a size. Gemini bills prompts over 200K tokens at that rate (Gemini 1.x: 128K).
LONG_THRESHOLD = 200_000
LONG_THRESHOLD_V1 = 128_000

TIER_FAMILIES = ("flash-lite", "flash", "pro", "ultra")
FAMILY_LABEL = {"flash-lite": "Flash-Lite", "flash": "Flash", "pro": "Pro", "ultra": "Ultra"}
ALIASES = {
    # key: (label, family or None for the flagship rule, rule text)
    "gemini_flash_lite": ("Gemini Flash-Lite", "flash-lite", "Newest GA Flash-Lite"),
    "gemini_flash": ("Gemini Flash", "flash", "Newest GA Flash"),
    "gemini_pro": ("Gemini Pro", "pro", "Newest GA Pro (a preview of a newer generation wins)"),
    "gemini_4": ("Gemini flagship", None, "Newest generation's flagship (highest output price, newest)"),
}
ALIAS_FOR_FAMILY = {"flash-lite": "gemini_flash_lite", "flash": "gemini_flash", "pro": "gemini_pro"}

_SKU_RX = re.compile(r"^Gemini\s+(?P<models>\d.+?)\s*(?:-\s*)?"
                     r"(?P<rest>(?:Global\s+|Regional\s+)?(?:Thinking\s+)?(?:Text|Input|Output)\b.*)$", re.I)
_ALT_RX = re.compile(r"^(?P<ver>\d+(?:\.\d+)?)(?:\s+(?P<fam>Flash[- ]Lite|[A-Za-z]+))?(?:\s+(?P<ga>GA))?"
                     r"(?:\s+(?P<extra>.+))?$", re.I)
# Words after the version that name another product or a variant, not a text model family.
_PRODUCT_WORDS = {"live", "image", "tts", "transcribe", "embedding", "embeddings", "audio", "native", "translate",
                  "robotics", "computer", "omni", "cyber", "permissive", "nano", "banana", "thinking", "text", "input",
                  "output", "global", "regional", "mm", "vision", "code", "codemender", "ga", "veo", "imagen"}
_REST_EXCLUDE = re.compile(r"Flex|Priority|Deferred|Off-?peak|Tun(?:ed|ing)|Provisioned|Storage|Thinking|Grounding|"
                           r"Audio|Video|Image|Live|Count|Search|Cyber|Permissive", re.I)

_MEM: dict = {}
_LOCK = threading.Lock()
_REFRESHING: set = set()


def live_enabled() -> bool:
    """False in tests (DILEMMA_NO_LIVE_METADATA=1): no cache reads, no fetches, fixed fallback prices."""
    return os.environ.get("DILEMMA_NO_LIVE_METADATA", "").strip() in ("", "0", "false")


# ---------------------------------------------------------------------------------------------------- SKU parsing

def norm_version(v: str) -> str:
    """'3.0' -> '3' and '4.0' -> '4' (Gemini 3 / 4 are named without .0), '2.0' kept (Gemini 2.0 Flash)."""
    v = v.strip()
    return v[:-2] if v.endswith(".0") and version_tuple(v)[0] >= 3 else v


def version_tuple(v: str) -> tuple:
    try:
        parts = [int(x) for x in str(v).split(".")]
    except ValueError:
        return (0, 0)
    return tuple((parts + [0])[:2])


def family_slug(fam: str) -> str:
    f = re.sub(r"[\s_]+", "-", fam.strip().lower())
    return "flash-lite" if f in ("flash-lite", "flashlite") else f


def family_label(fam: str) -> str:
    return FAMILY_LABEL.get(fam) or fam.capitalize()


def model_id(version: str, fam: str) -> str:
    return f"gemini-{version}-{fam}"


def model_name(version: str, fam: str) -> str:
    return f"Gemini {version} {family_label(fam)}"


def _price(sku: dict) -> float | None:
    try:
        r = sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"]
    except (KeyError, IndexError, TypeError):
        return None
    return float(r.get("units", 0) or 0) + (r.get("nanos", 0) or 0) / 1e9


def _kind(rest: str) -> str | None:
    r = rest.lower()
    if "text" not in r:
        return None
    if "cach" in r:          # "Input Text Caching", "Text Input Caching", "Text Cached Input"
        return "cached"
    if "input" in r:
        return "input"
    if "output" in r:
        return "output"
    return None


def _models_in(models_part: str) -> list[tuple[str, str, bool]]:
    """'3.0 Pro / 3.1 Pro / 3 Pro Image' -> [('3', 'pro', False), ('3.1', 'pro', False)]; variants with an extra word
    ('3 Pro Image', '3.8 Flash Cyber', '2.0 Flash Live') are other products and dropped."""
    alts = [_ALT_RX.match(x.strip()) for x in models_part.split("/")]
    if not all(alts):
        return []
    out, fam_next = [], None
    for a in reversed(alts):
        fam = a["fam"]
        if fam and fam.lower() in _PRODUCT_WORDS:
            fam_next = None
            continue
        fam_next = fam or fam_next
        if not a["extra"] and fam_next:
            out.append((norm_version(a["ver"]), family_slug(fam_next), bool(a["ga"])))
    return list(reversed(out))


def parse_billing(sku_list: list[dict]) -> list[dict]:
    """Every Gemini text model with per-token SKUs: [{id, name, family, version, input_per_m, output_per_m,
    cached_per_m, long_context, batch, regional, scope, skus}], unsorted. Prices are $ per 1M tokens."""
    slots: dict[tuple, dict] = {}
    meta: dict[tuple, dict] = {}
    for s in sku_list:
        d = " ".join((s.get("description") or "").split())
        m = _SKU_RX.match(d)
        if not m:
            continue
        rest = m["rest"]
        batch = bool(re.search(r"\bBatch\b", rest, re.I))
        if _REST_EXCLUDE.search(rest):
            continue
        if (s.get("category") or {}).get("usageType", "OnDemand") != "OnDemand":
            continue
        try:
            unit = s["pricingInfo"][0]["pricingExpression"]
        except (KeyError, IndexError, TypeError):
            continue
        if unit.get("usageUnit", "count") != "count":
            continue
        kind = _kind(rest)
        price = _price(s)
        if not kind or price is None:
            continue
        scope = "regional" if re.search(r"\bRegional\b", rest, re.I) else "global"
        long_ = bool(re.search(r"\bLong\b", rest, re.I))
        for ver, fam, ga in _models_in(m["models"]):
            key = (ver, fam)
            meta.setdefault(key, {"skus": set()})
            slot = slots.setdefault(key, {})
            k = ("batch" if batch else "online", scope, long_, kind)
            rank = (0 if ga else 1, len(d))   # GA SKUs win ("2.5 Flash GA" over the preview "2.5 Flash"), then shortest
            if k not in slot or rank < slot[k][1]:
                slot[k] = (price * 1e6, rank, d)
    out = []
    for (ver, fam), slot in slots.items():
        def get(mode, scope, long_, kind):
            v = slot.get((mode, scope, long_, kind))
            return v[0] if v else None
        scope = "global" if get("online", "global", False, "input") is not None else "regional"
        inp, outp = get("online", scope, False, "input"), get("online", scope, False, "output")
        if inp is None or outp is None:
            continue
        cached = get("online", scope, False, "cached")
        entry = {
            "id": model_id(ver, fam), "name": model_name(ver, fam), "family": fam, "version": ver,
            "input_per_m": round(inp, 6), "output_per_m": round(outp, 6),
            "cached_per_m": round(cached if cached is not None else inp * 0.1, 6),
            "cached_estimated": cached is None, "scope": scope, "long_context": None, "batch": None, "regional": None,
            "skus": sorted({v[2] for k, v in slot.items() if k[0] == "online" and k[1] == scope}),
            "source": "billing-catalog",
        }
        lin, lout = get("online", scope, True, "input"), get("online", scope, True, "output")
        if lin is not None or lout is not None:
            lin = lin if lin is not None else inp
            lout = lout if lout is not None else outp
            if abs(lin - inp) > 1e-9 or abs(lout - outp) > 1e-9:
                lc = get("online", scope, True, "cached")
                entry["long_context"] = {
                    "threshold": LONG_THRESHOLD_V1 if version_tuple(ver)[0] < 2 else LONG_THRESHOLD,
                    "input_per_m": round(lin, 6), "output_per_m": round(lout, 6),
                    "cached_per_m": round(lc if lc is not None else entry["cached_per_m"] * lin / max(inp, 1e-12), 6),
                }
        for sc in ("global", "regional"):
            bi, bo = get("batch", sc, False, "input"), get("batch", sc, False, "output")
            if bi is not None and bo is not None:
                entry["batch"] = {"input_per_m": round(bi, 6), "output_per_m": round(bo, 6), "scope": sc}
                break
        if scope == "global":
            ri, ro = get("online", "regional", False, "input"), get("online", "regional", False, "output")
            if ri is not None and ro is not None:
                entry["regional"] = {"input_per_m": round(ri, 6), "output_per_m": round(ro, 6)}
        out.append(entry)
    return out


# ---------------------------------------------------------------------------------------------------- Model Garden

def parse_garden(publisher_models: list[dict]) -> list[dict]:
    """[{id, stage_raw, index}] for google/gemini-* publisher models, in listing order."""
    out = []
    for i, m in enumerate(publisher_models or []):
        name = str(m.get("name") or "")
        mid = name.rsplit("/", 1)[-1]
        if not mid.startswith("gemini-"):
            continue
        out.append({"id": mid, "stage_raw": m.get("launchStage"), "index": i})
    return out


def fetch_garden(project: str, region: str = "us-central1", timeout: float = 60) -> list[dict]:
    """Model Garden publisher models of google (all pages), raw. Uses the app's project as quota project."""
    from aiops_dilemma import auth
    headers = auth.auth_headers(project)
    loc = region if region and region != "global" else "us-central1"
    regions.require(loc)   # the region becomes the host name below, and the request carries a token
    out, page = [], ""
    for _ in range(50):
        url = GARDEN_URL.format(region=loc) + "?pageSize=200&listAllVersions=true" + (f"&pageToken={page}" if page else "")
        data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout))
        out += data.get("publisherModels", [])
        page = data.get("nextPageToken")
        if not page:
            break
    return out


def _stage(raw: str | None) -> str | None:
    r = str(raw or "").upper()
    if r == "GA":
        return "ga"
    if "PREVIEW" in r or r in ("EXPERIMENTAL", "EARLY_ACCESS"):
        return "preview"
    if r in ("DEPRECATED", "LEGACY"):
        return "deprecated"
    return None


STAGE_LABEL = {"ga": "GA", "preview": "Preview", "deprecated": "Deprecated", "unlisted": "Not in Model Garden yet"}


def join(models: list[dict], garden: list[dict] | None, first_seen: dict | None = None) -> list[dict]:
    """Add Model Garden id / stage / listing index and first-seen dates; sort newest first within family."""
    first_seen = first_seen or {}
    by_id = {g["id"]: g for g in garden or []}
    out = []
    for m in models:
        m = dict(m)
        mid = m["id"]
        hit = by_id.get(mid)
        if not hit and garden:
            pre = [g for g in garden if re.match(rf"^{re.escape(mid)}-(preview|exp)\b", g["id"])]
            pre.sort(key=lambda g: (0 if _stage(g["stage_raw"]) == "ga" else 1, -g["index"]))
            hit = pre[0] if pre else None
        m["model_garden_id"] = hit["id"] if hit else None
        m["garden_index"] = hit["index"] if hit else -1
        m["stage"] = _stage(hit["stage_raw"]) if hit else None
        m["first_seen"] = first_seen.get(mid)
        out.append(m)
    if garden:
        listed_max: dict[str, tuple] = {}
        for m in out:
            if m["model_garden_id"]:
                v = version_tuple(m["version"])
                listed_max[m["family"]] = max(listed_max.get(m["family"], v), v)
        for m in out:
            if not m["model_garden_id"]:
                newer = listed_max.get(m["family"])
                m["stage"] = "deprecated" if newer and newer > version_tuple(m["version"]) else "unlisted"
    for m in out:
        m["stage_label"] = STAGE_LABEL.get(m["stage"])
        m["deprecated"] = m["stage"] == "deprecated"
        m["group"] = family_label(m["family"]) if m["family"] in TIER_FAMILIES else f"Gemini {version_tuple(m['version'])[0]}"
    out.sort(key=lambda m: (family_rank(m), *[-x for x in newest_key(m)]))
    return out


def newest_key(m: dict) -> tuple:
    """Sort key, larger = newer: version, first seen by this app, Model Garden listing position."""
    fs = m.get("first_seen") or ""
    try:
        fs_n = int(fs.replace("-", "")) if fs else 0
    except ValueError:
        fs_n = 0
    return (*version_tuple(m["version"]), fs_n, m.get("garden_index", -1))


def family_rank(m: dict) -> tuple:
    """Tier families first (Flash-Lite, Flash, Pro, Ultra), then codename families of newer generations first."""
    f = m["family"]
    if f in TIER_FAMILIES:
        return (0, TIER_FAMILIES.index(f), "")
    return (1, -version_tuple(m["version"])[0], f)


# ---------------------------------------------------------------------------------------------------- aliases

def _is_ga(m: dict) -> bool:
    return m.get("stage") in ("ga", None)


def _newest(cands: list[dict]) -> dict | None:
    return max(cands, key=newest_key) if cands else None


def resolve_alias(key: str, models: list[dict]) -> dict | None:
    """The live model a role alias stands for (rules in the module docstring), or None (use the fallback)."""
    live = [m for m in models if not m.get("deprecated")]
    label, fam, _ = ALIASES[key]
    if fam:
        cands = [m for m in live if m["family"] == fam]
        ga = [m for m in cands if _is_ga(m)]
        best = _newest(ga)
        if best:
            newer_gen = [m for m in cands if version_tuple(m["version"])[0] > version_tuple(best["version"])[0]]
            return _newest(newer_gen) or best
        return _newest(cands)
    gens = [m for m in live if version_tuple(m["version"])[0] >= 4]
    if not gens:
        return None
    top = max(version_tuple(m["version"])[0] for m in gens)
    gen = [m for m in gens if version_tuple(m["version"])[0] == top]
    gen = [m for m in gen if _is_ga(m)] or gen
    hi = max(m["output_per_m"] for m in gen)
    return _newest([m for m in gen if m["output_per_m"] >= hi - 1e-9])


def alias_label(key: str, model: dict | None) -> str:
    if key == "gemini_4":
        major = version_tuple(model["version"])[0] if model else 4
        return f"Gemini {major} flagship"
    return ALIASES[key][0]


def alias_for_id(target: str | None) -> str:
    """Role alias that replaces a model id no longer offered: same family ('gemini-3.6-flash' -> gemini_flash),
    codename / newer generations -> gemini_4, anything else -> gemini_flash_lite. model.js aliasForId mirrors it."""
    t = str(target or "").lower()
    if t in ALIASES:
        return t
    m = re.match(r"^gemini-(\d+)(?:\.\d+)?-(.+)$", t)
    if not m:
        return "gemini_flash_lite"
    fam = m.group(2)
    if fam.startswith("flash-lite"):
        return "gemini_flash_lite"
    if fam.startswith("flash"):
        return "gemini_flash"
    if fam.startswith("pro"):
        return "gemini_pro"
    return "gemini_4" if int(m.group(1)) >= 4 else "gemini_flash_lite"


# ---------------------------------------------------------------------------------------------------- catalog

def _fallback_models() -> list[dict]:
    out = []
    for key, g in GEMINI_MODELS.items():
        m = {k: v for k, v in g.items() if k not in ("id", "long_input_per_m", "long_output_per_m", "default_max_rps")}
        lc = g.get("long_context")
        if not lc and (g.get("long_input_per_m", g["input_per_m"]) != g["input_per_m"]
                       or g.get("long_output_per_m", g["output_per_m"]) != g["output_per_m"]):
            lc = {"threshold": LONG_THRESHOLD, "input_per_m": g["long_input_per_m"], "output_per_m": g["long_output_per_m"],
                  "cached_per_m": round(g["cached_per_m"] * g["long_input_per_m"] / g["input_per_m"], 6)}
        m.update({"id": g["model_id"], "family": g["family"], "long_context": lc, "source": "fallback",
                  "stage": g.get("stage"), "stage_label": STAGE_LABEL.get(g.get("stage")), "deprecated": False,
                  "model_garden_id": None, "garden_index": -1, "first_seen": None, "batch": None, "regional": None,
                  "context_window": g.get("max_context"), "fallback_for": key})
        m["group"] = family_label(m["family"]) if m["family"] in TIER_FAMILIES else f"Gemini {version_tuple(m['version'])[0]}"
        out.append(m)
    return out


def build(sku_list: list[dict] | None, garden_raw: list[dict] | None, first_seen: dict | None = None,
          now: float | None = None, errors: dict | None = None) -> dict:
    """The catalog: {models, aliases, meta}. `sku_list` None/empty or nothing parsed -> fallback models only."""
    now = now or time.time()
    parsed = parse_billing(sku_list or [])
    garden = parse_garden(garden_raw) if garden_raw else None
    live = bool(parsed)
    models = join(parsed, garden, first_seen) if live else _fallback_models()
    for m in models:
        m.setdefault("context_window", None)
    aliases = {}
    fb = {m["fallback_for"]: m for m in _fallback_models()}
    for key, (label, fam, rule) in ALIASES.items():
        hit = resolve_alias(key, models) if live else None
        target = hit or fb[key]
        if not hit and live:      # a family missing from the live catalog: keep the reference model, labelled
            models.append(target)
        aliases[key] = {"key": key, "label": alias_label(key, target), "rule": rule, "model_id": target["id"],
                        "name": target["name"], "source": target["source"]}
    return {
        "format": FORMAT,
        "models": models,
        "aliases": aliases,
        "meta": {"source": "billing-catalog" if live else "fallback", "updated_at": now,
                 "model_garden": garden is not None, "billing": live,
                 "counts": {"models": len(models), "shown": sum(1 for m in models if not m.get("deprecated")),
                            "garden_listed": len(garden or [])},
                 "errors": errors or {}},
    }


def _engine_entry(m: dict) -> dict:
    """Model entry plus the fields the cost engines read (long_input_per_m / long_output_per_m, model_id)."""
    e = copy.deepcopy(m)
    lc = m.get("long_context") or {}
    e["model_id"] = m["id"]
    e["long_input_per_m"] = lc.get("input_per_m", m["input_per_m"])
    e["long_output_per_m"] = lc.get("output_per_m", m["output_per_m"])
    e.setdefault("max_context", m.get("context_window") or 1048576)
    return e


def engine_map(cat: dict) -> dict:
    """{alias key or model id: entry} for costmodel / model.js. Aliases first, each a copy of its model with
    id = alias, model_id = the model, alias = True, role / rule."""
    by_id = {m["id"]: m for m in cat["models"]}
    out = {}
    for key, a in cat["aliases"].items():
        m = by_id.get(a["model_id"])
        if not m:
            continue
        e = _engine_entry(m)
        e.update({"id": key, "alias": True, "role": a["label"], "rule": a["rule"]})
        out[key] = e
    for mid, m in by_id.items():
        out.setdefault(mid, _engine_entry(m))
    return out


def models_list(cat: dict) -> list[dict]:
    """/api/catalog `gemini_models`: the documented fields per model, newest first within family."""
    keep = ("id", "name", "family", "group", "version", "stage", "stage_label", "deprecated", "input_per_m",
            "output_per_m", "cached_per_m", "cached_estimated", "long_context", "batch", "regional", "scope", "source",
            "first_seen", "context_window", "model_garden_id", "skus")
    roles = {}
    for key, a in cat["aliases"].items():
        roles.setdefault(a["model_id"], []).append(key)
    return [{**{k: m.get(k) for k in keep}, "released": m.get("first_seen"), "roles": roles.get(m["id"], [])}
            for m in cat["models"]]


# ---------------------------------------------------------------------------------------------------- cache

def _cache_file():
    return paths.cache_dir() / "gemini_catalog.json"


def _seen_file():
    return paths.cache_dir() / "gemini_first_seen.json"


def _read_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def cached() -> dict | None:
    """The last live catalog (memory, else disk), never fetched; None in tests or before the first fetch."""
    if not live_enabled():
        return None
    with _LOCK:
        mem = _MEM.get("cat")
    if mem:
        return mem
    d = _read_json(_cache_file())
    if d and d.get("format") == FORMAT and d.get("meta", {}).get("source") == "billing-catalog":
        with _LOCK:
            _MEM["cat"] = d
        return d
    return None


def fetch(project: str | None, region: str = "us-central1") -> dict:
    """Build the live catalog now (Billing SKUs at most a day old + Model Garden listing) and store it."""
    from aiops_dilemma import pricing
    errors = {}
    sku_list = None
    try:
        sku_list = pricing.skus("vertex", project, max_age=TTL_S)
    except Exception as e:
        errors["billing"] = str(e)[:300]
    garden_raw = None
    try:
        proj = pricing._quota_project(project)
        if proj:
            garden_raw = fetch_garden(proj, region)
        else:
            errors["model_garden"] = "no project"
    except Exception as e:
        errors["model_garden"] = str(e)[:300]
    seen = _read_json(_seen_file()) or {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cat = build(sku_list, garden_raw, seen, errors=errors)
    if cat["meta"]["source"] == "billing-catalog":
        changed = False
        for m in cat["models"]:
            if m["source"] == "billing-catalog" and m["id"] not in seen:
                seen[m["id"]] = today
                m["first_seen"] = today
                changed = True
        if changed:
            cat = build(sku_list, garden_raw, seen, errors=errors)   # re-sort / re-resolve with the dates
        try:
            paths.cache_dir().mkdir(parents=True, exist_ok=True)
            if changed:
                _seen_file().write_text(json.dumps(seen, indent=1, sort_keys=True))
            tmp = _cache_file().with_suffix(".tmp")
            tmp.write_text(json.dumps(cat))
            tmp.replace(_cache_file())
        except Exception:
            pass
        with _LOCK:
            _MEM["cat"] = cat
    return cat


def refresh_bg(project: str | None, region: str = "us-central1") -> bool:
    """Refresh in a background thread (one at a time); ON_UPDATE callbacks run when a live catalog landed."""
    with _LOCK:
        if "bg" in _REFRESHING:
            return False
        _REFRESHING.add("bg")

    def work():
        try:
            cat = fetch(project, region)
            if cat["meta"]["source"] == "billing-catalog":
                for cb in list(ON_UPDATE):
                    try:
                        cb()
                    except Exception:
                        log.exception("gemini ON_UPDATE callback failed")
        except Exception:
            log.exception("gemini catalog refresh failed")
        finally:
            with _LOCK:
                _REFRESHING.discard("bg")

    threading.Thread(target=work, daemon=True).start()
    return True


def refreshing() -> bool:
    with _LOCK:
        return "bg" in _REFRESHING


def get(project: str | None = None, region: str = "us-central1", refresh: bool = False) -> dict:
    """The Gemini catalog: live (cached a day; stale -> served while a background refresh runs), else fallback.

    Blocks only when there is no live catalog on disk yet (first run) or `refresh`; tests always get the fallback."""
    if not live_enabled():
        return build(None, None)
    if refresh:
        cat = fetch(project, region)
        return cat if cat["meta"]["source"] == "billing-catalog" else (cached() or cat)
    cat = cached()
    if cat:
        if time.time() - float(cat["meta"].get("updated_at") or 0) > TTL_S:
            refresh_bg(project, region)
        return cat
    with _LOCK:
        failed_at = _MEM.get("failed_at", 0)
    if time.time() - failed_at < 600:      # a failed first fetch: don't retry on every call for 10 minutes
        return build(None, None, errors=_MEM.get("errors"))
    cat = fetch(project, region)
    if cat["meta"]["source"] != "billing-catalog":
        with _LOCK:
            _MEM["failed_at"] = time.time()
            _MEM["errors"] = cat["meta"].get("errors")
    return cat


def offered() -> dict | None:
    """{model id: model} of the cached live catalog, or None when there is none (nothing to validate against)."""
    cat = cached()
    return {m["id"]: m for m in cat["models"]} if cat else None


def check_target(value, field: str, allow_open_maas: bool = True) -> tuple:
    """Normalize a hybrid target: open_maas, a role alias or a live model id. A model id the live catalog no longer
    offers -> its family's alias + a warning; a deprecated one is kept with a warning. Without a live catalog any
    gemini-* id is kept (it can't be checked)."""
    v = str(value or "").strip().lower()
    if v == "open_maas" and allow_open_maas:
        return v, None
    if v in ALIASES:
        return v, None
    if v.startswith("gemini-"):
        live = offered()
        if live is None:
            return v, None
        m = live.get(v)
        if m and not m.get("deprecated"):
            return v, None
        alias = alias_for_id(v)
        cat = cached() or {}
        a = (cat.get("aliases") or {}).get(alias) or {}
        repl = f"{a.get('label', ALIASES[alias][0])} (latest: {a.get('name', '?')})"
        if m:
            return v, f"{m['name']} ({field}) is deprecated; consider {repl}."
        return alias, f"Gemini model {v!r} ({field}) is no longer offered in the Cloud Billing Catalog; using {repl}."
    return value, None
