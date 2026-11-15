"""Dynamic model metadata discovery from Model Garden and Hugging Face Hub."""

import copy
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request

from aiops_dilemma import hfmeta, hftoken, paths, regions
from aiops_dilemma.catalog import SEED_MODELS

log = logging.getLogger(__name__)

HF = "https://huggingface.co/{id}/resolve/main/{file}"
HF_API = "https://huggingface.co/api/models"
CACHE = paths.cache_dir()   # DILEMMA_CACHE_DIR overrides
MODEL_CACHE_TTL = 86400  # 24 hours


def _get_json(url: str, timeout: int = 20):
    """GET a huggingface.co JSON document (with the Hugging Face token when one is set)."""
    req = urllib.request.Request(url, headers=hftoken.headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def weight_bytes(model: dict) -> float:
    """BF16 weight size in bytes: from the HF safetensors index, else params_b, else seed catalog."""
    if model.get("params_b"):
        return float(model["params_b"]) * 1e9 * 2
    hf_id = model.get("hf_id")
    mg_id = model.get("model_garden_id")
    live = _live_entry(model)
    if live and live.get("params_b"):
        return float(live["params_b"]) * 1e9 * 2
    for s in SEED_MODELS:
        if (hf_id and s["hf_id"].lower() == hf_id.lower()) or (mg_id and s["model_garden_id"].lower() == mg_id.lower()):
            return float(s["params_b"]) * 1e9 * 2
    entry = find_model(model) if (hf_id or mg_id) else None   # curated catalog (e.g. Kimi K2: 1T, no size in the id)
    if entry and entry.get("params_b"):
        return float(entry["params_b"]) * 1e9 * 2
    if hf_id:
        try:
            idx = _get_json(HF.format(id=hf_id, file="model.safetensors.index.json"))
            return float(idx["metadata"]["total_size"])
        except Exception:
            pass
        try:  # single-file models: read the safetensors header for tensor sizes
            req = urllib.request.Request(HF.format(id=hf_id, file="model.safetensors"),
                                         method="HEAD", headers=hftoken.headers())
            with urllib.request.urlopen(req, timeout=20) as r:
                return float(r.headers.get("Content-Length") or r.headers.get("X-Linked-Size"))
        except Exception:
            pass
        # Fallback: parse parameter count from model name like "27b" or "8b"
        m = re.search(r"(\d+(?:\.\d+)?)b\b", hf_id, re.I)
        if m:
            return float(m.group(1)) * 1e9 * 2
    if mg_id:
        m = re.search(r"(\d+(?:\.\d+)?)b\b", mg_id, re.I)
        if m:
            return float(m.group(1)) * 1e9 * 2
    raise SystemExit("can't determine model size: set model.params_b (billions of parameters) in the config")


def _live_entry(model: dict) -> dict | None:
    """The curated catalog entry when its size comes from live Hugging Face metadata (else None: seeds / rules)."""
    if hfmeta.disabled() or not (model.get("hf_id") or model.get("model_garden_id")):
        return None
    try:
        e = find_model(model)
    except Exception:
        return None
    return e if e and (e.get("field_sources") or {}).get("params_b") == "hf" else None


def active_params_b(model: dict) -> float:
    """Active parameters in billions (for MoE models like Gemma 4 26B-A4B, returns 4.0)."""
    if model.get("active_params_b"):
        return float(model["active_params_b"])
    hf_id = model.get("hf_id") or ""
    mg_id = model.get("model_garden_id") or ""
    live = _live_entry(model)
    if live and live.get("active_params_b"):
        # the curated entry's active size (live formula, or the published figure when the formula disagrees)
        return float(live["active_params_b"])
    for s in SEED_MODELS:
        if (hf_id and s["hf_id"].lower() == hf_id.lower()) or (mg_id and s["model_garden_id"].lower() == mg_id.lower()):
            return float(s.get("active_params_b") or s["params_b"])
    # Check for "-a4b" / "-a3b" MoE naming convention
    m = re.search(r"-a(\d+(?:\.\d+)?)b\b", hf_id or mg_id, re.I)
    if m:
        return float(m.group(1))
    return weight_bytes(model) / 2e9


# INT4 (4-bit weights: AWQ / GPTQ / W4A16 / compressed-tensors) is 0.5 byte per parameter plus group scales and the
# layers kept in 16 bits (attention, shared experts, embeddings): x1.2. Kimi K2 Thinking: ~1.03T params, 594 GB.
INT4_BYTES_PER_PARAM = 0.5
INT4_OVERHEAD = 1.2
INT4_KEYS = ("int4", "w4a16", "awq", "gptq")   # `prequantized` keys that are 4-bit integer checkpoints


# MXFP4 (OpenAI gpt-oss): the expert weights are 4-bit E2M1 with a shared 8-bit scale per 32 values (4.25 bits per
# parameter); attention, router, embeddings and biases stay in BF16. Sized from the Hub's per-dtype safetensors counts
# when the live metadata has them (`mxfp4_weight_gb`, hfmeta.mxfp4_weight_bytes), else ~0.55 byte per parameter.
MXFP4_BYTES_PER_PARAM = 0.55
MXFP4_KEYS = ("mxfp4",)          # `prequantized` keys that are MXFP4 checkpoints
GPU_ONLY_PRECISIONS = ("int4", "mxfp4")   # 4-bit GPU kernels (Marlin, Triton MXFP4 MoE): no TPU path in this tool


def bytes_per_param(precision: str | None) -> float:
    p = (precision or "bf16").lower()
    return (1.0 if p == "fp8" else INT4_BYTES_PER_PARAM * INT4_OVERHEAD if p == "int4" else
            MXFP4_BYTES_PER_PARAM if p == "mxfp4" else 2.0)


def mxfp4_weights(entry: dict | None, params_b: float | None = None) -> dict:
    """{gb, source, note} of a model's MXFP4 weights: from the live Hugging Face safetensors counts when known
    (source "hf"), else params x 0.55 byte (source "estimate")."""
    entry = entry or {}
    if entry.get("mxfp4_weight_gb"):
        return {"gb": float(entry["mxfp4_weight_gb"]), "source": "hf",
                "note": entry.get("mxfp4_weight_note") or "from the Hub's safetensors counts"}
    pb = float(params_b if params_b is not None else entry.get("params_b") or 0)
    return {"gb": pb * MXFP4_BYTES_PER_PARAM, "source": "estimate",
            "note": f"estimated at ~{MXFP4_BYTES_PER_PARAM} byte per parameter (4.25-bit experts, the rest in BF16)"}


def weight_bytes_at(entry: dict | None, precision: str | None, params_b: float | None = None) -> float:
    """Weight bytes of a catalog entry at a precision (MXFP4 from the checkpoint's real size when known)."""
    pb = float(params_b if params_b is not None else (entry or {}).get("params_b") or 0)
    if (precision or "").lower() == "mxfp4":
        return mxfp4_weights(entry, pb)["gb"] * 1e9
    return pb * 1e9 * bytes_per_param(precision)


def precision_bytes(model: dict) -> float:
    """Weight bytes at the model's precision (BF16 2, FP8 1, INT4 0.6 bytes per parameter; MXFP4 the checkpoint's
    size from live Hugging Face metadata, else 0.55)."""
    if (model.get("precision") or "").lower() == "mxfp4":
        if model.get("mxfp4_weight_gb"):
            return float(model["mxfp4_weight_gb"]) * 1e9
        try:
            entry = find_model(model) if (model.get("hf_id") or model.get("model_garden_id")) else None
        except Exception:
            entry = None
        if entry and entry.get("mxfp4_weight_gb") and not (
                model.get("params_b") and entry.get("params_b")
                and abs(float(model["params_b"]) - float(entry["params_b"])) > 0.05 * float(entry["params_b"])):
            return float(entry["mxfp4_weight_gb"]) * 1e9
    return weight_bytes(model) / 2 * bytes_per_param(model.get("precision"))


def int4_checkpoint(entry: dict | None) -> str | None:
    """Model Garden id of the model's 4-bit checkpoint: its own id when it is published in INT4, else a pre-quantized
    AWQ / GPTQ / W4A16 / INT4 checkpoint from `prequantized`; None when there is none (INT4 isn't offered then)."""
    if not entry:
        return None
    if entry.get("native_precision") == "int4":
        return entry.get("model_garden_id") or entry.get("hf_id")
    pq = entry.get("prequantized") or {}
    return next((pq[k] for k in INT4_KEYS if pq.get(k)), None)


def mxfp4_checkpoint(entry: dict | None) -> str | None:
    """Model Garden id of the model's MXFP4 checkpoint: its own id when it is published in MXFP4 (gpt-oss), else a
    pre-quantized `-mxfp4` checkpoint; None when there is none (MXFP4 isn't offered then)."""
    if not entry:
        return None
    if entry.get("native_precision") == "mxfp4":
        return entry.get("model_garden_id") or entry.get("hf_id")
    pq = entry.get("prequantized") or {}
    return next((pq[k] for k in MXFP4_KEYS if pq.get(k)), None)


_VERIFIED_MEM: dict[tuple, tuple[float, list[dict]]] = {}


def verified_options(project: str | None, region: str, model_garden_id: str) -> list[dict]:
    """Model Garden's verified deployment configs (machine + container) for this model (memoized for an hour)."""
    if not project or project == "my-gcp-project":
        return []
    key = (project, region, model_garden_id)
    hit = _VERIFIED_MEM.get(key)
    if hit and time.time() - hit[0] < 3600:
        return copy.deepcopy(hit[1])
    out = _verified_options(project, region, model_garden_id)
    _VERIFIED_MEM[key] = (time.time(), out)
    return copy.deepcopy(out)


def _verified_options(project: str, region: str, model_garden_id: str) -> list[dict]:
    try:
        import vertexai
        from vertexai import model_garden

        from aiops_dilemma import auth
        vertexai.init(project=project, location=regions.require(region), credentials=auth.credentials(project))
        opts = model_garden.OpenModel(model_garden_id).list_deploy_options()
    except Exception as e:  # model not in Model Garden, no access, etc.
        print(f"  (no Model Garden deploy options: {str(e)[:120]})")
        return []
    out = []
    for o in opts:
        ms = o.dedicated_resources.machine_spec
        acc = ms.accelerator_type
        if isinstance(acc, int):  # some SDK versions return the raw enum value
            from google.cloud.aiplatform_v1.types import AcceleratorType
            acc = AcceleratorType(acc)
        acc = acc.name if hasattr(acc, "name") else str(acc)
        c = o.container_spec
        out.append({"machine_type": ms.machine_type, "accelerator_type": acc,
                    "accelerator_count": ms.accelerator_count, "image": c.image_uri,
                    "args": list(c.args), "env": {e.name: e.value for e in c.env},
                    "predict_route": c.predict_route or "/generate", "health_route": c.health_route or "/ping",
                    "port": (c.ports[0].container_port if c.ports else 8080)})
    return out


def inspect_hf_model(hf_id: str) -> dict:
    """Model specs (total / MoE active parameters, context, native precision, license) read live from the Hugging
    Face Hub (hfmeta: model API + config.json); cached like the catalog's metadata."""
    info: dict = {"hf_id": hf_id}
    if not hfmeta.valid_id(hf_id):
        return {**info, "error": "not a Hugging Face model id (org/name)"}
    rec = hfmeta.fetch(hf_id)
    if rec.get("status") == "ok":
        hfmeta._save_raw(hf_id, rec)
    meta = hfmeta.parse(rec)
    if not meta:
        return {**info, "error": f"not found on Hugging Face (HTTP {rec.get('http_status')})"}
    f = meta["fields"]
    info.update({k: f[k] for k in ("params_b", "active_params_b", "max_context", "checkpoint_precision",
                                   "native_precision", "released", "license", "gated") if f.get(k) is not None})
    if f.get("dense") and f.get("params_b") and not f.get("active_params_b"):
        info["active_params_b"] = f["params_b"]
    info.update(hf_id=meta["hf_id"], source="hf", notes=meta["notes"])
    return info


def _list_vertex_model_garden(project: str | None, region: str) -> list[str]:
    """Model Garden ids of deployable open models (see _list_model_garden, which also returns Hugging Face ids)."""
    return _list_model_garden(project, region)[0]


def _list_model_garden(project: str | None, region: str) -> tuple[list[str], dict[str, str]]:
    """(Model Garden ids `publisher/model@version`, {id: Hugging Face id}) of deployable open models.

    The Hugging Face id is the deploy recipe's MODEL_ID (e.g. `google/gemma-4-26B-A4B-it`, canonical casing); ids
    whose recipe loads from Cloud Storage have none. Falls back to the SDK listing (ids only)."""
    if not project or project == "my-gcp-project":
        return [], {}
    try:
        import vertexai
        from google.cloud.aiplatform import initializer
        from google.cloud.aiplatform_v1beta1 import types
        from vertexai.model_garden._model_garden import _NATIVE_MODEL_FILTER, _ModelGardenClientWithOverride

        from aiops_dilemma import auth
        vertexai.init(project=project, location=regions.require(region), credentials=auth.credentials(project))
        client = initializer.global_config.create_client(
            client_class=_ModelGardenClientWithOverride, credentials=initializer.global_config.credentials,
            location_override="us-central1")
        req = types.ListPublisherModelsRequest(parent="publishers/*", list_all_versions=True,
                                               filter=_NATIVE_MODEL_FILTER)
        ids, hf = [], {}
        for page in client.list_publisher_models(req).pages:
            for pm in page.publisher_models:
                acts = pm.supported_actions
                if not acts.multi_deploy_vertex.multi_deploy_vertex:
                    continue
                raw = re.sub(r"publishers/(hf-|)|models/", "", pm.name) + "@" + pm.version_id
                ids.append(raw)
                specs = [d.container_spec for d in acts.multi_deploy_vertex.multi_deploy_vertex]
                if acts.deploy and acts.deploy.container_spec:
                    specs.insert(0, acts.deploy.container_spec)
                for c in specs:
                    mid = next((e.value for e in c.env if e.name == "MODEL_ID"), None)
                    if mid and hfmeta.valid_id(mid):
                        hf[raw] = mid
                        break
        return ids, hf
    except Exception as e:
        print(f"  (Model Garden listing with Hugging Face ids failed, using the SDK listing: {str(e)[:120]})")
    try:
        import vertexai
        from vertexai import model_garden

        from aiops_dilemma import auth
        vertexai.init(project=project, location=regions.require(region), credentials=auth.credentials(project))
        if hasattr(model_garden, "list_deployable_models"):
            res = model_garden.list_deployable_models()
            return (list(res) if isinstance(res, (list, tuple, set)) else []), {}
    except Exception:
        pass
    return [], {}


def _search_hf_open_models(limit: int = 8) -> list[str]:
    """Trending Hugging Face text-generation repos from key families (raw ids, curated like Model Garden ids)."""
    out = []
    try:
        q = urllib.parse.urlencode({"filter": "text-generation", "sort": "Trending", "limit": limit})
        for item in _get_json(f"{HF_API}?{q}", timeout=8):
            mid = item.get("id", "")
            if mid and re.search(r"(instruct|-it\b|r1|qwen|gemma|llama|mistral)", mid, re.I):
                out.append(mid)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Curation: raw Model Garden ids -> deployable text LLMs, one entry per model (quantized variants folded in)
# ---------------------------------------------------------------------------------------------------------------------

CURATED_FORMAT = 2   # bump when curation rules change: older caches are re-curated
CURATED_CACHE = f"models_curated_v{CURATED_FORMAT}.json"
LEGACY_CACHE = "models_live.json"   # format 1 (uncurated); only its Model Garden ids are reused

# Not text LLMs (image / video / vision-language / embeddings / rerankers / diffusion) or not serveable as one.
_NON_TEXT = re.compile(
    r"(blip|llava|paligemma|flux|wan2|wan-ai|cogvideo|diffusion|stable-diffusion|sdxl|imagen|embed|rerank|"
    r"-vl-|-vl@|qwen3-vl|-vision|image-to-text|visual-question|-dicom|-t2v-|-i2v-|-ti2v-|whisper|clip|siglip|"
    r"owlvit|detr|yolo|sam\b|segment|tts|audio|speech|ocr|gguf|image|cosmos|vqa|retrieval|dia-\d)", re.I)
# Hugging Face pipeline tags of models that aren't text generators (live counterpart of _NON_TEXT).
NON_TEXT_PIPELINES = {
    "feature-extraction", "sentence-similarity", "text-classification", "text-ranking", "token-classification",
    "fill-mask", "zero-shot-classification", "question-answering", "table-question-answering", "summarization",
    "translation", "text-to-image", "image-to-image", "image-to-text", "image-classification", "object-detection",
    "image-segmentation", "mask-generation", "depth-estimation", "zero-shot-image-classification",
    "unconditional-image-generation", "text-to-video", "image-to-video", "video-classification",
    "automatic-speech-recognition", "text-to-speech", "text-to-audio", "audio-classification", "audio-to-audio",
    "document-question-answering", "visual-question-answering", "time-series-forecasting", "image-feature-extraction"}
# Pre-quantized checkpoint suffixes -> precision key in `prequantized`.
_QUANT = re.compile(r"-(fp8(?:-dynamic)?|awq|gptq|quantized[.\-_]w8a8|w8a8|nvfp4|mxfp4|fp4|int4|int8)$", re.I)
_QUANT_KEY = {"fp8": "fp8", "fp8-dynamic": "fp8", "awq": "awq", "gptq": "gptq", "w8a8": "w8a8", "nvfp4": "nvfp4",
              "mxfp4": "mxfp4", "fp4": "nvfp4", "int4": "int4", "int8": "int8"}
_BASE = re.compile(r"(^|-)(base|pt)(-|$)", re.I)
_INSTRUCT_TOKENS = {"it", "instruct", "chat"}
_NOISE_SUFFIX = re.compile(r"(-gg-hf|-hf|-bf16)$", re.I)

GENERAL_TASKS = ["extraction", "summarization", "reasoning", "coding", "chat"]
SPECIALTY_TASKS = {"code": ["coding"], "medical": ["medical", "extraction", "summarization"],
                   "translation": ["translation"], "therapeutics": ["therapeutics"], "safety": ["safety"],
                   "math": ["math", "reasoning"], "multilingual": ["multilingual", "extraction", "summarization", "chat"]}

# Ordered (first match wins): key regex -> facts. `fam` = family shown to users; gen = sortable generation number;
# rel = release month (None when unknown); ctx = max context; params/active override the name-derived sizes.
_FAMILY_RULES: list[tuple[str, dict]] = [
    # Gemma and Gemma-based specialists
    (r"^medgemma-1\.5", {"fam": "Gemma", "gen": 3, "rel": "2026-01", "ctx": 131072, "spec": "medical"}),
    (r"^medgemma", {"fam": "Gemma", "gen": 3, "rel": "2025-05", "ctx": 131072, "spec": "medical"}),
    (r"^translategemma", {"fam": "Gemma", "gen": 3, "rel": "2026-01", "ctx": 131072, "spec": "translation"}),
    (r"^txgemma", {"fam": "Gemma", "gen": 2, "rel": "2025-03", "ctx": 8192, "spec": "therapeutics"}),
    (r"^codegemma", {"fam": "Gemma", "gen": 1, "rel": "2024-04", "ctx": 8192, "spec": "code", "legacy": True}),
    (r"^gemma-sea-lion-v4", {"fam": "Gemma", "gen": 3, "rel": "2025-08", "ctx": 131072, "spec": "multilingual"}),
    (r"^gemma-4-e2b", {"fam": "Gemma", "gen": 4, "rel": "2026-04", "ctx": 131072, "params": 5.1, "active": 2.3}),
    (r"^gemma-4-e4b", {"fam": "Gemma", "gen": 4, "rel": "2026-04", "ctx": 131072, "params": 8.0, "active": 4.5}),
    (r"^gemma-4-26b-a4b", {"fam": "Gemma", "gen": 4, "rel": "2026-04", "ctx": 262144, "params": 26.0, "active": 4.0}),
    (r"^gemma-4-31b", {"fam": "Gemma", "gen": 4, "rel": "2026-04", "ctx": 262144}),
    (r"^gemma-4-", {"fam": "Gemma", "gen": 4, "rel": None, "ctx": 131072}),
    (r"^gemma-3n-e2b", {"fam": "Gemma", "gen": 3, "rel": "2025-06", "ctx": 32768, "params": 5.0, "active": 2.0}),
    (r"^gemma-3n-e4b", {"fam": "Gemma", "gen": 3, "rel": "2025-06", "ctx": 32768, "params": 8.0, "active": 4.0}),
    (r"^gemma-3-1b", {"fam": "Gemma", "gen": 3, "rel": "2025-03", "ctx": 32768}),
    (r"^gemma-3-", {"fam": "Gemma", "gen": 3, "rel": "2025-03", "ctx": 131072}),
    (r"^gemma-2-", {"fam": "Gemma", "gen": 2, "rel": "2024-06", "ctx": 8192, "legacy": True}),
    (r"^gemma-1\.1", {"fam": "Gemma", "gen": 1.1, "rel": "2024-04", "ctx": 8192, "legacy": True}),
    (r"^gemma-\d+b", {"fam": "Gemma", "gen": 1, "rel": "2024-02", "ctx": 8192, "legacy": True}),
    # Llama
    (r"^llama-guard-4", {"fam": "Llama", "gen": 4, "rel": "2025-04", "ctx": 131072, "spec": "safety"}),
    (r"^llama-guard-3", {"fam": "Llama", "gen": 3.1, "rel": "2024-07", "ctx": 131072, "spec": "safety"}),
    (r"^llama-4-maverick-17b-128e", {"fam": "Llama", "gen": 4, "rel": "2025-04", "ctx": 1048576,
                                     "params": 400.0, "active": 17.0}),
    (r"^llama-4-scout-17b-16e", {"fam": "Llama", "gen": 4, "rel": "2025-04", "ctx": 10485760,
                                 "params": 109.0, "active": 17.0}),
    (r"^llama-3\.3", {"fam": "Llama", "gen": 3.3, "rel": "2024-12", "ctx": 131072}),
    (r"^llama-3[.\-]2", {"fam": "Llama", "gen": 3.2, "rel": "2024-09", "ctx": 131072}),
    (r"^llama-3\.1", {"fam": "Llama", "gen": 3.1, "rel": "2024-07", "ctx": 131072}),
    (r"^llama-guard-3", {"fam": "Llama", "gen": 3.1, "rel": "2024-07", "ctx": 131072, "spec": "safety"}),
    (r"^(meta-)?llama-3-", {"fam": "Llama", "gen": 3, "rel": "2024-04", "ctx": 8192, "legacy": True}),
    (r"^llama-2", {"fam": "Llama", "gen": 2, "rel": "2023-07", "ctx": 4096, "legacy": True}),
    (r"^codellama", {"fam": "Llama", "gen": 2, "rel": "2023-08", "ctx": 16384, "spec": "code", "legacy": True}),
    # Qwen
    (r"^qwen3-coder-next", {"fam": "Qwen", "gen": 3, "rel": None, "ctx": 262144, "params": 80.0, "active": 3.0,
                            "spec": "code"}),
    (r"^qwen3-coder-\d", {"fam": "Qwen", "gen": 3, "rel": "2025-07", "ctx": 262144, "spec": "code"}),
    (r"^qwen3-next", {"fam": "Qwen", "gen": 3, "rel": "2025-09", "ctx": 262144}),
    (r"^qwen3-.*-2507", {"fam": "Qwen", "gen": 3, "rel": "2025-07", "ctx": 262144}),
    (r"^qwen3\.8", {"fam": "Qwen", "gen": 3.8, "rel": None, "ctx": 262144}),
    (r"^qwen3\.6", {"fam": "Qwen", "gen": 3.6, "rel": None, "ctx": 262144}),
    (r"^qwen3\.5", {"fam": "Qwen", "gen": 3.5, "rel": "2026-02", "ctx": 262144}),
    (r"^qwen3-", {"fam": "Qwen", "gen": 3, "rel": "2025-04", "ctx": 131072}),
    (r"^qwq", {"fam": "Qwen", "gen": 2.5, "rel": "2025-03", "ctx": 131072}),
    (r"^qwen2\.5", {"fam": "Qwen", "gen": 2.5, "rel": "2024-09", "ctx": 131072}),
    (r"^qwen2-", {"fam": "Qwen", "gen": 2, "rel": "2024-06", "ctx": 32768, "legacy": True}),
    # Mistral AI
    (r"^ministral-3-", {"fam": "Mistral AI", "gen": 3, "rel": "2025-12", "ctx": 262144}),
    (r"^mistral-small-24b", {"fam": "Mistral AI", "gen": 3, "rel": "2025-01", "ctx": 32768}),
    (r"^mixtral-8x22b", {"fam": "Mistral AI", "gen": 1, "rel": "2024-04", "ctx": 65536, "params": 141.0,
                         "active": 39.0, "legacy": True}),
    (r"^mixtral-8x7b", {"fam": "Mistral AI", "gen": 1, "rel": "2023-12", "ctx": 32768, "params": 46.7,
                        "active": 12.9, "legacy": True}),
    (r"^mistral-7b-.*v0\.3", {"fam": "Mistral AI", "gen": 1, "rel": "2024-05", "ctx": 32768, "legacy": True}),
    (r"^mistral-7b-.*v0\.2", {"fam": "Mistral AI", "gen": 1, "rel": "2023-12", "ctx": 32768, "legacy": True}),
    (r"^mistral-7b", {"fam": "Mistral AI", "gen": 1, "rel": "2023-09", "ctx": 8192, "legacy": True}),
    (r"^mistral-large-3", {"fam": "Mistral AI", "gen": 3, "rel": "2025-12", "ctx": 262144, "params": 675.0,
                           "active": 41.0}),
    (r"^mistral-nemo", {"fam": "Mistral AI", "gen": 2, "rel": "2024-07", "ctx": 131072, "params": 12.2}),
    # DeepSeek (no size in the Model Garden name for the full-size models)
    (r"^deepseek-r1-0528-qwen3", {"fam": "DeepSeek", "gen": 1, "rel": "2025-05", "ctx": 131072}),
    (r"^deepseek-r1-distill", {"fam": "DeepSeek", "gen": 1, "rel": "2025-01", "ctx": 131072}),
    (r"^deepseek-r1-0528$", {"fam": "DeepSeek", "gen": 1, "rel": "2025-05", "ctx": 163840, "params": 671.0,
                             "active": 37.0}),
    (r"^deepseek-r1$", {"fam": "DeepSeek", "gen": 1, "rel": "2025-01", "ctx": 163840, "params": 671.0,
                        "active": 37.0}),
    (r"^deepseek-v3-2", {"fam": "DeepSeek", "gen": 3.2, "rel": "2025-12", "ctx": 163840, "params": 671.0,
                         "active": 37.0}),
    (r"^deepseek-v3-1", {"fam": "DeepSeek", "gen": 3.1, "rel": "2025-08", "ctx": 163840, "params": 671.0,
                         "active": 37.0}),
    (r"^deepseek-v3-0324", {"fam": "DeepSeek", "gen": 3, "rel": "2025-03", "ctx": 163840, "params": 671.0,
                            "active": 37.0}),
    (r"^deepseek-v3$", {"fam": "DeepSeek", "gen": 3, "rel": "2024-12", "ctx": 163840, "params": 671.0,
                        "active": 37.0}),
    # Other open-weight lines
    # native: the precision the weights are published in (no BF16 checkpoint). Verified 2026-11-09: the Hugging Face
    # configs of Kimi K2 Thinking / K2.5 / K2.6 / K2.7-Code carry a compressed-tensors quantization_config (INT4
    # weights, group size 32, "pack-quantized"; ~594 GB checkpoints) and Model Garden's recipes serve them as-is,
    # even on 8 x H100 (640 GB); Kimi K2 Instruct (-0905) is published in FP8 (~1.03 TB, block-wise e4m3).
    (r"^kimi-k2\.7-code", {"fam": "Kimi", "gen": 2.7, "rel": None, "ctx": 262144, "params": 1000.0, "active": 32.0,
                           "spec": "code", "native": "int4"}),
    (r"^kimi-k2\.6", {"fam": "Kimi", "gen": 2.6, "rel": None, "ctx": 262144, "params": 1000.0, "active": 32.0,
                      "native": "int4"}),
    (r"^kimi-k2\.5", {"fam": "Kimi", "gen": 2.5, "rel": "2026-01", "ctx": 262144, "params": 1000.0, "active": 32.0,
                      "native": "int4"}),
    (r"^kimi-k2-thinking", {"fam": "Kimi", "gen": 2, "rel": "2025-11", "ctx": 262144, "params": 1000.0,
                            "active": 32.0, "native": "int4"}),
    (r"^kimi-k2-instruct-0905", {"fam": "Kimi", "gen": 2, "rel": "2025-09", "ctx": 262144, "params": 1000.0,
                                 "active": 32.0, "native": "fp8"}),
    (r"^kimi-k2-instruct", {"fam": "Kimi", "gen": 2, "rel": "2025-07", "ctx": 131072, "params": 1000.0,
                            "active": 32.0, "native": "fp8"}),
    (r"^glm-4\.5-air", {"fam": "GLM", "gen": 4.5, "rel": "2025-07", "ctx": 131072, "params": 106.0, "active": 12.0}),
    (r"^glm-4\.5", {"fam": "GLM", "gen": 4.5, "rel": "2025-07", "ctx": 131072, "params": 355.0, "active": 32.0}),
    (r"^glm-4\.7-flash", {"fam": "GLM", "gen": 4.7, "rel": "2026-01", "ctx": 202752, "params": 30.0, "active": 3.0}),
    (r"^glm-4\.7", {"fam": "GLM", "gen": 4.7, "rel": "2025-12", "ctx": 202752, "params": 355.0, "active": 32.0}),
    (r"^glm-5(\.[12])?$", {"fam": "GLM", "gen": 5, "rel": "2026-02", "ctx": 202752, "params": 744.0, "active": 40.0}),
    (r"^minimax-m2", {"fam": "MiniMax", "gen": 2, "rel": "2025-10", "ctx": 196608, "params": 230.0, "active": 10.0}),
    (r"^phi-4-reasoning", {"fam": "Phi", "gen": 4, "rel": "2025-04", "ctx": 32768, "params": 14.7}),
    (r"^phi-4$", {"fam": "Phi", "gen": 4, "rel": "2024-12", "ctx": 16384, "params": 14.7}),
    (r"^phi-3\.5-moe", {"fam": "Phi", "gen": 3.5, "rel": "2024-08", "ctx": 131072, "params": 41.9, "active": 6.6,
                        "legacy": True}),
    (r"^phi-3\.5-mini", {"fam": "Phi", "gen": 3.5, "rel": "2024-08", "ctx": 131072, "params": 3.8, "legacy": True}),
    (r"^phi-3-mini-128k", {"fam": "Phi", "gen": 3, "rel": "2024-04", "ctx": 131072, "params": 3.8, "legacy": True}),
    (r"^phi-3-mini", {"fam": "Phi", "gen": 3, "rel": "2024-04", "ctx": 4096, "params": 3.8, "legacy": True}),
    (r"^phi-3-small-128k", {"fam": "Phi", "gen": 3, "rel": "2024-05", "ctx": 131072, "params": 7.4, "legacy": True}),
    (r"^phi-3-small", {"fam": "Phi", "gen": 3, "rel": "2024-05", "ctx": 8192, "params": 7.4, "legacy": True}),
    (r"^phi-3-medium-128k", {"fam": "Phi", "gen": 3, "rel": "2024-05", "ctx": 131072, "params": 14.0, "legacy": True}),
    (r"^phi-3-medium", {"fam": "Phi", "gen": 3, "rel": "2024-05", "ctx": 4096, "params": 14.0, "legacy": True}),
    # gpt-oss is published in MXFP4 only (HF config quantization_config quant_method "mxfp4", verified 2026-11-09;
    # Model Garden's recipes serve that checkpoint with no --quantization flag). Live metadata replaces this.
    (r"^gpt-oss-120b", {"fam": "gpt-oss", "gen": 1, "rel": "2025-08", "ctx": 131072, "params": 117.0, "active": 5.1,
                        "native": "mxfp4"}),
    (r"^gpt-oss-20b", {"fam": "gpt-oss", "gen": 1, "rel": "2025-08", "ctx": 131072, "params": 21.0, "active": 3.6,
                       "native": "mxfp4"}),
    (r"^nemotron-3-nano", {"fam": "Nemotron", "gen": 3, "rel": "2025-12", "ctx": 1048576}),
    (r"^nemotron-3-(super|ultra)", {"fam": "Nemotron", "gen": 3, "rel": None, "ctx": 1048576}),
    (r"^falcon", {"fam": "Falcon", "gen": 1, "rel": "2023-05", "ctx": 2048, "legacy": True}),
    (r"^vicuna-.*-16k", {"fam": "Vicuna", "gen": 1.5, "rel": "2023-08", "ctx": 16384, "legacy": True}),
    (r"^vicuna", {"fam": "Vicuna", "gen": 1.5, "rel": "2023-08", "ctx": 4096, "legacy": True}),
    (r"^vibethinker", {"fam": "VibeThinker", "gen": 1, "rel": None, "ctx": 40960, "spec": "math"}),
    (r"^muse-", {"fam": "Muse", "gen": None, "rel": None, "ctx": 131072}),
    (r"^gemma-3-1b", {"fam": "Gemma", "gen": 3, "rel": "2025-03", "ctx": 32768}),
]
_FAMILY_RE = [(re.compile(p, re.I), f) for p, f in _FAMILY_RULES]

# Publisher (Model Garden) -> family when no rule matches; HF org for inferred hf_id.
_PUBLISHER_FAMILY = {"google": "Gemma", "meta": "Llama", "qwen": "Qwen", "mistral-ai": "Mistral AI",
                     "mistralai": "Mistral AI", "deepseek-ai": "DeepSeek", "openai": "gpt-oss", "nvidia": "Nemotron",
                     "tiiuae": "Falcon", "lmsys": "Vicuna", "moonshotai": "Kimi", "zai-org": "GLM",
                     "minimaxai": "MiniMax", "microsoft": "Phi"}
_HF_ORG = {"meta": "meta-llama", "qwen": "Qwen", "mistral-ai": "mistralai", "weiboai": "WeiboAI",
           "minimaxai": "MiniMaxAI"}

_WORDS = {"gemma": "Gemma", "medgemma": "MedGemma", "translategemma": "TranslateGemma", "txgemma": "TxGemma",
          "codegemma": "CodeGemma", "codellama": "Code Llama", "llama": "Llama", "qwq": "QwQ", "mistral": "Mistral",
          "mixtral": "Mixtral", "ministral": "Ministral", "deepseek": "DeepSeek", "nemotron": "Nemotron",
          "falcon": "Falcon", "vicuna": "Vicuna", "vibethinker": "VibeThinker", "sea": "SEA", "lion": "LION",
          "oss": "oss", "gpt": "gpt", "next": "Next", "coder": "Coder", "guard": "Guard", "text": "Text",
          "glm": "GLM", "minimax": "MiniMax", "kimi": "Kimi", "phi": "Phi", "moe": "MoE", "speciale": "Speciale"}


def _split_id(raw: str) -> tuple[str, str]:
    """(publisher, version key) of a Model Garden id `publisher/model@version` or an HF repo id."""
    pub = raw.split("/")[0].lower() if "/" in raw else ""
    key = raw.split("@")[-1] if "@" in raw else raw.split("/")[-1]
    return pub, key.lower()


def display_name(key: str) -> str:
    """Readable model name from a version key: gemma-4-26b-a4b-it -> "Gemma 4 26B-A4B"."""
    k = _NOISE_SUFFIX.sub("", key.lower())
    k = re.sub(r"^meta-llama-", "llama-", k)
    k = re.sub(r"^llama-3-2-", "llama-3.2-", k)
    k = re.sub(r"^deepseek-v(\d)-(\d)(?=-|$)", r"deepseek-v\1.\2", k)
    toks = [t for t in k.split("-") if t and t not in _INSTRUCT_TOKENS]
    out: list[str] = []
    for i, t in enumerate(toks):
        if re.fullmatch(r"\d+(\.\d+)?b", t) or re.fullmatch(r"\d+x\d+b", t):
            out.append(t[:-1] + "B")
        elif re.fullmatch(r"[ae]\d+(\.\d+)?b", t):
            if t[0] == "a" and out and out[-1].endswith("B"):
                out[-1] += "-" + t.upper()
            else:
                out.append(t.upper())
        elif re.fullmatch(r"\d+e", t) and out and out[-1].endswith("B"):
            out[-1] += "-" + t.upper()
        elif re.fullmatch(r"r1|v\d.*", t) and out and out[-1] == "DeepSeek":
            out[-1] = "DeepSeek-" + t.upper()
        elif t == "oss" and out and out[-1] == "gpt":
            out[-1] = "gpt-oss"
        elif re.fullmatch(r"qwen\d.*", t):
            out.append("Qwen" + t[4:])
        elif re.fullmatch(r"v\d.*", t) or re.fullmatch(r"\d.*", t):
            out.append(t)
        else:
            out.append(_WORDS.get(t, t.capitalize()))
    return " ".join(out).replace("Gemma SEA LION", "Gemma-SEA-LION").replace("GLM ", "GLM-")


def _rule(key: str) -> dict:
    for rx, facts in _FAMILY_RE:
        if rx.search(key):
            return facts
    return {}


def _sizes(key: str, facts: dict) -> tuple[float | None, float | None]:
    if facts.get("params"):
        return float(facts["params"]), float(facts.get("active") or facts["params"])
    k = re.sub(r"-a\d+(?:\.\d+)?b\b", "", key)
    m = re.search(r"(?:^|-)(\d+(?:\.\d+)?)b\b", k)
    if not m:
        return None, None
    params = float(m.group(1))
    act = re.search(r"-a(\d+(?:\.\d+)?)b\b", key)
    return params, float(act.group(1)) if act else params


def _thinking(key: str) -> bool:
    if re.search(r"instruct-2507", key):
        return False
    return bool(re.search(r"(r1|qwq|thinking|reasoning|gemma-4|qwen3|gpt-oss|nemotron-3|vibethinker)", key))


def _group_key(key: str) -> str:
    """Instruct and base checkpoints of one model share this key (llama-2-13b-chat ~ llama-2-13b)."""
    k = _NOISE_SUFFIX.sub("", key)
    return "-".join(t for t in k.split("-") if t not in _INSTRUCT_TOKENS)


def _is_instruct(key: str) -> bool:
    return bool(set(_NOISE_SUFFIX.sub("", key).split("-")) & _INSTRUCT_TOKENS)


SOURCE_LABELS = {"hf": "Hugging Face", "model_garden": "Model Garden", "rules": "built-in estimate"}
_PROVENANCE_FIELDS = {"params_b": "size", "active_params_b": "active parameters", "max_context": "context",
                      "released": "release date"}


def _iso_day(ts) -> str | None:
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(int(ts)))
    except (TypeError, ValueError):
        return None


def annotate(m: dict) -> dict:
    """Fill the curated fields (family, generation, released, legacy, specialty, precisions…) on a model entry.

    Live Hugging Face metadata (hfmeta.cached, never fetched here) wins for size, active parameters (MoE formula),
    context, native precision, release month, license; the rule table (_FAMILY_RULES, the seed catalog and sizes in
    the model name) fills what the Hub can't provide. `field_sources` records "hf" / "model_garden" / "rules" per
    field and `data_source` summarises it for the UI ("Data: Hugging Face, updated <date>" / built-in estimate)."""
    pub, key = _split_id(m.get("model_garden_id") or m.get("hf_id") or m.get("id") or "")
    key = (m.get("id") or key).lower()
    facts = _rule(key)
    p, a = _sizes(key, facts)
    meta = hfmeta.cached(m.get("hf_id"))
    hf = (meta or {}).get("fields") or {}
    src: dict[str, str] = dict(m.get("field_sources") or {})
    m.setdefault("id", key)
    m.setdefault("name", display_name(key))
    if meta and meta.get("hf_id") and m.get("hf_id") and meta["hf_id"].lower() == str(m["hf_id"]).lower():
        m["hf_id"] = meta["hf_id"]                              # canonical casing

    fam = facts.get("fam") or (m.get("family") if m.get("_seed") else None)
    if fam:
        m["family"], src["family"] = fam, "rules"
    else:
        author = str(m.get("hf_id") or "").split("/")[0].lower()
        if meta and _PUBLISHER_FAMILY.get(author):
            m["family"], src["family"] = _PUBLISHER_FAMILY[author], "hf"
        else:
            m["family"] = _PUBLISHER_FAMILY.get(pub) or (pub.replace("-", " ").title() if pub else "Other")
            src["family"] = "model_garden"

    if hf.get("params_b"):
        m["params_b"], src["params_b"] = hf["params_b"], "hf"
    elif m.get("params_b") or p:
        m["params_b"] = m.get("params_b") or p
        src.setdefault("params_b", "rules")
    published = facts.get("active") or (a if a and p and a != p else None)   # "-a3b" in the name / rule table
    if hf.get("active_params_b") and published and abs(hf["active_params_b"] - published) / published > 0.15:
        # The formula disagrees with the published active size by more than 15%: trust the published figure.
        log.info("active params for %s: Hugging Face config gives %.1fB, published %.1fB; using the published",
                 key, hf["active_params_b"], published)
        m["active_params_b"], src["active_params_b"] = float(published), "rules"
    elif hf.get("active_params_b"):
        m["active_params_b"], src["active_params_b"] = hf["active_params_b"], "hf"
    elif hf.get("dense") and hf.get("params_b") and not published:
        m["active_params_b"], src["active_params_b"] = hf["params_b"], "hf"
    else:
        moe = hf.get("moe") or {}
        if published:
            m["active_params_b"] = float(published)
        elif src.get("params_b") == "hf" and moe.get("experts"):
            # MoE whose config the formula doesn't understand: routed share of the weights + 10% always active
            share = min(1.0, float(moe.get("experts_per_token") or 1) / float(moe["experts"]) + 0.1)
            m["active_params_b"] = round(float(m["params_b"]) * share, 1)
        elif src.get("params_b") == "hf":
            m["active_params_b"] = m["params_b"]                 # dense by its name: active = the Hub's total
        elif not m.get("active_params_b"):
            m["active_params_b"] = a or m.get("params_b")
        src["active_params_b"] = "rules"
    if hf.get("max_context"):
        m["max_context"], src["max_context"] = int(hf["max_context"]), "hf"
    else:
        m.setdefault("max_context", facts.get("ctx") or 131072)
        src.setdefault("max_context", "rules")
    if hf.get("released"):
        m["released"], src["released"] = hf["released"], "hf"
    else:
        m.setdefault("released", facts.get("rel"))
        src.setdefault("released", "rules")
    for k in ("license", "gated", "checkpoint_precision", "model_type"):
        if hf.get(k) is not None:
            m[k] = hf[k]
            src[k] = "hf"
    m.setdefault("generation", facts.get("gen"))
    m.setdefault("legacy", bool(facts.get("legacy")))
    m.setdefault("specialty", facts.get("spec"))
    m.setdefault("supports_thinking", _thinking(key))
    m.setdefault("supports_json", True)
    m.setdefault("tasks", list(SPECIALTY_TASKS.get(m["specialty"], GENERAL_TASKS)))
    params = float(m.get("params_b") or 0)
    m.setdefault("quality_tier", "frontier" if params >= 30 else ("balanced" if params >= 14 else "compact"))
    pq = dict(m.get("prequantized") or {})
    m["prequantized"] = pq
    if meta and hf.get("checkpoint_precision"):
        native = hf.get("native_precision")                     # the published checkpoint decides (None = BF16 etc.)
        src["native_precision"] = "hf"
        if not native and m.get("native_precision") and not m.get("fp8_only"):
            m.pop("native_precision", None)
    else:
        native = m.get("native_precision") or facts.get("native")
        if native:
            src["native_precision"] = "rules"
    if native:
        m["native_precision"] = native
    if native == "fp8":
        m["fp8_only"] = True
    # INT4 only with a real 4-bit checkpoint (published in INT4, or pre-quantized AWQ / GPTQ / W4A16 in Model Garden);
    # MXFP4 only for models published in MXFP4 (gpt-oss: then it is the only option unless Model Garden also lists a
    # BF16 checkpoint) or with a pre-quantized `-mxfp4` checkpoint.
    int4 = native == "int4" or any(pq.get(k) for k in INT4_KEYS)
    mx = native == "mxfp4" or any(pq.get(k) for k in MXFP4_KEYS)
    m["precisions"] = (["int4"] if native == "int4" else
                       ((["bf16"] if pq.get("bf16") else []) + ["mxfp4"]) if native == "mxfp4" else
                       (["fp8"] if m.get("fp8_only") else ["bf16", "fp8"]) + (["int4"] if int4 else [])
                       + (["mxfp4"] if mx else [])
                       + [k for k in pq if k not in ("bf16", "fp8") and k not in INT4_KEYS and k not in MXFP4_KEYS])
    if mx:
        if hf.get("mxfp4_weight_gb") and native == "mxfp4":
            m["mxfp4_weight_gb"], m["mxfp4_weight_note"] = hf["mxfp4_weight_gb"], hf.get("mxfp4_weight_note")
            src["mxfp4_weight_gb"] = "hf"
        else:
            m.pop("mxfp4_weight_gb", None)
            src["mxfp4_weight_gb"] = "rules"
    src["precisions"] = ("hf" if src.get("native_precision") == "hf" and native else
                         "model_garden" if pq or m.get("fp8_only") else src.get("native_precision", "rules"))
    m.setdefault("verified_in_model_garden", False)
    est = [f for f in _PROVENANCE_FIELDS if src.get(f) == "rules" and m.get(f) is not None]
    day = _iso_day((meta or {}).get("updated_at"))
    if meta:
        label = f"Hugging Face, updated {day}"
        if est:
            label += "; built-in estimate for " + ", ".join(_PROVENANCE_FIELDS[f] for f in est)
    else:
        label = "built-in estimate"
    m["field_sources"] = src
    m["data_source"] = {"kind": ("hf" if meta and not est else "mixed" if meta else "rules"), "label": label,
                        "hf_id": (meta or {}).get("hf_id"), "updated_at": day, "estimated_fields": est,
                        "notes": list((meta or {}).get("notes") or [])}
    if not m.get("summary"):
        act = float(m.get("active_params_b") or params)
        size = (f"MoE, {params:g}B total / {act:g}B active" if params and act < params * 0.6
                else f"{params:g}B dense" if params else "size unknown")
        bits = [f"{m['name']} ({size}, {int(m['max_context']) // 1024}K context)."]
        if m["specialty"]:
            bits.append(f"Specialised for {m['specialty']}.")
        if m["legacy"]:
            bits.append("Older generation: kept for comparison, superseded by newer models.")
        if native:
            bits.append(f"Published in {native.upper()} (no BF16 checkpoint).")
        if pq:
            bits.append("Pre-quantized checkpoints in Model Garden: " + ", ".join(k.upper() for k in pq) + ".")
        m["summary"] = " ".join(bits)
    m.pop("_seed", None)
    return m


def _infer_hf(raw: str, key: str) -> str:
    pub, _ = _split_id(raw)
    return raw if "@" not in raw else f"{_HF_ORG.get(pub, pub)}/{key}"


def curate_models(garden_ids: list[str], seeds: list[dict] | None = None,
                  source: str = "vertex_model_garden", hf_map: dict[str, str] | None = None) -> list[dict]:
    """Curate raw Model Garden (or HF) ids into deployable text LLMs merged with the seed catalog.

    Drops non-text models and base checkpoints that have an instruct sibling, folds pre-quantized checkpoints
    (`-fp8`, `-awq`, `-gptq`, `-w8a8`, `-nvfp4`) into their base model's `prequantized` map, and annotates every
    entry (family, generation, released, legacy, specialty, precisions). `hf_map` ({raw id: Hugging Face id}, from
    the Model Garden listing's deploy recipes) gives each entry its exact Hugging Face id; otherwise it is inferred
    from the publisher and version name. Models with no size in their name or rules are kept when their cached
    Hugging Face metadata has one.
    """
    hf_map = hf_map or {}
    seeds = copy.deepcopy(SEED_MODELS if seeds is None else seeds)
    by_key: dict[str, dict] = {}
    for s in seeds:
        s["_seed"] = True
        s.setdefault("verified_in_model_garden", False)
        by_key[s["id"].lower()] = s

    plain: dict[str, str] = {}             # version key -> raw id
    quant: list[tuple[str, str, str]] = []  # (base key, scheme, raw id)
    unsized: list[str] = []
    for raw in dict.fromkeys(garden_ids or []):
        if not raw or _NON_TEXT.search(raw):
            continue
        _, key = _split_id(raw)
        if not re.search(r"[a-z]", key):
            continue                       # bare version tags ("001", "latest"): no model name to curate
        q = _QUANT.search(key)
        stem = key[:q.start()] if q else re.sub(r"-bf16$", "", key)
        task = ((hfmeta.cached(hf_map.get(raw) or _infer_hf(raw, stem)) or {}).get("fields") or {}).get("pipeline_tag")
        if task in NON_TEXT_PIPELINES:
            continue                       # the Hub says it isn't a text generator (embeddings, ASR, diffusion…)
        if _sizes(stem, _rule(stem))[0] is None and not (hfmeta.cached(hf_map.get(raw) or _infer_hf(raw, stem))
                                                          or {}).get("fields", {}).get("params_b"):
            unsized.append(raw)            # no size in the name and no known size: can't plan hardware for it
            continue
        if q:
            scheme = re.sub(r"^quantized[.\-_]", "", q.group(1).lower())
            quant.append((stem, _QUANT_KEY.get(scheme, scheme), raw))
            continue
        if _BASE.search(stem):
            continue
        plain.setdefault(re.sub(r"-gg-hf$", "", stem), raw)

    def base_of(stem: str) -> dict | str | None:
        for k in (stem, stem + "-it", stem + "-instruct"):
            if k in by_key:
                return by_key[k]
            if k in plain:
                return k
        return None

    fp8_only: dict[str, str] = {}          # Model Garden only has the FP8 checkpoint (e.g. Llama 3.1 405B, GLM-5)
    for stem, scheme, raw in quant:
        if scheme == "fp8" and base_of(stem) is None and not _BASE.search(stem):
            fp8_only.setdefault(stem, raw)
            plain.setdefault(stem, raw)

    instruct_groups = {_group_key(k) for k in list(plain) + list(by_key) if _is_instruct(k)}
    for key, raw in plain.items():
        if not _is_instruct(key) and _group_key(key) in instruct_groups:
            continue                       # base checkpoint of a model with an instruct sibling
        if key in by_key:
            e = by_key[key]
            if source == "vertex_model_garden":
                if e.get("model_garden_id") and e["model_garden_id"] != raw:
                    e.setdefault("aliases", []).append(e["model_garden_id"])
                e["model_garden_id"] = raw
                e["verified_in_model_garden"] = True
                if hf_map.get(raw):
                    e["hf_id"] = hf_map[raw]
            continue
        hf = hf_map.get(raw) or _infer_hf(raw, key)
        by_key[key] = {"id": key, "model_garden_id": raw, "hf_id": hf,
                       "verified_in_model_garden": source == "vertex_model_garden", "dynamic_source": source}
        if key in fp8_only:
            by_key[key].update(fp8_only=True, prequantized={"fp8": raw})

    for stem, scheme, raw in quant:
        e = base_of(stem)
        if isinstance(e, dict) and raw != e.get("model_garden_id"):
            e.setdefault("prequantized", {}).setdefault(scheme, raw)
    _LAST_UNSIZED[:] = unsized

    out = [annotate(m) for m in by_key.values()]
    out = [m for m in out if m.get("params_b")]
    out.sort(key=lambda m: (m["legacy"], m["specialty"] is not None, m["family"],
                            -float(m.get("generation") or 0), float(m.get("params_b") or 0), m["id"]))
    return out


_LAST_UNSIZED: list[str] = []


def family_counts(models: list[dict]) -> list[dict]:
    """[{name, count, legacy}] per family, most models first (legacy = how many of them are older generations)."""
    agg: dict[str, dict] = {}
    for m in models:
        f = agg.setdefault(m.get("family") or "Other", {"name": m.get("family") or "Other", "count": 0, "legacy": 0})
        f["count"] += 1
        f["legacy"] += int(bool(m.get("legacy")))
    return sorted(agg.values(), key=lambda f: (-f["count"], f["name"]))


def _cached_garden_ids() -> list[str]:
    """Model Garden ids from the last listing (curated cache, else the format-1 cache), for offline re-curation."""
    for name in (CURATED_CACHE, LEGACY_CACHE):
        try:
            d = json.loads((CACHE / name).read_text())
        except Exception:
            continue
        if d.get("garden_ids"):
            return list(d["garden_ids"])
        if name == LEGACY_CACHE:
            return [m["model_garden_id"] for m in d.get("models", [])
                    if m.get("model_garden_id") and (m.get("verified_in_model_garden")
                                                     or m.get("dynamic_source") == "vertex_model_garden")]
    return []


_MEM: dict[str, tuple[float, dict]] = {}


def _build(garden_ids: list[str], hf_ids: list[str], sources: list[str], updated_at: int,
           garden_hf: dict[str, str] | None = None) -> dict:
    """Curated payload from raw listings (cheap: re-run on every cache load so rule changes apply at once)."""
    garden_hf = dict(garden_hf or {})
    models = curate_models(garden_ids, hf_map=garden_hf)
    unsized = list(_LAST_UNSIZED)
    if hf_ids:
        known = {m["id"] for m in models} | {str(m.get("hf_id", "")).lower() for m in models}
        extra = [m for m in curate_models(hf_ids, seeds=[], source="huggingface_hub")
                 if m["id"] not in known and str(m.get("hf_id", "")).lower() not in known]
        for m in extra:
            m["summary"] += " Discovered on Hugging Face Hub (not in Model Garden)."
        models += extra
    with_hf = sum(1 for m in models if (m.get("data_source") or {}).get("kind") in ("hf", "mixed"))
    return {"updated_at": updated_at, "format": CURATED_FORMAT, "sources": sources, "models": models,
            "families": family_counts(models), "garden_ids": list(garden_ids), "hf_ids": list(hf_ids),
            "garden_hf": garden_hf, "unsized_ids": unsized,
            "metadata": {"models": len(models), "hugging_face": with_hf, "built_in": len(models) - with_hf,
                         "refreshing": hfmeta.refreshing(), "last_refresh": hfmeta._BG.get("last")}}


def hf_targets(payload: dict) -> list[str]:
    """Hugging Face ids to keep fresh: every curated model plus unsized Model Garden ids (sizes may come from HF)."""
    ids = [m.get("hf_id") for m in payload.get("models") or []]
    gh = payload.get("garden_hf") or {}
    for raw in payload.get("unsized_ids") or []:
        _, key = _split_id(raw)
        q = _QUANT.search(key)
        ids.append(gh.get(raw) or _infer_hf(raw, key[:q.start()] if q else key))
    return [i for i in dict.fromkeys(ids) if hfmeta.valid_id(i)]


_HF_KICK = {"at": 0.0}


def _kick_hf_refresh(payload: dict, force: bool = False) -> None:
    """Start a background Hugging Face refresh of stale entries (at most every 10 minutes unless forced)."""
    if hfmeta.disabled() or (not force and time.time() - _HF_KICK["at"] < 600):
        return
    _HF_KICK["at"] = time.time()
    try:
        hfmeta.refresh_bg(hf_targets(payload))
    except Exception as e:
        print(f"  (Hugging Face metadata refresh not started: {str(e)[:120]})")


def refresh_after_token_change() -> dict:
    """A Hugging Face token was saved or removed: drop cached 401 / 403 metadata and refetch it in the background
    (hfmeta.ON_UPDATE then rebuilds the server's catalogs). Returns {invalidated, refreshing}."""
    n = hfmeta.invalidate_auth_failures()
    if hfmeta.disabled():
        return {"invalidated": n, "refreshing": False}

    def work():
        try:
            hfmeta.refresh_bg(hf_targets(discover_models()))
            _HF_KICK["at"] = time.time()
        except Exception as e:
            print(f"  (Hugging Face metadata refresh not started: {str(e)[:120]})")
    threading.Thread(target=work, daemon=True, name="hf-token-refresh").start()
    return {"invalidated": n, "refreshing": True}


def _hf_sig() -> tuple:
    """Changes when new Hugging Face metadata lands (this process or another one writing the cache)."""
    try:
        mt = (paths.cache_dir() / "hf_meta").stat().st_mtime
    except OSError:
        mt = 0.0
    return hfmeta.version(), mt, hfmeta.disabled()


def _cached_garden_hf() -> dict[str, str]:
    try:
        return dict(json.loads((CACHE / CURATED_CACHE).read_text()).get("garden_hf") or {})
    except Exception:
        return {}


def discover_models(project: str | None = None, region: str = "us-central1", refresh: bool = False) -> dict:
    """Curated catalog of deployable open text LLMs: seeds + Model Garden (+ HF on refresh), cached 24h.

    The cache stores the raw listings (`garden_ids`, `garden_hf`, `hf_ids`); curation runs on load, merging the
    per-model Hugging Face metadata cache (hfmeta), which is refreshed in the background (7-day TTL). Returns
    {updated_at, format, sources, models, families: [{name, count, legacy}], garden_ids, garden_hf, hf_ids,
    unsized_ids, metadata: {models, hugging_face, built_in, refreshing}}.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / CURATED_CACHE
    sig = _hf_sig()
    if not refresh:
        try:
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime < MODEL_CACHE_TTL:
                hit = _MEM.get(str(cache_path))
                if not (hit and hit[0] == (mtime, sig)):
                    d = json.loads(cache_path.read_text())
                    if d.get("format") != CURATED_FORMAT:
                        raise ValueError("old cache format")
                    hit = ((mtime, sig), _build(d.get("garden_ids") or [], d.get("hf_ids") or [],
                                                d.get("sources") or [], d.get("updated_at") or int(mtime),
                                                d.get("garden_hf") or {}))
                    _MEM[str(cache_path)] = hit
                    _kick_hf_refresh(hit[1])
                return copy.deepcopy(hit[1])
        except Exception:
            pass

    offline = _MEM.get("offline")
    if not refresh and not project and offline and time.time() - offline[0][0] < 600 and offline[0][1] == sig:
        return copy.deepcopy(offline[1])
    sources = ["seed_catalog"]
    garden_ids, garden_hf = _list_model_garden(project, region)
    if garden_ids:
        sources.append("vertex_model_garden")
    else:
        garden_ids, garden_hf = _cached_garden_ids(), _cached_garden_hf()
        if garden_ids:
            sources.append("vertex_model_garden_cached")
    hf_ids = _search_hf_open_models(limit=8) if refresh else []
    if hf_ids:
        sources.append("huggingface_hub")
    payload = _build(garden_ids, hf_ids, sources, int(time.time()), garden_hf)
    if "vertex_model_garden" in sources or hf_ids:   # a live listing: cache it for 24h
        try:
            cache_path.write_text(json.dumps({k: payload[k] for k in ("updated_at", "format", "sources", "garden_ids",
                                                                      "garden_hf", "hf_ids")}, indent=1))
            _MEM.pop(str(cache_path), None)
        except Exception:
            pass
    else:   # seeds (+ the last listing): recomputed cheaply, never pinned to disk so a later live listing wins
        _MEM["offline"] = ((time.time(), sig), payload)
    _kick_hf_refresh(payload, force=refresh)
    return copy.deepcopy(payload)


def find_model(model: dict, models: list[dict] | None = None) -> dict | None:
    """Catalog entry for a config `model` block (matched on model_garden_id, aliases or hf_id), or None."""
    ids = {str(x).lower() for x in (model.get("model_garden_id"), model.get("hf_id")) if x}
    if not ids:
        return None
    if models is None:
        try:
            models = discover_models(None).get("models", [])
        except Exception:
            models = []
        models = models or curate_models([])
    for m in models:
        cand = {str(m.get("model_garden_id", "")).lower(), str(m.get("hf_id", "")).lower(),
                *(str(a).lower() for a in m.get("aliases") or [])}
        if cand & ids:
            return m
    return None
