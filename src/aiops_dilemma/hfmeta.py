"""Model metadata from the Hugging Face Hub (live source for the curated model catalog).

For every curated model with a Hugging Face id this module reads the Hub's model API
(`https://huggingface.co/api/models/{id}`: safetensors parameter counts per dtype, createdAt / lastModified, tags,
cardData incl. license, gated) and the model's `config.json` (nested `text_config` for multimodal wrappers), and
derives: total parameters, **active parameters for mixture-of-experts models** (one formula, `moe_active_params`),
context window, the checkpoint's native precision (`quantization_config` / `torch_dtype`), release month and license.

Metadata of most models, gated ones included, needs no token; `config.json` of gated repos (Llama, some Gemma /
Mistral) does: add a token in the app's Settings or set `HF_TOKEN` (hftoken: env > saved file), otherwise those
fields fall back to the built-in rules in modelinfo. After a token is saved or removed, `invalidate_auth_failures()`
drops the cached 401 / 403 records so the next refresh refetches them.

Cached per model under `<cache>/hf_meta/` for `TTL_S` (7 days; failures for `FAIL_TTL_S`). The cache stores the
trimmed raw API + config; parsing runs on load, so parser changes apply without refetching. Nothing here blocks the
UI: `cached()` never touches the network and `refresh_bg()` fetches stale entries in a background thread.
`DILEMMA_NO_LIVE_METADATA=1` (tests) disables both reading the cache and fetching.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from aiops_dilemma import hftoken, paths

log = logging.getLogger(__name__)

API_URL = "https://huggingface.co/api/models/{id}"
FILE_URL = "https://huggingface.co/{id}/resolve/main/{file}"
TTL_S = 7 * 86400          # refetch after a week
FAIL_TTL_S = 86400         # retry failures (network, rate limit) after a day; 404s count as failures too
FORMAT = 1                 # raw record format on disk
SOURCE_HF = "hf"
_ID_RX = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def disabled() -> bool:
    return os.environ.get("DILEMMA_NO_LIVE_METADATA", "").strip() not in ("", "0", "false")


def _dir():
    return paths.cache_dir() / "hf_meta"


def valid_id(hf_id: str | None) -> bool:
    return bool(hf_id and _ID_RX.match(hf_id) and not hf_id.startswith("gs:"))


def _path(hf_id: str):
    return _dir() / (hf_id.lower().replace("/", "__") + ".json")


# ------------------------------------------------------------------------------------------------------- fetching
def _get(url: str, timeout: float = 15.0) -> tuple[int, dict | None]:
    """(HTTP status, JSON body or None). Network errors return (0, None)."""
    headers = hftoken.headers()     # Bearer token when one is set (env HF_TOKEN / HUGGING_FACE_HUB_TOKEN, Settings)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return 0, None


_API_KEEP = ("id", "createdAt", "lastModified", "gated", "safetensors", "tags", "pipeline_tag", "library_name",
             "author", "private", "disabled")
_CARD_KEEP = ("license", "license_name", "base_model", "pipeline_tag")


def _trim(obj, depth: int = 0):
    """Drop what parsing never needs (chat templates, long token lists) so cache files stay small."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("chat_template", "chat_template_jinja", "id2label", "label2id", "tokenizer_config",
                     "suppress_tokens", "begin_suppress_tokens", "bad_words_ids"):
                continue
            if isinstance(v, str) and len(v) > 300:
                continue
            if isinstance(v, list) and len(v) > 400:
                v = {"_len": len(v)}
            out[k] = _trim(v, depth + 1) if depth < 6 else v
        return out
    return obj


def fetch(hf_id: str) -> dict:
    """Fetch and trim one model's Hub metadata + config.json (network). Returns the raw cache record."""
    rec = {"format": FORMAT, "hf_id": hf_id, "fetched_at": int(time.time()), "status": "error", "api": None,
           "config": None, "config_status": None}
    code, api = _get(API_URL.format(id=hf_id))
    if code == 200 and isinstance(api, dict):
        card = api.get("cardData") or {}
        rec["api"] = {**{k: api.get(k) for k in _API_KEEP if k in api},
                      "cardData": {k: card.get(k) for k in _CARD_KEEP if k in card},
                      "config": _trim(api.get("config") or {})}
        rec["status"] = "ok"
        rec["hf_id"] = api.get("id") or hf_id          # canonical casing (the Hub redirects case variants)
        ccode, cfg = _get(FILE_URL.format(id=rec["hf_id"], file="config.json"))
        rec["config_status"] = ccode
        if ccode == 200 and isinstance(cfg, dict):
            rec["config"] = _trim(cfg)
    else:
        rec["status"] = "not_found" if code in (401, 404) else "error"
        rec["http_status"] = code
    return rec


def _load_raw(hf_id: str) -> dict | None:
    try:
        return json.loads(_path(hf_id).read_text())
    except Exception:
        return None


def _save_raw(hf_id: str, rec: dict) -> None:
    try:
        _dir().mkdir(parents=True, exist_ok=True)
        tmp = _path(hf_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, separators=(",", ":")))
        tmp.replace(_path(hf_id))
    except Exception as e:   # a read-only cache dir only costs a refetch later
        log.debug("hf_meta: can't write cache for %s: %s", hf_id, e)


def is_stale(hf_id: str, now: float | None = None) -> bool:
    rec = _load_raw(hf_id)
    if not rec or rec.get("format") != FORMAT:
        return True
    age = (now or time.time()) - float(rec.get("fetched_at") or 0)
    return age > (TTL_S if rec.get("status") == "ok" else FAIL_TTL_S)


# ------------------------------------------------------------------------------------------------------- parsing
def _text_config(cfg: dict) -> dict:
    """The language model's config: `text_config` / `llm_config` of a multimodal wrapper, else the config itself."""
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(k), dict) and cfg[k].get("hidden_size"):
            return {**{kk: vv for kk, vv in cfg.items() if not isinstance(vv, dict)}, **cfg[k]}
    return cfg


def _first(d: dict, *keys, default=None):
    """First present value. Per-layer lists (e.g. Gemma 3n's intermediate_size) collapse to their largest number."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, (list, tuple)):
            nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
            v = max(nums) if nums else None
        if v not in (None, "", 0):
            return v
    return default


def detect_precision(cfg: dict | None, api_cfg: dict | None = None, st_params: dict | None = None) -> str | None:
    """Native checkpoint precision: int4 / int8 / fp8 / nvfp4 / mxfp4 / bf16 (None when unknown).

    From `quantization_config` (compressed-tensors num_bits 4 int -> int4, 8 float -> fp8, nvfp4 format -> nvfp4;
    awq / gptq -> int4 (or their bits); fp8 / fbgemm_fp8 -> fp8; modelopt FP4 -> nvfp4, FP8 -> fp8; mxfp4), else
    `torch_dtype` / `dtype` (bfloat16 / float16 / float32 -> bf16), else the safetensors dtype that dominates."""
    cfg, api_cfg = cfg or {}, api_cfg or {}
    tc = _text_config(cfg)
    qc = cfg.get("quantization_config") or tc.get("quantization_config") or api_cfg.get("quantization_config")
    if isinstance(qc, dict) and qc:
        m = str(qc.get("quant_method") or "").lower().replace("_", "-")
        fmt = str(qc.get("format") or "").lower()
        if m == "compressed-tensors":
            if "nvfp4" in fmt or "mxfp4" in fmt:
                return "nvfp4" if "nvfp4" in fmt else "mxfp4"
            groups = [g.get("weights") or {} for g in (qc.get("config_groups") or {}).values() if isinstance(g, dict)]
            bits = [int(w.get("num_bits") or 16) for w in groups if w]
            kinds = {str(w.get("type") or "int").lower() for w in groups if w}
            if bits:
                b = min(bits)
                if b <= 4:
                    return "nvfp4" if "float" in kinds else "int4"
                if b == 8:
                    return "fp8" if "float" in kinds or "float-quantized" in fmt else "int8"
            if "float-quantized" in fmt:
                return "fp8"
            if "pack-quantized" in fmt:
                return "int4"
            return None
        if m in ("awq", "gptq", "auto-round", "autoround"):
            b = int(qc.get("bits") or qc.get("w_bit") or 4)
            return "int4" if b <= 4 else "int8"
        if m in ("fp8", "fbgemm-fp8"):
            return "fp8"
        if m == "modelopt":
            algo = str(qc.get("quant_algo") or (qc.get("quantization") or {}).get("quant_algo") or "").upper()
            if "FP4" in algo:
                return "nvfp4"
            if "FP8" in algo:
                return "fp8"
            return None
        if m == "mxfp4":
            return "mxfp4"
        if m == "bitsandbytes":
            return "int4" if qc.get("load_in_4bit") else "int8"
        return None
    dtype = str(_first(cfg, "torch_dtype", "dtype") or _first(tc, "torch_dtype", "dtype") or "").lower()
    if dtype in ("bfloat16", "float16", "float32", "bf16", "fp16"):
        return "bf16"
    if st_params:
        top = max(st_params.items(), key=lambda kv: kv[1])[0].upper()
        if top.startswith("F8"):
            return "fp8"
        if top in ("BF16", "F16", "F32"):
            return "bf16"
    return None


def context_length(cfg: dict | None, api_cfg: dict | None = None) -> int | None:
    """Declared max context: max_position_embeddings (or seq_length / n_positions); when it equals the pre-scaling
    window of a rope_scaling block (factor x original_max_position_embeddings), the scaled window; else
    model_max_length. Implausible values (tokenizer defaults like 1e30) are ignored."""
    cfg, api_cfg = cfg or {}, api_cfg or {}
    tc = _text_config(cfg)
    ctx = _first(tc, "max_position_embeddings", "seq_length", "n_positions", "max_sequence_length") or \
        _first(cfg, "max_position_embeddings", "seq_length")
    rs = tc.get("rope_scaling") or cfg.get("rope_scaling") or {}
    if isinstance(rs, dict) and rs.get("factor") and rs.get("original_max_position_embeddings"):
        scaled = int(float(rs["factor"]) * int(rs["original_max_position_embeddings"]))
        if not ctx or int(ctx) <= int(rs["original_max_position_embeddings"]):
            ctx = scaled
    elif isinstance(rs, dict) and rs.get("factor") and ctx and \
            str(rs.get("rope_type") or rs.get("type") or "").lower() == "linear":
        ctx = int(float(rs["factor"]) * int(ctx))                # linear position interpolation (e.g. 4K x 4 = 16K)
    if not ctx:
        ctx = _first(cfg, "model_max_length") or _first(tc, "model_max_length")
    try:
        ctx = int(ctx)
    except (TypeError, ValueError):
        return None
    return ctx if 512 <= ctx <= 20_000_000 else None


def _experts(tc: dict) -> tuple[int, int]:
    """(routed experts, experts per token); (0, 0) for dense models."""
    n = int(_first(tc, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", default=0) or 0)
    k = int(_first(tc, "num_experts_per_tok", "experts_per_token", "moe_topk", "top_k_experts",
                   "num_experts_per_token", "moe_k", default=0) or 0)
    if tc.get("enable_moe_block") is False:
        return 0, 0
    return (n, k) if n > 1 and 0 < k < n else (0, 0)


def _moe_layers(tc: dict) -> int:
    """How many decoder layers have a routed-expert block (MTP / next-token-prediction layers not included)."""
    L = int(tc.get("num_hidden_layers") or 0)
    if isinstance(tc.get("moe_layers"), list):
        return len(tc["moe_layers"])
    if isinstance(tc.get("moe_layer_freq"), list):                # per-layer flags (MiMo): 1 = MoE layer
        return sum(1 for x in tc["moe_layer_freq"] if x)
    fk = int(tc.get("first_k_dense_replace") or 0)               # DeepSeek-V2/V3, Kimi K2, GLM MoE
    freq = int(tc.get("moe_layer_freq") or 1) if isinstance(tc.get("moe_layer_freq"), int) else 1
    if fk or "moe_layer_freq" in tc:
        return sum(1 for i in range(fk, L) if i % max(1, freq) == 0)
    step = int(tc.get("interleave_moe_layer_step") or 0)       # Llama 4 (Maverick: every 2nd layer)
    if step > 1:
        return L // step
    mlp_only = tc.get("mlp_only_layers") or []                   # Qwen MoE
    sparse = int(tc.get("decoder_sparse_step") or 1)
    if isinstance(mlp_only, dict):
        mlp_only = []
    return sum(1 for i in range(L) if i not in mlp_only and (i + 1) % max(1, sparse) == 0)


def _attn_params(tc: dict) -> float:
    H = int(tc.get("hidden_size") or 0)
    heads = int(tc.get("num_attention_heads") or 1)
    if tc.get("kv_lora_rank"):                                    # multi-head latent attention (DeepSeek, Kimi)
        qr, kvr = int(tc.get("q_lora_rank") or 0), int(tc["kv_lora_rank"])
        nope, rope = int(tc.get("qk_nope_head_dim") or 128), int(tc.get("qk_rope_head_dim") or 64)
        v = int(tc.get("v_head_dim") or 128)
        q = (H * qr + qr * heads * (nope + rope)) if qr else H * heads * (nope + rope)
        return q + H * (kvr + rope) + kvr * heads * (nope + v) + heads * v * H
    hd = int(tc.get("head_dim") or (H // max(1, heads)))
    kvh = int(tc.get("num_key_value_heads") or heads)
    return H * heads * hd * 2 + 2 * H * kvh * hd


def config_params(tc: dict) -> dict | None:
    """Rough parameter count of a decoder from its config (embeddings, attention, dense MLP, experts), used to sanity-
    check the Hub's safetensors total and to size packed checkpoints. None without the basic fields."""
    H, L, V = (int(tc.get(k) or 0) for k in ("hidden_size", "num_hidden_layers", "vocab_size"))
    if not (H and L and V):
        return None
    n, k = _experts(tc)
    attn = _attn_params(tc)
    dense_i = int(_first(tc, "intermediate_size_mlp", "intermediate_size", default=0) or 0)
    embed = V * H * (1 if tc.get("tie_word_embeddings") else 2)
    if not n:
        return {"total": embed + L * (attn + 3 * H * dense_i), "expert": 0, "moe_layers": 0, "attn": attn}
    moe_i = int(_first(tc, "moe_intermediate_size", "expert_intermediate_size", default=0) or tc.get("intermediate_size") or 0)
    expert = 3 * H * moe_i
    shared = (int(tc.get("n_shared_experts") or 0) * expert
              + 3 * H * int(tc.get("shared_expert_intermediate_size") or 0))
    if str(tc.get("model_type", "")).startswith("llama4"):
        shared += expert                                         # Llama 4: one shared expert per MoE layer
    n_moe = _moe_layers(tc)
    mtp = int(_first(tc, "num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers", default=0) or 0)
    moe_layer = attn + n * expert + shared + H * n
    dense_layer = attn + 3 * H * dense_i
    total = embed + (L - n_moe) * dense_layer + n_moe * moe_layer + mtp * (moe_layer + 2 * H * H)
    return {"total": total, "expert": expert, "shared": shared, "moe_layers": n_moe, "mtp_layers": mtp,
            "attn": attn, "experts": n, "top_k": k}


def moe_active_params(tc: dict, total: float) -> float | None:
    """Active parameters per token of a mixture-of-experts model, from its config and total parameter count.

    active = total - (MoE layers + MTP layers) x (routed experts - experts per token) x expert size
                   - MTP layers x (attention + active and shared experts + 2 x hidden^2 projection)
    with expert size = 3 x hidden x expert intermediate size (gate, up, down). Everything else (embeddings, attention,
    dense layers, shared experts, the dense MLP Gemma 4 runs next to its experts, vision towers) stays active, so the
    formula needs only the expert dimensions: Mixtral (num_local_experts, intermediate_size), DeepSeek-V2/V3 / Kimi K2
    / GLM MoE (n_routed_experts, moe_intermediate_size, first_k_dense_replace, n_shared_experts, MTP layers), Qwen MoE
    (num_experts, decoder_sparse_step, mlp_only_layers, shared_expert_intermediate_size), gpt-oss
    (experts_per_token), Gemma 4 (top_k_experts), Llama 4 (interleave_moe_layer_step). None for dense models.
    """
    p = config_params(tc)
    if not p or not p.get("experts"):
        return None
    inactive = (p["moe_layers"] + p["mtp_layers"]) * (p["experts"] - p["top_k"]) * p["expert"]
    mtp_active = p["mtp_layers"] * (p["attn"] + p["top_k"] * p["expert"] + p["shared"]
                                    + 2 * int(tc.get("hidden_size") or 0) ** 2)
    active = float(total) - inactive - mtp_active
    # Sanity: the routed experts alone make up about total x k / n of the active weights; far below that the config
    # has fields this formula doesn't model.
    floor = float(total) * p["top_k"] / p["experts"] * 0.5
    return active if floor < active < float(total) else None


# Bytes per element of the Hub's safetensors dtypes (per-dtype parameter counts in the model API).
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1,
               "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
MXFP4_BITS = 4.25          # 4-bit E2M1 values + one shared 8-bit scale per 32-value block


def mxfp4_weight_bytes(st_params: dict | None, tc: dict | None = None) -> tuple[float, str] | None:
    """Bytes of an MXFP4 checkpoint from the Hub's per-dtype safetensors counts: the quantized (expert) tensors are
    stored as U8 and sized at 4.25 bits per parameter, everything kept in BF16 / F32 (attention, router, embeddings,
    biases) at its own width. The Hub counts the packed expert tensors at their logical parameter count (gpt-oss-120b:
    U8 114.66B = 36 layers x 128 experts x 3 x 2880^2, BF16 2.17B -> ~65 GB, the size of the checkpoint); when the U8
    count instead matches the packed size (~0.53 x the config's expert parameters) it is taken as bytes.
    Returns (bytes, explanation) or None without U8 counts."""
    st = {str(k).upper(): float(v or 0) for k, v in (st_params or {}).items()}
    u8 = st.get("U8") or 0.0
    if u8 <= 0:
        return None
    other = sum(v * DTYPE_BYTES.get(k, 2) for k, v in st.items() if k != "U8")
    cp = config_params(tc) if tc else None
    experts = (cp["expert"] * cp["experts"] * (cp["moe_layers"] + cp["mtp_layers"])) if cp and cp.get("experts") else 0
    packed = bool(experts and abs(u8 - experts * MXFP4_BITS / 8) < 0.1 * experts * MXFP4_BITS / 8)
    q = u8 if packed else u8 * MXFP4_BITS / 8
    note = (f"MXFP4 weights {q / 1e9:,.1f} GB ({u8 / 1e9:,.2f}B {'bytes' if packed else 'parameters at 4.25 bits'}) "
            f"+ {other / 1e9:,.1f} GB kept in 16 bits, from the Hub's safetensors counts")
    return q + other, note


_MULTIMODAL_KEYS = ("vision_config", "audio_config", "vision_tower", "mm_projector_type")


def parse(rec: dict | None) -> dict | None:
    """Catalog fields from a raw cache record (fetch()), each one only when the Hub really provides it.

    Returns {hf_id, updated_at, fields: {name: value}, notes: [...], status} or None for missing / failed records.
    Fields: params_b, active_params_b, max_context, checkpoint_precision, native_precision (int4 / fp8 / mxfp4 only:
    the precisions the planner models), mxfp4_weight_gb + mxfp4_weight_note (MXFP4 checkpoints, from the per-dtype
    safetensors counts), released (YYYY-MM from createdAt), license, gated, model_type, last_modified.
    """
    if not rec or rec.get("status") != "ok" or not rec.get("api"):
        return None
    api, cfg = rec["api"], rec.get("config") or {}
    api_cfg = api.get("config") or {}
    tc = _text_config(cfg) if cfg else {}
    st = api.get("safetensors") or {}
    st_params = st.get("parameters") or {}
    f: dict = {}
    notes: list[str] = []

    prec = detect_precision(cfg, api_cfg, st_params)
    if prec:
        f["checkpoint_precision"] = prec
        if prec in ("int4", "fp8", "mxfp4"):
            f["native_precision"] = prec
        if prec == "mxfp4":
            mb = mxfp4_weight_bytes(st_params, tc)
            if mb:
                f["mxfp4_weight_gb"], f["mxfp4_weight_note"] = round(mb[0] / 1e9, 2), mb[1]

    total = float(st.get("total") or 0)
    qmethod = str(((cfg.get("quantization_config") or tc.get("quantization_config") or api_cfg.get("quantization_config")
                    or {}).get("quant_method")) or "").lower()
    cp = config_params(tc) if tc else None
    if total and qmethod in ("awq", "gptq") and st_params.get("I32"):
        # packed 4-bit weights: 8 values per int32 element
        total = st_params["I32"] * 8 + sum(v for kk, v in st_params.items() if kk != "I32")
        notes.append("packed 4-bit checkpoint: total from unpacked int32 weights")
    if total:
        f["params_b"] = round(total / 1e9, 2)

    if tc and total:
        n, _k = _experts(tc)
        multimodal = any(k in cfg for k in _MULTIMODAL_KEYS)
        if n:
            act = moe_active_params(tc, total)
            # Consistency guard: for text-only configs the config-derived total must agree with the checkpoint's
            # (else the formula doesn't understand this architecture): fall back to the rules.
            # (multimodal checkpoints add vision / audio towers: only a config *above* the checkpoint is suspicious)
            off = (cp["total"] - total) / total if act and cp else 0.0
            if act and (off > 0.15 or (not multimodal and off < -0.15)):
                log.info("hf_meta %s: config total %.1fB vs safetensors %.1fB, MoE active params not trusted",
                         rec.get("hf_id"), cp["total"] / 1e9, total / 1e9)
                notes.append("MoE active parameters: config doesn't match the checkpoint, built-in estimate used")
                act = None
            f["moe"] = {"experts": n, "experts_per_token": _k}
            if act:
                f["active_params_b"] = round(act / 1e9, 2)
            else:
                f["moe_unresolved"] = True
                if not any("MoE active" in x for x in notes):
                    notes.append("MoE active parameters: config fields not understood, built-in estimate used")
        else:
            f["dense"] = True

    ctx = context_length(cfg, api_cfg) if (cfg or api_cfg) else None
    if ctx:
        f["max_context"] = ctx
    if api.get("createdAt"):
        f["released"] = str(api["createdAt"])[:7]
    card = api.get("cardData") or {}
    lic = card.get("license_name") if card.get("license") == "other" and card.get("license_name") else card.get("license")
    if lic:
        f["license"] = lic if isinstance(lic, str) else ", ".join(map(str, lic))
    if "gated" in api:
        f["gated"] = api.get("gated") not in (False, None, "false")
    if api.get("pipeline_tag"):
        f["pipeline_tag"] = api["pipeline_tag"]
    mt = (tc.get("model_type") if tc else None) or api_cfg.get("model_type")
    if mt:
        f["model_type"] = mt
    if api.get("lastModified"):
        f["last_modified"] = str(api["lastModified"])[:10]
    if rec.get("config_status") not in (200, None) and not cfg:
        notes.append("config.json not readable" + (" (gated: add a Hugging Face token in Settings or set HF_TOKEN)"
                                                   if rec.get("config_status") in (401, 403) else ""))
    return {"hf_id": rec.get("hf_id"), "updated_at": int(rec.get("fetched_at") or 0), "fields": f, "notes": notes,
            "status": "ok"}


# --------------------------------------------------------------------------------------------- cache + refresh
_MEM: dict[str, tuple[float, dict | None]] = {}
_VERSION = [0]               # bumped whenever new metadata lands (modelinfo re-curates on change)


def version() -> int:
    return _VERSION[0]


def _safe_parse(hf_id: str) -> dict | None:
    """parse() that never raises: one model with an unexpected config must not break the whole catalog."""
    try:
        return parse(_load_raw(hf_id))
    except Exception as e:   # noqa: BLE001
        print(f"  (Hugging Face metadata for {hf_id} not usable: {type(e).__name__}: {str(e)[:100]}; using built-in values)")
        return None


def cached(hf_id: str | None) -> dict | None:
    """Parsed metadata from the on-disk cache (any age; never touches the network)."""
    if disabled() or not valid_id(hf_id):
        return None
    p = _path(hf_id)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    hit = _MEM.get(p.name)
    if hit and hit[0] == mtime:
        return hit[1]
    meta = _safe_parse(hf_id)
    _MEM[p.name] = (mtime, meta)
    return meta


def invalidate_auth_failures() -> int:
    """Delete cached records that failed with 401 / 403 (model API or config.json), so the next refresh refetches
    them with the current token. Returns how many were dropped."""
    n = 0
    try:
        files = list(_dir().glob("*.json"))
    except OSError:
        return 0
    for p in files:
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if rec.get("http_status") in (401, 403) or rec.get("config_status") in (401, 403):
            try:
                p.unlink()
                _MEM.pop(p.name, None)
                n += 1
            except OSError:
                pass
    if n:
        _VERSION[0] += 1
    return n


def refresh(hf_ids, force: bool = False, workers: int = 6) -> dict:
    """Fetch stale (or all, `force`) models now. Returns counts {requested, fetched, ok, failed}."""
    ids = [i for i in dict.fromkeys(hf_ids or []) if valid_id(i)]
    todo = [i for i in ids if force or is_stale(i)]
    stats = {"requested": len(ids), "fetched": 0, "ok": 0, "failed": 0}
    if disabled() or not todo:
        return stats
    rate_limited = threading.Event()

    def one(hf_id):
        if rate_limited.is_set():
            return None
        rec = fetch(hf_id)
        if rec.get("http_status") == 429:
            rate_limited.set()
            return None
        _save_raw(hf_id, rec)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(one, todo):
            if rec is None:
                continue
            stats["fetched"] += 1
            stats["ok" if rec["status"] == "ok" else "failed"] += 1
    if stats["fetched"]:
        _VERSION[0] += 1
    if rate_limited.is_set():
        log.warning("hf_meta: Hugging Face rate limit hit; the rest is fetched on the next refresh")
        stats["rate_limited"] = True
    return stats


ON_UPDATE: list[Callable[[], None]] = []    # called (in the background thread) after new metadata was fetched
_BG = {"thread": None, "last": None}
_BG_LOCK = threading.Lock()


def refresh_bg(hf_ids, force: bool = False) -> bool:
    """Fetch stale entries in a background thread (one at a time). True if a refresh started."""
    if disabled():
        return False
    ids = [i for i in dict.fromkeys(hf_ids or []) if valid_id(i)]
    if not force and not any(is_stale(i) for i in ids):
        return False
    with _BG_LOCK:
        t = _BG["thread"]
        if t and t.is_alive():
            return False

        def work():
            try:
                stats = refresh(ids, force=force)
                _BG["last"] = {**stats, "at": int(time.time())}
                if stats.get("fetched"):
                    for cb in list(ON_UPDATE):
                        try:
                            cb()
                        except Exception as e:
                            log.debug("hf_meta: update callback failed: %s", e)
            except Exception as e:
                log.warning("hf_meta: background refresh failed: %s", e)

        _BG["thread"] = threading.Thread(target=work, daemon=True, name="hf-meta-refresh")
        _BG["thread"].start()
    return True


def refreshing() -> bool:
    t = _BG["thread"]
    return bool(t and t.is_alive())
