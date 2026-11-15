"""GPU and Google Cloud TPU machine shapes, Gemini on Agent Platform tiers, and seed Model Garden metadata.

Which accelerators and machine shapes exist in a region comes from the Compute Engine API (hardware.py); this module
holds the one table no API provides, CHIP_SPECS, and the hand-made shapes used as a fallback when the API can't be
reached (FALLBACK_MACHINES). GPUS = CHIP_SPECS + FALLBACK_MACHINES (the offline view, and what tests use).
"""

# ------------------------------------------------------------------------------------------------- CHIP_SPECS
# Per-chip performance (no Google API publishes these). Sources: NVIDIA data sheets (L4, RTX PRO 6000 Blackwell Server
# Edition, A100 40/80 GB SXM, H100 SXM, H200 SXM, B200 HGX) and Google Cloud TPU system-architecture pages (v5e,
# v6e Trillium, v5p); checked 2026-11. Approximate and per accelerator:
#   mem_gb  memory (VRAM / HBM)      tflops  dense BF16 tensor TFLOPS     bw_gbs  memory bandwidth (GB/s)
#   nvlink  fast interconnect (NVLink for GPUs, ICI for TPUs -> better tensor-parallel scaling)
#   family  machine family for Billing SKU matching; sku / sku_rx / host_bundled: how pricing.py finds the SKUs.
# An accelerator the Compute Engine API lists but this table lacks is shown as "no performance data" and not
# estimated (planner.py). Add a row here (and its SKU patterns) to plan it.
CHIP_SPECS = {
    "NVIDIA_L4": {"name": "L4", "kind": "gpu", "bw_gbs": 300, "mem_gb": 24, "tflops": 121, "nvlink": False,
                  "family": "G2", "sku": {"vertex": "L4", "gce": "Nvidia L4 GPU"}},
    "NVIDIA_RTX_PRO_6000": {"name": "RTX PRO 6000", "kind": "gpu", "bw_gbs": 1792, "mem_gb": 96, "tflops": 500,
                            "nvlink": False, "family": "G4", "sku": {"vertex": "RTX 6000", "gce": "RTX 6000 96GB"}},
    "NVIDIA_TESLA_A100": {"name": "A100 40GB", "kind": "gpu", "bw_gbs": 1555, "mem_gb": 40, "tflops": 312,
                          "nvlink": True, "family": "A2", "sku": {"vertex": "A100", "gce": "Nvidia Tesla A100 GPU"}},
    "NVIDIA_A100_80GB": {"name": "A100 80GB", "kind": "gpu", "bw_gbs": 2039, "mem_gb": 80, "tflops": 312,
                         "nvlink": True, "family": "A2",
                         "sku": {"vertex": "A100 80GB", "gce": "Nvidia Tesla A100 80GB GPU"}},
    "NVIDIA_H100_80GB": {"name": "H100 80GB", "kind": "gpu", "bw_gbs": 3350, "mem_gb": 80, "tflops": 989,
                         "nvlink": True, "family": "A3", "sku": {"vertex": "H100 80GB", "gce": "Nvidia H100 80GB GPU"}},
    "NVIDIA_H200_141GB": {
        "name": "H200 141GB", "kind": "gpu", "bw_gbs": 4800, "mem_gb": 141, "tflops": 989, "nvlink": True,
        "family": "A3", "sku": {"vertex": "H200", "gce": "Nvidia H200 141GB GPU"},
        # A3 Ultra SKUs: "Online/Batch Prediction NVIDIA H200 141GB in <region>", "H200 141GB GPU running in Americas"
        # (per GPU, host included: no separate A3 Ultra core/RAM SKUs).
        "sku_rx": {"v_gpu": r"Online/Batch Prediction NVIDIA H200 141GB in", "c_gpu": r"^H200 141GB GPU running in",
                   "c_gpu_commit": r"^Commitment v1: H200 141GB GPU running in .* for {term} Year"},
        "host_bundled": True},
    # A4 (NVIDIA B200, Blackwell: FP8 and FP4 tensor cores): 180 GB HBM3e and ~8 TB/s per GPU, ~2,250 dense BF16
    # TFLOPS; billed per GPU slice with the host included.
    "NVIDIA_B200": {
        "name": "B200 180GB", "kind": "gpu", "bw_gbs": 8000, "mem_gb": 180, "tflops": 2250, "nvlink": True,
        "family": "A4", "sku": {"vertex": "B200", "gce": "A4 Nvidia B200 (1 gpu slice)"},
        "sku_rx": {"c_gpu_commit": r"^Commitment v1: A4 Nvidia B200 \(1 gpu slice\) in .* for {term} Year"},
        "host_bundled": True},
    # Google Cloud TPU options (HBM per chip, BF16 TFLOPS, ICI interconnect)
    "TPU_V5E": {"name": "Cloud TPU v5e", "kind": "tpu", "bw_gbs": 819, "mem_gb": 16, "tflops": 197, "nvlink": True,
                "family": "TPU v5e", "sku": {"vertex": "TPU v5e", "gce": "TPU v5e"}},
    "TPU_V6E": {"name": "Cloud TPU v6e (Trillium)", "kind": "tpu", "bw_gbs": 1600, "mem_gb": 32, "tflops": 918,
                "nvlink": True, "family": "TPU v6e", "sku": {"vertex": "TPU v6e", "gce": "TPU v6e"}},
    "TPU_V5P": {"name": "Cloud TPU v5p", "kind": "tpu", "bw_gbs": 2765, "mem_gb": 95, "tflops": 459, "nvlink": True,
                "family": "TPU v5p", "sku": {"vertex": "TPU v5p", "gce": "TPU v5p"}},
}

# ------------------------------------------------------------------------------------------ FALLBACK_MACHINES
# Hand-made machine shapes per accelerator count: (machine type, vCPUs, host RAM GB). Fallback only: the Compute
# Engine API (hardware.py) decides which shapes exist in the chosen region; these names are preferred when a count has
# several shapes there (e.g. g2-standard-12 among the 1 x L4 machines).
FALLBACK_MACHINES = {
    "NVIDIA_L4": {1: ("g2-standard-12", 12, 48), 2: ("g2-standard-24", 24, 96), 4: ("g2-standard-48", 48, 192),
                  8: ("g2-standard-96", 96, 384)},
    "NVIDIA_RTX_PRO_6000": {1: ("g4-standard-48", 48, 180), 2: ("g4-standard-96", 96, 360),
                            4: ("g4-standard-192", 192, 720), 8: ("g4-standard-384", 384, 1440)},
    "NVIDIA_TESLA_A100": {1: ("a2-highgpu-1g", 12, 85), 2: ("a2-highgpu-2g", 24, 170), 4: ("a2-highgpu-4g", 48, 340),
                          8: ("a2-highgpu-8g", 96, 680)},
    "NVIDIA_A100_80GB": {1: ("a2-ultragpu-1g", 12, 170), 2: ("a2-ultragpu-2g", 24, 340),
                         4: ("a2-ultragpu-4g", 48, 680), 8: ("a2-ultragpu-8g", 96, 1360)},
    "NVIDIA_H100_80GB": {1: ("a3-highgpu-1g", 26, 234), 2: ("a3-highgpu-2g", 52, 468), 4: ("a3-highgpu-4g", 104, 936),
                         8: ("a3-highgpu-8g", 208, 1872)},
    "NVIDIA_H200_141GB": {8: ("a3-ultragpu-8g", 224, 2952)},
    "NVIDIA_B200": {8: ("a4-highgpu-8g", 224, 3968)},
    "TPU_V5E": {1: ("ct5lp-hightpu-1t", 24, 48), 4: ("ct5lp-hightpu-4t", 96, 192), 8: ("ct5lp-hightpu-8t", 192, 384)},
    "TPU_V6E": {1: ("ct6e-standard-1t", 44, 176), 4: ("ct6e-standard-4t", 176, 704), 8: ("ct6e-standard-8t", 180, 1440)},
    "TPU_V5P": {4: ("ct5p-hightpu-4t", 208, 448), 8: ("ct5p-hightpu-8t", 208, 448)},
}

GPUS = {acc: {**spec, "machines": dict(FALLBACK_MACHINES.get(acc, {}))} for acc, spec in CHIP_SPECS.items()}

# ------------------------------------------------------------------------------------------ MXFP4 support
# Where vLLM documents MXFP4 (gpt-oss) serving, per its gpt-oss recipe (docs.vllm.ai/projects/recipes, OpenAI/GPT-OSS,
# read 2026-11-09): "you can run it on NVIDIA H100, H200, B200 [...]. We are actively working on ensuring this model
# can work on Ampere, Ada Lovelace, and RTX 5090", and "GPT-OSS works on Ampere devices by default, using the
# TRITON_ATTN attention backend and Marlin MXFP4 MoE" (A100). Model Garden's verified recipes (live,
# modelinfo.verified_options) are the other authority: on 2026-11-09 (us-central1) gpt-oss-20b had vLLM recipes on
# 1x RTX PRO 6000, 1x A100 80GB (env VLLM_ATTENTION_BACKEND=TRITON_ATTN), 1x / 8x H100 and 8x H200 (+ SGLang 8x H100),
# gpt-oss-120b on 2x RTX PRO 6000, 2x / 8x H100, 8x H200 (+ SGLang 8x H100 with EAGLE3 and 8x B200); none passes
# --quantization. Other GPUs (L4: Ada) are offered but labelled untested; TPUs aren't offered.
# Fallback when the live recipes can't be read (no project / not signed in): accelerators Model Garden had a gpt-oss
# vLLM recipe for on 2026-11-09 beyond the vLLM-documented ones. Labelled as such, never shown as "verified".
MXFP4_RECIPES_SEEN = {"NVIDIA_RTX_PRO_6000": "2026-11-09"}
MXFP4_VLLM_DOCUMENTED = {
    "NVIDIA_H100_80GB": "Hopper", "NVIDIA_H200_141GB": "Hopper", "NVIDIA_B200": "Blackwell",
    "NVIDIA_A100_80GB": "Ampere (Triton attention + Marlin MXFP4 kernels)",
    "NVIDIA_TESLA_A100": "Ampere (Triton attention + Marlin MXFP4 kernels)",
}
ACCELERATORS = GPUS

# Tensor-parallel scaling efficiency (PCIe vs NVLink/ICI), used for ranking & speed proxy.
TP_EFF = {False: {1: 1.0, 2: 0.90, 4: 0.80, 8: 0.65}, True: {1: 1.0, 2: 0.95, 4: 0.90, 8: 0.85}}

# Reference per-accelerator-hour prices (USD, us-central1 baseline) used as fallback when Cloud Billing API is offline.
FALLBACK_PRICES = {
    "NVIDIA_L4": {"vertex": 1.44, "vertex_cud_1y": 1.01, "vertex_cud_3y": 0.74, "on_demand": 1.16, "cud_1y": 0.73, "cud_3y": 0.46, "management_fee": 0.28},
    "NVIDIA_RTX_PRO_6000": {"vertex": 6.13, "vertex_cud_1y": 3.88, "vertex_cud_3y": 2.64, "on_demand": 4.95, "cud_1y": 3.12, "cud_3y": 1.98, "management_fee": 1.18},
    "NVIDIA_TESLA_A100": {"vertex": 4.58, "vertex_cud_1y": 3.05, "vertex_cud_3y": 2.11, "on_demand": 3.67, "cud_1y": 2.31, "cud_3y": 1.47, "management_fee": 0.91},
    "NVIDIA_A100_80GB": {"vertex": 6.32, "vertex_cud_1y": 4.21, "vertex_cud_3y": 2.91, "on_demand": 5.07, "cud_1y": 3.19, "cud_3y": 2.03, "management_fee": 1.25},
    "NVIDIA_H100_80GB": {"vertex": 13.75, "vertex_cud_1y": 9.40, "vertex_cud_3y": 6.55, "on_demand": 11.06, "cud_1y": 7.42, "cud_3y": 4.86, "management_fee": 2.69},
    "NVIDIA_B200": {"vertex": 20.94, "vertex_cud_1y": 13.29, "vertex_cud_3y": 9.02, "on_demand": 16.11, "cud_1y": 11.12, "cud_3y": 7.09, "management_fee": 2.42},
    "NVIDIA_H200_141GB": {"vertex": 15.90, "vertex_cud_1y": 10.90, "vertex_cud_3y": 7.60, "on_demand": 12.80, "cud_1y": 8.60, "cud_3y": 5.65, "management_fee": 3.10},
    "TPU_V5E": {"vertex": 1.50, "vertex_cud_1y": 1.02, "vertex_cud_3y": 0.69, "on_demand": 1.20, "cud_1y": 0.76, "cud_3y": 0.54, "management_fee": 0.30},
    "TPU_V6E": {"vertex": 3.35, "vertex_cud_1y": 2.28, "vertex_cud_3y": 1.54, "on_demand": 2.70, "cud_1y": 1.70, "cud_3y": 1.21, "management_fee": 0.65},
    "TPU_V5P": {"vertex": 5.25, "vertex_cud_1y": 3.58, "vertex_cud_3y": 2.42, "on_demand": 4.20, "cud_1y": 2.65, "cud_3y": 1.89, "management_fee": 1.05},
}

# Google Gemini on Agent Platform (pay-per-use closed MaaS for long-context diversion, peak shaving, and Layer-3 incident failover).
# Fallback only (labelled "fallback"): gemini.py lists every Gemini text model live from the Cloud Billing Catalog +
# Model Garden and resolves these role aliases dynamically. Values below are what the aliases resolved to on
# 2026-11-14, $ per 1M tokens; tests price with them, so their numbers stay fixed.
GEMINI_MODELS = {
    "gemini_flash_lite": {
        "id": "gemini_flash_lite",
        "name": "Gemini 3.5 Flash-Lite",
        "model_id": "gemini-3.5-flash-lite",
        "family": "flash-lite",
        "stage": "ga",
        "version": "3.5",
        "input_per_m": 0.30,
        "output_per_m": 2.50,
        "cached_per_m": 0.03,
        "long_input_per_m": 0.30,
        "long_output_per_m": 2.50,
        "max_context": 1048576,
        "default_max_rps": 200,
        "summary": "Lowest-cost Gemini tier on Agent Platform with a 1M context window; peak shaving, high-volume failover, long-context extraction.",
    },
    "gemini_flash": {
        "id": "gemini_flash",
        "name": "Gemini 3.8 Flash",
        "model_id": "gemini-3.8-flash",
        "family": "flash",
        "stage": "ga",
        "version": "3.8",
        "input_per_m": 1.50,
        "output_per_m": 7.50,
        "cached_per_m": 0.15,
        "long_input_per_m": 1.50,
        "long_output_per_m": 7.50,
        "max_context": 1048576,
        "default_max_rps": 150,
        "summary": "Workhorse Gemini tier on Agent Platform with 1M context and thinking; >32K long-context routing and zero-idle-cost failover.",
    },
    "gemini_pro": {
        "id": "gemini_pro",
        "name": "Gemini 3.1 Pro",
        "model_id": "gemini-3.1-pro",
        "family": "pro",
        "stage": "preview",
        "version": "3.1",
        "input_per_m": 2.00,
        "output_per_m": 12.00,
        "cached_per_m": 0.20,
        "long_input_per_m": 4.00,
        "long_output_per_m": 18.00,
        # "(Long)" SKUs: prompts over 200K tokens
        "long_context": {"threshold": 200000, "input_per_m": 4.00, "output_per_m": 18.00, "cached_per_m": 0.40},
        "max_context": 1048576,
        "default_max_rps": 80,
        "summary": "Frontier Gemini reasoning tier on Agent Platform for complex >128K documents.",
    },
    "gemini_4": {
        "id": "gemini_4",
        "name": "Gemini 4 Carbon",
        "model_id": "gemini-4-carbon",
        "family": "carbon",
        "stage": "ga",
        "version": "4",
        "input_per_m": 4.00,
        "output_per_m": 20.00,
        "cached_per_m": 0.20,
        "long_input_per_m": 4.00,
        "long_output_per_m": 20.00,
        "max_context": 1048576,
        "default_max_rps": 50,
        "summary": "Newest Gemini generation's flagship on Agent Platform (Billing Catalog \"Gemini 4 Carbon\", same price as Argon).",
    },
}

_GEMINI_PRESET_RPS = {"gemini_flash_lite": 150, "gemini_flash": 100, "gemini_pro": 50, "gemini_4": 50}


def maas_presets(gemini: dict | None = None) -> dict:
    """MaaS presets; Gemini entries take their prices from `gemini` (live catalog) or GEMINI_MODELS."""
    gem = gemini or GEMINI_MODELS
    out = {"vertex_open_maas": {"name": "Agent Platform Open-Model MaaS (e.g. Gemma / Llama / DeepSeek)",
                                "input_per_m": 0.15, "output_per_m": 0.60, "cached_per_m": 0.015, "max_rps": 10}}
    for k, rps in _GEMINI_PRESET_RPS.items():
        g = gem.get(k) or GEMINI_MODELS[k]   # older cached catalogs may lack newer tiers (e.g. gemini_4)
        out[k] = {"name": f"Google {g['name']} on Agent Platform", "input_per_m": g["input_per_m"],
                  "output_per_m": g["output_per_m"], "cached_per_m": g["cached_per_m"], "max_rps": rps}
    return out


MAAS_PRESETS = maas_presets()

# Seed Model Garden open text LLMs. modelinfo.curate_models() merges them with the live Model Garden listing and fills
# generation / released / legacy / specialty / precisions (every model: bf16 + fp8, FP8 = vLLM on-the-fly unless a
# pre-quantized Model Garden checkpoint exists).
SEED_MODELS = [
    {
        "id": "gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B-A4B",
        "family": "Gemma",
        "model_garden_id": "google/gemma4@gemma-4-26b-a4b-it",
        "hf_id": "google/gemma-4-26b-a4b-it",
        "params_b": 26.0,
        "active_params_b": 4.0,
        "max_context": 262144,
        "supports_thinking": True,
        "supports_json": True,
        "tasks": ["extraction", "summarization", "chat", "reasoning"],
        "quality_tier": "balanced",
        "summary": "Google's MoE open model (26B total, 4B active parameters, 256K context). Delivers near-4B inference speed with 27B-class quality and optional thinking.",
    },
    {
        "id": "gemma-4-31b-it",
        "name": "Gemma 4 31B",
        "family": "Gemma",
        "model_garden_id": "google/gemma4@gemma-4-31b-it",
        "hf_id": "google/gemma-4-31b-it",
        "params_b": 31.0,
        "active_params_b": 31.0,
        "max_context": 262144,
        "supports_thinking": True,
        "supports_json": True,
        "tasks": ["reasoning", "coding", "extraction", "summarization"],
        "quality_tier": "frontier",
        "summary": "Dense 31B Google open model with 256K context and strong reasoning, coding, and structured extraction accuracy.",
    },
    {
        "id": "gemma-4-12b-it",
        "name": "Gemma 4 12B",
        "family": "Gemma",
        "model_garden_id": "google/gemma4@gemma-4-12b-it",
        "hf_id": "google/gemma-4-12b-it",
        "params_b": 12.0,
        "active_params_b": 12.0,
        "max_context": 131072,
        "supports_thinking": True,
        "supports_json": True,
        "tasks": ["extraction", "summarization", "chat", "reasoning"],
        "quality_tier": "compact",
        "summary": "Dense 12B Gemma 4 with thinking and 128K context; fits one 40 GB+ GPU in BF16, or one L4 in FP8.",
    },
    {
        "id": "gemma-4-e4b-it",
        "name": "Gemma 4 E4B",
        "family": "Gemma",
        "model_garden_id": "google/gemma4@gemma-4-e4b-it",
        "hf_id": "google/gemma-4-e4b-it",
        "params_b": 8.0,
        "active_params_b": 4.5,
        "max_context": 131072,
        "supports_thinking": True,
        "supports_json": True,
        "tasks": ["extraction", "summarization", "chat"],
        "quality_tier": "compact",
        "summary": "Edge-sized Gemma 4 (about 4.5B effective, 8B with per-layer embeddings, 128K context): the cheapest Gemma 4 to serve for simple extraction and chat.",
    },
    {
        "id": "gemma-4-e2b-it",
        "name": "Gemma 4 E2B",
        "family": "Gemma",
        "model_garden_id": "google/gemma4@gemma-4-e2b-it",
        "hf_id": "google/gemma-4-e2b-it",
        "params_b": 5.1,
        "active_params_b": 2.3,
        "max_context": 131072,
        "supports_thinking": True,
        "supports_json": True,
        "tasks": ["extraction", "chat"],
        "quality_tier": "compact",
        "summary": "Smallest Gemma 4 (about 2.3B effective, 5.1B with per-layer embeddings, 128K context) for high-volume, simple tasks.",
    },
    {
        "id": "gemma-3-12b-it",
        "name": "Gemma 3 12B",
        "family": "Gemma",
        "model_garden_id": "google/gemma3@gemma-3-12b-it",
        "hf_id": "google/gemma-3-12b-it",
        "params_b": 12.2,
        "active_params_b": 12.2,
        "max_context": 131072,
        "supports_thinking": False,
        "supports_json": True,
        "tasks": ["extraction", "chat", "summarization"],
        "quality_tier": "compact",
        "summary": "Compact 12B open model with 128K context that fits comfortably on a single L4 (FP8) or A100/RTX PRO 6000.",
    },
    {
        "id": "gemma-3-27b-it",
        "name": "Gemma 3 27B",
        "family": "Gemma",
        "model_garden_id": "google/gemma3@gemma-3-27b-it",
        "hf_id": "google/gemma-3-27b-it",
        "params_b": 27.4,
        "active_params_b": 27.4,
        "max_context": 131072,
        "supports_thinking": False,
        "supports_json": True,
        "tasks": ["extraction", "summarization", "reasoning", "chat"],
        "quality_tier": "balanced",
        "summary": "Proven 27B dense model with 128K context window and broad Agent Platform GPU & TPU verified deployment options.",
    },
    {
        "id": "llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
        "family": "Llama",
        "model_garden_id": "meta/llama3_3@llama-3.3-70b-instruct",
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "params_b": 70.6,
        "active_params_b": 70.6,
        "max_context": 131072,
        "supports_thinking": False,
        "supports_json": True,
        "tasks": ["reasoning", "coding", "summarization", "extraction"],
        "quality_tier": "frontier",
        "summary": "High-accuracy 70B dense open model for complex enterprise reasoning and coding; requires multi-GPU or FP8 high-VRAM shapes.",
    },
    {
        "id": "llama-3.1-8b-instruct",
        "name": "Llama 3.1 8B",
        "family": "Llama",
        "model_garden_id": "meta/llama3_1@llama-3.1-8b-instruct",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "params_b": 8.0,
        "active_params_b": 8.0,
        "max_context": 131072,
        "supports_thinking": False,
        "supports_json": True,
        "tasks": ["chat", "extraction", "summarization"],
        "quality_tier": "compact",
        "summary": "Fast, low-latency 8B model with 128K context; fits on a single L4 GPU or TPU v5e/v6e chip for high-throughput triage.",
    },
    {
        "id": "qwen2.5-32b-instruct",
        "name": "Qwen2.5 32B",
        "family": "Qwen",
        "model_garden_id": "qwen/qwen2_5@qwen2.5-32b-instruct",
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32.5,
        "active_params_b": 32.5,
        "max_context": 131072,
        "supports_thinking": False,
        "supports_json": True,
        "tasks": ["coding", "reasoning", "extraction", "summarization"],
        "quality_tier": "frontier",
        "summary": "Strong 32B dense model for structured JSON schemas, multilingual extraction, and code generation.",
    },
    {
        "id": "deepseek-r1-distill-qwen-32b",
        "name": "DeepSeek-R1 Distill Qwen 32B",
        "family": "DeepSeek",
        "model_garden_id": "deepseek-ai/deepseek-r1@deepseek-r1-distill-qwen-32b",
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "params_b": 32.5,
        "active_params_b": 32.5,
        "max_context": 131072,
        "supports_thinking": True,
        "supports_json": True,
        "tasks": ["reasoning", "coding", "extraction"],
        "quality_tier": "frontier",
        "summary": "Reasoning-specialized 32B distilled model with native chain-of-thought thinking tokens.",
    },
    {
        "id": "mistral-small-24b-instruct",
        "name": "Mistral Small 24B (2501)",
        "family": "Mistral AI",
        "model_garden_id": "mistralai/mistral@mistral-small-24b-instruct-2501",
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "params_b": 24.0,
        "active_params_b": 24.0,
        "max_context": 32768,
        "supports_thinking": False,
        "supports_json": True,
        "tasks": ["chat", "extraction", "coding"],
        "quality_tier": "balanced",
        "summary": "Low-latency 24B model optimized for short-to-medium context (<32K) structured outputs and function calling.",
    },
]

# Serving images for benchmark endpoints (deployment.image; null = the Model Garden verified recipe's image, which
# changes over time). "pinned" is an image this tool's maintainers have run long requests on; `tested` is the date
# of that check (None = not yet tested) and `notes` what was checked. The pre-flight check catches a bad image on
# any run.
SERVING_IMAGES = [
    {"key": "pinned",
     # Pinned by digest: tags such as `gemma4` can be moved to a new build at any time.
     "image": ("us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve"
               "@sha256:3fbc0e08c46e7736145c9effbcc8839682c80b15b413335146f4428fc3523c8a"),
     "tag": "gemma4", "built": "2026-04-02",
     # A CUDA build of vLLM: GPUs only (TPUs need a TPU build). Tested with long requests on these GPUs.
     "accelerator_kinds": ["gpu"],
     "tested_accelerators": ["NVIDIA_RTX_PRO_6000", "NVIDIA_H100_80GB", "NVIDIA_A100_80GB"],
     "label": "Pinned: tested by aiops-dilemma", "tested": "2026-11-09",
     "tested_on": "Gemma 4 26B-A4B FP8, 1x RTX PRO 6000 (g4-standard-48), us-central1, inference timeout 600 s",
     "notes": ("Gemma 4 day-0 vLLM build (tag gemma4, built 2026-04-02). A forced 12,000-token stream completed in "
               "92 s and a ~110K-token prompt in 14 s. The current verified recipe image "
               "(pytorch-vllm-serve:20260713_0916_RC01, built 2026-07-13) cut the same stream at 31 s, even with "
               "the 600 s timeout.")},
]

# Cloud Billing uses city/area names in SKU descriptions.
REGION_NAMES = {
    "us-central1": "Iowa", "us-east1": "South Carolina", "us-east4": "Virginia", "us-east5": "Columbus",
    "us-west1": "Oregon", "us-west4": "Las Vegas", "us-south1": "Dallas",
    "europe-west1": "Belgium", "europe-west2": "London", "europe-west3": "Frankfurt",
    "europe-west4": "Netherlands", "europe-west9": "Paris", "europe-north1": "Finland",
    "asia-east1": "Taiwan", "asia-northeast1": "Tokyo", "asia-northeast3": "Seoul",
    "asia-south1": "Mumbai", "asia-southeast1": "Singapore", "australia-southeast1": "Sydney",
    "me-west1": "Tel Aviv", "northamerica-northeast2": "Toronto",
}

# ------------------------------------------------------------------------------------------- reference benchmarks
# Per-GPU throughput measured by this project on the fixed reference workload (examples/reference-workload.yaml,
# workload "reference-v1": Gemma 4 26B-A4B FP8, ~5K-token prompts incl. a 1,800-token system prompt, ~500-token JSON
# answers, thinking off, nothing over 32K, the pinned serving image). planner.reference_throughput uses them as the
# primary throughput estimate (scaled to other models / output lengths) when there is no benchmark of your own.
# Add an entry from a finished run with `dilemma reference-entry -c runs/<name>/workload.yaml`.
REFERENCE_WORKLOAD = {
    "key": "reference-v1", "model": "Gemma 4 26B-A4B", "model_garden_id": "google/gemma4@gemma-4-26b-a4b-it",
    "precision": "fp8", "avg_input_tokens": 5000, "avg_output_tokens": 500, "system_prompt_tokens": 1800,
    "system_prompt_cache_hit": 0.9, "thinking_share": 0.0, "pct_over_32k": 0.0,
    "summary": "Gemma 4 26B-A4B FP8, ~5K-token prompts, 500-token answers, no thinking",
}
REFERENCE_FIELDS = {   # field -> required
    "accelerator_type": True, "accelerator_count": True, "machine_type": True, "model_garden_id": True,
    "precision": True, "workload": True, "short_prompt_tps_per_chip": True, "measured_at": True, "image": True,
    "notes": False, "source": True,
    # measured averages of the run (the decode adjustment uses them; default: the reference workload's nominal)
    "avg_input_tokens": False, "avg_output_tokens": False, "peak_concurrency": False, "p95_latency_s": False,
}
REFERENCE_BENCHMARKS: list[dict] = [
    # Measured with examples/reference-workload.yaml (dilemma auto + dilemma reference-entry). The 128 level is
    # capped by the recipe's --max-num-seqs=128, so peaks at c128 are this setup's ceiling, not the GPU's.
    {"accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 1, "machine_type": "a3-highgpu-1g", "model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "precision": "fp8", "workload": "reference-v1", "short_prompt_tps_per_chip": 22259, "measured_at": "2026-11-09", "image": "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve@sha256:3fbc0e08c46e7736145c9effbcc8839682c80b15b413335146f4428fc3523c8a", "notes": "peak c128, p95 41 s, 300 ok / 0 errors; image args from deployment.image", "source": "measured", "avg_input_tokens": 5190, "avg_output_tokens": 444, "peak_concurrency": 128, "p95_latency_s": 40.6},
    {"accelerator_type": "NVIDIA_RTX_PRO_6000", "accelerator_count": 1, "machine_type": "g4-standard-48", "model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "precision": "fp8", "workload": "reference-v1", "short_prompt_tps_per_chip": 16984, "measured_at": "2026-11-09", "image": "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve@sha256:3fbc0e08c46e7736145c9effbcc8839682c80b15b413335146f4428fc3523c8a", "notes": "peak c128, p95 51 s, 300 ok / 0 errors; image args from deployment.image", "source": "measured", "avg_input_tokens": 5240, "avg_output_tokens": 451, "peak_concurrency": 128, "p95_latency_s": 50.6},
    {"accelerator_type": "NVIDIA_A100_80GB", "accelerator_count": 1, "machine_type": "a2-ultragpu-1g", "model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "precision": "fp8", "workload": "reference-v1", "short_prompt_tps_per_chip": 10372, "measured_at": "2026-11-09", "image": "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve@sha256:3fbc0e08c46e7736145c9effbcc8839682c80b15b413335146f4428fc3523c8a", "notes": "peak c128, p95 102 s, 300 ok / 0 errors; image args from deployment.image (args adapted from the 1x NVIDIA_RTX_PRO_6000 recipe)", "source": "measured", "avg_input_tokens": 5245, "avg_output_tokens": 458, "peak_concurrency": 128, "p95_latency_s": 101.6},
]


def validate_reference(entry: dict) -> dict:
    """A clean copy of one REFERENCE_BENCHMARKS entry; raises ValueError naming what's wrong."""
    missing = [k for k, req in REFERENCE_FIELDS.items() if req and entry.get(k) in (None, "")]
    if missing:
        raise ValueError(f"reference benchmark entry is missing {', '.join(missing)}")
    if entry["accelerator_type"] not in GPUS:
        raise ValueError(f"unknown accelerator_type {entry['accelerator_type']}")
    if entry["source"] != "measured":
        raise ValueError("reference benchmarks must be measured (source: measured)")
    out = {k: entry.get(k) for k in REFERENCE_FIELDS}
    out["accelerator_count"] = int(out["accelerator_count"])
    out["short_prompt_tps_per_chip"] = float(out["short_prompt_tps_per_chip"])
    if out["short_prompt_tps_per_chip"] <= 0:
        raise ValueError("short_prompt_tps_per_chip must be > 0")
    return out


def reference_benchmarks(extra: list[dict] | None = None) -> list[dict]:
    """Validated reference entries: REFERENCE_BENCHMARKS + `extra` + a JSON list in $DILEMMA_REFERENCE_BENCHMARKS
    (a file path; handy before an entry is added here). Invalid entries are skipped with a warning."""
    import json
    import os
    items = list(REFERENCE_BENCHMARKS) + list(extra or [])
    path = os.environ.get("DILEMMA_REFERENCE_BENCHMARKS", "").strip()
    if path and os.path.exists(path):
        try:
            data = json.loads(open(path).read())
            items += data if isinstance(data, list) else [data]
        except ValueError as e:
            print(f"warning: {path}: {e}")
    out = []
    for e in items:
        try:
            out.append(validate_reference(e))
        except (ValueError, TypeError, KeyError) as err:
            print(f"warning: skipped reference benchmark {e.get('accelerator_type') if isinstance(e, dict) else e}: {err}")
    return out
