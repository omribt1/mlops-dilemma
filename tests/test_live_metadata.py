"""Live metadata: Hugging Face model metadata (hfmeta + modelinfo.annotate), Compute Engine accelerator listing
(hardware + planner), and the faster auth status. No network: HTTP is mocked.

The configs below are minimal hand-written excerpts of public config.json files (only the numeric fields the
formulas read) with the Hub's safetensors totals; published active sizes are the model cards' figures."""

import json
import time

import pytest

from aiops_dilemma import auth, catalog, hardware, hfmeta, modelinfo, planner

# ------------------------------------------------------------------------------------------------- fixtures
GEMMA4_26B = {  # google/gemma-4-26B-A4B-it (multimodal wrapper: text_config + vision/audio towers)
    "dtype": "bfloat16", "vision_config": {"hidden_size": 1152}, "audio_config": {"hidden_size": 1024},
    "text_config": {"hidden_size": 2816, "intermediate_size": 2112, "moe_intermediate_size": 704,
                    "num_hidden_layers": 30, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
                    "vocab_size": 262144, "tie_word_embeddings": True, "num_experts": 128, "top_k_experts": 8,
                    "enable_moe_block": True, "max_position_embeddings": 262144, "model_type": "gemma4_text"}}
QWEN3_30B = {"torch_dtype": "bfloat16", "hidden_size": 2048, "intermediate_size": 6144, "moe_intermediate_size": 768,
             "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
             "vocab_size": 151936, "tie_word_embeddings": False, "num_experts": 128, "num_experts_per_tok": 8,
             "decoder_sparse_step": 1, "mlp_only_layers": [], "max_position_embeddings": 40960,
             "model_type": "qwen3_moe"}
_MLA = {"hidden_size": 7168, "intermediate_size": 18432, "moe_intermediate_size": 2048, "num_hidden_layers": 61,
        "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "num_experts_per_tok": 8, "n_shared_experts": 1, "moe_layer_freq": 1, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16"}
KIMI_K2 = {**_MLA, "num_attention_heads": 64, "num_key_value_heads": 64, "vocab_size": 163840, "n_routed_experts": 384,
           "first_k_dense_replace": 1, "max_position_embeddings": 131072, "model_type": "kimi_k2",
           "rope_scaling": {"factor": 32.0, "original_max_position_embeddings": 4096, "type": "yarn"},
           "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}}
DEEPSEEK_V3 = {**_MLA, "num_attention_heads": 128, "num_key_value_heads": 128, "vocab_size": 129280,
               "n_routed_experts": 256, "first_k_dense_replace": 3, "num_nextn_predict_layers": 1,
               "max_position_embeddings": 163840, "model_type": "deepseek_v3",
               "rope_scaling": {"factor": 40, "original_max_position_embeddings": 4096, "type": "yarn"},
               "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}
MIXTRAL = {"torch_dtype": "bfloat16", "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
           "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000, "tie_word_embeddings": False,
           "num_local_experts": 8, "num_experts_per_tok": 2, "max_position_embeddings": 32768, "model_type": "mixtral"}
GPT_OSS_120B = {"hidden_size": 2880, "intermediate_size": 2880, "num_hidden_layers": 36, "num_attention_heads": 64,
                "num_key_value_heads": 8, "head_dim": 64, "vocab_size": 201088, "tie_word_embeddings": False,
                "num_local_experts": 128, "experts_per_token": 4, "num_experts_per_tok": 4,
                "max_position_embeddings": 131072, "model_type": "gpt_oss",
                "quantization_config": {"quant_method": "mxfp4"}}
QWEN25_32B = {"torch_dtype": "bfloat16", "hidden_size": 5120, "intermediate_size": 27648, "num_hidden_layers": 64,
              "num_attention_heads": 40, "num_key_value_heads": 8, "vocab_size": 152064, "tie_word_embeddings": False,
              "max_position_embeddings": 32768, "model_type": "qwen2"}

# (config, safetensors total, published active billions)
PUBLISHED = {
    "google/gemma-4-26B-A4B-it": (GEMMA4_26B, 25_805_936_206, 4.0),
    "Qwen/Qwen3-30B-A3B": (QWEN3_30B, 30_532_122_624, 3.3),
    "moonshotai/Kimi-K2-Instruct": (KIMI_K2, 1_026_408_232_448, 32.0),
    "deepseek-ai/DeepSeek-V3": (DEEPSEEK_V3, 684_489_845_504, 37.0),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (MIXTRAL, 46_702_792_704, 12.9),
    "openai/gpt-oss-120b": (GPT_OSS_120B, 116_829_156_672, 5.1),
}


def _rec(hf_id, cfg, total, created="2025-04-29T00:00:00.000Z", params=None, config_status=200, gated=False,
         license_="apache-2.0", fetched_at=None):
    return {"format": hfmeta.FORMAT, "hf_id": hf_id, "fetched_at": fetched_at or int(time.time()), "status": "ok",
            "config_status": config_status, "config": cfg,
            "api": {"id": hf_id, "createdAt": created, "lastModified": "2026-01-01T00:00:00.000Z", "gated": gated,
                    "safetensors": {"parameters": params or {"BF16": total}, "total": total},
                    "cardData": {"license": license_}, "config": {}}}


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Live metadata enabled against an empty temp cache (no network: fetches must be mocked by the test)."""
    monkeypatch.delenv("DILEMMA_NO_LIVE_METADATA", raising=False)
    monkeypatch.setenv("DILEMMA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(hfmeta, "_MEM", {})
    monkeypatch.setattr(hardware, "_MEM", {})
    monkeypatch.setattr(hfmeta, "_get", lambda *a, **k: pytest.fail("no network in tests"))
    monkeypatch.setattr(modelinfo, "_kick_hf_refresh", lambda *a, **k: None)
    return tmp_path


# ------------------------------------------------------------------------------------------- HF parsing
@pytest.mark.parametrize("hf_id", list(PUBLISHED))
def test_moe_active_params_match_published_sizes(hf_id):
    cfg, total, published = PUBLISHED[hf_id]
    meta = hfmeta.parse(_rec(hf_id, cfg, total))
    act = meta["fields"]["active_params_b"]
    assert abs(act - published) / published <= 0.15, (hf_id, act, published)
    assert meta["fields"]["params_b"] == round(total / 1e9, 2)
    assert meta["fields"]["moe"]["experts"] > meta["fields"]["moe"]["experts_per_token"]


def test_dense_model_active_equals_total_and_annotate_labels_hf(live):
    meta = hfmeta.parse(_rec("Qwen/Qwen2.5-32B-Instruct", QWEN25_32B, 32_763_876_352))
    assert meta["fields"]["dense"] and "active_params_b" not in meta["fields"]
    hfmeta._save_raw("Qwen/Qwen2.5-32B-Instruct", _rec("Qwen/Qwen2.5-32B-Instruct", QWEN25_32B, 32_763_876_352,
                                                        created="2024-09-17T00:00:00.000Z"))
    m = modelinfo.annotate({"id": "qwen2.5-32b-instruct", "model_garden_id": "qwen/qwen2@qwen2.5-32b-instruct",
                            "hf_id": "Qwen/Qwen2.5-32B-Instruct"})
    assert m["params_b"] == 32.76 and m["active_params_b"] == 32.76
    assert m["max_context"] == 32768            # the config's declared window (the old rule said 128K)
    assert m["released"] == "2024-09" and m["license"] == "apache-2.0"
    assert {m["field_sources"][k] for k in ("params_b", "active_params_b", "max_context", "released")} == {"hf"}
    assert m["field_sources"]["family"] == "rules"
    assert m["data_source"]["kind"] == "hf" and m["data_source"]["label"].startswith("Hugging Face, updated ")


def test_config_total_mismatch_falls_back_to_rules():
    bad = {**MIXTRAL, "vocab_size": 1_200_000}      # config total ~20% above the checkpoint: formula not trusted
    meta = hfmeta.parse(_rec("x/mixtral-like", bad, 46_702_792_704))
    assert "active_params_b" not in meta["fields"]
    assert any("built-in estimate" in n for n in meta["notes"])


@pytest.mark.parametrize("qc,expected", [
    ({"quant_method": "compressed-tensors", "format": "pack-quantized",
      "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int", "group_size": 32}}}}, "int4"),
    ({"quant_method": "compressed-tensors", "format": "float-quantized",
      "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}}}, "fp8"),
    ({"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
      "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}}}}, "nvfp4"),
    ({"quant_method": "fp8", "fmt": "e4m3"}, "fp8"),
    ({"quant_method": "awq", "bits": 4}, "int4"),
    ({"quant_method": "gptq", "bits": 4}, "int4"),
    ({"quant_method": "gptq", "bits": 8}, "int8"),
    ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, "nvfp4"),
    ({"quant_method": "modelopt", "quant_algo": "FP8"}, "fp8"),
    ({"quant_method": "mxfp4"}, "mxfp4"),
])
def test_quantization_detection(qc, expected):
    assert hfmeta.detect_precision({"quantization_config": qc, "torch_dtype": "bfloat16"}) == expected


def test_precision_from_dtype_and_safetensors():
    assert hfmeta.detect_precision({"torch_dtype": "bfloat16"}) == "bf16"
    assert hfmeta.detect_precision({"text_config": {"hidden_size": 8, "dtype": "float16"}}) == "bf16"
    assert hfmeta.detect_precision({}, {}, {"F8_E4M3": 9e11, "BF16": 4e9}) == "fp8"
    assert hfmeta.detect_precision({}) is None


def test_native_precision_feeds_precisions(live):
    """INT4 checkpoint -> int4 only; FP8 checkpoint -> fp8 only; NVFP4 isn't offered as INT4; BF16 clears a stale
    rule-table native precision."""
    int4_cfg = {**KIMI_K2, "quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized",
                                                   "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int"}}}}}
    nvfp4_cfg = {**QWEN25_32B, "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    for hf_id, cfg in (("moonshotai/Kimi-K2-Thinking", int4_cfg), ("moonshotai/Kimi-K2-Instruct", KIMI_K2),
                       ("nvidia/some-model-nvfp4", nvfp4_cfg), ("moonshotai/Kimi-K2-Instruct-0905", QWEN25_32B)):
        hfmeta._save_raw(hf_id, _rec(hf_id, cfg, 1_026_408_232_448))
    k_int4 = modelinfo.annotate({"id": "kimi-k2-thinking", "hf_id": "moonshotai/Kimi-K2-Thinking",
                                 "model_garden_id": "moonshotai/kimi-k2@kimi-k2-thinking"})
    assert k_int4["native_precision"] == "int4" and k_int4["precisions"] == ["int4"]
    assert k_int4["field_sources"]["native_precision"] == "hf"
    k_fp8 = modelinfo.annotate({"id": "kimi-k2-instruct", "hf_id": "moonshotai/Kimi-K2-Instruct"})
    assert k_fp8["native_precision"] == "fp8" and k_fp8["precisions"] == ["fp8"]
    nv = modelinfo.annotate({"id": "some-model-32b-nvfp4", "hf_id": "nvidia/some-model-nvfp4"})
    assert nv["checkpoint_precision"] == "nvfp4" and "int4" not in nv["precisions"] and not nv.get("native_precision")
    # The rule table says Kimi K2 Instruct 0905 is FP8; a BF16 checkpoint on the Hub wins.
    bf = modelinfo.annotate({"id": "kimi-k2-instruct-0905", "hf_id": "moonshotai/Kimi-K2-Instruct-0905"})
    assert not bf.get("native_precision") and bf["precisions"][:2] == ["bf16", "fp8"]


@pytest.mark.parametrize("cfg,expected", [
    ({"max_position_embeddings": 131072}, 131072),
    ({"max_position_embeddings": 4096, "rope_scaling": {"factor": 8.0, "original_max_position_embeddings": 4096}},
     32768),
    ({"max_position_embeddings": 163840, "rope_scaling": {"factor": 40, "original_max_position_embeddings": 4096}},
     163840),
    ({"text_config": {"hidden_size": 8, "max_position_embeddings": 262144}}, 262144),
    ({"model_max_length": 8192}, 8192),
    ({"model_max_length": 1e30}, None),
    ({}, None),
])
def test_context_length(cfg, expected):
    assert hfmeta.context_length(cfg) == expected


def test_gated_model_without_config_mixes_hf_and_rules(live):
    """Llama: the Hub API gives size and dates without a token, config.json is gated -> context from the rules."""
    hf_id = "meta-llama/Llama-3.3-70B-Instruct"
    hfmeta._save_raw(hf_id, _rec(hf_id, None, 70_553_706_496, created="2024-11-26T16:08:47.000Z", config_status=401,
                                 gated=True, license_="llama3.3"))
    m = modelinfo.annotate({"id": "llama-3.3-70b-instruct", "hf_id": hf_id,
                            "model_garden_id": "meta/llama3-3@llama-3.3-70b-instruct"})
    assert m["params_b"] == 70.55 and m["field_sources"]["params_b"] == "hf"
    assert m["max_context"] == 131072 and m["field_sources"]["max_context"] == "rules"
    assert m["gated"] is True and m["license"] == "llama3.3"
    assert m["data_source"]["kind"] == "mixed" and "built-in estimate for" in m["data_source"]["label"] and \
        "context" in m["data_source"]["label"]
    assert any("HF_TOKEN" in n for n in m["data_source"]["notes"])


def test_no_hf_metadata_is_labelled_built_in():
    m = modelinfo.annotate({"id": "gemma-4-26b-a4b-it", "hf_id": "google/gemma-4-26b-a4b-it"})
    assert m["data_source"]["kind"] == "rules" and m["data_source"]["label"] == "built-in estimate"
    assert m["field_sources"]["params_b"] == "rules" and m["active_params_b"] == 4.0
    assert set(m["data_source"]["estimated_fields"]) >= {"params_b", "max_context"}


def test_curate_uses_model_garden_hf_ids_and_hf_sizes_unsized_models(live):
    """The listing's MODEL_ID gives the exact HF id; a model without a size in its name is kept when the Hub has
    its size."""
    hfmeta._save_raw("google/gemma-4-26B-A4B-it", _rec("google/gemma-4-26B-A4B-it", GEMMA4_26B, 25_805_936_206,
                                                       created="2026-03-11T21:25:57.000Z"))
    ids = ["google/gemma4@gemma-4-26b-a4b-it", "tencent/tencent-hy3@hy3"]
    hf_map = {"google/gemma4@gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B-it", "tencent/tencent-hy3@hy3": "tencent/Hy3"}
    modelinfo.curate_models(ids, seeds=[], hf_map=hf_map)
    assert "tencent/tencent-hy3@hy3" in modelinfo._LAST_UNSIZED          # no size in the name, nothing cached yet
    hfmeta._save_raw("tencent/Hy3", _rec("tencent/Hy3", QWEN3_30B, 30_532_122_624))
    out = {m["id"]: m for m in modelinfo.curate_models(ids, hf_map=hf_map)}
    g = out["gemma-4-26b-a4b-it"]
    assert g["hf_id"] == "google/gemma-4-26B-A4B-it" and g["params_b"] == 25.81
    assert 3.4 <= g["active_params_b"] <= 4.6 and g["field_sources"]["active_params_b"] == "hf"
    assert g["released"] == "2026-03"
    assert "hy3" in out and out["hy3"]["params_b"] == 30.53 and out["hy3"]["field_sources"]["params_b"] == "hf"
    assert "tencent/tencent-hy3@hy3" not in modelinfo._LAST_UNSIZED


# ------------------------------------------------------------------------------------------- HF cache / TTL
def test_refresh_caches_and_respects_ttl(live, monkeypatch):
    calls = []

    def fake_get(url, timeout=15.0):
        calls.append(url)
        if "/api/models/" in url:
            return 200, {"id": "Qwen/Qwen3-30B-A3B", "createdAt": "2025-04-28T00:00:00Z", "gated": False,
                         "safetensors": {"parameters": {"BF16": 30_532_122_624}, "total": 30_532_122_624},
                         "cardData": {"license": "apache-2.0"}, "config": {"model_type": "qwen3_moe"}}
        if url.endswith("config.json"):
            return 200, {**QWEN3_30B, "chat_template": "x" * 5000}
        return 404, None

    monkeypatch.setattr(hfmeta, "_get", fake_get)
    assert hfmeta.cached("qwen/qwen3-30b-a3b") is None and hfmeta.is_stale("qwen/qwen3-30b-a3b")
    st = hfmeta.refresh(["qwen/qwen3-30b-a3b"])
    assert st == {"requested": 1, "fetched": 1, "ok": 1, "failed": 0} and len(calls) == 2
    raw = json.loads(hfmeta._path("qwen/qwen3-30b-a3b").read_text())
    assert "chat_template" not in raw["config"] and raw["hf_id"] == "Qwen/Qwen3-30B-A3B"
    meta = hfmeta.cached("qwen/qwen3-30b-a3b")
    assert meta["hf_id"] == "Qwen/Qwen3-30B-A3B" and 3.0 < meta["fields"]["active_params_b"] < 3.6
    assert hfmeta.refresh(["qwen/qwen3-30b-a3b"])["fetched"] == 0 and len(calls) == 2      # fresh: no refetch
    raw["fetched_at"] -= hfmeta.TTL_S + 10
    hfmeta._path("qwen/qwen3-30b-a3b").write_text(json.dumps(raw))
    assert hfmeta.is_stale("qwen/qwen3-30b-a3b")
    assert hfmeta.refresh(["qwen/qwen3-30b-a3b"])["fetched"] == 1 and len(calls) == 4      # a week later: refetched


def test_failures_are_cached_for_a_day(live, monkeypatch):
    monkeypatch.setattr(hfmeta, "_get", lambda url, timeout=15.0: (401, None))
    assert hfmeta.refresh(["org/missing"]) == {"requested": 1, "fetched": 1, "ok": 0, "failed": 1}
    assert hfmeta.cached("org/missing") is None and not hfmeta.is_stale("org/missing")
    raw = json.loads(hfmeta._path("org/missing").read_text())
    assert raw["status"] == "not_found"
    raw["fetched_at"] -= hfmeta.FAIL_TTL_S + 10
    hfmeta._path("org/missing").write_text(json.dumps(raw))
    assert hfmeta.is_stale("org/missing")


def test_disabled_env_reads_nothing(live, monkeypatch):
    hfmeta._save_raw("Qwen/Qwen3-30B-A3B", _rec("Qwen/Qwen3-30B-A3B", QWEN3_30B, 30_532_122_624))
    assert hfmeta.cached("Qwen/Qwen3-30B-A3B")
    monkeypatch.setenv("DILEMMA_NO_LIVE_METADATA", "1")
    assert hfmeta.cached("Qwen/Qwen3-30B-A3B") is None
    assert hfmeta.refresh(["Qwen/Qwen3-30B-A3B"], force=True)["fetched"] == 0
    assert not hfmeta.refresh_bg(["Qwen/Qwen3-30B-A3B"], force=True)


def test_inspect_endpoint_uses_hf_metadata(live, monkeypatch):
    monkeypatch.setattr(hfmeta, "fetch", lambda hf_id: _rec("mistralai/Mixtral-8x7B-Instruct-v0.1", MIXTRAL,
                                                            46_702_792_704))
    info = modelinfo.inspect_hf_model("mistralai/mixtral-8x7b-instruct-v0.1")
    assert info["params_b"] == 46.7 and abs(info["active_params_b"] - 12.9) < 0.5 and info["max_context"] == 32768
    assert info["source"] == "hf" and info["hf_id"] == "mistralai/Mixtral-8x7B-Instruct-v0.1"


# ------------------------------------------------------------------------------------------- Compute Engine
def _raw_listing(region="us-east4", with_h100=False):
    z = f"{region}-a"
    accs = [{"name": "nvidia-l4", "zone": z, "max_cards": 8}, {"name": "nvidia-l4-vws", "zone": z, "max_cards": 8},
            {"name": "nvidia-gb200", "zone": z, "max_cards": 4}, {"name": "nvidia-tesla-t4", "zone": z, "max_cards": 4},
            {"name": "ct6e", "zone": z, "max_cards": 8}, {"name": "nvidia-tesla-a100", "zone": z, "max_cards": 16}]
    mts = [("g2-standard-4", "nvidia-l4", 1, 4, 16), ("g2-standard-12", "nvidia-l4", 1, 12, 48),
           ("g2-standard-32", "nvidia-l4", 1, 32, 128), ("g2-standard-24", "nvidia-l4", 2, 24, 96),
           ("g2-standard-48", "nvidia-l4", 4, 48, 192), ("a4x-highgpu-4g", "nvidia-gb200", 4, 140, 884),
           ("ct6e-standard-4t", "ct6e", 4, 180, 720), ("a2-highgpu-1g", "nvidia-tesla-a100", 1, 12, 85),
           ("a2-megagpu-16g", "nvidia-tesla-a100", 16, 96, 1360)]
    if with_h100:
        accs.append({"name": "nvidia-h100-80gb", "zone": z, "max_cards": 8})
        mts.append(("a3-highgpu-1g", "nvidia-h100-80gb", 1, 26, 234))
    return {"format": hardware.FORMAT, "region": region, "fetched_at": 1_762_700_000, "accelerator_types": accs,
            "machine_types": [{"name": n, "zone": z, "accelerator": a, "count": c, "vcpus": v, "memory_gb": m}
                              for n, a, c, v, m in mts]}


def test_app_key_mapping():
    assert hardware.app_key("nvidia-h100-80gb") == "NVIDIA_H100_80GB"
    assert hardware.app_key("nvidia-rtx-pro-6000") == "NVIDIA_RTX_PRO_6000"
    assert hardware.app_key("nvidia-tesla-a100") == "NVIDIA_TESLA_A100"
    assert hardware.app_key("ct5lp") == "TPU_V5E" and hardware.app_key("ct6e") == "TPU_V6E"
    assert hardware.app_key("tpu7x") == "TPU_7X" and hardware.app_key("nvidia-l4-vws") is None


def test_parse_picks_shapes_and_counts():
    p = hardware.parse(_raw_listing())
    acc = p["accelerators"]
    assert acc["NVIDIA_L4"]["machines"][1][0] == "g2-standard-12"        # the fallback catalog's choice wins
    assert acc["NVIDIA_L4"]["machines"][2] == ("g2-standard-24", 24, 96)
    assert 16 not in acc["NVIDIA_TESLA_A100"]["machines"]                 # TP over 8 GPUs isn't planned
    assert acc["NVIDIA_GB200"]["has_specs"] is False and acc["TPU_V6E"]["kind"] == "tpu"
    assert acc["NVIDIA_TESLA_T4"]["machines"] == {}                        # attachable only: no shape
    assert "NVIDIA_L4_VWS" not in acc


def test_region_merge_labels_unrated_and_not_offered(live):
    p = hardware._cache_path("us-east4")
    p.write_text(json.dumps(_raw_listing()))
    rs = hardware.region_shapes("us-east4")
    assert rs["source"] == "compute_api"
    assert set(rs["accelerators"]) == {"NVIDIA_L4", "TPU_V6E", "NVIDIA_TESLA_A100"}
    assert rs["accelerators"]["NVIDIA_L4"]["mem_gb"] == catalog.CHIP_SPECS["NVIDIA_L4"]["mem_gb"]
    assert rs["accelerators"]["TPU_V6E"]["machines"] == {4: ("ct6e-standard-4t", 180, 720)}
    unrated = {u["accelerator_type"]: u for u in rs["unrated"]}
    assert set(unrated) == {"NVIDIA_GB200", "NVIDIA_TESLA_T4"}
    assert "no performance data" in unrated["NVIDIA_GB200"]["note"] and "a4x-highgpu-4g" in unrated["NVIDIA_GB200"]["note"]
    gone = {g["accelerator_type"]: g["note"] for g in rs["not_offered"]}
    assert gone["NVIDIA_H100_80GB"] == "Not offered in us-east4" and "NVIDIA_L4" not in gone
    # no listing for another region: the built-in catalog, nothing marked missing
    fb = hardware.region_shapes("europe-west4")
    assert fb["source"] == "fallback" and set(fb["accelerators"]) == set(catalog.GPUS) and not fb["not_offered"]


def _cfg(region, model=None, precision="bf16"):
    from aiops_dilemma import config
    return config.normalize({"region": region, "project": "my-gcp-project",
                             "model": model or {"model_garden_id": "meta/llama3_1@llama-3.1-8b-instruct",
                                                "hf_id": "meta-llama/Llama-3.1-8B-Instruct", "precision": precision},
                             "traffic": {"peak_input_tps": 1000, "average_input_tps": 500, "floor_input_tps": 100}})


def test_planner_only_proposes_shapes_offered_in_the_region(live, monkeypatch):
    monkeypatch.setattr(planner.pricing, "prices_for", lambda cfg, acc, n: {"vertex": 1.0, "vertex_cud_3y": 0.5})
    hardware._cache_path("us-east4").write_text(json.dumps(_raw_listing()))
    p = planner.plan(_cfg("us-east4"))
    accs = {c["accelerator_type"] for c in p["candidates"]}
    assert accs <= {"NVIDIA_L4", "TPU_V6E", "NVIDIA_TESLA_A100"} and "NVIDIA_H100_80GB" not in accs
    assert "NVIDIA_GB200" not in accs                                    # no performance data: not estimated
    tpu = next((c for c in p["candidates"] if c["accelerator_type"] == "TPU_V6E"), None)
    assert tpu is None or tpu["machine_type"] == "ct6e-standard-4t"      # the live shape, not ct6e-standard-1t
    av = p["availability"]
    assert av["source"] == "compute_api"
    assert any(x["accelerator_type"] == "NVIDIA_H100_80GB" and x["note"] == "Not offered in us-east4"
               for x in av["not_offered"])
    assert any(x["accelerator_type"] == "NVIDIA_GB200" for x in av["unrated"])
    # With H100 in the listing it is proposed again.
    hardware._cache_path("us-east4").write_text(json.dumps(_raw_listing(with_h100=True)))
    p2 = planner.plan(_cfg("us-east4"))
    assert "NVIDIA_H100_80GB" in {c["accelerator_type"] for c in p2["candidates"]}
    # A manually chosen accelerator the region lacks says so.
    cfg = _cfg("us-east4")
    hardware._cache_path("us-east4").write_text(json.dumps(_raw_listing()))
    cfg["deployment"].update(mode="manual", accelerator_type="NVIDIA_B200", accelerator_count=8)
    assert planner.plan(cfg)["chosen"]["region_note"] == "Not offered in us-east4"


def test_fetch_parses_paged_compute_responses(monkeypatch):
    pages = {
        "acceleratorTypes": [{"items": {"zones/us-east4-a": {"acceleratorTypes": [
            {"name": "nvidia-l4", "maximumCardsPerInstance": 8}]}}, "nextPageToken": "p2"},
            {"items": {"zones/us-east4-b": {"acceleratorTypes": [{"name": "nvidia-h100-80gb",
                                                                   "maximumCardsPerInstance": 8}]},
                       "zones/us-west1-a": {"acceleratorTypes": [{"name": "nvidia-b200"}]}}}],
    }
    calls = []

    def fake(url, headers, timeout=60.0):
        calls.append(url)
        if "acceleratorTypes" in url:
            return pages["acceleratorTypes"][1 if "pageToken=p2" in url else 0]
        zone = url.split("/zones/")[1].split("/")[0]
        mt = {"us-east4-a": [{"name": "g2-standard-12", "guestCpus": 12, "memoryMb": 49152,
                              "accelerators": [{"guestAcceleratorType": "nvidia-l4", "guestAcceleratorCount": 1}]}],
              "us-east4-b": [{"name": "a3-highgpu-1g", "guestCpus": 26, "memoryMb": 239616,
                              "accelerators": [{"guestAcceleratorType": "nvidia-h100-80gb",
                                                "guestAcceleratorCount": 1}]}]}[zone]
        return {"items": mt}

    monkeypatch.setattr(hardware, "_get_json", fake)
    raw = hardware.fetch("my-project", "us-east4", headers={})
    assert {a["name"] for a in raw["accelerator_types"]} == {"nvidia-l4", "nvidia-h100-80gb"}   # us-west1 dropped
    assert {(m["name"], m["zone"], m["memory_gb"]) for m in raw["machine_types"]} == {
        ("g2-standard-12", "us-east4-a", 48), ("a3-highgpu-1g", "us-east4-b", 234)}
    p = hardware.parse(raw)
    assert p["accelerators"]["NVIDIA_H100_80GB"]["machines"][1][0] == "a3-highgpu-1g"
    assert sum("machineTypes" in c for c in calls) == 2                  # one call per zone of the region


def test_reference_entries_still_validate_against_chip_specs():
    assert set(catalog.GPUS) == set(catalog.CHIP_SPECS)
    assert all(catalog.GPUS[k]["machines"] == catalog.FALLBACK_MACHINES[k] for k in catalog.GPUS)


# ------------------------------------------------------------------------------------------- auth speed
def test_gcloud_config_read_from_files(tmp_path, monkeypatch):
    (tmp_path / "configurations").mkdir()
    (tmp_path / "active_config").write_text("work\n")
    (tmp_path / "configurations" / "config_work").write_text("[core]\naccount = someone@example.com\n"
                                                             "project = my-gcp-project\n")
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path))
    monkeypatch.setenv("DILEMMA_GCLOUD_CONFIG_FILES", "1")
    for k in ("CLOUDSDK_CORE_PROJECT", "CLOUDSDK_CORE_ACCOUNT", "CLOUDSDK_ACTIVE_CONFIG_NAME", "GOOGLE_CLOUD_PROJECT",
              "GCLOUD_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(auth, "_run", lambda *a, **k: pytest.fail("no gcloud subprocess needed"))
    assert auth.gcloud_config_value("account") == "someone@example.com"
    assert auth.default_project() == "my-gcp-project"


def test_gcloud_calls_are_cached_and_served_stale(monkeypatch):
    n = {"calls": 0}

    def fake_run(cmd, timeout=15):
        n["calls"] += 1
        return 0, f"value-{n['calls']}"

    monkeypatch.setattr(auth, "_run", fake_run)
    assert auth._run_cached(["gcloud", "x"]) == (0, "value-1")
    assert auth._run_cached(["gcloud", "x"]) == (0, "value-1") and n["calls"] == 1
    key = ("gcloud", "x")
    auth._GCLOUD_MEM[key] = (time.time() - auth.GCLOUD_CACHE_TTL_S - 1, (0, "old"))
    assert auth._run_cached(["gcloud", "x"]) == (0, "old")              # stale answer at once, refresh behind
    for _ in range(100):
        if auth._GCLOUD_MEM[key][1] != (0, "old"):
            break
        time.sleep(0.02)
    assert auth._GCLOUD_MEM[key][1] == (0, "value-2")
    auth._invalidate_caches()
    assert key not in auth._GCLOUD_MEM


def test_status_repeated_calls_are_fast(monkeypatch):
    slow = {"n": 0}

    def fake_run(cmd, timeout=15):
        slow["n"] += 1
        time.sleep(0.3)
        return 0, "x@example.com" if "account" in cmd else ("my-gcp-project" if "project" in cmd else "tok")

    monkeypatch.setattr(auth, "_run", fake_run)
    monkeypatch.setattr(auth, "_adc", lambda: (_ for _ in ()).throw(RuntimeError("no adc")))
    for k in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    t = time.time()
    first = auth.status()
    t1 = time.time() - t
    t = time.time()
    second = auth.status()
    assert time.time() - t < 0.1 and t1 < 0.85          # concurrent first call (3 x 0.3 s serial would be 0.9 s)
    assert first["account"] == second["account"] == "x@example.com" and slow["n"] == 3


def test_per_layer_lists_and_bad_configs_never_break_the_catalog(monkeypatch):
    """Gemma 3n lists intermediate_size per layer; any unparseable config must fall back, not raise."""
    from aiops_dilemma import hfmeta
    cfg = {"model_type": "gemma3n", "hidden_size": 2048, "num_hidden_layers": 30,
           "intermediate_size": [8192] * 10 + [16384] * 20, "vocab_size": 262144}
    assert hfmeta.config_params(cfg)["total"] > 0
    monkeypatch.setattr(hfmeta, "_load_raw", lambda hf_id: {"api": {}, "config": {"num_hidden_layers": "x"}})
    def boom(raw):
        raise TypeError("unexpected config shape")
    monkeypatch.setattr(hfmeta, "parse", boom)
    assert hfmeta._safe_parse("org/model") is None
