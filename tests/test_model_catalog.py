"""Curated model catalog (docs/ui-design-plan.md, Round 2 "Model catalog"): curation, precision, cost-only scoring."""

import json

import pytest

from aiops_dilemma import catalog, config, deploy, modelinfo, planner, recommender, server

# A slice of a real Model Garden listing: text LLMs, base checkpoints, pre-quantized duplicates,
# image / video / vision models, embeddings, specialists and older generations.
GARDEN_IDS = [
    "google/gemma4@gemma-4-26b-a4b", "google/gemma4@gemma-4-26b-a4b-it", "google/gemma4@gemma-4-31b",
    "google/gemma4@gemma-4-31b-it", "google/gemma4@gemma-4-12b", "google/gemma4@gemma-4-12b-it",
    "google/gemma4@gemma-4-e4b", "google/gemma4@gemma-4-e4b-it", "google/gemma4@gemma-4-e2b-it",
    "google/gemma3@gemma-3-12b-it", "google/gemma3@gemma-3-12b-pt", "google/gemma3@gemma-3-12b-it-quantized.w8a8",
    "google/gemma3@gemma-3-27b-it", "google/gemma3@gemma-3-27b-pt",
    "google/gemma2@gemma-2-27b", "google/gemma2@gemma-2-27b-it",
    "google/gemma@gemma-7b", "google/gemma@gemma-7b-it", "google/gemma@gemma-7b-it-gg-hf",
    "google/paligemma@paligemma2-10b-mix-448", "google/paligemma@paligemma2-28b-pt-896",
    "google/medgemma@medgemma-27b-text-it", "google/medgemma@medgemma-4b-it-dicom", "google/medgemma@medgemma-4b-pt",
    "google/translategemma@translategemma-12b-it", "google/diffusiongemma@diffusiongemma-26b-a4b-it",
    "qwen/qwen3@qwen3-32b", "qwen/qwen3@qwen3-32b-fp8", "qwen/qwen3@qwen3-14b-base", "qwen/qwen3@qwen3-14b",
    "qwen/qwen3@qwen3-30b-a3b", "qwen/qwen3@qwen3-30b-a3b-fp8", "qwen/qwen3@qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-vl@qwen3-vl-32b-instruct", "qwen/qwen3-vl@qwen3-vl-32b-instruct-fp8",
    "qwen/qwen3-embedding@qwen3-embedding-8b", "qwen/qwen2@qwen2.5-32b-instruct", "qwen/qwen2@qwen2.5-32b-instruct-awq",
    "qwen/qwen3-coder@qwen3-coder-30b-a3b-instruct", "qwen/qwen3-coder@qwen3-coder-30b-a3b-instruct-fp8",
    "meta/llama4@llama-4-maverick-17b-128e", "meta/llama4@llama-4-maverick-17b-128e-instruct",
    "meta/llama4@llama-4-maverick-17b-128e-instruct-fp8", "meta/llama3-3@llama-3.3-70b-instruct",
    "meta/llama3_1@llama-3.1-405b-fp8", "meta/llama3_1@llama-3.1-405b-instruct-fp8",
    "meta/llama3-2@llama-3.2-90b-vision-instruct", "meta/llama2@llama-2-13b", "meta/llama2@llama-2-13b-chat",
    "meta/llama-2-quantized@llama-2-13b-chat-awq", "meta/llama-2-quantized@llama-2-13b-chat-gptq",
    "meta/codellama-7b-hf@codellama-13b-hf", "meta/codellama-7b-hf@codellama-13b-instruct-hf",
    "lmsys/lmsys-vicuna-7b@vicuna-13b-v1.5", "mistral-ai/mistral@mistral-7b-instruct-v0.1",
    "mistral-ai/mistral@mistral-7b-v0.1", "mistral-ai/mixtral@mixtral-8x7b-instruct-v0.1",
    "mistralai/ministral-3@ministral-3-14b-instruct-2512", "mistralai/ministral-3@ministral-3-14b-base-2512",
    "deepseek-ai/deepseek-v3@deepseek-v3", "openai/gpt-oss@gpt-oss-20b",
    "nvidia/nemotron-3@nemotron-3-nano-30b-a3b-bf16", "nvidia/nemotron-3@nemotron-3-nano-30b-a3b-fp8",
    "salesforce/blip2-opt-2.7-b@blip2-opt-2.7b", "liuhaotian/pytorch-llava@llava-v1.6-34b-hf",
    "black-forest-labs/flux-2-klein-4b@flux.2-klein-4b", "black-forest-labs/flux-2-klein-4b@flux.2-klein-4b-fp8",
    "wan-ai/wan2-2@wan2.2-t2v-a14b-diffusers", "contextualai/rerank-v2-instruct@6b", "qodo/qodo-embed-1-7b-v1@001",
]


@pytest.fixture(scope="module")
def curated():
    return {m["id"]: m for m in modelinfo.curate_models(GARDEN_IDS)}


def test_curation_keeps_deployable_text_llms_only(curated):
    mg_ids = {m["model_garden_id"] for m in curated.values()}
    blob = " ".join(mg_ids).lower()
    for bad in ("paligemma", "blip", "llava", "flux", "wan2", "qwen3-vl", "embed", "rerank", "diffusion", "vision",
                "dicom", "qodo"):
        assert bad not in blob, bad
    # Base / pretrained checkpoints with an instruct sibling are dropped; instruct and hybrid models stay.
    for base in ("gemma-4-26b-a4b", "gemma-4-31b", "gemma-4-12b", "gemma-3-12b-pt", "gemma-2-27b", "gemma-7b",
                 "qwen3-14b-base", "llama-2-13b", "codellama-13b-hf", "mistral-7b-v0.1", "medgemma-4b-pt",
                 "llama-4-maverick-17b-128e", "ministral-3-14b-base-2512", "llama-3.1-405b"):
        assert base not in curated, base
    for keep in ("gemma-4-26b-a4b-it", "qwen3-32b", "qwen3-14b", "qwen3-30b-a3b-instruct-2507", "gpt-oss-20b",
                 "deepseek-v3", "nemotron-3-nano-30b-a3b", "medgemma-27b-text-it"):
        assert keep in curated, keep
    assert "gemma-7b-it" in curated and "gemma-7b-it-gg-hf" not in curated
    # Seeds include the newest Gemma 4 sizes, marked verified when Model Garden lists them.
    for gid in ("gemma-4-12b-it", "gemma-4-e4b-it", "gemma-4-e2b-it"):
        assert curated[gid]["verified_in_model_garden"] is True


def test_quantized_variants_fold_into_precisions(curated):
    assert not any(k.endswith(("-fp8", "-awq", "-gptq", "w8a8", "-nvfp4")) for k in curated)
    assert curated["qwen3-32b"]["prequantized"] == {"fp8": "qwen/qwen3@qwen3-32b-fp8"}
    assert curated["llama-2-13b-chat"]["prequantized"] == {
        "awq": "meta/llama-2-quantized@llama-2-13b-chat-awq", "gptq": "meta/llama-2-quantized@llama-2-13b-chat-gptq"}
    assert curated["gemma-3-12b-it"]["prequantized"] == {"w8a8": "google/gemma3@gemma-3-12b-it-quantized.w8a8"}
    assert curated["qwen2.5-32b-instruct"]["prequantized"]["awq"] == "qwen/qwen2@qwen2.5-32b-instruct-awq"
    assert curated["llama-4-maverick-17b-128e-instruct"]["prequantized"]["fp8"].endswith("-instruct-fp8")
    assert curated["nemotron-3-nano-30b-a3b"]["model_garden_id"].endswith("-bf16")
    # Every model offers BF16 and FP8 (FP8 on the fly unless pre-quantized) ...
    for m in curated.values():
        if not m.get("fp8_only") and m.get("native_precision") not in ("int4", "mxfp4"):
            assert m["precisions"][:2] == ["bf16", "fp8"], m["id"]
    # ... models published in MXFP4 (gpt-oss) offer only that ...
    assert curated["gpt-oss-20b"]["precisions"] == ["mxfp4"] and curated["gpt-oss-20b"]["native_precision"] == "mxfp4"
    assert curated["gemma-4-26b-a4b-it"]["prequantized"] == {}
    # ... except models Model Garden only ships as an FP8 checkpoint.
    l405 = curated["llama-3.1-405b-instruct"]
    assert l405["fp8_only"] and l405["precisions"] == ["fp8"] and l405["model_garden_id"].endswith("-fp8")


def test_curated_fields_legacy_specialty_families(curated):
    keys = {"id", "name", "family", "params_b", "active_params_b", "max_context", "generation", "released", "legacy",
            "specialty", "precisions", "prequantized", "verified_in_model_garden", "model_garden_id", "hf_id",
            "supports_thinking", "supports_json", "summary"}
    for m in curated.values():
        assert keys <= set(m), (m["id"], keys - set(m))
    for old in ("llama-2-13b-chat", "gemma-2-27b-it", "gemma-7b-it", "vicuna-13b-v1.5", "mistral-7b-instruct-v0.1",
                "mixtral-8x7b-instruct-v0.1", "codellama-13b-instruct-hf"):
        assert curated[old]["legacy"] is True, old
    for new in ("gemma-4-26b-a4b-it", "qwen3-32b", "llama-3.3-70b-instruct", "gpt-oss-20b"):
        assert curated[new]["legacy"] is False, new
    assert curated["medgemma-27b-text-it"]["specialty"] == "medical"
    assert curated["translategemma-12b-it"]["specialty"] == "translation"
    assert curated["codellama-13b-instruct-hf"]["specialty"] == "code"
    assert curated["qwen3-coder-30b-a3b-instruct"]["specialty"] == "code"
    assert curated["gemma-4-26b-a4b-it"]["specialty"] is None
    # Display names, families, generations and sizes.
    assert curated["gemma-4-26b-a4b-it"]["name"] == "Gemma 4 26B-A4B"
    assert curated["qwen3-32b"]["name"] == "Qwen3 32B"
    assert curated["llama-4-maverick-17b-128e-instruct"]["name"] == "Llama 4 Maverick 17B-128E"
    assert curated["gemma-4-26b-a4b-it"]["generation"] == 4 and curated["gemma-4-26b-a4b-it"]["released"] == "2026-04"
    assert curated["mixtral-8x7b-instruct-v0.1"]["params_b"] == 46.7
    assert curated["mixtral-8x7b-instruct-v0.1"]["active_params_b"] == 12.9
    assert curated["llama-4-maverick-17b-128e-instruct"]["active_params_b"] == 17.0
    assert curated["deepseek-v3"]["params_b"] == 671.0
    fams = {f["name"]: f for f in modelinfo.family_counts(list(curated.values()))}
    assert {"Gemma", "Qwen", "Llama", "Mistral AI", "DeepSeek"} <= set(fams)
    assert not {"Google", "Meta", "Mistral-Ai", "Mistralai", "Gemma 4"} & set(fams)
    assert fams["Gemma"]["count"] == sum(m["family"] == "Gemma" for m in curated.values())
    assert fams["Llama"]["legacy"] >= 2


def test_discover_models_recurates_old_cache(tmp_path, monkeypatch):
    """A format-1 cache (uncurated) is re-curated from its Model Garden ids; nothing is pinned to disk offline."""
    monkeypatch.setattr(modelinfo, "CACHE", tmp_path)
    monkeypatch.setattr(modelinfo, "_MEM", {})
    old = {"sources": ["seed_catalog", "vertex_model_garden"],
           "models": [{"model_garden_id": g, "dynamic_source": "vertex_model_garden"} for g in GARDEN_IDS]}
    (tmp_path / modelinfo.LEGACY_CACHE).write_text(json.dumps(old))
    d = modelinfo.discover_models(None)
    assert d["format"] == modelinfo.CURATED_FORMAT
    assert "vertex_model_garden_cached" in d["sources"]
    assert {f["name"] for f in d["families"]} >= {"Gemma", "Qwen", "Llama"}
    assert not any("paligemma" in m["model_garden_id"] for m in d["models"])
    assert not (tmp_path / modelinfo.CURATED_CACHE).exists()
    # A live listing is cached as raw ids and curated on load.
    monkeypatch.setattr(modelinfo, "_list_model_garden", lambda p, r: (GARDEN_IDS, {}))
    d2 = modelinfo.discover_models("some-project", refresh=True)
    raw = json.loads((tmp_path / modelinfo.CURATED_CACHE).read_text())
    assert raw["format"] == modelinfo.CURATED_FORMAT and raw["garden_ids"] == GARDEN_IDS and "models" not in raw
    assert modelinfo.discover_models("some-project")["models"] == d2["models"]


def test_catalog_endpoint_passes_families(monkeypatch):
    payload = {"models": modelinfo.curate_models(GARDEN_IDS), "sources": ["seed_catalog"], "updated_at": 1}
    payload["families"] = modelinfo.family_counts(payload["models"])
    monkeypatch.setattr(server.modelinfo, "discover_models", lambda *a, **k: payload)
    monkeypatch.setattr(server.auth, "status", lambda: {"project": "my-gcp-project"})
    monkeypatch.setattr(server.pricing, "gemini_prices", lambda *a, **k: catalog.GEMINI_MODELS)
    cat = server._build_catalog("my-gcp-project", "us-central1")
    assert cat["families"] == payload["families"] and cat["families"][0]["count"] > 0


# Precision -----------------------------------------------------------------------------------------------------------

def _plan_cfg(model: dict) -> dict:
    return config.normalize({"name": "p", "project": "my-gcp-project", "region": "us-central1", "model": model})


def test_fp8_uses_prequantized_checkpoint_else_on_the_fly(curated, monkeypatch):
    monkeypatch.setattr(modelinfo, "discover_models", lambda *a, **k: {"models": list(curated.values())})
    q = {"model_garden_id": "qwen/qwen3@qwen3-32b", "hf_id": "Qwen/qwen3-32b"}
    dm = planner.deployment_model(_plan_cfg({**q, "precision": "fp8"}))
    assert dm["quantization"] == "prequantized" and dm["model_garden_id"] == "qwen/qwen3@qwen3-32b-fp8"
    assert dm["base_model_garden_id"] == q["model_garden_id"] and dm["hf_id"] == "Qwen/qwen3-32b-fp8"
    bf = planner.deployment_model(_plan_cfg({**q, "precision": "bf16"}))
    assert bf["model_garden_id"] == q["model_garden_id"] and bf["quantization"] is None
    g = planner.deployment_model(_plan_cfg({"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it",
                                            "hf_id": "google/gemma-4-26b-a4b-it", "precision": "fp8"}))
    assert g["quantization"] == "on_the_fly" and g["model_garden_id"] == "google/gemma4@gemma-4-26b-a4b-it"
    # The plan carries it to the deployer; weights are sized at the chosen precision either way.
    p8 = planner.plan(_plan_cfg({**q, "precision": "fp8"}))
    p16 = planner.plan(_plan_cfg({**q, "precision": "bf16"}))
    assert p8["deployment_model"]["quantization"] == "prequantized"
    assert p8["weights_gb"] == pytest.approx(p16["weights_gb"] / 2, abs=0.1)


def test_deploy_args_quantization_flag():
    cfg = {"model": {"model_garden_id": "google/gemma4@gemma-4-26b-a4b-it", "precision": "fp8"},
           "deployment": {"extra_args": []}}
    ch = {"accelerator_count": 1}
    _, args, _, _ = deploy._args(cfg, ch, None)
    assert "--quantization=fp8" in args
    cfg["model"]["prequantized_checkpoint"] = True
    _, args, _, _ = deploy._args(cfg, ch, None)
    assert not any(a.startswith("--quantization") for a in args)


# Recommender: always cost-oriented ----------------------------------------------------------------------------------

def _m(mid, params, active=None, **kw):
    return modelinfo.annotate({"id": mid, "model_garden_id": f"x/y@{mid}", "hf_id": f"x/{mid}", "params_b": params,
                               "active_params_b": active or params, "max_context": 262144, **kw})


def test_priority_is_ignored_and_cheapest_qualifying_model_wins(curated):
    req = {"avg_input_tokens": 7000, "pct_over_32k": 2.0, "pct_over_128k": 0.5}
    runs = [recommender.recommend({"task_type": "extraction", **({"priority": p} if p else {})}, req,
                                  models=list(curated.values()))["recommendations"]
            for p in (None, "cost_efficiency", "balanced", "max_quality")]
    ranks = [[(m["id"], m["match_score"]) for m in r] for r in runs]
    assert all(r == ranks[0] for r in ranks)
    assert all("priority" not in m["score_breakdown"] and "cost" in m["score_breakdown"] for m in runs[0])
    # Synthetic catalog: same generation and context, so only cost and the quality floor separate them.
    models = [_m("tiny-2b", 2.0, released="2026-01"), _m("dense-70b", 70.0, released="2026-01"),
              _m("moe-30b-a3b", 30.0, 3.0, released="2026-01"), _m("dense-14b", 14.0, released="2026-01")]
    recs = recommender.recommend({"task_type": "extraction"}, {}, models=models)["recommendations"]
    order = [m["id"] for m in recs]
    assert order[0] == "moe-30b-a3b"                       # cheapest to serve that is big enough
    assert order.index("dense-14b") < order.index("dense-70b")
    assert order[-1] == "tiny-2b"                          # cheapest, but below the quality floor for extraction


def test_legacy_and_specialty_models_hidden_unless_selected_or_matching(curated):
    models = list(curated.values())
    ids = lambda r: {m["id"] for m in r["recommendations"]}  # noqa: E731
    r = recommender.recommend({"task_type": "extraction"}, {}, models=models)
    assert not any(curated[i]["legacy"] or curated[i]["specialty"] for i in ids(r))
    assert "gemma-4-26b-a4b-it" in ids(r)
    # A code specialist shows up for coding; legacy stays hidden unless it is the selection.
    rc = recommender.recommend({"task_type": "coding"}, {}, models=models)
    assert "qwen3-coder-30b-a3b-instruct" in ids(rc) and "codellama-13b-instruct-hf" not in ids(rc)
    old = curated["llama-2-13b-chat"]
    rs = recommender.recommend({"task_type": "extraction"}, {}, models=models,
                               selected_model={"model_garden_id": old["model_garden_id"]})
    sel = [m for m in rs["recommendations"] if m["is_selected"]]
    assert [m["id"] for m in sel] == ["llama-2-13b-chat"]
    assert sel[0]["fits_longest_prompt"] is False and sel[0]["blocking"]   # 4K context < 32K short-prompt bucket
    assert sel[0]["recommended_precision"] in ("bf16", "fp8")
    # Precision reason says how FP8 is done.
    big = next(m for m in r["recommendations"] if m["recommended_precision"] == "fp8" and m["prequantized"].get("fp8"))
    assert "pre-quantized" in big["precision_reason"]


def test_old_config_with_priority_still_loads():
    cfg = config.normalize({"use_case": {"task_type": "extraction", "priority": "max_quality"}})
    res = recommender.recommend(cfg["use_case"], cfg["requests"], models=[_m("m-27b", 27.0)])
    assert res["recommendations"][0]["id"] == "m-27b"
