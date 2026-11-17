"""Deploy the model to a Agent Platform dedicated endpoint (Model Garden vLLM container), and tear it down.

Uses the Model Garden verified container for the chosen machine when there is one, otherwise the
generic vLLM serving image. Custom container args drop Model Garden's routes, so they are set
explicitly — without them Agent Platform health-checks a path vLLM doesn't serve and the deploy never
becomes ready. Benchmark deployments use one replica (min = max).

A custom serving container (your image, command, arguments, environment, routes, shared memory, a Python patch)
is layered on top by serving.container_spec; see serving.py.
"""

import json
import re
import time
from pathlib import Path

from aiops_dilemma import auth, regions, serving

# Fallback when the model has no Model Garden verified config (tags are dated; override with deployment.image).
VLLM_IMAGE = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20260713_0916_RC01"


def image_supports(image: str | None, accelerator_type: str) -> str | None:
    """None if `image` can serve on `accelerator_type`, else why not (catalog SERVING_IMAGES accelerator_kinds)."""
    from aiops_dilemma.catalog import GPUS, SERVING_IMAGES
    if not image:
        return None
    entry = next((x for x in SERVING_IMAGES if x["image"] == image), None)
    kinds = (entry or {}).get("accelerator_kinds")
    kind = (GPUS.get(accelerator_type) or {}).get("kind", "gpu")
    if kinds and kind not in kinds:
        return (f"the serving image {image.rsplit('/', 1)[-1][:60]} is a {'/'.join(k.upper() for k in kinds)} build and "
                f"can't run on {accelerator_type} ({kind.upper()}): use the Model Garden verified recipe "
                "(deployment.image empty) or a TPU image")
    return None


def valid_image_ref(image: str) -> bool:
    """Artifact Registry, Container Registry or Docker Hub (docker.io/...) image reference, as the web app checks it."""
    return serving.valid_image_ref(image)


def _project(cfg: dict) -> str:
    """The session's project, else the app's (a job started by the web app gets it as $GOOGLE_CLOUD_PROJECT)."""
    return auth.real_project(cfg.get("project")) or auth.real_project(auth.app_project()) or "my-gcp-project"


def _args(cfg: dict, chosen: dict, verified: dict | None, any_image: str | None = None) -> tuple[str, list[str], dict, dict]:
    m = cfg["model"]
    if verified:
        image, args, env = verified["image"], list(verified["args"]), dict(verified["env"])
        routes = {"predict": verified["predict_route"], "health": verified["health_route"], "port": verified["port"]}
        # Drop speculative-decoding / precision flags from the verified args; we control them.
        args = [a for a in args if not a.startswith(("--quantization", "--max-model-len"))]
        if cfg["deployment"].get("image"):   # pinned or custom image with the verified recipe's arguments
            image = cfg["deployment"]["image"]
    else:
        image = cfg["deployment"].get("image") or any_image or VLLM_IMAGE
        args = ["python", "-m", "vllm.entrypoints.api_server", "--host=0.0.0.0", "--port=8080",
                f"--model={m.get('hf_id') or m['model_garden_id']}", "--gpu-memory-utilization=0.9",
                f"--tensor-parallel-size={chosen['accelerator_count']}"]
        env = {"MODEL_ID": m.get("hf_id") or m["model_garden_id"], "DEPLOY_SOURCE": "mlops-dilemma"}
        routes = {"predict": "/generate", "health": "/ping", "port": 8080}
    # Pre-quantized or natively FP8 / INT4 / MXFP4 checkpoints carry their quantization config, which vLLM reads: only
    # FP8 from a BF16 checkpoint needs the flag (INT4 and MXFP4 never quantize on the fly; Model Garden's gpt-oss
    # recipes pass no --quantization either).
    if (m.get("precision") == "fp8" and not m.get("prequantized_checkpoint")
            and (planner_dm(cfg).get("quantization") != "native")):
        args.append("--quantization=fp8")
    if m.get("max_model_len"):
        args.append(f"--max-model-len={m['max_model_len']}")
    if cfg["deployment"].get("args_mode") != "replace":   # replace: serving.container_spec uses extra_args alone
        args = serving.merge_args(args, cfg["deployment"].get("extra_args") or [])
    return image, args, env, routes


def planner_dm(cfg: dict) -> dict:
    from aiops_dilemma import planner
    try:
        return planner.deployment_model(cfg)
    except (Exception, SystemExit):
        return {}


def adapt_recipe(recipe: dict, count: int) -> dict:
    """A Model Garden verified recipe for another machine, with the tensor-parallel degree set to `count`.

    Used when the chosen accelerator has no recipe (e.g. L4, A100 for Gemma 4): the recipe's weights path (Model
    Garden's copy, no Hugging Face token needed), serving flags and routes are kept, so results stay comparable
    across GPUs. The generic fallback (Hugging Face weights, no reasoning parser) remains for models without any."""
    out = dict(recipe, args=[re.sub(r"^(--tensor-parallel-size|--tp-size)=\d+$", rf"\g<1>={count}", a)
                             for a in recipe["args"]])
    if not any(a.startswith(("--tensor-parallel-size", "--tp-size")) for a in out["args"]):
        out["args"].append(f"--tensor-parallel-size={count}")
    return out


def base_recipe(opts: list[dict], accelerator_type: str) -> dict | None:
    """The vLLM recipe (no speculative decoding) to adapt for an accelerator without its own: one from the same
    machine family first (A100 40GB reuses the A100 80GB recipe, keeping e.g. gpt-oss's VLLM_ATTENTION_BACKEND=
    TRITON_ATTN for Ampere), else the first."""
    from aiops_dilemma.catalog import GPUS
    pool = [o for o in opts if "speculative" not in " ".join(o["args"]) and "vllm" in o["image"]]
    fam = (GPUS.get(accelerator_type) or {}).get("family")
    same = [o for o in pool if fam and (GPUS.get(o["accelerator_type"]) or {}).get("family") == fam]
    return (same or pool or [None])[0]


def deploy(cfg: dict, plan: dict, run_dir: Path, replicas: int = 1) -> dict:
    ch = plan["chosen"]
    why = image_supports((cfg.get("deployment") or {}).get("image"), ch["accelerator_type"])
    if why:
        raise SystemExit(f"Can't deploy: {why}.")
    dm = plan.get("deployment_model") or {}
    if (dm.get("quantization") == "prequantized" and cfg["model"].get("precision") in ("fp8", "int4", "mxfp4")
            and dm.get("base_model_garden_id") == cfg["model"].get("model_garden_id")):
        # FP8 / INT4 / MXFP4 with a Model Garden pre-quantized checkpoint (planner.deployment_model): deploy it as-is.
        cfg = {**cfg, "model": {**cfg["model"], "model_garden_id": dm["model_garden_id"],
                                "hf_id": dm.get("hf_id") or cfg["model"].get("hf_id"), "prequantized_checkpoint": True}}
    d_cfg = cfg.get("deployment") or {}
    if d_cfg.get("existing_endpoint") and d_cfg.get("dedicated_dns"):
        state = {
            "endpoint": d_cfg["existing_endpoint"],
            "dedicated_dns": d_cfg["dedicated_dns"],
            "region": cfg["region"],
            **{k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
            "precision": cfg["model"].get("precision", "bf16"),
            "replicas": replicas,
            "attached_existing": True,
        }
        (run_dir / "endpoint.json").write_text(json.dumps(state, indent=2))
        print(f"attached existing endpoint {state['endpoint']} ({state['dedicated_dns']})")
        return state

    import vertexai
    from vertexai import model_garden

    from aiops_dilemma.modelinfo import verified_options

    proj = _project(cfg)
    opts = verified_options(proj, cfg["region"], cfg["model"]["model_garden_id"])
    ver = next((o for o in opts
                if o["accelerator_type"] == ch["accelerator_type"] and o["accelerator_count"] == ch["accelerator_count"]
                and "speculative" not in " ".join(o["args"]) and "vllm" in o["image"]), None)
    any_image = next((o["image"] for o in opts if "vllm" in o["image"]), None)
    adapted_from = None
    if not ver:   # no recipe for this machine: reuse the model's recipe from another one (same flags and weights path)
        base = base_recipe(opts, ch["accelerator_type"])
        if base:
            ver, adapted_from = adapt_recipe(base, ch["accelerator_count"]), \
                f"{base['accelerator_count']}x {base['accelerator_type']}"
    if cfg["deployment"].get("image") and not valid_image_ref(cfg["deployment"]["image"]):
        raise SystemExit(f"deployment.image '{cfg['deployment']['image']}' doesn't look like an Artifact Registry, "
                         "Container Registry or Docker Hub image (e.g. us-docker.pkg.dev/my-project/my-repo/vllm:tag "
                         "or docker.io/vllm/vllm-openai:v0.31.0)")
    d_srv = dict(cfg["deployment"])
    problems = serving.normalize(d_srv)
    if problems:
        raise SystemExit("Can't deploy the custom serving container: " + "; ".join(problems))
    if serving.context_warning(cfg):
        print(f"warning: {serving.context_warning(cfg)}")
    image, args, env, routes = _args(cfg, ch, ver, any_image)
    # extra_args are already merged into args (append mode); replace mode takes them from d_srv
    spec = serving.container_spec({**d_srv, "extra_args": d_srv["extra_args"] if d_srv["args_mode"] == "replace" else []},
                                  {"image": image, "args": args, "env": env, "routes": routes}, ch.get("machine_type"),
                                  hf_token=_hf_token() if d_srv.get("pass_hf_token") else None)
    if d_srv.get("pass_hf_token") and not spec["record"].get("hf_token_passed"):
        print("warning: pass_hf_token is set but no Hugging Face token is available; gated weights won't load")
    image, args, env, routes = spec["image"], spec["args"], spec["env"], spec["routes"]
    image_requested = image
    if serving.is_dockerhub(image):
        serving.ensure_dockerhub_repo(proj, regions.require(cfg["region"]))
        image = serving.dockerhub_ar_uri(image, proj, regions.require(cfg["region"]))
        serving.warm_remote_image(image, proj)
        print(f"Docker Hub image {image_requested} -> pulled through {image}")
    source = ("tuned config" if d_srv.get("tuned") else "custom container" if serving.is_custom(d_srv)
              else "deployment.image" if cfg["deployment"].get("image") else "verified recipe" if ver else "fallback")
    if adapted_from and d_srv["args_mode"] != "replace":
        source += f" (args adapted from the {adapted_from} recipe)"
    print(f"serving image: {image} ({source})")
    vertexai.init(project=proj, location=regions.require(cfg["region"]), credentials=auth.credentials(proj))
    name = f"dilemma-{cfg['name']}"[:60]
    secs = inference_timeout_s(cfg)
    print(f"deploying {cfg['model']['model_garden_id']} on {ch['accelerator_count']}x {ch['accelerator_type']} "
          f"({ch['machine_type']}) in {cfg['region']} — typically 10-30 minutes...")
    common = dict(machine_type=ch["machine_type"], accelerator_type=ch["accelerator_type"],
                  accelerator_count=ch["accelerator_count"], min_replica_count=replicas, max_replica_count=replicas)
    container = dict(serving_container_image_uri=image, serving_container_args=args,
                     serving_container_environment_variables=env, serving_container_predict_route=routes["predict"],
                     serving_container_health_route=routes["health"], serving_container_ports=[routes["port"]],
                     serving_container_shared_memory_size_mb=spec["shared_memory_mb"],
                     serving_container_deployment_timeout=7200)
    if spec["command"]:
        container["serving_container_command"] = spec["command"]
    if (cfg.get("deployment") or {}).get("one_step_deploy") and serving.is_custom(d_srv):
        raise SystemExit("one_step_deploy can't use a custom serving container (command, env, routes, patch); "
                         "turn one_step_deploy off")
    if (cfg.get("deployment") or {}).get("one_step_deploy"):
        # Model Garden's one-step deploy: simpler, but the endpoint keeps the default ~30 s request limit, which
        # cuts reasoning and long-prompt requests. Kept as an opt-in fallback.
        ep = model_garden.OpenModel(cfg["model"]["model_garden_id"]).deploy(
            accept_eula=True, **common, **container, deploy_request_timeout=7200,
            endpoint_display_name=name, model_display_name=name, use_dedicated_endpoint=True,
            reservation_affinity_type="NO_RESERVATION")
        state = {"endpoint": ep.resource_name, "dedicated_dns": ep.gca_resource.dedicated_endpoint_dns,
                 "inference_timeout_s": None}
    else:
        state = _deploy_with_timeout(cfg, run_dir, name, secs, common, container, ch)
    state.update({"region": cfg["region"], **{k: ch[k] for k in ("machine_type", "accelerator_type", "accelerator_count")},
                  "precision": cfg["model"].get("precision", "bf16"), "replicas": replicas,
                  # provenance: what actually served the benchmark (analyze copies it into measured.json)
                  "image": image, "image_source": source, "args": args, **spec["record"],
                  **({"image_requested": image_requested} if image_requested != image else {})})
    (run_dir / "endpoint.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    return state


def _hf_token() -> str | None:
    """The Hugging Face token for pass_hf_token: the job's HF_TOKEN (the app passes a memory-only token that way),
    else the saved one."""
    import os
    from aiops_dilemma import hftoken
    return os.environ.get("HF_TOKEN") or hftoken.token()


def _publisher_model(model_garden_id: str) -> str:
    """'google/gemma4@gemma-4-26b-a4b-it' -> 'publishers/google/models/gemma4' (Model Garden source for weights access)."""
    pub, rest = model_garden_id.split("/", 1)
    return f"publishers/{pub}/models/{rest.split('@')[0]}"


def _deploy_with_timeout(cfg: dict, run_dir: Path, name: str, secs: int, common: dict, container: dict, ch: dict) -> dict:
    """Create a dedicated endpoint with a long inference timeout, then upload + deploy the model into it.

    The request timeout can only be set when the endpoint is created (updates are silently ignored), and Model
    Garden's one-step deploy creates its own endpoint with the ~30 s default. endpoint.json is written as soon as
    the endpoint exists so a failed deploy can still be torn down."""
    from google.cloud import aiplatform
    from vertexai import model_garden
    mg_id = cfg["model"]["model_garden_id"]
    try:   # the one-step deploy accepts the EULA itself; here accept it explicitly (no-op when already accepted)
        om = model_garden.OpenModel(mg_id)
        if hasattr(om, "accept_model_license_agreement"):
            om.accept_model_license_agreement()
    except Exception as e:   # noqa: BLE001
        print(f"note: EULA check skipped ({str(e)[:120]})")
    ep = aiplatform.Endpoint.create(display_name=name, dedicated_endpoint_enabled=True, inference_timeout=secs)
    state = {"endpoint": ep.resource_name, "dedicated_dns": ep.gca_resource.dedicated_endpoint_dns,
             "region": cfg["region"], "partial": True}
    (run_dir / "endpoint.json").write_text(json.dumps(state, indent=2))
    got = ep.gca_resource.client_connection_config.inference_timeout.seconds
    print(f"created endpoint {ep.resource_name} (inference timeout {got or 'default'}s)")
    model = aiplatform.Model.upload(display_name=name, model_garden_source_model_name=_publisher_model(mg_id), **container)
    state["model"] = model.resource_name
    (run_dir / "endpoint.json").write_text(json.dumps(state, indent=2))
    model.deploy(endpoint=ep, **common, deploy_request_timeout=7200, reservation_affinity_type="NO_RESERVATION",
                 deployed_model_display_name=name)
    ep = aiplatform.Endpoint(ep.resource_name)
    state.pop("partial", None)
    state["dedicated_dns"] = ep.gca_resource.dedicated_endpoint_dns or state["dedicated_dns"]
    state["inference_timeout_s"] = int(ep.gca_resource.client_connection_config.inference_timeout.seconds) or None
    print(f"endpoint inference timeout: {state['inference_timeout_s'] or 'default (~30 s)'}")
    return state


MIN_INFERENCE_TIMEOUT_S = 300
MAX_INFERENCE_TIMEOUT_S = 3600


def inference_timeout_s(cfg: dict) -> int:
    """Per-request timeout for the endpoint: deployment.inference_timeout_s, else the latency budget (>= 5 min).

    LLM requests (long prompts, reasoning, a deep queue at peak) routinely take minutes; the endpoint's default
    limit is far shorter, so long requests would be cut and counted as failures."""
    d = cfg.get("deployment") or {}
    want = d.get("inference_timeout_s") or (cfg.get("constraints") or {}).get("latency_budget_s") or 600
    return int(min(MAX_INFERENCE_TIMEOUT_S, max(MIN_INFERENCE_TIMEOUT_S, float(want))))


def set_inference_timeout(cfg: dict, endpoint_name: str, retries: int = 12, wait_s: float = 30) -> int | None:
    """Raise the endpoint's inference timeout (ClientConnectionConfig); Model Garden's deploy() can't set it."""
    from google.cloud import aiplatform_v1
    from google.protobuf import duration_pb2
    secs = inference_timeout_s(cfg)
    region = regions.require(endpoint_name.split("/locations/")[1].split("/")[0])   # becomes the API host
    # update_endpoint() accepts client_connection_config but silently ignores it on dedicated endpoints; the
    # long-running update applies it. It fails while another update is in flight (right after a deploy), so retry.
    client = aiplatform_v1.EndpointServiceClient(credentials=auth.credentials(_project(cfg)),
                                                 client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"})
    last = None
    for attempt in range(retries):
        try:
            ep = client.get_endpoint(name=endpoint_name)
            ep.client_connection_config.inference_timeout = duration_pb2.Duration(seconds=secs)
            client.update_endpoint_long_running(request={"endpoint": ep}).result(timeout=600)
            got = client.get_endpoint(name=endpoint_name).client_connection_config.inference_timeout.seconds
            if got:
                print(f"endpoint inference timeout set to {got}s")
                return int(got)
            last = "update accepted but the timeout is still unset"
        except Exception as e:   # noqa: BLE001
            last = str(e)[:200]
        time.sleep(wait_s)
    print(f"warning: could not set the endpoint inference timeout ({last}); long requests may be cut")
    return None


def teardown(cfg: dict, run_dir: Path):
    from google.cloud import aiplatform

    path = run_dir / "endpoint.json"
    if not path.exists():
        print("no endpoint recorded for this run")
        return
    state = json.loads(path.read_text())
    if state.get("attached_existing"):
        path.unlink()
        print(f"detached existing endpoint {state['endpoint']} (left running)")
        return
    proj = _project(cfg)
    aiplatform.init(project=proj, location=regions.require(state["region"]), credentials=auth.credentials(proj))
    ep = aiplatform.Endpoint(state["endpoint"])
    # Models deployed on this endpoint by this tool: the one we uploaded, plus the model Model Garden's one-step
    # deploy creates (display name dilemma-<session>). Collect them before undeploying; no cost, but no clutter.
    models = {state["model"]} if state.get("model") else set()
    name = f"dilemma-{cfg['name']}"[:60]
    try:
        for dm in ep.list_models():
            m = aiplatform.Model(dm.model)
            if m.display_name == name or dm.display_name == name:
                models.add(m.resource_name)
    except Exception as e:   # noqa: BLE001
        print(f"note: could not list the endpoint's models: {str(e)[:120]}")
    ep.undeploy_all()
    ep.delete()
    for mr in sorted(models):
        try:
            aiplatform.Model(mr).delete()
        except Exception as e:   # noqa: BLE001
            print(f"note: could not delete model {mr}: {str(e)[:120]}")
    path.unlink()
    print(f"deleted {state['endpoint']}")
