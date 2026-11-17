"""Custom serving containers for benchmark endpoints: your own image, command, vLLM arguments, environment, routes,
shared memory and an optional Python patch (e.g. a tuned MoE tile schedule or custom Triton kernels).

Everything lives under `deployment:` in the workload file (all optional; the default is Model Garden's verified recipe):

    image: docker.io/vllm/vllm-openai:v0.31.0   # Artifact Registry, Container Registry or Docker Hub (pulled through
                                                # an Artifact Registry remote repository: Agent Platform only runs
                                                # images from Artifact Registry)
    command: [vllm, serve]                      # container entrypoint; empty = the image's
    args_mode: replace                          # append (default): extra_args are merged into the recipe's arguments,
                                                # a flag given here replaces the recipe's; replace: extra_args are the
                                                # full argument list
    extra_args: [google/gemma-4-26B-A4B-it, --max-num-seqs=256, ...]
    env: {VLLM_USE_V2_MODEL_RUNNER: "1"}
    routes: {predict: /v1/chat/completions, health: /health, port: 8080}
    shared_memory_mb: 81920                     # /dev/shm; empty = 16 GB, or ~43% of the machine's memory when
                                                # HF_HOME points into /dev/shm (weights cached in RAM)
    pass_hf_token: true                         # give the endpoint your Hugging Face token (gated weights)
    sitecustomize: "..."                        # Python run at interpreter start inside the server (needs `command`)
    tuned: {source: alpha_inference, title: ..., hypothesis: ...}   # provenance, set by `import_candidate`

`import_candidate` turns an alpha_inference / AlphaEvolve candidate (best_candidate.py, or the same dict as JSON) into
these fields. The candidate file is parsed, never executed: its returned dict must be a literal.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import pprint
import re
import time

UPSTREAM_VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.31.0"
DOCKERHUB_REPO = "dilemma-dockerhub"          # Artifact Registry remote repository that proxies Docker Hub
DEFAULT_SHM_MB = 16384
OPENAI_ROUTES = {"predict": "/v1/chat/completions", "health": "/health", "port": 8080}
PATCH_ENV = "DILEMMA_PATCH_B64"
PATCH_DIR = "/tmp/dilemma_patch"
ARGS_MODES = ("append", "replace")
SECRET_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"}

MAX_ARGS, MAX_ARG_LEN = 256, 16384
MAX_CMD, MAX_CMD_LEN = 64, 8192
MAX_ENV, MAX_ENV_VAL = 64, 32768
MAX_PATCH = 512 * 1024
MAX_CANDIDATE = 1024 * 1024

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_ROUTE = re.compile(r"^/[A-Za-z0-9._~/:-]{0,200}$")
_AR_GCR = re.compile(r"^(?:[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9][a-z0-9.:-]*(?:/[a-z0-9][a-z0-9._-]*){2,}"
                     r"|(?:[a-z]+\.)?gcr\.io/[a-z0-9][a-z0-9.:-]*(?:/[a-z0-9][a-z0-9._-]*)+)"
                     r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[a-f0-9]{64})?$")
_DOCKERHUB = re.compile(r"^docker\.io/((?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)?[a-z0-9]+(?:[._-][a-z0-9]+)*)"
                        r"(:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[a-f0-9]{64})?$")


# ---------------------------------------------------------------- images

def is_dockerhub(image: str | None) -> bool:
    return bool(_DOCKERHUB.match(str(image or "").strip()))


def valid_image_ref(image: str | None) -> bool:
    """Artifact Registry, Container Registry, or Docker Hub written as docker.io/<namespace>/<image>[:tag|@digest]."""
    v = str(image or "").strip()
    return bool(_AR_GCR.match(v) or _DOCKERHUB.match(v))


def dockerhub_ar_uri(image: str, project: str, region: str) -> str:
    """docker.io/vllm/vllm-openai:v0.31.0 -> <region>-docker.pkg.dev/<project>/dilemma-dockerhub/vllm/vllm-openai:v0.31.0
    (official images: docker.io/ubuntu -> .../library/ubuntu)."""
    m = _DOCKERHUB.match(image.strip())
    if not m:
        raise ValueError(f"not a Docker Hub image: {image}")
    path, ref = m.group(1), m.group(2) or ":latest"
    if "/" not in path:
        path = f"library/{path}"
    return f"{region}-docker.pkg.dev/{project}/{DOCKERHUB_REPO}/{path}{ref}"


def ensure_dockerhub_repo(project: str, region: str, session=None, wait_s: float = 120) -> str:
    """Create the Artifact Registry remote repository that proxies Docker Hub, unless it exists. Returns its name.

    Agent Platform only runs images from Artifact Registry; a remote repository pulls (and caches) the Docker Hub image
    on first use, with no mirroring step. Cached versions older than 14 days are cleaned up."""
    if session is None:
        from google.auth.transport.requests import AuthorizedSession

        from aiops_dilemma import auth
        session = AuthorizedSession(auth.credentials(project))
    base = f"https://artifactregistry.googleapis.com/v1/projects/{project}/locations/{region}/repositories"
    name = f"projects/{project}/locations/{region}/repositories/{DOCKERHUB_REPO}"
    r = session.get(f"{base}/{DOCKERHUB_REPO}", timeout=30)
    if r.status_code == 200:
        return name
    manual = (f"gcloud artifacts repositories create {DOCKERHUB_REPO} --project={project} --location={region} "
              "--repository-format=docker --mode=remote-repository --remote-docker-repo=DOCKER-HUB")
    if r.status_code not in (404,):
        raise SystemExit(f"Can't check the Artifact Registry repository for Docker Hub images ({r.status_code}: "
                         f"{r.text[:200]}). Create it once with:\n  {manual}")
    body = {"format": "DOCKER", "mode": "REMOTE_REPOSITORY",
            "description": "Docker Hub pull-through cache for mlops-dilemma benchmark endpoints",
            "remoteRepositoryConfig": {"dockerRepository": {"publicRepository": "DOCKER_HUB"}},
            "cleanupPolicies": {"delete-old": {"id": "delete-old", "action": "DELETE",
                                               "condition": {"olderThan": f"{14 * 86400}s"}}},
            "cleanupPolicyDryRun": False}
    r = session.post(base, params={"repositoryId": DOCKERHUB_REPO}, json=body, timeout=60)
    if r.status_code == 409:
        return name
    if r.status_code >= 300:
        raise SystemExit(f"Can't create the Artifact Registry remote repository for Docker Hub images "
                         f"({r.status_code}: {r.text[:200]}). Create it once with:\n  {manual}\nor push the image to "
                         "Artifact Registry yourself and use that URI.")
    op = r.json()
    deadline = time.time() + wait_s
    while not op.get("done") and time.time() < deadline:
        time.sleep(3)
        op = session.get(f"https://artifactregistry.googleapis.com/v1/{op['name']}", timeout=30).json()
    if op.get("error"):
        raise SystemExit(f"Creating the Docker Hub remote repository failed: {op['error']}. Create it with:\n  {manual}")
    print(f"created Artifact Registry remote repository {name} (Docker Hub pull-through cache)")
    return name


MANIFEST_ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json",
                              "application/vnd.docker.distribution.manifest.list.v2+json",
                              "application/vnd.docker.distribution.manifest.v2+json",
                              "application/vnd.oci.image.manifest.v1+json"])


def warm_remote_image(ar_uri: str, project: str, session=None, retries: int = 3) -> None:
    """Pull the image's manifest through the remote repository, so Artifact Registry caches it before Agent Platform
    checks that the image exists (a remote repository only holds an image after its first pull; the model upload
    fails with 404 otherwise). Layers are fetched when the endpoint pulls the image."""
    if session is None:
        from google.auth.transport.requests import AuthorizedSession

        from aiops_dilemma import auth
        session = AuthorizedSession(auth.credentials(project))
    host, _, rest = ar_uri.partition("/")
    if "@" in rest:
        path, ref = rest.split("@", 1)
    else:
        path, _, ref = rest.rpartition(":") if ":" in rest.rsplit("/", 1)[-1] else (rest, "", "latest")
    url = f"https://{host}/v2/{path}/manifests/{ref}"
    last = None
    for attempt in range(retries):
        r = session.get(url, headers={"Accept": MANIFEST_ACCEPT}, timeout=120)
        if r.status_code == 200:
            return
        last = f"{r.status_code}: {r.text[:200]}"
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"Couldn't pull {ar_uri} through Artifact Registry ({last}). Check that the Docker Hub image and "
                     "tag exist.")


# ---------------------------------------------------------------- arguments

def flag_name(arg: str) -> str | None:
    """'--max-num-seqs=256' -> '--max-num-seqs'; positional arguments -> None."""
    return arg.split("=", 1)[0] if isinstance(arg, str) and arg.startswith("--") else None


def merge_args(base: list[str], extra: list[str]) -> list[str]:
    """`base` + `extra`, where a flag in `extra` replaces the same flag in `base` (`--x=1` or `--x 1` forms)."""
    names = {flag_name(a) for a in extra} - {None}
    out, i = [], 0
    while i < len(base):
        a, n = base[i], flag_name(base[i])
        if n in names:
            # two-token form: drop the value too
            if "=" not in a and i + 1 < len(base) and not str(base[i + 1]).startswith("--"):
                i += 1
            i += 1
            continue
        out.append(a)
        i += 1
    return out + list(extra)


def arg_value(args: list[str], name: str) -> str | None:
    for i, a in enumerate(args or []):
        if a == name and i + 1 < len(args):
            return str(args[i + 1])
        if isinstance(a, str) and a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


# ---------------------------------------------------------------- validation

def _str_list(v, what: str, max_items: int, max_len: int, errors: list[str]) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        v = [x for x in (s.strip() for s in v.splitlines()) if x]
    if not isinstance(v, list) or not all(isinstance(x, (str, int, float)) for x in v):
        errors.append(f"deployment.{what} must be a list of strings")
        return []
    v = [str(x) for x in v]
    if len(v) > max_items:
        errors.append(f"deployment.{what} has {len(v)} items (at most {max_items})")
    if any(len(x) > max_len for x in v):
        errors.append(f"deployment.{what}: an item is longer than {max_len} characters")
    if any("\x00" in x for x in v):
        errors.append(f"deployment.{what}: items can't contain NUL characters")
    return v


def normalize(d: dict, cloud: bool = False) -> list[str]:
    """Normalize the custom-serving fields of a `deployment` block in place; returns the problems found (empty = OK)."""
    errors: list[str] = []
    img = d.get("image")
    if img is not None:
        img = str(img).strip() or None
        d["image"] = img
        if img and not valid_image_ref(img):
            errors.append(f"deployment.image '{img[:120]}' isn't an Artifact Registry, Container Registry or Docker Hub "
                          "image (e.g. us-docker.pkg.dev/my-project/my-repo/vllm:tag or docker.io/vllm/vllm-openai:v0.31.0)")
    d["extra_args"] = _str_list(d.get("extra_args"), "extra_args", MAX_ARGS, MAX_ARG_LEN, errors)
    d["command"] = _str_list(d.get("command"), "command", MAX_CMD, MAX_CMD_LEN, errors) or None
    mode = str(d.get("args_mode") or "append").strip().lower()
    if mode not in ARGS_MODES:
        errors.append(f"deployment.args_mode must be one of {', '.join(ARGS_MODES)}")
        mode = "append"
    d["args_mode"] = mode
    if mode == "replace" and not d["extra_args"]:
        errors.append("deployment.args_mode is replace but extra_args is empty: give the full argument list")

    env = d.get("env") or {}
    if isinstance(env, list):   # ["KEY=VALUE", ...]
        pairs = {}
        for x in env:
            k, sep, v = str(x).partition("=")
            if not sep:
                errors.append(f"deployment.env entry '{str(x)[:40]}' is not KEY=VALUE")
            pairs[k.strip()] = v
        env = pairs
    if not isinstance(env, dict):
        errors.append("deployment.env must be a mapping of NAME: value")
        env = {}
    clean = {}
    for k, v in env.items():
        k = str(k).strip()
        if not _ENV_KEY.match(k):
            errors.append(f"deployment.env: '{k[:40]}' is not a valid variable name")
        elif k.upper().startswith("AIP_"):
            errors.append(f"deployment.env: {k} is reserved by Agent Platform (AIP_*)")
        elif k.upper() in SECRET_ENV:
            errors.append(f"deployment.env: don't put {k} in the workload file; use pass_hf_token instead")
        elif k == PATCH_ENV:
            errors.append(f"deployment.env: {k} is set from deployment.sitecustomize")
        else:
            sv = "" if v is None else (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            if len(sv) > MAX_ENV_VAL:
                errors.append(f"deployment.env: {k} is longer than {MAX_ENV_VAL} characters")
            clean[k] = sv
    if len(clean) > MAX_ENV:
        errors.append(f"deployment.env has {len(clean)} variables (at most {MAX_ENV})")
    d["env"] = clean

    routes = d.get("routes")
    if routes:
        if not isinstance(routes, dict):
            errors.append("deployment.routes must be {predict, health, port}")
            routes = None
        else:
            r = {}
            for k in ("predict", "health"):
                if routes.get(k):
                    if not _ROUTE.match(str(routes[k])):
                        errors.append(f"deployment.routes.{k} must be a path like /v1/chat/completions")
                    r[k] = str(routes[k])
            if routes.get("port") not in (None, ""):
                try:
                    p = int(routes["port"])
                    if not 1 <= p <= 65535:
                        raise ValueError
                    r["port"] = p
                except (TypeError, ValueError):
                    errors.append("deployment.routes.port must be a port number (1-65535)")
            routes = r or None
    d["routes"] = routes or None

    shm = d.get("shared_memory_mb")
    if shm in (None, ""):
        d["shared_memory_mb"] = None
    else:
        try:
            shm = int(float(shm))
            if not 1024 <= shm <= 1048576:
                raise ValueError
            d["shared_memory_mb"] = shm
        except (TypeError, ValueError):
            errors.append("deployment.shared_memory_mb must be a whole number of MB between 1024 and 1048576")
            d["shared_memory_mb"] = None

    d["pass_hf_token"] = bool(d.get("pass_hf_token", False))
    if cloud and d["pass_hf_token"]:
        errors.append("Passing the Hugging Face token to an endpoint is turned off on this shared deployment")

    patch = d.get("sitecustomize")
    if patch is not None and not isinstance(patch, str):
        errors.append("deployment.sitecustomize must be Python source text")
        patch = None
    patch = (patch or "").strip("\n") or None
    if patch and len(patch.encode()) > MAX_PATCH:
        errors.append(f"deployment.sitecustomize is larger than {MAX_PATCH // 1024} KB")
    if patch:
        try:
            ast.parse(patch)
        except SyntaxError as e:
            errors.append(f"deployment.sitecustomize is not valid Python (line {e.lineno}: {e.msg})")
    d["sitecustomize"] = patch
    if patch and not d["command"]:
        errors.append("deployment.sitecustomize needs deployment.command (the patch wraps the container's command, "
                      "e.g. [vllm, serve])")

    tuned = d.get("tuned")
    if tuned is not None and not isinstance(tuned, dict):
        tuned = None
    d["tuned"] = {str(k)[:40]: str(v)[:600] for k, v in tuned.items() if v is not None} if tuned else None
    return errors


def is_custom(d: dict) -> bool:
    """True when the serving setup differs from the recipe beyond image/extra args (for provenance and reference runs)."""
    d = d or {}
    return bool(d.get("command") or d.get("args_mode") == "replace" or d.get("env") or d.get("routes")
                or d.get("sitecustomize") or d.get("tuned"))


# ---------------------------------------------------------------- patch (sitecustomize)

_HOOKS = '''\
# Generated by mlops-dilemma: runs at interpreter start inside the serving container (sitecustomize.py).
import importlib.abc as _dl_abc
import sys as _dl_sys

_DL_HOOKS = {}


class _DlLoader(_dl_abc.Loader):
    def __init__(self, orig, callbacks):
        self._orig, self._cbs = orig, callbacks

    def create_module(self, spec):
        return self._orig.create_module(spec) if hasattr(self._orig, "create_module") else None

    def exec_module(self, module):
        self._orig.exec_module(module)
        for cb in list(self._cbs):
            try:
                cb(module)
            except Exception as e:  # a failed patch must not take the server down
                print(f"sitecustomize: patch on {module.__name__} failed: {e}", file=_dl_sys.stderr)


class _DlFinder(_dl_abc.MetaPathFinder):
    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in _DL_HOOKS or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            for finder in _dl_sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _DlLoader(spec.loader, _DL_HOOKS[fullname])
                    return spec
            return None
        finally:
            self._busy.discard(fullname)


_dl_sys.meta_path.insert(0, _DlFinder())


def register_vllm_patch(module_name, patch_fn):
    """Run patch_fn(module) right after `module_name` is imported (or now, if it already is)."""
    _DL_HOOKS.setdefault(module_name, []).append(patch_fn)
    if module_name in _dl_sys.modules:
        try:
            patch_fn(_dl_sys.modules[module_name])
        except Exception as e:
            print(f"sitecustomize: patch on {module_name} failed: {e}", file=_dl_sys.stderr)
'''

_MOE = '''
_DL_MOE_TUNED_CONFIG = {cfg}


def _dl_apply_moe_schedule(fused_moe):
    def get_moe_configs(E, N, dtype, block_n=None, block_k=None):
        return {{int(k): dict(v) for k, v in _DL_MOE_TUNED_CONFIG.items()}}
    fused_moe.get_moe_configs = get_moe_configs


register_vllm_patch("vllm.model_executor.layers.fused_moe.fused_moe", _dl_apply_moe_schedule)
'''


def compose_patch(moe_tuned_config: dict | None = None, custom_kernel_code: str = "") -> str | None:
    """sitecustomize.py source: a post-import hook (`register_vllm_patch`, the same contract alpha_inference kernels
    use), an optional fused-MoE tile schedule ({batch size M: {BLOCK_SIZE_M, ...}}) and optional kernel code."""
    moe = {}
    for m, tile in (moe_tuned_config or {}).items():
        if isinstance(tile, dict) and tile:
            moe[int(m)] = {str(k): int(v) for k, v in tile.items()}
    code = (custom_kernel_code or "").strip()
    if not moe and not code:
        return None
    parts = [_HOOKS]
    if moe:
        parts.append(_MOE.format(cfg=pprint.pformat(dict(sorted(moe.items())), width=110)))
    if code:
        parts.append("\n# ---- custom kernel code\n" + code + "\n")
    return "".join(parts)


def wrap_command(command: list[str]) -> list[str]:
    """bash wrapper that writes deployment.sitecustomize (from $DILEMMA_PATCH_B64) into a directory on PYTHONPATH,
    then execs the original command with the container args."""
    boot = (f"mkdir -p {PATCH_DIR} && python3 -c \"import base64, os, pathlib; "
            f"pathlib.Path('{PATCH_DIR}/sitecustomize.py').write_bytes(base64.b64decode(os.environ['{PATCH_ENV}']))\" "
            f"&& export PYTHONPATH=\"{PATCH_DIR}:${{PYTHONPATH:-}}\" && exec \"$@\"")
    return ["bash", "-lc", boot, "--", *command]


# ---------------------------------------------------------------- container spec

def recommended_shm_mb(machine_type: str | None) -> int:
    """~43% of the machine's memory for /dev/shm when the weights are cached there (the container disk is ~50 GB)."""
    mt = str(machine_type or "").lower()
    ram = 64
    if m := re.search(r"-standard-(\d+)$", mt):
        ram = int(m.group(1)) * (15 / 4 if mt.startswith("g4-") else 4)
    elif m := re.search(r"^(a2|a3|a4)-(highgpu|ultragpu|megagpu|edgegpu)-(\d+)g$", mt):
        per = {"a3": 234, "a4": 480}.get(m.group(1), 170 if m.group(2) == "ultragpu" else 85)
        ram = int(m.group(3)) * per
    elif m := re.search(r"^g2-standard-(\d+)$", mt):
        ram = int(m.group(1)) * 4
    return int(max(8192, min(int(ram * 1024 * 0.43) // 1024 * 1024, 524288)))


def container_spec(d: dict, base: dict, machine_type: str | None, hf_token: str | None = None) -> dict:
    """Final container from the recipe-derived `base` {image, args, env, routes} and the custom fields in `d`
    (deployment). Returns {image, command, args, env, routes, shared_memory_mb, record}; `record` is what
    endpoint.json keeps (no token, the patch only as size + sha256)."""
    replace = d.get("args_mode") == "replace"
    image = d.get("image") or base["image"]
    args = list(d.get("extra_args") or []) if replace else merge_args(base["args"], d.get("extra_args") or [])
    env = {} if replace else dict(base.get("env") or {})
    env.update(d.get("env") or {})
    routes = dict(OPENAI_ROUTES if replace else base["routes"])
    routes.update(d.get("routes") or {})
    if replace and not (d.get("routes") or {}).get("port") and arg_value(args, "--port"):
        routes["port"] = int(arg_value(args, "--port"))
    command = list(d.get("command") or [])
    patch = d.get("sitecustomize")
    record_env = dict(env)
    if patch:
        env[PATCH_ENV] = base64.b64encode(patch.encode()).decode("ascii")
        command = wrap_command(command)
    if d.get("pass_hf_token") and hf_token:
        env["HF_TOKEN"] = hf_token
    shm = d.get("shared_memory_mb")
    if not shm:
        shm = recommended_shm_mb(machine_type) if str(env.get("HF_HOME", "")).startswith("/dev/shm") else DEFAULT_SHM_MB
    # How the load test reaches the server: Model Garden containers answer Agent Platform's chat-completions route; a
    # container serving vLLM's OpenAI API itself is reached through :streamRawPredict (body passed to the predict
    # route), and only knows its served model name, not "default".
    raw = routes.get("predict", "").endswith("/chat/completions") and (replace or bool(d.get("command")))
    served = None
    if raw:
        served = (arg_value(args, "--served-model-name") or "").split()[0:1] or \
            ([args[0]] if args and not str(args[0]).startswith("-") else [])
        served = served[0] if served else None
    record = {"api": "raw" if raw else None, "served_model": served,
              "command": list(d.get("command") or []) or None, "args_mode": d.get("args_mode") or "append",
              "env": record_env or None, "routes": routes, "shared_memory_mb": shm,
              "hf_token_passed": bool(d.get("pass_hf_token") and hf_token),
              "patch": ({"bytes": len(patch.encode()), "sha256": hashlib.sha256(patch.encode()).hexdigest()}
                        if patch else None),
              "tuned": d.get("tuned") or None}
    return {"image": image, "command": command or None, "args": args, "env": env, "routes": routes,
            "shared_memory_mb": int(shm), "record": {k: v for k, v in record.items() if v not in (None, False)}}


# ---------------------------------------------------------------- import (alpha_inference candidates)

CANDIDATE_KEYS = ("plain_title", "hypothesis", "category", "vllm_args", "env_vars", "moe_tuned_config",
                  "custom_kernel_code")


def parse_candidate(text: str) -> dict:
    """The dict an alpha_inference candidate returns, without running it: the literal `return {...}` of
    get_candidate_config() in a candidate .py file, or the same dict as JSON."""
    if len(text.encode()) > MAX_CANDIDATE:
        raise ValueError(f"the file is larger than {MAX_CANDIDATE // 1024} KB")
    s = text.strip()
    if s.startswith("{"):
        try:
            data = json.loads(s)
        except ValueError as e:
            raise ValueError(f"not valid JSON: {e}") from None
    else:
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            raise ValueError(f"not valid Python (line {e.lineno}: {e.msg})") from None
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_candidate_config"),
                  None)
        if fn is None:
            raise ValueError("no get_candidate_config() function found (expected an alpha_inference candidate file)")
        ret = next((n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None), None)
        if ret is None:
            raise ValueError("get_candidate_config() has no return value")
        try:
            data = ast.literal_eval(ret.value)
        except (ValueError, SyntaxError):
            raise ValueError("get_candidate_config() must return a literal dict (it is read, never run)") from None
    if not isinstance(data, dict):
        raise ValueError("the candidate is not a dict")
    args = data.get("vllm_args")
    if not isinstance(args, list) or not args or not all(isinstance(a, str) for a in args):
        raise ValueError("the candidate has no vllm_args list")
    env = data.get("env_vars") or {}
    if not isinstance(env, dict):
        raise ValueError("env_vars must be a dict")
    moe = data.get("moe_tuned_config") or {}
    if not isinstance(moe, dict) or not all(isinstance(t, dict) for t in moe.values()):
        raise ValueError("moe_tuned_config must map batch sizes to tile dicts")
    try:
        {int(m): {str(k): int(v) for k, v in t.items()} for m, t in moe.items()}
    except (TypeError, ValueError):
        raise ValueError("moe_tuned_config values must be whole numbers") from None
    code = data.get("custom_kernel_code") or ""
    if not isinstance(code, str):
        raise ValueError("custom_kernel_code must be a string")
    return {k: data.get(k) for k in CANDIDATE_KEYS}


def needed_context(req: dict) -> int:
    """Context length the workload's longest prompts need (prompt + answer), for a --max-model-len check."""
    out = int(float(req.get("avg_output_tokens") or 0) * 2 + float(req.get("thinking_share") or 0)
              * float(req.get("avg_thinking_tokens") or 0) * 1.5) + 512
    if float(req.get("pct_over_128k") or 0) > 0:
        return int(max(128500, float(req.get("large_mean") or 188000) * 1.25)) + out
    if float(req.get("pct_over_32k") or 0) > 0:
        return 128000 + out
    return 32000 + out


def context_warning(cfg: dict) -> str | None:
    """A warning when explicit vLLM arguments cap the context below what the workload's longest prompts need."""
    d = cfg.get("deployment") or {}
    mml = arg_value(d.get("extra_args") or [], "--max-model-len")
    if not mml:
        return None
    try:
        mml = int(float(mml))
    except ValueError:
        return None
    need = needed_context(cfg.get("requests") or {})
    if mml >= need:
        return None
    return (f"--max-model-len is {mml:,}, below the ~{need:,} tokens this workload's longest prompts need: those "
            "requests will fail")


def import_candidate(text: str, cfg: dict, filename: str = "", image: str | None = None,
                     cloud: bool = False) -> dict:
    """Map an alpha_inference candidate onto `deployment` fields. Returns {"deployment": {...}, "summary": {...},
    "warnings": [...]}; nothing is saved."""
    c = parse_candidate(text)
    args = list(c["vllm_args"])
    # Speculative decoding is left out: on vLLM 0.31 it conflicts with structured (JSON-schema) output under load
    # (xgrammar rejects the speculated tokens and the request is terminated), and benchmark prompts ask for JSON.
    dropped = [a for a in args if flag_name(a) == "--speculative-config"]
    args = [a for a in args if flag_name(a) != "--speculative-config"]
    env = {str(k): str(v) for k, v in (c.get("env_vars") or {}).items()}
    model_arg = args[0] if not args[0].startswith("-") else None
    from_hf = bool(model_arg) and not model_arg.startswith(("gs://", "/"))
    if from_hf:
        env.setdefault("HF_HOME", "/dev/shm/hf_cache")   # weights in RAM: the container disk is ~50 GB
    patch = compose_patch(c.get("moe_tuned_config"), c.get("custom_kernel_code") or "")
    cur = (cfg.get("deployment") or {}).get("image")
    img = image or (cur if cur and is_dockerhub(cur) else None) or UPSTREAM_VLLM_IMAGE
    dep = {"image": img, "command": ["vllm", "serve"], "args_mode": "replace", "extra_args": args, "env": env,
           "routes": dict(OPENAI_ROUTES, port=int(arg_value(args, "--port") or 8080)),
           "shared_memory_mb": None, "pass_hf_token": bool(from_hf and not cloud), "sitecustomize": patch,
           "tuned": {"source": "alpha_inference", "file": filename[:200], "title": str(c.get("plain_title") or "")[:200],
                     "hypothesis": str(c.get("hypothesis") or "")[:600], "category": str(c.get("category") or "")[:40],
                     "imported_at": time.strftime("%Y-%m-%d")}}
    warnings = []
    if dropped:
        warnings.append("Speculative decoding (--speculative-config) was left out: with structured JSON output it fails "
                        "under load in vLLM 0.31 (\"grammar rejected tokens\"). If your prompts don't use a JSON "
                        "schema, you can add it back under vLLM arguments: " + dropped[0][:200])
    m = cfg.get("model") or {}
    want = (m.get("hf_id") or "").lower()
    if model_arg and want and model_arg.lower() != want:
        warnings.append(f"The tuned config serves {model_arg}; this workload's model is {m.get('hf_id')}. The benchmark "
                        f"measures {model_arg}.")
    q = arg_value(args, "--quantization")
    prec = (m.get("precision") or "bf16").lower()
    tuned_fp8 = (q or "").lower() == "fp8" or "fp8" in (model_arg or "").lower()
    if tuned_fp8 != (prec == "fp8"):
        warnings.append(f"The tuned config serves {'FP8' if tuned_fp8 else q or 'BF16'}; the workload says "
                        f"{prec.upper()}. The benchmark uses the tuned config's precision.")
    mml = arg_value(args, "--max-model-len")
    need = needed_context(cfg.get("requests") or {})
    if mml and int(float(mml)) < need:
        warnings.append(f"--max-model-len is {int(float(mml)):,}, but this workload has prompts that need about "
                        f"{need:,} tokens: those requests will fail. Tune on your own traffic, or raise "
                        f"--max-model-len (the other settings were tuned for {int(float(mml)):,}).")
    tp = arg_value(args, "--tensor-parallel-size") or arg_value(args, "--tp-size") or "1"
    count = (cfg.get("deployment") or {}).get("accelerator_count")
    if count and int(tp) != int(count):
        warnings.append(f"The tuned config uses {tp} GPU(s) per replica (--tensor-parallel-size); this workload's "
                        f"hardware has {count}.")
    if c.get("custom_kernel_code"):
        warnings.append("The config includes custom kernel code that runs inside the model server. Review it before "
                        "deploying.")
    if from_hf and cloud:
        warnings.append("The weights come from Hugging Face. This shared deployment can't pass a Hugging Face token, "
                        "so gated models won't load.")
    if is_dockerhub(img):
        warnings.append(f"{img} is on Docker Hub: the first deploy creates an Artifact Registry remote repository "
                        f"({DOCKERHUB_REPO}) that pulls and caches it.")
    moe_n = len(c.get("moe_tuned_config") or {})
    summary = {"title": dep["tuned"]["title"], "hypothesis": dep["tuned"]["hypothesis"],
               "category": dep["tuned"]["category"], "image": img, "args": len(args),
               "env": sorted(env), "moe_tile_sizes": moe_n,
               "kernel_lines": len((c.get("custom_kernel_code") or "").splitlines()),
               "max_model_len": int(float(mml)) if mml else None, "needed_context": need,
               "dropped_args": dropped}
    return {"deployment": dep, "summary": summary, "warnings": warnings}


def apply_import(cfg: dict, dep: dict) -> dict:
    """cfg with the imported deployment fields (hardware, endpoint settings and the rest are kept)."""
    out = {**cfg, "deployment": {**(cfg.get("deployment") or {}), **dep}}
    return out
