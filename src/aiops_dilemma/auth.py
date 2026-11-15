"""Google Cloud authentication: status (which identity ADC resolves to), default project, browser-completed
sign-in (gcloud ``--no-launch-browser`` flow driven over pipes), project listing and API checks."""

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid

CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
CRM_SEARCH_URL = "https://cloudresourcemanager.googleapis.com/v3/projects:search"
SERVICE_USAGE_URL = "https://serviceusage.googleapis.com/v1/projects/{project}/services/{service}"
METADATA_EMAIL_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"

LOGIN_START_TIMEOUT_S = 20      # time allowed for gcloud to print the sign-in URL
LOGIN_FINISH_TIMEOUT_S = 60     # time allowed for gcloud to exchange the code and exit
LOGIN_TTL_S = 10 * 60           # abandoned sign-ins are killed after this
PROJECTS_TTL_S = 5 * 60
PROJECTS_CAP = 500
# Test hook only: a JSON file with [{project_id, name, project_number}] served instead of the live listing.
PROJECTS_STUB_ENV = "DILEMMA_TEST_PROJECTS_FILE"

URL_RE = re.compile(r"https://accounts\.google\.com/[^\s'\"<>]+")
PROJECT_ID_RE = re.compile(r"^(?:[a-z0-9][a-z0-9.\-]*[a-z0-9]:)?[a-z][a-z0-9\-]{4,28}[a-z0-9]$")


class AuthError(Exception):
    """An auth/project operation failed; ``status`` is the HTTP status the server should use."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return 0, out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        msg = getattr(e, "output", b"")
        return 1, (msg.decode("utf-8", errors="replace").strip() if isinstance(msg, bytes) and msg else str(e))


def _http_get(url: str, *, token: str | None = None, params: dict | None = None, headers: dict | None = None,
              timeout: float = 10.0) -> tuple[int, dict]:
    """GET returning (status, JSON body or {}). Network errors return (0, {"error": message}). Tests patch this."""
    import requests

    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, params=params, headers=h, timeout=timeout)
    except Exception as e:
        return 0, {"error": str(e)[:200]}
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text[:500]}
    return r.status_code, body if isinstance(body, dict) else {"data": body}


GCLOUD_CACHE_TTL_S = 30          # gcloud config reads / token checks are reused this long (status is polled)
_GCLOUD_MEM: dict[tuple, tuple[float, tuple[int, str]]] = {}
_GCLOUD_LOCK = threading.Lock()


GCLOUD_STALE_S = 600              # an older answer is still returned at once while a fresh one is fetched
_GCLOUD_BG: set = set()


def _run_cached(cmd: list[str], timeout: int = 15, ttl: float = GCLOUD_CACHE_TTL_S) -> tuple[int, str]:
    """`_run` memoized (gcloud takes ~1.5 s per call): fresh for `ttl`, then served stale for up to GCLOUD_STALE_S
    while a background call refreshes it. Cleared on sign-in / account changes (_invalidate_caches)."""
    key = tuple(cmd)

    def call():
        res = _run(cmd, timeout=timeout)
        with _GCLOUD_LOCK:
            _GCLOUD_MEM[key] = (time.time(), res)
            _GCLOUD_BG.discard(key)
        return res

    with _GCLOUD_LOCK:
        hit = _GCLOUD_MEM.get(key)
        age = time.time() - hit[0] if hit else None
        if hit and age < ttl:
            return hit[1]
        if hit and age < GCLOUD_STALE_S:
            if key not in _GCLOUD_BG:
                _GCLOUD_BG.add(key)
                threading.Thread(target=call, daemon=True).start()
            return hit[1]
    return call()


def _gcloud_config_dir() -> str:
    return os.environ.get("CLOUDSDK_CONFIG", "").strip() or os.path.join(os.path.expanduser("~"), ".config", "gcloud")


def _gcloud_config_file_value(key: str) -> tuple[bool, str | None]:
    """(found, value) of core/<key> read directly from gcloud's active configuration file (read-only, instant).
    found=False when the files can't be read (then the caller asks gcloud). DILEMMA_GCLOUD_CONFIG_FILES=0 disables
    it (tests use a fake gcloud)."""
    if os.environ.get("DILEMMA_GCLOUD_CONFIG_FILES", "1").strip() in ("0", "false"):
        return False, None
    env = os.environ.get(f"CLOUDSDK_CORE_{key.upper()}", "").strip()
    if env:
        return True, env
    import configparser
    d = _gcloud_config_dir()
    try:
        active = os.environ.get("CLOUDSDK_ACTIVE_CONFIG_NAME", "").strip()
        if not active:
            with open(os.path.join(d, "active_config")) as f:
                active = f.read().strip() or "default"
        path = os.path.join(d, "configurations", f"config_{active}")
        if not os.path.isfile(path):
            return False, None
        cp = configparser.ConfigParser()
        cp.read(path)
        val = cp.get("core", key, fallback="").strip()
        return True, (val or None)
    except Exception:
        return False, None


def gcloud_config_value(key: str) -> str | None:
    """gcloud's core/<key> (account, project): from its config file when readable, else `gcloud config get-value`
    (cached briefly)."""
    found, val = _gcloud_config_file_value(key)
    if found:
        return val
    rc, out = _run_cached(["gcloud", "config", "get-value", key], timeout=8)
    if rc == 0 and out and out != "(unset)" and "ERROR" not in out:
        val = out.splitlines()[-1].strip()
        return val if val and val != "(unset)" else None
    return None


def default_project() -> str | None:
    """The machine's default project: environment, else gcloud config. Only a default for the app's own project
    (see app_project); the app never writes gcloud config."""
    for env_key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        val = os.environ.get(env_key, "").strip()
        if val and val != "(unset)":
            return val
    return gcloud_config_value("project")


def real_project(project: str | None) -> str | None:
    """`project` unless it is empty or the placeholder `my-gcp-project`."""
    project = (project or "").strip()
    return project if project and project not in ("my-gcp-project", "(unset)") else None


# The project this app uses for its Google API calls (resource paths and quota/billing). Chosen in the app's project
# picker and kept in memory only: choosing it never runs `gcloud config set project` or changes the environment,
# so other tools on the machine (and the ADC file) are left alone. Unset -> default_project(), memoized briefly.
_APP: dict = {"project": None, "default": None, "default_at": 0.0}
DEFAULT_PROJECT_TTL_S = 300


def app_project() -> str | None:
    """The app's project: the one chosen in the app, else the machine default (env / gcloud config) when unset."""
    if _APP["project"]:
        return _APP["project"]
    if time.time() - _APP["default_at"] > DEFAULT_PROJECT_TTL_S:
        _APP["default"], _APP["default_at"] = default_project(), time.time()
    return _APP["default"]


def credentials(project: str | None = None):
    """ADC with a fresh token whose quota (billing) project is `project`, default the app's project.

    `gcloud auth application-default login` stamps the then-current gcloud project into the ADC file as its quota
    project; a wrong stamp (e.g. a `sys-...` project without the Agent Platform API) makes every call fail with
    "API has not been used in project ...". Setting the quota project per call makes that stamp irrelevant: every
    Google API call this app makes goes through this helper (or auth_headers) and bills the app's project."""
    import google.auth.transport.requests

    creds, _ = _google_default()
    if not creds.valid:   # refresh the shared credentials once, so copies below carry the token
        try:
            creds.refresh(google.auth.transport.requests.Request())
        except Exception:
            pass
    proj = real_project(project) or real_project(app_project())
    if proj and hasattr(creds, "with_quota_project"):
        try:
            creds = creds.with_quota_project(proj)
        except Exception:   # credential types without quota-project support keep their own
            pass
    if not creds.valid:
        creds.refresh(google.auth.transport.requests.Request())
    return creds


def auth_headers(project: str | None = None, creds=None) -> dict:
    """HTTP headers for a Google REST call: bearer token + `x-goog-user-project` (the quota project)."""
    creds = creds or credentials(project)
    h = {"Authorization": f"Bearer {creds.token}"}
    quota = getattr(creds, "quota_project_id", None)
    if quota:
        h["x-goog-user-project"] = quota
    return h


# ------------------------------------------------------------------------------------------- credentials / identity
def credential_type(creds) -> str:
    """Classify an ADC credentials object: user, service_account, compute_metadata, external_account, impersonated."""
    checks = [
        ("google.auth.compute_engine.credentials", "Credentials", "compute_metadata"),
        ("google.oauth2.service_account", "Credentials", "service_account"),
        ("google.auth.impersonated_credentials", "Credentials", "impersonated"),
        ("google.auth.external_account", "Credentials", "external_account"),
        ("google.auth.external_account_authorized_user", "Credentials", "external_account"),
        ("google.oauth2.credentials", "Credentials", "user"),
    ]
    import importlib

    for mod, cls, label in checks:
        try:
            klass = getattr(importlib.import_module(mod), cls)
        except Exception:
            continue
        if isinstance(creds, klass):
            return label
    return type(creds).__name__.lower()


try:
    import google.auth as _google_auth
    _ORIG_DEFAULT = _google_auth.default
except Exception:   # pragma: no cover
    _ORIG_DEFAULT = None
_ADC_FILE_MEM: dict = {}


def _google_default():
    """google.auth.default(), minus its slow part: for an ADC *file* without a project, google-auth shells out to
    `gcloud config config-helper` (~1.7 s) just to find a project, on every call. The file is loaded directly instead
    (same credentials) and the loaded credentials are reused while the file is unchanged. Anything else (metadata
    server, workload identity, a patched google.auth.default in tests) goes through google.auth.default."""
    import google.auth
    if google.auth.default is _ORIG_DEFAULT:
        from google.auth import _cloud_sdk
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip() or \
            _cloud_sdk.get_application_default_credentials_path()
        try:
            st = os.stat(path)
        except OSError:
            st = None
        if st is not None:
            key = (path, st.st_mtime, st.st_size)
            hit = _ADC_FILE_MEM.get("file")
            if hit and hit[0] == key:
                return hit[1], hit[2]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                creds, proj = google.auth.load_credentials_from_file(path, scopes=[CLOUD_SCOPE])
            _ADC_FILE_MEM["file"] = (key, creds, proj)
            return creds, proj
    return google.auth.default(scopes=[CLOUD_SCOPE])


def _adc():
    """(credentials, adc_project) with a fresh token. Raises on missing/invalid credentials."""
    import google.auth.transport.requests

    creds, adc_proj = _google_default()
    if not creds.valid:
        creds.refresh(google.auth.transport.requests.Request())
    return creds, adc_proj


_TOKENINFO_MEM: dict[str, tuple[float, str | None]] = {}


def _identity_key(creds) -> str:
    """Stable, non-secret key per identity (hash), for caches."""
    parts = [credential_type(creds), str(getattr(creds, "service_account_email", "") or ""),
             str(getattr(creds, "refresh_token", "") or ""), str(getattr(creds, "client_id", "") or "")]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _adc_account(creds, ctype: str) -> tuple[str | None, str | None]:
    """(email, how it was found) for the ADC identity; never raises."""
    try:
        if ctype in ("service_account", "impersonated", "external_account"):
            email = getattr(creds, "service_account_email", None) or getattr(creds, "signer_email", None)
            if email:
                return email, "credentials"
        if ctype == "compute_metadata":
            email = getattr(creds, "service_account_email", None)
            if email and email != "default":
                return email, "metadata"
            code, body = _http_get(METADATA_EMAIL_URL, headers={"Metadata-Flavor": "Google"}, timeout=2)
            text = (body.get("text") or "").strip() if code == 200 else ""
            return (text or None), ("metadata" if text else None)
        if ctype == "user" and getattr(creds, "token", None):
            key = hashlib.sha256(creds.token.encode()).hexdigest()
            hit = _TOKENINFO_MEM.get(key)
            if hit and time.time() - hit[0] < 600:
                return hit[1], ("tokeninfo" if hit[1] else None)
            code, body = _http_get(TOKENINFO_URL, params={"access_token": creds.token}, timeout=5)
            email = body.get("email") if code == 200 else None
            _TOKENINFO_MEM.clear()
            _TOKENINFO_MEM[key] = (time.time(), email)
            if email:
                return email, "tokeninfo"
    except Exception:
        pass
    return None, None


SERVICE_IDENTITY_TYPES = ("compute_metadata", "service_account")


def gcloud_available() -> bool:
    return shutil.which("gcloud") is not None


def sign_in_available(ctype: str | None, *, gcloud_ok: bool, runtime: str | None, key_file: bool) -> bool:
    """Whether the browser sign-in can work and matters. It needs gcloud. A service-account key file in
    GOOGLE_APPLICATION_CREDENTIALS takes precedence over a signed-in user, and on Cloud Run the runtime service
    account is the identity, so both hide it. Metadata credentials on a VM or Cloud Workstation keep it: an ADC
    login there takes precedence over the machine's service account."""
    if not gcloud_ok:
        return False
    if ctype == "service_account" and key_file:
        return False
    if ctype in SERVICE_IDENTITY_TYPES and runtime == "cloud_run":
        return False
    return True


def status() -> dict:
    """Current sign-in state: which identity Application Default Credentials resolve to, gcloud's account, project."""
    _reap_login()
    from concurrent.futures import ThreadPoolExecutor
    # The slow parts run side by side: gcloud's token check (a subprocess, cached briefly), gcloud config reads
    # (files, or subprocesses when the files can't be read) and the ADC token / tokeninfo lookup below.
    pool = ThreadPoolExecutor(max_workers=3)
    tok_f = pool.submit(_run_cached, ["gcloud", "auth", "print-access-token"], 10)
    acc_f = pool.submit(gcloud_config_value, "account")
    gcloud_project = default_project()
    _APP["default"], _APP["default_at"] = gcloud_project, time.time()
    project = _APP["project"] or gcloud_project
    account = acc_f.result()

    adc_ok = False
    adc_error = None
    ctype = None
    adc_account = adc_account_source = None
    quota_project = None
    try:
        creds, adc_proj = _adc()
        if not project and adc_proj:
            project = adc_proj
        adc_ok = bool(creds.token)
        ctype = credential_type(creds)
        quota_project = getattr(creds, "quota_project_id", None)
        adc_account, adc_account_source = _adc_account(creds, ctype)
    except Exception as e:
        adc_error = str(e)[:200]

    gcloud_token_ok = False
    rc_tok, tok_out = tok_f.result()
    pool.shutdown(wait=False)
    if rc_tok == 0 and tok_out and not tok_out.startswith("ERROR"):
        gcloud_token_ok = True
    tok_msg = "" if gcloud_token_ok else tok_out[:200]

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    authenticated = adc_ok or gcloud_token_ok
    login = _LOGIN.get("id")
    gcloud_ok = gcloud_available()
    runtime = "cloud_run" if os.environ.get("K_SERVICE") else None
    service_identity = adc_ok and ctype in SERVICE_IDENTITY_TYPES
    return {
        "authenticated": authenticated,
        "adc_ok": adc_ok,
        "gcloud_token_ok": gcloud_token_ok,
        "account": account,                       # gcloud CLI's active account
        "credential_type": ctype,                 # user | service_account | compute_metadata | external_account | ...
        "adc_account": adc_account,               # the identity ADC resolved to, when knowable
        "adc_account_source": adc_account_source,  # tokeninfo | credentials | metadata
        "quota_project": quota_project,           # the quota project stamped in the ADC file (informational)
        "gcloud_project": gcloud_project,         # the machine default: $GOOGLE_CLOUD_PROJECT or gcloud config
        "app_project": real_project(project),     # what this app uses for its calls (resource paths + quota)
        "app_project_source": "app" if _APP["project"] else ("default" if project else None),
        # The app sets the quota project per call, so a different ADC quota project no longer breaks anything; the UI
        # mentions it so the user knows where other tools' calls are billed.
        "quota_project_differs": bool(adc_ok and quota_project and real_project(project)
                                      and quota_project != real_project(project)),
        "google_application_credentials": bool(gac),
        "adc_source": ("GOOGLE_APPLICATION_CREDENTIALS" if gac else
                       "metadata_server" if ctype == "compute_metadata" else
                       "gcloud_adc_file" if adc_ok else None),
        "project": project or "my-gcp-project",
        "project_configured": bool(project),
        "login_in_progress": bool(login),
        "gcloud_available": gcloud_ok,
        "runtime": runtime,                       # "cloud_run" when K_SERVICE is set, else None
        # The app runs as a service account (Cloud Run's runtime SA, a key file): the UI shows "Running as <SA>"
        # and hides "Sign in", which needs gcloud and couldn't override that identity anyway.
        "service_identity": service_identity,
        "sign_in_available": sign_in_available(ctype if adc_ok else None, gcloud_ok=gcloud_ok, runtime=runtime,
                                               key_file=bool(gac)),
        "error": None if authenticated else (adc_error or tok_msg),
        "adc_error": adc_error,
    }


def set_project(project: str | None) -> dict:
    """Choose the app's project (in memory, for this server process). Never runs `gcloud config set project` and
    never changes the environment: the machine's gcloud default and the ADC file stay as they are. Empty or the
    placeholder `my-gcp-project` goes back to the machine default."""
    project = (project or "").strip()
    if project and not PROJECT_ID_RE.match(project):
        raise AuthError("That doesn't look like a project ID.", 400)
    _APP["project"] = real_project(project)
    return status()


def identity() -> str | None:
    """The account ADC currently resolves to (email), or None; cheap (tokeninfo is memoized). Never raises."""
    try:
        creds, _ = _adc()
        return _adc_account(creds, credential_type(creds))[0]
    except Exception:
        return None


def project_access(project: str | None) -> dict:
    """Whether the signed-in identity can use `project`: {project, ok: True/False/None, message}. Used after an
    account switch, so the app doesn't silently keep a project the new account can't use."""
    proj = real_project(project)
    if not proj:
        return {"project": None, "ok": None, "message": None}
    try:
        data = _load_projects()
    except AuthError as e:
        return {"project": proj, "ok": None, "message": str(e)}
    ids = {p["project_id"] for p in data.get("projects") or []}
    if proj in ids:
        return {"project": proj, "ok": True, "message": None}
    if data.get("source") in ("resource_manager", "gcloud", "stub") and ids:
        return {"project": proj, "ok": False,
                "message": f"The signed-in account can't see project {proj}. Choose a project for this app."}
    return {"project": proj, "ok": None, "message": data.get("message")}


# ----------------------------------------------------------------------------------------------- browser sign-in
_LOGIN: dict = {}
_LOGIN_LOCK = threading.RLock()


def _login_cmd(mode: str) -> list[str]:
    if mode == "gcloud":
        return ["gcloud", "auth", "login", "--no-launch-browser"]
    return ["gcloud", "auth", "application-default", "login", "--no-launch-browser"]


def _pump(stream, sink: list, cond: threading.Condition) -> None:
    """Copy a pipe into ``sink`` (list of str chunks) as bytes arrive (prompts have no trailing newline)."""
    try:
        fd = stream.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            with cond:
                sink.append(chunk.decode("utf-8", errors="replace"))
                cond.notify_all()
    except Exception:
        pass
    finally:
        with cond:
            sink.append("")   # EOF marker keeps the list non-empty
            cond.notify_all()


def _output(st: dict) -> str:
    return "".join(st["out"]) + "".join(st["err"])


def find_login_url(text: str) -> str | None:
    """The Google sign-in URL in gcloud's output, once it is complete (followed by whitespace)."""
    for m in URL_RE.finditer(text):
        end = m.end()
        if end < len(text) and text[end].isspace():
            return m.group(0)
    return None


def _kill(st: dict) -> None:
    proc = st.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _reap_login() -> None:
    """Kill an abandoned sign-in (older than LOGIN_TTL_S) or forget one whose process already exited."""
    with _LOGIN_LOCK:
        if not _LOGIN:
            return
        if time.time() - _LOGIN["started"] > LOGIN_TTL_S or _LOGIN["proc"].poll() is not None:
            _kill(_LOGIN)
            _LOGIN.clear()


def _trim(text: str, url: str | None) -> str:
    if url:
        text = text.replace(url, "<sign-in link>")
    text = re.sub(r"Once finished, enter the verification code provided in your browser:\s*", "", text)
    text = re.sub(r"Go to the following link in your browser, and complete the sign-in\s+prompts:\s*", "", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[-600:]


def login_start(mode: str = "adc", force: bool = False) -> dict:
    """Start gcloud's no-launch-browser sign-in in the background; returns {login_id, url, mode} within ~20 s."""
    mode = "gcloud" if mode == "gcloud" else "adc"
    _reap_login()
    with _LOGIN_LOCK:
        if _LOGIN:
            if force:
                _kill(_LOGIN)
                _LOGIN.clear()
            elif _LOGIN["mode"] == mode:
                return {"login_id": _LOGIN["id"], "url": _LOGIN["url"], "mode": mode, "resumed": True}
            else:
                raise AuthError("Another sign-in is already in progress. Finish or cancel it first.", 409)
        env = {**os.environ, "CLOUDSDK_CORE_DISABLE_PROMPTS": "1", "PYTHONUNBUFFERED": "1"}
        # Prompts are disabled so gcloud doesn't stop to ask Y/n on a VM or when GOOGLE_APPLICATION_CREDENTIALS is set
        # (that question would consume the pasted code); the code itself is read with input(), which still works.
        try:
            proc = subprocess.Popen(_login_cmd(mode), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, start_new_session=True)
        except FileNotFoundError:
            raise AuthError("The gcloud CLI isn't installed on the machine running this app. Install the Google "
                            "Cloud SDK there, or set GOOGLE_APPLICATION_CREDENTIALS to a key file.", 503)
        except Exception as e:
            raise AuthError(f"Couldn't start gcloud: {e}", 500)
        cond = threading.Condition()
        st = {"id": uuid.uuid4().hex, "mode": mode, "proc": proc, "started": time.time(), "url": None,
              "out": [], "err": [], "cond": cond}
        for stream, sink in ((proc.stdout, st["out"]), (proc.stderr, st["err"])):
            threading.Thread(target=_pump, args=(stream, sink, cond), daemon=True).start()

        deadline = time.time() + LOGIN_START_TIMEOUT_S
        url = None
        with cond:
            while True:
                url = find_login_url(_output(st))
                if url or proc.poll() is not None or time.time() >= deadline:
                    break
                cond.wait(timeout=0.2)
        if not url or proc.poll() is not None:
            time.sleep(0.1)   # let the pipe readers drain
            _kill(st)
            msg = _trim(_output(st), None)
            if proc.returncode not in (None, 0) and msg:
                raise AuthError(f"gcloud stopped before showing a sign-in link: {msg}", 502)
            raise AuthError("gcloud didn't print a Google sign-in link within "
                            f"{LOGIN_START_TIMEOUT_S} s.{(' Output: ' + msg) if msg else ''}", 504)
        st["url"] = url
        with cond:
            st["url_at"] = (len("".join(st["out"])), len("".join(st["err"])))
        _LOGIN.clear()
        _LOGIN.update(st)
        return {"login_id": st["id"], "url": url, "mode": mode, "resumed": False}


def _take_login(login_id: str | None) -> dict:
    _reap_login()
    with _LOGIN_LOCK:
        if not _LOGIN or (login_id and login_id != _LOGIN["id"]):
            raise AuthError("This sign-in expired or was cancelled. Start again.", 404)
        st = dict(_LOGIN)
        _LOGIN.clear()
        return st


def login_finish(login_id: str, code: str) -> dict:
    """Send the code Google showed to gcloud, wait for it to exit; {ok, message, mode}."""
    code = (code or "").strip()
    if not code:
        raise AuthError("Paste the code Google showed you.", 400)
    if any(c in code for c in "\r\n"):
        raise AuthError("The code should be a single line.", 400)
    st = _take_login(login_id)
    proc = st["proc"]
    try:
        proc.stdin.write((code + "\n").encode())
        proc.stdin.flush()
        proc.stdin.close()
    except Exception:
        pass
    try:
        rc = proc.wait(timeout=LOGIN_FINISH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill(st)
        return {"ok": False, "mode": st["mode"],
                "message": f"gcloud didn't finish within {LOGIN_FINISH_TIMEOUT_S} s. Start the sign-in again."}
    time.sleep(0.1)   # let the pipe readers drain
    o, e = st.get("url_at", (0, 0))
    msg = _trim("".join(st["out"])[o:] + "\n" + "".join(st["err"])[e:], st["url"])
    _invalidate_caches()
    return {"ok": rc == 0, "mode": st["mode"], "message": msg or ("Signed in." if rc == 0 else f"gcloud exited {rc}.")}


def login_cancel(login_id: str | None = None) -> dict:
    """Kill the in-progress sign-in (any, when login_id is empty)."""
    try:
        st = _take_login(login_id)
    except AuthError:
        return {"cancelled": False}
    _kill(st)
    return {"cancelled": True}


@atexit.register
def _kill_login_at_exit() -> None:
    with _LOGIN_LOCK:
        if _LOGIN:
            _kill(_LOGIN)
            _LOGIN.clear()


def trigger_login(project: str | None = None, mode: str = "adc") -> dict:
    """Backward-compatible entry point: sets the project, then starts the browser-completed sign-in (non-blocking)."""
    if project and project != "my-gcp-project":
        set_project(project)
    st = status()
    try:
        st.update(login_start(mode))
        st["login_started"] = True
    except AuthError as e:
        st["login_started"] = False
        st["login_output"] = str(e)
    return st


# ----------------------------------------------------------------------------------------------- projects
_PROJECTS_MEM: dict[str, tuple[float, dict]] = {}


def _invalidate_caches() -> None:
    """Forget per-identity caches (project lists, tokeninfo) and the memoized default project."""
    _PROJECTS_MEM.clear()
    _TOKENINFO_MEM.clear()
    _GCLOUD_MEM.clear()
    _ADC_FILE_MEM.clear()
    _APP["default_at"] = 0.0


def _quota_refused(body: dict) -> bool:
    """A 4xx caused by the quota project itself (the identity can't use it, or the API is off there)."""
    text = json.dumps(body)[:4000]
    return any(k in text for k in ("USER_PROJECT_DENIED", "SERVICE_DISABLED", "serviceusage.services.use",
                                   "quota project", "has not been used in project"))


def _get_billed(url: str, token: str, project: str | None, **kw) -> tuple[int, dict]:
    """GET with `project` as the quota project (x-goog-user-project). If that project refuses the call, retry
    without it: identity-wide calls (listing projects, checking another project) must still work right after an
    account switch, before a project this account can use has been chosen."""
    proj = real_project(project)
    code, body = _http_get(url, token=token, headers={"x-goog-user-project": proj} if proj else None, **kw)
    if proj and code in (400, 403) and _quota_refused(body):
        code, body = _http_get(url, token=token, **kw)
    return code, body


def _project_row(project_id: str, name: str | None, number) -> dict:
    return {"project_id": project_id, "name": name or project_id, "project_number": str(number) if number else None}


def _search_projects(token: str, project: str | None = None) -> tuple[list[dict] | None, int, str | None]:
    """Cloud Resource Manager v3 projects:search (ACTIVE only, paged, capped). (rows | None, http status, error)."""
    rows: list[dict] = []
    page = None
    for _ in range(20):
        params = {"query": "state:ACTIVE", "pageSize": 300}
        if page:
            params["pageToken"] = page
        code, body = _get_billed(CRM_SEARCH_URL, token, project, params=params, timeout=15)
        if code != 200:
            err = (body.get("error") or {}) if isinstance(body.get("error"), dict) else {"message": body.get("error")}
            return (rows or None) if code == 0 else None, code, str(err.get("message") or body.get("text") or code)[:300]
        for p in body.get("projects") or []:
            if (p.get("state") or "ACTIVE") != "ACTIVE" or not p.get("projectId"):
                continue
            num = (p.get("name") or "").split("/")[-1] if (p.get("name") or "").startswith("projects/") else None
            rows.append(_project_row(p["projectId"], p.get("displayName"), num))
            if len(rows) >= PROJECTS_CAP:
                return rows, 200, None
        page = body.get("nextPageToken")
        if not page:
            break
    return rows, 200, None


def _gcloud_projects() -> tuple[list[dict] | None, str | None]:
    rc, out = _run(["gcloud", "projects", "list", "--format=json", "--filter=lifecycleState:ACTIVE",
                    f"--limit={PROJECTS_CAP}"], timeout=40)
    if rc != 0:
        return None, out[-300:]
    try:
        data = json.loads(out[out.index("["):] if "[" in out else out)
    except Exception:
        return None, "Couldn't read gcloud's project list."
    rows = [_project_row(p["projectId"], p.get("name"), p.get("projectNumber")) for p in data
            if p.get("projectId") and (p.get("lifecycleState") or "ACTIVE") == "ACTIVE"]
    return rows[:PROJECTS_CAP], None


def _is_untitled(p: dict) -> bool:
    return p["project_id"].startswith("sys-") or (p.get("name") or "").lower().startswith("untitled project")


def sort_projects(rows: list[dict], current: str | None) -> list[dict]:
    """Current project first, then by display name; ``sys-…`` / "Untitled project" ones last."""
    return sorted(rows, key=lambda p: (p["project_id"] != current, _is_untitled(p),
                                       (p.get("name") or p["project_id"]).casefold(), p["project_id"]))


def filter_projects(rows: list[dict], q: str | None) -> list[dict]:
    q = (q or "").strip().casefold()
    if not q:
        return rows
    return [p for p in rows if q in p["project_id"].casefold() or q in (p.get("name") or "").casefold()
            or q == (p.get("project_number") or "")]


def _load_projects() -> dict:
    """{projects, source, message} for the signed-in identity, cached PROJECTS_TTL_S. Raises AuthError(401)."""
    stub = os.environ.get(PROJECTS_STUB_ENV)
    if stub:
        rows = json.loads(open(stub).read())
        return {"projects": [_project_row(p["project_id"], p.get("name"), p.get("project_number")) for p in rows],
                "source": "stub", "message": None}
    creds = None
    adc_error = None
    try:
        creds, _ = _adc()
    except Exception as e:
        adc_error = str(e)[:200]
    key = _identity_key(creds) if creds is not None else "gcloud"
    hit = _PROJECTS_MEM.get(key)
    if hit and time.time() - hit[0] < PROJECTS_TTL_S:
        return hit[1]

    api_msg = None
    if creds is not None:
        rows, code, err = _search_projects(creds.token, app_project())
        if rows is not None and code == 200:
            res = {"projects": rows, "source": "resource_manager", "message": None}
            _PROJECTS_MEM[key] = (time.time(), res)
            return res
        api_msg = err
        if code == 401:
            adc_error = err or "Credentials were rejected."
    rows, gerr = _gcloud_projects()
    if rows is not None:
        res = {"projects": rows, "source": "gcloud", "message": None}
        _PROJECTS_MEM[key] = (time.time(), res)
        return res
    if creds is None or adc_error:
        raise AuthError("Sign in to Google Cloud to see your projects." +
                        (f" ({adc_error})" if adc_error and creds is not None else ""), 401)
    res = {"projects": [], "source": "none",
           "message": f"Couldn't list your projects ({api_msg or gerr or 'permission denied'}). You can still type a project ID."}
    return res


def list_projects(q: str | None = None, current: str | None = None) -> dict:
    """Projects the signed-in identity can access: {projects: [{project_id, name, project_number}], current, ...}."""
    data = _load_projects()
    current = current or app_project()
    rows = sort_projects(filter_projects(data["projects"], q), current)
    return {"projects": rows, "total": len(data["projects"]), "current": current,
            "source": data["source"], "message": data["message"]}


SERVICES = [
    ("aiplatform.googleapis.com", "Agent Platform API"),
    ("cloudbilling.googleapis.com", "Cloud Billing API"),
]


def check_project(project_id: str) -> dict:
    """Whether the Agent Platform and Cloud Billing APIs are enabled (Service Usage API). Never raises for API errors."""
    if not PROJECT_ID_RE.match(project_id or ""):
        raise AuthError("That doesn't look like a project ID.", 400)
    out = {"project_id": project_id, "services": [], "message": None}
    try:
        creds, _ = _adc()
        token = creds.token
    except Exception:
        out["message"] = "Sign in to check which APIs are enabled."
        token = None
    for svc, title in SERVICES:
        row = {"name": svc, "title": title, "enabled": None,
               "enable_url": f"https://console.cloud.google.com/apis/library/{svc}?project={project_id}"}
        if token:
            code, body = _get_billed(SERVICE_USAGE_URL.format(project=project_id, service=svc), token, project_id,
                                     timeout=8)
            if code == 200:
                row["enabled"] = body.get("state") == "ENABLED"
            elif not out["message"]:
                err = body.get("error") if isinstance(body.get("error"), dict) else {}
                out["message"] = ("You don't have permission to see this project's APIs." if code == 403 else
                                  f"Couldn't check this project's APIs ({err.get('message') or code}).")[:300]
        out["services"].append(row)
    return out
