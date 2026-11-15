"""The Hugging Face access token: where it comes from, where it is stored, and how it is checked.

Gated repos (Llama, some Gemma) answer the Hub's model API without a token but refuse `config.json` and
`tokenizer.json` (401): their context / active parameters then fall back to built-in rules and prompts are counted
with a character heuristic. A read-only token fixes that.

Precedence (first one set wins):
  1. `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN` environment variables (read-only from the app: shown as "env")
  2. the token saved from the app's Settings, in a per-user config file `<config dir>/hf_token`
     (`~/.config/aiops-dilemma/hf_token`, or `$XDG_CONFIG_HOME/aiops-dilemma`; `DILEMMA_CONFIG_DIR` overrides the
     directory, e.g. for tests), file mode 0600 inside a 0700 directory
  3. when that directory can't be written (read-only filesystem, e.g. a container), the token saved from Settings is
     kept in this process's memory only (source "memory") and lost on restart.

The token is never written to sessions, workload.yaml, catalog caches, anything under runs/ or the repo, never
logged, and never returned by the API (only `hint()`: "hf_…abcd").

`DILEMMA_TEST_HF_WHOAMI_STUB=1` (tests and UI checks only) replaces the whoami call: tokens starting with
"hf_valid" are accepted as user "test-user" (orgs "test-org") and never sent to huggingface.co, anything else is
refused with 401.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
FILE_NAME = "hf_token"
STUB_ENV = "DILEMMA_TEST_HF_WHOAMI_STUB"

_LOCK = threading.Lock()
_MEM: dict = {"token": None}          # read-only filesystem fallback
_WHO: dict[str, dict] = {}            # sha256(token) -> whoami result (user name / orgs), this process only


def config_dir() -> Path:
    d = os.environ.get("DILEMMA_CONFIG_DIR", "").strip()
    if d:
        return Path(d)
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or str(Path.home() / ".config")
    return Path(base) / "aiops-dilemma"


def token_path() -> Path:
    return config_dir() / FILE_NAME


def _env_token() -> tuple[str | None, str | None]:
    for k in ENV_VARS:
        v = os.environ.get(k, "").strip()
        if v:
            return v, k
    return None, None


def _file_token() -> str | None:
    try:
        v = token_path().read_text().strip()
        return v or None
    except OSError:
        return None


def current() -> tuple[str | None, str]:
    """(token, source) with source env | file | memory | none."""
    tok, _ = _env_token()
    if tok:
        return tok, "env"
    with _LOCK:
        mem = _MEM["token"]
    if mem:
        return mem, "memory"
    tok = _file_token()
    return (tok, "file") if tok else (None, "none")


def token() -> str | None:
    return current()[0]


def headers(extra: dict | None = None) -> dict:
    """HTTP headers for a huggingface.co request: User-Agent, plus `Authorization: Bearer <token>` when one is set."""
    h = {"User-Agent": "aiops-dilemma", **(extra or {})}
    tok = token()
    if tok and _stub() and tok.startswith("hf_valid"):
        return h          # test stub tokens are never sent to huggingface.co
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _stub() -> bool:
    return os.environ.get(STUB_ENV, "").strip() not in ("", "0", "false")


def hint(tok: str | None) -> str | None:
    """A recognisable but useless fragment: 'hf_…abcd'."""
    if not tok:
        return None
    return f"{tok[:3]}…{tok[-4:]}" if len(tok) >= 10 else "…"


def _key(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()


def validate(tok: str, timeout: float = 10.0) -> dict:
    """Check a token with the Hub's whoami-v2. {ok, user, orgs} or {ok: False, error, status}."""
    tok = (tok or "").strip()
    if not tok:
        return {"ok": False, "status": 400, "error": "Enter a token."}
    if any(c.isspace() for c in tok) or len(tok) > 400:
        return {"ok": False, "status": 400, "error": "That doesn't look like a Hugging Face token."}
    if _stub():
        if tok.startswith("hf_valid"):
            res = {"ok": True, "user": "test-user", "orgs": ["test-org"]}
        else:
            res = {"ok": False, "status": 401, "error": "Hugging Face didn't accept this token (invalid or revoked)."}
    else:
        req = urllib.request.Request(WHOAMI_URL, headers={"User-Agent": "aiops-dilemma",
                                                          "Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.load(r)
            orgs = [o.get("name") for o in (body.get("orgs") or []) if isinstance(o, dict) and o.get("name")]
            res = {"ok": True, "user": body.get("name"), "orgs": orgs}
        except urllib.error.HTTPError as e:
            res = {"ok": False, "status": e.code,
                   "error": ("Hugging Face didn't accept this token (invalid or revoked)." if e.code in (401, 403)
                             else f"Hugging Face answered HTTP {e.code}; try again later.")}
        except Exception as e:   # network
            res = {"ok": False, "status": 502, "error": f"Couldn't reach Hugging Face ({type(e).__name__}); try again."}
    if res.get("ok"):
        _WHO[_key(tok)] = {"user": res.get("user"), "orgs": res.get("orgs") or []}
    return res


def _write_file(tok: str) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    tmp = d / (FILE_NAME + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (tok + "\n").encode())
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, d / FILE_NAME)


def save(tok: str) -> str:
    """Store a (validated) token: the config file, or memory when the filesystem is read-only. Returns where."""
    tok = tok.strip()
    with _LOCK:
        try:
            _write_file(tok)
            _MEM["token"] = None
            return "file"
        except OSError as e:
            log.warning("Hugging Face token: config directory not writable (%s); kept in memory only",
                        type(e).__name__)
            _MEM["token"] = tok
            return "memory"


def delete() -> bool:
    """Forget the saved token (file and memory). True if there was one. The environment variable can't be removed."""
    with _LOCK:
        had = bool(_MEM["token"])
        _MEM["token"] = None
        try:
            token_path().unlink()
            had = True
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("Hugging Face token: can't remove the file (%s)", type(e).__name__)
    return had


def status(check: bool = True) -> dict:
    """{set, source, hint, user, orgs, env_var, path}: never the token. `check` validates a token whose user isn't
    known yet in this process (one whoami call, cached)."""
    tok, src = current()
    out = {"set": bool(tok), "source": src, "hint": hint(tok), "user": None, "orgs": [],
           "env_var": _env_token()[1] if src == "env" else None,
           "path": str(token_path()) if src == "file" else None}
    if tok:
        who = _WHO.get(_key(tok))
        if who is None and check:
            res = validate(tok, timeout=5.0)
            who = _WHO.get(_key(tok))
            if not res.get("ok"):
                out["invalid"] = res.get("status") in (401, 403)
                out["error"] = res.get("error")
        if who:
            out.update(user=who.get("user"), orgs=who.get("orgs") or [])
    return out


def job_env() -> dict:
    """Environment for a child process (benchmark job) when the token lives only in this process's memory."""
    tok, src = current()
    return {"HF_TOKEN": tok} if tok and src == "memory" else {}
