"""Identity-Aware Proxy: verify the signed header IAP adds to every request, and who it says the user is.

On the Cloud Run deployment every request must carry `x-goog-iap-jwt-assertion`, an ES256 JWT signed by IAP
(https://cloud.google.com/iap/docs/signed-headers-howto). Checking it in the app means a request that reaches the
service some other way (e.g. a principal with run.invoker calling the run.app URL with its own identity token) is
refused, whatever the path. Checks: signature against IAP's public keys (cached 1 h), issuer
`https://cloud.google.com/iap`, expiry, and the exact audience: `DILEMMA_IAP_AUDIENCE` (the deploy script sets it),
else built from Cloud Run's metadata server (`/projects/<number>/locations/<region>/services/<K_SERVICE>`). When
neither is available every request is refused (fail closed); a failed metadata read is retried after 30 s.

Required in cloud mode (paths.cloud_mode()); `DILEMMA_REQUIRE_IAP=0|1` overrides. `verify` returns
{"email", "sub"} or raises IapError.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request

from aiops_dilemma import paths

CERTS_URL = "https://www.gstatic.com/iap/verify/public_key"
ISSUER = "https://cloud.google.com/iap"
HEADER = "x-goog-iap-jwt-assertion"
METADATA = "http://metadata.google.internal/computeMetadata/v1/"
_LOCK = threading.Lock()
_CERTS: dict = {"at": 0.0, "keys": None}
_AUD: dict = {"value": None, "failed_at": 0.0}
AUD_RETRY_S = 30


class IapError(Exception):
    pass


def required() -> bool:
    f = paths._flag("DILEMMA_REQUIRE_IAP")
    return f if f is not None else paths.cloud_mode()


def _get(url: str, headers: dict | None = None, timeout: float = 5.0) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def _certs() -> dict:
    with _LOCK:
        if _CERTS["keys"] and time.time() - _CERTS["at"] < 3600:
            return _CERTS["keys"]
    keys = json.loads(_get(CERTS_URL))
    with _LOCK:
        _CERTS.update(at=time.time(), keys=keys)
    return keys


def expected_audience() -> str | None:
    """The exact audience for this Cloud Run service, or None when it can't be determined (then nothing passes)."""
    env = os.environ.get("DILEMMA_IAP_AUDIENCE", "").strip()
    if env:
        return env
    with _LOCK:
        if _AUD["value"]:
            return _AUD["value"]
        if time.time() - _AUD["failed_at"] < AUD_RETRY_S:   # don't hammer the metadata server, but never give up
            return None
    value = None
    svc = os.environ.get("K_SERVICE")
    if svc:
        try:
            h = {"Metadata-Flavor": "Google"}
            number = _get(METADATA + "project/numeric-project-id", h).strip()
            region = _get(METADATA + "instance/region", h).strip().rsplit("/", 1)[-1]   # projects/N/regions/R
            if number.isdigit() and region:
                value = f"/projects/{number}/locations/{region}/services/{svc}"
        except Exception:
            value = None
    with _LOCK:
        if value:
            _AUD["value"] = value
        else:
            _AUD["failed_at"] = time.time()
    return value


def _aud_ok(aud) -> bool:
    """Exactly the expected audience. No pattern fallback: an IAP token issued for another project's service with
    the same name must not pass, and an unknown audience means refuse (fail closed)."""
    auds = aud if isinstance(aud, list) else [aud]
    want = expected_audience()
    return bool(want) and want in auds


def verify(token: str | None) -> dict:
    if not token:
        raise IapError("missing IAP header")
    from google.auth import jwt
    try:
        claims = jwt.decode(token, certs=_certs(), verify=True, audience=None)
    except Exception as e:
        raise IapError(f"invalid IAP header: {type(e).__name__}")
    if claims.get("iss") != ISSUER:
        raise IapError("invalid IAP header: issuer")
    if not _aud_ok(claims.get("aud")):
        raise IapError("invalid IAP header: audience")
    email = claims.get("email") or ""
    return {"email": email, "sub": claims.get("sub") or ""}
