"""Cloud mode (Cloud Run behind IAP): IAP header verified on every request, cloud-only features turned off, file
paths from configs confined to the sessions folder, audit lines naming the IAP user, job.json started_by."""
import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from google.auth import crypt, jwt

from aiops_dilemma import iap, paths, prompts, server, traffic

AUD = "/projects/123456789/locations/us-central1/services/aiops-dilemma"


def _key():
    k = ec.generate_private_key(ec.SECP256R1())
    priv = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                           serialization.NoEncryption()).decode()
    pub = k.public_key().public_bytes(serialization.Encoding.PEM,
                                      serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


GOOD_PRIV, GOOD_PUB = _key()
OTHER_PRIV, _ = _key()


def token(priv=GOOD_PRIV, kid="k1", **over):
    now = int(time.time())
    claims = {"iss": iap.ISSUER, "aud": AUD, "email": "alice@example.com", "sub": "accounts.google.com:1",
              "iat": now, "exp": now + 600}
    claims.update(over)
    return jwt.encode(crypt.ES256Signer.from_string(priv, kid), claims).decode()


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "aiops-dilemma")
    monkeypatch.setenv("DILEMMA_IAP_AUDIENCE", AUD)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("DILEMMA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(server, "CLOUD", True)
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(iap, "_certs", lambda: {"k1": GOOD_PUB})
    return TestClient(server.app)


def H(tok=None):
    return {iap.HEADER: tok if tok is not None else token()}


def test_every_path_needs_a_valid_iap_header(cloud):
    for path in ("/", "/api/sessions", "/assets/app.js", "/api/sessions/x/export/report.md", "/api/does-not-exist"):
        assert cloud.get(path).status_code == 401, path
    assert cloud.post("/api/sessions/x/run", json={}).status_code == 401
    bad = [token(priv=OTHER_PRIV), token(aud="/projects/1/locations/us-central1/services/other"),
           token(iss="https://evil.example"), token(exp=int(time.time()) - 60, iat=int(time.time()) - 700),
           "not-a-jwt", ""]
    for t in bad:
        assert cloud.get("/api/sessions", headers=H(t)).status_code == 401
    assert cloud.get("/api/sessions", headers=H()).status_code == 200


def test_no_api_docs_in_cloud_mode(cloud):
    # FastAPI's docs routes are registered at import (local mode in tests); the gate still applies to them.
    assert cloud.get("/docs").status_code == 401


def test_cloud_only_features_are_off(cloud):
    for method, path, body in [("post", "/api/auth/login/start", {}), ("post", "/api/auth/login/finish", {"code": "x"}),
                               ("post", "/api/auth/login/cancel", {}), ("post", "/api/auth/login", {"trigger_oauth": True}),
                               ("post", "/api/settings/hf-token", {"token": "hf_validFAKEtoken0000abcd"}),
                               ("delete", "/api/settings/hf-token", None),
                               ("post", "/api/auth/login", {"project": "someone-elses-project"})]:
        r = getattr(cloud, method)(path, headers=H(), **({"json": body} if body is not None else {}))
        assert r.status_code == 403, (path, r.status_code, r.text)
    r = cloud.get("/api/projects", headers=H()).json()
    assert r["locked"] and [p["project_id"] for p in r["projects"]] == ["my-gcp-project"]
    assert cloud.get("/api/settings", headers=H()).json()["managed"] is True


def test_cloud_flags_in_auth_status(cloud):
    class R:
        class state:
            user = {"email": "alice@example.com"}
    st = server._with_cloud({"adc_ok": True}, R)
    assert st["cloud"] == {"project_locked": "my-gcp-project", "hf_token_managed": True, "sign_in": False,
                           "user": "alice@example.com"}


def test_audit_lines_name_the_iap_user(cloud, capsys):
    cloud.get("/api/sessions", headers=H())                       # reads aren't logged
    cloud.post("/api/sessions/demo", headers=H(), json={"config": {"name": "demo"}})
    cloud.get("/api/sessions", headers=H(token(priv=OTHER_PRIV)))   # denied
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    acts = [ln for ln in lines if ln["audit"]["event"] == "action"]
    assert len(acts) == 1 and acts[0]["audit"]["user"] == "alice@example.com"
    assert acts[0]["audit"]["method"] == "POST" and acts[0]["audit"]["session"] == "demo"
    assert acts[0]["logging.googleapis.com/labels"]["user"] == "alice@example.com"
    denied = [ln for ln in lines if ln["audit"]["event"] == "access_denied"]
    assert denied and denied[0]["severity"] == "WARNING" and denied[0]["audit"]["status"] == 401


def test_local_mode_needs_no_header_and_logs_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(server.app)
    assert c.get("/api/sessions").status_code == 200
    c.post("/api/sessions/demo", json={"config": {"name": "demo"}})
    assert not [ln for ln in capsys.readouterr().out.splitlines() if '"audit"' in ln]


def test_audience_exact_and_fail_closed(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "aiops-dilemma")
    monkeypatch.setattr(iap, "_AUD", {"value": None, "failed_at": 0.0})
    answers = {"project/numeric-project-id": "987", "instance/region": "projects/987/regions/europe-west4"}
    monkeypatch.setattr(iap, "_get", lambda url, headers=None, timeout=5.0: answers[url.split("v1/")[1]])
    assert iap.expected_audience() == "/projects/987/locations/europe-west4/services/aiops-dilemma"
    assert iap._aud_ok("/projects/987/locations/europe-west4/services/aiops-dilemma")
    assert not iap._aud_ok("/projects/111/locations/europe-west4/services/aiops-dilemma")   # same name, other project
    # metadata unreachable: nothing passes (no pattern fallback), and the read is retried later
    monkeypatch.setattr(iap, "_AUD", {"value": None, "failed_at": 0.0})
    calls = []
    def down(*a, **k):
        calls.append(1)
        raise OSError("no metadata")
    monkeypatch.setattr(iap, "_get", down)
    assert not iap._aud_ok("/projects/111/locations/us-east1/services/aiops-dilemma")
    assert not iap._aud_ok("/projects/987/locations/europe-west4/services/aiops-dilemma")
    n = len(calls)
    monkeypatch.setattr(iap, "_AUD", {"value": None, "failed_at": iap._AUD["failed_at"] - iap.AUD_RETRY_S - 1})
    monkeypatch.setattr(iap, "_get", lambda url, headers=None, timeout=5.0: answers[url.split("v1/")[1]])
    assert iap._aud_ok("/projects/987/locations/europe-west4/services/aiops-dilemma")   # recovered after the retry
    assert n >= 1


def test_paths_confined_on_a_shared_deployment(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "s").mkdir(parents=True)
    ok = runs / "s" / "uploaded_prompts.jsonl"
    ok.write_text('{"prompt": "hi"}\n')
    monkeypatch.setenv("DILEMMA_RUNS_DIR", str(runs))
    assert paths.check_readable("/etc/passwd")            # local: anything readable
    monkeypatch.setenv("DILEMMA_CONFINE_PATHS", "1")
    assert paths.check_readable(ok) == ok.resolve()
    for bad in ("/etc/passwd", "/proc/self/environ", str(runs / "s" / ".." / ".." / "secret.txt")):
        with pytest.raises(paths.PathNotAllowed):
            paths.check_readable(bad)
    link = runs / "s" / "link.txt"
    link.symlink_to("/etc/passwd")
    with pytest.raises(paths.PathNotAllowed):
        paths.check_readable(link)
    # the three readers that take a path from a session's config
    cfg = {"_path": str(runs / "s" / "workload.yaml"), "prompts": {"file": "/proc/self/environ"},
           "requests": {}}
    with pytest.raises(SystemExit, match="only files inside the session folder"):
        prompts.from_file(cfg, tmp_path / "out.jsonl")
    with pytest.raises(paths.PathNotAllowed):
        prompts._load_custom_system_prompts({"_path": cfg["_path"], "prompts": {"system_prompts_file": "/etc/hostname"}})
    with pytest.raises(paths.PathNotAllowed):
        traffic.from_csv("/etc/passwd")


def test_cloud_mode_switches(monkeypatch):
    assert not paths.cloud_mode() and not iap.required() and not paths.confine_paths()
    monkeypatch.setenv("K_SERVICE", "aiops-dilemma")
    assert paths.cloud_mode() and iap.required() and paths.confine_paths()
    monkeypatch.setenv("DILEMMA_CLOUD", "0")
    assert not paths.cloud_mode() and not iap.required()


def test_job_records_who_started_it(tmp_path, monkeypatch):
    from aiops_dilemma import jobs
    monkeypatch.setenv("DILEMMA_JOB_USER", "alice@example.com")
    rec = jobs.JobRecorder(tmp_path, steps=["load"], mode="load", source="cli", command="test")
    assert rec.state["started_by"] == "alice@example.com"
