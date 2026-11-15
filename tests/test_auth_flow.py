"""Browser-completed sign-in (fake gcloud over pipes), project listing, API checks, ADC identity detection.

No network: HTTP goes through auth._http_get (patched) and gcloud is tests/fake_gcloud.py on PATH."""

import json
import os
import re
import stat
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aiops_dilemma import auth, server

FAKE = Path(__file__).resolve().parent / "fake_gcloud.py"
FAKE_URL = re.search(r'FAKE_URL = \((.*?)\)\n', FAKE.read_text(), re.S).group(1)
FAKE_URL = "".join(re.findall(r'"([^"]*)"', FAKE_URL))


@pytest.fixture()
def fake_gcloud(tmp_path, monkeypatch):
    """Put a fake `gcloud` first on PATH; returns a reader for its call log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "gcloud"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    logf = tmp_path / "gcloud.log"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GCLOUD_LOG", str(logf))
    monkeypatch.delenv("FAKE_GCLOUD_MODE", raising=False)
    monkeypatch.delenv("FAKE_GCLOUD_PROJECTS", raising=False)
    monkeypatch.delenv(auth.PROJECTS_STUB_ENV, raising=False)
    auth.login_cancel()
    auth._invalidate_caches()
    yield lambda: [json.loads(line) for line in logf.read_text().splitlines()] if logf.exists() else []
    auth.login_cancel()


@pytest.fixture()
def no_adc(monkeypatch):
    """ADC unavailable (so status() stays offline)."""
    import google.auth
    import google.auth.exceptions

    def boom(*a, **k):
        raise google.auth.exceptions.DefaultCredentialsError("no ADC in tests")

    monkeypatch.setattr(google.auth, "default", boom)


def test_find_login_url_requires_complete_url():
    text = f"Go to the following link in your browser:\n\n    https://accounts.google.com/o/oauth2/auth?a=1&b=2\n\nOnce"
    assert auth.find_login_url(text) == "https://accounts.google.com/o/oauth2/auth?a=1&b=2"
    assert auth.find_login_url("    https://accounts.google.com/o/oauth2/auth?a=1&b") is None   # still streaming
    assert auth.find_login_url("no link here\n") is None


def test_login_start_parses_url_and_finish_success(fake_gcloud, no_adc):
    res = auth.login_start("adc")
    assert res["url"] == FAKE_URL and res["login_id"] and res["resumed"] is False
    first = fake_gcloud()[0]
    assert first["argv"] == ["auth", "application-default", "login", "--no-launch-browser"]
    assert first["disable_prompts"] == "1"       # a VM's Y/n prompt must not eat the code
    # one sign-in at a time: the same mode resumes it, another mode is refused
    again = auth.login_start("adc")
    assert again["login_id"] == res["login_id"] and again["resumed"] is True
    with pytest.raises(auth.AuthError) as e:
        auth.login_start("gcloud")
    assert e.value.status == 409
    out = auth.login_finish(res["login_id"], "good-code")
    assert out["ok"] is True and "Credentials saved" in out["message"]
    assert FAKE_URL not in out["message"]
    assert {"argv": ["auth", "application-default", "login", "--no-launch-browser"], "code": "good-code"} in fake_gcloud()
    with pytest.raises(auth.AuthError) as e:          # consumed
        auth.login_finish(res["login_id"], "good-code")
    assert e.value.status == 404


def test_login_finish_failure_reports_message(fake_gcloud, no_adc):
    res = auth.login_start("gcloud")
    assert fake_gcloud()[0]["argv"] == ["auth", "login", "--no-launch-browser"]
    out = auth.login_finish(res["login_id"], "wrong")
    assert out["ok"] is False and "Malformed auth code" in out["message"]
    assert "code_challenge" not in out["message"] and "Once finished" not in out["message"]


def test_login_cancel_kills_process(fake_gcloud, no_adc):
    res = auth.login_start("adc")
    proc = auth._LOGIN["proc"]
    assert proc.poll() is None
    assert auth.login_cancel(res["login_id"]) == {"cancelled": True}
    assert proc.poll() is not None
    assert auth.login_cancel(res["login_id"]) == {"cancelled": False}
    assert auth.login_start("gcloud")["resumed"] is False     # free again


def test_login_stale_is_reaped(fake_gcloud, no_adc, monkeypatch):
    res = auth.login_start("adc")
    proc = auth._LOGIN["proc"]
    auth._LOGIN["started"] -= auth.LOGIN_TTL_S + 1
    with pytest.raises(auth.AuthError):
        auth.login_finish(res["login_id"], "good-code")
    assert proc.poll() is not None


def test_login_start_errors(fake_gcloud, no_adc, monkeypatch):
    monkeypatch.setenv("FAKE_GCLOUD_MODE", "fail_start")
    with pytest.raises(auth.AuthError) as e:
        auth.login_start("adc")
    assert e.value.status == 502 and "something went wrong" in str(e.value)
    monkeypatch.setenv("FAKE_GCLOUD_MODE", "no_url")
    monkeypatch.setattr(auth, "LOGIN_START_TIMEOUT_S", 1)
    t = time.time()
    with pytest.raises(auth.AuthError) as e:
        auth.login_start("adc")
    assert e.value.status == 504 and time.time() - t < 5
    assert not auth._LOGIN
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(auth.AuthError) as e:
        auth.login_start("adc")
    assert e.value.status == 503 and "isn't installed" in str(e.value)


def test_login_endpoints(fake_gcloud, no_adc, monkeypatch):
    refreshed = []
    monkeypatch.setattr(server, "_refresh_catalogs_after_login", lambda all_=False: refreshed.append(1) or True)
    monkeypatch.setattr(server.auth, "status", lambda: {"authenticated": True, "project": "my-gcp-project"})
    c = TestClient(server.app)
    r = c.post("/api/auth/login/start", json={"mode": "adc"})
    assert r.status_code == 200 and r.json()["url"].startswith("https://accounts.google.com/")
    bad = c.post("/api/auth/login/finish", json={"login_id": r.json()["login_id"], "code": ""})
    assert bad.status_code == 400
    fin = c.post("/api/auth/login/finish", json={"login_id": r.json()["login_id"], "code": "good-code"}).json()
    assert fin["login_ok"] is True and fin["authenticated"] is True and fin["catalog_refreshing"] is True
    assert refreshed == [1]
    assert c.post("/api/auth/login/finish", json={"login_id": "nope", "code": "x"}).status_code == 404
    # legacy trigger_oauth no longer blocks: it starts the flow and returns the link
    legacy = c.post("/api/auth/login", json={"trigger_oauth": True}).json()
    assert legacy["login_started"] is True and legacy["url"].startswith("https://accounts.google.com/")
    assert c.post("/api/auth/login/cancel", json={}).json() == {"cancelled": True}


def test_refresh_catalogs_after_login_only_fallback(monkeypatch):
    started = []
    monkeypatch.setattr(server, "_refresh_catalog_bg", lambda key, project, region: started.append((key, project, region)))
    monkeypatch.setattr(server, "_CATALOG_MEM", {
        "default_us-central1": {"auth": {"authenticated": False}, "gemini": {}},
        "my-gcp-project_europe-west4": {"auth": {"authenticated": True}, "gemini": {"gemini_flash": {"source": "fallback"}}},
        "other_us-east1": {"auth": {"authenticated": True}, "gemini": {"gemini_flash": {"source": "billing-catalog"}}},
    })
    assert server._refresh_catalogs_after_login() is True
    assert started == [("default_us-central1", None, "us-central1"),
                       ("my-gcp-project_europe-west4", "my-gcp-project", "europe-west4")]


# ------------------------------------------------------------------------------------------- ADC identity
def _patch_adc(monkeypatch, creds, project=None):
    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda *a, **k: (creds, project))
    monkeypatch.setattr(auth, "_run", lambda cmd, timeout=15: (1, "ERROR: offline") if "print-access-token" in cmd
                        else (0, "you@example.com") if "account" in cmd else (0, "my-gcp-project"))


def test_status_user_credentials(monkeypatch):
    from google.oauth2 import credentials

    creds = credentials.Credentials(token="user-token", refresh_token="r", quota_project_id="my-gcp-project")
    _patch_adc(monkeypatch, creds)
    calls = []

    def fake_get(url, **kw):
        calls.append((url, kw.get("params")))
        return 200, {"email": "you@example.com", "scope": "x"}

    monkeypatch.setattr(auth, "_http_get", fake_get)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    auth._invalidate_caches()
    st = auth.status()
    assert st["credential_type"] == "user" and st["adc_account"] == "you@example.com"
    assert st["adc_account_source"] == "tokeninfo" and st["quota_project"] == "my-gcp-project"
    assert st["google_application_credentials"] is False and st["adc_source"] == "gcloud_adc_file"
    assert st["authenticated"] is True
    assert calls == [(auth.TOKENINFO_URL, {"access_token": "user-token"})]
    assert "user-token" not in json.dumps(st)       # never return tokens
    auth.status()
    assert len(calls) == 1                          # tokeninfo cached per token


def test_status_service_account(monkeypatch, tmp_path):
    from google.oauth2 import service_account

    creds = service_account.Credentials(signer=object(), service_account_email="sa@my-gcp-project.iam.gserviceaccount.com",
                                        token_uri="https://oauth2.googleapis.com/token")
    creds.token = "sa-token"
    _patch_adc(monkeypatch, creds)
    monkeypatch.setattr(auth, "_http_get", lambda *a, **k: pytest.fail("no HTTP needed for a service account"))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "key.json"))
    st = auth.status()
    assert st["credential_type"] == "service_account"
    assert st["adc_account"] == "sa@my-gcp-project.iam.gserviceaccount.com"
    assert st["google_application_credentials"] is True and st["adc_source"] == "GOOGLE_APPLICATION_CREDENTIALS"


def test_status_compute_metadata(monkeypatch):
    from google.auth import compute_engine

    creds = compute_engine.Credentials()
    creds.token = "md-token"
    _patch_adc(monkeypatch, creds)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(auth, "_http_get", lambda url, **kw: (200, {"text": "123-compute@developer.gserviceaccount.com"})
                        if url == auth.METADATA_EMAIL_URL else pytest.fail(url))
    st = auth.status()
    assert st["credential_type"] == "compute_metadata" and st["adc_source"] == "metadata_server"
    assert st["adc_account"] == "123-compute@developer.gserviceaccount.com"


def test_credential_type_external_account():
    from google.auth import identity_pool

    creds = identity_pool.Credentials(audience="//iam.googleapis.com/x", subject_token_type="urn:ietf:params:oauth:token-type:jwt",
                                      token_url="https://sts.googleapis.com/v1/token", credential_source={"file": "/tmp/x"})
    assert auth.credential_type(creds) == "external_account"


# ------------------------------------------------------------------------------------------- projects
class _Creds:
    token = "tok"
    valid = True
    refresh_token = "r"


def _crm_pages(pages):
    """Fake CRM search: list of page bodies chained by nextPageToken."""
    calls = []

    def fake_get(url, *, token=None, params=None, **kw):
        assert url == auth.CRM_SEARCH_URL and token == "tok"
        calls.append(dict(params))
        idx = int(params.get("pageToken") or 0)
        return 200, pages[idx]

    return fake_get, calls


def test_projects_paging_sorting_filter(monkeypatch, fake_gcloud):
    pages = [
        {"projects": [
            {"projectId": "zeta-prod", "displayName": "Zeta prod", "name": "projects/3", "state": "ACTIVE"},
            {"projectId": "sys-12345678901", "displayName": "Untitled project", "name": "projects/4", "state": "ACTIVE"},
            {"projectId": "old-one", "displayName": "Old", "name": "projects/5", "state": "DELETE_REQUESTED"},
        ], "nextPageToken": "1"},
        {"projects": [
            {"projectId": "alpha-dev", "displayName": "alpha dev", "name": "projects/1", "state": "ACTIVE"},
            {"projectId": "my-gcp-project", "displayName": "Mine", "name": "projects/2", "state": "ACTIVE"},
        ]},
    ]
    fake_get, calls = _crm_pages(pages)
    monkeypatch.setattr(auth, "_adc", lambda: (_Creds(), None))
    monkeypatch.setattr(auth, "_http_get", fake_get)
    c = TestClient(server.app)
    r = c.get("/api/projects", params={"current": "my-gcp-project"}).json()
    assert [p["project_id"] for p in r["projects"]] == ["my-gcp-project", "alpha-dev", "zeta-prod", "sys-12345678901"]
    assert r["projects"][0] == {"project_id": "my-gcp-project", "name": "Mine", "project_number": "2"}
    assert r["source"] == "resource_manager" and r["total"] == 4
    assert len(calls) == 2 and calls[0]["query"] == "state:ACTIVE" and calls[1]["pageToken"] == "1"
    # filtered from the cache (no new API calls)
    r = c.get("/api/projects", params={"q": "ZETA", "current": "my-gcp-project"}).json()
    assert [p["project_id"] for p in r["projects"]] == ["zeta-prod"] and len(calls) == 2
    assert [p["project_id"] for p in c.get("/api/projects", params={"q": "alpha d"}).json()["projects"]] == ["alpha-dev"]


def test_projects_fallback_to_gcloud(monkeypatch, fake_gcloud):
    monkeypatch.setattr(auth, "_adc", lambda: (_Creds(), None))
    monkeypatch.setattr(auth, "_http_get", lambda *a, **k: (403, {"error": {"message": "API not enabled"}}))
    monkeypatch.setenv("FAKE_GCLOUD_PROJECTS", json.dumps([
        {"projectId": "b-proj", "name": "B", "projectNumber": "22", "lifecycleState": "ACTIVE"},
        {"projectId": "a-proj", "name": "A", "projectNumber": "11", "lifecycleState": "ACTIVE"},
    ]))
    r = auth.list_projects(current="none")
    assert r["source"] == "gcloud" and [p["project_id"] for p in r["projects"]] == ["a-proj", "b-proj"]
    assert any(e["argv"][:2] == ["projects", "list"] for e in fake_gcloud())


def test_projects_permission_error_is_soft(monkeypatch, fake_gcloud):
    monkeypatch.setattr(auth, "_adc", lambda: (_Creds(), None))
    monkeypatch.setattr(auth, "_http_get", lambda *a, **k: (403, {"error": {"message": "Permission denied"}}))
    r = TestClient(server.app).get("/api/projects", params={"current": "x"})
    assert r.status_code == 200 and r.json()["projects"] == [] and "type a project ID" in r.json()["message"]


def test_projects_unauthenticated_401(monkeypatch, fake_gcloud):
    def no_creds():
        raise RuntimeError("Your default credentials were not found")

    monkeypatch.setattr(auth, "_adc", no_creds)
    r = TestClient(server.app).get("/api/projects", params={"current": "x"})
    assert r.status_code == 401 and "Sign in" in r.json()["detail"]
    # rejected token (API 401) with gcloud also failing -> 401 too
    monkeypatch.setattr(auth, "_adc", lambda: (_Creds(), None))
    monkeypatch.setattr(auth, "_http_get", lambda *a, **k: (401, {"error": {"message": "Request had invalid credentials"}}))
    assert TestClient(server.app).get("/api/projects", params={"current": "x"}).status_code == 401


def test_projects_cap(monkeypatch, fake_gcloud):
    page = {"projects": [{"projectId": f"p-{i:04d}x", "displayName": f"P {i:04d}", "name": f"projects/{i}"}
                         for i in range(300)], "nextPageToken": "0"}
    monkeypatch.setattr(auth, "_adc", lambda: (_Creds(), None))
    monkeypatch.setattr(auth, "_http_get", lambda *a, **k: (200, page))
    assert len(auth.list_projects(current="x")["projects"]) == auth.PROJECTS_CAP


def test_project_check(monkeypatch):
    monkeypatch.setattr(auth, "_adc", lambda: (_Creds(), None))
    states = {"aiplatform.googleapis.com": (200, {"state": "ENABLED"}),
              "cloudbilling.googleapis.com": (200, {"state": "DISABLED"})}
    monkeypatch.setattr(auth, "_http_get", lambda url, **k: states[url.rsplit("/", 1)[-1]])
    c = TestClient(server.app)
    r = c.get("/api/projects/my-gcp-project/check").json()
    assert [(s["name"], s["enabled"]) for s in r["services"]] == [("aiplatform.googleapis.com", True),
                                                                  ("cloudbilling.googleapis.com", False)]
    assert "console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=my-gcp-project" in r["services"][0]["enable_url"]
    monkeypatch.setattr(auth, "_http_get", lambda url, **k: (403, {"error": {"message": "denied"}}))
    r = c.get("/api/projects/my-gcp-project/check").json()
    assert all(s["enabled"] is None for s in r["services"]) and "permission" in r["message"]
    assert c.get("/api/projects/BAD_ID/check").status_code == 400
