"""Shared fixtures: tests run without reference benchmarks (catalog.REFERENCE_BENCHMARKS filled from real runs would
otherwise move every estimated figure); tests that need them pass their own."""
import os

import pytest

# No live metadata in tests: the Hugging Face model cache (hfmeta) and the Compute Engine accelerator listing
# (hardware) are neither read nor fetched, so tests see the built-in catalog unless they opt in (monkeypatch).
os.environ["DILEMMA_NO_LIVE_METADATA"] = "1"
# Read gcloud's account / project with `gcloud config get-value` (the fake gcloud), never from ~/.config/gcloud.
os.environ["DILEMMA_GCLOUD_CONFIG_FILES"] = "0"


@pytest.fixture(autouse=True)
def _no_hf_token(monkeypatch, tmp_path_factory):
    """No Hugging Face token leaks in from the environment or the user's ~/.config/aiops-dilemma: every test gets an
    empty config dir and no HF_TOKEN / HUGGING_FACE_HUB_TOKEN (tests that need a token set their own)."""
    from aiops_dilemma import hftoken
    for k in hftoken.ENV_VARS + (hftoken.STUB_ENV,):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DILEMMA_CONFIG_DIR", str(tmp_path_factory.mktemp("config")))
    hftoken._MEM["token"] = None
    hftoken._WHO.clear()


@pytest.fixture(autouse=True)
def _no_reference_benchmarks(monkeypatch):
    from aiops_dilemma import catalog
    monkeypatch.setattr(catalog, "REFERENCE_BENCHMARKS", [])
    monkeypatch.delenv("DILEMMA_REFERENCE_BENCHMARKS", raising=False)


@pytest.fixture(autouse=True)
def _fresh_app_project():
    """The app's chosen project and the memoized default live in auth._APP: start every test without either."""
    from aiops_dilemma import auth
    saved = dict(auth._APP)
    auth._APP.update(project=None, default=None, default_at=0.0)
    auth._GCLOUD_MEM.clear()
    auth._ADC_FILE_MEM.clear()
    yield
    auth._APP.clear()
    auth._APP.update(saved)


@pytest.fixture(autouse=True, scope="session")
def _catalog_cache_in_tmp(tmp_path_factory):
    """Catalogs the server builds during tests (without live metadata) go to a temp dir, never over the user's
    ~/.cache/aiops-dilemma/catalog_*.json, which a running `dilemma ui` would then serve."""
    from aiops_dilemma import server
    saved = server.CACHE_DIR
    server.CACHE_DIR = tmp_path_factory.mktemp("catalog-cache")
    yield
    server.CACHE_DIR = saved


@pytest.fixture(autouse=True)
def _local_mode(monkeypatch):
    """Tests run as a local app (no IAP gate, no path confinement) unless they opt into cloud mode."""
    for k in ("K_SERVICE", "DILEMMA_CLOUD", "DILEMMA_REQUIRE_IAP", "DILEMMA_CONFINE_PATHS", "DILEMMA_IAP_AUDIENCE",
              "DILEMMA_JOB_USER"):
        monkeypatch.delenv(k, raising=False)
