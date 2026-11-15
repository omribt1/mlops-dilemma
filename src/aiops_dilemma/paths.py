"""Where sessions and caches live; both can be moved with environment variables (e.g. onto a Cloud Run volume).

- ``DILEMMA_RUNS_DIR``: sessions (``<runs>/<session>/``), default ``<repo>/runs``.
- ``DILEMMA_CACHE_DIR``: catalog, price and model caches, default ``~/.cache/aiops-dilemma``.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def runs_dir() -> Path:
    return Path(os.environ.get("DILEMMA_RUNS_DIR", "").strip() or ROOT_DIR / "runs")


def cache_dir() -> Path:
    return Path(os.environ.get("DILEMMA_CACHE_DIR", "").strip() or Path.home() / ".cache" / "aiops-dilemma")


# ---- Cloud deployment (Cloud Run behind IAP) ----------------------------------------------------------------------
def _flag(name: str) -> bool | None:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def cloud_mode() -> bool:
    """Running as a shared cloud deployment: on Cloud Run (`K_SERVICE` is set) or `DILEMMA_CLOUD=1`.
    `DILEMMA_CLOUD=0` turns it off (e.g. a local container test that sets K_SERVICE)."""
    f = _flag("DILEMMA_CLOUD")
    return f if f is not None else bool(os.environ.get("K_SERVICE"))


def confine_paths() -> bool:
    """File paths taken from a session's config may only point inside the sessions folder and the repo's examples
    (cloud mode, or `DILEMMA_CONFINE_PATHS=1`). Locally, configs may name any file you can read."""
    f = _flag("DILEMMA_CONFINE_PATHS")
    return f if f is not None else cloud_mode()


def allowed_roots() -> list[Path]:
    return [runs_dir(), ROOT_DIR / "blog" / "example", ROOT_DIR / "examples"]


class PathNotAllowed(PermissionError):
    pass


def check_readable(path: str | Path) -> Path:
    """`path` resolved (symlinks and `..` followed); raises PathNotAllowed when paths are confined and it falls
    outside allowed_roots(), so a config can't read e.g. /proc/self/environ or credentials on a shared deployment."""
    p = Path(path).expanduser().resolve()
    if confine_paths() and not any(p.is_relative_to(r.resolve()) for r in allowed_roots()):
        raise PathNotAllowed(f"{path}: only files inside the session folder can be used on this deployment "
                             "(upload the file instead)")
    return p
