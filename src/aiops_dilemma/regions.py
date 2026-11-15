"""Google Cloud region names: the one rule for every place a region reaches a URL host or an API endpoint.

Regions are built into host names (`https://{region}-aiplatform.googleapis.com/...`, the Agent Platform SDK's
`{location}-aiplatform.googleapis.com`) and those requests carry credentials, so an unchecked value such as
`evil.example#` would send the token to another host. Every region is checked against `us-central1`-style names
(letters, a dash, letters, digits; e.g. europe-west4, northamerica-northeast1, me-central2) before use.
"""
from __future__ import annotations

import re

REGION_RE = re.compile(r"^[a-z]{2,20}-[a-z]{2,20}[0-9]{1,2}$")
DEFAULT = "us-central1"


class InvalidRegion(ValueError):
    pass


def is_valid(region, allow_global: bool = False) -> bool:
    return isinstance(region, str) and (bool(REGION_RE.fullmatch(region)) or (allow_global and region == "global"))


def require(region, allow_global: bool = False) -> str:
    """The region itself, or InvalidRegion: for code that puts it into a host name or endpoint."""
    if not is_valid(region, allow_global):
        raise InvalidRegion(f"not a Google Cloud region: {str(region)[:40]!r}")
    return region


def or_default(region, default: str = DEFAULT) -> str:
    """For configs read from disk: an invalid region falls back to the default instead of failing."""
    return region if is_valid(region) else default
