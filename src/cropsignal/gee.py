"""
Earth Engine authentication.

Resolution order:
  1. GEE_SERVICE_ACCOUNT + GEE_PRIVATE_KEY environment variables
  2. a .env file in this repo root
  3. a .env file in a sibling repo (so an existing local setup is reused
     rather than asking for the same credentials twice)
  4. cached user credentials from `earthengine authenticate`
"""
from __future__ import annotations

import os
from pathlib import Path

import ee

REPO_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_ENVS = [
    REPO_ROOT / ".env",
    REPO_ROOT.parent / "crop-stress-prediction" / ".env",
    REPO_ROOT.parent / "zaminai" / ".env",
]

_initialized = False


def _load_dotenv(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def init(verbose: bool = True) -> None:
    """Initialise Earth Engine once per process."""
    global _initialized
    if _initialized:
        return

    env = dict(os.environ)
    for candidate in FALLBACK_ENVS:
        loaded = _load_dotenv(candidate)
        for k, v in loaded.items():
            env.setdefault(k, v)

    sa, key = env.get("GEE_SERVICE_ACCOUNT"), env.get("GEE_PRIVATE_KEY")
    if sa and key:
        creds = ee.ServiceAccountCredentials(sa, key_data=key.replace("\\n", "\n"))
        ee.Initialize(creds)
        if verbose:
            print(f"Earth Engine ready (service account {sa})")
    else:
        ee.Initialize()
        if verbose:
            print("Earth Engine ready (cached user credentials)")
    _initialized = True
