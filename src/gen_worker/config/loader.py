"""Layered config-source loader for `Settings`.

Source precedence (highest -> lowest) mirrors koanf on the Go side:
    1. init_kwargs   — programmatic overrides passed to `load_settings(...)`
    2. environment   — orchestrator-injected pod env wins over everything below
    3. .env file     — ./.env, if present (local-dev convenience)
    4. /run/secrets  — k8s/docker secret-style per-key files, if dir exists
    5. yaml          — /etc/gen-worker/config.yaml or ./gen-worker.yaml, if file exists
    6. struct defaults

The env-name -> field-name mapping is a small static dict (`_ENV_TO_FIELD`)
at the top of this file. Every Settings field corresponds to exactly one
env name and one .env / yaml / secret key — see the table below.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable

import msgspec

from .settings import Settings


# Env name -> Settings field name. The env name is what gen-orchestrator
# injects into the worker pod and what /run/secrets file names should match.
# yaml + .env entries may use EITHER the env name or the field name; both
# resolve to the same field via the normalize step below.
_ENV_TO_FIELD: Dict[str, str] = {
    "HF_TOKEN": "hf_token",
    "HF_HOME": "hf_home",
    "TENSORHUB_PUBLIC_URL": "tensorhub_public_url",
    "ORCHESTRATOR_PUBLIC_ADDR": "orchestrator_public_addr",
    "WORKER_ID": "worker_id",
    "WORKER_MODE": "worker_mode",
    "WORKER_JWT": "worker_jwt",
    "TRAINER_JOB_SPEC_PATH": "trainer_job_spec_path",
    "RUNPOD_POD_ID": "runpod_pod_id",
    "WORKER_DISCONNECTED_TIMEOUT_S": "worker_disconnected_timeout_s",
}

_FIELD_NAMES = frozenset(_ENV_TO_FIELD.values())

_YAML_CANDIDATE_PATHS = (
    "/etc/gen-worker/config.yaml",
    "./gen-worker.yaml",
)
_SECRETS_DIR = "/run/secrets"
_DOTENV_PATH = "./.env"


def _normalize_key(raw: str) -> str | None:
    """Map a raw source key (env name OR field name) to a Settings field name.

    Returns None when the key doesn't correspond to any known field — sources
    can contain arbitrary keys (think a system .env with hundreds of entries);
    we silently skip the ones that don't match a Settings field.
    """
    key = raw.strip()
    if key in _ENV_TO_FIELD:
        return _ENV_TO_FIELD[key]
    if key in _FIELD_NAMES:
        return key
    return None


def _load_env() -> Dict[str, str]:
    """Read every Settings-relevant env var that's actually set."""
    out: Dict[str, str] = {}
    for env_name, field in _ENV_TO_FIELD.items():
        val = os.environ.get(env_name)
        if val is not None:
            out[field] = val
    return out


def _load_dotenv(path: str | None = None) -> Dict[str, str]:
    """Tiny `.env` parser — `KEY=VALUE` lines, `#` comments, blank lines.

    Avoids the python-dotenv dep. We don't support shell-style quoting or
    interpolation; values are taken literally. Lines that don't parse are
    silently skipped.
    """
    p = Path(path if path is not None else _DOTENV_PATH)
    if not p.is_file():
        return {}
    out: Dict[str, str] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            raw_key, _, raw_val = line.partition("=")
            field = _normalize_key(raw_key)
            if field is None:
                continue
            val = raw_val.strip()
            # Strip surrounding quotes if present — `.env` files commonly
            # wrap values in quotes for readability; we treat them as syntax.
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            out[field] = val
    except OSError:
        pass
    return out


def _load_secrets_dir(path: str | None = None) -> Dict[str, str]:
    """Read each Settings field from `<path>/<env-name>` if the file exists.

    k8s/docker-compose secrets mount one file per secret, named after the
    secret key. Skips silently when the dir doesn't exist (the common case
    on dev hosts).
    """
    p = Path(path if path is not None else _SECRETS_DIR)
    if not p.is_dir():
        return {}
    out: Dict[str, str] = {}
    for env_name, field in _ENV_TO_FIELD.items():
        f = p / env_name
        if not f.is_file():
            continue
        try:
            val = f.read_text(encoding="utf-8").rstrip("\n")
        except OSError:
            continue
        out[field] = val
    return out


def _load_yaml(paths: Iterable[str] | None = None) -> Dict[str, str]:
    """Read settings from the first existing yaml file in `paths`."""
    for raw in paths if paths is not None else _YAML_CANDIDATE_PATHS:
        p = Path(raw)
        if not p.is_file():
            continue
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: Dict[str, str] = {}
        for raw_key, raw_val in data.items():
            field = _normalize_key(str(raw_key))
            if field is None:
                continue
            if raw_val is None:
                continue
            out[field] = str(raw_val)
        return out
    return {}


def _normalize_init_kwargs(init_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Init kwargs are passed by callers using field names directly; just
    filter out anything that isn't a known field so a typo at the call site
    raises a clearer error from msgspec.convert later."""
    return {k: v for k, v in init_kwargs.items() if k in _FIELD_NAMES}


def load_settings(**init_kwargs: Any) -> Settings:
    """Build a fresh `Settings`. Call once at startup.

    Layers from lowest precedence to highest, merging dict-update style so
    later layers overwrite earlier ones. Then hands the merged dict to
    `msgspec.convert(..., strict=False)` which performs lossless string→typed
    coercion (env vars arrive as strings; numeric / bool fields get parsed)
    while still rejecting non-fitting values (e.g. `WORKER_MODE=garbage` is
    not a member of the Literal so it still raises).
    """
    merged: Dict[str, Any] = {}
    merged.update(_load_yaml())
    merged.update(_load_secrets_dir())
    merged.update(_load_dotenv())
    merged.update(_load_env())
    merged.update(_normalize_init_kwargs(init_kwargs))
    return msgspec.convert(merged, Settings, strict=False)
