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

import functools
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
    "GRPC_CA_BUNDLE": "grpc_ca_bundle",
    "WORKER_ID": "worker_id",
    "WORKER_JWT": "worker_jwt",
    "ENDPOINT_LOCK_PATH": "endpoint_lock_path",
    "RUNPOD_POD_ID": "runpod_pod_id",
    "WORKER_DISCONNECTED_TIMEOUT_S": "worker_disconnected_timeout_s",
    "WORKER_IMAGE_DIGEST": "worker_image_digest",
    "WORKER_GIT_COMMIT": "worker_git_commit",
    "TENSORHUB_URL": "tensorhub_url",
    "TENSORHUB_TOKEN": "tensorhub_token",
    "TENSORHUB_CACHE_DIR": "tensorhub_cache_dir",
    "TENSORHUB_CAS_DIR": "tensorhub_cas_dir",
    "CIVITAI_API_KEY": "civitai_api_key",
    "COZY_HF_DOWNLOAD_STALL_TIMEOUT_S": "hf_download_stall_timeout_s",
    "COZY_HF_DOWNLOAD_MAX_SECONDS": "hf_download_max_seconds",
    "COZY_HF_MAX_REPO_BYTES": "hf_max_repo_bytes",
    "GEN_WORKER_ATTACHED_LORA_MAX": "attached_lora_max",
    "GEN_WORKER_ATTACHED_LORA_MAX_BYTES": "attached_lora_max_bytes",
    "GEN_WORKER_COMPILE_CACHE": "compile_cache_path",
    "GEN_WORKER_COMPILE_CACHE_URL": "compile_cache_url",
    "GEN_WORKER_COMPILE_ALLOW_COLD": "compile_allow_cold",
}

# Secondary env names for a field, consulted only when the primary name is
# unset or empty (mirrors the historical `os.getenv(A) or os.getenv(B)`).
_ENV_ALIASES: Dict[str, str] = {
    "CIVITAI_TOKEN": "civitai_api_key",
    "HUGGING_FACE_HUB_TOKEN": "hf_token",
}

_FIELD_NAMES = frozenset(_ENV_TO_FIELD.values())

# Field-type metadata for source-value normalization: sources deliver strings;
# non-str fields get stripped, empty values fall back to the struct default
# (an exported-but-empty env var must not crash startup), and bool fields use
# the worker's historical truthy set rather than msgspec's stricter parse.
_FIELD_TYPES: Dict[str, type] = {
    f.name: f.type for f in msgspec.structs.fields(Settings) if isinstance(f.type, type)
}
_TRUTHY = ("1", "true", "yes")

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
    for env_name, field in _ENV_ALIASES.items():
        if out.get(field):
            continue  # primary name wins when non-empty
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
            import yaml

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


def _normalize_values(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare source strings for msgspec.convert per `_FIELD_TYPES`."""
    out: Dict[str, Any] = {}
    for field, val in merged.items():
        ftype = _FIELD_TYPES.get(field, str)
        if ftype is not str and isinstance(val, str):
            val = val.strip()
            if not val:
                continue  # empty => struct default
            if ftype is bool:
                val = val.lower() in _TRUTHY
        out[field] = val
    return out


def load_settings(**init_kwargs: Any) -> Settings:
    """Build a fresh `Settings`. Call once at startup.

    Layers from lowest precedence to highest, merging dict-update style so
    later layers overwrite earlier ones. Then hands the merged dict to
    `msgspec.convert(..., strict=False)` which performs lossless string→typed
    coercion (env vars arrive as strings; numeric / bool fields get parsed)
    while still rejecting non-fitting values.
    """
    merged: Dict[str, Any] = {}
    merged.update(_load_yaml())
    merged.update(_load_secrets_dir())
    merged.update(_load_dotenv())
    merged.update(_load_env())
    merged.update(_normalize_init_kwargs(init_kwargs))
    return msgspec.convert(_normalize_values(merged), Settings, strict=False)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached `Settings` for call sites that aren't handed the
    startup instance (standalone CLI paths, module-level constants). Same
    sources, loaded lazily on first use. Tests clear via the autouse
    `_fresh_settings_cache` fixture (`get_settings.cache_clear()`)."""
    return load_settings()
