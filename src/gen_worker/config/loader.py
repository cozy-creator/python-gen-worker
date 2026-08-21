"""Layered config-source loader for Settings. Precedence (highest -> lowest): init_kwargs, environment, .env, /run/secrets, yaml, struct defaults — mirrors koanf on the Go side. _ENV_TO_FIELD maps each env name to exactly one Settings field."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable

import msgspec

from .settings import Settings


_ENV_TO_FIELD: Dict[str, str] = {
    "HF_TOKEN": "hf_token",
    "HF_HOME": "hf_home",
    "TENSORHUB_PUBLIC_URL": "tensorhub_public_url",
    "ORCHESTRATOR_PUBLIC_ADDR": "orchestrator_public_addr",
    "WORKER_ID": "worker_id",
    "WORKER_JWT": "bootstrap_worker_jwt",
    "WORKER_RELEASE_ID": "worker_release_id",
    "ENDPOINT_LOCK_PATH": "endpoint_lock_path",
    "COZY_WEIGHTS_CAS": "weights_cas_root",
    "COZY_GRAPH_CAS": "graph_cas_root",
    "COZY_ARTIFACTS": "artifacts_root",
    "COZY_ENDPOINT_STATE": "endpoint_state_root",
    "COZY_HOME": "cozy_home",
    "BAKED_PROGRAM_CAS_ROOT": "baked_program_cas_root",
    "RUNPOD_POD_ID": "runpod_pod_id",
    "GEN_WORKER_CONFIG_SNAPSHOT_PATH": "config_snapshot_path",
    "WORKER_CONFIG_GENERATION": "boot_config_generation",
    "WORKER_IMAGE_DIGEST": "worker_image_digest",
    "GEN_WORKER_BOOT_RECORD": "boot_record_path",
    "GEN_WORKER_WATCHDOG_FLATNESS_FLOOR_S": "watchdog_flatness_floor_s",
    "TENSORHUB_URL": "tensorhub_url",
    "TENSORHUB_TOKEN": "tensorhub_token",
    "TENSORHUB_CACHE_DIR": "tensorhub_cache_dir",
    "TENSORHUB_FILL_SOURCE_DIR": "tensorhub_fill_source_dir",
    "CIVITAI_API_KEY": "civitai_api_key",
    "GEN_WORKER_C2PA_CERT_PEM": "c2pa_cert_pem",
    "GEN_WORKER_C2PA_CERT_PATH": "c2pa_cert_path",
    "GEN_WORKER_C2PA_ALG": "c2pa_alg",
    "GEN_WORKER_C2PA_TA_URL": "c2pa_ta_url",
}

_ENV_ALIASES: Dict[str, str] = {
    "CIVITAI_TOKEN": "civitai_api_key",
}

_FIELD_NAMES = frozenset(_ENV_TO_FIELD.values())

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


_OWNED_PREFIXES = ("GEN_WORKER_", "TENSORHUB_", "WORKER_", "COZY_")

REFUSED_KEY_MATERIAL: Dict[str, str] = {
    "GEN_WORKER_C2PA_KEY_PEM": (
        "a C2PA PRIVATE KEY must never be delivered to a pod (th#1307 — tenant "
        "code runs in this process and can read it). Signing is hub-side: the "
        "hub holds the key and signs claims over POST /v1/worker/c2pa/sign"
    ),
    "GEN_WORKER_C2PA_KEY_PATH": (
        "a C2PA PRIVATE KEY must never be delivered to a pod (th#1307 — tenant "
        "code runs in this process and can read it). Signing is hub-side: the "
        "hub holds the key and signs claims over POST /v1/worker/c2pa/sign"
    ),
}

_OWNED_NON_SETTINGS: frozenset[str] = frozenset(REFUSED_KEY_MATERIAL) | frozenset({
    "GEN_WORKER_COMPUTE_CHILD",
    "GEN_WORKER_COMPUTE_UID",
    "GEN_WORKER_CHILD_SOCKET",
    "GEN_WORKER_CHILD_CMD",
    "GEN_WORKER_CHILD_LIVENESS_FD",
    "GEN_WORKER_CHILD_WATCHDOG_PING_S",
    "GEN_WORKER_GROUP_ORDINAL",
    "GEN_WORKER_HOST_SIBLINGS",
    "GEN_WORKER_SESSION_ID",
    "GEN_WORKER_MINT_CHILD",
    "GEN_WORKER_SEAL_LIB_MEMO",
    "GEN_WORKER_SUPERVISED",
    "WORKER_EXECUTION_TOPOLOGY",
    "GEN_WORKER_PROBE",
    "GEN_WORKER_PROBE_PUBLISH_ARMED",
    "GEN_WORKER_LOG_LEVEL",
    "GEN_WORKER_CG_KEYSET",
    "GEN_WORKER_LOCAL_OUTPUT_DIR",
    "GEN_WORKER_NATIVE_KERNELS",
    "GEN_WORKER_NATIVE_KERNELS_LIB",
    "GEN_WORKER_VIDEO_ENCODE_CONCURRENCY",
    "GEN_WORKER_URL_FETCH_ALLOWED_HOSTS",
    "GEN_WORKER_INTERNAL_OBJECT_HOSTS",
    "GEN_WORKER_HOST_MOVE_GUARD",
    "GEN_WORKER_SUPERVISOR",
    "GEN_WORKER_POSTMORTEM_FILE",
    "COZY_HTTP_CONNECT_TIMEOUT_S",
    "COZY_HTTP_READ_TIMEOUT_S",
    "COZY_HTTP_WRITE_TIMEOUT_S",
    "COZY_HTTP_POOL_TIMEOUT_S",
    "COZY_HTTP_TOTAL_TIMEOUT_S",
    "COZY_CONVERT_WORKDIR",
    "COZY_CONVERT_SCRATCH_TTL_S",
    "COZY_CLONE_DOWNLOAD_ATTEMPTS",
    "COZY_CIVITAI_DOWNLOAD_ATTEMPTS",
    "GEN_WORKER_FLEET_LINE_FILE",
})


class RefusedKeyMaterialError(ValueError):
    """A config source delivered secret material this pod refuses to hold."""


class UnknownSettingError(ValueError):
    """A key inside an owned namespace matches no Settings field."""


def _normalize_key(raw: str, *, strict: bool = False) -> str | None:
    key = raw.strip()
    if key in _ENV_TO_FIELD:
        return _ENV_TO_FIELD[key]
    if key in _FIELD_NAMES:
        return key
    if strict and (why := REFUSED_KEY_MATERIAL.get(key.upper())):
        raise RefusedKeyMaterialError(
            f"{key} was delivered to this pod through a config source: {why}"
        )
    if strict and key.upper() in _OWNED_NON_SETTINGS:
        return None
    if strict and any(key.upper().startswith(p) for p in _OWNED_PREFIXES):
        raise UnknownSettingError(
            f"{key!r} is in a gen-worker-owned namespace but matches no "
            f"Settings field. Fix the spelling or add the field; a config key "
            f"we own must never be accepted and ignored."
        )
    return None


def _load_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for env_name, field in _ENV_TO_FIELD.items():
        val = os.environ.get(env_name)
        if val is not None:
            out[field] = val
    for env_name, field in _ENV_ALIASES.items():
        if out.get(field):
            continue
        val = os.environ.get(env_name)
        if val is not None:
            out[field] = val
    return out


def unrecognised_owned_env() -> list[str]:
    """Owned-namespace process-env names this build knows nothing about."""
    known = set(_ENV_TO_FIELD) | set(_ENV_ALIASES) | set(_OWNED_NON_SETTINGS)
    return sorted(
        name for name in os.environ
        if name not in known
        and any(name.startswith(p) for p in _OWNED_PREFIXES)
    )


def _load_dotenv(path: str | None = None) -> Dict[str, str]:
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
            field = _normalize_key(raw_key, strict=True)
            if field is None:
                continue
            val = raw_val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            out[field] = val
    except OSError:
        pass
    return out


def _load_secrets_dir(path: str | None = None) -> Dict[str, str]:
    p = Path(path if path is not None else _SECRETS_DIR)
    if not p.is_dir():
        return {}
    out: Dict[str, str] = {}
    try:
        present = sorted(f.name for f in p.iterdir() if f.is_file())
    except OSError:
        return {}
    for name in present:
        field = _normalize_key(name, strict=True)
        if field is None:
            continue
        try:
            out[field] = (p / name).read_text(encoding="utf-8").rstrip("\n")
        except OSError:
            continue
    return out


def _load_yaml(paths: Iterable[str] | None = None) -> Dict[str, str]:
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
            field = _normalize_key(str(raw_key), strict=True)
            if field is None:
                continue
            if raw_val is None:
                continue
            out[field] = str(raw_val)
        return out
    return {}


def _normalize_init_kwargs(init_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in init_kwargs.items() if k in _FIELD_NAMES}


def _normalize_values(merged: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field, val in merged.items():
        ftype = _FIELD_TYPES.get(field, str)
        if ftype is not str and isinstance(val, str):
            val = val.strip()
            if not val:
                continue
            if ftype is bool:
                val = val.lower() in _TRUTHY
        out[field] = val
    return out


def load_settings(**init_kwargs: Any) -> Settings:
    """Build a fresh `Settings`."""
    merged: Dict[str, Any] = {}
    merged.update(_load_yaml())
    merged.update(_load_secrets_dir())
    merged.update(_load_dotenv())
    merged.update(_load_env())
    merged.update(_normalize_init_kwargs(init_kwargs))
    return msgspec.convert(_normalize_values(merged), Settings, strict=False)
