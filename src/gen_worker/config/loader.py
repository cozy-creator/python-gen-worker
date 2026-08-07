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
    # The ENV name is hub-injected and fixed; the FIELD is renamed
    # (pgw#848) so no call site can read it as the live credential.
    "WORKER_JWT": "bootstrap_worker_jwt",
    "WORKER_RELEASE_ID": "worker_release_id",
    "WORKER_MODE": "worker_mode",
    "ENDPOINT_LOCK_PATH": "endpoint_lock_path",
    "RUNPOD_POD_ID": "runpod_pod_id",
    "GEN_WORKER_CONFIG_SNAPSHOT_PATH": "config_snapshot_path",
    "WORKER_CONFIG_GENERATION": "boot_config_generation",
    "WORKER_IMAGE_DIGEST": "worker_image_digest",
    "GEN_WORKER_BOOT_RECORD": "boot_record_path",
    "TENSORHUB_URL": "tensorhub_url",
    "TENSORHUB_TOKEN": "tensorhub_token",
    "TENSORHUB_CACHE_DIR": "tensorhub_cache_dir",
    "TENSORHUB_CAS_DIR": "tensorhub_cas_dir",
    "TENSORHUB_FILL_SOURCE_DIR": "tensorhub_fill_source_dir",
    "CIVITAI_API_KEY": "civitai_api_key",
    "GEN_WORKER_C2PA_CERT_PEM": "c2pa_cert_pem",
    "GEN_WORKER_C2PA_CERT_PATH": "c2pa_cert_path",
    # th#1307: GEN_WORKER_C2PA_KEY_PEM / _KEY_PATH are deliberately NOT
    # mapped — a private key must never be readable by tenant code in this
    # process. content_credentials.configure() refuses to start if either is
    # present in the environment.
    "GEN_WORKER_C2PA_ALG": "c2pa_alg",
    "GEN_WORKER_C2PA_TA_URL": "c2pa_ta_url",
}

# Secondary env names for a field, consulted only when the primary name is
# unset or empty (mirrors the historical `os.getenv(A) or os.getenv(B)`).
_ENV_ALIASES: Dict[str, str] = {
    "CIVITAI_TOKEN": "civitai_api_key",
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


#: Namespaces this program OWNS. A key inside one of them is addressed to us,
#: so an unrecognised spelling is a misconfiguration and not somebody else's
#: variable — see :func:`_normalize_key`.
_OWNED_PREFIXES = ("GEN_WORKER_", "TENSORHUB_", "WORKER_", "COZY_")

#: Owned-namespace names that are deliberately NOT Settings fields. Each one
#: is read by a named mechanism other than this loader, and listing it here is
#: what keeps :func:`_normalize_key`'s refusal from firing on a legitimate
#: value. Anything not here and not in `_ENV_TO_FIELD` is a typo.
_OWNED_NON_SETTINGS: frozenset[str] = frozenset({
    # th#1307: refused at boot by content_credentials.configure(); deliberately
    # never bound to a field, because a private key must not be readable by
    # tenant code in this process.
    "GEN_WORKER_C2PA_KEY_PEM",
    "GEN_WORKER_C2PA_KEY_PATH",
    # pgw#929 CHILD IPC HANDOFF: parent-minted, per-child, no config origin.
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
    "GEN_WORKER_AOT_ENTRY_CHILD",
    "GEN_WORKER_SEAL_LIB_MEMO",
    "GEN_WORKER_SUPERVISED",
    "WORKER_EXECUTION_TOPOLOGY",
    # pgw#980: the live-edit probe marking and its separate publish arming.
    # Read by `procsplit.actions` — the PARENT's security boundary, which must
    # be readable with no Settings in hand, and which tenant-adjacent code in
    # the compute child must not be able to reach through the config surface.
    "GEN_WORKER_PROBE",
    "GEN_WORKER_PROBE_PUBLISH_ARMED",
    # pgw#929 library/standalone-tool knobs; see scripts/config_reads_allowlist.txt.
    "GEN_WORKER_LOG_LEVEL",
    "GEN_WORKER_LOCAL_CELLS_DIR",
    "GEN_WORKER_LOCAL_OUTPUT_DIR",
    "GEN_WORKER_MINT_RESUME_DIR",
    "GEN_WORKER_MINT_RESUME_MAX_BYTES",
    "GEN_WORKER_AOT_HOST_COMPILE_JOBS",
    "GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF",
    "GEN_WORKER_NATIVE_KERNELS",
    "GEN_WORKER_NATIVE_KERNELS_LIB",
    "GEN_WORKER_SVDQ_ENGINE",
    "GEN_WORKER_VIDEO_ENCODER",
    "GEN_WORKER_VIDEO_ENCODE_CONCURRENCY",
    "GEN_WORKER_URL_FETCH_ALLOWED_HOSTS",
    "GEN_WORKER_INTERNAL_OBJECT_HOSTS",
    "GEN_WORKER_MINT_IN_PROCESS",
    "GEN_WORKER_HOST_MOVE_GUARD",
    "GEN_WORKER_FORBID_CPU_OFFLOAD",
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
    "COZY_CELL_EPOCH",
})


class UnknownSettingError(ValueError):
    """A key inside an owned namespace matches no Settings field.

    pgw#931 deliverable 8. `_normalize_key` used to return None for anything
    unrecognised and every source layer then silently skipped it, so a typo'd
    ``TENSORHUB_CHACE_DIR`` in `.env` or `/run/secrets` was accepted and inert
    — the operator's intent evaporated with no diagnostic anywhere. Same hole
    the hub side closed at `config/config.go:1281` (th#1500 deliverable 4).

    Deliberately scoped to `_OWNED_PREFIXES`: the process environment legitimately
    carries hundreds of variables belonging to CUDA, Python, the OS and the pod
    runtime, and refusing those would make the worker unbootable. A key in a
    namespace we own is addressed to us, so we owe it an answer.
    """


def _normalize_key(raw: str, *, strict: bool = False) -> str | None:
    """Map a raw source key (env name OR field name) to a Settings field name.

    Returns None for a key that is not ours. When `strict`, an unrecognised key
    inside an owned namespace raises `UnknownSettingError` instead of being
    silently dropped.
    """
    key = raw.strip()
    if key in _ENV_TO_FIELD:
        return _ENV_TO_FIELD[key]
    if key in _FIELD_NAMES:
        return key
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


def unrecognised_owned_env() -> list[str]:
    """Owned-namespace process-env names this build knows nothing about.

    REPORTED, never refused — and the asymmetry with the file sources is the
    whole point. `.env` / yaml / `/run/secrets` are hand-authored by an
    operator addressing this program, so an unrecognised key there is a typo
    and `load_settings` raises. The process environment is assembled by
    another program: measured 2026-08-03, Tensorhub injects owned-namespace
    names this worker has no reader for (`GEN_WORKER_OOM_PROBE`,
    `GEN_WORKER_PROCESS_SPLIT`), so refusing here would turn a hub-side
    addition into a fleet of dead pods.

    The residue still matters — a misspelled `GEN_WORKER_PREFR_AOT` in a
    release declaration is silently inert today — so it is named at boot
    instead of vanishing.
    """
    known = set(_ENV_TO_FIELD) | set(_ENV_ALIASES) | set(_OWNED_NON_SETTINGS)
    return sorted(
        name for name in os.environ
        if name not in known
        and any(name.startswith(p) for p in _OWNED_PREFIXES)
    )


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
            field = _normalize_key(raw_key, strict=True)
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
    try:
        present = sorted(f.name for f in p.iterdir() if f.is_file())
    except OSError:
        return {}
    for name in present:
        # Raises UnknownSettingError on an owned-namespace filename we do not
        # bind: a mounted secret nobody reads is a secret that did not arrive.
        field = _normalize_key(name, strict=True)
        if field is None:
            continue
        try:
            out[field] = (p / name).read_text(encoding="utf-8").rstrip("\n")
        except OSError:
            continue
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
            field = _normalize_key(str(raw_key), strict=True)
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
