"""Module-level helpers for RequestContext: parsing, SSRF checks, hashing, etc.

These are pure utilities with no dependency on RequestContext or
_RequestOutputStream. Pulled out of the monolithic request_context.py so the
class files only hold class bodies.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import re
import socket
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from ..api.errors import OutputTooLargeError
from ..models.refs import parse_model_ref

logger = logging.getLogger(__name__)


# Canonical hint-key names + back-compat aliases. Callers read via
# `_resolve_hint_first_string` so the alias fallbacks can be removed in one
# place once gen-orchestrator stops emitting the legacy names.
# Migration recipe:
#   1. gen-orchestrator emits canonical key alongside each alias for one release
#      (so a mixed fleet of workers/orchestrators keeps working during rollout).
#   2. Flip this list to contain only the canonical key (first element).
#   3. Stop emitting aliases in gen-orchestrator.
_HINT_KEYS_DESTINATION_REPO: tuple[str, ...] = ("destination_repo", "repo", "output_repo")
_HINT_KEYS_JOB_ID:           tuple[str, ...] = ("job_id", "conversion_job_id", "training_job_id")
_HINT_KEYS_EXECUTION_KIND:   tuple[str, ...] = ("kind", "execution_kind")


def _resolve_hint_first_string(
    *sources: Dict[str, Any],
    keys: tuple[str, ...],
    fallback: Any = "",
) -> str:
    """Return the first non-empty string value found by scanning `keys` across
    each source dict in order.

    Designed to replace multi-option fallback ladders like
    `hints.get("destination_repo") or hints.get("repo") or spec.get("destination_repo") …`
    with a single call. Sources are scanned in the order given; each source is
    checked for every key in `keys` before moving to the next source. Returns
    `fallback` (trimmed to string) if nothing matches.

    Callers that only need a single dict can pass it as the only positional:
    `_resolve_hint_first_string(hints, keys=_HINT_KEYS_JOB_ID)`.
    """
    for src in sources:
        if not src:
            continue
        for k in keys:
            v = src.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return str(fallback or "").strip()


_PUBLIC_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_STALE_MIRROR_CLAIM_ERROR_CODES = {"source_version_not_found", "source_variants_not_found"}
_MAX_OUTPUT_FILE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB hard cap per file.
_FILE_API_HTTP_TIMEOUT_S = 60
_FILE_API_STREAM_CHUNK_TIMEOUT_S = 120
_FILE_API_STREAM_FINALIZE_TIMEOUT_S = 600
_FILE_API_STREAM_REPLAY_TIMEOUT_S = 600
_FILE_API_STREAM_ABORT_TIMEOUT_S = 15


def _http_request(
    method: str,
    url: str,
    token: str,
    owner: Optional[str] = None,
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
) -> urllib.request.Request:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    owner = (owner or "").strip()
    if owner:
        req.add_header("X-Cozy-Owner", owner)
    if content_type:
        req.add_header("Content-Type", content_type)
    return req


def _encode_ref_for_url(ref: str) -> str:
    ref = ref.strip().lstrip("/")
    parts = [urllib.parse.quote(p, safe="") for p in ref.split("/") if p]
    return "/".join(parts)


def _infer_mime_type(ref: str, head: bytes) -> str:
    # Prefer magic bytes when available.
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"

    # Fall back to extension.
    import mimetypes

    guessed, _ = mimetypes.guess_type(ref)
    return guessed or "application/octet-stream"


def _normalize_output_ref(ref: str) -> str:
    out = str(ref or "").strip()
    if not out:
        raise ValueError("invalid ref")
    lower = out.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        raise ValueError("output ref must be a logical file ref, not a URL")
    return out.lstrip("/")


def _infer_tensors_format(ref_or_path: str) -> str:
    leaf = str(ref_or_path or "").strip().lower()
    if leaf.endswith(".safetensors"):
        return "safetensors"
    if leaf.endswith(".bin"):
        return "bin"
    if leaf.endswith(".pt"):
        return "pt"
    if leaf.endswith(".pth"):
        return "pth"
    if leaf.endswith(".ckpt"):
        return "ckpt"
    return "unknown"


def _require_worker_capability_token() -> str:
    # The per-request worker_capability_token is plumbed from
    # JobExecutionRequest.worker_capability_token via RequestContext.
    # Callers should prefer RequestContext._get_worker_capability_token(),
    # which uses the per-request token directly and only falls back here
    # when none is available. Without a per-request token, file
    # operations cannot proceed.
    raise RuntimeError("worker_capability_token is required for file operations")


def _parse_owner_repo(value: str) -> tuple[str, str]:
    raw = str(value or "").strip().strip("/")
    if "/" not in raw:
        raise ValueError("destination_repo must be in '<owner>/<repo>' format")
    owner, repo = raw.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        raise ValueError("destination_repo must be in '<owner>/<repo>' format")
    return owner, repo


def _parse_owner_repo_with_optional_tag(value: str) -> tuple[str, str, str]:
    raw = str(value or "").strip().strip("/")
    tag = ""
    if ":" in raw:
        raw, tag = raw.rsplit(":", 1)
        tag = str(tag or "").strip().lower()
    owner, repo = _parse_owner_repo(raw)
    return owner, repo, tag


def _normalize_destination_repo_tags(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in list(values or []):
        tag = str(item or "").strip().lower()
        if not tag:
            continue
        if not _PUBLIC_TAG_RE.match(tag):
            raise ValueError("destination_repo_tags contains an invalid tag")
        if tag == "latest":
            raise ValueError("destination_repo_tags must not include latest")
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    out.sort()
    return out


def _decode_unverified_jwt_claims(token: str) -> Dict[str, Any]:
    raw = str(token or "").strip()
    if raw.count(".") < 2:
        return {}
    try:
        parts = raw.split(".")
        payload_b64 = parts[1]
        pad = "=" * ((4 - (len(payload_b64) % 4)) % 4)
        payload = base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii"))
        parsed = json.loads(payload.decode("utf-8"))
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _normalize_repo_name(value: str) -> str:
    return str(value or "").strip().strip("/").lower()


def _enforce_output_file_size_limit(size_bytes: int) -> None:
    size = int(size_bytes)
    if size < 0:
        raise ValueError("size_bytes must be non-negative")
    if size > _MAX_OUTPUT_FILE_BYTES:
        raise OutputTooLargeError(size_bytes=size, max_bytes=_MAX_OUTPUT_FILE_BYTES)


def _assert_token_repo_scope_matches_destination(
    token: str,
    owner: str,
    repo: str,
    *,
    required_permissions: Optional[List[str]] = None,
) -> None:
    claims = _decode_unverified_jwt_claims(token)
    if not claims:
        raise ValueError("worker_capability_token must be a structured JWT")

    destination_owner = _normalize_repo_name(owner)
    destination_repo = _normalize_repo_name(repo)
    cap_kind = _normalize_repo_name(str(claims.get("cap_kind") or ""))
    if cap_kind != "worker_capability":
        raise ValueError("worker_capability_token must have cap_kind=worker_capability")
    if required_permissions is None:
        needed = ["repo-version:create"]
    else:
        needed = [str(p or "").strip() for p in list(required_permissions) if str(p or "").strip()]
    repos_read = [str(v or "").strip() for v in list(claims.get("tensor_repos_read") or [])]
    repos_update_legacy = [str(v or "").strip() for v in list(claims.get("tensor_repos_update") or [])]
    repos_version_create = [str(v or "").strip() for v in list(claims.get("tensor_repos_version_create") or [])]
    repos_variant_create = [str(v or "").strip() for v in list(claims.get("tensor_repos_variant_create") or [])]
    if not repos_version_create:
        repos_version_create = list(repos_update_legacy)
    if not repos_variant_create:
        repos_variant_create = list(repos_update_legacy)
    create_claim = claims.get("tensor_repo_create")
    create_policy = create_claim if isinstance(create_claim, dict) else {}
    create_owner = _normalize_repo_name(str(create_policy.get("owner") or ""))
    create_allowed_names = [_normalize_repo_name(str(v or "")) for v in list(create_policy.get("allowed_names") or [])]
    create_allow_any_name = bool(create_policy.get("allow_any_name"))

    def _repo_match(values: List[str]) -> bool:
        for raw in values:
            try:
                scoped_owner, scoped_repo = _parse_owner_repo(_normalize_repo_name(raw))
            except ValueError:
                continue
            if _normalize_repo_name(scoped_owner) == destination_owner and _normalize_repo_name(scoped_repo) == destination_repo:
                return True
        return False

    for permission in needed:
        if permission == "tensor-repo:read":
            if _repo_match(repos_read) or _repo_match(repos_version_create) or _repo_match(repos_variant_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token read scope")
        if permission == "repo-version:create":
            if _repo_match(repos_version_create) or _repo_match(repos_variant_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token repo-version:create scope")
        if permission == "repo-variant:create":
            if _repo_match(repos_variant_create) or _repo_match(repos_version_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token repo-variant:create scope")
        if permission == "tensor-repo:update":
            # Legacy alias.
            if _repo_match(repos_version_create) or _repo_match(repos_variant_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token update scope")
        if permission == "tensor-repo:create":
            if create_owner != destination_owner:
                raise ValueError("destination_repo owner does not match worker_capability_token create scope")
            if create_allow_any_name:
                continue
            if destination_repo in create_allowed_names:
                continue
            raise ValueError("destination_repo is not in worker_capability_token create allow-list")
        raise ValueError(f"unsupported required permission '{permission}'")


def _is_private_ip_str(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _url_is_blocked(url_str: str) -> bool:
    try:
        u = urllib.parse.urlparse(url_str)
    except Exception:
        return True
    if u.scheme not in ("http", "https"):
        return True
    host = (u.hostname or "").strip()
    if not host:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = str(sockaddr[0])
        if _is_private_ip_str(ip_str):
            return True
    return False


def _canonicalize_model_ref_string(raw: str) -> str:
    """
    Best-effort normalization of Cozy/HF model ref strings for allowlisting and caching identity.

    If the string doesn't parse as a phase-1 model ref, return it unchanged.

    Issue #17: consult the per-worker provider index (set as a contextvar
    by the worker before invoking the model manager) so HF / civitai refs
    are decoded with the correct provider. Refs not in the index default
    to ``"tensorhub"`` — matching the wire-format contract.
    """
    s = (raw or "").strip()
    if not s:
        return s
    try:
        from ..models.ref_downloader import lookup_provider_for_ref  # lazy: pulls requests

        provider = lookup_provider_for_ref(s)
        parsed = parse_model_ref(s, provider=provider)
        if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
            return parsed.tensorhub.canonical()
        if parsed.provider == "hf" and parsed.hf is not None:
            return parsed.hf.canonical()
        return s
    except Exception:
        return s


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
