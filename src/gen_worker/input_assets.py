"""Fail-closed materialization of RunJob input ``Asset`` values (v4).

Canonical MessagePack keeps the caller's opaque stored refs. Tensorhub ships an
ordered, credential-free ``RunJob.input_assets`` manifest; the assigned worker
validates the payload's private refs against it, resolves fresh transport URLs
once per attempt via the worker capability, downloads and verifies the exact
attested bytes, and sets only worker-local ``local_path`` on every occurrence.
Caller HTTP(S) refs stay public transports and never enter the manifest.
Capabilities, URLs, opaque refs, and resolver bodies never enter errors/logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import blake3 as blake3_mod
import msgspec

from .api.errors import CanceledError, RetryableError, ValidationError
from .api.types import Asset, AudioAsset, ImageAsset, VideoAsset
from .request_context._helpers import _infer_mime_type, _url_is_blocked

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 50 << 20  # matches tensorhub's default media cap
_DOWNLOAD_TIMEOUT_S = 120
_RESOLVE_TIMEOUT_S = 30
_MAX_RESOLVE_BODY = 8 << 20
_MAX_WALK_DEPTH = 32
_CHUNK = 1 << 20
_INPUT_DIR_PREFIX = "gen-worker-inputs-"
_RESOLVE_PATH = "/api/v1/worker/input-assets/resolve"
_MANIFEST_KINDS = ("media", "image", "video", "audio")
_HEX = set("0123456789abcdef")

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
}

# (url, headers, body) -> (http_status, response_body). Injectable for tests.
ResolveTransport = Callable[[str, Mapping[str, str], bytes], tuple[int, bytes]]


class InputManifestEntry(msgspec.Struct, frozen=True):
    """One dispatched ``RunJob.input_assets`` entry (immutable identity)."""

    asset_id: str
    source_ref: str
    blake3: str
    size_bytes: int
    kind: str
    mime_type: str


class _ResolvedInputAsset(msgspec.Struct, forbid_unknown_fields=True):
    asset_id: str
    source_ref: str
    blake3: str
    size_bytes: int
    kind: str
    mime_type: str
    url: str
    url_expires_at: str


class _ResolveResponse(msgspec.Struct, forbid_unknown_fields=True):
    assets: list[_ResolvedInputAsset]


def manifest_from_run_job(entries: Any) -> tuple[InputManifestEntry, ...]:
    """Convert ``RunJob.input_assets`` proto rows to typed manifest entries."""
    return tuple(
        InputManifestEntry(
            asset_id=str(e.asset_id),
            source_ref=str(e.source_ref),
            blake3=str(e.blake3),
            size_bytes=int(e.size_bytes),
            kind=str(e.kind),
            mime_type=str(e.mime_type),
        )
        for e in entries or ()
    )


@dataclass(frozen=True)
class _AssetOccurrence:
    path: str
    asset: Asset


@dataclass
class _DownloadUnit:
    occurrences: list[_AssetOccurrence]
    entry: InputManifestEntry | None = None  # None = public caller transport
    url: str = ""  # transport ref (public) or resolver-minted URL (private)


@dataclass
class _ScopeState:
    path: Path
    assets: tuple[Asset, ...]


_scope_lock = threading.Lock()
_scope_states: dict[str, _ScopeState] = {}


def _scope_key(request_id: str, attempt: int) -> str:
    raw = f"{str(request_id or '').strip()}\0{int(attempt)}".encode()
    return hashlib.sha256(raw).hexdigest()


def inputs_dir_for_request(request_id: str, attempt: int = 0) -> Path:
    """Return the active worker-owned directory, or a non-existent sentinel."""
    key = _scope_key(request_id, attempt)
    with _scope_lock:
        state = _scope_states.get(key)
    if state is not None:
        return state.path
    return Path(tempfile.gettempdir()) / f".{_INPUT_DIR_PREFIX}absent-{key}"


def _iter_asset_occurrences(
    obj: Any, path: str = "$", depth: int = 0
) -> Iterator[_AssetOccurrence]:
    """Deterministic traversal: msgspec struct declaration order, list/tuple
    index, lexicographically sorted string map keys. Unordered containers and
    non-string map keys are defensively rejected (build discovery already
    refuses such schemas)."""
    if depth > _MAX_WALK_DEPTH:
        raise ValidationError("input_asset_payload_too_deep: nested Asset walk exceeded its limit")
    if isinstance(obj, Asset):
        yield _AssetOccurrence(path=path, asset=obj)
        return
    if isinstance(obj, msgspec.Struct):
        for name in obj.__struct_fields__:
            yield from _iter_asset_occurrences(getattr(obj, name), f"{path}.{name}", depth + 1)
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            yield from _iter_asset_occurrences(item, f"{path}[{index}]", depth + 1)
    elif isinstance(obj, (set, frozenset)):
        for item in obj:
            for _ in _iter_asset_occurrences(item, f"{path}[*]", depth + 1):
                raise ValidationError(
                    f"input_asset_unordered_container: {path} carries an Asset "
                    "inside an unordered set/frozenset"
                )
    elif isinstance(obj, dict):
        # Map keys may contain caller data: error paths use sorted positions,
        # never the keys themselves.
        for index, key in enumerate(_sorted_string_keys(obj, path)):
            yield from _iter_asset_occurrences(obj[key], f"{path}[{index}]", depth + 1)


def _sorted_string_keys(obj: dict, path: str) -> list[str]:
    for key in obj:
        if not isinstance(key, str):
            raise ValidationError(
                f"input_asset_map_key_invalid: {path} mapping key is not a string"
            )
    return sorted(obj)


def _iter_assets(obj: Any, depth: int = 0) -> Iterator[Asset]:
    """Compatibility iterator: ordered, duplicate-preserving Asset objects."""
    for occurrence in _iter_asset_occurrences(obj, depth=depth):
        yield occurrence.asset


def _declared_kind(asset: Asset) -> str:
    if isinstance(asset, ImageAsset):
        return "image"
    if isinstance(asset, VideoAsset):
        return "video"
    if isinstance(asset, AudioAsset):
        return "audio"
    return "media"


def _classify_ref(ref: str, path: str) -> bool:
    """True = private opaque stored ref; False = public HTTP(S) transport.

    Mirrors tensorhub's submit-time classifier so ``input_payload`` and the
    manifest can never disagree: padded/empty refs and every non-HTTP scheme
    are invalid.
    """
    value = str(ref or "")
    if not value or value != value.strip():
        raise ValidationError(f"invalid_input_asset_ref: {path} ref is empty or padded")
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        raise ValidationError(f"invalid_input_asset_ref: {path} ref is not parseable") from None
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https"):
        if not parsed.netloc or parsed.hostname is None:
            raise ValidationError(f"invalid_input_asset_url: {path} has an invalid transport URL")
        return False
    if scheme or value.startswith("//"):
        raise ValidationError(
            f"unsupported_input_asset_scheme: {path} requires HTTP(S) or an opaque stored ref"
        )
    return True


def _manifest_entry_valid(entry: InputManifestEntry) -> bool:
    return (
        bool(entry.asset_id.strip())
        and bool(entry.source_ref)
        and entry.source_ref == entry.source_ref.strip()
        and entry.size_bytes > 0
        and entry.kind in _MANIFEST_KINDS
        and bool(entry.mime_type.strip())
        and len(entry.blake3) == 64
        and set(entry.blake3) <= _HEX
    )


def _mime_allowed(mime: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    mime = (mime or "").lower()
    for item in allowed:
        item = str(item or "").lower()
        if not item:
            continue
        if mime == item:
            return True
        if item.endswith("/*") and mime.startswith(item[:-1]):
            return True
    return False


def _declared_kind_allows(asset: Asset, mime: str) -> bool:
    normalized = str(mime or "").lower()
    if isinstance(asset, ImageAsset):
        return normalized.startswith("image/")
    if isinstance(asset, VideoAsset):
        return normalized.startswith("video/")
    if isinstance(asset, AudioAsset):
        return normalized.startswith("audio/")
    return True


def _entry_kind_allows(kind: str, mime: str) -> bool:
    if kind == "media":
        return True
    return str(mime or "").lower().startswith(f"{kind}/")


def _requires_image_decode(occurrences: list[_AssetOccurrence]) -> bool:
    return any(
        isinstance(occurrence.asset, ImageAsset)
        or int(occurrence.asset.url_max_width or 0) > 0
        or int(occurrence.asset.url_max_height or 0) > 0
        or int(occurrence.asset.url_max_pixels or 0) > 0
        for occurrence in occurrences
    )


def _decode_image(path: Path, occurrences: list[_AssetOccurrence]) -> str:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - image endpoints install the images extra
        raise RetryableError(
            "input_asset_decoder_unavailable: worker image decoder is unavailable"
        ) from None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                mime = str(image.get_format_mimetype() or "").lower()
                # Tensorhub's assignment-specific bounds are intentionally
                # tighter than Pillow's process-global bomb threshold. Check
                # them from the header before any pixel buffer is allocated.
                _validate_image_dimensions(occurrences, int(width), int(height))
                image.verify()
            # ``verify`` checks container integrity without decoding pixels.
            # Reopen and load to prove the first image is actually decodable.
            with Image.open(path) as image:
                image.load()
    except ValidationError:
        raise
    except Exception:
        raise ValidationError(
            f"input_asset_decode_failed: {occurrences[0].path} is not a decodable image"
        ) from None
    return mime


def _validate_image_dimensions(
    occurrences: list[_AssetOccurrence], width: int, height: int
) -> None:
    for occurrence in occurrences:
        asset = occurrence.asset
        max_width = int(asset.url_max_width or 0)
        max_height = int(asset.url_max_height or 0)
        max_pixels = int(asset.url_max_pixels or 0)
        if (
            (max_width > 0 and width > max_width)
            or (max_height > 0 and height > max_height)
            or (max_pixels > 0 and width * height > max_pixels)
        ):
            raise ValidationError(
                f"input_asset_dimensions_exceeded: {occurrence.path} exceeds its image bounds"
            )


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise CanceledError("canceled")


def _effective_size_cap(occurrences: list[_AssetOccurrence]) -> int:
    caps = [
        int(occurrence.asset.url_max_bytes or 0)
        for occurrence in occurrences
        if int(occurrence.asset.url_max_bytes or 0) > 0
    ]
    return min(caps) if caps else _DEFAULT_MAX_BYTES


def _default_resolve_transport(
    url: str, headers: Mapping[str, str], body: bytes
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_RESOLVE_TIMEOUT_S) as response:
            return int(response.status), response.read(_MAX_RESOLVE_BODY + 1)
    except urllib.error.HTTPError as err:
        try:
            payload = err.read(_MAX_RESOLVE_BODY + 1)
        except Exception:
            payload = b""
        return int(err.code), payload


def _resolve_private_urls(
    manifest: Sequence[InputManifestEntry],
    *,
    request_id: str,
    attempt: int,
    file_base_url: str,
    capability_token: str,
    resolve_transport: ResolveTransport | None,
    cancel_check: Callable[[], bool] | None,
) -> list[str]:
    """Exactly one bounded resolver POST per attempt. Returns URLs aligned to
    ``manifest`` order. Response identity must byte-match the dispatched
    manifest; anything else is a platform malfunction (retryable)."""
    _check_cancel(cancel_check)
    if not file_base_url or not capability_token:
        raise RetryableError(
            "input_asset_resolution_unavailable: worker has no resolver credentials"
        )
    transport = resolve_transport or _default_resolve_transport
    url = file_base_url.rstrip("/") + _RESOLVE_PATH
    body = json.dumps({"request_id": request_id, "attempt": int(attempt)}).encode()
    headers = {
        "Authorization": f"Bearer {capability_token}",
        "Content-Type": "application/json",
    }
    try:
        status, raw = transport(url, headers, body)
    except CanceledError:
        raise
    except Exception:
        raise RetryableError(
            "input_asset_resolution_unavailable: resolver request failed"
        ) from None
    _check_cancel(cancel_check)
    if status == 409:
        # This attempt is no longer the assigned one (or its manifest moved):
        # stop without publishing anything; the hub owns the successor.
        raise CanceledError("canceled")
    if status != 200:
        raise RetryableError(
            "input_asset_resolution_unavailable: resolver refused the request"
        )
    if len(raw) > _MAX_RESOLVE_BODY:
        raise RetryableError(
            "input_asset_response_mismatch: resolver response exceeds its size bound"
        )
    try:
        decoded = msgspec.json.decode(raw, type=_ResolveResponse, strict=True)
    except (msgspec.DecodeError, msgspec.ValidationError):
        raise RetryableError(
            "input_asset_response_mismatch: resolver response is malformed"
        ) from None
    if len(decoded.assets) != len(manifest):
        raise RetryableError(
            "input_asset_response_mismatch: resolver response does not match the manifest"
        )
    urls: list[str] = []
    for entry, resolved in zip(manifest, decoded.assets):
        if (
            resolved.asset_id != entry.asset_id
            or resolved.source_ref != entry.source_ref
            or resolved.blake3 != entry.blake3
            or resolved.size_bytes != entry.size_bytes
            or resolved.kind != entry.kind
            or resolved.mime_type != entry.mime_type
        ):
            raise RetryableError(
                "input_asset_response_mismatch: resolver response does not match the manifest"
            )
        if not resolved.url or not _parse_rfc3339(resolved.url_expires_at):
            raise RetryableError(
                "input_asset_response_mismatch: resolver response is malformed"
            )
        urls.append(resolved.url)
    return urls


def _parse_rfc3339(value: str) -> bool:
    try:
        datetime.fromisoformat(str(value or ""))
    except ValueError:
        return False
    return True


def _internal_object_hosts() -> frozenset[str]:
    """GEN_WORKER_INTERNAL_OBJECT_HOSTS: exact hostnames of a deployment's
    internal object store (datacenter MinIO/NFS gateway). Honored ONLY for
    resolver-minted private-input URLs; caller public transports always face
    the full SSRF policy."""
    raw = os.environ.get("GEN_WORKER_INTERNAL_OBJECT_HOSTS", "")
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def _validate_transport_url(url: str, path: str, *, resolver_minted: bool = False) -> None:
    try:
        parsed = urllib.parse.urlsplit(str(url or ""))
    except ValueError:
        raise ValidationError(
            f"invalid_input_asset_url: {path} has an invalid transport URL"
        ) from None
    if parsed.scheme.lower() not in ("http", "https") or parsed.hostname is None:
        raise ValidationError(f"invalid_input_asset_url: {path} has an invalid transport URL")
    if resolver_minted and parsed.hostname.lower() in _internal_object_hosts():
        return
    try:
        blocked = _url_is_blocked(url)
    except Exception:
        raise RetryableError(
            "input_asset_validation_failed: transport policy check failed"
        ) from None
    if blocked:
        raise ValidationError(f"input_asset_url_not_allowed: {path} transport is blocked")


def _download(
    unit: _DownloadUnit,
    dest_dir: Path,
    index: int,
    cancel_check: Callable[[], bool] | None,
) -> tuple[Path, int, str]:
    _check_cancel(cancel_check)
    occurrences = unit.occurrences
    entry = unit.entry
    cap = _effective_size_cap(occurrences)
    if entry is not None:
        if entry.size_bytes > cap:
            raise ValidationError(
                f"input_asset_too_large: {occurrences[0].path} exceeds its byte cap"
            )
        cap = entry.size_bytes
    req = urllib.request.Request(unit.url, headers={"User-Agent": "gen-worker"})
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f"input-{index}-", suffix=".part", dir=dest_dir)
    except OSError:
        raise RetryableError(
            "input_asset_materialization_failed: worker temporary storage is unavailable"
        ) from None
    tmp = Path(tmp_name)
    total = 0
    head = b""
    hasher = blake3_mod.blake3() if entry is not None else None
    try:
        try:
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as response:
                raw_length = str(response.headers.get("Content-Length") or "").strip()
                expected_length = int(raw_length) if raw_length.isdigit() else 0
                if expected_length > cap:
                    raise ValidationError(
                        f"input_asset_too_large: {occurrences[0].path} exceeds its byte cap"
                    )
                with os.fdopen(fd, "wb") as output:
                    fd = -1
                    while True:
                        _check_cancel(cancel_check)
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        if not head:
                            head = chunk[:64]
                        total += len(chunk)
                        if total > cap:
                            if entry is not None:
                                raise RetryableError(
                                    "input_asset_integrity_failed: "
                                    f"{occurrences[0].path} bytes exceed their attested size"
                                )
                            raise ValidationError(
                                f"input_asset_too_large: {occurrences[0].path} exceeds its byte cap"
                            )
                        if hasher is not None:
                            hasher.update(chunk)
                        output.write(chunk)
                if expected_length and total != expected_length:
                    raise RetryableError(
                        "input_asset_download_failed: authorized transport was truncated"
                    )
        except (CanceledError, ValidationError, RetryableError):
            raise
        except Exception:
            raise RetryableError(
                "input_asset_download_failed: authorized transport could not be downloaded"
            ) from None

        _check_cancel(cancel_check)
        if entry is not None:
            if total != entry.size_bytes:
                raise RetryableError(
                    f"input_asset_integrity_failed: {occurrences[0].path} bytes do not "
                    "match their attested size"
                )
            assert hasher is not None
            if hasher.hexdigest() != entry.blake3:
                raise RetryableError(
                    f"input_asset_integrity_failed: {occurrences[0].path} bytes do not "
                    "match their attested BLAKE3"
                )
        mime = _infer_mime_type(unit.url, head)
        if _requires_image_decode(occurrences):
            mime = _decode_image(tmp, occurrences)
        _check_cancel(cancel_check)
        if entry is not None and not _entry_kind_allows(entry.kind, mime):
            raise ValidationError(
                f"input_asset_kind_mismatch: {occurrences[0].path} content is not its "
                "manifest media kind"
            )
        for occurrence in occurrences:
            asset = occurrence.asset
            if not _declared_kind_allows(asset, mime):
                raise ValidationError(
                    f"input_asset_kind_mismatch: {occurrence.path} content is not its declared media kind"
                )
            if not _mime_allowed(mime, tuple(asset.url_allowed_mime_types or ())):
                raise ValidationError(
                    f"input_asset_kind_mismatch: {occurrence.path} content is not allowed"
                )

        final = dest_dir / f"input-{index}{_EXT_BY_MIME.get(mime, '')}"
        try:
            os.replace(tmp, final)
        except OSError:
            raise RetryableError(
                "input_asset_materialization_failed: worker temporary storage is unavailable"
            ) from None
        return final, total, mime
    except BaseException:
        if fd >= 0:
            os.close(fd)
        tmp.unlink(missing_ok=True)
        raise


def _remove_scope_state(state: _ScopeState) -> None:
    for asset in state.assets:
        asset.local_path = None
    path = state.path
    temp_root = Path(tempfile.gettempdir()).resolve()
    if path.parent.resolve() != temp_root or not path.name.startswith(_INPUT_DIR_PREFIX):
        logger.error("refusing unsafe input-asset cleanup target")
        return
    if path.is_symlink():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def cleanup_input_assets(request_id: str, attempt: int = 0) -> None:
    """Remove one attempt's materialized inputs and clear every assigned path.

    Scopes are keyed by (request_id, attempt): cleaning attempt N+1 can never
    remove attempt N's directory, and vice versa.
    """
    key = _scope_key(request_id, attempt)
    with _scope_lock:
        state = _scope_states.pop(key, None)
    if state is not None:
        _remove_scope_state(state)


def _start_scope(request_id: str, attempt: int, assets: tuple[Asset, ...]) -> _ScopeState:
    key = _scope_key(request_id, attempt)
    path: Path | None = None
    try:
        path = Path(tempfile.mkdtemp(prefix=_INPUT_DIR_PREFIX))
        path.chmod(0o700)
    except OSError:
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)
        raise RetryableError(
            "input_asset_materialization_failed: worker temporary storage is unavailable"
        ) from None
    state = _ScopeState(path=path, assets=assets)
    with _scope_lock:
        previous = _scope_states.pop(key, None)
        _scope_states[key] = state
    if previous is not None:
        _remove_scope_state(previous)
    return state


def _collect_units(
    occurrences: list[_AssetOccurrence],
    manifest: Sequence[InputManifestEntry],
) -> list[_DownloadUnit]:
    """Classify occurrences, fence the private sequence against the dispatched
    manifest, and return download units in payload first-occurrence order."""
    units: list[_DownloadUnit] = []
    public_by_url: dict[str, _DownloadUnit] = {}
    private_by_key: dict[tuple[str, str], _DownloadUnit] = {}
    for occurrence in occurrences:
        ref = str(occurrence.asset.ref or "")
        if _classify_ref(ref, occurrence.path):
            key = (ref, _declared_kind(occurrence.asset))
            unit = private_by_key.get(key)
            if unit is None:
                unit = _DownloadUnit(occurrences=[])
                private_by_key[key] = unit
                units.append(unit)
            unit.occurrences.append(occurrence)
        else:
            unit = public_by_url.get(ref)
            if unit is None:
                unit = _DownloadUnit(occurrences=[], url=ref)
                public_by_url[ref] = unit
                units.append(unit)
            unit.occurrences.append(occurrence)

    # The ordered unique private sequence must equal the dispatched manifest
    # exactly (count, order, ref, kind) with valid immutable metadata BEFORE
    # any resolver call or GET. A mismatch is a non-retryable contract breach.
    private_sequence = list(private_by_key)
    manifest_sequence = [(entry.source_ref, entry.kind) for entry in manifest]
    if private_sequence != manifest_sequence:
        raise ValidationError(
            "input_asset_manifest_mismatch: dispatched manifest does not match "
            "the payload's private inputs"
        )
    for entry, key in zip(manifest, private_sequence):
        if not _manifest_entry_valid(entry):
            raise ValidationError(
                "input_asset_manifest_invalid: dispatched manifest entry is malformed"
            )
        private_by_key[key].entry = entry
    return units


def materialize_input_assets(
    payload: Any,
    request_id: str,
    *,
    attempt: int = 0,
    manifest: Sequence[InputManifestEntry] = (),
    file_base_url: str = "",
    capability_token: str = "",
    cancel_check: Callable[[], bool] | None = None,
    resolve_transport: ResolveTransport | None = None,
) -> int:
    """Materialize all ordered input assets transactionally.

    Returns the number of distinct downloads. Duplicate occurrences retain
    their payload positions and share one worker-owned path. Every decoded
    Asset field is preserved — only ``local_path`` is assigned. Any failure
    clears assigned paths and removes the whole attempt directory before
    propagating a stable causal error.
    """
    cleanup_input_assets(request_id, attempt)
    occurrences = list(_iter_asset_occurrences(payload))
    for occurrence in occurrences:
        # A local path in RunJob input is caller-controlled wire data. Never
        # use it as evidence that the asset was materialized.
        occurrence.asset.local_path = None
    _check_cancel(cancel_check)
    manifest = tuple(manifest)
    if not occurrences:
        if manifest:
            raise ValidationError(
                "input_asset_manifest_mismatch: dispatched manifest does not match "
                "the payload's private inputs"
            )
        return 0

    units = _collect_units(occurrences, manifest)
    private_units = [unit for unit in units if unit.entry is not None]
    if private_units and int(attempt) <= 0:
        raise ValidationError(
            "input_asset_attempt_invalid: private inputs require a positive attempt"
        )

    # Validate the complete public transport set before the resolver call and
    # before downloading the first item, so a later blocked ref cannot cause
    # partial materialization work.
    for unit in units:
        if unit.entry is None:
            _validate_transport_url(unit.url, unit.occurrences[0].path)

    if private_units:
        urls = _resolve_private_urls(
            manifest,
            request_id=request_id,
            attempt=int(attempt),
            file_base_url=file_base_url,
            capability_token=capability_token,
            resolve_transport=resolve_transport,
            cancel_check=cancel_check,
        )
        by_entry = dict(zip(manifest, urls))
        for unit in private_units:
            assert unit.entry is not None
            unit.url = by_entry[unit.entry]
            _validate_transport_url(unit.url, unit.occurrences[0].path, resolver_minted=True)

    state = _start_scope(
        request_id,
        attempt,
        tuple(occurrence.asset for occurrence in occurrences),
    )
    try:
        for index, unit in enumerate(units):
            local_path, size_bytes, mime_type = _download(unit, state.path, index, cancel_check)
            for occurrence in unit.occurrences:
                occurrence.asset.local_path = str(local_path)
            logger.info(
                "materialized input asset index=%d aliases=%d bytes=%d mime=%s private=%s",
                index,
                len(unit.occurrences),
                size_bytes,
                mime_type,
                unit.entry is not None,
            )
        return len(units)
    except BaseException:
        cleanup_input_assets(request_id, attempt)
        raise


__all__ = [
    "InputManifestEntry",
    "cleanup_input_assets",
    "inputs_dir_for_request",
    "manifest_from_run_job",
    "materialize_input_assets",
]
