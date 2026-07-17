"""Fail-closed materialization of RunJob input ``Asset`` values.

Tensorhub owns stored-ref authorization and presigning. The worker accepts only
the resulting HTTP(S) transport refs, downloads each distinct ref once into a
fresh worker-owned directory, and writes that local path back to every ordered
occurrence. Opaque stored refs and caller-provided filesystem paths never cross
this trust boundary.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import msgspec

from .api.errors import CanceledError, RetryableError, ValidationError
from .api.types import Asset, AudioAsset, ImageAsset, VideoAsset
from .request_context._helpers import _infer_mime_type, _url_is_blocked

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 50 << 20  # matches tensorhub's default media cap
_DOWNLOAD_TIMEOUT_S = 120
_MAX_WALK_DEPTH = 32
_CHUNK = 1 << 20
_INPUT_DIR_PREFIX = "gen-worker-inputs-"

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
}


@dataclass(frozen=True)
class _AssetOccurrence:
    path: str
    asset: Asset


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
    elif isinstance(obj, dict):
        # Dict keys may contain caller data. Use stable insertion positions in
        # causal errors while preserving the value traversal order exactly.
        for index, item in enumerate(obj.values()):
            yield from _iter_asset_occurrences(item, f"{path}[{index}]", depth + 1)


def _iter_assets(obj: Any, depth: int = 0) -> Iterator[Asset]:
    """Compatibility iterator: ordered, duplicate-preserving Asset objects."""
    for occurrence in _iter_asset_occurrences(obj, depth=depth):
        yield occurrence.asset


def _transport_url(ref: str, path: str) -> str:
    value = str(ref or "").strip()
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        raise ValidationError(
            f"invalid_input_asset_url: {path} has an invalid transport URL"
        ) from None
    scheme = parsed.scheme.lower()
    if not scheme:
        raise ValidationError(
            f"unresolved_input_asset: {path} requires an authorized HTTP(S) transport URL"
        )
    if scheme not in ("http", "https"):
        raise ValidationError(f"unsupported_input_asset_scheme: {path} requires HTTP(S)")
    if not parsed.netloc or parsed.hostname is None:
        raise ValidationError(f"invalid_input_asset_url: {path} has an invalid transport URL")
    return value


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


def _requires_image_decode(occurrences: list[_AssetOccurrence]) -> bool:
    return any(
        isinstance(occurrence.asset, ImageAsset)
        or int(occurrence.asset.url_max_width or 0) > 0
        or int(occurrence.asset.url_max_height or 0) > 0
        or int(occurrence.asset.url_max_pixels or 0) > 0
        for occurrence in occurrences
    )


def _decode_image(path: Path, occurrence_path: str) -> tuple[str, int, int]:
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
                image.verify()
            # ``verify`` checks container integrity without decoding pixels.
            # Reopen and load to prove the first image is actually decodable.
            with Image.open(path) as image:
                image.load()
    except Exception:
        raise ValidationError(
            f"input_asset_decode_failed: {occurrence_path} is not a decodable image"
        ) from None
    return mime, int(width), int(height)


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


def _download(
    url: str,
    occurrences: list[_AssetOccurrence],
    dest_dir: Path,
    index: int,
    cancel_check: Callable[[], bool] | None,
) -> tuple[Path, int, str]:
    _check_cancel(cancel_check)
    cap = _effective_size_cap(occurrences)
    req = urllib.request.Request(url, headers={"User-Agent": "gen-worker"})
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f"input-{index}-", suffix=".part", dir=dest_dir)
    except OSError:
        raise RetryableError(
            "input_asset_materialization_failed: worker temporary storage is unavailable"
        ) from None
    tmp = Path(tmp_name)
    total = 0
    head = b""
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
                            raise ValidationError(
                                f"input_asset_too_large: {occurrences[0].path} exceeds its byte cap"
                            )
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
        mime = _infer_mime_type(url, head)
        if _requires_image_decode(occurrences):
            mime, width, height = _decode_image(tmp, occurrences[0].path)
            _validate_image_dimensions(occurrences, width, height)
        _check_cancel(cancel_check)
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
    """Remove one attempt's materialized inputs and clear every assigned path."""
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


def materialize_input_assets(
    payload: Any,
    request_id: str,
    *,
    attempt: int = 0,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Materialize all ordered input assets transactionally.

    The return value is the number of distinct HTTP(S) downloads. Duplicate
    occurrences retain their payload positions and share one worker-owned
    path. Any failure clears assigned paths and removes the whole attempt
    directory before propagating a stable causal error.
    """
    cleanup_input_assets(request_id, attempt)
    occurrences = list(_iter_asset_occurrences(payload))
    for occurrence in occurrences:
        # A local path in RunJob input is caller-controlled wire data. Never
        # use it as evidence that the asset was materialized.
        occurrence.asset.local_path = None
    _check_cancel(cancel_check)
    if not occurrences:
        return 0

    grouped: dict[str, list[_AssetOccurrence]] = {}
    for occurrence in occurrences:
        url = _transport_url(occurrence.asset.ref, occurrence.path)
        grouped.setdefault(url, []).append(occurrence)

    # Validate the complete transport set before downloading the first item,
    # so a later opaque/blocked ref cannot cause partial materialization work.
    for occurrences_for_url in grouped.values():
        url = str(occurrences_for_url[0].asset.ref).strip()
        try:
            blocked = _url_is_blocked(url)
        except Exception:
            raise RetryableError(
                "input_asset_validation_failed: transport policy check failed"
            ) from None
        if blocked:
            raise ValidationError(
                f"input_asset_url_not_allowed: {occurrences_for_url[0].path} transport is blocked"
            )

    state = _start_scope(
        request_id,
        attempt,
        tuple(occurrence.asset for occurrence in occurrences),
    )
    try:
        for index, (url, duplicates) in enumerate(grouped.items()):
            local_path, size_bytes, mime_type = _download(
                url, duplicates, state.path, index, cancel_check
            )
            for occurrence in duplicates:
                asset = occurrence.asset
                asset.local_path = str(local_path)
                asset.size_bytes = size_bytes
                asset.mime_type = mime_type
            logger.info(
                "materialized input asset index=%d aliases=%d bytes=%d mime=%s",
                index,
                len(duplicates),
                size_bytes,
                mime_type,
            )
        return len(grouped)
    except BaseException:
        cleanup_input_assets(request_id, attempt)
        raise


__all__ = [
    "cleanup_input_assets",
    "inputs_dir_for_request",
    "materialize_input_assets",
]
