"""LocalRequestContext — RequestContext subclass for ``gen-worker run``.

Build the right kind-specific context for the selected method (RequestContext
for inference, ConversionContext for conversion, DatasetContext for dataset,
TrainingContext for training). Override the orchestrator-backed bits with
local-mode equivalents:

- ``emitter`` writes JSON lines to stderr so ``ctx.emit / progress / log``
  events are visible without competing with the stdout result.
- ``save_bytes`` / ``save_file`` materialize files under
  ``./.gen-worker-run/outputs/<ref>`` and return an Asset with ``local_path``
  set (no tensorhub upload).
- ``publish_repo_revision`` and ``materialize_blob`` on ConversionContext
  print the would-be payload to stderr and return a fake response, gated
  by ``--allow-publish``.
- ``_canceled`` is toggled by the installed SIGINT handler in ``run.py``.

Construction is shaped so the only producer of a LocalRequestContext is the
``build_local_context`` factory below — keeps the wiring choices in one place.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..api.types import Asset
from ..request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)


_LOCAL_OUTPUT_DIR_NAME = ".gen-worker-run"


def _local_request_id() -> str:
    return f"local-{uuid.uuid4().hex[:8]}"


def _stderr_emitter(event: Dict[str, Any]) -> None:
    """JSON-line emitter writing to stderr.

    Each event is one self-contained JSON object on its own line, so a tail-
    like consumer can ``jq`` over the stream. Falls back silently on any
    serialization error — emit is best-effort by contract.
    """
    try:
        line = json.dumps(event, separators=(",", ":"), sort_keys=True, default=str)
    except Exception:
        return
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _local_outputs_root() -> Path:
    # cozy sets GEN_WORKER_LOCAL_OUTPUT_DIR so generated assets land in a
    # user-facing dir (e.g. ~/.cache/cozy/outputs) instead of being buried in
    # the endpoint's install dir, while serve still runs with cwd=endpoint dir
    # so discovery can find `main`. Falls back to cwd/.gen-worker-run as before.
    env = os.environ.get("GEN_WORKER_LOCAL_OUTPUT_DIR")
    if env and env.strip():
        root = Path(env).expanduser()
    else:
        root = Path.cwd() / _LOCAL_OUTPUT_DIR_NAME / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_local_bytes(ref: str, data: bytes) -> Asset:
    """Write data to ``./.gen-worker-run/outputs/<ref>`` and return an Asset.

    ``ref`` is allowed to contain ``/`` separators — we sanitize the leading
    slash and any ``..`` segments so a malicious / sloppy ref can't escape
    the local outputs dir.
    """
    safe = (ref or "").strip().lstrip("/")
    parts = [p for p in safe.split("/") if p and p != ".."]
    safe = "/".join(parts) or f"out-{uuid.uuid4().hex[:8]}"
    dest = _local_outputs_root() / safe
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return Asset(
        ref=safe,
        local_path=str(dest),
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _save_local_file(ref: str, src: "str | os.PathLike[str]") -> Asset:
    p = Path(src)
    data = p.read_bytes()
    return _save_local_bytes(ref, data)


class LocalRequestContextMixin:
    """Shared behavior for every local context subclass.

    Cannot be a plain mixin on the base RequestContext directly because the
    conversion / dataset / training subclasses each add their own producer-
    RPC methods we want to neutralize. Each Local* subclass below mixes this
    in plus the matching parent class.
    """

    def __init__(self, *args: Any, allow_publish: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._local_allow_publish = bool(allow_publish)

    def save_bytes(self, ref: str, data: bytes) -> Asset:  # type: ignore[override]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_bytes expects bytes")
        return _save_local_bytes(ref, bytes(data))

    def save_file(self, ref: str, local_path: "str | os.PathLike[str]") -> Asset:  # type: ignore[override]
        return _save_local_file(ref, local_path)


class LocalRequestContext(LocalRequestContextMixin, RequestContext):
    """Inference-kind local context."""


class LocalConversionContext(LocalRequestContextMixin, ConversionContext):
    """Conversion-kind local context.

    ``publish_repo_revision`` / ``materialize_blob`` are stubbed: they print
    the would-be payload to stderr and return a fake response unless
    ``--allow-publish`` was passed (in which case we delegate to the real
    implementation — useful for round-tripping against a dev tensorhub).
    """

    def publish_repo_revision(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        if self._local_allow_publish:
            return super().publish_repo_revision(**kwargs)
        _stderr_emitter({
            "kind": "publish_repo_revision.stubbed",
            "would_publish": kwargs,
        })
        return {
            "stubbed": True,
            "destination_repo": kwargs.get("destination_repo"),
            "revision_id": f"local-{uuid.uuid4().hex[:8]}",
        }

    def materialize_blob(self, digest: str, dest: "str | os.PathLike[str]") -> Path:  # type: ignore[override]
        if self._local_allow_publish:
            return super().materialize_blob(digest, dest)
        # Local stub: look in the tensorhub CAS for a matching snapshot. If
        # nothing's there we can't materialize — surface a typed error so
        # the tenant adjusts (run without --offline first, or seed the CAS).
        from ..models.cache_paths import tensorhub_cas_dir
        d = (digest or "").strip()
        if not d:
            raise ValueError("materialize_blob: empty digest")
        prefix = d.split(":", 1)[-1]
        candidate = tensorhub_cas_dir() / "blobs" / prefix
        if not candidate.exists():
            raise FileNotFoundError(
                f"materialize_blob: blob {digest!r} not found in local CAS "
                f"({tensorhub_cas_dir()}); rerun without --offline or pre-seed "
                "the CAS by running an unrelated job that produces this blob."
            )
        out = Path(dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(candidate.read_bytes())
        return out


class LocalDatasetContext(LocalRequestContextMixin, DatasetContext):
    """Dataset-kind local context."""

    def materialize_blob(self, digest: str, dest: "str | os.PathLike[str]") -> Path:  # type: ignore[override]
        # Same fallback as ConversionContext.
        from ..models.cache_paths import tensorhub_cas_dir
        d = (digest or "").strip()
        if not d:
            raise ValueError("materialize_blob: empty digest")
        prefix = d.split(":", 1)[-1]
        candidate = tensorhub_cas_dir() / "blobs" / prefix
        if not candidate.exists():
            raise FileNotFoundError(
                f"materialize_blob: blob {digest!r} not found in local CAS "
                f"({tensorhub_cas_dir()})"
            )
        out = Path(dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(candidate.read_bytes())
        return out


class LocalTrainingContext(LocalRequestContextMixin, TrainingContext):
    """Training-kind local context."""


def build_local_context(
    *,
    kind: str,
    allow_publish: bool = False,
    request_id: Optional[str] = None,
    owner: Optional[str] = None,
    emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> RequestContext:
    """Factory: build the right context subclass for ``kind``.

    ``kind`` is the endpoint's declared kind string from discover_manifest
    (``inference`` / ``conversion`` / ``dataset`` / ``training``). Unknown
    kinds fall back to ``RequestContext`` because the SDK guarantees the
    base methods on every variant.
    """
    rid = request_id or _local_request_id()
    em = emitter if emitter is not None else _stderr_emitter
    common: Dict[str, Any] = {
        "request_id": rid,
        "job_id": rid,
        "emitter": em,
        "owner": owner or os.getenv("USER") or "local-dev",
        # Honor local_output_dir so save_bytes routes through the resolved-
        # local-path branch on the base class (matches production semantics).
        "local_output_dir": str(_local_outputs_root()),
    }

    k = (kind or "").strip().lower()
    if k == "conversion":
        return LocalConversionContext(allow_publish=allow_publish, **common)
    if k == "dataset":
        return LocalDatasetContext(allow_publish=allow_publish, **common)
    if k == "training":
        return LocalTrainingContext(allow_publish=allow_publish, **common)
    return LocalRequestContext(allow_publish=allow_publish, **common)


__all__ = [
    "build_local_context",
    "LocalRequestContext",
    "LocalConversionContext",
    "LocalDatasetContext",
    "LocalTrainingContext",
    "_stderr_emitter",
]
