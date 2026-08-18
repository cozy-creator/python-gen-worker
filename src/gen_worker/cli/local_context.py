"""LocalRequestContext — RequestContext subclass for ``gen-worker run``.

Build the context for the selected method — ``LocalRequestContext`` for
inference, ``LocalJobContext`` for every producer kind and for ``@job``.
There are TWO classes here, not five, and that is pgw#1306's point: kind
selects a WIRE shape, never a context class and never write authority. Override
the orchestrator-backed bits with local-mode equivalents:

- ``emitter`` writes JSON lines to stderr so ``ctx.emit / progress / log``
  events are visible without competing with the stdout result.
- ``save_bytes`` / ``save_file`` materialize files under
  ``./.gen-worker-run/outputs/<ref>`` and return an Asset with ``local_path``
  set (no tensorhub upload).
- ``materialize_blob`` on ``LocalJobContext`` falls back to the local CAS
  unless ``--allow-publish`` delegates to the real implementation. ONE answer
  for every producer kind (pgw#1306).
  (Checkpoint publishing goes through ``gen_worker.convert.publish_flavors``,
  which talks to tensorhub directly and fails loudly without credentials.)
- ``_canceled`` is toggled by the installed SIGINT handler in ``run.py``.

Construction is shaped so the only producer of a LocalRequestContext is the
``build_local_context`` factory below — keeps the wiring choices in one place.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gen_worker._vendor.tensorfs import CASRef

from ..api.decorators import KINDS
from ..api.types import Asset
from ..models.cache_paths import open_worker_cas
from ..request_context import (
    REF_ORIGIN_PAYLOAD,
    JobContext,
    RequestContext,
)

_LOCAL_OUTPUT_DIR_NAME = ".gen-worker-run"

#: Every kind this factory builds a context for: the @endpoint vocabulary
#: (one home, `api.decorators.KINDS`) plus `job`, which is @job's own.
LOCAL_CONTEXT_KINDS = KINDS + ("job",)


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
        super().__init__(*args, **kwargs)
        self._local_allow_publish = bool(allow_publish)

    def save_bytes(self, ref: str, data: bytes) -> Asset:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_bytes expects bytes")
        return _save_local_bytes(ref, bytes(data))

    def save_file(
        self, ref: str, local_path: "str | os.PathLike[str]", *, create: bool = False
    ) -> Asset:
        # `create` is a no-op locally: dev runs overwrite freely.
        return _save_local_file(ref, local_path)


class LocalRequestContext(LocalRequestContextMixin, RequestContext):
    """Inference-kind local context."""


def _materialize_local_blob(digest: str, dest: str | os.PathLike[str]) -> Path:
    ref = CASRef.parse(digest)
    source = open_worker_cas().verify_object(ref)
    output = Path(dest)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return output


class LocalJobContext(LocalRequestContextMixin, JobContext):
    """The local context every producer body gets: ``@job`` runs and every
    producer-kind ``@endpoint`` alike.

    ``materialize_blob`` is stubbed against the local CAS unless
    ``--allow-publish`` was passed (in which case we delegate to the real
    implementation — useful for round-tripping against a dev tensorhub).

    **ONE class for every producer kind (pgw#1306), and that is the defect
    fix.** Three kind-selected classes disagreed about whether
    ``--allow-publish`` was honored, so ``gen-worker run --offline`` on a
    training endpoint reached the REAL hub-backed ``materialize_blob`` in the
    loop whose entire purpose is to not have a hub. One class cannot disagree
    with itself.
    """

    def materialize_blob(
        self, digest: str, dest: "str | os.PathLike[str]",
        *, origin: str = REF_ORIGIN_PAYLOAD,
    ) -> Path:
        if self._local_allow_publish:
            return super().materialize_blob(digest, dest, origin=origin)
        # Local stub: look in the tensorhub CAS for a matching snapshot. If
        # nothing's there we can't materialize — surface a typed error so
        # the tenant adjusts (run without --offline first, or seed the CAS).
        d = (digest or "").strip()
        if not d:
            raise ValueError("materialize_blob: empty digest")
        try:
            return _materialize_local_blob(d, dest)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"materialize_blob: blob {digest!r} not found in local CAS "
                f"({open_worker_cas().root}); rerun without --offline or pre-seed "
                "the CAS by running an unrelated job that produces this blob."
            ) from None


def build_local_context(
    *,
    kind: str,
    allow_publish: bool = False,
    request_id: Optional[str] = None,
    owner: Optional[str] = None,
    emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
    publishes: bool = False,
    emits_media: Optional[bool] = None,
) -> RequestContext:
    """Factory: build the context for ``kind``.

    ``kind`` is the endpoint's declared kind string from discover_manifest, or
    ``job`` for a ``@job`` run. It answers exactly one question — *is this body
    a producer?* — and every producer answer is the same class. An unrecognized
    kind is a REFUSAL, never a base ``RequestContext``: the base class silently
    lacks the producer surface (``materialize_blob``, ``save_checkpoint``, the
    dataset writers), so the substitution surfaces as a missing attribute deep
    inside a tenant body — a typo'd kind reading as "this endpoint has no
    publisher".
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
        # The hub-write declaration travels into the local run too: an author
        # who forgot publishes=True must hit the refusal HERE, in the dev
        # loop, and not for the first time on a rented pod.
        "publishes": bool(publishes),
        # Jobs only, and the same argument: an author who forgot
        # emits_media=True meets the refusal in the dev loop, not on a pod.
        "emits_media": emits_media,
    }

    k = (kind or "").strip().lower()
    # Every producer kind — and @job — gets the SAME class. The list is spelled
    # out rather than written as `!= "inference"` so an unknown kind still hits
    # the refusal below instead of being silently promoted to a producer.
    if k in ("job", "conversion", "eval", "dataset", "training"):
        return LocalJobContext(allow_publish=allow_publish, **common)
    if k == "inference":
        return LocalRequestContext(allow_publish=allow_publish, **common)
    raise ValueError(
        f"build_local_context: unknown kind {kind!r}. The declared kinds are "
        f"{', '.join(LOCAL_CONTEXT_KINDS)} — a kind this factory does not know "
        f"is a typo or an unported kind, and handing it a base RequestContext "
        f"would drop the surface its handler is written against."
    )


__all__ = [
    "build_local_context",
    "LocalRequestContext",
    "LocalJobContext",
    "_stderr_emitter",
]
