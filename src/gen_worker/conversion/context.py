"""ConversionContext — tenant-facing helpers for transform jobs.

Exposes:
  - cancelled: bool                                  — cooperative cancellation
  - mktemp() -> Path                                 — scratch dir (auto-cleaned)
  - checkpoint_dir(*, key) -> Path                   — persistent scratch
                                                       (survives worker restart)
  - open_output_writer() -> StreamingWriter
  - load_reference_model(ref) -> PreTrainedModel     — dynamic secondary-model load
  - copy_unconverted_components(source, out_dir, *, skip=[]) -> None
  - emit(event, data) -> None                        — telemetry passthrough

ConversionContext is a thin wrapper around ``RequestContext``; the library
builds it from the underlying request-level context before invoking the
tenant function.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ..request_context import RequestContext

    from .source import Source
    from .writer import StreamingWriter


class ConversionContext:
    """Tenant-facing context for transform-kind jobs."""

    def __init__(
        self,
        *,
        request_context: "RequestContext",
        source: "Source | None",
    ) -> None:
        self._req = request_context
        # Source is None for dataset-generation tenants that are
        # model-agnostic. Tenants that operate on a checkpoint declare
        # ``source: Source`` in their signature; the library builds it and
        # passes it here.
        self._source = source
        # Lazy-created temp roots; one per job invocation.
        self._mktemp_root: Path | None = None
        self._open_writers: list["StreamingWriter"] = []

    # ----- cancellation -----------------------------------------------

    @property
    def cancelled(self) -> bool:
        """Return True if the scheduler has signaled cancellation.

        Tenant polls between iterations; library sets when the worker
        receives a cancel gRPC message. Poll granularity is the tenant's
        choice — per-tensor inside iter_tensors loops, or per-spec / per-step
        for whole-model functions.
        """
        # RequestContext exposes a cancel event / flag — hook up to whichever
        # mechanism the host framework uses.
        return bool(getattr(self._req, "is_cancelled", False))

    # ----- scratch dirs -----------------------------------------------

    def mktemp(self) -> Path:
        """Return a job-scoped scratch directory. Contents are NOT persisted.

        Auto-cleaned at job end. Each call returns a fresh subdir so tenants
        can use it as ``out_dir`` for ``model.save_pretrained(ctx.mktemp())``
        without collision.
        """
        if self._mktemp_root is None:
            base = getattr(self._req, "scratch_dir", None) or tempfile.gettempdir()
            self._mktemp_root = Path(
                tempfile.mkdtemp(
                    prefix=f"txform-{getattr(self._req, 'request_id', 'x')}-",
                    dir=str(base),
                )
            )
        return Path(tempfile.mkdtemp(dir=str(self._mktemp_root)))

    def checkpoint_dir(self, *, key: str) -> Path:
        """Return a PERSISTENT scratch dir keyed by (job_id, key).

        Survives worker restart — intended for transformers.Trainer.output_dir
        so ``resume_from_checkpoint=True`` can pick up where a preempted job
        left off. Auto-cleaned only when the job reaches a terminal state.

        Callers should use a stable key within a single job (e.g. the
        spec index). The returned path is the SAME across worker restarts.
        """
        job_id = getattr(self._req, "job_id", None) or getattr(self._req, "request_id", "x")
        base = getattr(self._req, "persistent_scratch_dir", None)
        if base is None:
            # Fallback: /tmp — not persistent across host restarts, but better
            # than nothing. Host runtime should provide persistent_scratch_dir.
            base = Path(tempfile.gettempdir()) / "txform-persistent" / str(job_id)
        else:
            base = Path(base) / str(job_id)
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        dir_path = base / safe_key
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    # ----- output writer ----------------------------------------------

    def open_output_writer(self) -> "StreamingWriter":
        """Return a fresh StreamingWriter for one output variant.

        Call once per entry in the tenant's specs list. The returned writer
        is scoped to a unique subdirectory under mktemp(); tenants don't
        pick paths themselves.
        """
        from .writer import StreamingWriter

        out_dir = self.mktemp()
        w = StreamingWriter(source=self._source, out_dir=out_dir)
        self._open_writers.append(w)
        return w

    # ----- secondary-model load ---------------------------------------

    def load_reference_model(self, ref: str) -> Any:
        """Dynamically load a secondary model by tensorhub ref.

        Prefer the declarative ``Annotated[Source, ModelRef(Src.PAYLOAD,
        "field_name")]`` pattern when the ref is known at dispatch time —
        that scopes the capability token correctly at token-mint time. This
        method is the escape hatch when the ref isn't statically known.

        Raises if the capability token doesn't already cover the ref.
        """
        # Deferred — requires gen-worker's existing ref-downloader path with
        # capability-token auth. The MVP implementation wires this up after
        # the per-function ref-registry is plumbed through orchestrator +
        # endpoint.lock. For now, raise a clear not-implemented error rather
        # than silently downloading without authorization checks.
        raise NotImplementedError(
            "ctx.load_reference_model is not yet implemented — use the "
            "declarative Annotated[Source, ModelRef(Src.PAYLOAD, ...)] "
            "pattern instead (orchestrator pre-scopes the capability token)."
        )

    # ----- layout passthrough escape hatch ----------------------------

    def copy_unconverted_components(
        self,
        source: "Source",
        out_dir: Path,
        *,
        skip: Iterable[str] = (),
    ) -> None:
        """Copy components from source → out_dir that the tenant didn't produce.

        For tenants that use ``source.as_hf_model() + model.save_pretrained()``
        and want source-layout passthrough for non-touched components. The
        ``skip`` iterable names components the tenant HAS produced (and thus
        should not be overwritten).
        """
        skip_set = set(skip)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for comp_name, comp in source.components.items():
            if comp_name in skip_set:
                continue
            dst = out_dir / comp_name
            if dst.exists():
                continue
            shutil.copytree(str(comp.path), str(dst))

    # ----- telemetry --------------------------------------------------

    def emit(self, event: str, data: dict) -> None:
        """Emit a telemetry event. Forwards to the underlying RequestContext."""
        emit_fn = getattr(self._req, "emit", None)
        if emit_fn is not None:
            emit_fn(event, data)


__all__ = ["ConversionContext"]
