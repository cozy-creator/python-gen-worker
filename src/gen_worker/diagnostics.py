from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Optional


DiagnosticEmitter = Callable[..., bool]

_diagnostic_emitter: ContextVar[Optional[DiagnosticEmitter]] = ContextVar(
    "gen_worker_diagnostic_emitter",
    default=None,
)


def emit_diagnostic_log(
    category: str,
    message: str = "",
    payload: dict[str, Any] | None = None,
    *,
    severity: str = "info",
    endpoint_class: str = "",
    function_name: str = "",
    image_digest: str = "",
) -> bool:
    """Emit a best-effort internal worker diagnostic.

    Returns False when called outside a worker-managed setup/request context or
    when the connected worker runtime does not support diagnostics. This helper
    never raises.
    """
    emitter = _diagnostic_emitter.get()
    if emitter is None:
        return False
    try:
        return bool(
            emitter(
                category=category,
                message=message,
                payload=payload,
                severity=severity,
                endpoint_class=endpoint_class,
                function_name=function_name,
                image_digest=image_digest,
            )
        )
    except Exception:
        return False


@contextmanager
def diagnostic_emitter_context(emitter: DiagnosticEmitter) -> Iterator[None]:
    token = _diagnostic_emitter.set(emitter)
    try:
        yield
    finally:
        _diagnostic_emitter.reset(token)
