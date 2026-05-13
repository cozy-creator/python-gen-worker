"""Private injection-marker types used by ``@training_function``.

The public ``Annotated[T, ModelRef(...)]`` decoration model was removed from
the inference SDK in gen-worker 0.7.0 (replaced by ``models={...}`` on
``@inference_function``), but ``@training_function`` still uses the
``Annotated[Source, _PayloadRef("wire_field")]`` shape internally to bind
secondary materialization parameters. Training's wire payload remains
JSON-shaped and the binding model from inference doesn't apply.

This module is **not** part of the public SDK surface; tenants of
``@training_function`` should not import it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Optional, get_args, get_origin


@dataclass(frozen=True)
class _PayloadRef:
    """Annotated-marker for a secondary materialization param on a
    ``@training_function`` signature.

    ``key`` is the wire-payload field name carrying the SourceRepo-shaped
    dict the worker uses to materialize the secondary :class:`Source`.
    """

    key: str


def _parse_payload_ref(annotation: Any) -> Optional[tuple[Any, _PayloadRef]]:
    """Return ``(base_type, _PayloadRef)`` if the annotation is
    ``Annotated[base_type, _PayloadRef(...)]``, else None.
    """
    origin = get_origin(annotation)
    if origin is not Annotated:
        return None
    args = get_args(annotation)
    if not args:
        return None
    base = args[0]
    for meta in args[1:]:
        if isinstance(meta, _PayloadRef):
            return base, meta
    return None


__all__ = ["_PayloadRef", "_parse_payload_ref"]
