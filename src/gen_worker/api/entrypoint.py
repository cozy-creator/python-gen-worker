"""``@entrypoint`` -- the stateless half of the ship-code-as-is surface.

An entrypoint is a FREE FUNCTION ``(payload, model, ctx) -> output``: pure
composition over a resident :class:`~gen_worker.api.model_base.Model`
instance the platform binds per (checkpoint x lane). The decorator is a
marker plus a shape check; payload/model types are read off the annotations
(statically at publish, live at dispatch).

NAME HAZARD, flagged for pgw#1372: ``gen_worker/entrypoint.py`` (the worker
process entry module) shares this name. ``from gen_worker import entrypoint``
resolves to THIS decorator through the package's lazy export table -- unless
some earlier import of ``gen_worker.entrypoint`` (the module) has already
bound the submodule onto the package. The derive path never imports that
module; the serving loader must not either before author code imports, or
the module gets renamed.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

ENTRYPOINT_ATTR = "__gen_worker_entrypoint__"


def entrypoint(fn: F) -> F:
    if not inspect.isfunction(fn):
        raise TypeError(
            f"@entrypoint decorates functions, got {type(fn).__name__}"
        )
    parameters = [
        parameter
        for parameter in inspect.signature(fn).parameters.values()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if len(parameters) < 3:
        raise TypeError(
            f"@entrypoint {fn.__name__}: an entrypoint takes at least "
            f"(payload, model, ctx); further parameters are platform-injected "
            f"FACTS by annotation (e.g. `turbo: Adapter | None`, "
            f"`loras: list[Adapter]`); got {len(parameters)} parameters"
        )
    setattr(fn, ENTRYPOINT_ATTR, True)
    return fn


__all__ = ["ENTRYPOINT_ATTR", "entrypoint"]
