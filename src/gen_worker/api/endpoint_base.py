"""``gen_worker.Endpoint`` -- the required base for ship-code-as-is endpoints.

Paul's main_v2.py review rulings (pgw#1367 program): every lanes-surface
endpoint class inherits this base, GENERIC over its model type --
``class SdxlEndpoint(Endpoint[SDXL])``. The class header is THE single
source of the endpoint's expected model type: statically extractable at
publish (no code execution), stamped into the release, and the type
``ctx.defaults()`` decodes to. The base carries typed no-op lifecycle hooks
and NOTHING else -- capabilities reach author code through ``ctx`` only, and
the ``@endpoint`` decorator stays the metadata carrier. The serving lane
(pgw#1372) owns this class's evolution; keep it minimal here.
"""

from __future__ import annotations

import typing
from typing import Any, Generic, Optional, TypeVar

M = TypeVar("M")


class Endpoint(Generic[M]):
    """Base class for ``@endpoint(lanes=...)`` classes.

    ``setup`` runs once per resident lane before any handler; ``teardown``
    runs when the instance is retired. Both are no-op hooks an endpoint
    overrides as needed -- handlers are ordinary public methods
    ``(self, ctx, payload)``.
    """

    def setup(self, ctx: Any) -> None:  # noqa: B027 - deliberate no-op hook
        """Load and mark models; the default does nothing."""

    def teardown(self, ctx: Any) -> None:  # noqa: B027 - deliberate no-op hook
        """Release what setup acquired; the default does nothing."""


def endpoint_model_type(cls: type) -> Optional[type]:
    """The model type in the class header (``Endpoint[SDXL]`` -> ``SDXL``).

    Static by construction: read off ``__orig_bases__``, never off an
    instance. ``None`` when the base is unparameterized.
    """

    for base in getattr(cls, "__orig_bases__", ()):
        if typing.get_origin(base) is Endpoint:
            arguments = typing.get_args(base)
            if len(arguments) == 1 and isinstance(arguments[0], type):
                return arguments[0]
    return None


__all__ = ["Endpoint", "endpoint_model_type"]
