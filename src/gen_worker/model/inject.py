"""Binding declared families onto a handler's parameters.

``@endpoint(families={...})`` says which family types a handler's parameters
have; this is where a set of RESOLVED instances becomes the keyword arguments
that call it. The check it performs is small and load-bearing: an instance must
be of the declared type, so a mix-up between two families — or between two
checkpoints of one family whose parameters got crossed — is a refusal here
rather than a wrong render nobody can explain.

:func:`fake_models` is the same shape with the fake backing, and it is what
makes greenfield B8 real: any handler runs hubless and GPU-less, through its own
code path, in CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ModelError, ModelRefusal
from .runtime import Model


def declared_models(target: object) -> dict[str, type[Model]]:
    """The families a decorated endpoint or JOB binds.

    Both decorators, because `@endpoint`/`@job` portability is a requirement:
    one body promotes between them unchanged, so the helper that binds its
    parameters has to read either declaration.

    Reads the decorator's own record rather than re-inspecting signatures: the
    decorator already validated that the parameters exist and are annotated
    with exactly these types, and re-deriving it here would be a second answer
    that can disagree with the first.
    """

    from ..api.decorators import ATTR
    from ..api.jobs import JOB_ATTR

    for attribute in (ATTR, JOB_ATTR):
        decl = getattr(target, attribute, None)
        families = getattr(decl, "families", None) if decl is not None else None
        if families:
            # The decorator stores a `Bind` per parameter (pgw#1346 K3); what a
            # caller binding kwargs wants is the CLASS, so the endpoint-coupled
            # axes are dropped here rather than leaked into every consumer.
            return {name: row.model for name, row in families.items()}
    return {}


def bind_models(
    families: Mapping[str, type[Model]],
    resolved: Mapping[str, Model],
) -> dict[str, Model]:
    """The handler kwargs for one call, checked against the declaration.

    Refuses a missing parameter and a type mismatch. It does NOT refuse an
    extra entry in ``resolved``: the worker may hold instances for several
    handlers of one class and hand the union in, and dropping the ones this
    handler does not name is the correct reading, not an error.
    """

    kwargs: dict[str, Model] = {}
    for name, declared in families.items():
        instance = resolved.get(name)
        if instance is None:
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                f"handler parameter {name!r} binds family {declared.FAMILY!r} and no "
                "resolved instance was supplied",
            )
        if not isinstance(instance, declared):
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                f"handler parameter {name!r} binds {declared.__name__} but the resolved "
                f"instance is a {type(instance).__name__}",
            )
        kwargs[name] = instance
    return kwargs


def fake_models(
    families: Mapping[str, type[Model]], *, seed: int = 0
) -> dict[str, Model]:
    """One FAKE instance per declared family parameter (greenfield B8).

    Every parameter gets its OWN instance even when two share a family type,
    because that is what the real path does and a test that shared one would
    not be exercising the two-checkpoint shape it looks like it is exercising.
    The per-parameter name is folded into the seed so two parameters of one
    family return different tensors — a handler that swapped them is then
    visible in the output rather than silently identical.
    """

    return {
        name: declared.fake(seed=seed + _offset(name))
        for name, declared in families.items()
    }


def fake_kwargs(target: object, *, seed: int = 0) -> dict[str, Model]:
    """:func:`fake_models` for a decorated endpoint, by declaration."""

    return fake_models(declared_models(target), seed=seed)


def _offset(name: str) -> int:
    # A small, stable, dependency-free spread over parameter names. It only has
    # to differ per name within one call; it is not identity and nothing keys
    # on it.
    return sum(index * ord(character) for index, character in enumerate(name, start=1))


def resolver_instances(
    families: Mapping[str, type[Model]], refs: Mapping[str, str]
) -> dict[str, Any]:
    """Resolve each declared parameter through the installed worker resolver.

    The worker's own path: ``refs`` names the checkpoint bound to each
    parameter (from the dispatch order), and each family's own
    ``instance(ref)`` reaches weights through the resolver the worker installed
    at boot. A parameter with no ref is a refusal, not a default — a family
    bound to "whatever was resident" is exactly the cross-request bleed the
    axis split exists to prevent.
    """

    out: dict[str, Any] = {}
    for name, declared in families.items():
        ref = str(refs.get(name, "") or "").strip()
        if not ref:
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                f"handler parameter {name!r} binds family {declared.FAMILY!r} and the "
                "request names no checkpoint for it",
            )
        out[name] = declared.instance(ref)
    return out


__all__ = [
    "bind_models",
    "declared_models",
    "fake_models",
    "fake_kwargs",
    "resolver_instances",
]
