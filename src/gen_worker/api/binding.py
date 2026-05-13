"""Binding model for `@inference_function(models=...)` — `Repo` + `Dispatch`.

Replaces the old `Annotated[T, ModelRef(...)]` injection pattern. See
`progress.json` issue #9 (decorator-table-model-bindings) for the spec.

A binding is one of two **picks**:

- **Fixed pick** (`Repo`): function pins one specific `(repo, flavor?, tag?)`.
  Tag defaults to ``"prod"``. Flavor is optional. Resolves to a concrete
  checkpoint at deploy time.
- **Dispatch pick** (`Dispatch`): function pins a *set* of `(repo, flavor?,
  tag?)` picks keyed by a single discriminator field in the payload (a
  ``Literal[...]``-typed field). At invoke time the discriminator selects
  which pick is used.

Each pick supports an optional ``.allow_override(*classes)`` chainable
method. When called with one or more pipeline class arguments, the invoker
may substitute the default pick with an arbitrary ref of their choice —
subject to the constraint that the supplied ref's ``pipeline_class`` must
be in the explicit allowlist.

All modifier methods return new immutable instances; chain order is
commutative. Bare zero-arg ``.allow_override()`` raises ``ValueError`` at
decoration time — the framework does NOT auto-derive the constraint from
the function's param annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Union


def _qualname(t: Any) -> str:
    """Return a class's fully-qualified name (`module.qualname`)."""
    mod = getattr(t, "__module__", "") or ""
    qn = getattr(t, "__qualname__", None) or getattr(t, "__name__", "") or ""
    if mod and qn:
        return f"{mod}.{qn}"
    return repr(t)


def _normalize_classes(classes: tuple[Any, ...]) -> tuple[str, ...]:
    """Normalize ``*classes`` (class objects or string FQNs) to a unique
    tuple of FQN strings, preserving order.

    A zero-arg call raises ``ValueError``. The framework requires the tenant
    to explicitly enumerate which pipeline classes the invoker may substitute.
    """
    if not classes:
        raise ValueError(
            "allow_override() requires at least one pipeline class argument. "
            "Pass class objects (preferred) or string FQNs, e.g. "
            "allow_override(MyPipelineClass) or allow_override('pkg.mod.MyPipelineClass')."
        )
    out: list[str] = []
    seen: set[str] = set()
    for c in classes:
        if isinstance(c, str):
            name = c.strip()
            if not name:
                raise ValueError("allow_override() string FQN must be non-empty")
        elif isinstance(c, type):
            name = _qualname(c)
        else:
            raise TypeError(
                f"allow_override() arguments must be class objects or string FQNs; "
                f"got {type(c).__name__}: {c!r}"
            )
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


@dataclass(frozen=True)
class Repo:
    """Repository handle — also usable as a binding with defaults.

    Module-level convention::

        flux = Repo("black-forest-labs/flux.2-klein-4b-turbo")

    A bare ``Repo("owner/repo")`` already serves as a valid fixed binding
    (tag defaults to ``"prod"``, no flavor, no override). Refine with the
    chainable modifiers::

        flux.flavor("nf4")
        flux.tag("canary").flavor("bf16")
        flux.flavor("nf4").allow_override(MyPipelineClass)

    All modifier methods return new immutable instances; chain order is
    commutative.
    """

    ref: str
    _flavor: str = ""
    _tag: str = "prod"
    _allow_override: bool = False
    _pipeline_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ValueError(f"Repo(ref=) must be a non-empty string, got {self.ref!r}")
        # Normalize ref (lowercase + trim). Keep validation light here; the
        # heavier ref grammar check lives in the manifest validator.
        object.__setattr__(self, "ref", self.ref.strip())

    def flavor(self, name: str) -> "Repo":
        """Return a new Repo bound to the given flavor."""
        name = str(name or "").strip()
        if not name:
            raise ValueError("Repo.flavor() requires a non-empty flavor name")
        return replace(self, _flavor=name)

    def tag(self, name: str) -> "Repo":
        """Return a new Repo bound to the given tag.

        Tag defaults to ``"prod"``; override only for non-prod rollouts.
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("Repo.tag() requires a non-empty tag name")
        return replace(self, _tag=name)

    def allow_override(self, *classes: Any) -> "Repo":
        """Return a new Repo whose default may be overridden at invoke time.

        ``*classes`` is the explicit allowlist of acceptable pipeline classes
        for caller-supplied overrides. Accepts class objects (preferred —
        autocomplete + import-time check) or string FQNs (escape hatch).
        A bare zero-arg call raises ``ValueError``.
        """
        fqns = _normalize_classes(classes)
        return replace(self, _allow_override=True, _pipeline_classes=fqns)


@dataclass(frozen=True)
class Dispatch:
    """Payload-driven dispatch pick.

    Construct via the free :func:`dispatch` function::

        dispatch(field="variant", table={
            "nf4": flux.flavor("nf4"),
            "int8": flux.flavor("int8"),
        })

    The discriminator ``field`` must be a ``Literal[...]``-typed member of
    the function's payload struct; every ``table`` key must be one of the
    Literal's members. Validated at decoration time by
    :func:`inference_function`.
    """

    field: str
    table: Mapping[str, "Repo"]
    _allow_override: bool = False
    _pipeline_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        f = str(self.field or "").strip()
        if not f:
            raise ValueError("dispatch() requires a non-empty discriminator field name")
        object.__setattr__(self, "field", f)

        if not self.table:
            raise ValueError("dispatch() requires a non-empty table")
        # Freeze the table to a plain dict (msgspec.to_builtins safety).
        frozen_table: dict[str, Repo] = {}
        for k, v in self.table.items():
            key = str(k or "").strip()
            if not key:
                raise ValueError(f"dispatch() table has empty key: {k!r}")
            if not isinstance(v, Repo):
                raise TypeError(
                    f"dispatch() table values must be Repo instances; "
                    f"got {type(v).__name__} for key {key!r}"
                )
            frozen_table[key] = v
        object.__setattr__(self, "table", frozen_table)

    def allow_override(self, *classes: Any) -> "Dispatch":
        """Return a new Dispatch whose default may be overridden at invoke time.

        Same semantics as :meth:`Repo.allow_override`.
        """
        fqns = _normalize_classes(classes)
        return replace(self, _allow_override=True, _pipeline_classes=fqns)


def dispatch(field: str, table: Mapping[str, Repo]) -> Dispatch:
    """Construct a :class:`Dispatch` binding.

    Args:
        field: The discriminator payload field name. Must reference a
            ``Literal[...]``-typed field on the function's payload struct.
        table: Mapping from Literal member values → :class:`Repo` picks.
            Every Literal value must appear as a key. Validated at
            decoration time.
    """
    return Dispatch(field=field, table=dict(table))


Binding = Union[Repo, Dispatch]


__all__ = [
    "Binding",
    "Dispatch",
    "Repo",
    "dispatch",
]
