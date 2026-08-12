"""The shape of the SDK's OWN declaration surface (pgw#942).

Nothing here concerns a served graph: this is the SDK stating which fields
its declaration types do and do not have. It was ``api.shape_contract`` until
pgw#1060, which took the name back — ``shape_contract`` is a cell-metadata
block about a compiled graph, and one name for two unrelated things is what
the naming rulings exist to stop.

A field that the SDK DELETED is a fact only the SDK knows. Before this
module, 25 of 28 endpoint packages asserted that knowledge for it — 91
``assert not hasattr(...)`` lines plus 46 ``for deleted in (...)`` loops,
each naming fields the pinned SDK cannot ship. That is one declaration shape
written 25 times by the parties least able to maintain it: one rename here
broke 25 downstream suites at once.

The table below is the single copy. :func:`assert_sdk_shape` checks the SDK
against itself; :func:`assert_declaration_shape` checks one endpoint's
concrete declaration, so an endpoint asserts only what *it* declares.

Adding a row is the deletion's completion, not paperwork: it is what makes
the deletion enforced everywhere at once, and it is the only place a reader
can find out when and why a field went away.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import msgspec

from .decorators import Compile, EndpointDecl, Resources, WorkerFunctionDecl
from .slot import Slot


class DeletedField(msgspec.Struct, frozen=True, kw_only=True):
    """One field the SDK removed, and the version that removed it."""

    owner: type
    name: str
    deleted_at: str
    reason: str

    def __str__(self) -> str:
        return f"{self.owner.__name__}.{self.name}"


#: Every field deleted from the declaration surface, newest cut last. A row
#: is removed ONLY if the field comes back (``compute_capability`` did, at
#: 0.75.0 / pgw#660 — which is exactly why it is absent here and why an
#: endpoint must never keep its own copy of this list).
DELETED_FIELDS: Tuple[DeletedField, ...] = (
    DeletedField(
        owner=Resources,
        name="vram_gb",
        deleted_at="0.60.0",
        reason=(
            "th#683 prove-and-profile MEASURES how much VRAM a function WANTS per "
            "lane and shape, so the gate-shaped v1 spelling does not come back. It "
            "split in two: `vram_gb_hint` is the optional first-build hint and is "
            "NEVER a gate, and `min_vram_gb` (pgw#660) is the hard floor the "
            "scheduler filters offers on. Declaring the hint when you meant the "
            "floor is the v2 regression this row now exists to prevent."
        ),
    ),
    DeletedField(
        owner=Resources,
        name="ram_gb",
        deleted_at="0.60.0",
        reason=(
            "same cut as `vram_gb`. Restored as the OPTIONAL allocation-time hint "
            "`ram_gb_hint` (pgw#670) after ie#492 sized a real host-RAM floor; the "
            "gate-shaped spelling did not come back."
        ),
    ),
    DeletedField(
        owner=EndpointDecl,
        name="regimes",
        deleted_at="0.60.0",
        reason=(
            "SDK v2 (pgw#647): the endpoint declares one envelope, not a set "
            "of named operating regimes the hub had to re-derive."
        ),
    ),
    DeletedField(
        owner=EndpointDecl,
        name="route",
        deleted_at="0.60.0",
        reason=(
            "SDK v2: routing is the hub's decision from bindings and declared "
            "compile axes; endpoint code never names a route."
        ),
    ),
    DeletedField(
        owner=Compile,
        name="guidance_scales",
        deleted_at="0.60.0",
        reason=(
            "SDK v2: guidance is a payload axis (`Compile(args=)` / the derived "
            "warm plan), not a compile-time enumeration on the spec."
        ),
    ),
    DeletedField(
        owner=Compile,
        name="lora_bucket",
        deleted_at="0.60.0",
        reason=(
            "gw#561: the resident traced rank bucket shapes the graph family whether "
            "or not `compile=` is present, so it moved to `@endpoint(lora_bucket=)` "
            "— i.e. `EndpointDecl.lora_bucket`. Two carriers for one fact was the bug."
        ),
    ),
    DeletedField(
        owner=EndpointDecl,
        name="weight_set",
        deleted_at="0.92.0",
        reason=(
            "th#1559 §4.22 (Paul): NFS volumes follow `@endpoint(kind=)` — inference gets "
            "one, no other kind does. `weight_set` asked the author to state a fact that "
            "was not theirs: an operator flipping a slot to `open` post-deploy invalidates "
            "the compiled-in value. Declared by 0 of 1,129 function rows, and the defaulted "
            "`getattr(decl, 'weight_set', None)` read is what kept that invisible."
        ),
    ),
    DeletedField(
        owner=Slot,
        name="default_config",
        deleted_at="0.60.0",
        reason=(
            "th#1116: the CATALOG owns per-model config; a code-side default was a "
            "second answer that could disagree with the live one."
        ),
    ),
    DeletedField(
        owner=Slot,
        name="share_components",
        deleted_at="0.60.0",
        reason=(
            "SDK v2: the component tree is DERIVED from `pipeline_cls`, and sharing "
            "is a deploy-binding decision (th#980), never an endpoint declaration."
        ),
    ),
)


class DeclarationShapeError(AssertionError):
    """A declaration carries a field the SDK deleted."""


def _rows_for(owner: type) -> Tuple[DeletedField, ...]:
    return tuple(row for row in DELETED_FIELDS if row.owner is owner)


def _check_instance(obj: Any, owner: type, where: str, failures: list[str]) -> None:
    for row in _rows_for(owner):
        if hasattr(obj, row.name):
            failures.append(
                f"{where}.{row.name} exists — deleted at gen-worker {row.deleted_at}: {row.reason}"
            )


def assert_sdk_shape() -> None:
    """Assert the SDK itself ships none of :data:`DELETED_FIELDS`.

    The SDK's own suite calls this; an endpoint never needs to. It is the
    assertion the 91 endpoint-side fossils were each approximating.
    """
    failures: list[str] = []
    for row in DELETED_FIELDS:
        owner = row.owner
        fields = getattr(owner, "__struct_fields__", None)
        names: Iterable[str]
        if fields is not None:
            names = fields
        else:
            names = getattr(owner, "__slots__", ()) or ()
        if row.name in tuple(names) or hasattr(owner, row.name):
            failures.append(
                f"{row} is back on the class but still listed as deleted "
                f"(at {row.deleted_at}) — remove the DELETED_FIELDS row or the field"
            )
    if failures:
        raise DeclarationShapeError("; ".join(failures))


def assert_declaration_shape(
    decl: Any,
    *,
    slots: Optional[Sequence[str]] = None,
) -> None:
    """Assert one endpoint's declaration carries no deleted field.

    ``decl`` is the object ``@endpoint`` attaches
    (``MyClass.__gen_worker_endpoint__``) — an :class:`EndpointDecl` or a
    :class:`WorkerFunctionDecl`. Its ``resources``, ``compile`` and every
    ``slots`` entry are checked with it, so::

        from gen_worker.testing import assert_declaration_shape

        assert_declaration_shape(m.SDXLFamily.__gen_worker_endpoint__)

    replaces every ``assert not hasattr(decl.resources, "vram_gb")`` in the
    suite. ``slots=`` narrows the walk to named slots; by default every
    declared slot is checked.

    The endpoint keeps asserting what it POSITIVELY declares (``decl.resources.gpu
    is True``, ``set(decl.slots) == {"pipeline"}``) — that is its own contract
    and belongs in its own suite. What it stops re-deriving is the SDK's.
    """
    failures: list[str] = []
    _check_instance(decl, EndpointDecl, "decl", failures)
    _check_instance(decl, WorkerFunctionDecl, "decl", failures)

    resources = getattr(decl, "resources", None)
    if resources is not None:
        _check_instance(resources, Resources, "decl.resources", failures)

    compile_spec = getattr(decl, "compile", None)
    if compile_spec is not None:
        _check_instance(compile_spec, Compile, "decl.compile", failures)

    declared: Mapping[str, Any] = getattr(decl, "slots", {}) or {}
    names = tuple(slots) if slots is not None else tuple(declared)
    for name in names:
        slot = declared.get(name)
        if slot is None:
            failures.append(f"decl.slots[{name!r}] is not declared")
            continue
        _check_instance(slot, Slot, f"decl.slots[{name!r}]", failures)

    if failures:
        raise DeclarationShapeError("; ".join(failures))


__all__ = [
    "DELETED_FIELDS",
    "DeclarationShapeError",
    "DeletedField",
    "assert_declaration_shape",
    "assert_sdk_shape",
]
