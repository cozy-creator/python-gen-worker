"""Can THIS process build the subject the parent named? Asked before any load.

pgw#1215 (th#1834 Phase 3). Every one of these checks lived in ``mint_child``,
and two other children — ``boot_trace_child`` and ``measure_child`` — already
reached into it for the same five symbols. They did so through a FUNCTION-LOCAL
import, at three call sites, for one reason: ``mint_child`` pulls the whole mint
in behind it, so importing it at module scope costs a child that only wants to
check its slots the entire compile brain.

That is the shape §1.17 forbids (imports belong at the top of the file), and it
was not fixable while the checks lived beside the driver. Here they have no
heavy dependency at all, so the three runtime imports become ordinary top-of-
file ones and the module that owns the mint stops being a library for children
that do not mint.

The refusals are DETERMINISTIC by construction — the endpoint's declaration,
the parent's bindings and the materialized trees are fixed for the life of one
request — which is why they are a distinct type from anything the compile can
raise: a caller may re-run a resource shortfall and must never re-run one of
these.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .child_contract import MintSlot


class PreflightRefused(RuntimeError):
    """A named, deterministic reason this process cannot build the subject.

    Never retried by the caller: re-running a named refusal buys a second
    billed compile for the same sentence.

    ``mint_phases`` carries a refusing mint's PARTIAL phase table when it got
    far enough to have one (pgw#825) — the entries it exported and compiled
    before refusing are real minutes and must reach the hub.

    ⚠️ Deliberately NOT named ``MintRefused``: that name is live in
    ``aot_inputs`` for "every declared graph class refused", and one name for
    both would make "this subject cannot be built here" and "nothing compiled"
    indistinguishable at the exit-code boundary, with no test going red
    (th#1834's census names this exact trap).
    """

    def __init__(
        self, *args: Any, mint_phases: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(*args)
        self.mint_phases: Dict[str, Any] = dict(mint_phases or {})


def select_specs(
    specs: Sequence[Any], function: str,
) -> Tuple[Any, List[Any]]:
    """The requested function's spec plus every sibling sharing its instance.

    Siblings matter: the warm plan is CLASS-scoped (pgw#654 — one compiled
    graph per graph class, so turbo and base seed the same capture). Minting
    from one function's view is how a sibling lane ends up with no compiled
    graph at all and every adopting boot fails its per-object proof (gw#612).
    """
    chosen = next((s for s in specs if s.name == function), None)
    if chosen is None:
        raise PreflightRefused(
            f"function {function!r} is not in this image "
            f"(have: {sorted(s.name for s in specs)})")
    if chosen.cls is None:
        raise PreflightRefused(
            f"function {function!r} is function-shaped — it has no instance "
            "to compile")
    siblings = [
        s for s in specs
        if s.cls is chosen.cls and s.instance_key == chosen.instance_key
    ]
    if chosen not in siblings:
        siblings.append(chosen)
    return chosen, siblings


def bind_slots(specs: Sequence[Any], resolved: Mapping[str, MintSlot]) -> None:
    """Install the PARENT's resolved slot bindings onto rediscovered specs.

    The child re-runs discovery in a fresh interpreter, so ``spec.models``
    comes back holding only what ``@endpoint`` declared. For a hub-catalog
    slot — ``Slot(cls, selected_by="model")`` with no ``default_checkpoint=``,
    which is sdxl's shape and every multi-checkpoint family's — the decorator
    declares no ref at all, and the resolution chain then has nothing to
    resolve: ``ctx.slots["pipeline"]`` raises before a graph is traced.

    Applied to the WHOLE sibling set, never one spec: ``instance_key`` is a
    live property over ``spec.models``, so binding the chosen function alone
    would move its key out from under its siblings and silently narrow the
    class-scoped warm plan (pgw#654) to one lane.
    """
    for spec in specs:
        for slot, res in resolved.items():
            if slot in spec.slots or slot in spec.models:
                spec.models[slot] = res.ref


def assert_slots_resolvable(
    specs: Sequence[Any],
    slots: Mapping[str, MintSlot],
    *,
    what: str = "",
) -> None:
    """pgw#969: refuse a request whose slots cannot resolve — before the load.

    ``slots`` + ``what`` rather than the whole request (pgw#1089): the
    boot-trace child resolves the same slots for the same reason and must get
    the same refusal, and a second copy of this check would be a second answer
    to "can this process trace the checkpoint the parent serves".

    Two distinct failures, both named rather than deferred to a handler's
    first dereference nine seconds later:

    * the parent sent no binding for a declared, non-optional slot (the wire
      gap this issue exists for), and
    * a slot the parent DID bind still fails the resolution chain here, which
      is real divergence between the two processes and never a normal path.

    An OPTIONAL slot the parent did not bind is deliberately silent: the
    parent's own warm context resolves it exactly this way (the deploy chose
    not to serve that lane), and refusing here would refuse a mint the parent
    can serve.
    """
    from . import warmup as warmup_mod

    bound = set(slots)
    problems: List[str] = []
    for spec in specs:
        missing = sorted(
            name for name, slot in spec.slots.items()
            if name not in bound
            and name not in spec.models
            and not getattr(slot, "optional", False)
        )
        for name in missing:
            problems.append(
                f"slot {name!r} of {spec.name!r}: the parent sent no resolved "
                f"binding for it and the endpoint declares no "
                f"default_checkpoint, so this child has no checkpoint to "
                f"trace (MintRequest.slots carries "
                f"{sorted(bound) or '(nothing)'})")
        errors = warmup_mod.resolved_slots_kwargs(spec, None)["slot_errors"]
        for name, why in sorted(errors.items()):
            resolved = slots.get(name)
            if resolved is not None:
                problems.append(
                    f"slot {name!r} of {spec.name!r}: the parent's binding "
                    f"{resolved.ref.path!r} "
                    f"crossed but does not resolve in this process — {why}")
    if not problems:
        return
    raise PreflightRefused(
        f"{what or 'slot resolution'}: cannot build the warmup request — "
        + "; ".join(problems)
        + ". A mint cannot trace a graph for a pipeline it was never told "
          "to load.")


def pick_compile_target(loaded: Dict[str, Any], cfg: Any) -> Tuple[str, Any]:
    """The loaded slot that actually carries the declared compile target(s).

    Public since pgw#1089: the boot-trace child must pick the SAME slot the
    mint picks, and two pickers would be two compile targets.
    """
    from . import compile_cache as cc

    for slot, obj in loaded.items():
        try:
            if cc.has_compile_target(obj, cfg):
                return slot, obj
        except Exception:
            continue
    raise PreflightRefused(
        f"no compile target resolved on any loaded slot "
        f"({sorted(loaded) or '(none)'}) for family {cfg.family!r} "
        f"targets={list(cfg.targets)}")


__all__ = [
    "PreflightRefused",
    "assert_slots_resolvable",
    "bind_slots",
    "pick_compile_target",
    "select_specs",
]
