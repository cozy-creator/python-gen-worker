"""Turning what the worker already loaded into a `Model` instance (pgw#1346).

This is the half the typed SDK was missing. pgw#1332 landed the declaration,
the codegen, the backings and the injection helpers; what nothing did was
CONSTRUCT an instance on a real pod, so `@endpoint(models={"flux": Flux1Dev})`
type-checked, published a manifest and then handed the handler nothing.

The gap was never the plumbing. It was one fact: a declaration's
``Runner.build`` makes a WEIGHTLESS module from config — that is what the mint
traces — so it cannot answer a request. Serving eagerly means reaching the
weight-bearing module the loader already produced, and until ``Runner.component``
existed nothing said which module that was. With it, the whole path is a lookup:

    loaded pipeline ──Runner.component──▶ {runner: module} ──▶ EagerBacking
    armed compiled graphs ─────────────────────────────────────────────▶ CompiledBacking
                                    │
                                    ▼
                          Model.adopt(...) ──▶ handler kwarg

Both backings are folded into one `DualBacking`, so a runner armed at this
request's bucket runs compiled and every other call runs eager, with no branch
in the handler and no second code path to keep honest.

The instance is built PER REQUEST, deliberately. A `Model` value carries the
checkpoint ref and that checkpoint's tuned values, and those are dispatch facts
— two requests on one warm pod can name two checkpoints. The expensive half
(the weights) is shared: this only wraps modules residency already holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from .backing import Backing
from .errors import ModelError, ModelRefusal
from .runtime import Model
from .spec import ModelSpec

M = TypeVar("M", bound=Model)


def _reach(tree: Any, path: str) -> Any:
    """Walk a dotted attribute path, answering ``None`` at the first gap.

    A missing component is not an error HERE: an optional lane, a pipeline that
    genuinely lacks a refiner, or a runner with no eager equivalent all reach
    this and all mean the same thing — no eager module for that runner. The
    refusal, when it matters, comes from the backing at call time and names the
    runner, which is the message an author can act on.
    """

    current = tree
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def eager_modules(spec: ModelSpec | None, tree: Any) -> dict[str, Any]:
    """The ``{runner: module}`` map an :class:`EagerBacking` is built from.

    ``spec`` is ``None`` on an adopt-only serve pod, where importing the
    declaration would acquire a model library the role forbids (pgw#1328). That
    is not a failure: such a pod serves compiled graphs, so an empty eager map is
    the correct answer and the compiled backing carries the request.
    """

    if spec is None or tree is None:
        return {}
    out: dict[str, Any] = {}
    for runner in spec.runners:
        if not runner.component:
            continue
        module = _reach(tree, runner.component)
        if module is not None:
            out[runner.name] = module
    return out


def instance_for(
    model: type[M],
    *,
    ref: str,
    tree: Any = None,
    compiled: Backing | None = None,
    stamped: str = "",
    lora_stamped: Sequence[str] = (),
    label: str = "",
) -> M:
    """One resolved instance, from what this pod already has.

    Refuses when NEITHER backing has anything, rather than handing back an
    instance whose every call fails later with no context. A model bound to a
    checkpoint the pod could not load is a dispatch-time fact and says so here,
    naming the model and the ref.
    """

    from .tuned import tuned_from_catalog

    eager = eager_modules(model.SPEC, tree)
    if not eager and compiled is None:
        raise ModelError(
            ModelRefusal.BACKING_MISSING,
            f"model {model.FAMILY!r} bound to {ref!r} has neither an eager module "
            f"nor an armed compiled graph on this pod. Either the checkpoint did not load, or "
            f"the declaration's runners name components this tree does not have "
            f"(declare `Runner(..., component=...)` for each runner that has an "
            f"eager equivalent).",
        )
    return model.adopt(
        ref=ref,
        tuned=tuned_from_catalog(model, stamped, lora_stamped),
        eager=eager or None,
        compiled=compiled,
        label=label or ref,
    )


def instances_for(
    binds: Mapping[str, Any],
    *,
    refs: Mapping[str, str],
    trees: Mapping[str, Any],
    compiled: Mapping[str, Backing] | None = None,
    stamped: Mapping[str, str] | None = None,
    lora_stamped: Mapping[str, Sequence[str]] | None = None,
    skip_unresolved: bool = False,
) -> dict[str, Model]:
    """One instance per declared handler parameter.

    ``binds`` is the decorator's own record (parameter -> :class:`Bind`), so the
    set of parameters filled here is exactly the set the decorator validated
    against the handler's annotations. A parameter with no resolved ref is a
    refusal and never a default: a model bound to "whatever was resident" is the
    cross-request bleed the axis split exists to prevent.

    ``skip_unresolved`` is for WARM-UP and nothing else. A warm pass runs before
    any dispatch has named a checkpoint, so "no ref yet" is the expected state
    there rather than a defect; skipping leaves the model un-warmed, which is
    honest, where refusing would break the boot of every endpoint whose models
    are hub-resolved. The serving path never passes it.
    """

    armed = compiled or {}
    values = stamped or {}
    overlays = lora_stamped or {}
    out: dict[str, Model] = {}
    for name, bind in binds.items():
        ref = str(refs.get(name, "") or "").strip()
        if not ref and skip_unresolved:
            continue
        if not ref:
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                f"handler parameter {name!r} binds model "
                f"{bind.model.FAMILY!r} and this request names no checkpoint for it",
            )
        out[name] = instance_for(
            bind.model,
            ref=ref,
            tree=trees.get(name),
            compiled=armed.get(name),
            stamped=values.get(name, ""),
            lora_stamped=overlays.get(name, ()),
            label=name,
        )
    return out


__all__ = ["eager_modules", "instance_for", "instances_for"]
