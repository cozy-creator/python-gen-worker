"""The typed Model SDK — the ONE authoring contract for a model (pgw#1332).

A model is described, declared, bound and called exactly one way. There is no
second: the runtime never accepts a bare pipeline or module, so a catalog model
and an author-defined inline one are syntactically identical at the call site
and catalog promotion is a copy rather than a rewrite.

```python
from gen_worker import RequestContext, endpoint
from gen_worker.model.catalog import Flux1Dev

@endpoint(families={"flux": Flux1Dev})
class Generate:
    def generate(self, ctx: RequestContext, p: In, flux: Flux1Dev) -> Out:
        steps = p.steps if p.steps is not None else flux.tuned.steps
        latents = flux.denoiser(resolution=1024, hidden_states=..., timestep=...)
```

Three axes, and they do not cross:

* **class** — buckets, graph classes, signatures, the loop, the scheduler.
  Checkpoint-free by construction (DESIGN-RULINGS §4.27), which is what lets
  one compiled cell serve sixteen fine-tunes.
* **checkpoint** — the weights binding, ``inst.tuned``, the ref label. This is
  the INSTANCE, and it is what a handler parameter receives.
* **request** — cancellation, logging, progress, seed, assets. This is ``ctx``,
  and it carries nothing about a model.

The pieces, in the order they run:

``spec``       what an author declares (``ModelSpec``, ``GraphModelSpec``, ``Runner``)
``export``     the declaration-time fake-tensor export — no GPU, no weights
``snapshot``   ``family_export_v1``, the versioned document that export writes
``codegen``    the snapshot becomes a typed module; no string-keyed lookups
``runtime``    ``Model`` — the instance a handler receives — sessions, tuned
``backing``    eager / compiled / fake, under one signature
``inject``     declared models become a handler call's keyword arguments
``tuned``      catalog-stamped values decode onto the instance, not onto ctx
``drift``      a mint-emitted ``recipe_v1`` still describes the declaration

`gen_worker.model.catalog` is SDK surface and is never imported by the worker
runtime: importing a declaration imports model code, and an adopt-only serve
role must not acquire diffusers by accident (pgw#1328).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - the eager spelling, for type checkers only
    from .backing import (
        Backing,
        BackingKind,
        CompiledBacking,
        DualBacking,
        EagerBacking,
        FakeBacking,
    )
    from .bind import Bind, as_bind
    from .drift import assert_recipe, assert_reference
    from .errors import ModelError, ModelRefusal
    from .inject import (
        bind_models,
        declared_models as bound_models,
        fake_models,
        fake_kwargs,
        resolver_instances,
    )
    from .residency import eager_modules, instance_for, instances_for
    from .runtime import (
        DecodeSession,
        Model,
        InstanceResolver,
        Tuned,
        instance_resolver,
        resolve,
        resolve_tuned,
        set_instance_resolver,
        tuned_fields,
        tuned_payload_fields,
    )
    from .snapshot import EXPORT_VERSION, ModelExport
    from .spec import (
        Bucket,
        BucketMap,
        CallExample,
        DEFAULT_LAYOUT,
        ModelSpec,
        GraphModelSpec,
        Loop,
        LoopKind,
        Parameter,
        Runner,
        Scheduler,
        SessionState,
        Stage,
        TunedValues,
        declared_model_specs,
    )
    from .tuned import tuned_from_catalog

#: Exported name -> (submodule, name in it). THE package index, and the reason
#: it exists instead of a block of eager imports (pgw#1331).
#:
#: ``inject`` binds declared families onto a handler's call, so it imports
#: ``gen_worker.api.decorators`` — and through it the endpoint registry, the
#: warm-up, and ``models.provision``. Eagerly re-exporting it here meant that
#: importing ``gen_worker.model.runtime`` — which a generated binding does, on
#: an adopt-only pod that will never register an endpoint — executed the whole
#: eager-capable worker's import graph, including every module that names a
#: model library inside a function. PEP 562 breaks that: importing this package
#: costs nothing, and asking for a name costs exactly that one submodule.
#:
#: ``scripts/lint_serve_role_closure.py`` is what holds it: with the eager block
#: back, the serve-role closure reaches ``diffusers`` through ten modules and
#: the fence says so by name.
_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "Backing": ("backing", "Backing"),
    "BackingKind": ("backing", "BackingKind"),
    "Bucket": ("spec", "Bucket"),
    "BucketMap": ("spec", "BucketMap"),
    "CallExample": ("spec", "CallExample"),
    "CompiledBacking": ("backing", "CompiledBacking"),
    "DEFAULT_LAYOUT": ("spec", "DEFAULT_LAYOUT"),
    "DecodeSession": ("runtime", "DecodeSession"),
    "DualBacking": ("backing", "DualBacking"),
    "EXPORT_VERSION": ("snapshot", "EXPORT_VERSION"),
    "EagerBacking": ("backing", "EagerBacking"),
    "FakeBacking": ("backing", "FakeBacking"),
    "ModelSpec": ("spec", "ModelSpec"),
    "Model": ("runtime", "Model"),
    "ModelError": ("errors", "ModelError"),
    "ModelExport": ("snapshot", "ModelExport"),
    "ModelRefusal": ("errors", "ModelRefusal"),
    "GraphModelSpec": ("spec", "GraphModelSpec"),
    "InstanceResolver": ("runtime", "InstanceResolver"),
    "Loop": ("spec", "Loop"),
    "LoopKind": ("spec", "LoopKind"),
    "Parameter": ("spec", "Parameter"),
    "Runner": ("spec", "Runner"),
    "Scheduler": ("spec", "Scheduler"),
    "SessionState": ("spec", "SessionState"),
    "Stage": ("spec", "Stage"),
    "Tuned": ("runtime", "Tuned"),
    "TunedValues": ("spec", "TunedValues"),
    "Bind": ("bind", "Bind"),
    "as_bind": ("bind", "as_bind"),
    "assert_recipe": ("drift", "assert_recipe"),
    "assert_reference": ("drift", "assert_reference"),
    "bind_models": ("inject", "bind_models"),
    "bound_models": ("inject", "declared_models"),
    "declared_model_specs": ("spec", "declared_model_specs"),
    "fake_models": ("inject", "fake_models"),
    "fake_kwargs": ("inject", "fake_kwargs"),
    "instance_resolver": ("runtime", "instance_resolver"),
    "eager_modules": ("residency", "eager_modules"),
    "instance_for": ("residency", "instance_for"),
    "instances_for": ("residency", "instances_for"),
    "resolve": ("runtime", "resolve"),
    "resolve_tuned": ("runtime", "resolve_tuned"),
    "resolver_instances": ("inject", "resolver_instances"),
    "set_instance_resolver": ("runtime", "set_instance_resolver"),
    "tuned_fields": ("runtime", "tuned_fields"),
    "tuned_from_catalog": ("tuned", "tuned_from_catalog"),
    "tuned_payload_fields": ("runtime", "tuned_payload_fields"),
}


def __getattr__(name: str) -> Any:
    row = _EXPORTS.get(name)
    if row is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, origin = row
    return getattr(import_module(f"{__name__}.{module}"), origin)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


#: **Two things are spelled ``Tuned``, and both spellings are deliberate.**
#: ``Flux1Dev.Tuned`` is the family's tuned-value SCHEMA — a class, declared on
#: the family, which is how the naming collision with the old free-standing
#: ``@family(...)`` decorator was resolved. ``Tuned[int]`` is a payload-field
#: ANNOTATION saying "fall back to the checkpoint's tuned value when the caller
#: omits this". They live at different addresses (a class attribute vs a
#: module-level alias) and they are two halves of one idea: the family declares
#: the shape, the payload says which fields defer to it.
#:
#: There is no second name for an INSTANCE (pgw#1346, Paul). Python's own
#: class/instance distinction carries it: ``Flux1Dev`` is the type a handler
#: parameter is annotated with, and the value it receives is an instance of
#: that type. The alias said in a name what the language already says in its
#: semantics, and two names for one object is two things a reader has to
#: reconcile. The dead spelling was ``FamilyInstance``.  # retired-spelling: a
#: note about a deleted alias is useless if it cannot name the alias
__all__ = [
    "DEFAULT_LAYOUT",
    "EXPORT_VERSION",
    "Backing",
    "BackingKind",
    "Bind",
    "Bucket",
    "BucketMap",
    "CallExample",
    "CompiledBacking",
    "DecodeSession",
    "DualBacking",
    "EagerBacking",
    "FakeBacking",
    "ModelSpec",
    "Model",
    "ModelError",
    "ModelExport",
    "ModelRefusal",
    "GraphModelSpec",
    "InstanceResolver",
    "Loop",
    "LoopKind",
    "Parameter",
    "Runner",
    "Scheduler",
    "SessionState",
    "Stage",
    "Tuned",
    "TunedValues",
    "as_bind",
    "assert_recipe",
    "assert_reference",
    "bind_models",
    "bound_models",
    "declared_model_specs",
    "eager_modules",
    "fake_models",
    "fake_kwargs",
    "instance_for",
    "instance_resolver",
    "instances_for",
    "resolve",
    "resolve_tuned",
    "resolver_instances",
    "set_instance_resolver",
    "tuned_fields",
    "tuned_from_catalog",
    "tuned_payload_fields",
]
