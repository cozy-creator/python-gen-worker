"""The typed Family SDK — the ONE authoring contract for a model (pgw#1332).

A family is how a model is described, declared, bound and called. There is no
second way: the runtime never accepts a bare pipeline or module, so a catalog
family and an author-defined inline family are syntactically identical at the
call site and catalog promotion is a copy rather than a rewrite.

```python
from gen_worker import RequestContext, endpoint
from gen_worker.family.catalog import Flux1Dev

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

``spec``       what an author declares (``Family``, ``GraphFamily``, ``Runner``)
``export``     the declaration-time fake-tensor export — no GPU, no weights
``snapshot``   ``family_export_v1``, the versioned document that export writes
``codegen``    the snapshot becomes a typed module; no string-keyed lookups
``runtime``    ``FamilyBinding``/``FamilyInstance``, sessions, tuned resolution
``backing``    eager / compiled / fake, under one signature
``inject``     declared families become a handler call's keyword arguments
``tuned``      catalog-stamped values decode onto the instance, not onto ctx
``drift``      a mint-emitted ``recipe_v1`` still describes the declaration

`gen_worker.family.catalog` is SDK surface and is never imported by the worker
runtime: importing a family declaration imports model code, and an adopt-only
serve role must not acquire diffusers by accident (pgw#1328).
"""

from __future__ import annotations

from .backing import (
    Backing,
    BackingKind,
    CompiledBacking,
    DualBacking,
    EagerBacking,
    FakeBacking,
)
from .drift import assert_recipe, assert_reference
from .errors import FamilyError, FamilyRefusal
from .inject import (
    bind_families,
    declared_families as bound_families,
    fake_families,
    fake_kwargs,
    resolver_instances,
)
from .runtime import (
    DecodeSession,
    FamilyBinding,
    InstanceResolver,
    Tuned,
    instance_resolver,
    resolve,
    resolve_tuned,
    set_instance_resolver,
    tuned_fields,
    tuned_payload_fields,
)
from .snapshot import EXPORT_VERSION, FamilyExport
from .tuned import tuned_from_catalog
from .spec import (
    DEFAULT_LAYOUT,
    Bucket,
    BucketMap,
    CallExample,
    Family,
    GraphFamily,
    Loop,
    LoopKind,
    Parameter,
    Runner,
    Scheduler,
    SessionState,
    Stage,
    TunedValues,
    declared_families,
)

#: **Two things are spelled ``Tuned``, and both spellings are deliberate.**
#: ``Flux1Dev.Tuned`` is the family's tuned-value SCHEMA — a class, declared on
#: the family, which is how the naming collision with the old free-standing
#: ``@family(...)`` decorator was resolved. ``Tuned[int]`` is a payload-field
#: ANNOTATION saying "fall back to the checkpoint's tuned value when the caller
#: omits this". They live at different addresses (a class attribute vs a
#: module-level alias) and they are two halves of one idea: the family declares
#: the shape, the payload says which fields defer to it.
#:
#: ``FamilyInstance`` is the READING of :class:`FamilyBinding` that matters at a
#: handler parameter: the class is the type, the value is the instance. One
#: object, two names, because both readings are load-bearing in the design and
#: a reader who arrives from either finds the thing they were told to look for.
FamilyInstance = FamilyBinding

__all__ = [
    "DEFAULT_LAYOUT",
    "EXPORT_VERSION",
    "Backing",
    "BackingKind",
    "Bucket",
    "BucketMap",
    "CallExample",
    "CompiledBacking",
    "DecodeSession",
    "DualBacking",
    "EagerBacking",
    "FakeBacking",
    "Family",
    "FamilyBinding",
    "FamilyError",
    "FamilyExport",
    "FamilyInstance",
    "FamilyRefusal",
    "GraphFamily",
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
    "assert_recipe",
    "assert_reference",
    "bind_families",
    "bound_families",
    "declared_families",
    "fake_families",
    "fake_kwargs",
    "instance_resolver",
    "resolve",
    "resolve_tuned",
    "resolver_instances",
    "set_instance_resolver",
    "tuned_fields",
    "tuned_from_catalog",
    "tuned_payload_fields",
]
