"""``Model`` — the value a handler parameter actually receives.

Paul's 2026-08-17 ruling, made mechanical: **the generated family class is the
TYPE; the injected value is a fully resolved INSTANCE** carrying graph, bound
weights and tuned values. So a handler writes

```python
def generate(self, ctx: RequestContext, p: In, flux: Flux1Dev) -> Out:
    latents = flux.denoiser(resolution=1024, hidden_states=..., timestep=...)
```

and ``flux`` is an instance of ``Flux1Dev``, not the class. Two parameters of
one family type are TWO checkpoints with independent weights and independent
tuning — ``flux_a: Flux1Dev, flux_b: Flux1Dev`` — which is why nothing here
holds per-instance state at module or class level (torchcg G13). A generated
singleton, a module-level registry of live instances, or a process-global
default would break that, and each is a codegen defect rather than a feature.

**The axis discipline, enforced rather than documented:**

| axis | lives on | examples |
|---|---|---|
| class | the generated class | buckets, graph classes, signatures, loop, scheduler |
| checkpoint | the instance | weights binding, ``tuned``, the ref label |
| request | ``ctx`` | cancellation, logging, progress, seed, assets |

Nothing crosses. In particular ``tuned`` is checkpoint-level and therefore
rides the instance: the values are still catalog-stamped per release slot
exactly as th#1116 stamps them today, and only the delivery address moved.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    Protocol,
    Self,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import msgspec

from ..families.base import GenerationDefaults
from .backing import Backing, BackingKind, DualBacking, EagerBacking, FakeBacking
from .errors import ModelError, ModelRefusal
from .snapshot import EagerExport, ExportedVariant, ModelExport
from .spec import ModelSpec


class InstanceResolver(Protocol):
    """How a dynamic ``ModelSpec.instance(ref)`` reaches real weights.

    The SDK does not own residency, downloads or placement, and must not grow
    them: those are worker policy, and pgw#1328's adopt-only role and the
    residency registry are where they live. So the resolution is a seam the
    worker fills at boot, and a process with no worker (CI, ``gen-worker dev``,
    an author's laptop) has none — which is exactly when :meth:`Model.
    fake` is the right answer rather than a silently-degraded real one.
    """

    def __call__(self, family: ModelSpec | None, name: str, ref: str) -> Backing: ...


_RESOLVER: InstanceResolver | None = None


def set_instance_resolver(resolver: InstanceResolver | None) -> InstanceResolver | None:
    """Install (or clear) the worker's backing resolver; returns the previous one.

    Deliberately a single process-wide seam and NOT a per-family registry: the
    thing that varies is which worker is hosting, never which family is asking,
    and a per-family hook would be a string-keyed lookup wearing a hat.
    """

    global _RESOLVER
    previous = _RESOLVER
    _RESOLVER = resolver
    return previous


def instance_resolver() -> InstanceResolver | None:
    return _RESOLVER


class DecodeSession:
    """A context-managed decode session for an autoregressive family.

    ``recipe_v1`` states a host-owned loop's per-step classes and who owns the
    state threaded between them, and then says outright that the ITERATION is
    the host's (torchcg G14). This object is that statement made usable: it owns
    the KV-cache allocation and teardown and exposes the same typed per-step
    callables, and it emits **no** driver, no step count and no termination
    condition, because the recipe deliberately refused to state one.

    It is single-use and refuses outside its context manager, so a cache cannot
    outlive the request that allocated it — the failure that turns a per-request
    allocation into a slow VRAM leak.
    """

    def __init__(self, instance: Model) -> None:
        self._instance = instance
        self._state: dict[str, Any] = {}
        self._open = False
        self._spent = False

    @property
    def state(self) -> dict[str, Any]:
        """The host-owned state threaded between steps (the KV cache, typically).

        Present only while the session is open. ``session_state`` on the recipe
        names who OWNS it: ``host`` when the caller passes it in and out as
        tensors, ``graph`` when the artifact carries it, ``none`` when there is
        none — this dict is the ``host`` case's home.
        """

        self._require_open()
        return self._state

    @property
    def instance(self) -> Model:
        self._require_open()
        return self._instance

    def _require_open(self) -> None:
        if not self._open:
            raise ModelError(
                ModelRefusal.SESSION_INVALID,
                "this decode session is not open; use `with inst.session() as s:` — a "
                "KV cache that outlives its context manager is a leak, not a cache",
            )

    def _enter(self) -> Self:
        if self._spent:
            raise ModelError(
                ModelRefusal.SESSION_INVALID,
                "a decode session is single-use; open a new one per request",
            )
        self._open = True
        self._spent = True
        return self

    def _exit(self) -> None:
        self._open = False
        self._state.clear()


class Model:
    """Base of every GENERATED model class: the type IS the class, the value
    IS the instance.

    ``class Flux1Dev(Model)`` is what a handler parameter is annotated with;
    the value it receives is an instance of it. There is deliberately no second
    name for the instance side (pgw#1346, Paul) — Python's own class/instance
    distinction already carries that reading, and the alias this replaces was
    one more name for one object.

    Generated subclasses add exactly three kinds of thing, and no others:

    * class-level identity — ``FAMILY``, ``EXPORT_DIGEST``, the bucket
      ``Literal`` aliases, the layout ``Literal`` alias, ``Tuned``;
    * one typed callable per runner, whose parameters are the ingress's
      parameters in the ingress's order (torchcg G2/G3);
    * nothing mutable. Class-level output is immutable identity, full stop.

    There is **no public constructor** (torchcg G12): a model cannot be called
    without a resolved instance, so a handler cannot accidentally reach a
    checkpoint-free class expecting weights to be there.
    """

    #: The family handle. Generated; never read as a lookup key.
    FAMILY: ClassVar[str] = ""
    #: The declaration-export digest these bindings were generated against.
    EXPORT_DIGEST: ClassVar[str] = ""
    #: The family's tuned-value SCHEMA. Values are catalog, per release slot.
    #: The bare base is the EAGER tier's honest answer for a model with no
    #: inference vocabulary at all (pgw#1346 K5/B5): an auxiliary or
    #: external-binary model registers no schema, so there is none to name.
    Tuned: ClassVar[type[GenerationDefaults]] = GenerationDefaults
    #: The committed ``family_export_v1`` document, loaded once per class.
    #: ``None`` on an EAGER model, which declares no graph classes and
    #: therefore exports ``eager_model_v1`` (below) instead — the F3 ruling's
    #: permanent tier, not a partially-generated binding.
    EXPORT: ClassVar[ModelExport | None] = None
    #: The committed ``eager_model_v1`` document, on an eager model only.
    EAGER: ClassVar[EagerExport | None] = None
    #: The declaration, when the catalog module that owns it is importable.
    #: ``None`` on an adopt-only serve pod, where the declaration's ``build``
    #: would import diffusers and the role forbids it (pgw#1328).
    SPEC: ClassVar[ModelSpec | None] = None

    __slots__ = ("_backing", "_label", "_ref", "_tuned")

    _backing: Backing
    _label: str
    _ref: str
    _tuned: GenerationDefaults

    def __init__(self) -> None:
        raise ModelError(
            ModelRefusal.BACKING_MISSING,
            f"{type(self).__name__} is a model TYPE; a value of it is a resolved instance. "
            f"Use {type(self).__name__}.instance(<checkpoint ref>) inside a handler, declare "
            "it in @endpoint(families={...}) so placement can prefetch the weights, or "
            f"{type(self).__name__}.fake() in a test.",
        )

    # ------------------------------------------------------------- construction

    @classmethod
    def _materialize(
        cls,
        *,
        ref: str,
        tuned: GenerationDefaults,
        backing: Backing,
        label: str = "",
    ) -> Self:
        """Build one instance. The ONLY way an instance comes into existence."""

        if not isinstance(tuned, cls.Tuned):
            raise ModelError(
                ModelRefusal.TUNED_INVALID,
                f"family {cls.FAMILY!r} expects {cls.Tuned.__name__} tuned values, got "
                f"{type(tuned).__name__}",
            )
        self = object.__new__(cls)
        object.__setattr__(self, "_ref", str(ref))
        object.__setattr__(self, "_tuned", tuned)
        object.__setattr__(self, "_backing", backing)
        object.__setattr__(self, "_label", str(label or ref))
        spec = cls.SPEC
        if spec is not None and spec.on_load is not None:
            spec.on_load(self)
        return self

    @classmethod
    def instance(cls, ref: str, *, tuned: GenerationDefaults | None = None) -> Self:
        """Resolve one checkpoint DYNAMICALLY, from inside a handler.

        This is the one parse-don't-validate boundary the design keeps: a
        request naming its own checkpoint is parsed HERE, once, and everything
        downstream holds a resolved instance. Static declaration in the
        decorator stays the default precisely because it is what lets placement
        prefetch the weights and verify the VRAM fit before the request lands.
        """

        resolver = instance_resolver()
        if resolver is None:
            raise ModelError(
                ModelRefusal.BACKING_MISSING,
                f"{cls.__name__}.instance({ref!r}) needs a worker to resolve weights and "
                "none is hosting this process. Declare the family in "
                "@endpoint(families={...}) so placement resolves it, or use "
                f"{cls.__name__}.fake() to run this handler hubless.",
            )
        backing = resolver(cls.SPEC, cls.FAMILY, str(ref))
        return cls._materialize(
            ref=str(ref), tuned=tuned if tuned is not None else cls.Tuned(), backing=backing
        )

    @classmethod
    def adopt(
        cls,
        *,
        ref: str,
        tuned: GenerationDefaults | None = None,
        eager: Mapping[str, Any] | None = None,
        compiled: Backing | None = None,
        label: str = "",
    ) -> Self:
        """Build an instance from backings the caller already has.

        The worker's own entry point: residency has the modules, adoption has
        the armed runners, and this folds them into one dual-backed instance
        whose handler-visible behaviour is identical either way.
        """

        return cls._materialize(
            ref=ref,
            tuned=tuned if tuned is not None else cls.Tuned(),
            backing=DualBacking(
                eager=None if eager is None else EagerBacking(eager), compiled=compiled
            ),
            label=label,
        )

    @classmethod
    def fake(cls, *, tuned: GenerationDefaults | None = None, seed: int = 0) -> Self:
        """A hubless, GPU-less, weight-less instance (greenfield B8).

        Every typed callable answers with shape-correct deterministic tensors
        derived from the declaration, so a handler's payload validation, bucket
        mapping, composition, streaming and asset saving are all exercised in
        CI by the real code path. It is not a mock of the SDK — it is the SDK,
        with the only part that needs a card replaced.
        """

        return cls._materialize(
            ref=f"fake:{cls.FAMILY}",
            tuned=tuned if tuned is not None else cls.Tuned(),
            backing=FakeBacking(cls.FAMILY, seed=seed),
            label=f"fake:{cls.FAMILY}",
        )

    # ------------------------------------------------------------ instance facts

    @property
    def ref(self) -> str:
        """The checkpoint this instance is bound to. Checkpoint-level, per §4.27."""

        return self._ref

    @property
    def label(self) -> str:
        """A human-readable name for receipts and logs; the ref when unnamed."""

        return self._label

    @property
    def tuned(self) -> GenerationDefaults:
        """This checkpoint's catalog-stamped tuned values, as a frozen struct.

        Typed as the family's own ``Tuned`` schema in generated subclasses.
        Payload values still win over these — see :func:`resolve_tuned`, which
        is where that precedence lives now instead of in every handler.
        """

        return self._tuned

    @property
    def backing(self) -> Backing:
        return self._backing

    def backing_kind(self, runner: str, bucket: Mapping[str, int] | None = None,
                     layout: str | None = None) -> BackingKind:
        """Which half answers this runner. For receipts, never for a branch."""

        variant = self.variant(runner, bucket, layout)
        backing = self._backing
        if isinstance(backing, DualBacking):
            return backing.kind_for(runner, variant)
        return backing.kind

    def with_tuned(self, tuned: GenerationDefaults) -> Self:
        """The same instance with different tuned values — a new object.

        Instances are immutable: rebinding tuned values in place would let one
        request's resolution leak into the next, which is precisely the
        cross-axis bleed the split exists to prevent.
        """

        return type(self)._materialize(
            ref=self._ref, tuned=tuned, backing=self._backing, label=self._label
        )

    # ---------------------------------------------------------------- invocation

    @classmethod
    def _graph_export(cls, what: str) -> ModelExport:
        """This class's ``family_export_v1``, or the eager tier's refusal.

        An EAGER model declares no graph classes, so every question phrased in
        graph vocabulary — which variant serves this bucket, what does the loop
        thread between steps — has no answer rather than a neutral one. Naming
        the tier in the refusal is what keeps it from reading as a broken
        binding (pgw#1346's F3 ruling: this is a permanent citizen).
        """

        export = cls.EXPORT
        if export is None:
            raise ModelError(
                ModelRefusal.FAMILY_INVALID,
                f"model {cls.FAMILY!r} is EAGER: it declares no graph classes, so it has "
                f"no {what}. An external-binary or non-PyTorch runtime is a permanent "
                "eager citizen and owes no graph (pgw#1346 F3); declare a GraphModelSpec "
                "if this model does have one.",
            )
        return export

    def variant(
        self,
        runner: str,
        bucket: Mapping[str, int] | None = None,
        layout: str | None = None,
    ) -> ExportedVariant:
        """The declared variant one call resolves to — an EXACT lookup.

        Never a rank: choosing which bucket serves a live call is
        ``ingress_selection_v1`` and choosing a layout is the hub's join with
        per-checkpoint layout metadata. This resolves a bucket the caller
        already decided.
        """

        export = type(self)._graph_export("runner variants")
        return export.runner(runner).variant(dict(bucket or {}), layout)

    def _invoke(
        self,
        runner: str,
        bucket: Mapping[str, int],
        layout: str | None,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Any:
        """The one path every generated typed callable goes through."""

        variant = self.variant(runner, bucket, layout)
        return self._backing.invoke(runner, variant, args, kwargs)

    @contextlib.contextmanager
    def session(self) -> Iterator[DecodeSession]:
        """Open a decode session for an autoregressive family.

        Refuses on a family whose loop is ``staged``: a staged composition has
        no state to thread between steps, so a session would be an empty object
        implying a capability the family does not have.
        """

        loop = type(self)._graph_export("declared loop").loop
        if loop.kind.value != "host":
            raise ModelError(
                ModelRefusal.SESSION_INVALID,
                f"family {type(self).FAMILY!r} declares a {loop.kind.value!r} loop; a decode "
                "session belongs to a host-owned (autoregressive) one",
            )
        session = DecodeSession(self)._enter()
        try:
            yield session
        finally:
            session._exit()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"{type(self).__name__}(ref={self._ref!r}, backing={self._backing.kind.value!r}, "
            f"tuned={self._tuned!r})"
        )


def resolve_tuned(
    payload: object, tuned: GenerationDefaults, fields: tuple[str, ...]
) -> dict[str, Any]:
    """Payload-over-``inst.tuned`` resolution, once, in the SDK (greenfield B3).

    Every handler in the fleet used to open with some spelling of
    ``steps = p.steps if p.steps is not None else ctx.defaults.steps``, one line
    per field, hand-written per endpoint. Each one is a place the precedence can
    be written backwards, and several were. The rule is now stated once: an
    EXPLICIT payload value wins; ``None`` means "no opinion" and falls through
    to the checkpoint's tuned value.

    ``RuntimeFormula`` reads the same resolution, so a pricing formula and the
    handler can no longer disagree about what ``steps`` was.
    """

    resolved: dict[str, Any] = {}
    for name in fields:
        value = getattr(payload, name, None)
        resolved[name] = getattr(tuned, name) if value is None else value
    return resolved


def tuned_fields(schema: type[GenerationDefaults]) -> tuple[str, ...]:
    """The tuned schema's own fields, excluding the version stamp."""

    return tuple(
        row.name for row in msgspec.structs.fields(schema) if row.name != "schema_version"
    )


class _TunedMarker:
    """The sentinel that makes ``Tuned[int]`` distinguishable from ``int | None``.

    Without it the two are the same annotation and nothing could tell a field
    that FALLS BACK to the checkpoint's tuned value from one that simply has no
    default. With it, resolution is derivable from the payload type rather than
    restated per handler — which is the whole of greenfield B3.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "Tuned"


#: The marker instance. One object, compared by identity.
TUNED: Final = _TunedMarker()

_T = TypeVar("_T")

#: A payload field annotated ``Tuned[int]`` resolves payload-over-``inst.tuned``
#: and reaches the handler as a final value. It is an ANNOTATION, not a
#: container: the runtime type is exactly ``int | None``, so msgspec decodes it
#: unchanged, the exported JSON Schema is byte-identical to the plain optional's,
#: and nothing downstream has to unwrap anything.
#:
#: ``T | None`` because that IS the wire shape — a field the caller may omit.
#: Pretending otherwise would make the payload schema lie about what a client
#: can send, which is the one thing a schema may never do.
Tuned = Annotated[_T | None, TUNED]


def tuned_payload_fields(payload_type: type) -> tuple[str, ...]:
    """The payload fields annotated ``Tuned[...]``, in declaration order.

    Derived from the type rather than listed by the handler, so a field added
    to the payload participates without anybody remembering to add it to a
    second list — the drift that makes per-handler resolution unreliable.
    """

    try:
        hints = get_type_hints(payload_type, include_extras=True)
    except Exception:  # noqa: BLE001 - an unresolvable hint resolves nothing
        return ()
    return tuple(
        name
        for name, hint in hints.items()
        if get_origin(hint) is Annotated
        and any(isinstance(extra, _TunedMarker) for extra in get_args(hint)[1:])
    )


def resolve(payload: object, instance: Model) -> dict[str, Any]:
    """Every ``Tuned[...]`` payload field, resolved against one instance.

    The B3 shape a handler actually wants: one call, no field list, and the
    same precedence ``RuntimeFormula`` reads — so a pricing formula and the
    handler can no longer disagree about what ``steps`` was.
    """

    return resolve_tuned(payload, instance.tuned, tuned_payload_fields(type(payload)))


__all__ = [
    "TUNED",
    "DecodeSession",
    "Model",
    "InstanceResolver",
    "Tuned",
    "instance_resolver",
    "resolve",
    "resolve_tuned",
    "set_instance_resolver",
    "tuned_fields",
    "tuned_payload_fields",
]
