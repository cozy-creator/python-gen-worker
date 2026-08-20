"""``@entrypoint`` — the stateless half of the pgw#1382 split.

An entrypoint is a MODULE-LEVEL function: pure request -> response logic
that holds no state and needs none (statelessness is STRUCTURAL — there is
no ``self`` to stash state on). The signature contract (Paul's ctx-first
ruling, 2026-08-17 — one parameter-order rule across the SDK, matching
``load(self, ctx)``; adapters-as-parameters ruling, same review)::

    @entrypoint
    def generate(ctx: RequestContext, payload: TextToImageInput,
                 model: SdxlModel) -> ImageOutput: ...

    @entrypoint
    def generate(ctx: RequestContext, payload: TextToImageInput,
                 model: SdxlModel, turbo: Adapter | None,
                 loras: list[Adapter]) -> ImageOutput: ...

    @entrypoint                       # WEIGHTLESS (pgw#1392): no slot at all
    def transform(ctx: RequestContext, payload: TransformInput
                  ) -> TransformOutput: ...

    @entrypoint(resources=Resources(  # the STAFFING ENVELOPE (pgw#1396)
        vcpus=16, max_gpu_count=4, max_gpus_per_execution_group=4,
        parallel=("sequence",),
        requires=LayoutRequirements(recommended="ram96g")))
    def generate(ctx: RequestContext, payload: GenerateInput,
                 video: H3Model) -> GenerateOutput: ...

    @entrypoint(streams=TokenDelta)   # INCREMENTAL OUTPUT (pgw#1576): the
    async def complete(ctx: RequestContext,   # chunk type is declared, the
                       payload: CompletionInput,   # terminal struct is still
                       model: QwenModel) -> Completion: ...  # RETURNED

    @entrypoint(kind="conversion",    # a PRODUCER (pgw#1406)
                publishes=True, env=("HF_TOKEN", "CIVITAI_API_KEY"))
    def cast_dtype(ctx: RequestContext, payload: CastDtypeInput
                   ) -> PublishResult: ...

    @entrypoint(child_calls=True)     # a WORKFLOW (pgw#1579): this body
    def make_video(ctx: RequestContext,   # composes OTHER endpoints through
                   payload: DjInput) -> DjOutput: ...  # ctx.call_endpoint

* first: ``ctx`` annotated :class:`~gen_worker.serving.context.RequestContext`
  — ONE class for inference and producers alike (pgw#1294/pgw#1306: *no kind
  selects a different class, because no kind decides what a body may write —
  the declaration does*). It carries the publisher surface
  (``save_checkpoint``, ``mktemp``, ``source_path``) and REFUSES it typed
  unless ``publishes=True`` was declared.
* second: the payload, a ``msgspec.Struct`` — the wire schema
* remaining, in any order and **possibly none**: model SLOTS (params
  annotated with a :class:`~gen_worker.serving.model.Model` subclass) and
  adapter SLOTS (``Adapter`` / ``Adapter | None`` single, ``list[Adapter]``
  the request's picks; the hub resolves what rides per deployment/request
  into them). The PARAMETER NAME is the slot name the request envelope
  keys per-slot picks on (``{"model": ref, "loras": [{"ref":…, "scale":…}]}``).
  **Zero slots is a valid declaration (pgw#1392)** — a CPU workflow helper
  has no weights to type, so it declares none, its envelope carries no model
  field, and nothing is ever made resident for it. Zero slots is legal;
  junk slots are not — a declared parameter must still BE a valid slot.
* return: a ``msgspec.Struct``

The decorator validates at import (typed refusal names the exact defect) and
stamps :data:`ENTRYPOINT_ATTR` with an :class:`EntrypointSpec` — the whole
publish-time extraction surface, readable without executing author code
beyond import.

**THE KWARG RULE (pgw#1406).** Decorator kwargs are declarations about
PLACEMENT AND AUTHORITY; the signature carries MODEL SLOTS. ``resources=``
(pgw#1396) and ``kind=`` are placement; ``publishes=`` / ``env=`` /
``emits_media=`` / ``child_calls=`` / ``handles=`` are authority, and together
they are the producer plane's whole declaration set — pgw#983 deleted ``@job``
and this is where its 27 conversion producers land (th#2173). The
"``resources=`` is the ONE kwarg" sentence described the surface on the day it
had one; it was never the principle.

**CAPABILITIES AND BEHAVIORAL DIVERGENCE ARE DECLARED, NEVER INFERRED**
(coordinator ruling, 2026-08-20, on pgw#1579/pgw#1580). ``child_calls=`` and
``handles=`` are back here rather than sniffed out of the body, because the
manifest is where a capability is asked for: inferring ``child_calls`` from the
presence of ``ctx.call_endpoint`` would mint an ``invoke_child`` credential for
anything that merely imports the surface.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, Literal, Tuple, Type, TypeVar, get_type_hints, overload,
)

import msgspec

from .context import Adapter, DistillationAdapter
from .model import Model

F = TypeVar("F", bound=Callable[..., Any])

#: The one attribute @entrypoint stamps on the author's function.
ENTRYPOINT_ATTR = "__cozy_entrypoint__"


class EntrypointDeclarationError(TypeError):
    """An @entrypoint function does not satisfy the signature contract."""


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """One declared slot parameter, in signature order after the payload.

    Kinds: ``model`` (a Model subclass — required), ``adapter``
    (``Adapter``/``DistillationAdapter`` required, ``| None`` optional —
    the ANNOTATION records which adapter KIND the slot takes; the hub
    enforces it at pick time, the worker at resolution), ``adapters``
    (``list[Adapter]`` — the request's picks, empty when none ride)."""

    name: str
    kind: Literal["model", "adapter", "adapters"]
    #: The model class for a model slot; :class:`Adapter` or
    #: :class:`DistillationAdapter` for adapter slots (the slot's KIND).
    annotation: type
    #: ``Adapter | None`` and ``list[Adapter]`` slots are not required.
    required: bool = True


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    """One entrypoint, statically extracted: name, payload schema, ordered
    slots (param name = slot name), return type."""

    name: str
    fn: Callable[..., Any]
    payload_type: Type[msgspec.Struct]
    slots: Tuple[SlotSpec, ...]
    return_type: Type[msgspec.Struct]
    #: The declared :class:`~gen_worker.api.resources.Resources`, or ``None``.
    #: se#755/pgw#1396: the STAFFING ENVELOPE — the machine this function is
    #: placed on. Function scope because the hub's unit of placement is the
    #: function (pgw#1394) and because `music-analysis` declares three
    #: different vCPU floors across three functions of one endpoint.
    resources: Any = None
    #: pgw#1406: this function MAY write tensors/repos to the hub. The
    #: declaration says MAY write; the request still says WHERE. It is the ONE
    #: justification for the hub minting the repo-write capability grant
    #: (th#2049), and the SDK stamps the same fact onto the context so the
    #: publisher surface refuses undeclared code before a byte moves — one
    #: fact, two enforcers, no drift.
    publishes: bool = False
    #: pgw#1406: the environment-variable NAMES this function reads. Values are
    #: release config and never manifest data; the hub validates config-layer
    #: env writes against this declaration.
    env: Tuple[str, ...] = ()
    #: pgw#1406: TRI-STATE. ``None`` = undeclared, the inference default (media
    #: IS the product of a request, and the hub grants ``upload_media``
    #: unconditionally on the request path). ``True`` = declared. ``False`` =
    #: an explicit opt-OUT, which the SDK enforces at the media surface. See
    #: the decorator docstring for why this one does not reach the hub.
    emits_media: bool | None = None
    #: pgw#1406: the WORKLOAD KIND, the hub's own vocabulary
    #: (``internal/functionkind``: inference | training | conversion | dataset
    #: | eval). ``""`` is undeclared and means ``inference``. NOT a write
    #: declaration — th#2052 cut that and ``publishes=`` replaced it — but the
    #: hub still derives three PLACEMENT facts from it, which is why a producer
    #: cannot leave it at the default. See the decorator docstring.
    kind: str = ""
    #: pgw#1579: this function makes endpoint-to-endpoint CHILD CALLS. It is
    #: the hub's ONE justification for minting the ``invoke_child`` capability
    #: grant (``scheduler_dispatch.go`` mints it only ``if
    #: subj.ChildCallsDeclared``), so an undeclared function's token cannot
    #: submit a child request and ``ctx.call_endpoint`` fails at the FIRST
    #: invoke — not at publish.
    child_calls: bool = False
    #: pgw#1580: the concrete execution-lane BODIES this function's code
    #: branches on (``ctx.execution_lane``) — a BEHAVIORAL-DIVERGENCE marker,
    #: never inventory and never a request for a lane. Tokens come from
    #: ``models.execution_lanes.known_execution_lane_bodies()``, the twin of
    #: the table the hub validates against; the execution axis (eager/compiled)
    #: is the platform's and is refused here.
    handles: Tuple[str, ...] = ()
    #: The class the author annotated on ``ctx``. The serve loop constructs
    #: THIS class, so a producer that annotates ``JobContext`` receives the
    #: publisher surface and an inference entrypoint is unchanged.
    ctx_type: type | None = None
    #: pgw#1576: the declared ``streams=`` chunk ANNOTATION, or ``None``. One
    #: struct type, or a discriminated union when a handler streams several
    #: shapes. Present means this function emits incremental output: publish
    #: reports ``incremental_output`` + ``delta_output_schema`` off THIS field
    #: without executing author code.
    delta_type: Any = None
    #: The same declaration as an isinstance-able tuple — what ``ctx.emit``
    #: refuses a foreign chunk against.
    delta_arms: Tuple[Type[msgspec.Struct], ...] = ()

    @property
    def model_params(self) -> Tuple[Tuple[str, type], ...]:
        return tuple(
            (slot.name, slot.annotation) for slot in self.slots if slot.kind == "model"
        )

    @property
    def adapter_params(self) -> Tuple[SlotSpec, ...]:
        return tuple(slot for slot in self.slots if slot.kind == "adapter")

    @property
    def model_classes(self) -> Tuple[type, ...]:
        """Referenced model classes, first-reference order, deduplicated."""
        seen: Dict[type, None] = {}
        for _, cls in self.model_params:
            seen.setdefault(cls)
        return tuple(seen)


def _refuse(fn: Callable[..., Any], message: str) -> EntrypointDeclarationError:
    return EntrypointDeclarationError(
        f"@entrypoint {fn.__module__}.{fn.__qualname__}: {message}"
    )


def _annotation_class(annotation: Any) -> type | None:
    """The concrete class of an annotation, seeing through Generic aliases
    (``RequestContext[D]`` -> ``RequestContext``)."""
    origin = typing.get_origin(annotation)
    candidate = origin if origin is not None else annotation
    return candidate if isinstance(candidate, type) else None


def _slot_of(fn: Callable[..., Any], name: str, annotation: Any) -> SlotSpec:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        arms = [arm for arm in typing.get_args(annotation) if arm is not type(None)]
        if len(arms) == 1 and arms[0] in (Adapter, DistillationAdapter):
            return SlotSpec(
                name=name, kind="adapter", annotation=arms[0], required=False)
        raise _refuse(
            fn,
            f"parameter {name!r}: the only optional slot forms are "
            f"`Adapter | None` and `DistillationAdapter | None`, got "
            f"{annotation!r}",
        )
    if origin is list:
        (item,) = typing.get_args(annotation) or (None,)
        if item is Adapter:
            return SlotSpec(
                name=name, kind="adapters", annotation=Adapter, required=False)
        raise _refuse(
            fn,
            f"parameter {name!r}: the only list slot form is "
            f"`list[Adapter]` (the request's adapter picks), got {annotation!r}",
        )
    concrete = _annotation_class(annotation)
    if concrete in (Adapter, DistillationAdapter):
        assert concrete is not None
        return SlotSpec(name=name, kind="adapter", annotation=concrete, required=True)
    if concrete is Model:
        raise _refuse(
            fn,
            f"parameter {name!r} is annotated with the bare Model base; "
            "declare the author's model class",
        )
    if concrete is not None and issubclass(concrete, Model):
        return SlotSpec(name=name, kind="model", annotation=concrete, required=True)
    raise _refuse(
        fn,
        f"parameter {name!r} must be a model slot (annotated with a "
        "gen_worker.Model subclass) or an adapter slot (Adapter / "
        f"Adapter | None); the annotation IS the declaration and the param "
        f"name is the slot name — got {annotation!r}",
    )


_ENV_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

#: The hub's workload-kind vocabulary, verbatim from
#: `internal/functionkind/functionkind.go:All`. NOT invented here — every value
#: is one the hub normalizes and reads, and `""` is the undeclared default that
#: `Normalize` maps onto `inference`.
KINDS = ("inference", "training", "conversion", "dataset", "eval")


def _validate_env_decl(fn: Callable[..., Any], env: Any) -> Tuple[str, ...]:
    """The declared env-var NAMES, validated at decoration.

    Semantics are v1's verbatim (`api/decorators.py:_validate_env_decl`,
    deleted with the catalog by pgw#983): a list/tuple of upper-snake names,
    <=64 chars, no repeats, a bare string refused rather than iterated into
    characters. Only the refusal TEXT changed, to name the author's line the
    way every other `@entrypoint` refusal does.
    """
    if env is None:
        return ()
    if isinstance(env, str) or not isinstance(env, (list, tuple)):
        raise _refuse(
            fn,
            f"env= must be a list/tuple of environment-variable NAMES, got "
            f"{type(env).__name__} — a bare string would iterate into "
            "characters, so it is refused rather than accepted",
        )
    out: list[str] = []
    for name in env:
        candidate = str(name or "").strip()
        if (
            not candidate
            or len(candidate) > 64
            or not ("A" <= candidate[0] <= "Z")
            or any(c not in _ENV_NAME_CHARS for c in candidate)
        ):
            raise _refuse(
                fn,
                f"env= name {name!r} is not a valid environment-variable name "
                "(UPPER_SNAKE, starting with a letter, <=64 chars)",
            )
        if candidate in out:
            raise _refuse(fn, f"env= repeats {name!r}")
        out.append(candidate)
    return tuple(out)


def _delta_declaration(
    fn: Callable[..., Any], streams: Any
) -> Tuple[Any, Tuple[Type[msgspec.Struct], ...]]:
    """``streams=`` normalized: ``(annotation, arms)``.

    Accepts one ``msgspec.Struct`` type, a tuple/list of them, or ``A | B`` —
    one handler legitimately streams more than one shape. The ANNOTATION is
    what publish turns into ``delta_output_schema`` (a union becomes a
    discriminated ``anyOf``); the ARMS are what ``ctx.emit`` type-checks
    against.
    """
    if streams is None:
        return None, ()
    if isinstance(streams, (tuple, list)):
        candidates = tuple(streams)
    elif typing.get_origin(streams) in (typing.Union, types.UnionType):
        candidates = typing.get_args(streams)
    else:
        candidates = (streams,)
    arms: list[Type[msgspec.Struct]] = []
    for candidate in candidates:
        concrete = _annotation_class(candidate)
        if concrete is None or not issubclass(concrete, msgspec.Struct):
            raise _refuse(
                fn,
                f"streams= takes the msgspec.Struct chunk type(s) this "
                f"function emits (gen_worker.TokenDelta, gen_worker.ItemDelta, "
                f"your own Delta subclass, or a tuple/union of them), got "
                f"{candidate!r} — the type is what publish reports as "
                f"delta_output_schema",
            )
        arms.append(concrete)
    if not arms:
        raise _refuse(fn, "streams= declares no chunk type")
    if len(arms) == 1:
        return arms[0], (arms[0],)
    # `Union[tuple(...)]` at runtime — the arms are only known here, so mypy
    # cannot read it as a type alias and is told so.
    annotation: Any = typing.Union[tuple(arms)]
    untagged = [
        arm.__name__
        for arm in arms
        if getattr(arm, "__struct_config__", None) is None
        or getattr(arm.__struct_config__, "tag", None) is None
    ]
    if untagged:
        raise _refuse(
            fn,
            f"streams= declares several chunk types and {untagged} carry no "
            "msgspec tag, so the published delta_output_schema could not tell "
            "them apart. Subclass gen_worker.Delta (tagged on `type`), or "
            "declare tag=True on your struct",
        )
    return annotation, tuple(arms)


def _validate_handles_decl(fn: Callable[..., Any], handles: Any) -> Tuple[str, ...]:
    """The declared lane BODIES, validated at decoration (pgw#1580).

    Every refusal here is one the hub's ``normalizeManifestHandles``
    (``internal/builder/manifest_contract.go``) would raise instead — after the
    image bake and the registry push. The vocabulary is not invented: it is
    ``models.execution_lanes.known_execution_lane_bodies()``, already
    documented in that module as *"the valid ``handles=`` declaration
    tokens"*, and the twin of ``precision.KnownExecutionLaneBodies`` the hub
    checks. The execution axis is refused with its own sentence rather than
    lumped in with "unknown token", because ``"fp8-w8a8-dynamic+compiled"`` is
    a legal LANE and an illegal DECLARATION, and the difference is the point.
    """
    from ..models.execution_lanes import (
        known_execution_lane_bodies,
        valid_execution_lane_body,
    )

    if handles is None:
        return ()
    if isinstance(handles, str) or not isinstance(handles, (list, tuple)):
        raise _refuse(
            fn,
            f"handles= must be a list/tuple of lane BODY tokens, got "
            f"{type(handles).__name__} — a bare string would iterate into "
            "characters, so it is refused rather than accepted",
        )
    known = ", ".join(known_execution_lane_bodies())
    out: list[str] = []
    for raw in handles:
        token = str(raw or "").strip().lower()
        if not token:
            raise _refuse(fn, "handles= carries an empty token")
        if "+" in token:
            raise _refuse(
                fn,
                f"handles= token {raw!r} carries an execution axis; declare "
                "the lane BODY — eager/compiled is platform-managed and is "
                "not something a body can branch on",
            )
        if not valid_execution_lane_body(token):
            raise _refuse(
                fn,
                f"handles= token {raw!r} is not a known lane body "
                f"(known: {known}) — the hub validates this against the same "
                "table AFTER the image bake, so it is refused here instead",
            )
        if token in out:
            raise _refuse(fn, f"handles= repeats {raw!r}")
        out.append(token)
    return tuple(out)


@overload
def entrypoint(fn: F) -> F: ...
@overload
def entrypoint(
    *,
    resources: Any = ...,
    kind: str = ...,
    publishes: bool = ...,
    env: Any = ...,
    emits_media: bool | None = ...,
    streams: Any = ...,
    child_calls: bool = ...,
    handles: Any = ...,
) -> Callable[[F], F]: ...


def entrypoint(
    fn: F | None = None,
    *,
    resources: Any = None,
    kind: str = "",
    publishes: bool = False,
    env: Any = None,
    emits_media: bool | None = None,
    streams: Any = None,
    child_calls: bool = False,
    handles: Any = None,
) -> F | Callable[[F], F]:
    """Mark a module-level function as an entrypoint (contract above).

    ``resources=`` is the PLACEMENT kwarg, and it is the
    STAFFING ENVELOPE (se#755/pgw#1396): the machine this function is placed
    on — ``Resources(vcpus=…, max_gpu_count=…,
    max_gpus_per_execution_group=…, parallel=…, requires=…)``. It is FUNCTION
    scope for three reasons, none of them convenience:

    * the hub's unit of placement IS the function — pgw#1394 folded the
      per-lane VRAM floor to function scope for exactly this reason
      (*"a function is placed on a single machine before anything knows which
      lane it will serve"*), and every term here is placed by that same code;
    * one endpoint's functions legitimately differ: ``music-analysis``
      declares ``vcpus=2``, ``vcpus=8``, ``vcpus=8`` across three functions,
      and a single endpoint- or model-scoped number is a silent over- or
      under-buy of two of them;
    * a WEIGHTLESS entrypoint (pgw#1392) has no ``Model`` class to hang it on,
      and weightless endpoints are precisely the ones a CPU floor unblocks.

    This does NOT reopen pgw#1382. That hardcut moved the *weight and lane*
    declarations (``model=``, ``lanes=``) onto the ``Model`` class header
    because they are properties of the WEIGHTS. A machine ask is a property of
    the CODE THAT RUNS, which is this function.

    ``vcpus`` is NOT a ``requires=`` term and must never become one: the hub
    already reads ``resources.vcpus`` and normalizes it to ``min_vcpus``
    itself, so a second spelling would land in one payload key from two
    directions. The asymmetry with ``min_host_ram_gb`` (which IS a
    ``requires=`` term, at ``recommended`` only) is deliberate: vCPUs are
    selectable and filterable at both providers, host RAM on a RunPod GPU pod
    is neither, so a host-RAM MINIMUM stays forbidden (Paul, 2026-07-11).

    ``kind=`` is the second PLACEMENT kwarg, and a producer CANNOT leave it at
    the default. The vocabulary is the hub's own — `internal/functionkind`:
    ``inference`` (the default) | ``training`` | ``conversion`` | ``dataset``
    | ``eval`` — so nothing is invented here. th#2052 cut kind from deciding
    WRITES and ``publishes=`` replaced it for that, but ``functionkind.go``
    itself draws the remaining line: *"Naming a ref is a READ; writing one is
    the function's `publishes` declaration and is not derived here."* The hub
    still derives three PLACEMENT facts from kind, and all three are wrong for
    a producer that leaves it at ``inference``:

    * ``NamesReservedRefs(kind) != inference`` — an inference row gets NO
      preflight resolution and NO dispatch-time read grant for the reserved
      ``source`` / ``text_encoder`` / ``destination`` / dataset-pin payload
      fields, so the dispatch carries no snapshot for them and
      ``serving/reserved_repos.py`` has nothing to pin against.
      **Do NOT read a ``ctx.source_path is None`` failure as proof that
      ``kind=`` is wrong.** pgw#1475 is exactly that symptom with ``kind=
      "conversion"`` correctly declared: the SDK's own materialization step
      had been deleted with ``executor.py``, and this note sent the first
      reader hunting for a declaration that was already right. Check
      ``git grep -n _set_source_path -- src/`` for a WRITER before suspecting
      the kind.
    * ``WorkerCapabilityTokenTTL`` — 1h for inference, 24h for conversion. A
      multi-hour quantization would spend its life renewing.
    * ``SizesDiskFromSource`` — true for conversion and eval only. An
      inference-kind row sizes a pod's disk without looking at the 13.9 GB
      source it is about to download.

    ``publishes=`` / ``env=`` / ``emits_media=`` are the AUTHORITY kwargs
    (pgw#1406) — the producer plane's declaration set, carried over verbatim
    from the ``@job`` pgw#983 deleted so the 27 ``cozy-creator/jobs``
    conversion producers have a migration target (th#2173):

    * ``publishes=True`` — this function MAY write tensors/repos to the hub.
      MAY write; the request still says WHERE. It is the hub's ONE
      justification for minting the repo-write capability grant (th#2049),
      and it reaches that decision UNCHANGED: the hub already decodes
      ``functions[].publishes``, and ``entrypoints[]`` folds into
      ``Functions`` at ``builder.ParseManifest``'s one decode site. The SDK
      stamps the same fact onto the context, so ``ctx.save_checkpoint`` /
      ``gen_worker.convert.publish_flavors`` refuse undeclared code before a
      byte moves — one fact, two enforcers.
    * ``env=("HF_TOKEN", "CIVITAI_API_KEY")`` — the env-var NAMES this code
      reads. Values are release config and never manifest data; the hub
      validates config-layer env writes against the declaration and already
      decodes ``functions[].env``.
    * ``emits_media=`` — TRI-STATE, and the one that does NOT reach the hub.
      Measured at tensorhub master before this was written: the request path
      grants ``upload_media`` UNCONDITIONALLY
      (``capability_subject_th2068.go:124``, ``UploadsMedia: true``) because
      a request's product IS media; only the JOB path gates it on a
      declaration. So a producer porting here LOSES NOTHING by declaring it
      and a hub read could only NARROW — which would fire on every existing
      endpoint that saves an image without declaring one. The kwarg is
      therefore accepted (so a producer ports verbatim) and enforced
      SDK-side: ``None`` is the undeclared inference default, ``True``
      declares emission, and an explicit ``False`` refuses the media surface
      at the call site.

      ON THE WIRE IT RIDES A JOB-KIND ROW ONLY (th#2177 narrowed this, and
      was right). The paragraph above is true of a row dispatched as a
      REQUEST and does not survive one dispatched as a JOB:
      ``jobCapabilitySubject`` sets ``UploadsMedia: d.EmitsMedia`` from
      ``endpoint_release_jobs.emits_media``, a column that already exists —
      so on that path the declaration is READ and it GATES, and withholding
      it would silently downgrade every producer that declared media. An
      INFERENCE row still emits nothing, because nothing there can store or
      read it; that half of th#2087's fence stays.

    ``streams=`` is the INCREMENTAL-OUTPUT declaration (pgw#1576): the
    ``msgspec.Struct`` type this function emits through :meth:`ctx.emit
    <gen_worker.serving.context.RequestContext.emit>` while it works. It does
    NOT change the return contract — a streaming entrypoint still returns one
    struct, because the wire carries two things and they promise different
    amounts::

        @entrypoint(streams=TokenDelta)
        async def complete(ctx: RequestContext, payload: CompletionInput,
                           model: QwenModel) -> Completion:
            parts = []
            async for token in engine.stream(payload.prompt):
                ctx.raise_if_cancelled()
                parts.append(token)
                ctx.emit(TokenDelta(text=token))
            return Completion(text="".join(parts))

    One handler may stream several shapes — ``streams=(TokenDelta, ItemDelta)``
    or ``streams=TokenDelta | ItemDelta`` — and the manifest publishes a
    DISCRIMINATED ``anyOf`` over them, which is why every arm must carry a
    msgspec tag (``gen_worker.Delta`` subclasses do).

    ``JobProgress`` deltas are ordered, live, never persisted and DROPPABLE by
    contract; ``JobResult`` is the single authoritative output every
    non-streaming caller and the request record read. The declaration is read
    statically at publish — ``incremental_output`` and ``delta_output_schema``
    come off ``EntrypointSpec.delta_type``, no author code executed — and
    ``ctx.emit`` refuses both an undeclared function and a chunk of another
    type, the same one-fact-two-enforcers shape as ``publishes=``.

    **An async-generator entrypoint is refused, and the refusal is the design.**
    Python forbids ``return <value>`` inside an async generator, so
    ``-> AsyncIterator[Delta]`` can express the droppable half of the wire and
    nothing else: the authoritative result would have to be folded out of the
    deltas by the platform (v1's accumulator, magic field-peeling over three
    blessed types) or smuggled in as a special last frame. Both are weaker than
    a plain ``return``, so the generator shape is refused at import naming
    ``streams=`` + ``ctx.emit`` as the successor.

    ``child_calls=`` / ``handles=`` are the other two AUTHORITY kwargs, restored
    by pgw#1579/pgw#1580 after the v1 hardcut deleted them while all three
    other layers kept speaking them. Both are DECLARATIONS, on the ruling that
    capabilities and behavioral divergence are never inferred from code shape:

    * ``child_calls=True`` — this function calls OTHER endpoints
      (``ctx.call_endpoint``). ``scheduler_dispatch.go`` mints the
      ``invoke_child`` capability grant only ``if subj.ChildCallsDeclared``,
      and ``manifestFunction.ChildCalls`` is where that flag comes from, so an
      undeclared workflow endpoint publishes fine, deploys fine, and then fails
      at its FIRST child call with ``child_calls_not_declared``. It is what
      ``private-inference-endpoints/dj-pipeline`` IS: ``make_video`` is
      ordinary CPU code that composes the DJ pipeline out of child requests.
      Inference from the body is refused BY DESIGN — it would mint the
      credential for anything that merely imports the surface.
    * ``handles=("fp8-w8a8-dynamic",)`` — the concrete lane BODIES this
      function's code BRANCHES ON, read back at runtime as
      ``ctx.execution_lane``. A behavioral-divergence marker and nothing else:
      it is not a request for a lane, not inventory, and it never narrows what
      the platform may serve. The hub validates the tokens against its
      concrete lane-body table; this decorator validates them against the SDK
      twin first, so the refusal names your line instead of arriving after the
      image bake. Undeclared is the norm — and ``ctx.execution_lane`` then
      REFUSES typed rather than handing back a default nobody checked, the same
      one-fact-two-enforcers shape as ``publishes=``. ``ctx.lane`` inside
      ``Model.load`` is deliberately NOT gated by this: that lane is the
      deploy's pick and ``ctx.lane.dtype`` is how every model loads at all,
      which is the ordinary path rather than a divergence.
    """

    if fn is None:
        def bind(inner: F) -> F:
            return entrypoint(  # type: ignore[call-overload,no-any-return]
                inner,
                resources=resources,
                kind=kind,
                publishes=publishes,
                env=env,
                emits_media=emits_media,
                streams=streams,
                child_calls=child_calls,
                handles=handles,
            )
        return bind

    from .context import RequestContext

    if not inspect.isfunction(fn):
        raise EntrypointDeclarationError(
            f"@entrypoint marks module-level functions, got "
            f"{type(fn).__name__} — the kwargs are resources= (pgw#1396), "
            "publishes=/env=/emits_media= (pgw#1406) and "
            "child_calls=/handles= (pgw#1579/pgw#1580); the Model class header "
            "carries lanes="
        )
    if not isinstance(publishes, bool):
        raise _refuse(
            fn,
            f"publishes= is a bool DECLARATION, got "
            f"{type(publishes).__name__} — it says this function MAY write to "
            "the hub; the request still says where",
        )
    if not isinstance(child_calls, bool):
        raise _refuse(
            fn,
            f"child_calls= is a bool DECLARATION, got "
            f"{type(child_calls).__name__} — it says this function calls OTHER "
            "endpoints, and it is what the hub mints the invoke_child grant "
            "against",
        )
    if emits_media is not None and not isinstance(emits_media, bool):
        raise _refuse(
            fn,
            f"emits_media= is a bool or None (undeclared), got "
            f"{type(emits_media).__name__}",
        )
    declared_kind = str(kind or "").strip().lower()
    if declared_kind and declared_kind not in KINDS:
        raise _refuse(
            fn,
            f"kind= must be one of {KINDS}, got {kind!r} — the vocabulary is "
            "the hub's (internal/functionkind), so a value it does not "
            "normalize would silently become 'inference'",
        )
    declared_env = _validate_env_decl(fn, env)
    delta_type, delta_arms = _delta_declaration(fn, streams)
    declared_handles = _validate_handles_decl(fn, handles)
    if resources is not None:
        from ..api.resources import Resources

        if not isinstance(resources, Resources):
            raise _refuse(
                fn,
                f"resources= takes a gen_worker.Resources, got "
                f"{type(resources).__name__} — the staffing envelope is a "
                "typed declaration so its refusals name your line, not a "
                "manifest key",
            )
    if "." in fn.__qualname__:
        raise _refuse(
            fn,
            "entrypoints are MODULE-LEVEL functions (statelessness is "
            "structural); methods and nested functions cannot be entrypoints",
        )
    try:
        hints = get_type_hints(fn)
    except Exception as exc:
        raise _refuse(fn, f"unresolvable type hints: {exc}") from exc

    parameters = list(inspect.signature(fn).parameters.values())
    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise _refuse(
                fn,
                f"parameter {parameter.name!r} is {parameter.kind.description}; "
                "the signature is plain positional: (ctx, payload, *slots)",
            )
    if len(parameters) < 2:
        raise _refuse(
            fn,
            f"takes {len(parameters)} parameters; the contract is "
            "(ctx: RequestContext, payload: msgspec.Struct) followed by "
            "ZERO OR MORE slots (<Model subclass or Adapter>) — pgw#1392: a "
            "weightless entrypoint declares no slot at all",
        )

    ctx_parameter, payload_parameter, slot_parameters = (
        parameters[0], parameters[1], parameters[2:],
    )

    # ONE context class, for inference and producers alike (pgw#1294/pgw#1306:
    # "no kind selects a different class, because no kind decides what a body
    # may write — the declaration does"). `RequestContext` carries the
    # publisher surface and refuses it unless `publishes=True` was declared,
    # so a producer needs no second annotation to migrate here (th#2173).
    ctx_type = _annotation_class(hints.get(ctx_parameter.name))
    if ctx_type is None or not issubclass(ctx_type, RequestContext):
        raise _refuse(
            fn,
            f"first parameter {ctx_parameter.name!r} must be annotated "
            "RequestContext — ctx comes FIRST (Paul's parameter-order "
            f"ruling), got {hints.get(ctx_parameter.name)!r}",
        )

    payload_type = _annotation_class(hints.get(payload_parameter.name))
    if payload_type is None or not issubclass(payload_type, msgspec.Struct):
        raise _refuse(
            fn,
            f"second parameter {payload_parameter.name!r} must be the "
            "payload, annotated with a msgspec.Struct schema; got "
            f"{hints.get(payload_parameter.name)!r}",
        )

    slots = tuple(
        _slot_of(fn, parameter.name, hints.get(parameter.name))
        for parameter in slot_parameters
    )
    # pgw#1392: ZERO model slots is a valid declaration. A weightless
    # entrypoint (a CPU workflow helper — dj-utils, music-analysis) has no
    # weights to type, so it declares none; the envelope then has no model
    # field at all and nothing is ever resident for it. Zero slots is legal,
    # JUNK slots are not — `_slot_of` above still refuses every parameter
    # that is neither a Model subclass nor an adapter form.

    # pgw#1576: name the STREAMING migration before the generic refusal does.
    # An async-generator body (or an `AsyncIterator[...]`/`Iterator[...]`
    # return) is the v1 streaming shape, and "must be a msgspec.Struct" sends
    # its author looking for a schema defect that is not there.
    returns_iterator = _annotation_class(hints.get("return")) in (
        collections.abc.AsyncIterator, collections.abc.AsyncGenerator,
        collections.abc.Iterator, collections.abc.Generator,
    )
    if inspect.isasyncgenfunction(fn) or inspect.isgeneratorfunction(fn) or (
        returns_iterator
    ):
        raise _refuse(
            fn,
            "a generator entrypoint cannot produce the request's authoritative "
            "result — Python forbids `return <value>` inside a generator, and "
            "JobResult is a separate wire channel from the droppable "
            "JobProgress deltas. Declare the chunk type instead and return the "
            "terminal struct: `@entrypoint(streams=TokenDelta)` + "
            "`ctx.emit(TokenDelta(text=...))` in the loop + "
            "`return <YourOutput>(...)` at the end (pgw#1576)",
        )

    return_type = _annotation_class(hints.get("return"))
    if return_type is None or not issubclass(return_type, msgspec.Struct):
        raise _refuse(
            fn,
            f"return type must be a msgspec.Struct, got {hints.get('return')!r}",
        )

    spec = EntrypointSpec(
        name=fn.__name__,
        fn=fn,
        payload_type=payload_type,
        slots=slots,
        return_type=return_type,
        resources=resources,
        kind=declared_kind,
        publishes=bool(publishes),
        env=declared_env,
        emits_media=emits_media,
        child_calls=bool(child_calls),
        handles=declared_handles,
        ctx_type=ctx_type,
        delta_type=delta_type,
        delta_arms=delta_arms,
    )
    setattr(fn, ENTRYPOINT_ATTR, spec)
    return fn


__all__ = [
    "ENTRYPOINT_ATTR",
    "EntrypointDeclarationError",
    "EntrypointSpec",
    "SlotSpec",
    "entrypoint",
]
