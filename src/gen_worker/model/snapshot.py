"""``family_export_v1`` — what a family's DECLARATION-TIME export states.

torchcg G16 fixes the direction and this document is its Python half: typed
bindings generate from a declaration-time fake-tensor export, **not** from the
mint-emitted recipe. The reason is not stylistic — a recipe is emitted by a
mint, so recipe-sourced bindings would need a GPU compile before a new family
could type-check, and a family PR would be un-reviewable until somebody rented
a card.

So the flow is:

```text
declaration  --fake-tensor export-->  family_export_v1  --codegen-->  bindings
      |                                      |
      |                                      +-- DeclaredRunner rows
      |                                                  |
      +-- mint (a real compile, later) --> recipe_v1 --> Recipe.assert_declaration
```

The recipe is then the DRIFT ASSERTION (a mint compiled something the bindings
do not describe) and the adopt-time reference from a runner handle to a class
identity. It is never the binding source: a generator that read it instead
would have inverted the dependency and left nothing to compare against.

**What this document deliberately does not carry.** No ``class_hash``: a class
hash folds ``target`` and the traced graph's device placement, so a CPU
fake-tensor export could not produce the hash a real ``sm_86`` mint will, and
a document stating one would be stating a fact it cannot know. Identity here is
the ingress digest, which IS machine-independent — and it is exactly what
``Recipe.assert_declaration`` compares, because equal ingress digests imply
equal signatures (the signature is a projection of the ingress).

No checkpoint-level field appears either, for the same reason ``recipe_v1``
refuses them: the declaration is the class level, and the field sets below are
closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .._vendor.torchcg.ingress import CallIngress, IngressError
from .._vendor.torchcg.recipe import (
    BucketAxisName,
    CallSignature,
    DeclaredRunner,
    FamilyName,
    IngressDigest,
    LayoutContract,
    LoopKind,
    ParameterName,
    RecipeError,
    RepeatKind,
    RunnerName,
    SchedulerValue,
    SessionState,
    call_signature,
    parse_bucket_axis_name,
    parse_family_name,
    parse_ingress_digest,
    parse_layout_contract,
    parse_parameter_name,
    parse_runner_name,
    parse_scheduler_name,
)
from .errors import ModelError, ModelRefusal

#: The schema this reader implements. A snapshot stamping any other value is
#: refused, never best-effort decoded: a generator that silently accepted a
#: newer vocabulary would emit bindings the fleet disagrees with, and binding
#: disagreements are invisible until they arm wrong.
EXPORT_VERSION: Final = 1

_DIGEST_HEX: Final = 32


def _canonical(document: Mapping[str, Any]) -> bytes:
    """The exact bytes a digest is taken over — ``CallIngress``'s own rule."""

    try:
        return json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ModelError(
            ModelRefusal.SNAPSHOT_INVALID, f"export document is not finite JSON: {exc}"
        ) from exc


def _fields(
    kind: str,
    raw: object,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ModelError(ModelRefusal.SNAPSHOT_INVALID, f"{kind} must be an object")
    present = {str(name) for name in raw}
    if not required <= present or not present <= (required | optional):
        raise ModelError(
            ModelRefusal.SNAPSHOT_INVALID,
            f"{kind} fields must be exactly {sorted(required)!r}"
            + (f" plus optional {sorted(optional)!r}" if optional else ""),
        )
    return raw


def _parse(kind: str, value: object, parse: Any) -> str:
    try:
        return str(parse(value))
    except RecipeError as exc:
        raise ModelError(
            ModelRefusal.SNAPSHOT_INVALID, f"{kind} {value!r} is not canonical: {exc}"
        ) from exc


def _layout_handle(value: object) -> str:
    """One ie#740 tensor-layout handle out of a document, validated.

    NOT ``parse_layout_contract``, and the difference is pgw#1346's K4 in one
    line: a RUNNER's ``layouts`` are torchcg contract handles and live under
    the recipe's identifier grammar, while a MODEL's ``layouts`` are the hub's
    registered tensor-layout contracts — ``plain.bf16@1`` — which that grammar
    refuses on sight. Two axes, two vocabularies, and decoding one with the
    other's parser rejects every real declaration.
    """

    from ..models.tensor_layout_contract import (
        LayoutDeclarationError,
        validate_layout_handle,
    )

    try:
        return validate_layout_handle(value, where="eager model export")
    except LayoutDeclarationError as exc:
        raise ModelError(
            ModelRefusal.SNAPSHOT_INVALID, f"layout handle {value!r} is not canonical: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ExportedOutput:
    """One tensor a variant RETURNS, as the export recorded it.

    ``CallIngress`` is deliberately an INGRESS contract — it says nothing about
    what a graph produces, because admission never needed to. The FAKE backing
    does: a handler running hubless and GPU-less has to receive something with
    the right dtype and rank or every line after the call is untested. So the
    egress is recorded here, in this document, rather than smuggled into
    torchcg's contract where it would rekey every corpus for a reason torchcg
    does not have.

    A dimension is an ``int`` or a symbol NAME. A symbol is not a type
    (torchcg G5): the fake backing binds it from the call's own inputs and
    refuses when the call does not determine it.
    """

    dtype: str
    shape: tuple[int | str, ...]

    def __post_init__(self) -> None:
        if not self.dtype or self.dtype != self.dtype.strip():
            raise ModelError(ModelRefusal.SNAPSHOT_INVALID, "output dtype must be canonical")
        shape = tuple(self.shape)
        if any(
            (type(dim) is not int and not isinstance(dim, str))
            or (type(dim) is int and dim < 0)
            or (isinstance(dim, str) and (not dim or dim != dim.strip()))
            for dim in shape
        ):
            raise ModelError(ModelRefusal.SNAPSHOT_INVALID, "output shape is not canonical")
        object.__setattr__(self, "shape", shape)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def as_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}

    @classmethod
    def decode(cls, raw: object) -> ExportedOutput:
        row = _fields("output", raw, frozenset(("dtype", "shape")))
        shape = row["shape"]
        if not isinstance(shape, list):
            raise ModelError(ModelRefusal.SNAPSHOT_INVALID, "output shape must be an array")
        return cls(dtype=str(row["dtype"]), shape=tuple(shape))


@dataclass(frozen=True, slots=True)
class ExportedVariant:
    """One (bucket, layout) of one runner, and the exact call it was exported at."""

    bucket: tuple[tuple[BucketAxisName, int], ...]
    layout: LayoutContract
    ingress: CallIngress
    ingress_digest: IngressDigest
    outputs: tuple[ExportedOutput, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ingress, CallIngress):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                "a variant must carry its exact CallIngress v1 value",
            )
        if self.ingress.digest() != str(self.ingress_digest):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                "variant ingress_digest does not restate its own ingress declaration",
            )
        bucket = tuple(
            (BucketAxisName(_parse("bucket axis", name, parse_bucket_axis_name)), int(value))
            for name, value in self.bucket
        )
        names = tuple(name for name, _ in bucket)
        if names != tuple(sorted(set(names))):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "bucket axes must be sorted and unique"
            )
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(
            self, "layout", LayoutContract(_parse("layout", self.layout, parse_layout_contract))
        )

    @property
    def selector(self) -> tuple[tuple[tuple[BucketAxisName, int], ...], LayoutContract]:
        return (self.bucket, self.layout)

    @property
    def signature(self) -> CallSignature:
        return call_signature(self.ingress)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": {str(name): value for name, value in self.bucket},
            "ingress": self.ingress.as_dict(),
            "ingress_digest": str(self.ingress_digest),
            "layout": str(self.layout),
            "outputs": [row.as_dict() for row in self.outputs],
        }

    @classmethod
    def decode(cls, raw: object) -> ExportedVariant:
        row = _fields(
            "variant",
            raw,
            frozenset(("bucket", "ingress", "ingress_digest", "layout", "outputs")),
        )
        bucket = row["bucket"]
        outputs = row["outputs"]
        if not isinstance(bucket, Mapping) or not isinstance(outputs, list):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "variant bucket/outputs are malformed"
            )
        try:
            ingress = CallIngress.decode(row["ingress"])
        except IngressError as exc:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, f"variant call ingress is invalid: {exc}"
            ) from exc
        return cls(
            bucket=tuple(
                sorted(
                    (BucketAxisName(str(name)), int(value)) for name, value in bucket.items()
                )
            ),
            layout=LayoutContract(str(row["layout"])),
            ingress=ingress,
            ingress_digest=IngressDigest(
                _parse("ingress digest", row["ingress_digest"], parse_ingress_digest)
            ),
            outputs=tuple(ExportedOutput.decode(item) for item in outputs),
        )


def _signature_shape(variant: ExportedVariant) -> tuple[Any, ...]:
    """Everything a generated binding reads off a variant, and nothing else.

    Concrete dimensions and symbol bounds are deliberately excluded: they vary
    per variant, and a runner whose variants differ ONLY in them is still one
    binding (torchcg G2/G5). The egress is included at dtype-and-rank only, for
    the same reason and with the same cut: a runner whose 512 bucket returns
    one tensor and whose 1024 bucket returns two is not one callable.
    """

    ingress = variant.ingress
    return (
        ingress.parameters,
        ingress.flat_arity,
        ingress.excluded_inputs,
        tuple(
            (
                row.name,
                row.position,
                row.param,
                row.param_position,
                row.path,
                row.exported_name,
                row.dtype,
                len(row.shape),
            )
            for row in ingress.inputs
        ),
        tuple((row.dtype, row.rank) for row in variant.outputs),
    )


@dataclass(frozen=True, slots=True)
class ExportedRunner:
    """One runner handle and every variant the declaration exported for it."""

    name: RunnerName
    axes: tuple[BucketAxisName, ...]
    variants: tuple[ExportedVariant, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", RunnerName(_parse("runner name", self.name, parse_runner_name))
        )
        axes = tuple(
            BucketAxisName(_parse("bucket axis", name, parse_bucket_axis_name))
            for name in self.axes
        )
        if axes != tuple(sorted(set(axes))):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                f"runner {self.name!r} axes must be sorted and unique",
            )
        object.__setattr__(self, "axes", axes)
        if not self.variants:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, f"runner {self.name!r} exported no variant"
            )
        for variant in self.variants:
            if tuple(name for name, _ in variant.bucket) != axes:
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID,
                    f"runner {self.name!r} variants must pin exactly its declared axes",
                )
        selectors = [variant.selector for variant in self.variants]
        if len(set(selectors)) != len(selectors):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                f"runner {self.name!r} exported two variants for one bucket and layout",
            )
        if selectors != sorted(selectors):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                f"runner {self.name!r} variants must be sorted by bucket, then layout",
            )
        shapes = {_signature_shape(variant) for variant in self.variants}
        if len(shapes) != 1:
            # ONE runner is ONE binding (torchcg G2). Two variants that project
            # onto different signatures cannot share a generated callable, and
            # picking one of them would make the binding lie about the other.
            raise ModelError(
                ModelRefusal.SIGNATURE_DISAGREEMENT,
                f"runner {self.name!r} exported variants that disagree about the call "
                "signature; they may differ only in concrete dimensions and symbol bounds",
            )

    @property
    def signature(self) -> CallSignature:
        """The one signature every variant of this runner shares."""

        return self.variants[0].signature

    @property
    def layouts(self) -> tuple[LayoutContract, ...]:
        return tuple(sorted({variant.layout for variant in self.variants}))

    @property
    def declared(self) -> DeclaredRunner:
        """The row ``Recipe.assert_declaration`` compares a mint against."""

        return DeclaredRunner(
            name=self.name,
            ingress_digests=tuple(variant.ingress_digest for variant in self.variants),
        )

    def variant(
        self, bucket: Mapping[str, int], layout: str | None = None
    ) -> ExportedVariant:
        """Resolve one variant by EXACT bucket and layout. This never ranks.

        Choosing which bucket serves a live call is ``ingress_selection_v1``
        and choosing a layout is the hub's join with per-checkpoint layout
        metadata — two separate contracts. This is a lookup, and it refuses
        rather than guess.
        """

        wanted = tuple(
            sorted((BucketAxisName(str(name)), int(value)) for name, value in bucket.items())
        )
        layouts = self.layouts
        if layout is None:
            if len(layouts) != 1:
                raise ModelError(
                    ModelRefusal.CALL_INVALID,
                    f"runner {str(self.name)!r} has classes for "
                    f"{[str(item) for item in layouts]!r}; name the traced layout rather "
                    "than leaving it to a lookup",
                )
            chosen = layouts[0]
        else:
            chosen = LayoutContract(str(layout))
        for variant in self.variants:
            if variant.selector == (wanted, chosen):
                return variant
        raise ModelError(
            ModelRefusal.CALL_INVALID,
            f"runner {str(self.name)!r} declares no variant for bucket {dict(wanted)!r} "
            f"at layout {str(chosen)!r}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "axes": [str(name) for name in self.axes],
            "variants": [variant.as_dict() for variant in self.variants],
        }

    @classmethod
    def decode(cls, raw: object) -> ExportedRunner:
        row = _fields("runner", raw, frozenset(("name", "axes", "variants")))
        axes, variants = row["axes"], row["variants"]
        if not isinstance(axes, list) or not isinstance(variants, list):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "runner axes and variants must be arrays"
            )
        return cls(
            name=RunnerName(str(row["name"])),
            axes=tuple(BucketAxisName(str(name)) for name in axes),
            variants=tuple(ExportedVariant.decode(variant) for variant in variants),
        )


@dataclass(frozen=True, slots=True)
class ExportedStage:
    """One loop stage, restated so codegen and the recipe read one order."""

    runner: RunnerName
    repeat: RepeatKind = RepeatKind.ONCE
    parameter: ParameterName | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"runner": str(self.runner), "repeat": self.repeat.value}
        if self.parameter is not None:
            row["parameter"] = str(self.parameter)
        return row

    @classmethod
    def decode(cls, raw: object) -> ExportedStage:
        row = _fields(
            "loop stage", raw, frozenset(("runner", "repeat")), frozenset(("parameter",))
        )
        repeat = str(row["repeat"])
        if repeat not in tuple(kind.value for kind in RepeatKind):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, f"unknown loop repeat {repeat!r}"
            )
        parameter = row.get("parameter")
        return cls(
            runner=RunnerName(_parse("runner name", row["runner"], parse_runner_name)),
            repeat=RepeatKind(repeat),
            parameter=None
            if parameter is None
            else ParameterName(_parse("parameter", parameter, parse_parameter_name)),
        )


@dataclass(frozen=True, slots=True)
class ExportedLoop:
    """The declared loop, or the declaration that the host owns the iteration."""

    kind: LoopKind
    stages: tuple[ExportedStage, ...]
    session_state: SessionState = SessionState.NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "session_state": self.session_state.value,
            "stages": [stage.as_dict() for stage in self.stages],
        }

    @classmethod
    def decode(cls, raw: object) -> ExportedLoop:
        row = _fields("loop", raw, frozenset(("kind", "session_state", "stages")))
        kind, session_state, stages = str(row["kind"]), str(row["session_state"]), row["stages"]
        if kind not in tuple(item.value for item in LoopKind):
            raise ModelError(ModelRefusal.SNAPSHOT_INVALID, f"unknown loop kind {kind!r}")
        if session_state not in tuple(item.value for item in SessionState):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, f"unknown session state {session_state!r}"
            )
        if not isinstance(stages, list) or not stages:
            raise ModelError(ModelRefusal.SNAPSHOT_INVALID, "loop stages must be an array")
        return cls(
            kind=LoopKind(kind),
            stages=tuple(ExportedStage.decode(stage) for stage in stages),
            session_state=SessionState(session_state),
        )


@dataclass(frozen=True, slots=True)
class ExportedParameter:
    """One integer count a counted stage reads, with inclusive bounds."""

    name: ParameterName
    minimum: int
    maximum: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": str(self.name), "minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def decode(cls, raw: object) -> ExportedParameter:
        row = _fields("parameter", raw, frozenset(("name", "minimum", "maximum")))
        return cls(
            name=ParameterName(_parse("parameter", row["name"], parse_parameter_name)),
            minimum=int(row["minimum"]),
            maximum=int(row["maximum"]),
        )


@dataclass(frozen=True, slots=True)
class ExportedScheduler:
    """One SAMPLER, the scheduler kind it names, and that kind's block.

    ``sampler`` is the key a checkpoint is stamped with (``inst.tuned.
    scheduler``) and ``name`` is the scheduler KIND the host implements. They
    are two different vocabularies and conflating them is the defect pgw#1346
    K10 records: ``euler`` and ``euler_trailing`` are one kind under two
    spacings, and ``euler_a`` is a different kind entirely.
    """

    sampler: str
    name: str
    parameters: tuple[tuple[ParameterName, SchedulerValue], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampler": str(self.sampler),
            "name": str(self.name),
            "parameters": {str(name): value for name, value in self.parameters},
        }

    @classmethod
    def decode(cls, raw: object) -> ExportedScheduler:
        row = _fields("scheduler", raw, frozenset(("sampler", "name", "parameters")))
        parameters = row["parameters"]
        if not isinstance(parameters, Mapping):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "scheduler parameters must be an object"
            )
        return cls(
            sampler=_parse("sampler name", row["sampler"], parse_parameter_name),
            name=_parse("scheduler name", row["name"], parse_scheduler_name),
            parameters=tuple(
                sorted(
                    (
                        ParameterName(_parse("parameter", name, parse_parameter_name)),
                        value,
                    )
                    for name, value in parameters.items()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class TunedRef:
    """Where a family's tuned schema lives, so a generated module can import it.

    A REFERENCE, never a copy of the struct: duplicating the schema into the
    generated module would create a second definition that can disagree with
    the first, which is the same defect ``Recipe`` avoids by keeping its body
    out of ``endpoint.lock``.
    """

    module: str
    qualname: str

    def __post_init__(self) -> None:
        for what, value in (("module", self.module), ("qualname", self.qualname)):
            if not value or value != value.strip() or not all(
                part.isidentifier() for part in value.split(".")
            ):
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID,
                    f"tuned schema {what} {value!r} is not an importable dotted name",
                )

    def as_dict(self) -> dict[str, Any]:
        return {"module": self.module, "qualname": self.qualname}

    @classmethod
    def decode(cls, raw: object) -> TunedRef:
        row = _fields("tuned reference", raw, frozenset(("module", "qualname")))
        return cls(module=str(row["module"]), qualname=str(row["qualname"]))


@dataclass(frozen=True, slots=True)
class ModelExport:
    """One family's declaration-time export: the binding source, versioned.

    Every field here is CLASS-LEVEL. A weight set, a checkpoint ref, a tuned
    VALUE or a per-request default is unrepresentable, not merely discouraged:
    every object above has a closed field set, so a document carrying one is
    refused at decode.
    """

    family: FamilyName
    buckets: tuple[tuple[BucketAxisName, tuple[int, ...]], ...]
    runners: tuple[ExportedRunner, ...]
    loop: ExportedLoop
    tuned: TunedRef
    parameters: tuple[ExportedParameter, ...] = ()
    #: The scheduler SET, sorted by sampler name (pgw#1346 K10). Empty when the
    #: family declares no scheduler.
    schedulers: tuple[ExportedScheduler, ...] = ()
    lora_tuned: TunedRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "family", FamilyName(_parse("family name", self.family, parse_family_name))
        )
        axis_names = tuple(name for name, _ in self.buckets)
        if axis_names != tuple(sorted(set(axis_names))):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "bucket axes must be sorted and unique"
            )
        if not self.runners:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "an export must declare at least one runner"
            )
        runner_names = tuple(runner.name for runner in self.runners)
        if runner_names != tuple(sorted(set(runner_names))):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "runners must be sorted by name and unique"
            )
        axes = dict(self.buckets)
        for runner in self.runners:
            unknown = sorted(set(runner.axes) - set(axes))
            if unknown:
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID,
                    f"runner {str(runner.name)!r} buckets on undeclared axis "
                    f"{str(unknown[0])!r}",
                )
            expected = {
                tuple(zip(runner.axes, combination, strict=True))
                for combination in _product(tuple(axes[name] for name in runner.axes))
            }
            for layout in runner.layouts:
                built = {
                    variant.bucket for variant in runner.variants if variant.layout == layout
                }
                missing = sorted(expected - built)
                if missing:
                    raise ModelError(
                        ModelRefusal.BUCKET_COVERAGE_INCOMPLETE,
                        f"runner {str(runner.name)!r} at layout {str(layout)!r} exported no "
                        f"variant for bucket {dict(missing[0])!r}; a generated closed type "
                        "must be total for every layout it offers",
                    )
        staged = {stage.runner for stage in self.loop.stages}
        unknown_runner = sorted(staged - set(runner_names))
        if unknown_runner:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                f"loop stages undeclared runner {str(unknown_runner[0])!r}",
            )
        unused = sorted(set(runner_names) - staged)
        if unused:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                f"runner {str(unused[0])!r} is exported but no loop stage runs it",
            )
        samplers = tuple(row.sampler for row in self.schedulers)
        if samplers != tuple(sorted(set(samplers))):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "schedulers must be sorted by sampler and unique"
            )

    # ---------------------------------------------------------------- reading

    def runner(self, name: str) -> ExportedRunner:
        wanted = _parse("runner name", name, parse_runner_name)
        for runner in self.runners:
            if str(runner.name) == wanted:
                return runner
        raise ModelError(
            ModelRefusal.SNAPSHOT_INVALID, f"export declares no runner {wanted!r}"
        )

    def declared_runners(self) -> tuple[DeclaredRunner, ...]:
        """The rows a mint-emitted recipe is asserted against (torchcg G16)."""

        return tuple(runner.declared for runner in self.runners)

    @property
    def axis_values(self) -> dict[str, tuple[int, ...]]:
        return {str(name): values for name, values in self.buckets}

    # ---------------------------------------------------------------- writing

    def as_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "v": EXPORT_VERSION,
            "family": str(self.family),
            "buckets": [
                {"name": str(name), "values": list(values)} for name, values in self.buckets
            ],
            "runners": [runner.as_dict() for runner in self.runners],
            "loop": self.loop.as_dict(),
            "parameters": [parameter.as_dict() for parameter in self.parameters],
            "tuned": self.tuned.as_dict(),
        }
        if self.schedulers:
            document["schedulers"] = [row.as_dict() for row in self.schedulers]
        if self.lora_tuned is not None:
            document["lora_tuned"] = self.lora_tuned.as_dict()
        return document

    def canonical(self) -> bytes:
        return _canonical(self.as_dict())

    def digest(self) -> str:
        """The 32-hex machine-independent digest a generated module pins."""

        return hashlib.sha256(self.canonical()).hexdigest()[:_DIGEST_HEX]

    def dumps(self) -> str:
        """The exact committed file body: canonical JSON plus a trailing newline."""

        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"

    # ---------------------------------------------------------------- decoding

    @classmethod
    def decode(cls, raw: object) -> ModelExport:
        row = _fields(
            "family export",
            raw,
            frozenset(("v", "family", "buckets", "runners", "loop", "parameters", "tuned")),
            frozenset(("schedulers", "lora_tuned")),
        )
        version = row["v"]
        if type(version) is not int or version != EXPORT_VERSION:
            raise ModelError(
                ModelRefusal.SNAPSHOT_VERSION_UNSUPPORTED,
                f"family export v={version!r} is not v{EXPORT_VERSION}; this reader has "
                "one version",
            )
        for name in ("buckets", "runners", "parameters"):
            if not isinstance(row[name], list):
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID, f"family export {name} must be an array"
                )
        buckets: list[tuple[BucketAxisName, tuple[int, ...]]] = []
        for axis in row["buckets"]:
            entry = _fields("bucket axis", axis, frozenset(("name", "values")))
            values = entry["values"]
            if not isinstance(values, list) or not values:
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID, "bucket values must be a non-empty array"
                )
            buckets.append(
                (
                    BucketAxisName(_parse("bucket axis", entry["name"], parse_bucket_axis_name)),
                    tuple(int(value) for value in values),
                )
            )
        schedulers = row.get("schedulers")
        if schedulers is not None and not isinstance(schedulers, list):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, "family export schedulers must be an array"
            )
        lora = row.get("lora_tuned")
        return cls(
            family=FamilyName(str(row["family"])),
            buckets=tuple(buckets),
            runners=tuple(ExportedRunner.decode(runner) for runner in row["runners"]),
            loop=ExportedLoop.decode(row["loop"]),
            tuned=TunedRef.decode(row["tuned"]),
            parameters=tuple(ExportedParameter.decode(item) for item in row["parameters"]),
            schedulers=()
            if schedulers is None
            else tuple(ExportedScheduler.decode(row) for row in schedulers),
            lora_tuned=None if lora is None else TunedRef.decode(lora),
        )

    @classmethod
    def loads(cls, payload: bytes | str) -> ModelExport:
        try:
            document = json.loads(payload)
        except ValueError as exc:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, f"family export is not JSON: {exc}"
            ) from exc
        return cls.decode(document)


#: The schema :class:`EagerExport` implements, versioned separately from
#: ``family_export_v1`` because it is a different document with a different
#: field set — not a degenerate case of one.
EAGER_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class EagerExport:
    """``eager_model_v1`` — what an EAGER declaration states, with no trace.

    pgw#1346 B5. ``family_export_v1`` is the record of a fake-tensor TRACE: its
    runners, buckets, loop and signatures all exist because something was
    exported. An eager-only :class:`~gen_worker.model.spec.ModelSpec` has none
    of that by definition — a model served by an external binary (vLLM,
    llama-server) or a runtime that loads itself has no graph to trace, and the
    F3 ruling makes that a PERMANENT state rather than a waiting room. Widening
    ``ModelExport`` to admit zero runners would have deleted the invariants
    that make a generated ``Literal`` exhaustive, for the benefit of documents
    that carry no ``Literal`` at all.

    So this is a second, smaller document with its own closed field set, and
    everything downstream is unchanged in shape: the declaration exports it
    (no torch, nothing to trace), codegen is a pure function of it, both halves
    are committed, and the fence is the same byte comparison.

    What it carries is exactly what an eager declaration can honestly state:
    the family handle, the tuned schema(s) when the model has an inference
    vocabulary, and the three layout axes — which are the ie#740 floors, and
    the reason this document exists at all rather than the declaration being
    dropped on the floor at migration time.
    """

    family: FamilyName
    tuned: TunedRef | None = None
    lora_tuned: TunedRef | None = None
    layouts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    layouts_undeclarable: str = ""
    layout_requirements: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "family", FamilyName(_parse("family name", self.family, parse_family_name))
        )
        components = tuple(component for component, _ in self.layouts)
        if components != tuple(sorted(set(components))):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                "eager export layout components must be sorted and unique",
            )
        for component, handles in self.layouts:
            if not handles or handles != tuple(sorted(set(handles))):
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID,
                    f"eager export layouts[{component!r}] must be a non-empty sorted "
                    "unique handle set",
                )
        if self.layouts and self.layouts_undeclarable:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                "eager export declares both layouts and layouts_undeclarable; a model "
                "either names the contracts its code executes or says why none is "
                "nameable, never both",
            )
        guarded = tuple(handle for handle, _ in self.layout_requirements)
        if guarded != tuple(sorted(set(guarded))):
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                "eager export layout_requirements must be sorted by handle and unique",
            )
        accepted = {handle for _, handles in self.layouts for handle in handles}
        unknown = sorted(set(guarded) - accepted)
        if unknown:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                f"eager export requires {unknown[0]!r}, which it does not accept; a "
                "requirement over nothing is never checked",
            )
        if self.lora_tuned is not None and self.tuned is None:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID,
                "eager export carries a lora tuned schema with no base one; a LoRA "
                "vocabulary refines a base one and there is none here to refine",
            )

    # ---------------------------------------------------------------- writing

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": EAGER_VERSION,
            "family": str(self.family),
            "layouts": [
                {"component": component, "contracts": list(handles)}
                for component, handles in self.layouts
            ],
            "layouts_undeclarable": self.layouts_undeclarable,
            "layout_requirements": [
                {"contract": handle, "minimum": minimum}
                for handle, minimum in self.layout_requirements
            ],
            "tuned": None if self.tuned is None else self.tuned.as_dict(),
            "lora_tuned": None if self.lora_tuned is None else self.lora_tuned.as_dict(),
        }

    def canonical(self) -> bytes:
        return _canonical(self.as_dict())

    def digest(self) -> str:
        """The 32-hex machine-independent digest a generated module pins."""

        return hashlib.sha256(self.canonical()).hexdigest()[:_DIGEST_HEX]

    def dumps(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"

    # ---------------------------------------------------------------- decoding

    @classmethod
    def decode(cls, raw: object) -> EagerExport:
        row = _fields(
            "eager model export",
            raw,
            frozenset(
                (
                    "v",
                    "family",
                    "layouts",
                    "layouts_undeclarable",
                    "layout_requirements",
                    "tuned",
                    "lora_tuned",
                )
            ),
        )
        version = row["v"]
        if type(version) is not int or version != EAGER_VERSION:
            raise ModelError(
                ModelRefusal.SNAPSHOT_VERSION_UNSUPPORTED,
                f"eager model export v={version!r} is not v{EAGER_VERSION}; this reader "
                "has one version",
            )
        for name in ("layouts", "layout_requirements"):
            if not isinstance(row[name], list):
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID,
                    f"eager model export {name} must be an array",
                )
        layouts: list[tuple[str, tuple[str, ...]]] = []
        for entry in row["layouts"]:
            item = _fields("eager layout demand", entry, frozenset(("component", "contracts")))
            contracts = item["contracts"]
            if not isinstance(contracts, list):
                raise ModelError(
                    ModelRefusal.SNAPSHOT_INVALID, "eager layout contracts must be an array"
                )
            layouts.append(
                (
                    str(item["component"]),
                    tuple(_layout_handle(handle) for handle in contracts),
                )
            )
        requirements: list[tuple[str, str]] = []
        for entry in row["layout_requirements"]:
            item = _fields(
                "eager layout requirement", entry, frozenset(("contract", "minimum"))
            )
            requirements.append((_layout_handle(item["contract"]), str(item["minimum"])))
        tuned = row["tuned"]
        lora = row["lora_tuned"]
        return cls(
            family=FamilyName(str(row["family"])),
            tuned=None if tuned is None else TunedRef.decode(tuned),
            lora_tuned=None if lora is None else TunedRef.decode(lora),
            layouts=tuple(layouts),
            layouts_undeclarable=str(row["layouts_undeclarable"]),
            layout_requirements=tuple(requirements),
        )

    @classmethod
    def loads(cls, payload: bytes | str) -> EagerExport:
        try:
            document = json.loads(payload)
        except ValueError as exc:
            raise ModelError(
                ModelRefusal.SNAPSHOT_INVALID, f"eager model export is not JSON: {exc}"
            ) from exc
        return cls.decode(document)


def _product(rows: Sequence[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    out: tuple[tuple[int, ...], ...] = ((),)
    for row in rows:
        out = tuple(prefix + (value,) for prefix in out for value in row)
    return out


__all__ = [
    "EAGER_VERSION",
    "EXPORT_VERSION",
    "EagerExport",
    "ExportedLoop",
    "ExportedOutput",
    "ExportedParameter",
    "ExportedRunner",
    "ExportedScheduler",
    "ExportedStage",
    "ExportedVariant",
    "ModelExport",
    "TunedRef",
]
