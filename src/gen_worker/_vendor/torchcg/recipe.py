"""The closed v1 vocabulary for one family's compiled composition.

A recipe NAMES a family's composition: which graph specializations an endpoint's
pipeline is made of, the loop between them, and the scheduler block that loop
runs under.  It does not program it.  Payload parsing, CFG policy, residency,
retries, telemetry, ingress ranking, and eager fallback are host code by
definition, not vocabulary extensions.

The document is machine-independent on purpose: it pins each specialization by its
16-hex specialization hash and its exact ``CallIngress`` v1 declaration, never by a
``cg-key-v2`` value.  The key is folded at adopt time from the pod's own
``sm``/``toolchain`` facts and its compile policy (tcg#80), so one recipe
digest is valid on every SKU.

It is CLASS-LEVEL and therefore checkpoint-free, exactly as the key is: weight
sets, checkpoint refs, tuned values, and per-request defaults are
unrepresentable here because every object below has a closed field set.  A
family is the graph level; an instance is family x weight-set; a request is
neither.  ``docs/RECIPE.md`` states the generation requirements that follow from
that, including why a generated family specialization is a TYPE whose callables are
reached through an instance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import Any, Final, NewType, Protocol

from .identity import CompiledGraphKey, from_axes, toolchain_axis_digest
from .ingress import CallIngress, CallInput, IngressError, PathStep

RECIPE_VERSION: Final = 1
_DIGEST_HEX: Final = 32
_SPECIALIZATION_HASH_HEX: Final = 16
_IDENTIFIER_MAX: Final = 48
_HEX: Final = frozenset("0123456789abcdef")

#: The one identifier grammar every generated symbol has to survive.  A recipe
#: refuses what a binding generator would have to mangle, so codegen never owns
#: an escaping rule: family, runner, bucket-axis, call-parameter, scheduler, and
#: scheduler-parameter names are all `[a-z][a-z0-9_]*`, 1..48 bytes.
IDENTIFIER_GRAMMAR: Final = "[a-z][a-z0-9_]*, 1..48 bytes"

#: Reserved in Python or Rust, so no generator can emit them as a symbol.  This
#: list is FROZEN DATA, not `keyword.kwlist`: a corpus whose refusals move with
#: the interpreter running it is not a contract.  It is the union of Python 3.13
#: keywords and soft keywords with the Rust 2021 strict, reserved, and weak
#: keywords.
RESERVED_IDENTIFIERS: Final[frozenset[str]] = frozenset(
    (
        # Python keywords and soft keywords.
        "and", "as", "assert", "async", "await", "break", "case", "class",
        "continue", "def", "del", "elif", "else", "except", "false", "finally",
        "for", "from", "global", "if", "import", "in", "is", "lambda", "match",
        "none", "nonlocal", "not", "or", "pass", "raise", "return", "true",
        "try", "type", "while", "with", "yield",
        # Rust 2021 strict, reserved, and weak keywords.
        "abstract", "become", "box", "const", "crate", "do", "dyn", "enum",
        "extern", "final", "fn", "impl", "let", "loop", "macro", "macro_rules",
        "mod", "move", "mut", "override", "priv", "pub", "ref", "self", "static",
        "struct", "super", "trait", "typeof", "union", "unsafe", "unsized",
        "use", "virtual", "where",
    )
)

FamilyName = NewType("FamilyName", str)
RunnerName = NewType("RunnerName", str)
BucketAxisName = NewType("BucketAxisName", str)
ParameterName = NewType("ParameterName", str)
SchedulerName = NewType("SchedulerName", str)
LayoutContract = NewType("LayoutContract", str)
GraphSpecializationHash = NewType("GraphSpecializationHash", str)
IngressDigest = NewType("IngressDigest", str)
RecipeDigest = NewType("RecipeDigest", str)

SchedulerValue = bool | int | float | str


class RecipeRefusal(StrEnum):
    """The closed set of reasons a v1 recipe document is refused."""

    RECIPE_VERSION_UNSUPPORTED = "recipe_version_unsupported"
    RECIPE_FIELDS_INVALID = "recipe_fields_invalid"
    IDENTIFIER_INVALID = "identifier_invalid"
    IDENTIFIER_RESERVED = "identifier_reserved"
    DIGEST_INVALID = "digest_invalid"
    BUCKET_AXIS_INVALID = "bucket_axis_invalid"
    BUCKET_INVALID = "bucket_invalid"
    BUCKET_COVERAGE_INCOMPLETE = "bucket_coverage_incomplete"
    LAYOUT_INVALID = "layout_invalid"
    DECLARATION_DRIFT = "declaration_drift"
    INGRESS_INVALID = "ingress_invalid"
    INGRESS_DIGEST_MISMATCH = "ingress_digest_mismatch"
    SIGNATURE_DISAGREEMENT = "signature_disagreement"
    RUNNER_INVALID = "runner_invalid"
    RUNNER_UNKNOWN = "runner_unknown"
    RUNNER_UNUSED = "runner_unused"
    LOOP_INVALID = "loop_invalid"
    PARAMETER_INVALID = "parameter_invalid"
    PARAMETER_UNKNOWN = "parameter_unknown"
    PARAMETER_UNUSED = "parameter_unused"
    SCHEDULER_INVALID = "scheduler_invalid"
    REFERENCE_MISMATCH = "reference_mismatch"


class RecipeError(ValueError):
    """A recipe document is incomplete, noncanonical, or ungeneratable."""

    def __init__(self, reason: RecipeRefusal, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason.value}: {detail}")


class ParameterKind(StrEnum):
    """How one call parameter's tensor leaves are shaped in the unflattened call."""

    TENSOR = "tensor"
    MAPPING = "mapping"
    SEQUENCE = "sequence"
    ABSENT = "absent"


class RepeatKind(StrEnum):
    """How many times one loop stage runs.  There is no third form."""

    ONCE = "once"
    COUNTED = "counted"


class LoopKind(StrEnum):
    """Whether the recipe states the loop, or states that it cannot.

    ``staged`` is a composition the vocabulary fully describes: an ordered list
    of stages, each running once or a declared number of times.  ``host`` is the
    data-dependent loop of an autoregressive family — the recipe names the
    per-step graph specializations and who owns session state, and says outright that
    the iteration is host-owned.  The vocabulary never pretends to describe a
    loop it cannot express.
    """

    STAGED = "staged"
    HOST = "host"


class SessionState(StrEnum):
    """Who owns the state threaded between steps of a loop."""

    NONE = "none"
    HOST = "host"
    GRAPH = "graph"


class RuntimeAxes(Protocol):
    """The two machine axes a recipe deliberately does not carry.

    ``torchcg.RuntimeCompatibility`` satisfies this structurally; the protocol
    exists so this module never imports it, and therefore never imports Torch.
    """

    @property
    def sm(self) -> str: ...

    @property
    def toolchain(self) -> Mapping[str, str]: ...


def _identifier(kind: str, value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > _IDENTIFIER_MAX:
        raise RecipeError(
            RecipeRefusal.IDENTIFIER_INVALID,
            f"{kind} must be a non-empty string of at most {_IDENTIFIER_MAX} bytes",
        )
    if not value[0].islower() or not value[0].isascii() or not value[0].isalpha():
        raise RecipeError(
            RecipeRefusal.IDENTIFIER_INVALID,
            f"{kind} {value!r} must match {IDENTIFIER_GRAMMAR}",
        )
    if any(not (character.isascii() and (character.islower() or character.isdigit()))
           and character != "_" for character in value):
        raise RecipeError(
            RecipeRefusal.IDENTIFIER_INVALID,
            f"{kind} {value!r} must match {IDENTIFIER_GRAMMAR}",
        )
    if value in RESERVED_IDENTIFIERS:
        raise RecipeError(
            RecipeRefusal.IDENTIFIER_RESERVED,
            f"{kind} {value!r} is reserved in Python or Rust and cannot be generated",
        )
    return value


def parse_family_name(value: object) -> FamilyName:
    """Parse an author-facing family handle."""

    return FamilyName(_identifier("family name", value))


def parse_runner_name(value: object) -> RunnerName:
    """Parse an author-facing runner handle."""

    return RunnerName(_identifier("runner name", value))


def parse_bucket_axis_name(value: object) -> BucketAxisName:
    """Parse a bucket-axis handle."""

    return BucketAxisName(_identifier("bucket axis name", value))


def parse_parameter_name(value: object) -> ParameterName:
    """Parse a call, loop, or scheduler parameter handle."""

    return ParameterName(_identifier("parameter name", value))


def parse_scheduler_name(value: object) -> SchedulerName:
    """Parse a scheduler handle."""

    return SchedulerName(_identifier("scheduler name", value))


def parse_layout_contract(value: object) -> LayoutContract:
    """Parse the tensor-layout contract token a graph specialization was traced against.

    The token set is the platform's tensor-layout vocabulary (DESIGN-RULINGS
    §1.32/§1.33). This contract records which one a specialization was traced against and
    never enumerates, interprets, or forks it: the COMPATIBLE / CONVERTIBLE /
    PRODUCIBLE / INCOMPATIBLE verdict is the hub's join of this document with
    per-checkpoint layout metadata, which is a SEPARATE document.
    """

    return LayoutContract(_identifier("layout contract", value))


def _hex(kind: str, value: object, width: int) -> str:
    if not isinstance(value, str) or len(value) != width or any(
        character not in _HEX for character in value
    ):
        raise RecipeError(
            RecipeRefusal.DIGEST_INVALID,
            f"{kind} must be {width} lowercase hexadecimal characters",
        )
    return value


def parse_specialization_hash(value: object) -> GraphSpecializationHash:
    """Parse the 16-hex graph-specialization hash that is the key's ``graph`` axis."""

    return GraphSpecializationHash(_hex("specialization hash", value, _SPECIALIZATION_HASH_HEX))


def parse_ingress_digest(value: object) -> IngressDigest:
    """Parse the 32-hex CallIngress v1 digest."""

    return IngressDigest(_hex("ingress digest", value, _DIGEST_HEX))


def parse_recipe_digest(value: object) -> RecipeDigest:
    """Parse the 32-hex recipe-document digest."""

    return RecipeDigest(_hex("recipe digest", value, _DIGEST_HEX))


def _canonical(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:  # pragma: no cover - constructors refuse first
        raise RecipeError(
            RecipeRefusal.RECIPE_FIELDS_INVALID, f"recipe document is not finite JSON: {exc}"
        ) from exc


def _fields(
    kind: str,
    raw: object,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise RecipeError(RecipeRefusal.RECIPE_FIELDS_INVALID, f"{kind} must be an object")
    present = {str(name) for name in raw}
    if not required <= present or not present <= (required | optional):
        raise RecipeError(
            RecipeRefusal.RECIPE_FIELDS_INVALID,
            f"{kind} fields must be exactly {sorted(required)!r}"
            + (f" plus optional {sorted(optional)!r}" if optional else ""),
        )
    return raw


@dataclass(frozen=True, slots=True)
class SignatureLeaf:
    """One tensor feed of a generated call, identified exactly as the ingress does."""

    name: str
    position: int
    path: tuple[PathStep, ...]
    dtype: str
    rank: int


@dataclass(frozen=True, slots=True)
class SignatureParameter:
    """One parameter of a generated call, in the ordered parameter axis."""

    name: ParameterName
    kind: ParameterKind
    leaves: tuple[SignatureLeaf, ...]


@dataclass(frozen=True, slots=True)
class CallSignature:
    """The generation-facing projection of one CallIngress.

    It is a pure function of the ingress, so a generated binding and the
    artifact's admission expectation are two readings of one identity.  Shapes
    are deliberately absent: concrete and symbolic dimensions vary per variant,
    while rank, dtype, order, and leaf identity do not.
    """

    parameters: tuple[SignatureParameter, ...]
    flat_arity: int


def call_signature(ingress: CallIngress) -> CallSignature:
    """Project one ingress onto the facts a binding generator needs."""

    rows: dict[str, list[CallInput]] = {name: [] for name in ingress.parameters}
    for row in ingress.inputs:
        rows[row.param].append(row)
    parameters: list[SignatureParameter] = []
    for name in ingress.parameters:
        leaves = rows[name]
        if not leaves:
            kind = ParameterKind.ABSENT
        elif len(leaves) == 1 and not leaves[0].path:
            kind = ParameterKind.TENSOR
        elif all(leaf.path and isinstance(leaf.path[0], str) for leaf in leaves):
            kind = ParameterKind.MAPPING
        elif all(leaf.path and type(leaf.path[0]) is int for leaf in leaves):
            kind = ParameterKind.SEQUENCE
        else:
            raise RecipeError(
                RecipeRefusal.INGRESS_INVALID,
                f"parameter {name!r} mixes container kinds and has no generatable type",
            )
        parameters.append(
            SignatureParameter(
                name=parse_parameter_name(name),
                kind=kind,
                leaves=tuple(
                    SignatureLeaf(
                        name=leaf.name,
                        position=leaf.position,
                        path=leaf.path,
                        dtype=leaf.dtype,
                        rank=len(leaf.shape),
                    )
                    for leaf in leaves
                ),
            )
        )
    return CallSignature(parameters=tuple(parameters), flat_arity=ingress.flat_arity)


@dataclass(frozen=True, slots=True)
class BucketAxis:
    """One family-wide shape axis and the closed set of values it may take."""

    name: BucketAxisName
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", parse_bucket_axis_name(self.name))
        if not isinstance(self.values, tuple) or not self.values or any(
            type(value) is not int or value <= 0 for value in self.values
        ):
            raise RecipeError(
                RecipeRefusal.BUCKET_AXIS_INVALID,
                f"bucket axis {self.name!r} must declare positive integer values",
            )
        if self.values != tuple(sorted(set(self.values))):
            raise RecipeError(
                RecipeRefusal.BUCKET_AXIS_INVALID,
                f"bucket axis {self.name!r} values must be sorted and unique",
            )

    def as_dict(self) -> dict[str, Any]:
        return {"name": str(self.name), "values": list(self.values)}

    @classmethod
    def decode(cls, raw: object) -> BucketAxis:
        row = _fields("bucket axis", raw, frozenset(("name", "values")))
        values = row["values"]
        if not isinstance(values, list):
            raise RecipeError(RecipeRefusal.BUCKET_AXIS_INVALID, "bucket values must be an array")
        return cls(name=parse_bucket_axis_name(row["name"]), values=tuple(values))


@dataclass(frozen=True, slots=True)
class GraphSpecializationVariant:
    """One graph specialization of one runner, pinned by identity, layout, and bucket.

    ``layout`` names the tensor-layout contract this class was TRACED against.
    An fp8-rowwise trace and a bf16 trace are different graphs, so layout is an
    axis of the specialization row, not a property of the weights bound to it later. The
    compiled backing accepts exactly its traced layout; an eager backing follows
    the fit ladder, which is host policy.
    """

    specialization_hash: GraphSpecializationHash
    ingress_digest: IngressDigest
    ingress: CallIngress
    layout: LayoutContract
    bucket: tuple[tuple[BucketAxisName, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "specialization_hash", parse_specialization_hash(self.specialization_hash)
        )
        object.__setattr__(self, "layout", parse_layout_contract(self.layout))
        object.__setattr__(self, "ingress_digest", parse_ingress_digest(self.ingress_digest))
        if not isinstance(self.ingress, CallIngress):
            raise RecipeError(
                RecipeRefusal.INGRESS_INVALID, "a variant must carry its exact CallIngress v1 value"
            )
        if self.ingress.digest() != self.ingress_digest:
            raise RecipeError(
                RecipeRefusal.INGRESS_DIGEST_MISMATCH,
                "variant ingress_digest does not restate its own ingress declaration",
            )
        bucket = tuple(
            (parse_bucket_axis_name(name), value) for name, value in self.bucket
        )
        if any(type(value) is not int for _, value in bucket):
            raise RecipeError(
                RecipeRefusal.BUCKET_INVALID, "bucket values must be integers"
            )
        names = tuple(name for name, _ in bucket)
        if names != tuple(sorted(set(names))):
            raise RecipeError(
                RecipeRefusal.BUCKET_INVALID, "bucket axes must be sorted and unique"
            )
        object.__setattr__(self, "bucket", bucket)

    @property
    def signature(self) -> CallSignature:
        return call_signature(self.ingress)

    def key(self, runtime: RuntimeAxes) -> CompiledGraphKey:
        """Fold this pinned specialization with the pod's own axes into the exact key."""

        # tcg#80. `compiler` imports no torch at module scope, so reading the
        # compile policy here keeps this module's no-Torch property intact --
        # and the policy is a pod axis exactly as sm and toolchain are: the
        # recipe pins the graph, the pod states what its compiler was told.
        from .compiler import compile_policy_digest

        return from_axes(
            {
                "compile_policy": compile_policy_digest(),
                "graph": str(self.specialization_hash),
                "sm": runtime.sm,
                "toolchain": toolchain_axis_digest(runtime.toolchain),
            }
        )

    @property
    def selector(self) -> tuple[tuple[tuple[BucketAxisName, int], ...], LayoutContract]:
        """The exact pair a lookup resolves by."""

        return (self.bucket, self.layout)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": {str(name): value for name, value in self.bucket},
            "specialization_hash": str(self.specialization_hash),
            "ingress": self.ingress.as_dict(),
            "ingress_digest": str(self.ingress_digest),
            "layout": str(self.layout),
        }

    @classmethod
    def decode(cls, raw: object) -> GraphSpecializationVariant:
        row = _fields(
            "variant",
            raw,
            frozenset(("bucket", "specialization_hash", "ingress", "ingress_digest", "layout")),
        )
        bucket = row["bucket"]
        if not isinstance(bucket, Mapping):
            raise RecipeError(RecipeRefusal.BUCKET_INVALID, "variant bucket must be an object")
        try:
            ingress = CallIngress.decode(row["ingress"])
        except IngressError as exc:
            raise RecipeError(
                RecipeRefusal.INGRESS_INVALID, f"variant call ingress is invalid: {exc}"
            ) from exc
        return cls(
            specialization_hash=parse_specialization_hash(row["specialization_hash"]),
            ingress_digest=parse_ingress_digest(row["ingress_digest"]),
            ingress=ingress,
            layout=parse_layout_contract(row["layout"]),
            bucket=tuple(
                sorted(
                    (parse_bucket_axis_name(name), value) for name, value in bucket.items()
                )
            ),
        )


def _signature_shape(ingress: CallIngress) -> tuple[Any, ...]:
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
    )


@dataclass(frozen=True, slots=True)
class RecipeRunner:
    """One author-facing runner handle and the bucketed specializations behind it."""

    name: RunnerName
    axes: tuple[BucketAxisName, ...]
    variants: tuple[GraphSpecializationVariant, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", parse_runner_name(self.name))
        axes = tuple(parse_bucket_axis_name(name) for name in self.axes)
        if axes != tuple(sorted(set(axes))):
            raise RecipeError(
                RecipeRefusal.RUNNER_INVALID,
                f"runner {self.name!r} axes must be sorted and unique",
            )
        object.__setattr__(self, "axes", axes)
        if not isinstance(self.variants, tuple) or not self.variants:
            raise RecipeError(
                RecipeRefusal.RUNNER_INVALID,
                f"runner {self.name!r} declares no graph specialization",
            )
        buckets = [tuple(name for name, _ in variant.bucket) for variant in self.variants]
        if any(names != axes for names in buckets):
            raise RecipeError(
                RecipeRefusal.BUCKET_INVALID,
                f"runner {self.name!r} variants must pin exactly its declared axes",
            )
        pinned = [variant.selector for variant in self.variants]
        if len(set(pinned)) != len(pinned):
            raise RecipeError(
                RecipeRefusal.BUCKET_INVALID,
                f"runner {self.name!r} declares two variants for one bucket and layout",
            )
        if pinned != sorted(pinned):
            raise RecipeError(
                RecipeRefusal.BUCKET_INVALID,
                f"runner {self.name!r} variants must be sorted by bucket, then layout",
            )
        shapes = {_signature_shape(variant.ingress) for variant in self.variants}
        if len(shapes) != 1:
            raise RecipeError(
                RecipeRefusal.SIGNATURE_DISAGREEMENT,
                f"runner {self.name!r} variants disagree about the call signature; "
                "they may differ only in concrete dimensions and symbol bounds",
            )

    @property
    def signature(self) -> CallSignature:
        """The one signature every variant of this runner shares.

        Across layouts too: one runner is one generated binding, so a trace whose
        layout changes the CALL (not just the weights) is a different runner.
        """

        return self.variants[0].signature

    @property
    def layouts(self) -> tuple[LayoutContract, ...]:
        """The tensor-layout contracts this runner has specializations for."""

        return tuple(sorted({variant.layout for variant in self.variants}))

    def variant(
        self,
        bucket: Mapping[BucketAxisName, int],
        layout: LayoutContract | None = None,
    ) -> GraphSpecializationVariant:
        """Resolve one variant by EXACT bucket and layout.  This never ranks.

        Choosing a bucket for a live call is ingress selection, a separate
        versioned contract (``ingress_selection_v1``); choosing a layout is the
        hub's join of this document with per-checkpoint layout metadata. This is
        a lookup, and it refuses rather than guess.
        """

        wanted = tuple(sorted((parse_bucket_axis_name(k), v) for k, v in bucket.items()))
        layouts = self.layouts
        if layout is None:
            if len(layouts) != 1:
                raise RecipeError(
                    RecipeRefusal.LAYOUT_INVALID,
                    f"runner {self.name!r} has specializations for "
                    f"{[str(item) for item in layouts]!r}; "
                    "name the traced layout rather than leaving it to a lookup",
                )
            layout = layouts[0]
        selector = (wanted, parse_layout_contract(layout))
        for variant in self.variants:
            if variant.selector == selector:
                return variant
        raise RecipeError(
            RecipeRefusal.BUCKET_INVALID,
            f"runner {self.name!r} declares no variant for bucket {dict(wanted)!r} "
            f"at layout {str(layout)!r}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "axes": [str(name) for name in self.axes],
            "variants": [variant.as_dict() for variant in self.variants],
        }

    @classmethod
    def decode(cls, raw: object) -> RecipeRunner:
        row = _fields("runner", raw, frozenset(("name", "axes", "variants")))
        axes, variants = row["axes"], row["variants"]
        if not isinstance(axes, list) or not isinstance(variants, list):
            raise RecipeError(
                RecipeRefusal.RUNNER_INVALID, "runner axes and variants must be arrays"
            )
        return cls(
            name=parse_runner_name(row["name"]),
            axes=tuple(parse_bucket_axis_name(name) for name in axes),
            variants=tuple(GraphSpecializationVariant.decode(variant) for variant in variants),
        )


@dataclass(frozen=True, slots=True)
class LoopStep:
    """One stage of the composition: a runner, and how many times it runs."""

    runner: RunnerName
    repeat: RepeatKind = RepeatKind.ONCE
    parameter: ParameterName | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "runner", parse_runner_name(self.runner))
        if not isinstance(self.repeat, RepeatKind):
            raise RecipeError(
                RecipeRefusal.LOOP_INVALID,
                f"loop step repeat must be one of {[kind.value for kind in RepeatKind]!r}",
            )
        if self.repeat is RepeatKind.COUNTED:
            if self.parameter is None:
                raise RecipeError(
                    RecipeRefusal.LOOP_INVALID,
                    f"counted stage {self.runner!r} names no count parameter",
                )
            object.__setattr__(self, "parameter", parse_parameter_name(self.parameter))
        elif self.parameter is not None:
            raise RecipeError(
                RecipeRefusal.LOOP_INVALID,
                f"stage {self.runner!r} runs once and cannot carry a count parameter",
            )

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"runner": str(self.runner), "repeat": self.repeat.value}
        if self.parameter is not None:
            row["parameter"] = str(self.parameter)
        return row

    @classmethod
    def decode(cls, raw: object) -> LoopStep:
        row = _fields(
            "loop step", raw, frozenset(("runner", "repeat")), frozenset(("parameter",))
        )
        repeat = row["repeat"]
        if repeat not in tuple(kind.value for kind in RepeatKind):
            raise RecipeError(
                RecipeRefusal.LOOP_INVALID, f"unknown loop repeat {repeat!r}"
            )
        parameter = row.get("parameter")
        return cls(
            runner=parse_runner_name(row["runner"]),
            repeat=RepeatKind(repeat),
            parameter=None if parameter is None else parse_parameter_name(parameter),
        )


@dataclass(frozen=True, slots=True)
class Loop:
    """The loop between a family's specializations, or the declaration that it is host-owned.

    An autoregressive family's iteration is data-dependent: it runs until the
    model says stop, and no count in a document can say that.  ``kind: host``
    is how the vocabulary says so out loud.  The recipe still states everything
    it CAN: the per-step graph specializations in order, and who owns the state threaded
    between steps.  What it will not do is pretend a host loop is a counted one.
    """

    kind: LoopKind
    stages: tuple[LoopStep, ...]
    session_state: SessionState = SessionState.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LoopKind) or not isinstance(
            self.session_state, SessionState
        ):
            raise RecipeError(
                RecipeRefusal.LOOP_INVALID, "loop kind and session_state must be v1 values"
            )
        if not isinstance(self.stages, tuple) or not self.stages:
            raise RecipeError(RecipeRefusal.LOOP_INVALID, "a loop must declare its stages")
        if self.kind is LoopKind.HOST and any(
            step.repeat is not RepeatKind.ONCE for step in self.stages
        ):
            raise RecipeError(
                RecipeRefusal.LOOP_INVALID,
                "a host-owned loop states its per-step specializations, never a repeat count; "
                "the iteration is data-dependent and belongs to the host",
            )

    @property
    def parameters(self) -> frozenset[ParameterName]:
        return frozenset(
            step.parameter for step in self.stages if step.parameter is not None
        )

    @property
    def runners(self) -> frozenset[RunnerName]:
        return frozenset(step.runner for step in self.stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "session_state": self.session_state.value,
            "stages": [step.as_dict() for step in self.stages],
        }

    @classmethod
    def decode(cls, raw: object) -> Loop:
        row = _fields("loop", raw, frozenset(("kind", "session_state", "stages")))
        kind, session_state, stages = row["kind"], row["session_state"], row["stages"]
        if kind not in tuple(item.value for item in LoopKind):
            raise RecipeError(RecipeRefusal.LOOP_INVALID, f"unknown loop kind {kind!r}")
        if session_state not in tuple(item.value for item in SessionState):
            raise RecipeError(
                RecipeRefusal.LOOP_INVALID, f"unknown session state {session_state!r}"
            )
        if not isinstance(stages, list):
            raise RecipeError(RecipeRefusal.LOOP_INVALID, "loop stages must be an array")
        return cls(
            kind=LoopKind(kind),
            stages=tuple(LoopStep.decode(step) for step in stages),
            session_state=SessionState(session_state),
        )


@dataclass(frozen=True, slots=True)
class RecipeParameter:
    """One integer count a counted stage reads, with its inclusive bounds."""

    name: ParameterName
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", parse_parameter_name(self.name))
        if type(self.minimum) is not int or type(self.maximum) is not int:
            raise RecipeError(
                RecipeRefusal.PARAMETER_INVALID, f"parameter {self.name!r} bounds must be integers"
            )
        if self.minimum < 1 or self.maximum < self.minimum:
            raise RecipeError(
                RecipeRefusal.PARAMETER_INVALID,
                f"parameter {self.name!r} bounds must satisfy 1 <= minimum <= maximum",
            )

    def as_dict(self) -> dict[str, Any]:
        return {"name": str(self.name), "minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def decode(cls, raw: object) -> RecipeParameter:
        row = _fields("parameter", raw, frozenset(("name", "minimum", "maximum")))
        return cls(
            name=parse_parameter_name(row["name"]),
            minimum=row["minimum"],
            maximum=row["maximum"],
        )


@dataclass(frozen=True, slots=True)
class Scheduler:
    """The named scheduler and its parameter block.  torchcg never interprets it.

    The block is vocabulary: the host implements the named scheduler and reads
    its own parameters.  It rides in the recipe because it is part of what makes
    two otherwise identical compositions different pipelines, so it must be
    inside the digest a consumer pins.
    """

    name: SchedulerName
    parameters: tuple[tuple[ParameterName, SchedulerValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", parse_scheduler_name(self.name))
        parameters = tuple(
            (parse_parameter_name(name), value) for name, value in self.parameters
        )
        for name, value in parameters:
            if isinstance(value, bool) or isinstance(value, (int, str)):
                continue
            if isinstance(value, float) and value == value and value not in (
                float("inf"),
                float("-inf"),
            ):
                continue
            raise RecipeError(
                RecipeRefusal.SCHEDULER_INVALID,
                f"scheduler parameter {name!r} must be a finite JSON scalar",
            )
        names = tuple(name for name, _ in parameters)
        if names != tuple(sorted(set(names))):
            raise RecipeError(
                RecipeRefusal.SCHEDULER_INVALID, "scheduler parameters must be sorted and unique"
            )
        object.__setattr__(self, "parameters", parameters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "parameters": {str(name): value for name, value in self.parameters},
        }

    @classmethod
    def decode(cls, raw: object) -> Scheduler:
        row = _fields("scheduler", raw, frozenset(("name", "parameters")))
        parameters = row["parameters"]
        if not isinstance(parameters, Mapping):
            raise RecipeError(
                RecipeRefusal.SCHEDULER_INVALID, "scheduler parameters must be an object"
            )
        return cls(
            name=parse_scheduler_name(row["name"]),
            parameters=tuple(
                sorted(
                    (parse_parameter_name(name), value) for name, value in parameters.items()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class DeclaredRunner:
    """What a DECLARATION-time fake-tensor export states for one runner.

    Typed bindings are generated from the declaration, not from this document.
    The recipe a mint emits is the DRIFT ASSERTION against it and the adopt-time
    reference: same runners, same specializations, same ingresses, or the mint compiled
    something the bindings do not describe.
    """

    name: RunnerName
    ingress_digests: tuple[IngressDigest, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", parse_runner_name(self.name))
        digests = tuple(parse_ingress_digest(value) for value in self.ingress_digests)
        if not digests:
            raise RecipeError(
                RecipeRefusal.DECLARATION_DRIFT,
                f"declared runner {self.name!r} states no ingress",
            )
        object.__setattr__(self, "ingress_digests", tuple(sorted(digests)))


@dataclass(frozen=True, slots=True)
class RecipeReference:
    """The digest-pinned reference an endpoint lock or crate embeds."""

    family: FamilyName
    digest: RecipeDigest
    version: int = RECIPE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", parse_family_name(self.family))
        object.__setattr__(self, "digest", parse_recipe_digest(self.digest))
        if self.version != RECIPE_VERSION:
            raise RecipeError(
                RecipeRefusal.RECIPE_VERSION_UNSUPPORTED,
                f"recipe reference v={self.version!r} is not v{RECIPE_VERSION}",
            )

    def as_dict(self) -> dict[str, Any]:
        return {"family": str(self.family), "v": self.version, "digest": str(self.digest)}

    @classmethod
    def decode(cls, raw: object) -> RecipeReference:
        row = _fields("recipe reference", raw, frozenset(("family", "v", "digest")))
        version = row["v"]
        if type(version) is not int or version != RECIPE_VERSION:
            raise RecipeError(
                RecipeRefusal.RECIPE_VERSION_UNSUPPORTED,
                f"recipe reference v={version!r} is not v{RECIPE_VERSION}",
            )
        return cls(
            family=parse_family_name(row["family"]),
            digest=parse_recipe_digest(row["digest"]),
            version=version,
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    """One family's composition: its specializations, its loop, and its scheduler."""

    family: FamilyName
    buckets: tuple[BucketAxis, ...]
    runners: tuple[RecipeRunner, ...]
    loop: Loop
    parameters: tuple[RecipeParameter, ...] = ()
    scheduler: Scheduler | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", parse_family_name(self.family))
        axis_names = tuple(axis.name for axis in self.buckets)
        if axis_names != tuple(sorted(set(axis_names))):
            raise RecipeError(
                RecipeRefusal.BUCKET_AXIS_INVALID, "bucket axes must be sorted and unique"
            )
        axes = {axis.name: axis.values for axis in self.buckets}
        if not isinstance(self.runners, tuple) or not self.runners:
            raise RecipeError(
                RecipeRefusal.RUNNER_INVALID, "a recipe must declare at least one runner"
            )
        runner_names = tuple(runner.name for runner in self.runners)
        if runner_names != tuple(sorted(set(runner_names))):
            raise RecipeError(
                RecipeRefusal.RUNNER_INVALID, "runners must be sorted by name and unique"
            )
        for runner in self.runners:
            unknown = sorted(set(runner.axes) - set(axes))
            if unknown:
                raise RecipeError(
                    RecipeRefusal.BUCKET_INVALID,
                    f"runner {runner.name!r} buckets on undeclared axis {unknown[0]!r}",
                )
            for variant in runner.variants:
                for name, value in variant.bucket:
                    if value not in axes[name]:
                        raise RecipeError(
                            RecipeRefusal.BUCKET_INVALID,
                            f"runner {runner.name!r} pins {name}={value}, "
                            f"which axis {name!r} does not declare",
                        )
            expected = {
                tuple(zip(runner.axes, combination, strict=True))
                for combination in product(*(axes[name] for name in runner.axes))
            }
            for layout in runner.layouts:
                built = {
                    variant.bucket for variant in runner.variants if variant.layout == layout
                }
                missing = sorted(expected - built)
                if missing:
                    raise RecipeError(
                        RecipeRefusal.BUCKET_COVERAGE_INCOMPLETE,
                        f"runner {runner.name!r} at layout {str(layout)!r} declares no variant "
                        f"for bucket {dict(missing[0])!r}; a generated closed type must be "
                        "total for every layout it offers",
                    )
        if not isinstance(self.loop, Loop):
            raise RecipeError(RecipeRefusal.LOOP_INVALID, "a recipe must declare its loop")
        staged = self.loop.runners
        unknown_runner = sorted(staged - set(runner_names))
        if unknown_runner:
            raise RecipeError(
                RecipeRefusal.RUNNER_UNKNOWN,
                f"loop stages undeclared runner {unknown_runner[0]!r}",
            )
        unused = sorted(set(runner_names) - staged)
        if unused:
            raise RecipeError(
                RecipeRefusal.RUNNER_UNUSED,
                f"runner {unused[0]!r} is declared but no loop stage runs it",
            )
        parameter_names = tuple(parameter.name for parameter in self.parameters)
        if parameter_names != tuple(sorted(set(parameter_names))):
            raise RecipeError(
                RecipeRefusal.PARAMETER_INVALID, "parameters must be sorted by name and unique"
            )
        counted = self.loop.parameters
        unknown_parameter = sorted(counted - set(parameter_names))
        if unknown_parameter:
            raise RecipeError(
                RecipeRefusal.PARAMETER_UNKNOWN,
                f"a counted stage reads undeclared parameter {unknown_parameter[0]!r}",
            )
        idle = sorted(set(parameter_names) - counted)
        if idle:
            raise RecipeError(
                RecipeRefusal.PARAMETER_UNUSED,
                f"parameter {idle[0]!r} is declared but no counted stage reads it",
            )

    def runner(self, name: RunnerName) -> RecipeRunner:
        """Resolve one runner handle.  Handler code uses GENERATED bindings instead.

        This exists for the generator and for diagnostics.  A handler that
        reaches a runner through a string it typed is exactly what the generated
        bindings exist to make unrepresentable.
        """

        wanted = parse_runner_name(name)
        for runner in self.runners:
            if runner.name == wanted:
                return runner
        raise RecipeError(
            RecipeRefusal.RUNNER_UNKNOWN, f"recipe declares no runner {str(wanted)!r}"
        )

    def as_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "v": RECIPE_VERSION,
            "family": str(self.family),
            "buckets": [axis.as_dict() for axis in self.buckets],
            "runners": [runner.as_dict() for runner in self.runners],
            "loop": self.loop.as_dict(),
            "parameters": [parameter.as_dict() for parameter in self.parameters],
        }
        if self.scheduler is not None:
            document["scheduler"] = self.scheduler.as_dict()
        return document

    def canonical(self) -> bytes:
        """The exact bytes the digest is taken over."""

        return _canonical(self.as_dict())

    def digest(self) -> RecipeDigest:
        """The 32-hex machine-independent digest a consumer pins at build."""

        return RecipeDigest(hashlib.sha256(self.canonical()).hexdigest()[:_DIGEST_HEX])

    def reference(self) -> RecipeReference:
        """The digest-pinned reference that rides beside an endpoint lock."""

        return RecipeReference(family=self.family, digest=self.digest())

    def assert_declaration(self, declared: Sequence[DeclaredRunner]) -> None:
        """Refuse a recipe that drifted from the declaration bindings came from.

        The direction matters and is the point of this method: the DECLARATION
        is the binding source, and this document asserts that what the mint
        actually compiled still matches it. Equal ingress digests imply equal
        signatures, because the signature is a projection of the ingress.
        """

        stated = {
            runner.name: tuple(sorted(variant.ingress_digest for variant in runner.variants))
            for runner in self.runners
        }
        expected = {runner.name: runner.ingress_digests for runner in declared}
        if len(expected) != len(declared):
            raise RecipeError(
                RecipeRefusal.DECLARATION_DRIFT, "a declaration names one runner twice"
            )
        missing = sorted(set(expected) - set(stated))
        if missing:
            raise RecipeError(
                RecipeRefusal.DECLARATION_DRIFT,
                f"the declaration states runner {str(missing[0])!r} and the recipe does not",
            )
        extra = sorted(set(stated) - set(expected))
        if extra:
            raise RecipeError(
                RecipeRefusal.DECLARATION_DRIFT,
                f"the recipe states runner {str(extra[0])!r} and the declaration does not",
            )
        for name, digests in sorted(expected.items()):
            if stated[name] != digests:
                raise RecipeError(
                    RecipeRefusal.DECLARATION_DRIFT,
                    f"runner {str(name)!r} was minted against a different call than it was "
                    f"declared with: declared {list(digests)!r}, recipe {list(stated[name])!r}",
                )

    def verify(self, reference: RecipeReference) -> None:
        """Refuse a reference that does not name this exact document."""

        if reference != self.reference():
            raise RecipeError(
                RecipeRefusal.REFERENCE_MISMATCH,
                f"reference {reference.as_dict()!r} does not name this recipe "
                f"({self.reference().as_dict()!r})",
            )

    @classmethod
    def decode(cls, raw: object) -> Recipe:
        """Decode one v1 document, refusing every other version loudly."""

        row = _fields(
            "recipe",
            raw,
            frozenset(("v", "family", "buckets", "runners", "loop", "parameters")),
            frozenset(("scheduler",)),
        )
        version = row["v"]
        if type(version) is not int or version != RECIPE_VERSION:
            raise RecipeError(
                RecipeRefusal.RECIPE_VERSION_UNSUPPORTED,
                f"recipe v={version!r} is not v{RECIPE_VERSION}; this reader has one version",
            )
        for field in ("buckets", "runners", "parameters"):
            if not isinstance(row[field], list):
                raise RecipeError(
                    RecipeRefusal.RECIPE_FIELDS_INVALID, f"recipe {field} must be an array"
                )
        scheduler = row.get("scheduler")
        return cls(
            family=parse_family_name(row["family"]),
            buckets=tuple(BucketAxis.decode(axis) for axis in row["buckets"]),
            runners=tuple(RecipeRunner.decode(runner) for runner in row["runners"]),
            loop=Loop.decode(row["loop"]),
            parameters=tuple(RecipeParameter.decode(item) for item in row["parameters"]),
            scheduler=None if scheduler is None else Scheduler.decode(scheduler),
        )

    @classmethod
    def loads(cls, payload: bytes | str) -> Recipe:
        """Decode one document from its stored bytes."""

        try:
            document = json.loads(payload)
        except ValueError as exc:
            raise RecipeError(
                RecipeRefusal.RECIPE_FIELDS_INVALID, f"recipe document is not JSON: {exc}"
            ) from exc
        return cls.decode(document)


def bucket_of(pairs: Sequence[tuple[str, int]]) -> tuple[tuple[BucketAxisName, int], ...]:
    """Build one canonical bucket from unvalidated pairs."""

    return tuple(sorted((parse_bucket_axis_name(name), value) for name, value in pairs))


__all__ = [
    "IDENTIFIER_GRAMMAR",
    "RECIPE_VERSION",
    "RESERVED_IDENTIFIERS",
    "BucketAxis",
    "BucketAxisName",
    "CallSignature",
    "DeclaredRunner",
    "FamilyName",
    "GraphSpecializationHash",
    "GraphSpecializationVariant",
    "IngressDigest",
    "LayoutContract",
    "Loop",
    "LoopKind",
    "LoopStep",
    "ParameterKind",
    "ParameterName",
    "Recipe",
    "RecipeDigest",
    "RecipeError",
    "RecipeParameter",
    "RecipeReference",
    "RecipeRefusal",
    "RecipeRunner",
    "RepeatKind",
    "RunnerName",
    "RuntimeAxes",
    "Scheduler",
    "SchedulerName",
    "SchedulerValue",
    "SessionState",
    "SignatureLeaf",
    "SignatureParameter",
    "bucket_of",
    "call_signature",
    "parse_bucket_axis_name",
    "parse_specialization_hash",
    "parse_family_name",
    "parse_ingress_digest",
    "parse_layout_contract",
    "parse_parameter_name",
    "parse_recipe_digest",
    "parse_runner_name",
    "parse_scheduler_name",
]
