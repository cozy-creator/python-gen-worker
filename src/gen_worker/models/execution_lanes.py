"""Lane vocabulary (th#913/gw#596) — the SHARED SPEC twin of tensorhub's
``internal/orchestrator/precision/lane.go``. Ids and semantics must stay
byte-identical across repos.

A lane is the FULL execution-strategy descriptor:
``<weights>-<activation>[-<scale>]+<execution>``, e.g.
``fp8-w8a8-dynamic+compiled``. Dual-form input: a coarse FAMILY
(``bf16 | fp8 | 4bit``) or a full descriptor.
"""

from __future__ import annotations

from typing import Iterable, Optional

import msgspec

FAMILY_BF16 = "bf16"
FAMILY_FP8 = "fp8"
FAMILY_4BIT = "4bit"
FAMILIES = (FAMILY_BF16, FAMILY_FP8, FAMILY_4BIT)

WEIGHTS_BF16 = "bf16"
WEIGHTS_FP8 = "fp8"
WEIGHTS_SVDQ_FP4 = "svdq-fp4"
WEIGHTS_SVDQ_INT4 = "svdq-int4"
WEIGHTS_NVFP4 = "nvfp4"

ACT_W16A16 = "w16a16"  # upcast-ahead (weights already at compute dtype)
ACT_W8A16 = "w8a16"  # fp8 storage, per-layer upcast at inference
ACT_W8A8 = "w8a8"  # fp8 GEMM with activation scales (torch scaled_mm)
ACT_W4A4 = "w4a4"

SCALE_STATIC = "static"
SCALE_DYNAMIC = "dynamic"

EXEC_EAGER = "eager"
EXEC_COMPILED = "compiled"


class ExecutionLane(msgspec.Struct, frozen=True, kw_only=True):
    weights: str
    activation: str
    scale: str = ""  # "" when the lane has no scale axis
    execution: str


def execution_lane_id(execution_lane: ExecutionLane) -> str:
    body = f"{execution_lane.weights}-{execution_lane.activation}"
    if execution_lane.scale:
        body += f"-{execution_lane.scale}"
    return f"{body}+{execution_lane.execution}"


def family_of(execution_lane: ExecutionLane) -> str:
    if execution_lane.weights == WEIGHTS_BF16:
        return FAMILY_BF16
    if execution_lane.weights == WEIGHTS_FP8:
        return FAMILY_FP8
    if execution_lane.weights in (WEIGHTS_SVDQ_FP4, WEIGHTS_SVDQ_INT4, WEIGHTS_NVFP4):
        return FAMILY_4BIT
    return ""


_EXECUTION_EITHER = "either"


class _ExecutionLaneBody(msgspec.Struct, frozen=True, kw_only=True):
    weights: str
    activation: str
    execution_support: str
    scale: str = ""


# THE lane table's rows, ranked best-first. Execution support is authoritative
# for what the platform CHOOSES: lane enumeration, validation, and binding
# resolution all read this one field. It is NOT a claim about what can be
# OBSERVED — ie#655: a compiled-only body serves eager whenever a serve-time
# recipe quantizes and the self-mint then declines, and a report of that state
# must name it (``observed_execution_lane``) rather than be coerced into the
# choosable set.
_KNOWN_BODIES: tuple[_ExecutionLaneBody, ...] = (
    # Eager w8a8 has not been measured; keep Tensorhub's compiled-only answer.
    _ExecutionLaneBody(weights=WEIGHTS_FP8, activation=ACT_W8A8,
              scale=SCALE_DYNAMIC, execution_support=EXEC_COMPILED),
    # Eager nvfp4 has not been measured; keep Tensorhub's compiled-only answer.
    _ExecutionLaneBody(weights=WEIGHTS_NVFP4, activation=ACT_W4A4,
              scale=SCALE_STATIC, execution_support=EXEC_COMPILED),
    _ExecutionLaneBody(weights=WEIGHTS_SVDQ_FP4, activation=ACT_W4A4,
              execution_support=EXEC_EAGER),
    _ExecutionLaneBody(weights=WEIGHTS_SVDQ_INT4, activation=ACT_W4A4,
              execution_support=EXEC_EAGER),
    _ExecutionLaneBody(weights=WEIGHTS_BF16, activation=ACT_W16A16,
              execution_support=_EXECUTION_EITHER),
    _ExecutionLaneBody(weights=WEIGHTS_FP8, activation=ACT_W8A16,
              execution_support=_EXECUTION_EITHER),
)


def _supports(body: _ExecutionLaneBody, execution: str) -> bool:
    return body.execution_support == _EXECUTION_EITHER or body.execution_support == execution


def _body_for_execution_lane(execution_lane: ExecutionLane) -> Optional[_ExecutionLaneBody]:
    for body in _KNOWN_BODIES:
        if (
            body.weights == execution_lane.weights
            and body.activation == execution_lane.activation
            and body.scale == execution_lane.scale
        ):
            return body
    return None


def _execution_lane_for_body(body: _ExecutionLaneBody, execution: str) -> ExecutionLane:
    return ExecutionLane(
        weights=body.weights,
        activation=body.activation,
        scale=body.scale,
        execution=execution,
    )


def known_execution_lanes() -> list[str]:
    """Every concrete lane id, ranked (table order, compiled before eager)."""
    out: list[str] = []
    for body in _KNOWN_BODIES:
        if _supports(body, EXEC_COMPILED):
            out.append(execution_lane_id(_execution_lane_for_body(body, EXEC_COMPILED)))
        if _supports(body, EXEC_EAGER):
            out.append(execution_lane_id(_execution_lane_for_body(body, EXEC_EAGER)))
    return out


def execution_lane_body_id(execution_lane: ExecutionLane) -> str:
    """The lane id without the execution axis (verdict/declaration token)."""
    body = f"{execution_lane.weights}-{execution_lane.activation}"
    if execution_lane.scale:
        body += f"-{execution_lane.scale}"
    return body


def known_execution_lane_bodies() -> list[str]:
    """Every concrete lane BODY token, ranked (table order). These are the
    valid `handles=` declaration tokens (th#1050) — execution axis excluded:
    author kernels declare the quant scheme, the platform owns eager/compiled."""
    return [
        execution_lane_body_id(_execution_lane_for_body(body, EXEC_EAGER))
        for body in _KNOWN_BODIES
    ]


def _body_for_token(token: str) -> Optional[_ExecutionLaneBody]:
    tok = str(token or "").strip().lower()
    for body in _KNOWN_BODIES:
        if execution_lane_body_id(_execution_lane_for_body(body, EXEC_EAGER)) == tok:
            return body
    return None


def valid_execution_lane_body(token: str) -> bool:
    """Twin of the hub's ``precision.ValidExecutionLaneBody``."""
    return _body_for_token(token) is not None


def execution_lane_of_body(token: str, compiled: bool) -> ExecutionLane:
    """PLAN: the lane a known BODY token is CHOSEN to execute as under live
    compile state. Raises ValueError on a token outside the table: the lane
    vocabulary is platform-wide (the hub joins verdicts, cells and pricing on
    it) and no caller extends it.

    Not for reporting — see ``observed_execution_lane`` (ie#655)."""
    body = _body_for_token(token)
    if body is None:
        raise ValueError(
            f"lane body {token!r} is not a known lane body "
            f"(known: {', '.join(known_execution_lane_bodies())})")
    return _planned_execution(
        _execution_lane_for_body(body, EXEC_EAGER), compiled)


def observed_execution_lane(token: str, compiled: bool) -> ExecutionLane:
    """REPORT: the lane weights of body ``token`` are OBSERVED executing as.

    ie#655: the table's execution support says which lanes the platform
    PLANS, and an observation is not a plan. ``_planned_execution`` therefore
    must not touch this path: wan-2.2 served w8a8 weights EAGER on an H100
    (its self-mint declined for `insufficient_vram`) and the compiled-only
    coercion rewrote the honest `+eager` into `+compiled` — an over-claim on
    the key that feeds pricing, verdicts, floors and cell identity, in the
    flattering direction. The body stays table-validated (the weights half is
    still vocabulary); the execution axis is whatever actually happened."""
    body = _body_for_token(token)
    if body is None:
        raise ValueError(
            f"lane body {token!r} is not a known lane body "
            f"(known: {', '.join(known_execution_lane_bodies())})")
    return _execution_lane_for_body(
        body, EXEC_COMPILED if compiled else EXEC_EAGER)


def most_quantized_body(tokens: Iterable[str]) -> str:
    """The most-quantized BODY among ``tokens`` (a bf16 VAE riding a w8a8
    pipeline is still the w8a8 lane), ties by table rank. Unknown tokens are
    dropped; an empty selection is the plain bf16 body."""
    ranked = {body: i for i, body in enumerate(known_execution_lane_bodies())}
    best, best_key = "", (2, len(ranked) + 1)
    for token in tokens:
        tok = str(token or "").strip().lower()
        if tok not in ranked:
            continue
        quant = 1 if tok.startswith(WEIGHTS_BF16 + "-") else 0
        key = (quant, ranked[tok])
        if not best or key < best_key:
            best, best_key = tok, key
    return best or f"{WEIGHTS_BF16}-{ACT_W16A16}"


class AppliedLane(msgspec.Struct, frozen=True, kw_only=True):
    """What a SERVE-TIME recipe actually did to the weights (pgw#1104).

    The lane a request reports must be the lane its weights execute. A
    binding names the CHECKPOINT the hub resolved; an endpoint that quantizes
    inside ``setup()`` (torchao ``quantize_``, wan-2.2's ``_quantize_fp8``,
    minimax-h3's ``serve_recipe.quantize_dit``) moves the executed lane away
    from it, and only the code that did the conversion can say so provably.
    ``body`` is a ``known_execution_lane_bodies()`` token — the same th#1050
    vocabulary ``handles=`` uses; eager/compiled stays the platform's axis."""

    component: str
    body: str
    modules: int = 0
    kept_bf16: int = 0

    def detail(self) -> str:
        return (
            f"component={self.component} applied={self.body} "
            f"modules={self.modules} kept_bf16={self.kept_bf16}")


def valid_execution_lane(execution_lane: ExecutionLane) -> bool:
    if execution_lane.execution not in (EXEC_EAGER, EXEC_COMPILED):
        return False
    body = _body_for_execution_lane(execution_lane)
    return body is not None and _supports(body, execution_lane.execution)


def parse_execution_lane(s: str) -> ExecutionLane:
    """Parse a FULL descriptor id. Raises ValueError on anything else."""
    raw = str(s or "").strip().lower()
    parts = raw.split("+")
    if len(parts) != 2:
        raise ValueError(
            f"lane {s!r}: want `<weights>-<activation>[-<scale>]+<execution>`")
    body, execution = parts
    if execution not in (EXEC_EAGER, EXEC_COMPILED):
        raise ValueError(f"lane {s!r}: execution must be compiled|eager")
    weights = ""
    for w in (WEIGHTS_SVDQ_FP4, WEIGHTS_SVDQ_INT4, WEIGHTS_NVFP4, WEIGHTS_BF16, WEIGHTS_FP8):
        if body == w or body.startswith(w + "-"):
            weights = w
            break
    if not weights:
        raise ValueError(f"lane {s!r}: unknown weight format")
    rest = body[len(weights):].lstrip("-")
    segs = rest.split("-") if rest else []
    if len(segs) == 1:
        execution_lane = ExecutionLane(weights=weights, activation=segs[0], execution=execution)
    elif len(segs) == 2:
        execution_lane = ExecutionLane(weights=weights, activation=segs[0], scale=segs[1], execution=execution)
    else:
        raise ValueError(
            f"lane {s!r}: want `<weights>-<activation>[-<scale>]+<execution>`")
    if not valid_execution_lane(execution_lane):
        raise ValueError(
            f"lane {s!r} is not a known lane (known: {', '.join(known_execution_lanes())})")
    return execution_lane


class ExecutionLaneSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Dual-form parse result: a family (lane is None) or a full descriptor."""

    family: str = ""
    execution_lane: Optional[ExecutionLane] = None

    @property
    def is_zero(self) -> bool:
        return not self.family and self.execution_lane is None


def parse_execution_lane_spec(s: str) -> ExecutionLaneSpec:
    """Dual-form: "" = auto, a family, or a full descriptor id."""
    raw = str(s or "").strip().lower()
    if not raw:
        return ExecutionLaneSpec()
    if raw in FAMILIES:
        return ExecutionLaneSpec(family=raw)
    execution_lane = parse_execution_lane(raw)
    return ExecutionLaneSpec(family=family_of(execution_lane), execution_lane=execution_lane)


def is_w8a8_flavor(token: str) -> bool:
    t = str(token or "").strip().lower()
    return t == "fp8-w8a8" or t.startswith("fp8-w8a8-")


def mandatory_traced_lane_of(flavor: str) -> str:
    """Traced weight lane a stored flavor MANDATES for fail-closed serving:
    "w8a8" for `#fp8-w8a8` (gw#534), "w4a4" for `#nvfp4-w4a4` (gw#540), ""
    otherwise. th#1059 twin of the hub's ``mandatoryTracedLane``; th#1361/
    pgw#1065 choke point — replaced by the hub-resolved lane once th#1721
    descriptors cover every ref."""
    t = str(flavor or "").strip().lower()
    if is_w8a8_flavor(t):
        return "w8a8"
    if t == "nvfp4-w4a4" or t.startswith("nvfp4-w4a4-"):
        return "w4a4"
    return ""


def _planned_execution(execution_lane: ExecutionLane, compiled: bool) -> ExecutionLane:
    """The execution axis of a PLANNED lane: the caller's preference, moved
    onto a mode the table says this body is chosen for. Planning only —
    an OBSERVED posture is a fact and coercing it is a lie (ie#655)."""
    execution = EXEC_COMPILED if compiled else EXEC_EAGER
    body = _body_for_execution_lane(execution_lane)
    if body is not None and not _supports(body, execution):
        execution = EXEC_COMPILED if _supports(body, EXEC_COMPILED) else EXEC_EAGER
    return ExecutionLane(
        weights=execution_lane.weights,
        activation=execution_lane.activation,
        scale=execution_lane.scale,
        execution=execution,
    )


def execution_lane_body_of_binding(flavor: str, storage_dtype: str) -> str:
    """The lane BODY a (flavor, cast/storage_dtype) binding names — the
    WEIGHTS half, with no execution axis. Split out of
    ``execution_lane_of_binding`` (ie#655) so the reporting path can stamp an
    observed execution onto it instead of a planned one."""
    # cycle: ladder imports lanes at module top
    from .ladder import (
        CLASS_FP8,
        CLASS_NVFP4_W4A4,
        CLASS_SVDQ_FP4,
        CLASS_SVDQ_INT4,
        classify_flavor_token,
    )

    if str(storage_dtype or "").strip().lower() in ("fp8", "fp8+te"):
        execution_lane = ExecutionLane(weights=WEIGHTS_FP8, activation=ACT_W8A16, execution="")
    elif (cls := classify_flavor_token(flavor)) == CLASS_FP8:
        if is_w8a8_flavor(flavor):
            execution_lane = ExecutionLane(weights=WEIGHTS_FP8, activation=ACT_W8A8,
                        scale=SCALE_DYNAMIC, execution="")
        else:
            execution_lane = ExecutionLane(weights=WEIGHTS_FP8, activation=ACT_W8A16, execution="")
    elif cls == CLASS_SVDQ_FP4:
        execution_lane = ExecutionLane(weights=WEIGHTS_SVDQ_FP4, activation=ACT_W4A4, execution="")
    elif cls == CLASS_SVDQ_INT4:
        execution_lane = ExecutionLane(weights=WEIGHTS_SVDQ_INT4, activation=ACT_W4A4, execution="")
    elif cls == CLASS_NVFP4_W4A4:
        execution_lane = ExecutionLane(weights=WEIGHTS_NVFP4, activation=ACT_W4A4,
                    scale=SCALE_STATIC, execution="")
    else:
        execution_lane = ExecutionLane(weights=WEIGHTS_BF16, activation=ACT_W16A16, execution="")
    return execution_lane_body_id(execution_lane)


def execution_lane_of_binding(flavor: str, storage_dtype: str, compiled: bool) -> ExecutionLane:
    """PLAN: the lane a (flavor, cast/storage_dtype) binding is chosen to
    execute as — the twin of tensorhub's ``LaneOfResolution``."""
    body = _body_for_token(execution_lane_body_of_binding(flavor, storage_dtype))
    assert body is not None  # every branch above names a table row
    return _planned_execution(
        _execution_lane_for_body(body, EXEC_EAGER), compiled)


class ExecutionLaneUnavailableError(ValueError):
    """Typed refusal: the instructed lane cannot be served on this worker.
    Always names the lane — never a silent fallback."""

    def __init__(self, execution_lane: str, detail: str) -> None:
        self.execution_lane = execution_lane
        self.detail = detail
        super().__init__(f"lane_unavailable: {execution_lane} — {detail}")


__all__ = [
    "ACT_W16A16",
    "AppliedLane",
    "ACT_W4A4",
    "ACT_W8A16",
    "ACT_W8A8",
    "EXEC_COMPILED",
    "EXEC_EAGER",
    "FAMILIES",
    "FAMILY_4BIT",
    "FAMILY_BF16",
    "FAMILY_FP8",
    "ExecutionLane",
    "ExecutionLaneSpec",
    "ExecutionLaneUnavailableError",
    "SCALE_DYNAMIC",
    "SCALE_STATIC",
    "WEIGHTS_BF16",
    "WEIGHTS_FP8",
    "WEIGHTS_NVFP4",
    "WEIGHTS_SVDQ_FP4",
    "WEIGHTS_SVDQ_INT4",
    "family_of",
    "mandatory_traced_lane_of",
    "known_execution_lane_bodies",
    "known_execution_lanes",
    "execution_lane_body_id",
    "execution_lane_id",
    "execution_lane_body_of_binding",
    "execution_lane_of_binding",
    "execution_lane_of_body",
    "most_quantized_body",
    "observed_execution_lane",
    "valid_execution_lane_body",
    "parse_execution_lane",
    "parse_execution_lane_spec",
    "valid_execution_lane",
]
