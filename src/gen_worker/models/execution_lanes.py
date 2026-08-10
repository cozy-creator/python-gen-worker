"""Lane vocabulary (th#913/gw#596) — the SHARED SPEC twin of tensorhub's
``internal/orchestrator/precision/lane.go``. Ids and semantics must stay
byte-identical across repos.

A lane is the FULL execution-strategy descriptor:
``<weights>-<activation>[-<scale>]+<execution>``, e.g.
``fp8-w8a8-dynamic+compiled``. Dual-form input: a coarse FAMILY
(``bf16 | fp8 | 4bit``) or a full descriptor.
"""

from __future__ import annotations

from typing import Optional

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


# THE lane table's rows, ranked best-first. Execution support is authoritative:
# lane enumeration, validation, and binding resolution all read this one field.
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


def _with_supported_execution(execution_lane: ExecutionLane, compiled: bool) -> ExecutionLane:
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


def execution_lane_of_binding(flavor: str, storage_dtype: str, compiled: bool) -> ExecutionLane:
    """The concrete lane a (flavor, cast/storage_dtype) binding executes as —
    the twin of tensorhub's ``LaneOfResolution``."""
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
    return _with_supported_execution(execution_lane, compiled)


class ExecutionLaneUnavailableError(ValueError):
    """Typed refusal: the instructed lane cannot be served on this worker.
    Always names the lane — never a silent fallback."""

    def __init__(self, execution_lane: str, detail: str) -> None:
        self.execution_lane = execution_lane
        self.detail = detail
        super().__init__(f"lane_unavailable: {execution_lane} — {detail}")


__all__ = [
    "ACT_W16A16",
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
    "execution_lane_of_binding",
    "parse_execution_lane",
    "parse_execution_lane_spec",
    "valid_execution_lane",
]
