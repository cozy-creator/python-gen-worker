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


class Lane(msgspec.Struct, frozen=True, kw_only=True):
    weights: str
    activation: str
    scale: str = ""  # "" when the lane has no scale axis
    execution: str


def lane_id(lane: Lane) -> str:
    body = f"{lane.weights}-{lane.activation}"
    if lane.scale:
        body += f"-{lane.scale}"
    return f"{body}+{lane.execution}"


def family_of(lane: Lane) -> str:
    if lane.weights == WEIGHTS_BF16:
        return FAMILY_BF16
    if lane.weights == WEIGHTS_FP8:
        return FAMILY_FP8
    if lane.weights in (WEIGHTS_SVDQ_FP4, WEIGHTS_SVDQ_INT4, WEIGHTS_NVFP4):
        return FAMILY_4BIT
    return ""


_EXECUTION_EITHER = "either"


class _LaneBody(msgspec.Struct, frozen=True, kw_only=True):
    weights: str
    activation: str
    execution_support: str
    scale: str = ""


# THE lane table's rows, ranked best-first. Execution support is authoritative:
# lane enumeration, validation, and binding resolution all read this one field.
_KNOWN_BODIES: tuple[_LaneBody, ...] = (
    # Eager w8a8 has not been measured; keep Tensorhub's compiled-only answer.
    _LaneBody(weights=WEIGHTS_FP8, activation=ACT_W8A8,
              scale=SCALE_DYNAMIC, execution_support=EXEC_COMPILED),
    # Eager nvfp4 has not been measured; keep Tensorhub's compiled-only answer.
    _LaneBody(weights=WEIGHTS_NVFP4, activation=ACT_W4A4,
              scale=SCALE_STATIC, execution_support=EXEC_COMPILED),
    _LaneBody(weights=WEIGHTS_SVDQ_FP4, activation=ACT_W4A4,
              execution_support=EXEC_EAGER),
    _LaneBody(weights=WEIGHTS_SVDQ_INT4, activation=ACT_W4A4,
              execution_support=EXEC_EAGER),
    _LaneBody(weights=WEIGHTS_BF16, activation=ACT_W16A16,
              execution_support=_EXECUTION_EITHER),
    _LaneBody(weights=WEIGHTS_FP8, activation=ACT_W8A16,
              execution_support=_EXECUTION_EITHER),
)


def _supports(body: _LaneBody, execution: str) -> bool:
    return body.execution_support == _EXECUTION_EITHER or body.execution_support == execution


def _body_for_lane(lane: Lane) -> Optional[_LaneBody]:
    for body in _KNOWN_BODIES:
        if (
            body.weights == lane.weights
            and body.activation == lane.activation
            and body.scale == lane.scale
        ):
            return body
    return None


def _lane_for_body(body: _LaneBody, execution: str) -> Lane:
    return Lane(
        weights=body.weights,
        activation=body.activation,
        scale=body.scale,
        execution=execution,
    )


def known_lanes() -> list[str]:
    """Every concrete lane id, ranked (table order, compiled before eager)."""
    out: list[str] = []
    for body in _KNOWN_BODIES:
        if _supports(body, EXEC_COMPILED):
            out.append(lane_id(_lane_for_body(body, EXEC_COMPILED)))
        if _supports(body, EXEC_EAGER):
            out.append(lane_id(_lane_for_body(body, EXEC_EAGER)))
    return out


def lane_body_id(lane: Lane) -> str:
    """The lane id without the execution axis (verdict/declaration token)."""
    body = f"{lane.weights}-{lane.activation}"
    if lane.scale:
        body += f"-{lane.scale}"
    return body


def known_lane_bodies() -> list[str]:
    """Every concrete lane BODY token, ranked (table order). These are the
    valid `handles=` declaration tokens (th#1050) — execution axis excluded:
    author kernels declare the quant scheme, the platform owns eager/compiled."""
    return [
        lane_body_id(_lane_for_body(body, EXEC_EAGER))
        for body in _KNOWN_BODIES
    ]


def valid_lane(lane: Lane) -> bool:
    if lane.execution not in (EXEC_EAGER, EXEC_COMPILED):
        return False
    body = _body_for_lane(lane)
    return body is not None and _supports(body, lane.execution)


def parse_lane(s: str) -> Lane:
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
        lane = Lane(weights=weights, activation=segs[0], execution=execution)
    elif len(segs) == 2:
        lane = Lane(weights=weights, activation=segs[0], scale=segs[1], execution=execution)
    else:
        raise ValueError(
            f"lane {s!r}: want `<weights>-<activation>[-<scale>]+<execution>`")
    if not valid_lane(lane):
        raise ValueError(
            f"lane {s!r} is not a known lane (known: {', '.join(known_lanes())})")
    return lane


class LaneSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Dual-form parse result: a family (lane is None) or a full descriptor."""

    family: str = ""
    lane: Optional[Lane] = None

    @property
    def is_zero(self) -> bool:
        return not self.family and self.lane is None


def parse_lane_spec(s: str) -> LaneSpec:
    """Dual-form: "" = auto, a family, or a full descriptor id."""
    raw = str(s or "").strip().lower()
    if not raw:
        return LaneSpec()
    if raw in FAMILIES:
        return LaneSpec(family=raw)
    lane = parse_lane(raw)
    return LaneSpec(family=family_of(lane), lane=lane)


def is_w8a8_flavor(token: str) -> bool:
    t = str(token or "").strip().lower()
    return t == "fp8-w8a8" or t.startswith("fp8-w8a8-")


def _with_supported_execution(lane: Lane, compiled: bool) -> Lane:
    execution = EXEC_COMPILED if compiled else EXEC_EAGER
    body = _body_for_lane(lane)
    if body is not None and not _supports(body, execution):
        execution = EXEC_COMPILED if _supports(body, EXEC_COMPILED) else EXEC_EAGER
    return Lane(
        weights=lane.weights,
        activation=lane.activation,
        scale=lane.scale,
        execution=execution,
    )


def lane_of_binding(flavor: str, storage_dtype: str, compiled: bool) -> Lane:
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
        lane = Lane(weights=WEIGHTS_FP8, activation=ACT_W8A16, execution="")
    elif (cls := classify_flavor_token(flavor)) == CLASS_FP8:
        if is_w8a8_flavor(flavor):
            lane = Lane(weights=WEIGHTS_FP8, activation=ACT_W8A8,
                        scale=SCALE_DYNAMIC, execution="")
        else:
            lane = Lane(weights=WEIGHTS_FP8, activation=ACT_W8A16, execution="")
    elif cls == CLASS_SVDQ_FP4:
        lane = Lane(weights=WEIGHTS_SVDQ_FP4, activation=ACT_W4A4, execution="")
    elif cls == CLASS_SVDQ_INT4:
        lane = Lane(weights=WEIGHTS_SVDQ_INT4, activation=ACT_W4A4, execution="")
    elif cls == CLASS_NVFP4_W4A4:
        lane = Lane(weights=WEIGHTS_NVFP4, activation=ACT_W4A4,
                    scale=SCALE_STATIC, execution="")
    else:
        lane = Lane(weights=WEIGHTS_BF16, activation=ACT_W16A16, execution="")
    return _with_supported_execution(lane, compiled)


class LaneUnavailableError(ValueError):
    """Typed refusal: the instructed lane cannot be served on this worker.
    Always names the lane — never a silent fallback."""

    def __init__(self, lane: str, detail: str) -> None:
        self.lane = lane
        self.detail = detail
        super().__init__(f"lane_unavailable: {lane} — {detail}")


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
    "Lane",
    "LaneSpec",
    "LaneUnavailableError",
    "SCALE_DYNAMIC",
    "SCALE_STATIC",
    "WEIGHTS_BF16",
    "WEIGHTS_FP8",
    "WEIGHTS_NVFP4",
    "WEIGHTS_SVDQ_FP4",
    "WEIGHTS_SVDQ_INT4",
    "family_of",
    "is_w8a8_flavor",
    "known_lane_bodies",
    "known_lanes",
    "lane_body_id",
    "lane_id",
    "lane_of_binding",
    "parse_lane",
    "parse_lane_spec",
    "valid_lane",
]
