from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["AdoptOutcome", "CompiledGraphAdoption", "EagerPhase"]


class EagerPhase(StrEnum):
    """Why a compile-declaring pipeline is serving EAGER — one shared token, riding self_mint_skipped/self_mint_started as phase and ArmOutcome.eager_reason. The VALUES are a wire contract: the hub groups worker_activity_events on them and joins to request rows, so renaming a member orphans history."""

    NO_FAMILY = "no_family"
    NO_CUDA = "no_cuda"
    NO_TOOLCHAIN = "no_toolchain"
    NO_COMPILE_TARGET = "no_compile_target"
    KEY_COMPUTATION_FAILED = "key_computation_failed"
    JIT_ARM_FAILED = "jit_arm_failed"
    MANDATORY_LANE_NEEDS_A_COMPILED_GRAPH = "mandatory_lane_needs_a_compiled_graph"

    COMPILED_GRAPH_QUARANTINED = "compiled_graph_quarantined"

    ARM_AXES_UNSTATEABLE = "arm_axes_unstateable"

    MINT_IN_PROGRESS = "mint_in_progress"

    HUB_ORDERED_EAGER = "hub_ordered_eager"

    ARM_PENDING = "arm_pending"
    NO_COMPILE_DECLARED = "no_compile_declared"
    UNCOMPILED = "uncompiled"
    BOOT_ENDED_UNCOMPILED = "boot_ended_uncompiled"

    GRAPH_BREAK = "graph_break"

    DECLARED_RANGE_EXCEEDED = "declared_range_exceeded"

    COMPILED_DEGRADED = "compiled_degraded"

    ARMED_TARGET_UNRESOLVED = "armed_target_unresolved"

    NO_COMPILE_CANDIDATES = "no_compile_candidates"

    ADOPTED_COMPILED_GRAPH_REFUSED = "adopted_compiled_graph_refused"

    OPERATOR_EAGER_ONLY = "operator_eager_only"

    MINT_UNAVAILABLE = "mint_unavailable"


@dataclass(frozen=True)
class AdoptOutcome:
    """The result of arming ONE candidate artifact on ONE pipeline."""

    armed: bool
    reason: str = ""
    detail: str = ""
    identity: str = ""

    def __bool__(self) -> bool:
        return self.armed

    @classmethod
    def hit(cls, identity: str = "") -> "AdoptOutcome":
        return cls(armed=True, identity=identity)

    @classmethod
    def miss(cls, reason: str, detail: str = "", identity: str = "") -> "AdoptOutcome":
        return cls(armed=False, reason=reason, detail=detail[:2000], identity=identity)


@dataclass(frozen=True)
class CompiledGraphAdoption:
    """One adoption ATTEMPT, measured, with the identity the hub fences on."""

    ref: str
    snapshot_digest: str
    artifact_kind: str
    arm_ms: int
    armed: bool
    reason: str = ""
    detail: str = ""
    pipeline_id: int = 0
    entry: str = ""
    compiled_graph_key: str = ""
