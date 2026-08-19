"""Complete, versioned attribution over one graph-specialization compile.

The totals form three nested partitions.  Each level has one explicit
residual so a new or lost phase becomes visible instead of silently changing
the meaning of a measured span::

    compile_s = child_boot_s + child_wall_s + reap_lag_s + parent_other_s
    child_wall_s = child_seal_s + child_torch_import_s + child_devlock_s
        + child_setup_s + child_trace_s + compile_wall_s + child_pack_s
        + child_other_s
    compile_wall_s = lowering_s + codegen_s + graph_passes_s
        + host_compile_s + compile_other_s

Triton, autotune, device-lock wait, and Inductor total are overlays nested
inside those members.  They are never summed into a partition.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Any, cast

# Bump when the partition changes shape.  A reader must never mix tables with
# different residual meanings.
SPANS_V = 2

PARTITION_KEYS: dict[str, tuple[str, ...]] = {
    "lowering_s": ("GraphLowering.run",),
    "codegen_s": ("GraphLowering.codegen",),
    "host_compile_s": ("AotCodeCompiler.compile",),
    "graph_passes_s": (
        "_recursive_pre_grad_passes",
        "_recursive_joint_graph_passes",
        "_recursive_post_grad_passes",
    ),
}

OVERLAY_KEYS: dict[str, tuple[str, ...]] = {
    "inductor_total_s": ("compile_fx.<locals>.fw_compiler_base",),
    "autotune_s": (
        "CachingAutotuner.benchmark_all_configs",
        "CachingAutotuner.coordinate_descent_tuning",
        "CachingAutotuner.combo_sequential_autotune",
    ),
}

PARTITIONS: dict[str, tuple[str, ...]] = {
    "compile_s": (
        "child_boot_s",
        "child_wall_s",
        "reap_lag_s",
        "parent_other_s",
    ),
    "child_wall_s": (
        "child_seal_s",
        "child_torch_import_s",
        "child_devlock_s",
        "child_setup_s",
        "child_trace_s",
        "compile_wall_s",
        "child_pack_s",
        "child_other_s",
    ),
    "compile_wall_s": (
        "lowering_s",
        "codegen_s",
        "graph_passes_s",
        "host_compile_s",
        "compile_other_s",
    ),
}

RESIDUALS = ("parent_other_s", "child_other_s", "compile_other_s")

# Recorded beside the partitions but not members of them.  ``parent_stage_s``
# and ``parent_spawn_s`` occur outside compile_s; the others nest inside a
# child member.  Summing any of these into a partition double-counts time.
SUBSPANS = (
    "child_interp_s",
    "triton_s",
    "autotune_s",
    "device_lock_wait_s",
    "inductor_total_s",
    "parent_stage_s",
    "parent_spawn_s",
)


def phase_snapshot() -> dict[str, float]:
    """Return every current Dynamo compilation counter, summed per key."""

    try:
        dynamo_utils = import_module("torch._dynamo.utils")

        return {
            str(name): float(sum(values))
            for name, values in cast(Any, dynamo_utils).compilation_time_metrics.items()
        }
    except Exception:  # telemetry must never fail a compile
        return {}


def phase_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return ``(partition, overlays, raw)`` between two snapshots."""

    raw = {
        name: round(float(after.get(name, 0.0)) - float(before.get(name, 0.0)), 3)
        for name in set(after) | set(before)
    }
    raw = {name: value for name, value in raw.items() if value > 0.0005}
    partition = {
        label: round(sum(raw.get(name, 0.0) for name in names), 3)
        for label, names in PARTITION_KEYS.items()
    }
    overlays = {
        label: value
        for label, names in OVERLAY_KEYS.items()
        if (value := round(sum(raw.get(name, 0.0) for name in names), 3))
    }
    triton = round(
        sum(
            value
            for name, value in raw.items()
            if "async_compile" in name or "triton" in name.lower()
        ),
        3,
    )
    if triton:
        overlays["triton_s"] = triton
    return partition, overlays, raw


class SpanLedger:
    """Small wall-clock ledger whose close operation always names residual."""

    def __init__(self) -> None:
        self.spans: dict[str, float] = {}
        self._started = time.monotonic()
        # Cross-process spans need a clock both parent and child can read.
        self.start_epoch = time.time()

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.spans[name] = round(
                self.spans.get(name, 0.0) + time.monotonic() - started,
                3,
            )

    def mark(self, name: str, seconds: float) -> None:
        self.spans[name] = round(float(seconds), 3)

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def close(self, total_name: str, residual_name: str) -> dict[str, float]:
        total = self.elapsed()
        named = sum(
            value for name, value in self.spans.items() if name not in (total_name, residual_name)
        )
        self.spans[total_name] = round(total, 3)
        self.spans[residual_name] = round(total - named, 3)
        return dict(self.spans)


def check(spans: Mapping[str, float], *, tolerance_s: float = 0.05) -> list[str]:
    """Return partition violations; an empty list is a closed ledger."""

    problems: list[str] = []
    for total, members in PARTITIONS.items():
        if total not in spans:
            continue
        present = [member for member in members if member in spans]
        if not present:
            continue
        got = sum(float(spans[member]) for member in present)
        want = float(spans[total])
        if abs(got - want) > tolerance_s:
            problems.append(
                f"{total}={want:.3f} but its members sum to {got:.3f} "
                f"(delta {got - want:+.3f}s over {sorted(present)!r}) - "
                "the partition is not exhaustive"
            )
        missing = [member for member in members if member not in spans]
        if missing:
            problems.append(
                f"{total}: partition member(s) {missing!r} were never recorded, "
                "so the residual silently absorbed them"
            )
    for name in RESIDUALS:
        if float(spans.get(name, 0.0)) < -tolerance_s:
            problems.append(
                f"{name}={spans[name]:.3f} is negative - a member of its "
                "partition is double-counted"
            )
    return problems


def dark_fraction(spans: Mapping[str, float]) -> float:
    """Return the fraction of compile_s carried only by residual names."""

    total = float(spans.get("compile_s", 0.0))
    if total <= 0:
        return 0.0
    dark = sum(float(spans.get(name, 0.0)) for name in RESIDUALS)
    return round(dark / total, 4)


__all__ = [
    "OVERLAY_KEYS",
    "PARTITIONS",
    "PARTITION_KEYS",
    "RESIDUALS",
    "SPANS_V",
    "SUBSPANS",
    "SpanLedger",
    "check",
    "dark_fraction",
    "phase_delta",
    "phase_snapshot",
]
