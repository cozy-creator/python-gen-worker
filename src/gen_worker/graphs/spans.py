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


#: Every span in this ledger is recorded to millisecond precision. It is ONE
#: number (tcg#86): the `round(..., 3)` quantum, the drop floor below, and the
#: partition tolerance all follow from it, and before this they were three
#: independent literals that happened to agree.
LEDGER_PRECISION = 3
_QUANTUM_S = 10.0 ** -LEDGER_PRECISION

#: A delta below HALF the recording quantum cannot be a real span -- it is the
#: rounding of two snapshots. Was a bare `0.0005`, which IS half the quantum
#: and said so nowhere.
_DROP_FLOOR_S = _QUANTUM_S / 2

def _rounding_bound_s() -> float:
    """The closure error PURE ROUNDING can produce, in seconds.

    Every span and every partition total is stored to `LEDGER_PRECISION`, so
    each contributes at most half a quantum; the widest partition plus its own
    rounded total is the worst chain. Derived, so it follows the precision.
    """
    widest = max(len(members) for members in PARTITIONS.values())
    return (widest + 1) * _QUANTUM_S / 2


#: Slack carried ON TOP of `_rounding_bound_s()` before a partition is called
#: broken.
#:
#: tcg#86 (b)+(d), and the split is the point. The bare `0.05` this replaces
#: was ~11x what rounding can explain, so most of it was never a rounding
#: argument at all -- it was unstated tolerance for a ledger whose members are
#: sampled around real work. The rounding half is DERIVED above; this half is
#: AUTHOR-DECLARED and **UNMEASURED**, and it is set so the total does not
#: move: no ledger that closed before this change fails after it.
#:
#: THE FALSIFIER: a `check()` that reports a violation an operator can trace
#: to a genuinely missing member says this slack is too wide; a partition that
#: quietly absorbs 40 ms of real work says the same. Tighten it against a
#: banked ledger, never by feel -- none exists yet, which is exactly why the
#: number kept its value.
#: 45.5 ms: exactly what the retired `0.05` carried above today's 4.5 ms
#: rounding bound, so the total is unchanged for the current partition shape
#: and GROWS with the widest partition, which the flat literal could not.
_CLOSURE_SLACK_S = 0.0455


def _closure_tolerance_s() -> float:
    return _rounding_bound_s() + _CLOSURE_SLACK_S


def phase_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return ``(partition, overlays, raw)`` between two snapshots."""

    raw = {
        name: round(
            float(after.get(name, 0.0)) - float(before.get(name, 0.0)),
            LEDGER_PRECISION,
        )
        for name in set(after) | set(before)
    }
    raw = {name: value for name, value in raw.items() if value > _DROP_FLOOR_S}
    partition = {
        label: round(sum(raw.get(name, 0.0) for name in names), LEDGER_PRECISION)
        for label, names in PARTITION_KEYS.items()
    }
    overlays = {
        label: value
        for label, names in OVERLAY_KEYS.items()
        if (value := round(
            sum(raw.get(name, 0.0) for name in names), LEDGER_PRECISION
        ))
    }
    triton = round(
        sum(
            value
            for name, value in raw.items()
            if "async_compile" in name or "triton" in name.lower()
        ),
        LEDGER_PRECISION,
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
                LEDGER_PRECISION,
            )

    def mark(self, name: str, seconds: float) -> None:
        self.spans[name] = round(float(seconds), LEDGER_PRECISION)

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def close(self, total_name: str, residual_name: str) -> dict[str, float]:
        total = self.elapsed()
        named = sum(
            value for name, value in self.spans.items() if name not in (total_name, residual_name)
        )
        self.spans[total_name] = round(total, LEDGER_PRECISION)
        self.spans[residual_name] = round(total - named, LEDGER_PRECISION)
        return dict(self.spans)


def check(
    spans: Mapping[str, float], *, tolerance_s: float | None = None
) -> list[str]:
    """Return partition violations; an empty list is a closed ledger."""

    if tolerance_s is None:
        tolerance_s = _closure_tolerance_s()
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
    # A FRACTION, not a span: one digit finer than the ledger's own precision so
    # the ratio of two millisecond figures is not itself quantized to them.
    # tcg#86: the lone `round(, 4)` among six `round(, 3)` now says why.
    return round(dark / total, LEDGER_PRECISION + 1)


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
