"""pgw#830: complete span coverage over one entry's compile.

The defect this closes
----------------------
pgw#757 named FIVE phases by reading dynamo's ``compilation_time_metrics``
(lowering / codegen / graph passes / host C++ / triton). On attempt nine's
72-entry regional mint those five summed to **742.6 s of a recorded
``compile_s`` of 1331.72 s — 56 %.** The other 44 % had no name, and a
number with no name cannot be optimized, argued about, or defended.

The reason the gap existed is structural, not a missing metric key:
``compile_s`` in the pooled path is the parent's Popen-to-reap wall of an
entry CHILD PROCESS, while ``phases`` only ever measured the inside of
``aot_compile``. Everything the child does before and after that call —
interpreter boot, ``import torch``, ``env_seal.establish``, the device-lock
install, ``torch.export.load`` of the staged program, the report write, the
parent's poll granularity — was inside the measured total and outside the
measured parts.

The model: three nested partitions, each exact
----------------------------------------------
Every level is a PARTITION with an explicit residual, so the sum is the
total by construction and a newly-introduced phase shows up as the residual
growing rather than as time silently vanishing::

    compile_s (parent Popen -> reap)
      = child_boot_s + child_wall_s + reap_lag_s

    child_wall_s (child module import -> report written)
      = child_seal_s + child_torch_import_s + child_devlock_s
      + child_program_load_s + compile_wall_s + child_other_s

    compile_wall_s (around aot_compile)
      = lowering_s + codegen_s + graph_passes_s + host_compile_s
      + compile_other_s

Overlays vs partition members
-----------------------------
``triton_s``, ``autotune_s`` and ``device_lock_wait_s`` are OVERLAYS, not
partition members: they are known to nest inside the members above them (the
triton keys sit inside codegen / host compile on a CUDA entry; a
device-benchmark flock wait happens inside ``aot_compile``). Summing them with
the partition was the second attribution bug — it inflates "attributed" while
leaving the residual unexplained. They are reported under their own names so a
reader can attribute WITHIN a member without double-counting the total.

``autotune_s`` (pgw#1006) is the compile-time Triton autotune benchmark —
``autotune_at_compile_time`` resolves True for AOTI, so the autotune block runs
INSIDE ``GraphLowering.codegen``. It is named because it decides two separate
questions and both were being answered from a residual: how much of a mint a
shared autotune cache could ever return (measured 3.1 % of a 390 s w8a8 SDXL
UNet entry — 12.1 s over 96 calls), and whether the selected config, which is
baked into the generated wrapper's grid expression and ``num_warps``, moved
between two mints of the same key.

The partition members are the four dynamo keys that are provably disjoint on
the pin (verified by the containment arithmetic on torch 2.13.0:
``GraphLowering.compile_to_fn = GraphLowering.codegen + AotCodeCompiler.compile``
exactly, and ``fx_codegen_and_compile - compile_to_fn - GraphLowering.run``
is 1.5 % noise).
"""

from __future__ import annotations

import contextlib
import time
from typing import Dict, Iterator, List, Mapping, Tuple

#: Bump when the partition changes shape, so a reader never mixes two ledgers.
SPANS_V = 1

#: Disjoint ``compilation_time_metrics`` leaves. These partition the inside of
#: ``aot_compile`` together with the ``compile_other_s`` residual.
PARTITION_KEYS: Dict[str, Tuple[str, ...]] = {
    "lowering_s": ("GraphLowering.run",),
    "codegen_s": ("GraphLowering.codegen",),
    "host_compile_s": ("AotCodeCompiler.compile",),
    "graph_passes_s": (
        "_recursive_pre_grad_passes",
        "_recursive_joint_graph_passes",
        "_recursive_post_grad_passes",
    ),
}

#: Reported, never summed into the partition — see the module docstring.
#: None of the autotune keys contain "triton"/"async_compile", so ``autotune_s``
#: and the ``triton_s`` overlay below do not double-count each other.
OVERLAY_KEYS: Dict[str, Tuple[str, ...]] = {
    "inductor_total_s": ("compile_fx.<locals>.fw_compiler_base",),
    "autotune_s": (
        "CachingAutotuner.benchmark_all_configs",
        "CachingAutotuner.coordinate_descent_tuning",
        "CachingAutotuner.combo_sequential_autotune",
    ),
}

#: The three-level partition, as {total: members}. ``check`` walks this, so
#: adding a member in one place is enough for the invariant to cover it.
PARTITIONS: Dict[str, Tuple[str, ...]] = {
    "compile_s": ("child_boot_s", "child_wall_s", "reap_lag_s"),
    "child_wall_s": (
        "child_seal_s", "child_torch_import_s", "child_devlock_s",
        "child_program_load_s", "compile_wall_s", "child_other_s",
    ),
    "compile_wall_s": (
        "lowering_s", "codegen_s", "graph_passes_s", "host_compile_s",
        "compile_other_s",
    ),
}

#: Residual member of each partition — the one that absorbs whatever the named
#: members did not claim. Named so the invariant can say WHICH bucket grew.
RESIDUALS = ("child_other_s", "compile_other_s")

#: Recorded alongside the partition but NOT a member of it: each is a split of
#: a member above it, kept because it is the number that decides a fix.
#: ``child_interp_s`` is the part of ``child_boot_s`` a persistent pool worker
#: would delete; ``triton_s`` / ``device_lock_wait_s`` nest inside
#: ``compile_wall_s``. Anything listed here must be excluded by a reader
#: summing the table.
SUBSPANS = ("child_interp_s", "triton_s", "autotune_s", "device_lock_wait_s",
            "inductor_total_s", "parent_stage_s", "parent_spawn_s")


def phase_snapshot() -> Dict[str, float]:
    """Every ``compilation_time_metrics`` counter, summed per key.

    The whole dict, not a whitelist: the residual arithmetic below needs the
    total, and a snapshot that pre-filtered could never notice a key the
    filter did not know about.
    """
    try:
        import torch._dynamo.utils as du

        return {
            str(k): float(sum(v))
            for k, v in du.compilation_time_metrics.items()}
    except Exception:  # noqa: BLE001 — telemetry never fails a compile
        return {}


def phase_delta(
    before: Mapping[str, float], after: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """``(partition, overlays, raw)`` seconds spent between two snapshots.

    ``raw`` is every key that moved, so a mint whose residual is large carries
    the evidence for naming it — the failure mode this issue exists to end is
    a gap whose contents nobody can see from the recorded telemetry.
    """
    raw = {
        k: round(float(after.get(k, 0.0)) - float(before.get(k, 0.0)), 3)
        for k in set(after) | set(before)
    }
    raw = {k: v for k, v in raw.items() if v > 0.0005}
    partition: Dict[str, float] = {}
    for label, keys in PARTITION_KEYS.items():
        partition[label] = round(sum(raw.get(k, 0.0) for k in keys), 3)
    overlays: Dict[str, float] = {}
    for label, keys in OVERLAY_KEYS.items():
        value = round(sum(raw.get(k, 0.0) for k in keys), 3)
        if value:
            overlays[label] = value
    triton = round(sum(
        v for k, v in raw.items()
        if "async_compile" in k or "triton" in k.lower()), 3)
    if triton:
        overlays["triton_s"] = triton
    return partition, overlays, raw


class SpanLedger:
    """Named wall-clock spans over one process's life, plus a residual.

    Not a tracing framework: a dict, a context manager and the arithmetic that
    makes the residual honest. The point is that ``close`` cannot be called
    without producing a number for the time nobody claimed.
    """

    def __init__(self) -> None:
        self.spans: Dict[str, float] = {}
        self._t0 = time.monotonic()
        #: Wall-clock twin of ``_t0``. The spans that cross a process boundary
        #: (a parent's spawn, a child's boot) can only be closed on a clock
        #: both processes read.
        self.start_epoch = time.time()

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.spans[name] = round(
                self.spans.get(name, 0.0) + time.monotonic() - started, 3)

    def mark(self, name: str, seconds: float) -> None:
        self.spans[name] = round(float(seconds), 3)

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def close(self, total_name: str, residual_name: str) -> Dict[str, float]:
        """Seal the ledger: record the total and the unclaimed remainder."""
        total = self.elapsed()
        named = sum(
            v for k, v in self.spans.items()
            if k != total_name and k != residual_name)
        self.spans[total_name] = round(total, 3)
        self.spans[residual_name] = round(total - named, 3)
        return dict(self.spans)


def check(spans: Mapping[str, float], *, tolerance_s: float = 0.05) -> List[str]:
    """Violations of the three-level partition. Empty means it reconciles.

    ``tolerance_s`` is absolute and small on purpose: these are the SAME
    clock's numbers added up, so any real drift is an attribution defect, not
    measurement noise. The one thing it must tolerate is per-span rounding
    (3 decimals x ~6 members).
    """
    out: List[str] = []
    for total, members in PARTITIONS.items():
        if total not in spans:
            continue
        present = [m for m in members if m in spans]
        if not present:
            continue
        got = sum(float(spans[m]) for m in present)
        want = float(spans[total])
        if abs(got - want) > tolerance_s:
            out.append(
                f"{total}={want:.3f} but its members sum to {got:.3f} "
                f"(delta {got - want:+.3f}s over {sorted(present)!r}) — the "
                f"partition is not exhaustive any more")
        missing = [m for m in members if m not in spans]
        if missing:
            out.append(
                f"{total}: partition member(s) {missing!r} were never "
                f"recorded, so the residual silently absorbed them")
    for name in RESIDUALS:
        if float(spans.get(name, 0.0)) < -tolerance_s:
            out.append(
                f"{name}={spans[name]:.3f} is NEGATIVE — a member of its "
                f"partition is double-counted (an overlay was summed as a "
                f"partition member, or two members nest)")
    return out


def dark_fraction(spans: Mapping[str, float]) -> float:
    """Fraction of ``compile_s`` that has no name beyond a residual bucket."""
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
