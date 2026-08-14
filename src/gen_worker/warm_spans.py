"""What a dynamo mint's ``warmup_forward`` hour is actually made of.

Nearly all of a dynamo mint lands in ``warmup_forward``, while the phase named
after the work (``inductor_compile``) measures zero: ``mint_child`` opens that
frame around ``_drain_router``, and the fleet mint arms COLD with no router, so
it measures an empty queue. Every compile happens INLINE inside the warm
forwards — so ``warmup_forward`` is a bucket holding compile AND execution AND
everything else with no way to tell them apart.

A mint hour with one name on it cannot be optimised — you cannot tell a
compile-bound mint from a warm plan that is simply running too many diffusion
steps. This module measures the split.

Same instrument as the AOT path (:mod:`torch_compiled_graphs.spans`): deltas
of dynamo's process-global ``compilation_time_metrics`` across a span. The KEY
SET is different and that difference is the whole point — the AOT partition
was derived for ``aot_compile`` and **silently under-reports a JIT compile**.
MEASURED on the pin (torch 2.13.0+cu130) over one ``torch.compile`` call,
5.06 s wall::

    AOT partition total ......... 1.104 s   (22 % of the compile)
      host_compile_s ............ 0.000 s   <- AotCodeCompiler.compile is AOT-only
    _compile.compile_inner ...... 5.054 s   (99.9 % of the compile)
      PyCodeCache.load_by_key_path 3.203 s  <- where a JIT compile actually goes
        async_compile.wait ...... 2.886 s

So the JIT total is ``_compile.compile_inner`` — the whole of ``torch.compile``,
front-end through kernel load — and the members below partition it. Anything
they do not claim lands in ``compile_other_s`` rather than vanishing, exactly
as in the AOT ledger.

Overlays (reported, never summed into the partition): ``triton_s`` and
``parallel_kernel_cpu_s`` nest inside ``kernel_compile_s``, and
``compile_file`` exceeds the wall because it sums the async-compile workers'
own threads.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from torch_compiled_graphs.spans import phase_delta, phase_snapshot

#: Bump when the partition changes shape, so a reader never mixes two ledgers.
WARM_SPANS_V = 1

#: The phase that waits out a hot-swap router's queue — which the fleet mint
#: arms empty, so it never measures a compile. Defined in
#: this leaf module rather than in :mod:`gen_worker.activity` because the mint
#: CHILD needs it and deliberately imports neither protobuf nor psutil;
#: ``activity`` re-exports it, so the phase vocabulary still has one home.
PHASE_ROUTER_DRAIN = "router_drain"

#: The JIT compile's own total: everything ``torch.compile`` does for one
#: frame, measured by dynamo itself. NOT the AOT ``compile_s`` — that one is a
#: parent's Popen-to-reap wall around a child process, which the inline path
#: has no analogue of.
TOTAL_KEY = "_compile.compile_inner"

#: Members of ``dynamo_compile_s``. Disjoint on the pin: the containment runs
#: compile_inner > call_user_compiler > fw_compiler_base > compile_fx_inner >
#: GraphLowering.compile_to_module > {codegen, PyCodeCache.load_by_key_path},
#: so taking the LEAVES (and never their parents) is what keeps the sum honest.
JIT_PARTITION_KEYS: Dict[str, Tuple[str, ...]] = {
    # Dynamo's front-end: bytecode tracing, guard construction, variable
    # building. Cheap on a diffusion unet and reported anyway, because "the
    # front-end is cheap" is a claim that should carry its own number.
    "tracing_s": ("bytecode_tracing", "build_guards", "variable_builder_call"),
    "graph_passes_s": (
        "_recursive_pre_grad_passes",
        "_recursive_joint_graph_passes",
        "_recursive_post_grad_passes",
    ),
    "lowering_s": ("GraphLowering.run",),
    "codegen_s": ("GraphLowering.codegen",),
    # The JIT counterpart of the AOT path's `host_compile_s`: writing the
    # generated module out and waiting for the kernels it names. This is where
    # a JIT compile's time actually is (63 % of the probe above).
    "kernel_compile_s": ("PyCodeCache.load_by_key_path",),
    # Kept so a cell minted through an AOT-shaped inline call is not silently
    # unattributed; zero on the ordinary JIT path, which is itself a fact.
    "host_compile_s": ("AotCodeCompiler.compile",),
}

#: Reported under their own names, never summed with the partition — each is
#: known to nest inside a member above, and adding them would inflate
#: "attributed" while leaving the residual unexplained.
JIT_OVERLAY_KEYS: Dict[str, Tuple[str, ...]] = {
    # Inside kernel_compile_s.
    "async_wait_s": ("async_compile.wait",),
    # Sums the async-compile WORKERS' own CPU, so it legitimately exceeds the
    # wall — it prices the parallelism, it does not partition it.
    "parallel_kernel_cpu_s": ("compile_file",),
    "aot_dispatch_s": ("create_aot_dispatcher_function",),
}


def _sum(raw: Mapping[str, float], keys: Tuple[str, ...]) -> float:
    return round(sum(float(raw.get(k, 0.0)) for k in keys), 3)


def partition(raw: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """``(partition, overlays)`` for one span's raw metric delta.

    ``compile_other_s`` is the explicit residual of ``dynamo_compile_s`` and
    can be negative only if the pin's containment changed — which is the
    signal this shape exists to surface, so it is reported, not clamped.
    """
    total = round(float(raw.get(TOTAL_KEY, 0.0)), 3)
    members = {
        label: _sum(raw, keys) for label, keys in JIT_PARTITION_KEYS.items()}
    members["dynamo_compile_s"] = total
    members["compile_other_s"] = round(
        total - sum(v for k, v in members.items() if k != "dynamo_compile_s"),
        3)
    overlays = {
        label: value for label, keys in JIT_OVERLAY_KEYS.items()
        if (value := _sum(raw, keys))
    }
    triton = round(sum(
        v for k, v in raw.items()
        if "triton" in k.lower() and k != "compile_file"), 3)
    if triton:
        overlays["triton_s"] = triton
    return members, overlays


class WarmLedger:
    """The warm plan's cost, per job and in total.

    Not a profiler: a wall clock and one metric delta per job. The invariant it
    keeps is that ``warm_wall_s = dynamo_compile_s + warm_execute_s`` by
    construction, so the mint's biggest row can finally answer "compile, or
    forward?" — and a warm plan that is slow for a THIRD reason shows up as
    ``warm_execute_s`` growing rather than as time with no name.
    """

    def __init__(self) -> None:
        self.jobs: List[Dict[str, Any]] = []
        self._raw: Dict[str, float] = {}
        self._wall = 0.0

    @contextlib.contextmanager
    def job(self, name: str) -> Iterator[None]:
        """Measure ONE warm forward. Never raises on the telemetry path — the
        job's own exception propagates untouched."""
        before = phase_snapshot()
        started = time.monotonic()
        try:
            yield
        finally:
            wall = time.monotonic() - started
            self._wall += wall
            try:
                _p, _o, raw = phase_delta(before, phase_snapshot())
                for key, value in raw.items():
                    self._raw[key] = round(
                        self._raw.get(key, 0.0) + float(value), 3)
                compile_s = round(float(raw.get(TOTAL_KEY, 0.0)), 3)
                self.jobs.append({
                    "job": name,
                    "wall_s": round(wall, 3),
                    "compile_s": compile_s,
                    "execute_s": round(wall - compile_s, 3),
                })
            except Exception:  # noqa: BLE001 — telemetry never fails a mint
                self.jobs.append({"job": name, "wall_s": round(wall, 3)})

    def table(self) -> Dict[str, Any]:
        """The flat, emittable ledger. Shaped like the AOT phase table
        (``totals`` / ``phases`` / ``overlays`` / per-unit rows) so an
        AOT-vs-JIT comparison stays ONE grouped query rather than two
        vocabularies."""
        members, overlays = partition(self._raw)
        compile_s = members["dynamo_compile_s"]
        wall = round(self._wall, 3)
        compiled = sum(1 for j in self.jobs if j.get("compile_s", 0.0) > 0.5)
        return {
            "spans_v": WARM_SPANS_V,
            "totals": {
                "warm_wall_s": wall,
                "warm_compile_s": compile_s,
                # The residual, and named as one: whatever the warm plan spent
                # NOT compiling — the forwards themselves, the payload builds,
                # the tempdirs, the decode.
                "warm_execute_s": round(wall - compile_s, 3),
                "warm_jobs": len(self.jobs),
                # A warm plan whose jobs mostly compile NOTHING is a plan
                # paying full forward cost for coverage it already has, which
                # is a different defect from a slow compile.
                "warm_jobs_compiling": compiled,
                # OMITTED, not zeroed, when the wall is unmeasurable: a ratio
                # over a zero denominator reported as 0.0 would read "this
                # mint compiled nothing", which is the exact class of lie this
                # module exists to delete.
                **({"compile_fraction": round(compile_s / wall, 4)}
                   if wall > 0 else {}),
            },
            "phases": {k: v for k, v in members.items()
                       if k != "dynamo_compile_s"},
            "overlays": overlays,
            "jobs": list(self.jobs),
        }


__all__ = [
    "JIT_OVERLAY_KEYS", "JIT_PARTITION_KEYS", "PHASE_ROUTER_DRAIN",
    "TOTAL_KEY", "WARM_SPANS_V", "WarmLedger", "partition",
]
