"""What a dynamo mint's ``warmup_forward`` hour is actually made of."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from gen_worker._vendor.torchcg.spans import phase_delta, phase_snapshot

WARM_SPANS_V = 1

PHASE_ROUTER_DRAIN = "router_drain"

TOTAL_KEY = "_compile.compile_inner"

JIT_PARTITION_KEYS: Dict[str, Tuple[str, ...]] = {
    "tracing_s": ("bytecode_tracing", "build_guards", "variable_builder_call"),
    "graph_passes_s": (
        "_recursive_pre_grad_passes",
        "_recursive_joint_graph_passes",
        "_recursive_post_grad_passes",
    ),
    "lowering_s": ("GraphLowering.run",),
    "codegen_s": ("GraphLowering.codegen",),
    "kernel_compile_s": ("PyCodeCache.load_by_key_path",),
    "host_compile_s": ("AotCodeCompiler.compile",),
}

JIT_OVERLAY_KEYS: Dict[str, Tuple[str, ...]] = {
    "async_wait_s": ("async_compile.wait",),
    "parallel_kernel_cpu_s": ("compile_file",),
    "aot_dispatch_s": ("create_aot_dispatcher_function",),
}


def _sum(raw: Mapping[str, float], keys: Tuple[str, ...]) -> float:
    return round(sum(float(raw.get(k, 0.0)) for k in keys), 3)


def partition(raw: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """``(partition, overlays)`` for one span's raw metric delta."""
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
    """The warm plan's cost, per job and in total."""

    def __init__(self) -> None:
        self.jobs: List[Dict[str, Any]] = []
        self._raw: Dict[str, float] = {}
        self._wall = 0.0

    @contextlib.contextmanager
    def job(self, name: str) -> Iterator[None]:
        """Measure ONE warm forward."""
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
        """The flat, emittable ledger."""
        members, overlays = partition(self._raw)
        compile_s = members["dynamo_compile_s"]
        wall = round(self._wall, 3)
        compiled = sum(1 for j in self.jobs if j.get("compile_s", 0.0) > 0.5)
        return {
            "spans_v": WARM_SPANS_V,
            "totals": {
                "warm_wall_s": wall,
                "warm_compile_s": compile_s,
                "warm_execute_s": round(wall - compile_s, 3),
                "warm_jobs": len(self.jobs),
                "warm_jobs_compiling": compiled,
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
