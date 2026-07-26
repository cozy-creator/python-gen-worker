"""Checkpoint-swap latency benchmark (pgw#654 / pgw#674 / WORKER-RESIDENCY-DESIGN).

Measures the PHYSICAL costs behind multi-checkpoint juggling, one narrow
case at a time, on a REAL GPU pod with already-materialized snapshot trees
(weights-locality: this never runs on the control-plane box — it refuses
without CUDA and under GEN_WORKER_FORBID_CPU_OFFLOAD).

Cache/eviction ladder measured here (design of record): VRAM -> host RAM ->
local-disk CAS (the pod NVMe IS content-addressed storage) -> NFS CAS
(remote disk, when mounted) -> re-download from R2 (tensorhub's remote CAS
origin). Point ``--checkpoint`` at trees on different tiers (local NVMe vs
an NFS mount) to compare source tiers; R2 cold-fetch is the store's job and
is measured by the worker's own transfer telemetry, not here.

Cases:
  load        disk -> VRAM, per component. Components load through the
              PRODUCTION path (``models.loading.load_component``), so a
              quantized flavor is materialized by its artifact loader and
              the row measures what serving actually pays (pgw#689).
  demote      VRAM -> host RAM (pinned swap cache built on first demote)
  promote     host RAM -> VRAM — the resident re-pick
  swap        component-first A -> B: only components whose content digest
              differs are touched (Paul's ruling: swap components, never
              whole pipelines — the pipeline object / compiled graphs are
              the stable identity). Two mechanisms measured:
                replace: load a fresh module and swap the attribute
                         (breaks compiled-graph object binding — the
                         anti-pattern, timed for comparison)
                copy:    in-place load_state_dict into the RESIDENT module
                         (preserves object identity; the design target).
              A DMD-distilled full-checkpoint sibling is the same case —
              pass it as B and serve the distilled recipe per request; same
              contract, no re-trace.
  stage       the pgw#674 rotation-preload path, per component:
              disk -> CPU load -> eager pin (prestage_module) -> H2D on the
              dedicated copy stream. THIS is the tier the RAM-staged
              single-buffer pays at rotation; the H2D row is the whole
              visible swap when the next model was preloaded.
  overlap     H2D copy on the dedicated copy stream (pinned staging) while
              a synthetic compute load runs — copy bandwidth and the
              interference cost on compute throughput. This bounds "does
              preloading slow serving".

Usage (pod-side):
  python -m gen_worker.benchmarks.swap_latency load    --checkpoint /path/A
  python -m gen_worker.benchmarks.swap_latency demote  --checkpoint /path/A
  python -m gen_worker.benchmarks.swap_latency swap    --checkpoint /path/A --to /path/B
  python -m gen_worker.benchmarks.swap_latency stage   --checkpoint /path/A
  python -m gen_worker.benchmarks.swap_latency overlap --gb 4
  python -m gen_worker.benchmarks.swap_latency all --checkpoint /path/A --to /path/B

Each case prints one JSON object per measurement row (machine-parseable)
plus rides the same rows through :func:`run_cases` for the
``gen_worker.diagnostics`` worker function (th#1198 payload path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_GiB = float(1 << 30)

ALL_CASES: Tuple[str, ...] = (
    "load", "demote", "swap", "stage", "overlap",
)

EmitFn = Callable[["Row"], None]


class OffPodError(RuntimeError):
    """This host must not run weight benchmarks (weights-locality rule)."""


def check_on_pod() -> None:
    if os.environ.get("GEN_WORKER_FORBID_CPU_OFFLOAD"):
        raise OffPodError(
            "refusing: GEN_WORKER_FORBID_CPU_OFFLOAD is set — this is the "
            "control-plane box; run this benchmark on a GPU pod "
            "(weights-locality rule)")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch-less host
        raise OffPodError(f"refusing: torch unavailable ({exc})") from exc
    if not torch.cuda.is_available():
        raise OffPodError("refusing: no CUDA device — run on a GPU pod")


@dataclass
class Row:
    case: str
    label: str
    seconds: float
    bytes: int = 0
    extra: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gib"] = round(self.bytes / _GiB, 3)
        d["gib_per_s"] = (
            round(self.bytes / _GiB / self.seconds, 2) if self.seconds > 0 else 0.0
        )
        return d


@dataclass
class Collector:
    """Row sink: collects for programmatic callers, optionally prints."""

    echo: bool = False
    rows: List[Row] = field(default_factory=list)

    def emit(self, row: Row) -> None:
        self.rows.append(row)
        if self.echo:
            print(json.dumps(row.as_dict(), sort_keys=True), flush=True)


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def _cuda_bytes() -> int:
    import torch

    return int(torch.cuda.memory_allocated())


def _component_names(tree: Path) -> List[str]:
    """Loadable component names from model_index.json (sorted). The module
    CLASS is not read here — :func:`_load_component` resolves it through the
    production loader, which is the one place that mapping may live."""
    idx = json.loads((tree / "model_index.json").read_text())
    return sorted(
        name for name, entry in idx.items()
        if not name.startswith("_")
        and isinstance(entry, list) and len(entry) == 2
        and entry[0] is not None and entry[1] is not None
    )


def component_digest(tree: Path, component: str) -> str:
    """Content address of one component subtree (file names + sizes + head/
    tail samples — fast, discriminates real weight differences without
    hashing multi-GB files end to end)."""
    root = tree / component
    if not root.is_dir():
        return ""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        st = p.stat()
        h.update(str(p.relative_to(root)).encode())
        h.update(str(st.st_size).encode())
        with open(p, "rb") as f:
            h.update(f.read(1 << 16))
            if st.st_size > (1 << 20):
                f.seek(-(1 << 16), 2)
                h.update(f.read(1 << 16))
    return h.hexdigest()


def swap_plan(tree_a: Path, tree_b: Path) -> Tuple[List[str], List[str]]:
    """(differing, shared_by_content_address) component names for A -> B.
    Pure planning — runs anywhere, no CUDA needed."""
    comps_b = _component_names(tree_b)
    differing: List[str] = []
    shared: List[str] = []
    for name in comps_b:
        if not (tree_b / name).is_dir():
            continue
        if component_digest(tree_a, name) == component_digest(tree_b, name):
            shared.append(name)
        else:
            differing.append(name)
    return differing, shared


def _load_component(tree: Path, name: str) -> Any:
    """One component, through the PRODUCTION component-load path.

    The benchmark's whole purpose is measuring what serving pays, so it must
    not own a second loader. ``models.loading.load_component`` is the same
    function the executor's component substitution and the pgw#674 rotation
    preloader call, and it routes quantized artifacts (w8a8 / w4a4) to their
    lane loaders. The reimplementation this replaced called
    ``cls.from_pretrained`` directly, which on any modelopt-produced flavor
    — i.e. every tree the fleet actually serves — died reconstructing
    ``NVIDIAModelOptConfig`` (pgw#689)."""
    from ..models.loading import load_component

    return load_component(tree, name)


def _module_bytes(obj: Any) -> int:
    import torch

    if not isinstance(obj, torch.nn.Module):
        return 0
    return sum(
        int(p.numel()) * int(p.element_size()) for p in obj.parameters()
    ) + sum(int(b.numel()) * int(b.element_size()) for b in obj.buffers())


def bench_load(tree: Path, emit: EmitFn) -> Dict[str, Any]:
    """disk -> VRAM per component; returns {name: module} for reuse."""
    loaded: Dict[str, Any] = {}
    total_t0 = time.monotonic()
    total_bytes = 0
    for name in _component_names(tree):
        if not (tree / name).is_dir():
            continue
        t0 = time.monotonic()
        obj = _load_component(tree, name)
        host_s = time.monotonic() - t0
        t1 = time.monotonic()
        import torch

        if isinstance(obj, torch.nn.Module):
            obj = obj.to("cuda")
        _sync()
        h2d_s = time.monotonic() - t1
        nbytes = _module_bytes(obj)
        total_bytes += nbytes
        loaded[name] = obj
        emit(Row("load", f"{tree.name}/{name}", host_s + h2d_s, nbytes,
                 {"disk_to_host_s": round(host_s, 3),
                  "host_to_vram_s": round(h2d_s, 3)}))
    emit(Row("load", f"{tree.name}/TOTAL", time.monotonic() - total_t0,
             total_bytes, {"vram_allocated": _cuda_bytes()}))
    return loaded


def bench_demote_promote(loaded: Dict[str, Any], emit: EmitFn) -> None:
    import torch

    mods = {n: m for n, m in loaded.items() if isinstance(m, torch.nn.Module)}
    nbytes = sum(_module_bytes(m) for m in mods.values())
    _sync()
    t0 = time.monotonic()
    for m in mods.values():
        m.to("cpu")
    _sync()
    torch.cuda.empty_cache()
    emit(Row("demote", "vram->host_ram", time.monotonic() - t0, nbytes))
    t1 = time.monotonic()
    for m in mods.values():
        m.to("cuda")
    _sync()
    emit(Row("promote", "host_ram->vram (resident re-pick)",
             time.monotonic() - t1, nbytes))


def bench_swap(
    tree_a: Path, tree_b: Path, loaded: Dict[str, Any], emit: EmitFn,
) -> None:
    """Component-first swap A -> B: only differing components move."""
    import torch

    differing, shared = swap_plan(tree_a, tree_b)
    emit(Row("swap", f"{tree_a.name}->{tree_b.name}/plan", 0.0, 0,
             {"differing": differing, "shared_by_content_address": shared}))

    # Mechanism 1 (anti-pattern, for comparison): fresh module + replace.
    t0 = time.monotonic()
    replaced_bytes = 0
    fresh: Dict[str, Any] = {}
    for name in differing:
        obj = _load_component(tree_b, name)
        if isinstance(obj, torch.nn.Module):
            obj = obj.to("cuda")
        fresh[name] = obj
        replaced_bytes += _module_bytes(obj)
    _sync()
    emit(Row("swap", "mechanism=replace (breaks compiled binding)",
             time.monotonic() - t0, replaced_bytes))

    # Mechanism 2 (design target): in-place state_dict copy into the
    # RESIDENT module — pipeline object + compiled graphs stay bound.
    t1 = time.monotonic()
    copied_bytes = 0
    for name in differing:
        resident = loaded.get(name)
        source = fresh.get(name)
        if (
            resident is None or source is None
            or not isinstance(resident, torch.nn.Module)
            or type(resident) is not type(source)
        ):
            continue
        resident.load_state_dict(source.state_dict(), strict=True)
        copied_bytes += _module_bytes(resident)
    _sync()
    emit(Row("swap", "mechanism=copy (in-place, object identity preserved)",
             time.monotonic() - t1, copied_bytes,
             {"note": "same contract => no re-trace; a DMD-distilled sibling "
                      "swaps identically and serves the distilled recipe via "
                      "per-request views"}))


def bench_stage(tree: Path, emit: EmitFn) -> None:
    """The pgw#674 rotation-preload tier, per component: disk -> CPU load ->
    eager pin (prestage_module) -> H2D on the dedicated copy stream. The H2D
    row is the whole visible swap when the next model was RAM-staged."""
    import torch

    from ..models.pinned_swap import prestage_module
    from ..models.staging import copy_stream

    total_pin = 0
    total_h2d_s = 0.0
    for name in _component_names(tree):
        if not (tree / name).is_dir():
            continue
        t0 = time.monotonic()
        obj = _load_component(tree, name)
        load_s = time.monotonic() - t0
        if not isinstance(obj, torch.nn.Module):
            continue
        nbytes = _module_bytes(obj)
        t1 = time.monotonic()
        pinned = prestage_module(obj)
        pin_s = time.monotonic() - t1
        total_pin += pinned
        stream = copy_stream()
        t2 = time.monotonic()
        if stream is not None:
            with torch.cuda.stream(stream):
                obj.to("cuda", non_blocking=True)
            stream.synchronize()
        else:  # pragma: no cover - CUDA guaranteed by check_on_pod
            obj.to("cuda")
            _sync()
        h2d_s = time.monotonic() - t2
        total_h2d_s += h2d_s
        emit(Row("stage", f"{tree.name}/{name}", load_s + pin_s + h2d_s, nbytes,
                 {"disk_to_host_s": round(load_s, 3),
                  "pin_s": round(pin_s, 3),
                  "pinned_bytes": pinned,
                  "h2d_copy_stream_s": round(h2d_s, 3)}))
        del obj
        torch.cuda.empty_cache()
    emit(Row("stage", f"{tree.name}/TOTAL-h2d (visible rotation cost when "
             "RAM-staged)", total_h2d_s, total_pin))


def bench_overlap(gb: float, emit: EmitFn) -> None:
    """H2D on the copy stream with pinned staging, concurrent with compute."""
    import torch

    from ..models.staging import copy_stream

    n = int(gb * _GiB / 2)  # fp16 elements
    staging_buf = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dst = torch.empty(n, dtype=torch.float16, device="cuda")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    def _compute(stop: threading.Event, out: List[int]) -> None:
        count = 0
        while not stop.is_set():
            _ = a @ b
            count += 1
        _sync()
        out.append(count)

    # Baseline compute throughput.
    stop = threading.Event()
    counts: List[int] = []
    t = threading.Thread(target=_compute, args=(stop, counts))
    _sync()
    t0 = time.monotonic()
    t.start()
    time.sleep(3.0)
    stop.set()
    t.join()
    base_rate = counts[0] / (time.monotonic() - t0)
    emit(Row("overlap", "compute baseline", 3.0, 0,
             {"matmul_per_s": round(base_rate, 1)}))

    # Baseline copy bandwidth (idle GPU) on the SAME copy stream the
    # rotation preload uses.
    stream = copy_stream()
    assert stream is not None  # check_on_pod guarantees CUDA
    _sync()
    t1 = time.monotonic()
    with torch.cuda.stream(stream):
        dst.copy_(staging_buf, non_blocking=True)
    stream.synchronize()
    emit(Row("overlap", "h2d idle (pinned, copy stream)",
             time.monotonic() - t1, staging_buf.numel() * 2))

    # Concurrent: compute on default stream + copy on the copy stream.
    stop = threading.Event()
    counts = []
    t = threading.Thread(target=_compute, args=(stop, counts))
    _sync()
    t2 = time.monotonic()
    t.start()
    with torch.cuda.stream(stream):
        dst.copy_(staging_buf, non_blocking=True)
    stream.synchronize()
    copy_s = time.monotonic() - t2
    stop.set()
    t.join()
    wall = time.monotonic() - t2
    rate = counts[0] / wall if wall > 0 else 0.0
    emit(Row("overlap", "h2d during compute", copy_s, staging_buf.numel() * 2,
             {"matmul_per_s": round(rate, 1),
              "compute_interference_pct": round(
                  100.0 * (1.0 - rate / base_rate), 1) if base_rate else 0.0,
              }))


def run_cases(
    cases: Tuple[str, ...],
    *,
    checkpoint: Optional[Path] = None,
    to: Optional[Path] = None,
    overlap_gb: float = 4.0,
    echo: bool = False,
) -> List[Row]:
    """Programmatic runner (the diagnostics worker function's entrypoint).
    Raises :class:`OffPodError` off-pod and ``ValueError`` on a case whose
    required tree is missing."""
    check_on_pod()
    unknown = sorted(set(cases) - set(ALL_CASES))
    if unknown:
        raise ValueError(f"unknown cases {unknown}; known: {list(ALL_CASES)}")
    needs_a = {"load", "demote", "swap", "stage"} & set(cases)
    if needs_a and checkpoint is None:
        raise ValueError(f"cases {sorted(needs_a)} require a checkpoint tree")
    if "swap" in cases and to is None:
        raise ValueError("the swap case requires a second tree (to=)")

    out = Collector(echo=echo)
    loaded: Dict[str, Any] = {}
    if {"load", "demote", "swap"} & set(cases):
        assert checkpoint is not None
        loaded = bench_load(checkpoint, out.emit)
    if "demote" in cases:
        bench_demote_promote(loaded, out.emit)
    if "swap" in cases:
        assert checkpoint is not None and to is not None
        bench_swap(checkpoint, to, loaded, out.emit)
    if "stage" in cases:
        # Release the resident copy first so staging measures a cold path.
        if loaded:
            import torch

            loaded.clear()
            torch.cuda.empty_cache()
        assert checkpoint is not None
        bench_stage(checkpoint, out.emit)
    if "overlap" in cases:
        bench_overlap(overlap_gb, out.emit)
    return out.rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("case", choices=ALL_CASES + ("all",))
    ap.add_argument("--checkpoint", type=Path, help="materialized snapshot tree A")
    ap.add_argument("--to", type=Path, help="snapshot tree B (swap target; a "
                    "fine-tune or a DMD-distilled sibling)")
    ap.add_argument("--gb", type=float, default=4.0, help="overlap copy size")
    args = ap.parse_args(argv)

    cases = ALL_CASES if args.case == "all" else (args.case,)
    if args.case == "all" and args.to is None:
        cases = tuple(c for c in cases if c != "swap")
    try:
        run_cases(
            cases,
            checkpoint=args.checkpoint,
            to=args.to,
            overlap_gb=args.gb,
            echo=True,
        )
    except (OffPodError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
