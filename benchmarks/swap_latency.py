"""Checkpoint-swap latency benchmark (pgw#654 / WORKER-RESIDENCY-DESIGN).

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
  load        disk -> VRAM, per component (from_pretrained + .to(cuda))
  demote      VRAM -> host RAM (pipe.to("cpu"))
  promote     host RAM -> VRAM (pipe.to("cuda")) — the resident re-pick
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
  overlap     H2D copy on a dedicated copy stream (pinned staging) while a
              synthetic compute load runs — copy bandwidth and the
              interference cost on compute throughput.

Usage (pod-side):
  python -m benchmarks.swap_latency load    --checkpoint /path/A
  python -m benchmarks.swap_latency demote  --checkpoint /path/A
  python -m benchmarks.swap_latency swap    --checkpoint /path/A --to /path/B
  python -m benchmarks.swap_latency overlap --gb 4
  python -m benchmarks.swap_latency all --checkpoint /path/A --to /path/B

Each case prints one JSON object per measurement row (machine-parseable)
plus a summary table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_GiB = float(1 << 30)


def _refuse_off_pod() -> None:
    if os.environ.get("GEN_WORKER_FORBID_CPU_OFFLOAD"):
        raise SystemExit(
            "refusing: GEN_WORKER_FORBID_CPU_OFFLOAD is set — this is the "
            "control-plane box; run this benchmark on a GPU pod "
            "(weights-locality rule)")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device — run on a GPU pod")


@dataclass
class Row:
    case: str
    label: str
    seconds: float
    bytes: int = 0
    extra: Optional[Dict[str, Any]] = None

    def emit(self) -> None:
        d = asdict(self)
        d["gib"] = round(self.bytes / _GiB, 3)
        d["gib_per_s"] = (
            round(self.bytes / _GiB / self.seconds, 2) if self.seconds > 0 else 0.0
        )
        print(json.dumps(d, sort_keys=True), flush=True)


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def _cuda_bytes() -> int:
    import torch

    return int(torch.cuda.memory_allocated())


def _component_entries(tree: Path) -> Dict[str, Tuple[str, str]]:
    """component name -> (library, class) from model_index.json."""
    idx = json.loads((tree / "model_index.json").read_text())
    out: Dict[str, Tuple[str, str]] = {}
    for name, entry in idx.items():
        if name.startswith("_") or not isinstance(entry, list) or len(entry) != 2:
            continue
        if entry[0] is None or entry[1] is None:
            continue
        out[name] = (str(entry[0]), str(entry[1]))
    return out


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


def _load_component(tree: Path, name: str, lib: str, cls_name: str) -> Any:
    module = importlib.import_module(lib)
    cls = getattr(module, cls_name)
    src = tree / name
    if callable(getattr(cls, "from_pretrained", None)):
        try:
            import torch

            return cls.from_pretrained(str(src), torch_dtype=torch.bfloat16)
        except TypeError:
            return cls.from_pretrained(str(src))
    raise SystemExit(f"component {name}: {lib}.{cls_name} has no from_pretrained")


def _module_bytes(obj: Any) -> int:
    import torch

    if not isinstance(obj, torch.nn.Module):
        return 0
    return sum(
        p.numel() * p.element_size() for p in obj.parameters()
    ) + sum(b.numel() * b.element_size() for b in obj.buffers())


def bench_load(tree: Path) -> Dict[str, Any]:
    """disk -> VRAM per component; returns {name: module} for reuse."""
    loaded: Dict[str, Any] = {}
    total_t0 = time.monotonic()
    total_bytes = 0
    for name, (lib, cls_name) in sorted(_component_entries(tree).items()):
        if not (tree / name).is_dir():
            continue
        t0 = time.monotonic()
        obj = _load_component(tree, name, lib, cls_name)
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
        Row("load", f"{tree.name}/{name}", host_s + h2d_s, nbytes,
            {"disk_to_host_s": round(host_s, 3),
             "host_to_vram_s": round(h2d_s, 3)}).emit()
    Row("load", f"{tree.name}/TOTAL", time.monotonic() - total_t0,
        total_bytes, {"vram_allocated": _cuda_bytes()}).emit()
    return loaded


def bench_demote_promote(loaded: Dict[str, Any]) -> None:
    import torch

    mods = {n: m for n, m in loaded.items() if isinstance(m, torch.nn.Module)}
    nbytes = sum(_module_bytes(m) for m in mods.values())
    _sync()
    t0 = time.monotonic()
    for m in mods.values():
        m.to("cpu")
    _sync()
    torch.cuda.empty_cache()
    Row("demote", "vram->host_ram", time.monotonic() - t0, nbytes).emit()
    t1 = time.monotonic()
    for m in mods.values():
        m.to("cuda")
    _sync()
    Row("promote", "host_ram->vram (resident re-pick)",
        time.monotonic() - t1, nbytes).emit()


def bench_swap(tree_a: Path, tree_b: Path, loaded: Dict[str, Any]) -> None:
    """Component-first swap A -> B: only differing components move."""
    import torch

    comps_b = _component_entries(tree_b)
    differing: List[str] = []
    shared: List[str] = []
    for name in sorted(comps_b):
        if not (tree_b / name).is_dir():
            continue
        if component_digest(tree_a, name) == component_digest(tree_b, name):
            shared.append(name)
        else:
            differing.append(name)
    Row("swap", f"{tree_a.name}->{tree_b.name}/plan", 0.0, 0,
        {"differing": differing, "shared_by_content_address": shared}).emit()

    # Mechanism 1 (anti-pattern, for comparison): fresh module + replace.
    t0 = time.monotonic()
    replaced_bytes = 0
    fresh: Dict[str, Any] = {}
    for name in differing:
        lib, cls_name = comps_b[name]
        obj = _load_component(tree_b, name, lib, cls_name)
        if isinstance(obj, torch.nn.Module):
            obj = obj.to("cuda")
        fresh[name] = obj
        replaced_bytes += _module_bytes(obj)
    _sync()
    Row("swap", "mechanism=replace (breaks compiled binding)",
        time.monotonic() - t0, replaced_bytes).emit()

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
    Row("swap", "mechanism=copy (in-place, object identity preserved)",
        time.monotonic() - t1, copied_bytes,
        {"note": "same contract => no re-trace; a DMD-distilled sibling "
                 "swaps identically and serves the distilled recipe via "
                 "per-request views"}).emit()


def bench_overlap(gb: float) -> None:
    """H2D on a copy stream with pinned staging, concurrent with compute."""
    import torch

    n = int(gb * _GiB / 2)  # fp16 elements
    staging = torch.empty(n, dtype=torch.float16, pin_memory=True)
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
    Row("overlap", "compute baseline", 3.0, 0,
        {"matmul_per_s": round(base_rate, 1)}).emit()

    # Baseline copy bandwidth (idle GPU).
    copy_stream = torch.cuda.Stream()
    _sync()
    t1 = time.monotonic()
    with torch.cuda.stream(copy_stream):
        dst.copy_(staging, non_blocking=True)
    copy_stream.synchronize()
    Row("overlap", "h2d idle (pinned, copy stream)",
        time.monotonic() - t1, staging.numel() * 2).emit()

    # Concurrent: compute on default stream + copy on the copy stream.
    stop = threading.Event()
    counts = []
    t = threading.Thread(target=_compute, args=(stop, counts))
    _sync()
    t2 = time.monotonic()
    t.start()
    with torch.cuda.stream(copy_stream):
        dst.copy_(staging, non_blocking=True)
    copy_stream.synchronize()
    copy_s = time.monotonic() - t2
    stop.set()
    t.join()
    wall = time.monotonic() - t2
    rate = counts[0] / wall if wall > 0 else 0.0
    Row("overlap", "h2d during compute", copy_s, staging.numel() * 2,
        {"matmul_per_s": round(rate, 1),
         "compute_interference_pct": round(
             100.0 * (1.0 - rate / base_rate), 1) if base_rate else 0.0,
         }).emit()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", choices=("load", "demote", "swap", "overlap", "all"))
    ap.add_argument("--checkpoint", type=Path, help="materialized snapshot tree A")
    ap.add_argument("--to", type=Path, help="snapshot tree B (swap target; a "
                    "fine-tune or a DMD-distilled sibling)")
    ap.add_argument("--gb", type=float, default=4.0, help="overlap copy size")
    args = ap.parse_args(argv)
    _refuse_off_pod()

    if args.case in ("load", "demote", "swap", "all") and not args.checkpoint:
        ap.error("--checkpoint is required for this case")
    if args.case in ("swap", "all") and not args.to:
        ap.error("--to is required for the swap case")

    loaded: Dict[str, Any] = {}
    if args.case in ("load", "demote", "swap", "all"):
        loaded = bench_load(args.checkpoint)
    if args.case in ("demote", "all"):
        bench_demote_promote(loaded)
    if args.case in ("swap", "all"):
        bench_swap(args.checkpoint, args.to, loaded)
    if args.case in ("overlap", "all"):
        bench_overlap(args.gb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
