#!/usr/bin/env python3
"""pgw#1089 — drive the BOOT-SIDE key derivation against a real endpoint, with
no compile anywhere.

This is the derivation half of the rig gauntlet's cycle and NOTHING else: it
composes the vehicle's compile target structure-only, exports each declared
graph class on fake tensors in its own OS process, and folds the ``ck1`` key.
It never calls AOTInductor, never links a ``.so`` and never publishes — so it
runs on a laptop under the standing "mints run on remote machines only" rule,
which explicitly permits fake-tensor tracing for key derivation.

What it proves, and why each is worth a run
-------------------------------------------
* **The derivation WORKS on a real endpoint** — the structure-only builder
  reaches the boot path (pgw#1080's owed widening), the export gates pass on
  fake parameters, and every declared class produces a keying block.
* **Parallelism is not an identity axis** — the same tree derived at K=1 and at
  K=N yields the identical key. The unit tests pin the arithmetic; this pins
  the PROCESSES, which is where `aot_compile_pool` measured its failures.
* **The memo is honest and fast** — a second derivation over the same closure
  reports ``hit``, and a poisoned memo reports ``invalidated`` while the FRESH
  hashes are what the key names.
* **pgw#847's economy, priced on real graphs** — ``FakeTensorProp`` ms beside
  export ms, per child. Reported; it decides nothing here.

What it CANNOT prove, and must not be read as proving
------------------------------------------------------
**derived == stamped.** A stamp needs a mint, a mint needs a compile, and a
compile is a pod. This rig's key is self-consistent, not corroborated. The
comparison against a real mint's stamp is the pod leg, and it is the only load-
bearing proof in pgw#1089.

Usage::

    python scripts/boot_key_rig.py                       # micro, K=auto
    python scripts/boot_key_rig.py --vehicle micro-4d
    python scripts/boot_key_rig.py --workers 1 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

#: The gauntlet's own gate: never start multi-core work on a loaded box.
MAX_START_LOAD_1MIN = 10.0


def assert_load_gate(limit: float = MAX_START_LOAD_1MIN) -> float:
    one, _five, _fifteen = os.getloadavg()
    if one > limit:
        raise SystemExit(
            f"boot-key rig: 1-min load {one:.2f} is above {limit:.1f} — this "
            f"spawns one process per trace share. Wait, or pass --workers 1.")
    return one


def _vehicle(name: str) -> Any:
    from harness import rig_vehicles

    return rig_vehicles.vehicle(name)


def _slots(veh: Any, tree: Path) -> Dict[str, Any]:
    from gen_worker.api.binding import ModelRef
    from gen_worker.mint_process import MintSlot

    return {"pipeline": MintSlot(
        ref=ModelRef(source="tensorhub", path=veh.ref_path, tag="prod"),
        path=str(tree))}


def _declared_hint(family: str) -> int:
    """The parent's read of the declaration's row count — sizes K and nothing
    else (the adapter fork can double it; the children enumerate the truth)."""
    from gen_worker import aot_declaration, aot_mint

    decl = aot_mint.export_declaration(family)
    if decl is None:
        raise SystemExit(f"family {family!r} has no export declaration")
    return len(list(aot_declaration.cell_plans(decl)))


def _spec(veh: Any) -> Any:
    from gen_worker.mint_process import CompileCellSpec

    cfg = veh.compile_cell()
    return CompileCellSpec(
        shapes=tuple(tuple(int(v) for v in row) for row in (cfg.shapes or ())),
        targets=tuple(str(t) for t in (cfg.targets or ())),
        family=str(cfg.family or ""),
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
    )


def derive_once(
    veh: Any, tree: Path, root: Path, *, workers: int, memo_dir: Optional[Path],
    label: str, trust_memo: bool = True,
) -> Any:
    from gen_worker import boot_key, fleet_cells

    cfg = veh.compile_cell()
    work = root / f"work-{label}"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    derived = boot_key.derive(
        function=veh.function,
        modules=tuple(veh.modules),
        family=str(cfg.family or ""),
        cfg=_spec(veh),
        slots=_slots(veh, tree),
        declared_hint=_declared_hint(str(cfg.family or "")),
        envelope=fleet_cells.declared_envelope_block(cfg),
        work_root=work,
        memo_dir=memo_dir,
        workers=int(workers),
        trust_memo=bool(trust_memo),
    )
    wall = time.monotonic() - t0
    return derived, wall


def _report(derived: Any, wall: float, label: str) -> Dict[str, Any]:
    per_class = sorted(derived.trace_ms.values()) or [0]
    return {
        "label": label,
        "key": derived.digest,
        "graph_axis": derived.axes().get("graph", ""),
        "classes": derived.traced,
        "K": derived.workers,
        "width_reason": derived.width_reason,
        "memo": derived.memo,
        "wall_s": round(wall, 2),
        "derive_ms": derived.wall_ms,
        "trace_ms_min": per_class[0],
        "trace_ms_median": int(statistics.median(per_class)),
        "trace_ms_max": per_class[-1],
        "nodes_total": sum(derived.nodes.values()),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="micro")
    ap.add_argument(
        "--workers", type=int, action="append", default=None,
        help="K values to run, in order. Repeat. Default: 1 then auto.")
    ap.add_argument("--root", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    load = assert_load_gate()
    veh = _vehicle(args.vehicle)
    # The vehicle's endpoint modules live in this checkout, not in site-packages
    # — so they must reach the TRACE CHILDREN too. On a pod this is nothing:
    # the endpoint is installed in the image and `child_env` only has to make
    # `gen_worker` unambiguous. Here the rig supplies the same thing an image
    # would, through the PYTHONPATH `child_env` already preserves.
    extra = [str(REPO / "tests"), *(str(p) for p in veh.syspath)]
    for entry in extra:
        if entry not in sys.path:
            sys.path.insert(0, entry)
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [p for p in extra if p not in inherited] + inherited)

    import tempfile

    root = Path(args.root) if args.root else Path(
        tempfile.mkdtemp(prefix="boot-key-rig-"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"boot-key rig: vehicle={veh.name} family={veh.family} "
          f"load={load:.2f} root={root}")

    tree = veh.build_checkpoint(root / "checkpoint")
    print(f"  checkpoint tree: {tree} "
          f"({veh.checkpoint_bytes(tree) / 1e6:.2f} MB, GENERATED — nothing "
          f"downloaded)")

    widths = args.workers or [1, 0]
    # Every width run is a COLD derivation: the memo is per-width so a K=4 run
    # cannot be answered by the K=1 run's memo, which would make the scaling
    # figure a measurement of the memo instead of the pool.
    rows: List[Dict[str, Any]] = []
    keys: Dict[int, str] = {}
    for index, k in enumerate(widths):
        label = f"K{k or 'auto'}-{index}"
        memo_dir = root / f"memo-{index}"
        derived, wall = derive_once(
            veh, tree, root, workers=k, memo_dir=memo_dir, label=label)
        row = _report(derived, wall, label)
        rows.append(row)
        keys[index] = derived.digest
        print(f"  [{label}] key={row['key']} classes={row['classes']} "
              f"K={row['K']} memo={row['memo']} wall={row['wall_s']}s "
              f"per-class ms min/med/max="
              f"{row['trace_ms_min']}/{row['trace_ms_median']}/"
              f"{row['trace_ms_max']}")

    # The MEMO PATH — the same closure, second time, no trace at all.
    hot, hot_wall = derive_once(
        veh, tree, root, workers=widths[-1], memo_dir=memo_dir, label="memo")
    print(f"  [memo]  key={hot.digest} memo={hot.memo} traced={hot.traced} "
          f"wall={hot_wall:.3f}s ({rows[-1]['wall_s'] / max(hot_wall, 1e-9):.0f}x "
          f"faster than the cold derive above)")

    distinct = sorted(set(keys.values()))
    print(f"\n  KEY AGREEMENT across {len(widths)} width(s): "
          f"{'SAME' if len(distinct) == 1 else 'DIVERGED ' + repr(distinct)}")

    # The memo, poisoned deliberately: the FRESH trace must win and say so.
    from gen_worker import boot_key

    closure = boot_key.closure_digest(
        veh.family, _spec(veh), function=veh.function)
    held = boot_key.read_memo(memo_dir, closure)
    poison = {name: dict(block) for name, block in held.items()}
    for block in poison.values():
        block["class_dims"] = [["__poison__", 999]]
    boot_key.write_memo(memo_dir, closure, poison)
    # `trust_memo=False` is the VERIFY posture: trace anyway, then rule.
    poisoned, wall = derive_once(
        veh, tree, root, workers=1, memo_dir=memo_dir, label="poisoned",
        trust_memo=False)
    print(f"  MEMO POISON: memo={poisoned.memo} key={poisoned.digest} "
          f"({'UNCHANGED — the fresh trace won' if poisoned.digest == distinct[0] else 'CHANGED — RED'})")

    out = {
        "vehicle": veh.name,
        "family": veh.family,
        "rows": rows,
        "key_agreement": len(distinct) == 1,
        "keys": distinct,
        "memo_hit": {
            "state": hot.memo, "traced": hot.traced,
            "wall_s": round(hot_wall, 3), "key": hot.digest,
            "same_key": hot.digest == distinct[0]},
        "memo_poison": {
            "state": poisoned.memo, "key": poisoned.digest,
            "held": poisoned.digest == distinct[0]},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args.json}")
    ok = (out["key_agreement"] and out["memo_poison"]["held"]
          and out["memo_poison"]["state"] == "invalidated"
          and out["memo_hit"]["state"] == "hit"
          and out["memo_hit"]["traced"] == 0
          and out["memo_hit"]["same_key"])
    print(f"\nboot-key rig: {'GREEN' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
