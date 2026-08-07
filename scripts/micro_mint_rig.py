#!/usr/bin/env python3
"""pgw#978 — the local micro-mint rig: the whole mint machinery, on this box.

    python scripts/micro_mint_rig.py            # full cycle
    python scripts/micro_mint_rig.py --stage mint   # stop after the child
    python scripts/micro_mint_rig.py --json out.json

WHAT THIS REPLACES. The loop it exists to invert is: change code -> publish to
PyPI -> build an image on the published version -> spawn a pod -> observe for the
FIRST time. Paul, 2026-08-06: *"we want to release new versions of
python-gen-worker to pypi only after we've proven they work."* Attempt twenty
burned ~4 h of that loop to learn one `ValueError` that fired 0.0 s into
`warmup_forward`.

WHAT IT ACTUALLY RUNS. Every leg below is the production code path, against a
randomly-initialized toy latent-diffusion model on this box's card:

  1. resolve      — the parent builds a real `MintSlot` (identity + bytes)
  2. handoff      — `mint_delegate.build_request` -> `MintRequest` -> a JSON file
  3. spawn        — `mint_process.run_mint` starts a REAL child interpreter
  4. load         — the child re-runs discovery and `run_setup` from scratch
  5. warm         — `warmup_forward` over the endpoint's own declared plan
  6. export       — real `torch.export` + real AOTInductor compile + link
  7. seal         — real cell key, real packed artifact, real envelope
  8. publish      — the real `CellPublisher` wire to a LOCAL hub (7 HTTP calls)
  9. adopt        — a SECOND OS process discovers and adopts the cell

Legs 1-5 are where 9 of attempt twenty's 12 walls were.

THE BOUNDS (Paul's amendment to his own local-inference rule, 2026-08-06 —
recorded in `WORKSPACE-GIT-POLICY.md`). These are hard and this script enforces
them rather than trusting the operator:

  * weights under 500 MB, generated locally, never downloaded;
  * a 4 GiB device budget, SPLIT deliberately between the mint child and the
    adopting process (they run at different times but the split is stated, not
    assumed, so neither leg is written against "the whole card");
  * `nice`, so a compile cannot starve the box's other agents;
  * a load gate: refuse to start above 1-min load 24;
  * `GEN_WORKER_HOST_MOVE_GUARD` untouched — the rig never disables it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

GIB = 1 << 30

#: The whole-rig device budget. Never raise this without re-reading the policy
#: carve-out — 4 GiB is half this box's 8 GiB card, which is what keeps a rig
#: run compatible with a desktop session and another agent's work.
RIG_VRAM_BUDGET_BYTES = 4 * GIB

#: The split. The mint child does the export and the compile, so it gets the
#: larger share; the adopting process only loads a packed cell and runs one
#: forward. Stated here rather than left to each leg's own idea of "the card":
#: the ENV_HOST_SIBLINGS divisor pattern in procsplit exists for the same
#: reason — two processes on one device must agree on the division up front.
MINT_VRAM_BYTES = 3 * GIB
ADOPT_VRAM_BYTES = RIG_VRAM_BUDGET_BYTES - MINT_VRAM_BYTES

#: Refuse to start above this 1-minute load. The box is shared with several
#: agent sessions; a compile that starts at load 30 finishes slower AND makes
#: everyone else slower.
MAX_START_LOAD_1MIN = 24.0

#: The size ceiling the policy carve-out states. Enforced, not documented.
MAX_WEIGHTS_BYTES = 500 * 1000 * 1000

#: pgw#997: WHAT the rig mints is now a choice. `tiny` is pgw#978's original
#: one-entry plumbing toy; `micro` is the org-worker-shaped
#: `examples/micro-diffusion` package — three export entries, container inputs,
#: generated weights, a Dockerfile. See `tests/harness/rig_vehicles.py`.
DEFAULT_VEHICLE = "tiny"


# ---------------------------------------------------------------------------
# Gates — refusals, before anything expensive
# ---------------------------------------------------------------------------


class RigRefused(RuntimeError):
    """A named precondition failure. Terminal, and never a rig defect."""


def assert_load_gate(limit: float = MAX_START_LOAD_1MIN) -> float:
    load1 = os.getloadavg()[0]
    if load1 > limit:
        raise RigRefused(
            f"1-minute load is {load1:.1f} (> {limit:.0f}); this box is shared "
            f"with other agent sessions. Wait, or run with --force-load.")
    return load1


def resolve_device(want: str = "auto") -> Dict[str, Any]:
    """Which device this cycle runs on, and — when it is not the card — WHY.

    pgw#983, found by this rig's first run: the repo pins `torch==2.13.0+cu130`
    and this box's NVIDIA driver is 570.211.01 (CUDA 12.8). A cu130 build needs
    a 580-series driver, so `torch.cuda.is_available()` is False here and the
    box cannot execute the fleet's own pinned torch at all.

    That does NOT stop the rig from being worth running. The nine walls of
    attempt twenty that this exists to catch were PLUMBING — endpoint
    discovery, slot binding across the delegation boundary, the child spawn,
    the declaration, the publish wire, the adopt filter — and every one of them
    runs identically on CPU. What CPU does not cover is stated rather than
    quietly implied: no VRAM cap enforcement, no device placement, no measured
    kernel lane, and an `sm` axis that comes from a probe rather than a card.

    So the device is RESOLVED and REPORTED, never assumed. `--device cuda`
    turns the fallback into a refusal for the day the driver is current.
    """
    import torch

    available = torch.cuda.is_available()
    if want == "cuda" and not available:
        raise RigRefused(
            f"--device cuda requested but torch {torch.__version__} reports no "
            f"CUDA device on this box (see pgw#983: driver is CUDA 12.8, the "
            f"pinned torch is cu130 and needs a 580-series driver)")
    if available and want in ("auto", "cuda"):
        major, minor = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        return {
            "device_kind": "cuda",
            "device": props.name,
            "sm": f"sm_{major}{minor}",
            "total_bytes": int(props.total_memory),
            "torch": torch.__version__,
            "device_ordinal": 0,
            "covers": "plumbing + device (VRAM cap, placement, measured lane)",
        }
    return {
        "device_kind": "cpu",
        "device": "cpu",
        "sm": "(probe)",
        "total_bytes": 0,
        "torch": torch.__version__,
        "device_ordinal": -1,
        "covers": "plumbing only — NO VRAM cap, NO placement, NO measured lane",
        "why_not_cuda": (
            f"torch {torch.__version__} reports no CUDA device "
            f"(pgw#983: cu130 build vs a CUDA 12.8 driver)"),
    }


def assert_host_move_guard() -> str:
    """The one env the rig must never have turned off.

    `GEN_WORKER_HOST_MOVE_GUARD` is on by default and disabled with `=0`. A rig
    that ran with it off would be exercising a placement path production
    forbids, and would report a green cycle for a configuration no pod runs.
    """
    value = os.environ.get("GEN_WORKER_HOST_MOVE_GUARD", "")
    if value.strip() == "0":
        raise RigRefused(
            "GEN_WORKER_HOST_MOVE_GUARD=0 is set. The rig refuses to run with "
            "the host-move guard disabled — see the workspace policy.")
    return value or "(default: on)"


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------


@dataclass
class Leg:
    name: str
    ok: bool = False
    seconds: float = 0.0
    detail: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        mark = "ok " if self.ok else "FAIL"
        return f"  {mark} {self.name:<14} {self.seconds:7.2f}s  {self.detail}"


@dataclass
class RigResult:
    legs: List[Leg] = field(default_factory=list)
    env: Dict[str, Any] = field(default_factory=dict)
    total_s: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.legs) and all(leg.ok for leg in self.legs)

    def add(self, leg: Leg) -> Leg:
        self.legs.append(leg)
        return leg

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "total_s": round(self.total_s, 2),
            "env": self.env,
            "legs": [
                {"name": lg.name, "ok": lg.ok, "seconds": round(lg.seconds, 3),
                 "detail": lg.detail, "facts": lg.facts}
                for lg in self.legs
            ],
        }

    def report(self) -> str:
        head = "MICRO-MINT RIG — " + ("CYCLE COMPLETE" if self.ok else "CYCLE FAILED")
        body = "\n".join(leg.line() for leg in self.legs)
        return (
            f"{head}\n{body}\n"
            f"  ---- full machinery cycle: {self.total_s:.1f}s "
            f"(the number that replaces 4 hours)")


def _mint_slot(tree: Path, ref_path: str) -> Any:
    """The parent's resolution: WHICH checkpoint, and WHERE its bytes are.

    Built through the real `ModelRef`/`MintSlot` types, because pgw#974's whole
    point is that a slot with bytes and no identity must be unconstructable —
    a rig that hand-rolled a dict would route around the guard it exists to
    exercise.
    """
    from gen_worker.api.binding import ModelRef
    from gen_worker.mint_process import MintSlot

    return MintSlot(
        ref=ModelRef(source="tensorhub", path=ref_path, tag="prod"),
        path=str(tree),
    )


def _mint_request(
    workdir: Path, tree: Path, capture: Path, veh: Any, *,
    ordinal: int = 0, cap_bytes: int = MINT_VRAM_BYTES,
) -> Any:
    """Built through `mint_delegate.build_request` — the REAL parent chain.

    Not a hand-written `MintRequest`: the thing under test is the handoff, and
    a hand-written request is the one shape the handoff can never produce.
    """
    from gen_worker import mint_delegate
    from gen_worker.mint_delegate import MintTask

    cfg = veh.compile_cell()
    pending = SimpleNamespace(
        family=veh.family, cell_key="", recipe=os.environ.get("PGW978_RECIPE", "aot"), cfg=cfg,
        capture_dir=capture, target=workdir / "cell.tar.gz", mint_root=workdir)
    # The key is the REAL one, derived the way the arming brain derives it, so
    # the child's own recomputation has something to agree with.
    task = MintTask(
        pending=pending, pipe=None,
        function=veh.function, modules=tuple(veh.modules),
        slots={"pipeline": _mint_slot(tree, veh.ref_path)}, device=ordinal,
        execution_lane="", configs={})
    return mint_delegate.build_request(
        task, workdir=workdir, cap_bytes=cap_bytes)


def run_cycle(
    root: Path, *, stage: str = "all", force_load: bool = False,
    device: str = "auto", vehicle: str = DEFAULT_VEHICLE,
) -> RigResult:
    from harness import rig_vehicles

    veh = rig_vehicles.vehicle(vehicle)
    for entry in veh.syspath:
        if entry not in sys.path:
            sys.path.insert(0, entry)
    result = RigResult()
    t0 = time.monotonic()

    # -- gates ---------------------------------------------------------------
    leg = result.add(Leg("gates"))
    g0 = time.monotonic()
    load1 = 0.0 if force_load else assert_load_gate()
    guard = assert_host_move_guard()
    dev = resolve_device(device)
    leg.ok, leg.seconds = True, time.monotonic() - g0
    leg.facts = {"load1": load1, "host_move_guard": guard, **dev,
                 "mint_vram_bytes": MINT_VRAM_BYTES,
                 "adopt_vram_bytes": ADOPT_VRAM_BYTES}
    leg.facts["vehicle"] = veh.name
    leg.facts["vehicle_covers"] = veh.covers
    leg.detail = (f"vehicle={veh.name} {dev['device']} {dev['sm']} "
                  f"torch={dev['torch']} load={load1:.1f} "
                  f"covers={dev['covers']}")
    result.env = leg.facts

    # -- weights -------------------------------------------------------------
    from harness.tiny_diffusion import SYNTHETIC_RUNTIME_ENV

    leg = result.add(Leg("weights"))
    g0 = time.monotonic()
    tree = veh.build_checkpoint(root / "checkpoint" / veh.name)
    size = veh.checkpoint_bytes(tree)
    if size > MAX_WEIGHTS_BYTES:
        raise RigRefused(
            f"generated checkpoint is {size / 1e6:.1f} MB, over the "
            f"{MAX_WEIGHTS_BYTES / 1e6:.0f} MB carve-out ceiling")
    leg.ok, leg.seconds = True, time.monotonic() - g0
    leg.facts = {"bytes": size, "tree": str(tree)}
    leg.detail = f"{size / 1e6:.1f} MB generated locally (no download)"

    # -- the handoff ---------------------------------------------------------
    workdir = root / "mint"
    workdir.mkdir(parents=True, exist_ok=True)
    capture = root / "capture"

    leg = result.add(Leg("handoff"))
    g0 = time.monotonic()
    request = _mint_request(workdir, tree, capture, veh,
                            ordinal=int(dev["device_ordinal"]),
                            cap_bytes=(MINT_VRAM_BYTES
                                       if dev["device_kind"] == "cuda" else 0))
    leg.ok, leg.seconds = True, time.monotonic() - g0
    entries = _declared_entries(veh)
    leg.facts = {"family": request.family, "recipe": request.recipe,
                 "slots": sorted(request.slots),
                 "cell_key": request.cell_key,
                 "vram_cap_bytes": request.vram_cap_bytes,
                 "declared_entries": entries}
    leg.detail = (f"family={request.family} recipe={request.recipe} "
                  f"slots={sorted(request.slots)} "
                  f"entries={len(entries)}")

    # -- the child -----------------------------------------------------------
    from gen_worker import mint_process as mp

    leg = result.add(Leg("mint-child"))
    g0 = time.monotonic()
    phases: List[str] = []
    env = dict(mp.child_env(request))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "tests"), str(REPO / "src"), *veh.syspath,
         env.get("PYTHONPATH", "")])
    env["PGW978_CHECKPOINT"] = str(tree)
    if dev["device_kind"] != "cuda":
        # pgw#983: a cell key needs an `sm` and this box can state none. The
        # probes are supplied, LOUDLY — see `install_synthetic_runtime_if_asked`
        # and the `synthetic_runtime` fact this leg reports.
        env[SYNTHETIC_RUNTIME_ENV] = "1"
    outcome = asyncio.run(mp.run_mint(
        request, workdir=workdir, env=env, observe_interval_s=2.0,
        on_frame=lambda f: phases.append(f.phase) if f.phase else None))
    leg.seconds = time.monotonic() - g0
    leg.ok = outcome.status == mp.MINTED
    report = outcome.report
    leg.facts = {
        "status": outcome.status,
        "exit_code": outcome.exit_code,
        "phases_seen": sorted(set(phases)),
        "phase_seconds": dict(report.phases) if report else {},
        "peak_vram_bytes": int(report.peak_vram_bytes) if report else 0,
        "cell_key": str(report.cell_key) if report else "",
        "artifact": str(outcome.artifact or ""),
        "detail": outcome.detail[:500],
        "stderr_tail": outcome.stderr_tail[-1200:],
        # Never implied away: a cell sealed under a supplied `sm` is a PLUMBING
        # artifact and must never reach a shared namespace.
        "synthetic_runtime": env.get(SYNTHETIC_RUNTIME_ENV) == "1",
    }
    if not leg.ok:
        leg.detail = f"{outcome.status}: {outcome.detail[:220]}"
        result.total_s = time.monotonic() - t0
        return result
    warm = float((report.phases if report else {}).get("warmup_forward", 0.0))
    from gen_worker import aot_serve as _serve

    packed = sorted(
        (_serve.unpack_metadata(Path(outcome.artifact or "")).get("entries") or {}))
    leg.facts["packed_entries"] = packed
    if entries and sorted(entries) != packed:
        # The declaration said N entries; the tarball carries M. A cycle that
        # reported green on a cell missing an arm would be the exact silent
        # loss the entry vocabulary exists to prevent.
        leg.ok = False
        leg.detail = (f"declared entries {sorted(entries)!r} but the packed "
                      f"cell carries {packed!r}")
        result.total_s = time.monotonic() - t0
        return result
    leg.detail = (
        f"minted {Path(outcome.artifact or '').name} "
        f"key={(report.cell_key if report else '')[:20]} "
        f"warm={warm:.2f}s peak={leg.facts['peak_vram_bytes'] / 1e9:.2f}GB "
        f"entries={len(packed)}")

    if stage == "mint":
        result.total_s = time.monotonic() - t0
        return result

    # -- publish -------------------------------------------------------------
    from harness.cell_hub import LocalCellHub

    hub = LocalCellHub()
    try:
        leg = result.add(Leg("publish"))
        g0 = time.monotonic()
        checkpoint_id = _publish(hub, request, Path(outcome.artifact or ""))
        leg.ok, leg.seconds = True, time.monotonic() - g0
        leg.facts = {"checkpoint_id": checkpoint_id,
                     "routes": hub.routes(),
                     "cas_bytes": hub.artifact_bytes()}
        leg.detail = (f"checkpoint={checkpoint_id[:22]} over "
                      f"{len(hub.routes())} real HTTP calls")

        if stage == "publish":
            result.total_s = time.monotonic() - t0
            return result

        # -- adopt, in a SECOND process --------------------------------------
        leg = result.add(Leg("adopt"))
        g0 = time.monotonic()
        adopted = _adopt_in_subprocess(
            hub.base, root, veh, tree,
            synthetic_runtime=bool(
                next(lg for lg in result.legs
                     if lg.name == "mint-child").facts.get("synthetic_runtime")))
        leg.seconds = time.monotonic() - g0
        leg.ok = bool(adopted.get("ok"))
        leg.facts = adopted
        parity = adopted.get("parity_max_abs")
        leg.detail = (
            f"pid={adopted.get('pid')} adopted "
            f"{str(adopted.get('cell_key'))[:20]} "
            f"({adopted.get('artifact_bytes', 0) / 1e6:.1f} MB"
            + (f", {len(adopted['entries'])} entries"
               if adopted.get("entries") else "") + ")"
            + (f" parity max|delta|="
               f"{max(parity.values()):.2e} over {len(parity)} arms"
               if parity else "")
            if leg.ok else
            (str(adopted.get("error") or adopted.get("miss_log") or "")[-400:]))
    finally:
        hub.close()

    result.total_s = time.monotonic() - t0
    return result


def _publish(hub: Any, request: Any, artifact: Path) -> str:
    """The real `CellPublisher`, against the local hub."""
    from gen_worker import aot_serve, fleet_cells

    # The cell's OWN recorded envelope, read back off the packed tarball —
    # never a dict rebuilt here. `CellPublisher` derives the key, the axes and
    # the identity axes from it, so a hand-built stand-in would publish a cell
    # describing something other than the bytes beside it.
    meta = aot_serve.unpack_metadata(artifact)
    publisher = fleet_cells.CellPublisher(
        base_url=hub.base,
        worker_jwt=lambda: "local-rig-worker-jwt",
        image_digest="sha256:" + "0" * 64,
    )
    if not publisher.enabled():
        raise RigRefused("the publisher reports no sink; the local hub is up")
    return publisher.publish(request.family, artifact, meta)


def _declared_entries(veh: Any) -> List[str]:
    """The entry names this vehicle's declaration says to mint.

    Reported by the handoff leg so a cycle states its own SIZE: the whole
    point of the micro vehicle is that this list has three rows and sdxl's
    has thirty-six. A vehicle whose family carries no declaration (the
    dynamo-recipe toy) reports none rather than failing.
    """
    from gen_worker import aot_declaration as ad
    from gen_worker.api.export_contract import export_declaration

    decl = export_declaration(veh.family)
    if decl is None:
        return []
    return [ad.plan_entry_name(p) for p in ad.cell_plans(decl)]


def _adopt_in_subprocess(
    base: str, root: Path, veh: Any, tree: Path, *,
    synthetic_runtime: bool = False,
) -> Dict[str, Any]:
    """A SECOND OS process doing the discovery.

    In-process adoption would be a different test: the whole cross-pod claim is
    that a cell minted by one interpreter is servable by another that shares
    nothing but the hub and the card. A fresh interpreter is the cheapest
    honest stand-in for the second pod.
    """
    cache = root / "adopt-cache" / veh.name
    cache.mkdir(parents=True, exist_ok=True)
    src = veh.adopt_source(base, cache)
    from harness.tiny_diffusion import SYNTHETIC_RUNTIME_ENV

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "tests"), str(REPO / "src"), *veh.syspath,
         env.get("PYTHONPATH", "")])
    # The SAME generated tree the mint traced. Deterministic weights are what
    # make that a re-derivation rather than a copy: a second pod with the seed
    # has the bytes, and the snapshot digest agrees without a download.
    env["PGW978_CHECKPOINT"] = str(tree)
    if synthetic_runtime:
        # The adopting side runs `aot_serve.verify`, which compares the cell's
        # stamped sm/torch/cuda against THIS runtime's. A second process that
        # did not get the same supplied probes would reject the cell the first
        # one just minted — and report it as a filter miss, which is exactly the
        # wrong diagnosis. Both sides state the same runtime or neither does.
        env[SYNTHETIC_RUNTIME_ENV] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("RIG_ADOPT "):
            out = json.loads(line[len("RIG_ADOPT "):])
            if not out.get("ok"):
                # A MISS is not a crash, and its reason lives in the typed
                # `aot-cells` events on stderr. Carrying it out is the whole
                # difference between "no cell" and "twelve cells, all rejected
                # on one axis" (pgw#824).
                out["miss_log"] = _adopt_miss_log(proc.stderr)
            return out
    return {"ok": False,
            "error": (proc.stderr or proc.stdout or "no adopt line")[-1500:]}


def _adopt_miss_log(stderr: str) -> str:
    lines = [ln for ln in stderr.splitlines()
             if "aot-cells" in ln or "aot_cells" in ln or "verify" in ln]
    return "\n".join(lines[-12:]) or stderr[-800:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="micro_mint_rig",
        description="pgw#978 — run the whole mint machinery locally.")
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get(
            "PGW978_ROOT", str(Path.home() / ".cache" / "gen-worker" / "micro-rig"))),
        help="scratch root (checkpoint is cached here between runs)")
    parser.add_argument(
        "--stage", choices=("mint", "publish", "all"), default="all",
        help="stop after this leg")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the machine-readable result here")
    parser.add_argument("--force-load", action="store_true",
                        help="skip the shared-box load gate (say why)")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="auto falls back to CPU and SAYS SO; cuda refuses instead")
    parser.add_argument("--clean", action="store_true",
                        help="discard the scratch root first")
    parser.add_argument(
        "--vehicle", default=os.environ.get("PGW997_VEHICLE", DEFAULT_VEHICLE),
        help="WHAT to mint: 'tiny' (pgw#978's one-entry plumbing toy) or "
             "'micro' (pgw#997's org-worker-shaped examples/micro-diffusion — "
             "3 entries, container inputs, generated weights)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        result = run_cycle(root, stage=args.stage, force_load=args.force_load,
                           device=args.device, vehicle=args.vehicle)
    except RigRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(result.report())
    if args.json:
        Path(args.json).write_text(
            json.dumps(result.as_dict(), indent=2, sort_keys=True, default=str))
    if not result.ok:
        for leg in result.legs:
            if not leg.ok and leg.facts.get("stderr_tail"):
                print("\n--- mint child stderr tail ---", file=sys.stderr)
                print(leg.facts["stderr_tail"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
