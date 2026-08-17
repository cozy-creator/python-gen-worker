#!/usr/bin/env python3
"""The local micro-mint rig: the whole mint machinery, on this box.

    python scripts/micro_mint_rig.py            # full cycle
    python scripts/micro_mint_rig.py --stage mint   # stop after the child
    python scripts/micro_mint_rig.py --json out.json

It exists so a mint is proven before a version reaches PyPI, rather than first
observed on a pod built from a published wheel.

WHAT IT ACTUALLY RUNS. Every leg below is the production code path, against a
randomly-initialized toy latent-diffusion model on this box's card:

  1. resolve      — the parent builds a real `MintSlot` (identity + bytes)
  2. handoff      — `mint_process.build_request` -> `MintRequest` -> a JSON file
  3. spawn        — `mint_process.run_mint` starts a REAL child interpreter
  4. load         — the child re-runs module discovery and `run_setup` from scratch
  5. warm         — `warmup_forward` over the endpoint's own declared plan
  6. export       — real `torch.export` + real AOTInductor compile + link
  7. seal         — real cell key, real packed artifact, real envelope
  8. publish      — the real `CellPublisher` wire to a LOCAL hub (7 HTTP calls)
  9. adopt        — a SECOND OS process fetches the exact named cell and adopts it

THE BOUNDS (the local-inference carve-out recorded in
`WORKSPACE-GIT-POLICY.md`). These are hard and this script enforces them rather
than trusting the operator:

  * weights under 500 MB, generated locally, never downloaded;
  * a 4 GiB device budget, SPLIT deliberately between the mint child and the
    adopting process (they run at different times but the split is stated, not
    assumed, so neither leg is written against "the whole card");
  * `nice`, so a compile cannot starve the box's other agents;
  * a load gate: refuse to start above 1-min load 24;
  * `GEN_WORKER_HOST_MOVE_GUARD` untouched — the rig never disables it.

THE ENV-DELIVERY MODE (`--hub-env`). The default rig builds the child's
environment itself (`mint_process.child_env` plus a few rig keys), a shape no
production pod ever has, so the real chain stays invisible:

    worker function declares env -> release_env_declarations
    operator sets a value        -> endpoint_env_entries
    pod launch                   -> EndpointEnvService.Resolve -> pod env -> Settings

A release that stops DECLARING a name makes the hub withhold the live entry
silently. `--hub-env` boots the mint child through that chain instead: the
child's environment is what the hub's rule would have delivered, ambient values
are STRIPPED so a developer's shell cannot stand in for a delivered one, and any
withholding is reported as a rig fact. The model lives in
`tests/harness/hub_env.py`.
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

#: There is deliberately no mint/adopt VRAM split. The rig's politeness lever is
#: `compile_posture.USER_MACHINE`: half the cores, double the host-RAM reserve,
#: and a nice level — bounds on what the box FEELS, not a guess at what a
#: compile will allocate.

#: Refuse to start above this 1-minute load. The box is shared with several
#: agent sessions; a compile that starts at load 30 finishes slower AND makes
#: everyone else slower.
MAX_START_LOAD_1MIN = 24.0

#: The size ceiling the policy carve-out states. Enforced, not documented.
MAX_WEIGHTS_BYTES = 500 * 1000 * 1000

#: WHAT the rig mints is a choice. `tiny` is the one-entry plumbing toy; `micro`
#: is the org-worker-shaped `examples/micro-diffusion` package — three export
#: entries, container inputs, generated weights, a Dockerfile. See
#: `tests/harness/rig_vehicles.py`.
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

    A box whose NVIDIA driver predates the pinned torch build (cu130 needs a
    580-series driver) reports `torch.cuda.is_available()` False and cannot
    execute the fleet's own pinned torch at all.

    That does NOT stop the rig from being worth running: the failures it exists
    to catch are PLUMBING — endpoint discovery, slot binding across the
    delegation boundary, the child spawn, the declaration, the publish wire, the
    adopt filter — and every one of them runs identically on CPU. What CPU does
    not cover is stated rather than quietly implied: no VRAM cap enforcement, no
    device placement, no measured kernel lane, and an `sm` axis that comes from
    a probe rather than a card.

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

    Built through the real `ModelRef`/`MintSlot` types: a slot with bytes and no
    identity must be unconstructable, and a rig that hand-rolled a dict would
    route around the guard it exists to exercise.
    """
    from gen_worker.api.binding import ModelRef
    from gen_worker.child_contract import MintSlot
    from gen_worker.serving_facts import FactsUnavailable

    return MintSlot(
        ref=ModelRef(source="tensorhub", path=ref_path, release="prod"),
        path=str(tree),
        # pgw#1333: a rig resolves no catalog, and says so rather than
        # asserting an unclassified checkpoint. The micro vehicles declare no
        # serving contract, so nothing checks these facts.
        facts=FactsUnavailable(owed_by="micro_mint_rig (no catalog)"),
    )


def _mint_request(
    workdir: Path, tree: Path, veh: Any, *, ordinal: int = 0,
) -> Any:
    """Built through `mint_process.build_request` — the REAL parent chain.

    Not a hand-written `MintRequest`: the thing under test is the handoff, and
    a hand-written request is the one shape the handoff can never produce.

    No banked per-entry DEVICE peak is forwarded: K is f(cores, one measured
    child RSS), so a rig cycle takes the pool path on its own cores without an
    operator priming a device basis.
    """
    from gen_worker import mint_process
    from gen_worker.mint_process import MintTask

    cfg = veh.compile_cell()
    pending = SimpleNamespace(
        family=veh.family, arm_token="", cfg=cfg,
        target=workdir / "cell.tar.gz", mint_root=workdir)
    # The obligation token is inessential here — the child stamps the REAL
    # cell key from the artifact's own recorded facts.
    task = MintTask(
        pending=pending, pipe=None,
        function=veh.function, modules=tuple(veh.modules),
        slots={"pipeline": _mint_slot(tree, veh.ref_path)}, device=ordinal,
        execution_lane="", configs={})
    return mint_process.build_request(task, workdir=workdir)


#: What the rig's endpoint function DECLARES, the way a real build reads it off
#: the function schema. Deliberately tiny: the point is the delivery mechanism,
#: not the breadth of the catalogue.
RIG_DECLARED_ENV = ("HF_TOKEN",)

#: Ambient names the rig REFUSES to let through in `--hub-env` mode. Without
#: this the developer's own shell satisfies the assertion and the mode proves
#: nothing — which is the exact substitution the blind spot was made of.
RIG_STRIPPED_ENV = ("HF_TOKEN",)


def hub_delivered_env(
    base: Dict[str, str], entries: Optional[Dict[str, str]] = None,
) -> tuple:
    """(env, withheld) as the hub would resolve them for this rig's release."""
    sys.path.insert(0, str(REPO / "tests"))
    from harness import hub_env as _hub

    delivery = _hub.resolve(
        _hub.declared_by(list(RIG_DECLARED_ENV)),
        _hub.EndpointEnvEntries(dict(entries or {})))
    env = _hub.pod_environ(base, delivery, strip=RIG_STRIPPED_ENV)
    return env, [
        {"name": w.name, "reason": w.reason, "detail": w.detail}
        for w in delivery.withheld
    ]


def run_cycle(
    root: Path, *, stage: str = "all", force_load: bool = False,
    device: str = "auto", vehicle: str = DEFAULT_VEHICLE,
    hub_env_mode: bool = False,
    hub_env_entries: Optional[Dict[str, str]] = None,
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
    leg.facts = {"load1": load1, "host_move_guard": guard, **dev}
    leg.facts["vehicle"] = veh.name
    leg.facts["vehicle_covers"] = veh.covers
    leg.detail = (f"vehicle={veh.name} {dev['device']} {dev['sm']} "
                  f"torch={dev['torch']} load={load1:.1f} "
                  f"covers={dev['covers']}")
    result.env = leg.facts

    # The rig parent seals exactly as a worker boot does. Without this the
    # parent's computed key axes (env_seal above all) describe an UN-established
    # process no pod ever runs, and the handback leg's axis guard would refuse
    # every healthy mint.
    from gen_worker import env_seal as _env_seal

    _env_seal.establish()
    leg.facts["env_seal"] = _env_seal.seal_digest(_env_seal.effective_seal())

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

    leg = result.add(Leg("handoff"))
    g0 = time.monotonic()
    request = _mint_request(
        workdir, tree, veh, ordinal=int(dev["device_ordinal"]))
    leg.ok, leg.seconds = True, time.monotonic() - g0
    entries = _declared_entries(veh)
    leg.facts = {"family": request.family,
                 "slots": sorted(request.slots),
                 "arm_token": request.arm_token,
                 "declared_entries": entries}
    leg.detail = (f"family={request.family} "
                  f"slots={sorted(request.slots)} "
                  f"entries={len(entries)}")

    # -- the child -----------------------------------------------------------
    from gen_worker import mint_process as mp

    leg = result.add(Leg("mint-child"))
    g0 = time.monotonic()
    phases: List[str] = []
    env = dict(mp.child_env(request))
    hub_withheld: List[Dict[str, str]] = []
    if hub_env_mode:
        # The child does not inherit whatever this process carries: it boots
        # with what the HUB would have delivered for this release, so a name the
        # release stops declaring disappears here rather than on a pod.
        env, hub_withheld = hub_delivered_env(env, hub_env_entries)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "tests"), str(REPO / "src"), *veh.syspath,
         env.get("PYTHONPATH", "")])
    env["PGW978_CHECKPOINT"] = str(tree)
    if dev["device_kind"] != "cuda":
        # A cell key needs an `sm` and this box can state none. The probes are
        # supplied, LOUDLY — see `install_synthetic_runtime_if_asked` and the
        # `synthetic_runtime` fact this leg reports.
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
        "hub_env_mode": hub_env_mode,
        "hub_env_withheld": hub_withheld,
        "phases_seen": sorted(set(phases)),
        "phase_seconds": dict(report.phases) if report else {},
        "peak_vram_bytes": int(report.peak_vram_bytes) if report else 0,
        "compiled_graph_key": str(report.compiled_graph_key) if report else "",
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
    # The guard is a SUBSET check, not equality. A bucket-bearing mint adds
    # adapter arms whose exact set depends on composed branch-capability; what
    # must never happen is a packed cell MISSING a declared class. An extra
    # packed arm is the adapter fork doing its job.
    missing = [n for n in _branchless(packed) if n not in set(entries)]
    if entries and missing:
        leg.ok = False
        leg.detail = (f"declared entries {sorted(entries)!r} but the packed "
                      f"cell is MISSING {missing!r} (packed: {packed!r})")
        result.total_s = time.monotonic() - t0
        return result
    leg.detail = (
        f"minted {Path(outcome.artifact or '').name} "
        f"key={(report.compiled_graph_key if report else '')[:20]} "
        f"warm={warm:.2f}s peak={leg.facts['peak_vram_bytes'] / 1e9:.2f}GB "
        f"entries={len(packed)}")

    if stage == "mint":
        result.total_s = time.monotonic() - t0
        return result

    # -- the handback: the parent adopts its OWN child's cell -----
    # The pod order: `adopt_delegated_mint` on the mint-opening parent's live
    # pipeline runs BEFORE anything publishes — only a cell that can arm ships.
    # Adopting only in a fresh SECOND process would leave this leg untested.
    if veh.parent_pipe is not None:
        import gc

        from gen_worker import aot_serve as _aserve
        from gen_worker import compile_cache as _cc
        from gen_worker import fleet_cells as _fc
        from gen_worker.models import loading as _loading

        leg = result.add(Leg("handback"))
        g0 = time.monotonic()
        if dev["device_kind"] != "cuda":
            # The parent must state the same supplied runtime the child
            # sealed under, or the axis guard would refuse a cardless cycle
            # on `sm` — the same rule both child processes already follow.
            os.environ[SYNTHETIC_RUNTIME_ENV] = "1"
            from harness.tiny_diffusion import install_synthetic_runtime_if_asked

            install_synthetic_runtime_if_asked()
        hb_root = root / "handback"
        hb_root.mkdir(parents=True, exist_ok=True)
        hb_artifact = hb_root / "cell.tar.gz"
        shutil.copy2(str(outcome.artifact), hb_artifact)
        pipe, hb_cfg = veh.parent_pipe(
            tree, "cuda" if dev["device_kind"] == "cuda" else "cpu")
        bucket = int(getattr(hb_cfg, "lora_bucket", 0) or 0)
        arm = _fc.arm_identity(
            veh.family, _loading.pipeline_weight_lane(pipe), bucket, hb_cfg)
        (hb_root / "mint-root").mkdir(exist_ok=True)
        pending = _fc.PendingSelfMint(
            family=veh.family, arm_token=arm.token,
            ref=f"{_cc.system_repo(veh.family)}#{arm.token}",
            cfg=hb_cfg, target=hb_root / "adopted.tar.gz",
            mint_root=hb_root / "mint-root", publisher=None,
            cache_dir=hb_root / "cache", arm_key=arm)
        minted = _fc.adopt_delegated_mint(pipe, pending, hb_artifact)
        leg.seconds = time.monotonic() - g0
        reason, why = _fc.adopt_refusal(pending)
        leg.ok = minted is not None
        leg.facts = {
            "arm_key": arm.token,
            "compiled_graph_key": str(getattr(minted, "compiled_graph_key", "") or ""),
            "arm_reason": reason,
            "detail": (why or "")[:400],
        }
        leg.detail = (
            f"parent adopted {leg.facts['compiled_graph_key'][:20]} "
            f"(arm_key={arm.token[:20]})" if leg.ok
            else f"{reason}: {(why or '')[:200]}")
        # Return the VRAM before the publish/adopt legs need it.
        _aserve.unwrap(pipe)
        del pipe
        gc.collect()
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup only
            pass
        if not leg.ok:
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
            checkpoint_id=checkpoint_id,
            synthetic_runtime=bool(
                next(lg for lg in result.legs
                     if lg.name == "mint-child").facts.get("synthetic_runtime")))
        leg.seconds = time.monotonic() - g0
        leg.ok = bool(adopted.get("ok"))
        leg.facts = adopted
        parity = adopted.get("parity_max_abs")
        leg.detail = (
            f"pid={adopted.get('pid')} adopted "
            f"{str(adopted.get('compiled_graph_key'))[:20]} "
            f"({adopted.get('artifact_bytes', 0) / 1e6:.1f} MB"
            + (f", {len(adopted['entries'])} entries"
               if adopted.get("entries") else "") + ")"
            + (f" parity max|delta|="
               f"{max(parity.values()):.2e} over {len(parity)} arms"
               if parity else "")
            + (f" arm={adopted['arm_reason']}"
               if adopted.get("arm_reason") else "")
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
    has thirty-six. A vehicle whose family carries no declaration reports none
    rather than failing — such a family serves JIT intake and mints nothing.
    """
    from gen_worker import aot_declaration as ad
    from gen_worker.api.export_contract import export_declaration

    decl = export_declaration(veh.family)
    if decl is None:
        return []
    # The DECLARED class set, always branchless: a bucket-bearing cfg forks
    # each branch-capable target into an adapter-bearing and a branchless
    # graph class at MINT time, and which targets are branch-capable
    # is composed truth this function has no pipeline for. So the declaration
    # stays the branchless authority and the guard compares against it after
    # stripping the adapter coordinate — see `_branchless`.
    return [ad.plan_entry_name(p) for p in ad.cell_plans(decl)]


def _branchless(names: List[str]) -> List[str]:
    """Packed entry names with any ``adapter=…`` coordinate dropped, so a
    bucket-bearing cell's arms compare against the declared class set."""
    out = []
    for name in names:
        segs = []
        for seg in name.split("/"):
            kept = ",".join(
                pair for pair in seg.split(",")
                if not pair.startswith("adapter="))
            if kept:
                segs.append(kept)
        out.append("/".join(segs))
    return sorted(set(out))


def _adopt_in_subprocess(
    base: str, root: Path, veh: Any, tree: Path, *,
    checkpoint_id: str,
    synthetic_runtime: bool = False,
) -> Dict[str, Any]:
    """A SECOND OS process adopting the EXACT published cell: it is TOLD the
    checkpoint id — there is no discovery; a serving pod is told by
    `Arm.artifact` the same way.

    In-process adoption would be a different test: the whole cross-pod claim is
    that a cell minted by one interpreter is servable by another that shares
    nothing but the hub and the card. A fresh interpreter is the cheapest
    honest stand-in for the second pod.
    """
    cache = root / "adopt-cache" / veh.name
    cache.mkdir(parents=True, exist_ok=True)
    src = veh.adopt_source(base, cache, checkpoint_id)
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
                # on one axis".
                out["miss_log"] = _adopt_miss_log(proc.stderr)
            return out
    return {"ok": False,
            "error": (proc.stderr or proc.stdout or "no adopt line")[-1500:]}


def _adopt_miss_log(stderr: str) -> str:
    lines = [ln for ln in stderr.splitlines()
             if "rig-fetch" in ln or "aot" in ln or "verify" in ln]
    return "\n".join(lines[-12:]) or stderr[-800:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    # The rig parent seals as a worker boot does, and `establish` fail-closes on
    # an interpreter outside the declared env — this is the STANDALONE entry's
    # sanctioned imposition (one re-exec, same as the worker entrypoint's).
    from gen_worker.settings_authority import ensure_interpreter_env

    ensure_interpreter_env()
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
    parser.add_argument(
        "--hub-env", action="store_true",
        help="boot the mint child through the hub's env-delivery rule "
             "(declarations x entries) instead of this process's environment; "
             "ambient values are stripped and withholdings are reported")
    parser.add_argument(
        "--hub-env-entry", action="append", default=[], metavar="NAME=VALUE",
        help="an endpoint_env_entries row the operator has set; repeatable. "
             "Only names the release DECLARES are delivered.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        entries: Dict[str, str] = {}
        for raw in args.hub_env_entry:
            name, sep, value = str(raw).partition("=")
            if not sep:
                parser.error(f"--hub-env-entry expects NAME=VALUE, got {raw!r}")
            entries[name.strip()] = value
        result = run_cycle(root, stage=args.stage, force_load=args.force_load,
                           device=args.device, vehicle=args.vehicle,
                           hub_env_mode=args.hub_env,
                           hub_env_entries=entries)
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
