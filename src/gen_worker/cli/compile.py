"""``gen-worker compile`` — pre-warm this card's compiled graphs.

pgw#1491. Paul: *"compile all graph-specializations for this card, if they
cannot be fetched already."* Three properties, all load-bearing:

**FETCH-FIRST.** Per specialization the hub is asked before anything is built —
it is the fleet artifact pool, so one machine's compile is every same-env
machine's fetch. Only a genuine miss pays for a build.

**NEVER BLOCKS SERVING.** This is a separate process. An endpoint that is up
keeps serving throughout, and its own background mint and this command key work
by the SAME CAS entries through the work ledger (``cli/work_ledger``): each
skips what the other landed, and whichever survives finishes the remainder.
That is Paul's "take over the running compile" implemented as a ledger rather
than as process adoption.

**WEIGHTLESS.** Nothing here downloads a checkpoint. The exported program is a
weightless trace and the mint runs against it, so ``compile`` legitimately runs
before ``download``.

## Address-free programs

Paul ruled the exported-program blob ADDRESS-FREE: ``torch.export.save`` is
deterministic per machine but produces different bytes across machines (14/14
graph identities matched, 0/14 blob digests did), so the blob is a local derived
artifact and only the graph HASH is portable. Consequently, when this machine's
graph CAS does not hold the program for a specialization, ``compile``
RE-DERIVES it locally from the committed source + lock (~2 min CPU) instead of
fetching one. There is no program-blob route to call and there must never be.

## The compiled floor

A graph executes atomically with by-reference weights — nothing pages
mid-graph — so below a certain grant no compiled specialization can ever be
armed. Under such a grant this command NO-OPS by name rather than minting
artifacts that could not be used. The floor is read from the endpoint's own
declared lane floor today; a measured peak-VRAM stamp (pgw#1494) supersedes it
when one exists. An UNKNOWN floor never reads as zero and never reads as
infinite: it simply does not fire the refusal.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import endpoint_lock as el
from . import work_ledger, workspace

logger = logging.getLogger(__name__)

#: Outcome vocabulary. Closed: every specialization ends in exactly one of
#: these and the summary counts them.
PRESENT = "present"        # already in this machine's CAS
FETCHED = "fetched"        # pulled from the hub's fleet pool
BUILT = "built"            # compiled here
CLAIMED = "claimed"        # another process holds the lease; it is doing it
BELOW_FLOOR = "below_floor"  # no grant this run targets could arm it
FAILED = "failed"


class CompileError(RuntimeError):
    """Compile could not start. Always names what was missing."""


@dataclass(frozen=True, slots=True)
class Spec:
    """One graph specialization to satisfy."""

    contract: str
    graph: str
    target: Any
    ingress: Any

    @property
    def short(self) -> str:
        return self.graph[:16]


@dataclass(frozen=True, slots=True)
class Outcome:
    spec: Spec
    state: str
    detail: str = ""
    wall_s: float = 0.0


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------


def specializations(lock_path: Path) -> Tuple[Spec, ...]:
    """Every specialization the committed lock claims, in document order.

    Read out of ``[derive].document`` rather than from a lifted top-level
    table, because nothing derivable is restated in the lock — a second list
    would be a second answer to what this endpoint compiles to.
    """
    block = el.read_derive_block(lock_path)
    if block is None:
        raise CompileError(
            f"{lock_path} carries no derive block — this endpoint has not been "
            f"traced.\n  Run `gen-worker lock` (author-time) first; the lock is "
            f"committed to the endpoint repo and distributed with it."
        )
    document = block.decoded()
    lanes = ((document.get("graphs") or {}).get("lanes")) or []
    out: List[Spec] = []
    for lane in lanes:
        for record in lane.get("graphs") or ():
            out.append(
                Spec(
                    contract=str(lane.get("contract") or ""),
                    graph=str(record.get("graph") or ""),
                    target=record.get("target"),
                    ingress=record.get("ingress"),
                )
            )
    return tuple(out)


# --------------------------------------------------------------------------
# The compiled floor
# --------------------------------------------------------------------------


def declared_floor_gb(endpoint_dir: Path) -> Optional[float]:
    """The endpoint's declared per-lane VRAM floor, or ``None`` if it declares
    none. ``None`` means UNKNOWN and is never treated as 0 or as infinite."""
    try:
        from ..discovery.discover import prime_sys_path
        from ..serving.loader import load_endpoint
        from ..serving.model import model_requires
    except Exception:  # noqa: BLE001 - an import failure is not a floor
        return None
    try:
        prime_sys_path(endpoint_dir)
        loaded = load_endpoint(endpoint_dir)
    except Exception:  # noqa: BLE001
        return None
    floors: List[float] = []
    for cls in loaded.models:
        for requirements in (model_requires(cls) or {}).values():
            # `model_requires` answers LayoutRequirements, whose floor lives on
            # `.minimum` (a RequirementTerms). Reading `min_vram_gb` off the
            # outer object returns None — which this function would have
            # rendered as "no floor declared" and the caller as "proceed". A
            # missing floor and a floor read one level too shallow are
            # indistinguishable downstream, so the access is explicit here.
            terms = getattr(requirements, "minimum", None) or requirements
            value = float(getattr(terms, "min_vram_gb", 0.0) or 0.0)
            if value > 0.0:
                floors.append(value)
    return min(floors) if floors else None


def grant_gb(stated: float) -> Optional[float]:
    """This run's VRAM grant: the stated one, else what the card has free.

    Probed ONCE and frozen (pgw#1495's shape), because a grant re-probed later
    is a grant that can disagree with the decision it justified.
    """
    if stated > 0.0:
        return stated
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return float(free) / float(1024 ** 3)
    except Exception:  # noqa: BLE001 - absent never renders as zero
        return None


# --------------------------------------------------------------------------
# Store + env
# --------------------------------------------------------------------------


def _env_identity(endpoint_dir: Path, sm: str, lockfile: Optional[Path]) -> Any:
    from .._vendor.torchcg.graph_identity import EnvIdentity
    from ..env_identity import (
        EnvIdentityError,
        compile_stack_from_lockfile,
        cuda_bucket,
        lockfile_beside,
    )

    resolved = lockfile or lockfile_beside(str(endpoint_dir))
    if resolved is None:
        raise CompileError(
            f"no uv.lock beside {endpoint_dir}: the artifact key carries the "
            f"COMPILE STACK (torch/triton/nvidia-*), and the lockfile is where "
            f"those versions are pinned. There is no second source."
        )
    try:
        stack = compile_stack_from_lockfile(Path(resolved), bucket=cuda_bucket())
    except EnvIdentityError as exc:
        raise CompileError(f"{resolved}: {exc}") from exc
    return EnvIdentity(stack=stack, sm=sm)


def _store(cas_root: Path) -> Any:
    """This machine's graph store, tiered under the hub when one is configured.

    The hub tier is READ-ONLY here: publishing a locally-built artifact back to
    the fleet pool is the pod-side mint's job (pgw#1471), which stamps driver
    range and measured peak. A CLI publishing unstamped artifacts would fill
    the pool with rows nothing can safely admit.
    """
    from ..serving.mint_store import worker_store

    return worker_store(Path(cas_root), None)


# --------------------------------------------------------------------------
# Per-specialization work
# --------------------------------------------------------------------------


def _ensure_program(spec: Spec, store: Any, rederive: Any, scratch: Path) -> Path:
    """The exported program for ``spec``, re-deriving it when this box lacks it.

    ADDRESS-FREE (Paul, 2026-08-19): ``torch.export.save`` is deterministic per
    machine but produces DIFFERENT bytes across machines for the same traced
    graph — 14/14 graph identities reproduce, 0/14 blob digests do. A serialized
    program is therefore a LOCAL derived artifact whose only portable name is
    the GRAPH, and ``graphs[].program`` has left the document contract
    entirely (torchcg 09c09b7a; `document.py` now contains zero occurrences).

    So there is exactly one lookup and it is by identity. A miss is answered by
    re-deriving locally — never by fetching, because there is nothing portable
    to fetch. A malformed key is a typed refusal from the store rather than a
    miss, so a wiring bug can never spend a pointless two-minute derive.
    """
    destination = scratch / f"{spec.short}.pt2"
    destination.parent.mkdir(parents=True, exist_ok=True)
    found = store.fetch_program(spec.graph, destination)
    if found is not None:
        return Path(found)
    rederive()
    found = store.fetch_program(spec.graph, destination)
    if found is None:
        raise CompileError(
            f"no exported program for graph {spec.short} even after re-deriving "
            f"from this source tree: the committed lock and this checkout "
            f"disagree about what this endpoint traces to. "
            f"`gen-worker lock --check` names the drift."
        )
    return Path(found)


def _build(
    spec: Spec, *, program: Path, cas_root: Path, sm: str, destination: Path,
) -> None:
    """Compile one specialization in its OWN process.

    A child, not a thread: inductor keeps process-global mutable state, and a
    mint that dies must not take the driver with it. ``destination`` is
    deliberately NOT created here — torchcg refuses a destination that already
    exists unless every byte matches, so an empty directory made "for tidiness"
    reads as occupied by something that is not the artifact.
    """
    from ..compile_cache import toolchain_digest

    request = {
        "blob": str(program),
        "graph": spec.graph,
        "target": spec.target,
        "ingress": spec.ingress,
        "target_arch": sm,
        "toolchain": dict(toolchain_digest()),
        "cas": str(cas_root),
        "destination": str(destination),
        "result": str(destination.parent / f"{spec.short}.result.json"),
    }
    request_path = destination.parent / f"{spec.short}.request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    argv = [
        "nice", "-n", "19",
        sys.executable, "-m", "gen_worker.serving.mint_child", str(request_path),
    ]
    # NO `env=`: a child inherits this process's environment by default, so
    # `env=dict(os.environ)` was a no-op that re-exported the whole environment
    # explicitly — an unresolvable bare `os.environ` binding that §1.18's guard
    # cannot classify, because "whatever happens to be set" is not a config
    # value anyone can name. The mint child reads its own inputs from the
    # request file written above; nothing here needs to hand it an env.
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise CompileError(
            f"mint child exited {completed.returncode} for graph {spec.short}"
        )


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def compile_all(
    *,
    endpoint_dir: Path,
    lock_path: Path,
    cas_root: Path,
    sm: str,
    lockfile: Optional[Path],
    only: int = 0,
    vram_budget_gb: float = 0.0,
) -> List[Outcome]:
    specs = specializations(lock_path)
    if only:
        specs = specs[:only]
    if not specs:
        logger.info(
            "compile: this endpoint's lock claims no compiled specializations "
            "— nothing to build (it serves eager by declaration)"
        )
        return []

    floor = declared_floor_gb(endpoint_dir)
    grant = grant_gb(vram_budget_gb)
    if floor is not None and grant is not None and grant < floor:
        logger.info(
            "compile: grant_below_compiled_floor(grant=%.1f GiB, floor=%.1f GiB) "
            "— a compiled graph executes atomically with by-reference weights, "
            "so nothing this endpoint compiles to could be armed under this "
            "grant. NO-OP: %d specialization(s) left unbuilt on purpose.",
            grant, floor, len(specs),
        )
        return [
            Outcome(spec, BELOW_FLOOR, f"grant {grant:.1f} GiB < floor {floor:.1f} GiB")
            for spec in specs
        ]
    if floor is None:
        logger.info(
            "compile: compiled floor UNKNOWN for this endpoint (no declared "
            "lane floor, no measured stamp) — proceeding; an unknown floor is "
            "not a floor of zero"
        )

    env = _env_identity(endpoint_dir, sm, lockfile)
    store = _store(cas_root)
    artifacts_dir = Path(endpoint_dir) / ".compiled-graphs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rederive_ran = [False]

    def rederive() -> None:
        """Regenerate this box's exported programs into the graph store, ONCE.

        Once per compile run, not once per specialization: one derive emits
        every graph the endpoint traces to, so a second call would re-pay a
        two-minute trace to produce bytes already on disk.
        """
        if not rederive_ran[0]:
            rederive_ran[0] = True
            _rederive_programs(endpoint_dir, cas_root, lockfile)

    outcomes: List[Outcome] = []
    for index, spec in enumerate(specs, start=1):
        started = time.monotonic()
        label = f"[{index}/{len(specs)}] {spec.contract} {spec.short}"
        try:
            if store.has_artifact(spec.graph, env):
                logger.info("%s: present", label)
                outcomes.append(Outcome(spec, PRESENT, wall_s=time.monotonic() - started))
                continue
        except Exception as exc:  # noqa: BLE001 — a store miss is not fatal
            logger.warning("%s: store lookup failed (%s); treating as a miss", label, exc)

        destination = artifacts_dir / spec.short
        try:
            with work_ledger.lease(Path(cas_root), f"{spec.graph}/{env.value}"):
                # Re-check under the lease: the holder we queued behind may
                # have landed exactly this artifact while we waited.
                if store.has_artifact(spec.graph, env):
                    logger.info("%s: present (landed while claimed)", label)
                    outcomes.append(
                        Outcome(spec, PRESENT, wall_s=time.monotonic() - started)
                    )
                    continue
                logger.info("%s: building", label)
                program = _ensure_program(spec, store, rederive, artifacts_dir)
                _build(
                    spec, program=program, cas_root=Path(cas_root), sm=sm,
                    destination=destination,
                )
                outcomes.append(
                    Outcome(spec, BUILT, wall_s=time.monotonic() - started)
                )
        except work_ledger.Busy:
            logger.info(
                "%s: claimed by another process — skipping (the work ledger is "
                "how compile and a serving mint share this)", label,
            )
            outcomes.append(Outcome(spec, CLAIMED, wall_s=time.monotonic() - started))
        except Exception as exc:  # noqa: BLE001 — one failure is not the run
            logger.error("%s: FAILED: %s: %s", label, type(exc).__name__, exc)
            outcomes.append(
                Outcome(spec, FAILED, f"{type(exc).__name__}: {exc}",
                        wall_s=time.monotonic() - started)
            )
    return outcomes


def _rederive_programs(
    endpoint_dir: Path, cas_root: Path, lockfile: Optional[Path]
) -> None:
    """Regenerate this machine's exported programs into the graph store.

    Returns nothing: since the address-free ruling the derive PUTS each program
    under its graph identity, and the caller asks the store by that identity.
    Handing back a map of addresses would be re-introducing exactly the
    per-machine address the ruling deleted.
    """
    import importlib

    from ..discovery.discover import prime_sys_path
    from ..env_identity import lockfile_beside
    from ..release.derive import DeriveError, derive_release
    from ..serving.loader import load_endpoint

    logger.info(
        "compile: exported program absent locally — re-deriving from committed "
        "source + lock (address-free blobs: the identity travels, the bytes "
        "do not)"
    )
    prime_sys_path(endpoint_dir)
    # `load_endpoint` is the ONE reader of endpoint.toml's `main =` and it has
    # already imported the module; re-importing by name is a dict lookup, not a
    # second load, and it keeps this file from parsing endpoint.toml itself.
    module = importlib.import_module(load_endpoint(endpoint_dir).module_name)
    try:
        derive_release(
            module,
            checkpoint_dir=endpoint_dir,
            lockfile=lockfile or lockfile_beside(str(endpoint_dir)),
            graph_cas=Path(cas_root),
            slot_checkpoints={},
        )
    except DeriveError as exc:
        raise CompileError(f"re-derive failed: {exc}") from exc


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    parser = sub.add_parser(
        "compile",
        help="Pre-warm this card's compiled graphs: fetch else build.",
        description=(
            "Satisfy every graph specialization the committed endpoint.lock "
            "claims, for this card's sm and this venv's compile stack. Fetches "
            "from the hub's fleet pool when it can, builds when it must, and "
            "shares the work with any serving process's background mint "
            "through the CAS work ledger. Weightless — no checkpoint needed."
        ),
    )
    parser.add_argument("endpoint_dir", nargs="?", default=".",
                        help="directory holding endpoint.toml (default: .)")
    parser.add_argument("--sm", default="",
                        help="target sm (e.g. sm_89); default: this card's")
    parser.add_argument("--graph-store", default="", metavar="DIR",
                        help="graph CAS root (default: the box graph CAS)")
    parser.add_argument("--lock", default="", metavar="PATH",
                        help="endpoint.lock (default: the one beside the endpoint)")
    parser.add_argument("--env-lockfile", default="", metavar="PATH",
                        help="uv.lock stating the compile stack (default: the "
                             "endpoint's own)")
    parser.add_argument("--only", type=int, default=0, metavar="N",
                        help="stop after N specializations (bring-up aid)")
    parser.add_argument("--vram-budget", type=float, default=0.0, metavar="GB",
                        help="the grant this compile targets; below the "
                             "endpoint's compiled floor it no-ops by name. "
                             "0 = probe this card once.")
    parser.set_defaults(_handler=run_compile)


def run_compile(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stderr, force=False,
    )
    endpoint_dir = Path(args.endpoint_dir).resolve()
    sm = args.sm or workspace.host_sm()
    if not sm:
        sys.stderr.write(
            "gen-worker compile: no --sm and no visible CUDA device. Compiled "
            "artifacts are keyed per-sm, so there is nothing to compile FOR. "
            "Pass --sm sm_89 to build for a card you will serve on.\n"
        )
        return 2
    lock_path = Path(args.lock) if args.lock else endpoint_dir / el.LOCK_FILENAME
    cas_root = Path(args.graph_store) if args.graph_store else workspace.graph_cas_root()
    try:
        outcomes = compile_all(
            endpoint_dir=endpoint_dir,
            lock_path=lock_path,
            cas_root=cas_root,
            sm=sm,
            lockfile=Path(args.env_lockfile) if args.env_lockfile else None,
            only=int(args.only or 0),
            vram_budget_gb=float(args.vram_budget or 0.0),
        )
    except CompileError as exc:
        sys.stderr.write(f"gen-worker compile: {exc}\n")
        return 1
    counts: Dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.state] = counts.get(outcome.state, 0) + 1
    sys.stderr.write(
        "gen-worker compile: "
        + ", ".join(f"{state}={counts[state]}" for state in sorted(counts))
        + f" (of {len(outcomes)})\n"
    )
    return 1 if counts.get(FAILED) else 0


__all__ = [
    "BELOW_FLOOR",
    "BUILT",
    "CLAIMED",
    "CompileError",
    "FAILED",
    "FETCHED",
    "Outcome",
    "PRESENT",
    "Spec",
    "add_subparser",
    "compile_all",
    "run_compile",
    "specializations",
]
