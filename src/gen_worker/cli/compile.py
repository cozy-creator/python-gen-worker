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

**SERVABLE, OR IT FAILED (pgw#1533).** The success criterion is not "the builds
returned". It is "the reader that arms graphs at boot can find what I wrote",
and it is checked by asking THAT reader, through a store object this command did
not publish through. Before this, ``compile`` reported ``built=14 (of 14)`` with
rc=0 after 26 minutes of real inductor work and left the serving path with 14
holes: ``Engine.compile`` banks bytes in torchcg's own engine cache, and
NOTHING lands in the ``(cg-graph-v1, cg-env-v2)`` band adoption reads unless
somebody calls ``publish_artifact`` — which the runtime mint did and this
command did not. It also never called ``put_graphs``, so even a correctly placed
artifact had no graph-set document to be found through. Both are now this
command's job, and the read-back is what proves it: a count of builds that
returned cannot go red on either failure, and did not.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import endpoint_lock as el
from . import work_ledger, workspace

logger = logging.getLogger(__name__)

#: Compile one specialization: (spec, program blob, destination) -> artifact.
#: The production implementation is :func:`_build` (one child process per
#: graph); an explicit one is the local seam — no GPU, no inductor — and
#: nothing else.
Builder = Callable[["Spec", Path, Path], Path]

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

    A MISS ARRIVES TWO WAYS, AND THIS ASKED FOR ONLY ONE (pgw#1525).
    ``LocalGraphStore.fetch_program`` returns ``None`` for a miss;
    ``WorkerGraphStore.fetch_program`` — which is what ``_store`` actually
    builds — RAISES ``ProgramBlobUnreachable``, and its own message calls the
    condition "ORDINARY". This function only tested for ``None``, so on the one
    state it exists for — a box that has never derived this endpoint — the
    raise escaped past ``rederive()`` and the re-derive branch was unreachable.
    Measured: `compile` on a wiped store answered ``failed=14 (of 14)`` in
    6.04 s having never logged "re-deriving", and told the user to run
    `compile`.

    Both spellings are now the same miss, and the answer to a miss is to derive.
    That includes the ``rotten`` arm of the same refusal: a serialized program
    is a DERIVED, DISPOSABLE, machine-scoped artifact, so bytes that fail their
    own scrub deserve exactly what absent bytes deserve — regeneration, not a
    report. The refusal survives where it is load-bearing: a malformed graph key
    is not a miss, so it is re-raised rather than paying for a derive that
    cannot possibly satisfy it, and a program still missing AFTER the derive is
    the typed ``CompileError`` below.
    """
    from .._vendor.torchcg import is_graph_hash
    from ..serving.mint_store import ProgramBlobUnreachable

    # Checked HERE, by the predicate, rather than recognised from the refusal's
    # wording. A malformed key is a wiring bug and no derive can satisfy it, so
    # it must not be swallowed as a miss; matching on message text to tell the
    # two apart would put this function one prose edit from silently paying for
    # a two-minute derive on every specialization.
    if not is_graph_hash(spec.graph):
        raise CompileError(
            f"{spec.graph!r} is not a cg-graph-v1 identity — the lock names a "
            f"graph no store can be asked for. `gen-worker lock` rewrites it."
        )

    destination = scratch / f"{spec.short}.pt2"
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _look() -> Optional[Path]:
        try:
            found = store.fetch_program(spec.graph, destination)
        except ProgramBlobUnreachable as exc:
            logger.debug("%s: no usable local program (%s)", spec.short, exc)
            return None
        return Path(found) if found is not None else None

    program = _look()
    if program is not None:
        return program
    rederive()
    program = _look()
    if program is None:
        raise CompileError(
            f"no exported program for graph {spec.short} even after re-deriving "
            f"from this source tree: the committed lock and this checkout "
            f"disagree about what this endpoint traces to. "
            f"`gen-worker lock --check` names the drift."
        )
    return program


def _build(
    spec: Spec, *, program: Path, cas_root: Path, sm: str, destination: Path,
) -> Path:
    """Compile one specialization in its OWN process, and return its artifact.

    A child, not a thread: inductor keeps process-global mutable state, and a
    mint that dies must not take the driver with it. ``destination`` is
    deliberately NOT created here — torchcg refuses a destination that already
    exists unless every byte matches, so an empty directory made "for tidiness"
    reads as occupied by something that is not the artifact.

    The child writes the artifact path it produced into ``result``, and this
    function READS it (pgw#1533). The request already asked for that file and
    nothing consumed it: the parent threw the child's only output away, so it
    had nothing to publish and could only count exit statuses — which is
    precisely how a green run left the serving path empty.
    """
    from ..compile_cache import toolchain_digest

    result_path = destination.parent / f"{spec.short}.result.json"
    request = {
        "blob": str(program),
        "graph": spec.graph,
        "target": spec.target,
        "ingress": spec.ingress,
        "target_arch": sm,
        "toolchain": dict(toolchain_digest()),
        "cas": str(cas_root),
        "destination": str(destination),
        "result": str(result_path),
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
    try:
        artifact = Path(result_path.read_text(encoding="utf-8").strip())
    except OSError as exc:
        raise CompileError(
            f"the mint child for graph {spec.short} exited 0 but wrote no "
            f"artifact path to {result_path} ({exc}) — a zero exit is not an "
            f"artifact, and there is nothing to publish"
        ) from exc
    if not artifact.exists():
        raise CompileError(
            f"the mint child for graph {spec.short} named {artifact} and it "
            f"does not exist"
        )
    return artifact


def _default_builder(cas_root: Path, sm: str) -> Builder:
    """THE production builder: one child process per specialization."""

    def build(spec: Spec, program: Path, destination: Path) -> Path:
        return _build(
            spec, program=program, cas_root=Path(cas_root), sm=sm,
            destination=destination,
        )

    return build


def endpoint_module(endpoint_dir: Path) -> str:
    """The module name adoption looks this endpoint's document up BY.

    Read through ``load_endpoint`` — the one reader of ``endpoint.toml``'s
    ``main =`` — because ``_adoption_source`` reads it the same way. Two
    spellings of "which document is this endpoint's" is exactly the drift that
    would make ``compile`` publish under a name ``up`` never asks for.

    An import failure here is a TYPED refusal, not a traceback (pgw#1537).
    pgw#1533 made this the first thing `compile` needs that can fail on the
    author's own code — `declared_floor_gb` calls `load_endpoint` too, but
    swallows everything, because an unreadable floor is genuinely "no floor
    stated". An unreadable MODULE NAME is not "no name": nothing can be
    published under a name that could not be read, so the run cannot deliver a
    servable endpoint and must say why in one sentence.
    """
    from ..discovery.discover import prime_sys_path
    from ..serving.loader import load_endpoint

    try:
        prime_sys_path(endpoint_dir)
        return str(load_endpoint(endpoint_dir).module_name)
    except Exception as exc:  # noqa: BLE001 — author code; any failure is theirs
        raise CompileError(
            f"cannot read {endpoint_dir}'s module name "
            f"({type(exc).__name__}: {exc}).\n"
            f"  `compile` publishes this endpoint's graph-set document under "
            f"that name and boot-time adoption looks it up by the same one, so "
            f"a document published under a guess would be invisible to `up`.\n"
            f"  Fix the import (this is the endpoint's own `main =` module), or "
            f"pass the name explicitly if you are compiling for a tree you "
            f"cannot import here."
        ) from exc


# --------------------------------------------------------------------------
# The serving reader — the only witness that counts
# --------------------------------------------------------------------------


def serving_reader(cas_root: Path) -> Any:
    """The store object BOOT-TIME ADOPTION builds, over this machine's CAS.

    Constructed exactly the way ``cli/daemon._adoption_source`` constructs it,
    and deliberately NOT the store this command publishes through. "My writer
    returned without raising" and "the reader that arms graphs can find it" are
    different claims, and pgw#1533 is what happens when only the first one is
    ever checked: the write succeeded, into torchcg's engine cache, and the
    reader saw nothing.
    """
    from .._vendor.tensorfs import LocalCAS
    from .._vendor.torchcg.store import LocalGraphStore

    return LocalGraphStore(LocalCAS(Path(cas_root)))


def unservable(cas_root: Path, specs: Tuple[Spec, ...], env: Any, module: str) -> List[str]:
    """What the serving reader still cannot find. Empty means servable.

    Reports the DOCUMENT as its own row: an endpoint whose artifacts are all
    present but whose graph-set document is missing adopts nothing, because
    adoption enumerates lanes out of the document and never reaches the
    artifacts. That was the second half of the same silent failure — nothing in
    this CLI called ``put_graphs``, so ``get_graphs`` answered a clean miss and
    ``up`` served eager over a full store.
    """
    reader = serving_reader(cas_root)
    gaps: List[str] = []
    try:
        document = reader.get_graphs(module)
    except Exception as exc:  # noqa: BLE001 — unreadable IS unservable
        document = None
        gaps.append(f"graph-set document {module!r}: unreadable ({exc})")
    if document is None and not gaps:
        gaps.append(f"graph-set document {module!r}: absent")
    for spec in specs:
        try:
            present = reader.has_artifact(spec.graph, env)
        except Exception as exc:  # noqa: BLE001
            gaps.append(f"artifact {spec.short}: unreadable ({exc})")
            continue
        if not present:
            gaps.append(f"artifact {spec.short}: absent at ({spec.short}, {env.value})")
    return gaps


def _publish_document(cas_root: Path, lock_path: Path, module: str) -> None:
    """Publish the committed lock's graph-set document into this box's store.

    The document is not derived here and must not be: the lock IS the authored
    document, ``specializations()`` already reads its ``[derive].document``
    block for the very specs being built, and adoption looks the same document
    up by module name. Writing a second one would be a second answer to what
    this endpoint compiles to.
    """
    from .._vendor.torchcg.document import GraphSetDocument

    block = el.read_derive_block(lock_path)
    if block is None:  # unreachable via compile_all, which refuses earlier
        return
    document = GraphSetDocument.decode(block.decoded()["graphs"])
    serving_reader(cas_root).put_graphs(module, document)
    logger.info(
        "compile: published graph-set document %s (%d lane(s), %d graph(s)) — "
        "adoption enumerates lanes out of this; artifacts alone adopt nothing",
        module, len(document.lanes),
        sum(len(lane.graphs) for lane in document.lanes),
    )


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Report:
    """What one ``compile`` run did, and whether it left anything servable.

    ``unservable`` is the verdict and ``outcomes`` is the narrative. They are
    separate fields because they answer different questions and the whole of
    pgw#1533 is that the narrative was mistaken for the verdict: fourteen
    ``BUILT`` rows and rc=0 over a serving path that could arm nothing.
    """

    outcomes: List[Outcome]
    unservable: List[str]


def compile_all(
    *,
    endpoint_dir: Path,
    lock_path: Path,
    cas_root: Path,
    sm: str,
    lockfile: Optional[Path],
    only: int = 0,
    vram_budget_gb: float = 0.0,
    module: str = "",
    store: Any = None,
    builder: Optional[Builder] = None,
) -> Report:
    specs = specializations(lock_path)
    if only:
        specs = specs[:only]
    if not specs:
        logger.info(
            "compile: this endpoint's lock claims no compiled specializations "
            "— nothing to build (it serves eager by declaration)"
        )
        return Report([], [])

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
        return Report(
            [
                Outcome(spec, BELOW_FLOOR,
                        f"grant {grant:.1f} GiB < floor {floor:.1f} GiB")
                for spec in specs
            ],
            [],
        )
    if floor is None:
        logger.info(
            "compile: compiled floor UNKNOWN for this endpoint (no declared "
            "lane floor, no measured stamp) — proceeding; an unknown floor is "
            "not a floor of zero"
        )

    from ..serving.mint import publish_compiled

    env = _env_identity(endpoint_dir, sm, lockfile)
    if store is None:
        store = _store(cas_root)
    build = builder if builder is not None else _default_builder(cas_root, sm)
    module = module or endpoint_module(endpoint_dir)
    # pgw#1526: the BOX cache, not `<endpoint>/.compiled-graphs`. This is mint
    # SCRATCH plus the pre-publish destination — machine-scoped by nature, and
    # nothing downstream reads it from the source tree: the artifact's only
    # durable address is the CAS entry `publish_compiled` writes below, keyed
    # (graph x env). Writing it into the endpoint made it look like endpoint
    # content and shipped 172 MB of it in a source tarball (cl#88).
    #
    # Box-wide is safe to SHARE because the leaf is `spec.short` — the graph
    # identity, content-addressed over the exported program — so two endpoints
    # collide only when they trace to the same graph, in which case the CAS
    # deduplicates them anyway.
    artifacts_dir = workspace.artifacts_root()
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
                artifact = build(spec, program, destination)
                # PUBLISH, then ASK THE READER. Neither step existed before
                # pgw#1533 and neither one alone is enough: the publish is what
                # puts the bytes in the band adoption addresses, and the
                # read-back is the only thing that can go red if it did not.
                published = publish_compiled(store, spec.graph, env, artifact)
                if not serving_reader(cas_root).has_artifact(spec.graph, env):
                    raise CompileError(
                        f"built {spec.short} and the publish reported "
                        f"{published or 'nothing'}, and the store adoption "
                        f"reads at boot still has no artifact at "
                        f"({spec.short}, {env.value}). The build is not the "
                        f"deliverable — an artifact the serving reader can "
                        f"find is."
                    )
                logger.info("%s: built and servable (%s)", label, published or "published")
                outcomes.append(
                    Outcome(spec, BUILT, published, wall_s=time.monotonic() - started)
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

    # The document is published for the WHOLE run, not per specialization: it
    # names every lane the lock claims, so a partial run still leaves adoption
    # able to enumerate and the mint able to fill the rest. Publishing it after
    # the artifacts is the same durability order the runtime mint uses —
    # nothing is announced before it exists.
    _publish_document(cas_root, lock_path, module)
    gaps = unservable(cas_root, specs, env, module)
    return Report(outcomes, gaps)


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


def summarize(report: Report) -> Tuple[str, int]:
    """What the operator is told, and what the shell is told, from ONE fact.

    Separated from :func:`run_compile` so the verdict is testable against real
    reports rather than only through argparse and a live inductor. The whole of
    pgw#1533 lives in the second element: for 26 minutes of real GPU work this
    returned 0 beside ``built=14 (of 14)`` while the serving path held 14 holes,
    because the exit code was computed from the outcome tally and the tally
    counted builds that returned.
    """
    counts: Dict[str, int] = {}
    for outcome in report.outcomes:
        counts[outcome.state] = counts.get(outcome.state, 0) + 1
    lines = [
        "gen-worker compile: "
        + ", ".join(f"{state}={counts[state]}" for state in sorted(counts))
        + f" (of {len(report.outcomes)})\n"
    ]
    if report.unservable:
        # NOT a warning. A run that leaves the serving path unable to arm what
        # it was asked for has failed at the only thing it was for, and saying
        # so quietly beside a green summary is the defect this fixed, not a
        # style choice.
        lines.append(
            f"gen-worker compile: NOT SERVABLE — {len(report.unservable)} "
            f"gap(s) in the store `gen-worker up` adopts from:\n"
        )
        lines.extend(f"  - {gap}\n" for gap in report.unservable)
        lines.append(
            "  A build that returned is not an artifact the serving path can "
            "find; this command reports the second.\n"
        )
        return "".join(lines), 1
    if counts.get(FAILED):
        return "".join(lines), 1
    if counts.get(BELOW_FLOOR) or not report.outcomes:
        return "".join(lines), 0
    lines.append(
        f"gen-worker compile: SERVABLE — the graph-set document and all "
        f"{len(report.outcomes)} artifact(s) are readable through the store "
        f"boot-time adoption uses.\n"
    )
    return "".join(lines), 0


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
        report = compile_all(
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
    summary, code = summarize(report)
    sys.stderr.write(summary)
    return code


__all__ = [
    "BELOW_FLOOR",
    "BUILT",
    "CLAIMED",
    "Builder",
    "CompileError",
    "FAILED",
    "FETCHED",
    "Outcome",
    "PRESENT",
    "Report",
    "Spec",
    "add_subparser",
    "compile_all",
    "endpoint_module",
    "run_compile",
    "serving_reader",
    "specializations",
    "summarize",
    "unservable",
]
