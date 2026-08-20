"""``gen-worker compile`` — pre-warm this card's compiled graphs.

pgw#1491. Paul: *"compile all graph-specializations for this card, if they
cannot be fetched already."* Three properties, all load-bearing:

**PRESENCE-FIRST, THROUGH THE ONE STORE.** Per specialization the store is
asked before anything is built — local CAS first, then the fleet pool when this
process has a release to adopt for (a pod does; a box `compile` on a committed
lock does not, and that is a stated `upstream=None`, not a different store).
Only a genuine miss pays for a build.

pgw#1573 corrected this paragraph, which claimed a hub consult this verb has
never had: `_store` passed `upstream=None` and the `FETCHED` outcome constant
was defined and never assigned anywhere in `src/`. A vocabulary member nothing
can emit is a proof condition that passes unconditionally, so it is deleted
rather than left reading like coverage.

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

**FIRST THE ONE THAT IS NEEDED (pgw#1545).** Paul, 2026-08-20: *"prioritize
generating the graph specialization for what we want to run first; the other
specializations can be compiled in the background while we do inference using
the .so that was compiled for the workflow we're actually using."* So the unit
of delivery is not the run, it is the FIRST SPECIALIZATION: ``--first`` names
the one the caller's workflow needs (defaulting to the document's own first
record, which the derive states is the all-defaults specialization — tcg
``document.py``, pgw#1384), the graph-set document is published BEFORE any
build, and under ``--fill background`` the verb returns the instant that one
artifact is servable while a detached child fills the rest. Measured before
this: 14 specializations at ~111 s each, nothing servable for ~26 minutes.

## Address-free programs

Paul ruled the exported-program blob ADDRESS-FREE: ``torch.export.save`` is
deterministic per machine but produces different bytes across machines (14/14
graph identities matched, 0/14 blob digests did), so the blob is a local derived
artifact and only the graph HASH is portable. Consequently, when this machine's
graph CAS does not hold the program for a specialization, ``compile``
RE-DERIVES it locally from the committed source + lock (~2 min CPU) instead of
fetching one. There is no program-blob route to call and there must never be.

## No floor (pgw#1587)

This command MINTS; it does not decide what a particular card can arm. It used
to no-op the whole run when the endpoint's declared lane number exceeded this
run's VRAM grant — a refusal to BUILD, taken from a declaration, sized to the
whole pipeline rather than to the compiled graph. Deleted. Servability is
settled on the serving card at load, by probe, and a card that cannot hold the
graph's working set serves EAGER with a typed reason. See the block above
:func:`_env_identity` for the whole argument.
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
PRESENT = "present"        # already readable through the store (local or fleet)
BUILT = "built"            # compiled here
REUSED = "reused"          # resolved out of the engine's own compile cache
CLAIMED = "claimed"        # another process holds the lease; it is doing it
FAILED = "failed"
# pgw#1587: `BELOW_FLOOR` is DELETED. A declared lane requirement is a claim
# about what a COMPILED GRAPH needs of a card, verified by a probe at serve —
# never permission to build. Nothing here refuses a mint on a declaration.

#: What this run does with the specializations that are NOT the first one.
#: Closed, and stated rather than inferred from a flag combination.
FILL_ALL = "all"                # build them here, in this process, in order
FILL_BACKGROUND = "background"  # hand them to a detached child and return
FILL_NONE = "none"              # build only the first one; say what was left
FILLS = (FILL_ALL, FILL_BACKGROUND, FILL_NONE)


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
# Which specialization goes first
# --------------------------------------------------------------------------


def facets(spec: Spec) -> Tuple[str, ...]:
    """Every name ``--first`` may address ONE specialization by.

    A specialization has no single human name — it is a graph identity, and an
    operator asking for "the 512x512 unet" is naming properties of its ingress.
    So the selector matches FACETS: the graph identity (and its short form),
    the lane contract, the target module path, and, from the ingress contract,
    each input's parameter name, dtype and ``AxBxC`` shape. One relation, so a
    refusal can print exactly what was addressable.
    """
    out: List[str] = [spec.graph, spec.short, spec.contract, spec.target]
    ingress = spec.ingress if isinstance(spec.ingress, dict) else {}
    for row in ingress.get("inputs") or ():
        if not isinstance(row, dict):
            continue
        out.append(str(row.get("param") or ""))
        out.append(str(row.get("dtype") or ""))
        shape = row.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            out.append("x".join(str(dimension) for dimension in shape))
    return tuple(dict.fromkeys(name for name in out if name))


def _selects(spec: Spec, term: str) -> bool:
    """One selector term against one specialization.

    Equality against a facet, or a prefix of the graph identity — a graph hash
    is 68 characters and nobody types one, but a distinguishing prefix is how
    every other content-addressed tool is driven. Eight characters minimum, so
    a short word can never accidentally prefix-match a hash.
    """
    if term in facets(spec):
        return True
    return len(term) >= 8 and spec.graph.startswith(term)


def select(specs: Tuple[Spec, ...], selector: str) -> Spec:
    """The specialization ``selector`` names — the one built FIRST.

    An empty selector is the DOCUMENT'S OWN default: the first record in
    document order. That is not "whatever happens to be first" — graph order is
    semantic and the derive states it, putting the all-defaults specialization
    first (torchcg ``document.py``, pgw#1384), which is exactly the one an
    unstated workflow runs.

    A selector matching nothing REFUSES. Falling back to the default would
    build a specialization the caller did not ask for and report success, and
    the whole point of the argument is that the first artifact is the one that
    serves.
    """
    if not selector.strip():
        return specs[0]
    terms = [term.strip() for term in selector.split(",") if term.strip()]
    for spec in specs:
        if all(_selects(spec, term) for term in terms):
            return spec
    addressable = sorted({name for spec in specs for name in facets(spec)
                          if not name.startswith("cg-graph-v1-")})
    raise CompileError(
        f"--first {selector!r} names no specialization this endpoint has.\n"
        f"  {len(specs)} specialization(s) are addressable by lane contract, "
        f"target, input parameter, dtype, AxBxC shape, or a graph-identity "
        f"prefix (>= 8 chars).\n"
        f"  Addressable names here: {', '.join(addressable) or '(none)'}"
    )


def order(specs: Tuple[Spec, ...], selector: str) -> Tuple[Spec, ...]:
    """``specs`` with the selected one first, everything else in document order.

    Document order is preserved for the remainder rather than re-sorted: it is
    the producer's stated priority for everything after the caller's own ask.
    """
    first = select(specs, selector)
    return (first,) + tuple(spec for spec in specs if spec is not first)


# --------------------------------------------------------------------------
# pgw#1587: THERE IS NO COMPILED FLOOR HERE ANY MORE
# --------------------------------------------------------------------------
#
# `declared_floor_gb` + `grant_gb` used to read the endpoint's declared
# per-lane VRAM number, compare it against this run's grant, and NO-OP the
# whole mint when the grant was smaller ("nothing this endpoint compiles to
# could be armed under this grant"). Two things were wrong with it and Paul
# named both (2026-08-20):
#
#   "Remove this '12gb floor'. I don't even know what this is. I didn't ask
#    for this. this is the completely wrong model."
#   "having some memory requirement per tensor-layout-contract makes sense.
#    But yeah, this limit is too high. And we should be able to serve
#    no-compiled below this limit."
#
#  1. THE NUMBER DESCRIBED THE WRONG THING. A declared lane number sized the
#     whole PIPELINE, while what a compiled graph needs of a card is its OWN
#     working set — its weights BY REFERENCE plus its activation peak — with
#     every component outside the graph free to be offloaded to make room.
#     SDXL is the case: park the two text encoders and the compiled UNet
#     serves on a 7.3 GiB card the 12 GiB number refused outright.
#  2. BUILDING IS NOT SERVING. This command mints artifacts; whether a
#     PARTICULAR card can arm one is decided on that card, at load, by probe
#     (`adopt_fit`, `partial_resident.probe_plan`) and never by a declaration
#     read here. A card that cannot hold the graph serves EAGER, loudly, with
#     a typed reason — it never refuses the job.
#
# The declaration itself is NOT deleted: `lanes={contract: "vram7g"}` stays as
# the per-tensor-layout-contract memory requirement — the hub's buying signal
# and the claim the probe verifies. What is deleted is its power to refuse.

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
    """THE store — the same constructor a pod and `gen-worker up` build.

    ``upstream=None``: a box compiling from a committed lock has no release to
    adopt for, and the fleet artifact route is release-scoped. That is a stated
    absence, not a second kind of store — and it is the ONLY thing that differs
    between this call and the pod's.

    Publishing a locally-built artifact back to the fleet pool is the pod-side
    mint's job (pgw#1471), which stamps driver range and measured peak. A CLI
    publishing unstamped artifacts would fill the pool with rows nothing can
    safely admit.
    """
    from ..serving.mint_store import graph_store

    return graph_store(Path(cas_root))


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
    from ..toolchain import toolchain_digest

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

    Read through ``loader.endpoint_module_name`` — the one reader of
    ``endpoint.toml``'s ``main =``, and the same one ``load_endpoint`` (which
    ``_adoption_source``'s callers use) resolves through. Two spellings of
    "which document is this endpoint's" is exactly the drift that would make
    ``compile`` publish under a name ``up`` never asks for.

    Deliberately NOT ``load_endpoint`` itself (pgw#1546): importing the
    author's module drags in torch and the model libraries — ~4.5 s measured
    on a warm all-present run — to answer a question the manifest states in
    one line. The name is what adoption keys by; the import proves nothing
    the publish needs.

    An unreadable name is a TYPED refusal, not a traceback (pgw#1537): nothing
    can be published under a name that could not be read, so the run cannot
    deliver a servable endpoint and must say why in one sentence.
    """
    from ..serving.loader import endpoint_module_name

    try:
        return endpoint_module_name(endpoint_dir)
    except Exception as exc:  # noqa: BLE001 — the manifest is the author's too
        raise CompileError(
            f"cannot read {endpoint_dir}'s module name "
            f"({type(exc).__name__}: {exc}).\n"
            f"  `compile` publishes this endpoint's graph-set document under "
            f"that name and boot-time adoption looks it up by the same one, so "
            f"a document published under a guess would be invisible to `up`.\n"
            f"  Fix endpoint.toml's `main = \"pkg.module\"`, or pass the name "
            f"explicitly if you are compiling for a tree you cannot read here."
        ) from exc


# --------------------------------------------------------------------------
# The serving reader — the only witness that counts
# --------------------------------------------------------------------------


def serving_reader(cas_root: Path) -> Any:
    """The store object BOOT-TIME ADOPTION builds, over this machine's CAS.

    pgw#1573: this used to be a DELIBERATELY DIFFERENT object from the one this
    command publishes through — an independence that was never the property
    that mattered. pgw#1533's defect was two BANDS (the engine's compile cache
    versus the ``(cg-graph-v1, cg-env-v2)`` serving band), not two Python
    objects over one band, and pgw#1561's defect was two FORMATS. Neither is
    detected by holding a second handle, and both are detected by the read-back
    below actually opening the bytes. So there is one store, built by
    :func:`_store`, and the witness is what proves it.

    Kept as a function because callers pass a CAS root, not a store — and
    because a second construction over the same root is now provably the same
    answer rather than a hedge against the writer.
    """
    return _store(Path(cas_root))


@dataclass(frozen=True, slots=True)
class Gap:
    """One thing the serving reader cannot find, ATTRIBUTED to its graph.

    ``graph`` is ``""`` for the graph-set document row. The attribution is what
    lets the verdict tell a gap this run promised to close from one it
    deliberately deferred to the background fill (pgw#1545) — a bare sentence
    could only be told apart by matching prose, which is how a vocabulary
    becomes folklore.
    """

    graph: str
    detail: str

    def __str__(self) -> str:
        return self.detail


def unservable(cas_root: Path, specs: Tuple[Spec, ...], env: Any, module: str) -> List[Gap]:
    """What the serving reader still cannot find. Empty means servable.

    Reports the DOCUMENT as its own row: an endpoint whose artifacts are all
    present but whose graph-set document is missing adopts nothing, because
    adoption enumerates lanes out of the document and never reaches the
    artifacts. That was the second half of the same silent failure — nothing in
    this CLI called ``put_graphs``, so ``get_graphs`` answered a clean miss and
    ``up`` served eager over a full store.
    """
    reader = serving_reader(cas_root)
    gaps: List[Gap] = []
    try:
        document = reader.get_graphs(module)
    except Exception as exc:  # noqa: BLE001 — unreadable IS unservable
        document = None
        gaps.append(Gap("", f"graph-set document {module!r}: unreadable ({exc})"))
    if document is None and not gaps:
        gaps.append(Gap("", f"graph-set document {module!r}: absent"))
    for spec in specs:
        try:
            present = reader.has_artifact(spec.graph, env)
        except Exception as exc:  # noqa: BLE001
            gaps.append(Gap(spec.graph, f"artifact {spec.short}: unreadable ({exc})"))
            continue
        if not present:
            gaps.append(Gap(
                spec.graph,
                f"artifact {spec.short}: absent at ({spec.short}, {env.value})"))
            continue
        # PRESENCE IS NOT LOADABILITY (pgw#1561). For as long as this census
        # stopped at `has_artifact`, a store full of bare `.pt2` ZIPs — bytes
        # the adopt loader refuses by TYPE — printed `SERVABLE`, and va#3
        # arm 2 paid the difference: 0/14 armed over 14 present positions. The
        # probe is two magic bytes per position (`artifact_skew`), so the
        # fully-warm census stays proportional to positions, not bytes; the
        # full open-through-the-real-loader proof runs where bytes change, in
        # :func:`witness_materializes` on every publish.
        try:
            skew = reader.artifact_skew(spec.graph, env)
        except Exception as exc:  # noqa: BLE001 — an unreadable probe is a gap
            skew = f"format probe failed ({exc})"
        if skew:
            gaps.append(Gap(spec.graph, f"artifact {spec.short}: {skew}"))
    return gaps


def witness_materializes(cas_root: Path, graph: str, env: Any) -> Optional[str]:
    """Fetch through the store adoption uses and OPEN through the real loader.

    ``None`` means the artifact will load at boot; a sentence names why it
    will not. This is the [[pgw#1533]] read-back with its blind spot removed
    (pgw#1561): `has_artifact` proved presence, and presence was never the
    gap — the freshly-built probe artifact of 2026-08-20 09:22 was PRESENT,
    `SERVABLE` was printed, and its bytes were a ZIP no loader opens. So the
    witness now materializes the fetched bytes with ``torchcg.serve
    .materialize`` — the exact function boot-time arming calls — and only
    that is allowed to say servable.

    Runs once per PUBLISHED artifact (14 × ~0.3 s on a full sd15 backfill),
    never on the warm present path — the census's two-byte skew probe covers
    that at position cost.
    """
    import tempfile

    from .._vendor.torchcg.serve import materialize

    reader = serving_reader(cas_root)
    with tempfile.TemporaryDirectory(prefix="pgw1561-witness-") as raw:
        destination = Path(raw) / "artifact"
        try:
            fetched = reader.fetch_artifact(graph, env, destination)
        except Exception as exc:  # noqa: BLE001 — a failed fetch is the verdict
            return f"fetch through the adoption store failed: {exc}"
        if fetched is None:
            return f"absent at ({graph[:16]}, {env.value})"
        try:
            materialize(fetched, Path(raw) / "unpacked")
        except Exception as exc:  # noqa: BLE001 — an unloadable blob is the verdict
            return f"fetched but does not MATERIALIZE ({type(exc).__name__}: {exc})"
    return None


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


class _EngineReuse:
    """Already-minted artifacts in the engine's own compile cache, resolvable
    WITHOUT torch, a program blob, or a child process (pgw#1546).

    ``Engine.compile`` banks every mint under its exact cg-key, and the only
    thing this command lacked to reuse one was the KEY — whose only derivation
    path loaded the exported program inside a mint child: measured 5.47 s per
    specialization of torch import + ``export.load`` bookkeeping against
    0.21 s of actual store work, 14 times over, to rediscover work already
    done. ``Engine.reuse_index`` answers the key from the store's own metadata,
    torch-free, and ``Engine.resolve`` fully verifies whatever it answers — an
    ADDRESS optimization, never an admission shortcut.

    The index is built lazily, once, on the first specialization the serving
    band is missing; a run whose artifacts are all published never pays it.
    Every failure here answers ``None`` — the build path is always the fallback.
    """

    def __init__(self, cas_root: Path, sm: str) -> None:
        self._cas_root = Path(cas_root)
        self._sm = sm
        self._engine: Any = None
        self._index: Optional[Dict[str, str]] = None

    def _load(self) -> None:
        from ..toolchain import toolchain_digest
        from .._vendor.tensorfs import LocalCAS
        from .._vendor.torchcg.engine import Engine

        self._engine = Engine(LocalCAS(self._cas_root))
        try:
            self._index = dict(
                self._engine.reuse_index(self._sm, dict(toolchain_digest()))
            )
        except Exception as exc:  # noqa: BLE001 — an unreadable cache is a miss
            logger.warning(
                "compile: engine reuse index unavailable (%s: %s); every miss "
                "will build", type(exc).__name__, exc,
            )
            self._index = {}
        if self._index:
            logger.info(
                "compile: engine cache holds %d already-minted artifact(s) for "
                "this (sm x toolchain); reusing instead of rebuilding",
                len(self._index),
            )

    def offers(self, graph: str) -> bool:
        """Whether the cache CLAIMS a mint for ``graph`` — an address answer;
        the claim is only trusted after :meth:`resolve` fully verifies it."""
        if self._index is None:
            self._load()
        assert self._index is not None
        return graph in self._index

    def resolve(self, spec: Spec, destination: Path) -> Optional[Path]:
        """The verified artifact directory for ``spec``, or ``None`` to build."""
        if self._index is None:
            self._load()
        assert self._index is not None
        key = self._index.get(spec.graph)
        if key is None:
            return None
        try:
            found = self._engine.resolve(key, destination)
        except Exception as exc:  # noqa: BLE001 — a rotten row costs a rebuild
            logger.warning(
                "%s: engine-cache reuse of %s failed (%s: %s); building",
                spec.short, key, type(exc).__name__, exc,
            )
            return None
        return Path(destination) if found is not None else None


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

    ``deferred`` is the third question (pgw#1545) and it is deliberately not
    folded into either: a specialization this run CHOSE not to build yet is
    absent from the store for a reason, and a verdict that cannot tell it from
    one that was promised and missing is back to guessing.
    """

    outcomes: List[Outcome]
    unservable: List[Gap]
    #: The specialization built FIRST, because it is the one that serves.
    priority: Optional[Spec] = None
    #: Handed to the background fill (or left unbuilt under ``--fill none``).
    deferred: Tuple[Spec, ...] = ()
    #: Where the background fill's own verdict lands. Empty = none started.
    fill: str = ""


# --------------------------------------------------------------------------
# The background fill
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fill:
    """The specializations this run deferred, and the two ways to run them.

    ``run`` builds them HERE, through the same per-specialization path the
    foreground used — it exists so the deferral is testable without a detached
    process, and production never calls it. ``argv`` is the real one: a whole
    ``gen-worker compile`` for the same endpoint with ``--fill all``, which
    resolves everything already built as PRESENT and continues from there. The
    fill is not a special mode with its own resume state; it is this verb.
    """

    specs: Tuple[Spec, ...]
    argv: Tuple[str, ...]
    log: Path
    verdict: Path
    run: Callable[[], List[Outcome]]


#: Start the fill and answer where its verdict will land. The production
#: implementation is :func:`detach`; an explicit one is the local seam.
FillRunner = Callable[[Fill], str]


def detach(fill: Fill) -> str:
    """Run the deferred specializations in a DETACHED, niced child.

    Detached, because the foreground's whole promise is that it returns as soon
    as the first artifact is servable — a caller that goes on to ``gen-worker
    up`` must not be waiting on the remainder. ``start_new_session`` puts the
    child in its own session so the parent's exit, its terminal and its process
    group signals do not take the fill with them.

    Niced, and only niced: CPU priority is what this process can actually give
    away. It does NOT arbitrate the card — what keeps the fill from starving
    live inference there is that it compiles one specialization at a time in a
    process of its own, which a serving process on the same card preempts the
    same way any other tenant does.
    """
    fill.log.parent.mkdir(parents=True, exist_ok=True)
    with open(fill.log, "ab", buffering=0) as handle:
        process = subprocess.Popen(
            list(fill.argv),
            stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
            start_new_session=True,
        )
    return f"pid {process.pid}, log {fill.log}, verdict {fill.verdict}"


def _fill_argv(
    *, endpoint_dir: Path, lock_path: Path, cas_root: Path, sm: str,
    lockfile: Optional[Path], only: int, vram_budget_gb: float, module: str,
    first: str, verdict: Path,
) -> Tuple[str, ...]:
    """The child that finishes this run: THIS verb, stated in full.

    Every input the parent resolved is restated explicitly — the sm it chose,
    the CAS it published into, the module name it published the document under.
    A child that re-derived any of them could publish under a different name or
    build for a different card and still exit 0, and nothing would disagree.
    """
    argv = [
        "nice", "-n", "19",
        sys.executable, "-m", "gen_worker.cli", "compile", str(endpoint_dir),
        "--sm", sm,
        "--graph-store", str(cas_root),
        "--lock", str(lock_path),
        "--module", module,
        "--first", first,
        "--fill", FILL_ALL,
        "--verdict", str(verdict),
    ]
    if lockfile is not None:
        argv += ["--env-lockfile", str(lockfile)]
    if only:
        argv += ["--only", str(only)]
    if vram_budget_gb > 0.0:
        argv += ["--vram-budget", str(vram_budget_gb)]
    return tuple(argv)


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
    first: str = "",
    fill: str = FILL_ALL,
    fill_runner: Optional[FillRunner] = None,
) -> Report:
    if fill not in FILLS:
        raise CompileError(f"--fill must be one of {FILLS}, got {fill!r}")
    specs = specializations(lock_path)
    if not specs:
        logger.info(
            "compile: this endpoint's lock claims no compiled specializations "
            "— nothing to build (it serves eager by declaration)"
        )
        return Report([], [])
    # THE PRIORITY ORDER, taken FIRST (pgw#1545): `--first` names the
    # specialization the caller's workflow actually runs and everything else
    # keeps document order behind it. Resolved here, before the floor probe
    # imports torch, so a selector that names nothing refuses in milliseconds
    # rather than after a load. `--only` then truncates the PRIORITY order,
    # which is the only reading of a bring-up aid that can honour both flags.
    specs = order(specs, first)
    if only:
        specs = specs[:only]

    env = _env_identity(endpoint_dir, sm, lockfile)
    if store is None:
        store = _store(cas_root)
    module = module or endpoint_module(endpoint_dir)

    def _known_present(spec: Spec) -> bool:
        """Present AND shaped like something a boot can load (pgw#1561).

        A position holding a bare-package blob from a pre-envelope publisher
        answered ``has_artifact`` TRUE, so every run short-circuited past the
        one thing that could repair it — the re-publish. A skewed position is
        a MISS here, which is exactly what routes it back through the publish
        (whose store-side migration arm replaces the skewed incumbent).

        Both questions go to the SAME store object (pgw#1573). The skew probe
        used to build a second one per specialization, which was both a second
        answer to "do I have this graph" and per-spec construction on the warm
        path pgw#1546 measured down to 0.95 s.
        """
        try:
            if not store.has_artifact(spec.graph, env):
                return False
        except Exception as exc:  # noqa: BLE001 — a store miss is not fatal
            logger.warning(
                "%s: store lookup failed (%s); treating as a miss",
                spec.short, exc,
            )
            return False
        try:
            skew = store.artifact_skew(spec.graph, env)
        except Exception:  # noqa: BLE001 — probe absence never blocks a run
            return True
        if skew:
            logger.warning(
                "%s: present but NOT loadable (%s) — treating as a miss so the "
                "publish path replaces it", spec.short, skew,
            )
            return False
        return True

    reuse = _EngineReuse(Path(cas_root), sm)

    from ..serving.mint import publish_compiled

    build = builder if builder is not None else _default_builder(cas_root, sm)
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

    def satisfy(spec: Spec, label: str) -> Outcome:
        """One specialization: present, or leased-built-published-witnessed.

        Returns an outcome for every path including failure — one graph's
        failure has never been the run's, and now that the run may be split
        across two processes it is the ONLY thing that could be.
        """
        started = time.monotonic()
        if _known_present(spec):
            logger.info("%s: present", label)
            return Outcome(spec, PRESENT, wall_s=time.monotonic() - started)

        destination = artifacts_dir / spec.short
        try:
            with work_ledger.lease(Path(cas_root), f"{spec.graph}/{env.value}"):
                # Re-check under the lease: the holder we queued behind may
                # have landed exactly this artifact while we waited. The same
                # loadability gate as everywhere (pgw#1561): a present-but-
                # skewed position must fall THROUGH to the publish that
                # replaces it, never short-circuit as done.
                if _known_present(spec):
                    logger.info("%s: present (landed while claimed)", label)
                    return Outcome(spec, PRESENT, wall_s=time.monotonic() - started)
                # REUSE BEFORE BUILD (pgw#1546): the engine cache may already
                # hold this exact mint, and resolving it needs no torch, no
                # program blob, and no child — 0.21 s against the child's
                # 5.47 s of bookkeeping to reach the same bytes.
                reused = reuse.resolve(spec, destination)
                if reused is not None:
                    artifact, state = reused, REUSED
                else:
                    logger.info("%s: building", label)
                    program = _ensure_program(spec, store, rederive, artifacts_dir)
                    artifact = build(spec, program, destination)
                    state = BUILT
                # PUBLISH, then ASK THE READER. Neither step existed before
                # pgw#1533 and neither one alone is enough: the publish is what
                # puts the bytes in the band adoption addresses, and the
                # read-back is the only thing that can go red if it did not.
                # The reuse arm runs the SAME two steps — a resolve that never
                # reached the serving band is the original defect exactly.
                published = publish_compiled(store, spec.graph, env, artifact)
                # The read-back MATERIALIZES (pgw#1561): `has_artifact` here
                # certified fourteen ZIPs no loader opens, on the strength of
                # their refs existing. Only bytes the real loader opens count.
                problem = witness_materializes(cas_root, spec.graph, env)
                if problem is not None:
                    raise CompileError(
                        f"{state} {spec.short} and the publish reported "
                        f"{published or 'nothing'}, and through the store "
                        f"boot-time adoption reads the artifact is NOT "
                        f"loadable: {problem}. The build is not the "
                        f"deliverable — an artifact the serving loader can "
                        f"OPEN is."
                    )
                logger.info(
                    "%s: %s and servable (%s)", label, state,
                    published or "published",
                )
                return Outcome(spec, state, published, wall_s=time.monotonic() - started)
        except work_ledger.Busy:
            logger.info(
                "%s: claimed by another process — skipping (the work ledger is "
                "how compile and a serving mint share this)", label,
            )
            return Outcome(spec, CLAIMED, wall_s=time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 — one failure is not the run
            logger.error("%s: FAILED: %s: %s", label, type(exc).__name__, exc)
            return Outcome(spec, FAILED, f"{type(exc).__name__}: {exc}",
                           wall_s=time.monotonic() - started)

    def satisfy_all(batch: Tuple[Spec, ...], *, offset: int) -> List[Outcome]:
        return [
            satisfy(spec, f"[{offset + index}/{len(specs)}] "
                          f"{spec.contract} {spec.short}")
            for index, spec in enumerate(batch, start=1)
        ]

    # The split. `specs` is already in priority order.
    priority = specs[0]
    foreground = specs if fill == FILL_ALL else specs[:1]
    deferred = () if fill == FILL_ALL else specs[1:]
    if deferred:
        logger.info(
            "compile: building %s FIRST (%s); %d specialization(s) deferred to "
            "the %s fill — the first artifact is what serves",
            priority.short, first or "the document's own default (all-defaults "
            "specialization, stated first by the derive)",
            len(deferred), fill,
        )

    # THE DOCUMENT FIRST, and this is a REVERSAL (pgw#1545). It used to be
    # published after every build, on the runtime mint's durability rule that
    # nothing is announced before it exists. That rule is about ARTIFACTS and
    # still governs them — but the document is not a claim that artifacts
    # exist. It is the authored lane list, read out of the committed lock, and
    # `AdoptSession` turns every record it cannot fetch into a HOLE: eager for
    # that graph, queued for the mint, never a refusal. Publishing it last is
    # therefore what made an incremental run unservable — thirteen artifacts on
    # disk and adoption unable to enumerate a single one, because the one row
    # it enumerates FROM had not landed yet.
    _publish_document(cas_root, lock_path, module)

    outcomes = satisfy_all(foreground, offset=0)

    fill_detail = ""
    if deferred and fill == FILL_BACKGROUND:
        verdict = _fill_dir(module) / "fill.json"
        runner = fill_runner if fill_runner is not None else detach
        try:
            fill_detail = runner(Fill(
                specs=deferred,
                argv=_fill_argv(
                    endpoint_dir=endpoint_dir, lock_path=lock_path,
                    cas_root=cas_root, sm=sm, lockfile=lockfile, only=only,
                    vram_budget_gb=vram_budget_gb, module=module, first=first,
                    verdict=verdict,
                ),
                log=_fill_dir(module) / "fill.log",
                verdict=verdict,
                run=lambda: satisfy_all(deferred, offset=1),
            ))
        except Exception as exc:  # noqa: BLE001 — a fill that will not start is
            # not a foreground failure: the priority artifact is servable and
            # this endpoint runs. It is stated, and the deferred rows stay
            # deferred, so the next `compile` finishes them.
            logger.error("compile: the background fill did not start: %s: %s",
                         type(exc).__name__, exc)
            fill_detail = f"NOT STARTED ({type(exc).__name__}: {exc})"

    gaps = unservable(cas_root, specs, env, module)
    return Report(outcomes, gaps, priority=priority, deferred=deferred,
                  fill=fill_detail)


def _fill_dir(module: str) -> Path:
    """Where a background fill's log and verdict live — one place per module,
    reused across runs so an operator has ONE path to read rather than a
    timestamped pile nobody prunes."""
    safe = "".join(char if char.isalnum() or char in "._-" else "-"
                   for char in module) or "endpoint"
    return workspace.artifacts_root() / "fill" / safe


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
            "Satisfy the graph specializations the committed endpoint.lock "
            "claims, for this card's sm and this venv's compile stack, "
            "starting with the one --first names. Fetches from the hub's fleet "
            "pool when it can, builds when it must, and shares the work with "
            "any serving process's background mint through the CAS work "
            "ledger. Weightless — no checkpoint needed. A specialization that "
            "is not built yet costs eager execution at request time and never "
            "a refusal, which is what makes --fill background safe."
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
    parser.add_argument("--first", default="", metavar="SELECTOR",
                        help="build THIS specialization first — the one the "
                             "workflow you are about to run needs. A "
                             "comma-separated conjunction over lane contract, "
                             "target, input parameter, dtype, AxBxC shape, or "
                             "a graph-identity prefix. Default: the "
                             "document's own first record (the all-defaults "
                             "specialization).")
    parser.add_argument("--fill", default=FILL_ALL, choices=list(FILLS),
                        help="what to do with the specializations that are not "
                             "--first. `all` (default) builds them here; "
                             "`background` returns as soon as the first one is "
                             "servable and finishes the rest in a detached "
                             "niced child; `none` builds only the first.")
    parser.add_argument("--module", default="", metavar="NAME",
                        help="publish the graph-set document under this module "
                             "name instead of importing the endpoint to read "
                             "it (for a tree this box cannot import)")
    parser.add_argument("--verdict", default="", metavar="PATH",
                        help="also write the verdict here as JSON: rc, the "
                             "state of every specialization, and every "
                             "remaining gap")
    parser.add_argument("--vram-budget", type=float, default=0.0, metavar="GB",
                        help="the VRAM grant this compile targets, passed to "
                             "the mint child as its residency budget. 0 = "
                             "probe this card once. It does NOT gate the "
                             "build: nothing here refuses a mint on a "
                             "declared number (pgw#1587).")
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
    # A gap this run DEFERRED is not a gap it failed to close (pgw#1545), and
    # the two are told apart by the gap's own attribution rather than by its
    # prose. Everything not attributable to a deferred specialization — the
    # graph-set document row included, because the document is published in the
    # foreground precisely so the deferred ones can land into it — is fatal.
    deferred_graphs = {spec.graph for spec in report.deferred}
    pending = [gap for gap in report.unservable if gap.graph in deferred_graphs]
    fatal = [gap for gap in report.unservable if gap.graph not in deferred_graphs]
    if fatal:
        # NOT a warning. A run that leaves the serving path unable to arm what
        # it was asked for has failed at the only thing it was for, and saying
        # so quietly beside a green summary is the defect this fixed, not a
        # style choice.
        lines.append(
            f"gen-worker compile: NOT SERVABLE — {len(fatal)} "
            f"gap(s) in the store `gen-worker up` adopts from:\n"
        )
        lines.extend(f"  - {gap.detail}\n" for gap in fatal)
        lines.append(
            "  A build that returned is not an artifact the serving path can "
            "find; this command reports the second.\n"
        )
        return "".join(lines), 1
    if counts.get(FAILED):
        return "".join(lines), 1
    if not report.outcomes:
        return "".join(lines), 0
    if pending:
        # THE PARTIAL VERDICT, and it is a different sentence on purpose. The
        # all-complete line below is a claim about EVERY specialization, so no
        # arrangement of half-done work may reach it: this branch is chosen by
        # asking the serving reader which deferred artifacts it still cannot
        # find, not by asking what this run intended.
        first = report.priority.short if report.priority is not None else "?"
        lines.append(
            f"gen-worker compile: SERVABLE FOR {first} — the graph-set "
            f"document and the {len(report.outcomes)} artifact(s) built here "
            f"are readable through the store boot-time adoption uses. "
            f"{len(pending)} specialization(s) are NOT built yet; serving arms "
            f"each one as it lands and falls back to eager for the rest.\n"
        )
        lines.extend(f"  - pending: {gap.detail}\n" for gap in pending)
        lines.append(
            f"  background fill: {report.fill}\n" if report.fill else
            "  no fill was started — re-run `gen-worker compile` to finish "
            "them (everything built already resolves as present).\n"
        )
        return "".join(lines), 0
    lines.append(
        f"gen-worker compile: SERVABLE — the graph-set document and all "
        f"{len(report.outcomes) + len(report.deferred)} artifact(s) are "
        f"readable through the store boot-time adoption uses.\n"
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
    verdict_path = Path(args.verdict) if getattr(args, "verdict", "") else None
    try:
        report = compile_all(
            endpoint_dir=endpoint_dir,
            lock_path=lock_path,
            cas_root=cas_root,
            sm=sm,
            lockfile=Path(args.env_lockfile) if args.env_lockfile else None,
            only=int(args.only or 0),
            vram_budget_gb=float(args.vram_budget or 0.0),
            module=str(getattr(args, "module", "") or ""),
            first=str(getattr(args, "first", "") or ""),
            fill=str(getattr(args, "fill", FILL_ALL) or FILL_ALL),
        )
    except CompileError as exc:
        sys.stderr.write(f"gen-worker compile: {exc}\n")
        if verdict_path is not None:
            _write_verdict(verdict_path, None, str(exc), 1)
        return 1
    summary, code = summarize(report)
    sys.stderr.write(summary)
    if verdict_path is not None:
        _write_verdict(verdict_path, report, summary, code)
    return code


def _write_verdict(
    path: Path, report: Optional[Report], summary: str, code: int
) -> None:
    """The run's verdict, durably, per specialization.

    THE BACKGROUND FILL'S ONLY WAY TO BE READ (pgw#1545). A detached child's
    exit status reaches nobody: its parent is gone by the time it finishes, and
    a serve pod's stdout goes nowhere. So the fill states its result where a
    later reader — an operator, the next `compile`, a test — can find it, and
    it states it PER SPECIALIZATION, because "the fill failed" over fourteen
    graphs is not a fact anyone can act on.

    Never raises: a verdict that could not be written must not turn a finished
    compile into a failed one.
    """
    payload: Dict[str, Any] = {"rc": int(code), "summary": summary}
    if report is not None:
        payload["priority"] = (
            report.priority.graph if report.priority is not None else "")
        payload["fill"] = report.fill
        payload["deferred"] = [spec.graph for spec in report.deferred]
        payload["specializations"] = [
            {"graph": outcome.spec.graph, "contract": outcome.spec.contract,
             "target": outcome.spec.target, "state": outcome.state,
             "detail": outcome.detail, "wall_s": round(outcome.wall_s, 3)}
            for outcome in report.outcomes
        ]
        payload["unservable"] = [
            {"graph": gap.graph, "detail": gap.detail}
            for gap in report.unservable
        ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
    except OSError as exc:
        logger.warning("compile: could not write the verdict to %s (%s)", path, exc)


__all__ = [
    "BUILT",
    "CLAIMED",
    "Builder",
    "CompileError",
    "FAILED",
    "FILLS",
    "FILL_ALL",
    "FILL_BACKGROUND",
    "FILL_NONE",
    "Fill",
    "FillRunner",
    "Gap",
    "Outcome",
    "PRESENT",
    "REUSED",
    "Report",
    "Spec",
    "add_subparser",
    "compile_all",
    "detach",
    "endpoint_module",
    "facets",
    "order",
    "run_compile",
    "select",
    "serving_reader",
    "specializations",
    "witness_materializes",
    "summarize",
    "unservable",
]
