"""``gen-worker compile`` — pre-warm this card's compiled graphs."""

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

Builder = Callable[["Spec", Path, Path], Path]

PRESENT = "present"
BUILT = "built"
REUSED = "reused"
CLAIMED = "claimed"
FAILED = "failed"

FILL_ALL = "all"
FILL_BACKGROUND = "background"
FILL_NONE = "none"
FILLS = (FILL_ALL, FILL_BACKGROUND, FILL_NONE)


class CompileError(RuntimeError):
    """Compile could not start."""


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


def specializations(lock_path: Path) -> Tuple[Spec, ...]:
    """Every specialization the committed lock claims, in document order."""
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


def facets(spec: Spec) -> Tuple[str, ...]:
    """Every name ``--first`` may address ONE specialization by."""
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
    if term in facets(spec):
        return True
    return len(term) >= 8 and spec.graph.startswith(term)


def select(specs: Tuple[Spec, ...], selector: str) -> Spec:
    """The specialization ``selector`` names — the one built FIRST."""
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
    """``specs`` with the selected one first, everything else in document order."""
    first = select(specs, selector)
    return (first,) + tuple(spec for spec in specs if spec is not first)


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
    from ..serving.mint_store import graph_store

    return graph_store(Path(cas_root))


def _ensure_program(spec: Spec, store: Any, rederive: Any, scratch: Path) -> Path:
    from .._vendor.torchcg import is_graph_hash
    from ..serving.mint_store import ProgramBlobUnreachable

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

    def build(spec: Spec, program: Path, destination: Path) -> Path:
        return _build(
            spec, program=program, cas_root=Path(cas_root), sm=sm,
            destination=destination,
        )

    return build


def endpoint_module(endpoint_dir: Path) -> str:
    """The module name adoption looks this endpoint's document up BY."""
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


def serving_reader(cas_root: Path) -> Any:
    """The store object BOOT-TIME ADOPTION builds, over this machine's CAS."""
    return _store(Path(cas_root))


@dataclass(frozen=True, slots=True)
class Gap:
    """One thing the serving reader cannot find, ATTRIBUTED to its graph."""

    graph: str
    detail: str

    def __str__(self) -> str:
        return self.detail


def unservable(cas_root: Path, specs: Tuple[Spec, ...], env: Any, module: str) -> List[Gap]:
    """What the serving reader still cannot find."""
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
        try:
            skew = reader.artifact_skew(spec.graph, env)
        except Exception as exc:  # noqa: BLE001 — an unreadable probe is a gap
            skew = f"format probe failed ({exc})"
        if skew:
            gaps.append(Gap(spec.graph, f"artifact {spec.short}: {skew}"))
    return gaps


def witness_materializes(cas_root: Path, graph: str, env: Any) -> Optional[str]:
    """Fetch through the store adoption uses and OPEN through the real loader."""
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
    from .._vendor.torchcg.document import GraphSetDocument

    block = el.read_derive_block(lock_path)
    if block is None:
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
        """Whether the cache CLAIMS a mint for ``graph`` — an address answer; the claim is only trusted after :meth:`resolve` fully verifies it."""
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


@dataclass(frozen=True, slots=True)
class Report:
    """What one ``compile`` run did, and whether it left anything servable."""

    outcomes: List[Outcome]
    unservable: List[Gap]
    priority: Optional[Spec] = None
    deferred: Tuple[Spec, ...] = ()
    fill: str = ""


@dataclass(frozen=True, slots=True)
class Fill:
    """The specializations this run deferred, and the two ways to run them."""

    specs: Tuple[Spec, ...]
    argv: Tuple[str, ...]
    log: Path
    verdict: Path
    run: Callable[[], List[Outcome]]


FillRunner = Callable[[Fill], str]


def detach(fill: Fill) -> str:
    """Run the deferred specializations in a DETACHED, niced child."""
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
    specs = order(specs, first)
    if only:
        specs = specs[:only]

    env = _env_identity(endpoint_dir, sm, lockfile)
    if store is None:
        store = _store(cas_root)
    module = module or endpoint_module(endpoint_dir)

    def _known_present(spec: Spec) -> bool:
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
    artifacts_dir = workspace.artifacts_root()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rederive_ran = [False]

    def rederive() -> None:
        """Regenerate this box's exported programs into the graph store, ONCE."""
        if not rederive_ran[0]:
            rederive_ran[0] = True
            _rederive_programs(endpoint_dir, cas_root, lockfile)

    def satisfy(spec: Spec, label: str) -> Outcome:
        """One specialization: present, or leased-built-published-witnessed."""
        started = time.monotonic()
        if _known_present(spec):
            logger.info("%s: present", label)
            return Outcome(spec, PRESENT, wall_s=time.monotonic() - started)

        destination = artifacts_dir / spec.short
        try:
            with work_ledger.lease(Path(cas_root), f"{spec.graph}/{env.value}"):
                if _known_present(spec):
                    logger.info("%s: present (landed while claimed)", label)
                    return Outcome(spec, PRESENT, wall_s=time.monotonic() - started)
                reused = reuse.resolve(spec, destination)
                if reused is not None:
                    artifact, state = reused, REUSED
                else:
                    logger.info("%s: building", label)
                    program = _ensure_program(spec, store, rederive, artifacts_dir)
                    artifact = build(spec, program, destination)
                    state = BUILT
                published = publish_compiled(store, spec.graph, env, artifact)
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
            logger.error("compile: the background fill did not start: %s: %s",
                         type(exc).__name__, exc)
            fill_detail = f"NOT STARTED ({type(exc).__name__}: {exc})"

    gaps = unservable(cas_root, specs, env, module)
    return Report(outcomes, gaps, priority=priority, deferred=deferred,
                  fill=fill_detail)


def _fill_dir(module: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "._-" else "-"
                   for char in module) or "endpoint"
    return workspace.artifacts_root() / "fill" / safe


def _rederive_programs(
    endpoint_dir: Path, cas_root: Path, lockfile: Optional[Path]
) -> None:
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
    """What the operator is told, and what the shell is told, from ONE fact."""
    counts: Dict[str, int] = {}
    for outcome in report.outcomes:
        counts[outcome.state] = counts.get(outcome.state, 0) + 1
    lines = [
        "gen-worker compile: "
        + ", ".join(f"{state}={counts[state]}" for state in sorted(counts))
        + f" (of {len(report.outcomes)})\n"
    ]
    deferred_graphs = {spec.graph for spec in report.deferred}
    pending = [gap for gap in report.unservable if gap.graph in deferred_graphs]
    fatal = [gap for gap in report.unservable if gap.graph not in deferred_graphs]
    if fatal:
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
