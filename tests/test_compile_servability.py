"""SERVABILITY: ``gen-worker compile`` succeeds only if the SERVING READER agrees.

The subject is one contract — *what this verb owes is not builds that returned,
it is an endpoint the boot-time adopt path can arm* — and everything here is a
precondition of it: that a built artifact lands in the band adoption addresses,
that the graph-set document adoption enumerates through gets published, that the
verdict and the exit code are computed from a reader rather than from a tally,
and that the mint child can find a CUDA root on the box it is actually running
on (no root, no artifact, so it belongs to the same claim).

pgw#1545 extends the same contract to a run that is deliberately INCOMPLETE:
the specialization the caller names is built first, the rest are deferred, and
the verdict has to stay honest about which is which — a deferred artifact is
not a missing one, and no arrangement of half-done work may print the
all-complete line. Its last two tests close the loop on the serving side,
because deferring is only safe if a specialization that is not built yet costs
eager execution and never a refusal.

Every test drives the REAL ``compile_all`` — its work ledger, its store, its
publish, its census — with only the inductor child replaced by a seam emitting a
real torchcg artifact (``tests/tcg_artifacts``: no inductor, no GPU), and the
serving pair drives the real ``AdoptSession`` with only the bytes->callable
loader seamed. The red arms are the point, because each of them was once GREEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest

import tcg_artifacts
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker._vendor.torchcg import CallIngress, CallInput
from gen_worker._vendor.torchcg.document import GraphRecord, GraphSetDocument, LaneGraphs
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker._vendor.torchcg.store import LocalGraphStore, PublishOutcome
from gen_worker.cli import workspace
from gen_worker.cli import compile as compile_cli
from gen_worker.cli import endpoint_lock as el

SM = "sm_89"
STACK: Tuple[Tuple[str, str], ...] = (("torch", "2.13.0"), ("triton", "3.6.0"))
ENV = EnvIdentity(stack=STACK, sm=SM)
MODULE = "sd15.main"
LANE = "sd15.diffusers-bf16@1"
TARGET = "unet"

#: Three graphs, because "all of them", "some of them" and "the ONE the caller
#: asked for first" are three different verdicts, and a one-graph fixture can
#: tell none of them apart. They differ in the two facets `--first` selects on
#: — target module path and input shape — because a fixture whose
#: specializations are indistinguishable cannot prove an ORDER was honoured.
GRAPHS = (
    "cg-graph-v1-" + "a" * 56,
    "cg-graph-v1-" + "b" * 56,
    "cg-graph-v1-" + "c" * 56,
)
TARGETS = ("unet", "unet", "vae")
SHAPES = ((2, 64, 64), (2, 128, 128), (1, 64, 64))


def _ingress(shape: Tuple[int, ...] = (2,)) -> CallIngress:
    return CallIngress(
        parameters=("value",),
        flat_arity=1,
        inputs=(CallInput("value", 0, "value", 0, (), "value", "float32", shape),),
    )


def document() -> GraphSetDocument:
    return GraphSetDocument(
        stack=STACK,
        lanes=(
            LaneGraphs(
                contract=LANE,
                targets=tuple(dict.fromkeys(TARGETS)),
                graphs=tuple(
                    GraphRecord(graph=graph, target=target, ingress=_ingress(shape))
                    for graph, target, shape in zip(GRAPHS, TARGETS, SHAPES)
                ),
            ),
        ),
    )


@pytest.fixture()
def endpoint(tmp_path: Path) -> Path:
    """An endpoint directory carrying the one thing ``compile`` reads: its lock.

    The lock IS the authored graph-set document — ``specializations()`` reads
    the specs out of it and ``_publish_document`` publishes the same bytes — so
    a fixture that wrote the document twice would be testing agreement between
    two things this design deliberately has only one of.
    """
    root = tmp_path / "endpoint"
    root.mkdir()
    # The compile stack is read from the endpoint's own uv.lock and NOWHERE
    # else (pgw#1489), so the env half of every artifact key comes from here.
    (root / "uv.lock").write_text(
        "version = 1\n"
        + "".join(
            f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
            for name, version in STACK
        )
    )
    raw = json.dumps(document().as_dict(), separators=(",", ":"), sort_keys=True)
    payload = json.dumps({"graphs": json.loads(raw)}, separators=(",", ":"))
    el.write_lock(
        root / el.LOCK_FILENAME,
        {},
        el.DeriveBlock(
            v=el.DERIVE_BLOCK_V,
            interface_v=el.torchcg_format_versions()[0],
            inputs_digest="0" * 64,
            document_digest=hashlib.sha256(payload.encode("ascii")).hexdigest(),
            document=payload,
            trace_device="cuda",
            endpoint="sd15.main:Sd15",
        ),
    )
    return root


def _seed_programs(store: LocalGraphStore, tmp_path: Path) -> None:
    """This box's serialized programs, banked under their graph identities.

    ``compile`` re-derives when they are absent; they are present here because
    the subject is what happens AFTER a build, not the derive.
    """
    for graph in GRAPHS:
        blob = tmp_path / f"{graph[-8:]}.pt2"
        blob.write_bytes(b"exported-program-" + graph.encode("ascii"))
        store.put_program(graph, blob)


def _builder(built: List[str]) -> compile_cli.Builder:
    """A compile that really produces an artifact, without inductor or a card."""

    def build(spec: compile_cli.Spec, program: Path, destination: Path) -> Path:
        assert program.is_file(), "the builder must be handed this box's program"
        destination.mkdir(parents=True, exist_ok=True)
        tcg_artifacts.aoti_package(
            destination / "model.pt2", graph_specialization=spec.graph)
        built.append(spec.graph)
        return destination

    return build


def _run(endpoint: Path, cas: Path, **kwargs: Any) -> compile_cli.Report:
    return compile_cli.compile_all(
        endpoint_dir=endpoint,
        lock_path=endpoint / el.LOCK_FILENAME,
        cas_root=cas,
        sm=SM,
        lockfile=None,
        module=MODULE,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Green arm
# --------------------------------------------------------------------------


def test_a_built_graph_lands_where_boot_time_adoption_reads_it(
    endpoint: Path, tmp_path: Path
) -> None:
    # pgw#1533: `built=14 (of 14)`, rc 0, 26 min of real GPU work, 0 artifacts armed.
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)
    built: List[str] = []

    report = _run(endpoint, cas, store=store, builder=_builder(built))

    assert built == list(GRAPHS), "every specialization must have been built"
    assert [outcome.state for outcome in report.outcomes] == (
        [compile_cli.BUILT] * len(GRAPHS))
    assert report.unservable == []

    # The witness that matters: a store object this run did not publish
    # through, built exactly the way `cli/daemon._adoption_source` builds it.
    reader = compile_cli.serving_reader(cas)
    assert reader.get_graphs(MODULE) == document()
    for graph in GRAPHS:
        assert reader.has_artifact(graph, ENV), f"{graph} is not where adoption looks"

    assert compile_cli.summarize(report)[1] == 0
    assert "SERVABLE" in compile_cli.summarize(report)[0]


def test_a_build_lands_in_the_box_cache_and_never_in_the_endpoint_tree(
    endpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """# pgw#1526: `compile` wrote its scratch to `<endpoint>/.compiled-graphs`.

    Machine-scoped bytes inside a source tree, untracked and unignored and
    indistinguishable from endpoint content — which is how 172 MB of local
    compile output took an sd15 source tarball from 75,164 to 59,408,521 bytes
    (cl#88) and killed the first build on an S3 fault over a residential link.

    Asserted at the PATH level rather than through adoption: the peer #1532
    lane reports adoption enumerating zero records over a full store for a
    cause of its own, so an adoption-based assertion here would go green or red
    for their reason instead of this one.
    """
    box = tmp_path / "box-artifacts"
    # Patched on `cli.workspace` itself, which is where the answer lives —
    # `compile` looks the attribute up at call time, so this is the real seam
    # rather than a re-export the module never promised.
    monkeypatch.setattr(workspace, "artifacts_root", lambda: box)
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)

    destinations: List[Path] = []

    def build(spec: compile_cli.Spec, program: Path, destination: Path) -> Path:
        destinations.append(destination)
        destination.mkdir(parents=True, exist_ok=True)
        tcg_artifacts.aoti_package(
            destination / "model.pt2", graph_specialization=spec.graph)
        return destination

    report = _run(endpoint, cas, store=store, builder=build)

    assert [outcome.state for outcome in report.outcomes] == (
        [compile_cli.BUILT] * len(GRAPHS))
    assert destinations, "the builder was never reached"
    for destination in destinations:
        assert box in destination.parents, destination
        assert endpoint not in destination.parents, destination
    # The endpoint tree is UNTOUCHED — the property a `.gitignore` or an
    # upload-exclusion can only paper over.
    assert not (endpoint / ".compiled-graphs").exists()


def test_a_second_run_over_a_full_store_reports_present_and_stays_servable(
    endpoint: Path, tmp_path: Path
) -> None:
    """The warm path — the ONLY path pgw#1491's acceptance ever exercised.

    It short-circuits at ``has_artifact`` before any build, which is exactly why
    a green acceptance certified a verb whose build path had never run. It must
    still be the census that decides, not the short-circuit.
    """
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)
    _run(endpoint, cas, store=store, builder=_builder([]))

    second: List[str] = []
    report = _run(endpoint, cas, store=store, builder=_builder(second))

    assert second == [], "nothing should be rebuilt over a full store"
    assert {outcome.state for outcome in report.outcomes} == {compile_cli.PRESENT}
    assert report.unservable == []
    assert compile_cli.summarize(report)[1] == 0


# --------------------------------------------------------------------------
# Red arms — each one is a compile that used to come out green
# --------------------------------------------------------------------------


class PublishesNowhere:
    """A store that reports a successful publish, believes itself, and wrote
    nothing where adoption reads.

    THE DEFECT ITSELF, in a dozen lines. ``Engine.compile`` banked bytes into
    torchcg's own engine cache and had every reason to call that a success —
    it could resolve them right back. What it could not do was answer for the
    ``(cg-graph-v1, cg-env-v2)`` band, and nothing asked it to.

    So this double ALSO answers ``has_artifact`` from its own record of what it
    published. That is the whole reason the read-back must go through a store
    object the run did not publish through: a writer asked about its own write
    is not a witness. Everything else is the real ``LocalGraphStore``.
    """

    def __init__(self, real: LocalGraphStore) -> None:
        self._real = real
        self.publishes = 0
        self._believed: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def publish_artifact(self, graph: str, env: Any, *args: Any, **kwargs: Any) -> Any:
        self.publishes += 1
        self._believed.add(graph)
        return PublishOutcome.PUBLISHED

    def has_artifact(self, graph: str, env: Any) -> bool:
        return graph in self._believed or self._real.has_artifact(graph, env)


def test_a_compile_that_publishes_nowhere_fails_loudly(
    endpoint: Path, tmp_path: Path
) -> None:
    # pgw#1533: the CLI never called `publish_artifact`; only the runtime mint did.
    cas = tmp_path / "graph-cas"
    real = LocalGraphStore(LocalCAS(cas))
    _seed_programs(real, tmp_path)
    store = PublishesNowhere(real)
    built: List[str] = []

    report = _run(endpoint, cas, store=store, builder=_builder(built))

    # The build genuinely happened and the publish genuinely claimed success.
    assert built == list(GRAPHS)
    assert store.publishes == len(GRAPHS)
    # And every one of them is a FAILURE, at the moment it happened.
    assert {outcome.state for outcome in report.outcomes} == {compile_cli.FAILED}
    assert all("serving reader" in outcome.detail for outcome in report.outcomes)
    assert len(report.unservable) == len(GRAPHS)
    summary, code = compile_cli.summarize(report)
    assert code == 1
    assert "NOT SERVABLE" in summary


def test_an_unpublished_graph_set_document_is_a_gap_even_when_every_artifact_is_there(
    endpoint: Path, tmp_path: Path
) -> None:
    """pgw#1533: the publish and the document are independent, and either gap alone
    serves eager.

    Adoption enumerates lanes out of the document and never reaches the
    artifacts without one, so a store full of correctly addressed bytes and no
    document adopts exactly nothing. The census must say so rather than counting
    the artifacts and calling it servable.
    """
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)
    _run(endpoint, cas, store=store, builder=_builder([]))

    specs = compile_cli.specializations(endpoint / el.LOCK_FILENAME)
    assert compile_cli.unservable(cas, specs, ENV, MODULE) == []
    # Ask under a name nothing published: the artifacts are still all present,
    # and that is not enough.
    gaps = compile_cli.unservable(cas, specs, ENV, "some.other")
    assert [gap.detail for gap in gaps] == [
        "graph-set document 'some.other': absent"]
    assert [gap.graph for gap in gaps] == [""], "the document row names no graph"


def test_an_artifact_under_the_wrong_env_is_absent_to_the_reader(
    endpoint: Path, tmp_path: Path
) -> None:
    """The env axis is real, and the census asks the reader on THIS card's axis.

    A publish that lands under a different ``cg-env-v2`` is the same silent
    failure wearing different clothes: the write succeeded, the position the
    boot reads is empty.
    """
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)
    _run(endpoint, cas, store=store, builder=_builder([]))

    specs = compile_cli.specializations(endpoint / el.LOCK_FILENAME)
    other = EnvIdentity(stack=STACK, sm="sm_90")
    gaps = compile_cli.unservable(cas, specs, other, MODULE)
    assert len(gaps) == len(GRAPHS)
    assert all("absent at" in gap.detail for gap in gaps)
    assert {gap.graph for gap in gaps} == set(GRAPHS), (
        "every gap is attributed, so a deferred one can be told from a missing one")


# --------------------------------------------------------------------------
# The exit code IS the verdict
# --------------------------------------------------------------------------


def test_the_shell_learns_unservable_even_when_every_build_returned() -> None:
    """pgw#1533: the exact shape of the night it was found — 14 BUILT, rc 0, 14 holes.

    Driven against a real ``Report`` rather than through argparse, because the
    thing under test is which fact the exit code is computed from.
    """
    spec = compile_cli.Spec(contract=LANE, graph=GRAPHS[0], target=TARGET, ingress={})
    green = compile_cli.Report([compile_cli.Outcome(spec, compile_cli.BUILT)], [])
    assert compile_cli.summarize(green)[1] == 0

    lying = compile_cli.Report(
        [compile_cli.Outcome(spec, compile_cli.BUILT)],
        [compile_cli.Gap(
            GRAPHS[0],
            "artifact cg-graph-v1-aaa: absent at (cg-graph-v1-aaa, cg-env-v2-x)")],
    )
    summary, code = compile_cli.summarize(lying)
    assert code == 1, "a build that returned is not an artifact anyone can serve"
    assert "built=1" in summary and "NOT SERVABLE" in summary


def test_a_below_floor_no_op_is_not_an_unservable_run(
    endpoint: Path, tmp_path: Path
) -> None:
    """Refusing to build under a grant that could never arm is a SUCCESS.

    It must not be swept into the new refusal: the store is legitimately empty
    and nothing was promised.
    """
    spec = compile_cli.Spec(contract=LANE, graph=GRAPHS[0], target=TARGET, ingress={})
    report = compile_cli.Report(
        [compile_cli.Outcome(spec, compile_cli.BELOW_FLOOR, "grant 1.0 < floor 4.0")],
        [],
    )
    assert compile_cli.summarize(report)[1] == 0


# --------------------------------------------------------------------------
# pgw#1533: the CUDA root the mint child points torch at — no root, no artifact
# --------------------------------------------------------------------------


def test_the_cuda_root_is_composed_where_this_process_can_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A box whose ``/usr/local`` is root-owned must not be told to write there.

    Measured: fourteen specializations died on ``FileNotFoundError:
    '/usr/local/cuda/include'`` while the toolkit sat in the venv's own
    ``nvidia-*`` wheels — which is what ``compose()`` builds a root FROM.
    """
    from gen_worker import cuda_root

    monkeypatch.setattr(cuda_root, "CUDA_ROOT", tmp_path / "usr" / "local" / "cuda")
    monkeypatch.setattr(cuda_root, "USER_CUDA_ROOT", tmp_path / "cache" / "cuda-root")

    # /usr/local absent and uncreatable by this process -> the user root.
    assert cuda_root.default_root() == tmp_path / "cache" / "cuda-root"

    # /usr/local writable -> the image answer, unchanged.
    (tmp_path / "usr" / "local").mkdir(parents=True)
    assert cuda_root.default_root() == tmp_path / "usr" / "local" / "cuda"

    # An existing root always wins, writable or not.
    (tmp_path / "usr" / "local" / "cuda").mkdir()
    assert cuda_root.default_root() == tmp_path / "usr" / "local" / "cuda"


def test_a_composition_names_where_it_wrote_not_where_the_constant_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CUDA_HOME`` comes from ``Composition.path``; reading the constant is
    what made this correct only on a pod."""
    from gen_worker import cuda_root

    root = tmp_path / "composed"
    root.mkdir()
    composed = cuda_root.compose(root)
    assert composed.path == str(root)
    assert composed.root == "preexisting"
    assert f"cuda_home: {root}" in composed.lines()


def test_a_relocated_root_says_so_rather_than_relocating_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gen_worker import cuda_root

    monkeypatch.setattr(cuda_root, "CUDA_ROOT", tmp_path / "usr" / "local" / "cuda")
    monkeypatch.setattr(cuda_root, "USER_CUDA_ROOT", tmp_path / "cache" / "cuda-root")
    monkeypatch.setattr(cuda_root, "wheel_cuda_root", lambda: "")

    composed = cuda_root.compose()

    assert composed.path == str(tmp_path / "cache" / "cuda-root")
    assert any("is neither present nor this process's to create" in note
               for note in composed.notes)



# --------------------------------------------------------------------------
# pgw#1546: the warm paths are structurally cheap, not merely fast
# --------------------------------------------------------------------------


class _PublishRefused:
    """A store proxy that fails the test the moment anything re-publishes."""

    def __init__(self, real: LocalGraphStore) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def publish_artifact(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an all-present run re-published an artifact")


def test_an_all_present_run_asks_no_policy_questions_and_republishes_nothing(
    endpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fully-warm run (pgw#1546): every artifact already in the serving
    band means NO floor read, NO grant probe, NO builder, NO publish — the
    ~4.5 s of author-module/torch import the old path paid was bookkeeping for
    a build that was never going to happen. The census still decides.
    """
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)
    _run(endpoint, cas, store=store, builder=_builder([]))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an all-present run consulted build policy")

    monkeypatch.setattr(compile_cli, "declared_floor_gb", refuse)
    monkeypatch.setattr(compile_cli, "grant_gb", refuse)

    report = _run(endpoint, cas, store=_PublishRefused(store), builder=refuse)

    assert {outcome.state for outcome in report.outcomes} == {compile_cli.PRESENT}
    assert report.unservable == []
    assert compile_cli.summarize(report)[1] == 0


def test_a_mint_in_the_engine_cache_is_reused_without_a_child_or_a_program(
    endpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pgw#1546's other warm state: present in the ENGINE cache (cg-key band),
    absent from the serving band — the exact state of the 140.5 s run. The old
    path spawned a mint child per specialization (torch import + export.load,
    5.47 s measured) purely to re-derive keys the cache already stores; the
    reuse path resolves them torch-free and still runs the [[pgw#1533]]
    publish + read-back, so the witness survives the optimization.

    The programs are deliberately NOT seeded: a reuse needs no program blob,
    and a builder that raises proves no child path was reached.
    """
    from gen_worker._vendor.tensorfs import LocalCAS as VendoredCAS
    from gen_worker._vendor.torchcg.engine import Engine
    from gen_worker import compile_cache

    monkeypatch.setattr(workspace, "artifacts_root", lambda: tmp_path / "box")
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    engine = Engine(VendoredCAS(cas))
    for graph in GRAPHS:
        artifact = tcg_artifacts.build(
            tmp_path / f"{graph[-8:]}.tar.gz",
            graph_specialization=graph, sm=SM,
            # Distinct witnesses: the cg-key hashes the graph CONTENT, not its
            # name, so two fixture graphs sharing one witness share one key and
            # the second import lands DIVERGENT instead of stored.
            witness=graph[-16:],
        )
        engine.import_artifact(tcg_artifacts.key_of(artifact), artifact)

    monkeypatch.setattr(
        compile_cache, "toolchain_digest",
        lambda: tuple(sorted(tcg_artifacts.TOOLCHAIN.items())),
    )

    def no_child(spec: Any, program: Any, destination: Any) -> Any:
        raise AssertionError(f"{spec.short}: a cached mint spawned a build child")

    report = _run(endpoint, cas, store=store, builder=no_child)

    assert [outcome.state for outcome in report.outcomes] == [
        compile_cli.REUSED for _ in GRAPHS]
    assert report.unservable == []
    reader = compile_cli.serving_reader(cas)
    for graph in GRAPHS:
        assert reader.has_artifact(graph, ENV), f"{graph} was reused but not published"
    summary, code = compile_cli.summarize(report)
    assert code == 0 and "SERVABLE" in summary
    assert f"reused={len(GRAPHS)}" in summary


def test_a_cached_mint_on_a_different_toolchain_is_not_reused(
    endpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED ARM: the reuse index answers only the exact (sm x toolchain) axis.
    A cache row minted under another compiler stack must fall through to the
    build path, never be republished as this env's artifact."""
    from gen_worker._vendor.tensorfs import LocalCAS as VendoredCAS
    from gen_worker._vendor.torchcg.engine import Engine
    from gen_worker import compile_cache

    monkeypatch.setattr(workspace, "artifacts_root", lambda: tmp_path / "box")
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)
    engine = Engine(VendoredCAS(cas))
    stale = tcg_artifacts.build(
        tmp_path / "stale.tar.gz",
        graph_specialization=GRAPHS[0], sm=SM,
        toolchain={"torch": "an-older-stack", "triton": "entirely"},
    )
    engine.import_artifact(tcg_artifacts.key_of(stale), stale)

    monkeypatch.setattr(
        compile_cache, "toolchain_digest",
        lambda: tuple(sorted(tcg_artifacts.TOOLCHAIN.items())),
    )
    built: List[str] = []
    report = _run(endpoint, cas, store=store, builder=_builder(built))

    assert built == list(GRAPHS), "a foreign-toolchain cache row must not satisfy this env"
    assert {outcome.state for outcome in report.outcomes} == {compile_cli.BUILT}
    assert report.unservable == []


def test_an_unreadable_module_name_is_a_typed_refusal_not_a_traceback(
    tmp_path: Path,
) -> None:
    """pgw#1537: `compile` reads the endpoint's module name and can now fail on
    the author's own import.

    `declared_floor_gb` calls `load_endpoint` too and swallows everything,
    because an unreadable floor genuinely IS "no floor stated". An unreadable
    module NAME is not "no name" — nothing can be published under a name that
    could not be read — so it refuses, in one sentence, with the cause.
    """
    empty = tmp_path / "not-an-endpoint"
    empty.mkdir()

    with pytest.raises(compile_cli.CompileError) as refusal:
        compile_cli.endpoint_module(empty)

    message = str(refusal.value)
    assert "cannot read" in message
    assert "graph-set document" in message, "it says WHY the name is needed"
    assert str(empty) in message, "and which endpoint it was asked about"


# --------------------------------------------------------------------------
# pgw#1545: FIRST the specialization the workflow needs; the rest in the
# background. Time-to-first-served is the number, not total mint wall.
# --------------------------------------------------------------------------


def _cas_with_programs(tmp_path: Path) -> Tuple[Path, LocalGraphStore]:
    cas = tmp_path / "graph-cas"
    store = LocalGraphStore(LocalCAS(cas))
    _seed_programs(store, tmp_path)
    return cas, store


def test_the_selector_addresses_a_specialization_by_the_facets_it_has(
    endpoint: Path,
) -> None:
    """A specialization has no human name, so `--first` matches its FACETS.

    Asserted through the real `specializations()` output rather than a hand-built
    Spec, because the facets are read off the ingress the lock actually carries.
    """
    specs = compile_cli.specializations(endpoint / el.LOCK_FILENAME)

    assert compile_cli.select(specs, "").graph == GRAPHS[0], (
        "an unstated selector is the document's own first record")
    assert compile_cli.select(specs, "vae").graph == GRAPHS[2]
    assert compile_cli.select(specs, "2x128x128").graph == GRAPHS[1]
    assert compile_cli.select(specs, "unet,2x128x128").graph == GRAPHS[1], (
        "terms are a conjunction: both must hold of the same specialization")
    assert compile_cli.select(specs, GRAPHS[2][:20]).graph == GRAPHS[2], (
        "a graph-identity prefix addresses one exactly")
    assert compile_cli.select(specs, LANE).graph == GRAPHS[0], (
        "a lane matches every graph in it, and the first one wins")

    ordered = compile_cli.order(specs, "vae")
    assert [spec.graph for spec in ordered] == [GRAPHS[2], GRAPHS[0], GRAPHS[1]], (
        "the selected one moves to the front; the rest keep document order")


def test_a_selector_that_names_nothing_refuses_rather_than_building_the_default(
    endpoint: Path,
) -> None:
    """The whole point of the argument is that the FIRST artifact is the one
    that serves. Silently building the default instead would report success
    over a specialization the caller never asked for."""
    specs = compile_cli.specializations(endpoint / el.LOCK_FILENAME)

    with pytest.raises(compile_cli.CompileError) as refusal:
        compile_cli.select(specs, "controlnet")

    message = str(refusal.value)
    assert "names no specialization" in message
    assert "vae" in message and "unet" in message, "it prints what IS addressable"


def test_only_truncates_the_PRIORITY_order_not_the_document_order(
    endpoint: Path, tmp_path: Path
) -> None:
    """The two bring-up flags have to compose.

    `--only` truncating before `--first` was resolved would make
    `--only 1 --first vae` refuse — the selector would be asked about a list the
    truncation had already removed its answer from.
    """
    cas, store = _cas_with_programs(tmp_path)
    built: List[str] = []

    report = _run(endpoint, cas, store=store, builder=_builder(built),
                  first="vae", only=1)

    assert built == [GRAPHS[2]]
    assert [outcome.spec.graph for outcome in report.outcomes] == [GRAPHS[2]]
    assert report.unservable == [], (
        "the census covers what this run took on, which is the truncated set")


def test_the_named_specialization_is_the_only_one_built_and_it_is_servable(
    endpoint: Path, tmp_path: Path
) -> None:
    """THE ACCEPTANCE, in miniature: one build, and the endpoint serves it.

    ``--fill none`` is the sharpest form of the claim — no background process
    to confuse the reading, so the store's contents are exactly what the
    priority build put there.
    """
    cas, store = _cas_with_programs(tmp_path)
    built: List[str] = []

    report = _run(endpoint, cas, store=store, builder=_builder(built),
                  first="vae", fill=compile_cli.FILL_NONE)

    assert built == [GRAPHS[2]], "only the specialization the caller named was built"
    assert report.priority is not None and report.priority.graph == GRAPHS[2]
    assert [spec.graph for spec in report.deferred] == [GRAPHS[0], GRAPHS[1]]

    reader = compile_cli.serving_reader(cas)
    assert reader.get_graphs(MODULE) == document(), (
        "the document is published, so adoption can enumerate and arm this one")
    assert reader.has_artifact(GRAPHS[2], ENV)
    assert not reader.has_artifact(GRAPHS[0], ENV)
    assert not reader.has_artifact(GRAPHS[1], ENV)


def test_the_graph_set_document_is_published_before_the_first_build(
    endpoint: Path, tmp_path: Path
) -> None:
    """A REVERSAL, and the property that makes incremental serving possible.

    The document used to be published after every build, so a run that had
    landed some artifacts and not others left adoption unable to enumerate a
    single one: the row it enumerates FROM had not landed. Observed from inside
    the builder — the only place that can see the store as it was BEFORE any
    artifact existed.
    """
    cas, store = _cas_with_programs(tmp_path)
    seen: List[Any] = []

    def build(spec: compile_cli.Spec, program: Path, destination: Path) -> Path:
        seen.append(compile_cli.serving_reader(cas).get_graphs(MODULE))
        destination.mkdir(parents=True, exist_ok=True)
        tcg_artifacts.aoti_package(
            destination / "model.pt2", graph_specialization=spec.graph)
        return destination

    _run(endpoint, cas, store=store, builder=build, fill=compile_cli.FILL_NONE)

    assert seen == [document()], (
        "the first build already ran against a store adoption can enumerate")


def test_a_deferred_run_never_reports_the_all_complete_line(
    endpoint: Path, tmp_path: Path
) -> None:
    """rc semantics: servable FOR the priority artifact, rc 0, and the
    all-complete sentence is unreachable while the reader still has holes."""
    cas, store = _cas_with_programs(tmp_path)

    report = _run(endpoint, cas, store=store, builder=_builder([]),
                  fill=compile_cli.FILL_NONE)
    summary, code = compile_cli.summarize(report)

    assert code == 0, "a deferred specialization is not a failure"
    assert f"SERVABLE FOR {report.outcomes[0].spec.short}" in summary
    assert "and all 3 artifact(s) are readable" not in summary
    assert summary.count("pending:") == 2
    assert "re-run `gen-worker compile`" in summary, (
        "no fill was started, so it says how they get finished")


def test_a_gap_in_what_this_run_PROMISED_is_still_fatal(
    endpoint: Path, tmp_path: Path
) -> None:
    """Deferral must not become a way to launder a failure.

    The priority build publishes nowhere — the pgw#1533 defect — while two
    specializations are legitimately deferred. The deferred ones stay silent;
    the promised one is NOT SERVABLE and rc is 1.
    """
    cas, real = _cas_with_programs(tmp_path)
    store = PublishesNowhere(real)

    report = _run(endpoint, cas, store=store, builder=_builder([]),
                  fill=compile_cli.FILL_NONE)
    summary, code = compile_cli.summarize(report)

    assert code == 1
    assert "NOT SERVABLE — 1 gap(s)" in summary, (
        "one gap is fatal; the other two were never promised by this run")
    assert {outcome.state for outcome in report.outcomes} == {compile_cli.FAILED}


def test_the_background_fill_finishes_what_the_foreground_deferred(
    endpoint: Path, tmp_path: Path
) -> None:
    """The fill runs the SAME per-specialization path, and the verdict is
    computed from the reader afterwards — which is the only way the
    all-complete line is ever reached."""
    cas, store = _cas_with_programs(tmp_path)
    built: List[str] = []
    handed: List[Tuple[str, ...]] = []

    def inline(fill: compile_cli.Fill) -> str:
        handed.append(tuple(spec.graph for spec in fill.specs))
        fill.run()
        return "ran inline"

    report = _run(endpoint, cas, store=store, builder=_builder(built),
                  first="vae", fill=compile_cli.FILL_BACKGROUND,
                  fill_runner=inline)
    summary, code = compile_cli.summarize(report)

    assert built == [GRAPHS[2], GRAPHS[0], GRAPHS[1]], (
        "priority first, then the rest in document order")
    assert handed == [(GRAPHS[0], GRAPHS[1])]
    assert report.unservable == []
    assert code == 0
    assert "and all 3 artifact(s) are readable" in summary


def test_the_detached_fill_is_THIS_verb_with_every_input_restated(
    endpoint: Path, tmp_path: Path
) -> None:
    """The fill is not a second code path with its own resume state.

    It is ``gen-worker compile --fill all`` for the same endpoint, so
    everything already built resolves as PRESENT and it continues from there.
    Every resolved input is restated on the argv: a child that re-derived its
    sm or its module name could publish somewhere else and still exit 0.
    """
    cas, store = _cas_with_programs(tmp_path)
    seen: List[compile_cli.Fill] = []

    def capture(fill: compile_cli.Fill) -> str:
        seen.append(fill)
        return f"pid 4242, log {fill.log}"

    _run(endpoint, cas, store=store, builder=_builder([]), first="vae",
         fill=compile_cli.FILL_BACKGROUND, fill_runner=capture)

    argv = seen[0].argv
    assert argv[:3] == ("nice", "-n", "19"), "the fill yields the CPU"
    assert "compile" in argv and str(endpoint) in argv
    for flag, value in (
        ("--sm", SM), ("--graph-store", str(cas)), ("--module", MODULE),
        ("--first", "vae"), ("--fill", compile_cli.FILL_ALL),
        ("--lock", str(endpoint / el.LOCK_FILENAME)),
        ("--verdict", str(seen[0].verdict)),
    ):
        assert argv[argv.index(flag) + 1] == value, flag
    # And the argv is a real one: the parser this verb installs accepts it.
    parser = argparse.ArgumentParser()
    compile_cli.add_subparser(parser.add_subparsers(dest="verb"))
    parsed = parser.parse_args(list(argv[argv.index("compile"):]))
    assert parsed.fill == compile_cli.FILL_ALL and parsed.module == MODULE


def test_detach_really_spawns_a_surviving_child_and_captures_its_output(
    tmp_path: Path
) -> None:
    """The PRODUCTION runner, exercised — not just the seam it hides behind.

    Every other test here states its own `fill_runner`, so `detach` would
    otherwise be correct code no test path calls, which is the exact defect
    class this repo keeps finding (pgw#1543 C1). Driven over a harmless argv:
    what is under test is that a child is started, survives being detached, and
    lands its output where the returned sentence says it will.
    """
    log = tmp_path / "fill" / "fill.log"
    fill = compile_cli.Fill(
        specs=(),
        argv=("nice", "-n", "19", sys.executable, "-c",
              "import os; print('filling', os.getsid(0) != %d)" % os.getsid(0)),
        log=log, verdict=tmp_path / "fill" / "fill.json",
        run=lambda: [],
    )

    detail = compile_cli.detach(fill)

    assert detail.startswith("pid ") and str(log) in detail
    pid = int(detail.split()[1].rstrip(","))
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert log.read_text(encoding="utf-8").strip() == "filling True", (
        "the child ran, in a session of its own, with its output captured")


def test_a_fill_that_cannot_start_is_stated_and_is_not_a_failure(
    endpoint: Path, tmp_path: Path
) -> None:
    """The priority artifact is servable and the endpoint runs. A fill that
    will not spawn leaves work undone, not a broken deliverable."""
    cas, store = _cas_with_programs(tmp_path)

    def refuses(fill: compile_cli.Fill) -> str:
        raise OSError("no fork for you")

    report = _run(endpoint, cas, store=store, builder=_builder([]),
                  fill=compile_cli.FILL_BACKGROUND, fill_runner=refuses)
    summary, code = compile_cli.summarize(report)

    assert code == 0
    assert "NOT STARTED (OSError: no fork for you)" in report.fill
    assert "NOT STARTED" in summary


def test_a_killed_fill_resumes_as_reuse_and_finishes_the_remainder(
    endpoint: Path, tmp_path: Path
) -> None:
    """Interruption safety, over the real store rather than a resume file.

    The fill dies after one of its two specializations. The re-run rebuilds
    NOTHING that landed — the finished ones resolve as PRESENT through the same
    store lookup a warm run uses — and builds only what is missing.
    """
    cas, store = _cas_with_programs(tmp_path)
    first_pass: List[str] = []

    def dies_after_one(fill: compile_cli.Fill) -> str:
        """A fill that lands ONE of its two and is then killed.

        Modelled the way production actually looks: the runner returns a pid
        and the parent goes on. A detached child's death is invisible to the
        foreground by construction, so the resume can only be proved by what is
        in the store afterwards — which is the point.
        """
        compile_cli.compile_all(
            endpoint_dir=endpoint, lock_path=endpoint / el.LOCK_FILENAME,
            cas_root=cas, sm=SM, lockfile=None, module=MODULE, store=store,
            builder=_builder(first_pass), first=fill.specs[0].graph,
            fill=compile_cli.FILL_NONE,
        )
        return "pid 4242 (killed after one)"

    _run(endpoint, cas, store=store, builder=_builder(first_pass), first="vae",
         fill=compile_cli.FILL_BACKGROUND, fill_runner=dies_after_one)
    assert first_pass == [GRAPHS[2], GRAPHS[0]], "two of three landed"

    second_pass: List[str] = []
    report = _run(endpoint, cas, store=store, builder=_builder(second_pass),
                  first="vae")

    assert second_pass == [GRAPHS[1]], "only the unfinished one was rebuilt"
    assert {outcome.state for outcome in report.outcomes} == {
        compile_cli.PRESENT, compile_cli.BUILT}
    assert report.unservable == []
    assert compile_cli.summarize(report)[1] == 0


def test_the_verb_writes_a_durable_per_specialization_verdict(
    endpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detached fill's exit status reaches nobody, so it states its result
    where a later reader can find it — per specialization, because "the fill
    failed" over fourteen graphs is not a fact anyone can act on.

    Driven through argparse and :func:`run_compile`, so the flags, the handler
    and the writer are all the real ones.
    """
    cas, store = _cas_with_programs(tmp_path)
    monkeypatch.setattr(workspace, "artifacts_root", lambda: tmp_path / "box")
    _run(endpoint, cas, store=store, builder=_builder([]))  # everything present

    verdict = tmp_path / "fill" / "fill.json"
    parser = argparse.ArgumentParser()
    compile_cli.add_subparser(parser.add_subparsers(dest="verb"))
    args = parser.parse_args([
        "compile", str(endpoint), "--sm", SM, "--graph-store", str(cas),
        "--lock", str(endpoint / el.LOCK_FILENAME), "--module", MODULE,
        "--verdict", str(verdict),
    ])

    assert args._handler(args) == 0

    banked = json.loads(verdict.read_text(encoding="utf-8"))
    assert banked["rc"] == 0
    assert "SERVABLE" in banked["summary"]
    assert [row["graph"] for row in banked["specializations"]] == list(GRAPHS)
    assert {row["state"] for row in banked["specializations"]} == {compile_cli.PRESENT}
    assert banked["unservable"] == []


# --------------------------------------------------------------------------
# ...and what SERVING does with a store that is only partly filled. This is
# the other half of pgw#1545: deferring is only safe because a specialization
# that is not built yet costs eager execution and NEVER a refusal.
# --------------------------------------------------------------------------


def _adopt_session(cas: Path, artifacts: Path, armed: List[str]) -> Any:
    """A REAL ``AdoptSession`` over the store `compile` published into.

    Only the bytes->callable loader is seamed (there is no card here); the
    fetch, the claim, the per-graph arm and the dispatcher are torchcg's own.
    """
    from gen_worker._vendor.torchcg.adopt import AdoptSession

    def loader(artifact: Path, record: Any, module: Any) -> Any:
        armed.append(record.graph)
        return lambda value: ("compiled", record.graph)

    return AdoptSession(
        LocalGraphStore(LocalCAS(cas)), document(), LANE, SM,
        loader=loader, artifacts_dir=artifacts, stack=STACK,
    )


def _marked_module() -> Any:
    import torch

    class Denoiser(torch.nn.Module):
        def forward(self, value: Any) -> Any:
            return ("eager", tuple(value.shape))

    return Denoiser()


def test_a_specialization_that_is_not_built_yet_serves_EAGER_and_never_refuses(
    endpoint: Path, tmp_path: Path
) -> None:
    """The property the whole deferral rests on.

    One artifact in the store, three in the document. The call that matches it
    dispatches compiled; the calls that do not run the author's own forward.
    Neither refuses, and neither waits for a build.
    """
    import torch

    cas, store = _cas_with_programs(tmp_path)
    _run(endpoint, cas, store=store, builder=_builder([]), first="vae",
         fill=compile_cli.FILL_NONE)

    armed: List[str] = []
    session = _adopt_session(cas, tmp_path / "adopted", armed)
    module = session.adopt(_marked_module())

    assert armed == [GRAPHS[2]], "only the built specialization was armed"
    assert [record.graph for record in session.adopted] == [GRAPHS[2]]
    assert [hole.record.graph for hole in session.holes] == [GRAPHS[0], GRAPHS[1]], (
        "the unbuilt ones are the mint work-list, in document order")

    assert module(torch.zeros(SHAPES[2])) == ("compiled", GRAPHS[2])
    assert module(torch.zeros(SHAPES[0])) == ("eager", SHAPES[0]), (
        "a request needing a specialization nobody built runs eager")
    assert module(torch.zeros((7, 7))) == ("eager", (7, 7)), (
        "and so does a shape no specialization in the document covers")


def test_the_fill_ARMS_into_a_live_session_without_a_reboot(
    endpoint: Path, tmp_path: Path
) -> None:
    """Per-artifact adoption: each specialization arms the instant it lands.

    The session is built while two specializations are still missing, a request
    is served eager for one of them, then the fill lands it and the SAME live
    module dispatches compiled — no restart, no re-adopt.
    """
    import torch

    cas, store = _cas_with_programs(tmp_path)
    _run(endpoint, cas, store=store, builder=_builder([]), first="vae",
         fill=compile_cli.FILL_NONE)

    armed: List[str] = []
    session = _adopt_session(cas, tmp_path / "adopted", armed)
    module = session.adopt(_marked_module())
    assert module(torch.zeros(SHAPES[0])) == ("eager", SHAPES[0])

    # The fill lands the deferred specializations, exactly as `--fill
    # background` would, through the same verb.
    _run(endpoint, cas, store=store, builder=_builder([]))

    hole = session.holes[0]
    fetched = LocalGraphStore(LocalCAS(cas)).fetch_artifact(
        hole.record.graph, session.env, tmp_path / "late" / "model")
    session.arm(hole.record, fetched)

    assert armed == [GRAPHS[2], GRAPHS[0]]
    assert module(torch.zeros(SHAPES[0])) == ("compiled", GRAPHS[0]), (
        "the live module took the late artifact")
    assert module(torch.zeros(SHAPES[1])) == ("eager", SHAPES[1]), (
        "and the one still unarmed is still eager, not refused")
    assert [hole.record.graph for hole in session.holes] == [GRAPHS[1]]
