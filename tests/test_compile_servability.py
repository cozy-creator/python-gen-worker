"""SERVABILITY: ``gen-worker compile`` succeeds only if the SERVING READER agrees.

The subject is one contract — *what this verb owes is not builds that returned,
it is an endpoint the boot-time adopt path can arm* — and everything here is a
precondition of it: that a built artifact lands in the band adoption addresses,
that the graph-set document adoption enumerates through gets published, that the
verdict and the exit code are computed from a reader rather than from a tally,
and that the mint child can find a CUDA root on the box it is actually running
on (no root, no artifact, so it belongs to the same claim).

Every test drives the REAL ``compile_all`` — its work ledger, its store, its
publish, its census — with only the inductor child replaced by a seam emitting a
real torchcg artifact (``tests/tcg_artifacts``: no torch, no GPU). The red arms
are the point, because each of them was once GREEN.
"""

from __future__ import annotations

import hashlib
import json
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

#: Two graphs, because "all of them" and "some of them" are different verdicts
#: and a one-graph fixture cannot tell them apart.
GRAPHS = (
    "cg-graph-v1-" + "a" * 56,
    "cg-graph-v1-" + "b" * 56,
)


def _ingress() -> CallIngress:
    return CallIngress(
        parameters=("value",),
        flat_arity=1,
        inputs=(CallInput("value", 0, "value", 0, (), "value", "float32", (2,)),),
    )


def document() -> GraphSetDocument:
    ingress = _ingress()
    return GraphSetDocument(
        stack=STACK,
        lanes=(
            LaneGraphs(
                contract=LANE,
                targets=(TARGET,),
                graphs=tuple(
                    GraphRecord(graph=graph, target=TARGET, ingress=ingress)
                    for graph in GRAPHS
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

    assert built == list(GRAPHS), "both specializations must have been built"
    assert [outcome.state for outcome in report.outcomes] == [
        compile_cli.BUILT, compile_cli.BUILT]
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

    assert [outcome.state for outcome in report.outcomes] == [
        compile_cli.BUILT, compile_cli.BUILT]
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
    assert gaps == ["graph-set document 'some.other': absent"]


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
    assert all("absent at" in gap for gap in gaps)


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
        ["artifact cg-graph-v1-aaa: absent at (cg-graph-v1-aaa, cg-env-v2-x)"],
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
