"""pgw#1371: the serving worker MINTS its own holes — the CALLER, driven.

``serving/mint.py`` shipped with no production caller: every reference to
``mint_holes`` was a definition or a re-export, so no pod could mint anything
and its typed wedge events were unreachable. These are the tests of the
caller, and they are driven end to end rather than asserted about.

Real everything except the compile: the fixture endpoint under
``tests/fixtures/serving_v2_endpoint``, real ``load_endpoint`` /
``EndpointHost`` / ``ServeLoop``, real torchcg publish-time discovery, a real
``AdoptSession`` arming real module forwards, a real ``LocalGraphStore`` over
a real tensorfs CAS, and the real th#2133 adopt-answer shape through
``HubGraphStore``. The COMPILE is the stated seam (``compiler=``) because
inductor needs a GPU this box does not run — and because a fake compile is
what makes the RE-USE proof legible: the second worker's compile callable is
invoked zero times, which is the whole claim.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List

import pytest
import torch

from gen_worker import activity as activity_mod
from gen_worker import compile_posture
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker._vendor.torchcg.discovery import discover_lane
from gen_worker._vendor.torchcg.document import (
    GraphRecord,
    GraphSetDocument,
    LaneGraphs,
)
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker._vendor.torchcg.requirements import RequirementsManifest
from gen_worker._vendor.torchcg.store import LocalGraphStore
from gen_worker.serving import DeployBinding, EndpointHost, load_endpoint
from gen_worker.serving import mint as mint_mod
from gen_worker.serving import self_mint as self_mint_mod
from gen_worker.serving.hub_store import HubGraphStore, ReleaseNotStamped
from gen_worker.serving.mint_store import TieredGraphStore
from gen_worker.serving.self_mint import SelfMint
from gen_worker.serving.serve_adoption import ServeAdoption

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
LANE = "sdxl.diffusers-bf16@1"
SM = "sm_89"
STACK: tuple[tuple[str, str], ...] = (("torch", torch.__version__),)
ENV = EnvIdentity(stack=STACK, sm=SM)
OVERRIDES: dict[str, Any] = {"steps": {"default": 2, "lo": 1, "hi": 8}}
#: A well-formed store address — rows are keyed by cg-graph-v1 hashes.
GRAPH_HASH = "cg-graph-v1-" + "0" * 56


# --------------------------------------------------------------------------
# fixtures: a real endpoint, a real derive, a real store
# --------------------------------------------------------------------------


@pytest.fixture()
def binding(tmp_path: Path) -> DeployBinding:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"seed": 7, "scheduler": {"prediction_type": "epsilon"}}))
    return DeployBinding(
        checkpoint_ref="ckpt:tiny@1", checkpoint_dir=root, model="sdxl",
        defaults=dict(OVERRIDES),
    )


def fresh_host(binding: DeployBinding, tmp_path: Path) -> EndpointHost:
    return EndpointHost(
        load_endpoint(FIXTURE_DIR), binding, lane_contract=LANE,
        output_dir=tmp_path / "outputs")


@pytest.fixture()
def document(binding: DeployBinding, tmp_path: Path) -> GraphSetDocument:
    """The publish-time derive, run for real, then STAMPED with per-graph
    program digests — pgw#1370's blob-in contract, which is what the mint
    fetches instead of re-tracing author code."""
    host = fresh_host(binding, tmp_path)  # load_endpoint puts the fixture on sys.path
    host.setup()
    from serving_v2_fixture.main import AspectRatio

    (instance,) = host.instances.values()
    model: Any = instance.model

    def drive() -> None:
        for index, ratio in enumerate(AspectRatio):
            host.dispatch(
                "generate", {"prompt": "trace", "aspect_ratio": str(ratio)},
                request_id=f"trace-{index}")

    lane = discover_lane(LANE, ("unet",), {"unet": model.pipe.unet}, drive)
    host.teardown()
    stamped = tuple(
        GraphRecord(
            graph=record.graph, target=record.target, ingress=record.ingress,
            program="sha256:" + hashlib.sha256(
                record.graph.encode()).hexdigest(),
        )
        for record in lane.graphs
    )
    return GraphSetDocument(stack=STACK, lanes=(LaneGraphs(
        contract=lane.contract, graphs=stamped, targets=lane.targets,
        unobserved_targets=lane.unobserved_targets),))


def manifest() -> RequirementsManifest:
    return RequirementsManifest(
        include_set=(("torch", torch.__version__),), sm_compiled=SM)


def elf(path: Path) -> Path:
    """A minimal 64-bit ELF. The mint reads DT_NEEDED off the artifact itself
    to build its compatibility set; an artifact naming no library is a
    legitimate answer (all-embedded Triton/SASS), so this is the smallest
    HONEST artifact rather than a stub of the reader."""
    raw = bytearray(64)
    raw[:4] = b"\x7fELF"
    raw[4:7] = bytes((2, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(raw))
    return path


class FakeCompiler:
    """The stated ``compiler=`` seam: a valid artifact, fast, on CPU.

    It COUNTS, because the count is the re-use proof — a second worker that
    adopts must invoke this zero times.
    """

    def __init__(self, block: threading.Event | None = None) -> None:
        self.calls: List[str] = []
        self.block = block

    def __call__(self, blob: Path, record: Any, destination: Path) -> Path:
        assert blob.is_file(), "the mint must fetch the graph blob before compiling"
        self.calls.append(record.graph)
        if self.block is not None:
            self.block.wait(timeout=20.0)
        return elf(destination)


def programs(tmp_path: Path) -> Callable[[str, Path], Path]:
    """``fetch_program``: the serialized ExportedProgram by digest. The mint
    NEVER re-traces — its input is bytes."""

    def fetch(digest: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(digest.encode())
        return destination

    return fetch


def counting_loader(
    armed: List[str], served: List[str] | None = None,
) -> Callable[[Path, Any, Any], Any]:
    """The artifact loader — the ONE stub, because bytes-to-callable is the
    AOTInductor runtime's job on the target GPU.

    Two lists, deliberately: ``armed`` records the graph at LOAD time (a
    forward was swapped in) and ``served`` records it at CALL time (a request
    ran through that forward). Collapsing them is how "it armed" gets
    mistaken for "it served", which are the two separate claims here.

    THREE arguments since pgw#1460/tcg#58: the third is the LIVE MODULE whose
    resident tensors the compiled graph binds by reference. This stub asserts
    it, because "the loader USES the record" is precisely what this file
    asserted while production discarded it.
    """

    def load(path: Path, record: Any, module: Any) -> Callable[..., Any]:
        assert path.is_file(), "an armed artifact must exist on disk"
        assert isinstance(module, torch.nn.Module), (
            "the loader is handed the module its constants bind from")
        armed.append(record.graph)

        def compiled(sample: torch.Tensor) -> torch.Tensor:
            if served is not None:
                served.append(record.graph)
            return sample

        return compiled

    return load


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    """Captured events as (kind, phase, detail, kwargs) — the kwargs matter
    because pgw#1441 moved the reuse COUNTS into typed numeric fields, and a
    recorder that dropped them could not tell a wired count from prose."""
    seen: List[tuple] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append(
            (kind, phase, detail, kw)))
    return seen


def booted(
    binding: DeployBinding,
    tmp_path: Path,
    store: Any,
    document: GraphSetDocument,
    armed: List[str],
    tag: str,
    served: List[str] | None = None,
) -> EndpointHost:
    host = fresh_host(binding, tmp_path)
    host.setup(
        store=store, document=document, sm=SM,
        loader=counting_loader(armed, served),
        artifacts_dir=tmp_path / f"adopted-{tag}", stack=STACK)
    return host


def a_mint(store: Any, tmp_path: Path, compiler: Any, tag: str, **kw: Any) -> SelfMint:
    return SelfMint(
        store=store, artifacts_dir=tmp_path / f"artifacts-{tag}",
        compiler=compiler, program_source=programs(tmp_path),
        posture=compile_posture.FLEET, vcpus=4, **kw)


# --------------------------------------------------------------------------
# THE PROOF: misses -> minted -> a second boot hits, compiling nothing
# --------------------------------------------------------------------------


def test_the_first_worker_mints_its_own_holes_and_a_second_one_adopts_them(
    binding: DeployBinding, document: GraphSetDocument, tmp_path: Path,
    wire: List[tuple],
) -> None:
    """The whole program in one run.

    Worker A boots onto an unminted (lane x sm), registers every graph as a
    hole, mints them in the background off the request path, publishes each,
    and arms each onto its own live dispatch with no reboot. Worker B boots
    against the SAME store and has nothing left to do.
    """
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    expected = [r.graph for r in document.lanes[0].graphs]
    assert len(expected) == 2

    armed_a: List[str] = []
    served_a: List[str] = []
    worker_a = booted(
        binding, tmp_path, store, document, armed_a, "a", served_a)
    # Adopt-first boot found nothing: every claimed graph is a hole, in
    # canonical DOCUMENT order (pgw#1384's default-parameter-first guarantee
    # is the derive's, and the miner must not disturb it).
    assert [h.record.graph for h in worker_a.holes] == expected
    assert armed_a == [], "nothing was in the store, so nothing armed at boot"

    compiler_a = FakeCompiler()
    mint_a = a_mint(store, tmp_path, compiler_a, "a")
    assert mint_a.status().state == self_mint_mod.NOT_ARMED

    assert mint_a.arm(worker_a).state == self_mint_mod.MINTING
    status = mint_a.join(timeout=60.0)

    assert status.state == self_mint_mod.COMPLETE, status.facts()
    assert (status.holes, status.landed, status.failed) == (2, 2, 0)
    assert status.remaining == 0
    # It compiled each hole exactly once, from the BLOB (never a re-trace).
    assert sorted(compiler_a.calls) == sorted(expected)
    # It published each artifact, and it armed each onto the LIVE session.
    for graph in expected:
        assert store.has_artifact(graph, ENV), f"{graph} was never published"
        assert store.get_manifest(graph, ENV) is not None
    assert sorted(armed_a) == sorted(expected), "the mint published but never armed"
    assert worker_a.holes == (), "a filled hole is not still a hole"

    # And the pod SERVES through what it just minted — one arming path, not
    # two: the freshly minted graph swapped in exactly the way an adopted one
    # does, with no reboot and on the live dispatch.
    out = worker_a.dispatch(
        "generate", {"prompt": "after the mint", "aspect_ratio": "1:1"},
        request_id="post-mint")
    assert out.model == "ckpt:tiny@1"
    assert served_a, "the request did not run through the minted forward"
    assert set(served_a) <= set(expected)

    # ---- THE RE-USE PROOF ------------------------------------------------
    armed_b: List[str] = []
    worker_b = booted(binding, tmp_path, store, document, armed_b, "b")
    assert worker_b.holes == (), "worker B still had holes: the mint did not stick"
    assert [r.graph for r in worker_b.adoption.adopted] == expected
    assert sorted(armed_b) == sorted(expected), "worker B did not arm the fleet's bytes"

    compiler_b = FakeCompiler()
    mint_b = a_mint(store, tmp_path, compiler_b, "b")
    status_b = mint_b.arm(worker_b)
    assert status_b.state == self_mint_mod.NOTHING_TO_MINT, status_b.facts()
    assert compiler_b.calls == [], (
        "the second worker COMPILED something; adoption is not re-use")
    # Nothing-to-do is a wire fact, never a silence.
    assert [e for e in wire if e[0] == self_mint_mod.KIND_SKIPPED]

    worker_a.teardown()
    worker_b.teardown()


def test_a_wedged_compile_is_condemned_and_says_so_on_the_wire(
    binding: DeployBinding, document: GraphSetDocument, tmp_path: Path,
    wire: List[tuple],
) -> None:
    """RED-VERIFY of the module's own law: a hanging compile must produce
    ``self_mint_wedged``, not a hang. The caller is what makes that event
    reachable at all — before this it had no production emitter."""
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    host = booted(binding, tmp_path, store, document, [], "w")
    block = threading.Event()
    compiler = FakeCompiler(block=block)
    mint = a_mint(store, tmp_path, compiler, "w", window_s=0.3)
    try:
        mint.arm(host)
        status = mint.join(timeout=30.0)
    finally:
        block.set()

    assert status.state == self_mint_mod.CONDEMNED, status.facts()
    assert status.landed == 0 and status.reason
    wedged = [e for e in wire if e[0] == mint_mod.KIND_MINT_WEDGED]
    assert len(wedged) == 1, f"the wedge never reached the wire: {wire}"
    assert wedged[0][1] == "no_measured_progress"
    host.teardown()


def test_the_instrument_never_renders_an_absent_mint_as_a_finished_one(
    binding: DeployBinding, document: GraphSetDocument, tmp_path: Path,
) -> None:
    """C11: not-running, nothing-to-do and cannot-run are three answers.

    All three would otherwise be ``holes=0 landed=0``, which is exactly how a
    mint that never started reads as a mint that succeeded.
    """
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    never = SelfMint()
    assert never.status().state == self_mint_mod.NOT_ARMED

    host = booted(binding, tmp_path, store, document, [], "i")
    # Armed with holes but no way to compile them: a NAMED refusal carrying
    # the hole count, not a silent zero.
    stuck = SelfMint(store=store, artifacts_dir=tmp_path / "art")
    status = stuck.arm(host)
    assert status.state == self_mint_mod.UNAVAILABLE
    assert status.holes == 2 and "compiler" in status.reason

    # Nothing to do, said differently from both of the above.
    empty = SelfMint(store=store, artifacts_dir=tmp_path / "art2",
                     compiler=FakeCompiler())
    assert empty.arm(object()).state == self_mint_mod.NOTHING_TO_MINT

    states = {
        never.status().state, status.state, empty.status().state,
    }
    assert len(states) == 3, states
    host.teardown()


def test_the_mint_counts_its_progress_on_its_own_activity(
    binding: DeployBinding, document: GraphSetDocument, tmp_path: Path,
) -> None:
    """The counted observable: holes remaining / minted / failed, carried by
    a ``compile:``-prefixed counter on the mint's OWN activity scope — so a
    serving request's counter can never refresh this mint's stall clock."""
    from gen_worker import progress as progress_mod

    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    host = booted(binding, tmp_path, store, document, [], "c")
    seen: List[float] = []
    gate = threading.Event()

    class Watching(FakeCompiler):
        def __call__(self, blob: Path, record: Any, destination: Path) -> Path:
            snaps = [s for s in progress_mod.snapshot()
                     if s.name == self_mint_mod.COUNTER]
            if snaps:
                seen.append(snaps[0].done)
                gate.set()
            return super().__call__(blob, record, destination)

    mint = a_mint(store, tmp_path, Watching(), "c")
    mint.arm(host)
    status = mint.join(timeout=60.0)

    assert status.state == self_mint_mod.COMPLETE
    assert gate.is_set(), "no counter was open while the mint was compiling"
    assert seen and seen[0] == 0.0, seen
    # The counter's stall window comes from its `compile:` family, not the
    # 300s default — one inductor compile of one class outruns that.
    assert progress_mod.window_for(self_mint_mod.COUNTER) == 600.0
    host.teardown()


# --------------------------------------------------------------------------
# the PRODUCTION seam: the hub's adopt answer -> a session -> an armed mint
# --------------------------------------------------------------------------


class StubTransport:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.asks = 0

    def release_compiled_graphs(self, release_id: str, lane: str, sm: str) -> Any:
        self.asks += 1
        if self.answer is None:
            raise ReleaseNotStamped(f"release {release_id} is not stamped")
        return self.answer

    def fetch_blob(self, url: str) -> bytes:  # pragma: no cover — all misses
        raise AssertionError("an all-miss answer fetches no artifact bytes")


def all_miss_answer(document: GraphSetDocument) -> dict:
    lane = document.lanes[0]
    return {
        "object": "release_compiled_graphs",
        "release_id": "release-1",
        "binding_generation": 0,
        "env_compile_stack": [list(row) for row in document.stack],
        "lane": lane.contract,
        "lane_stamped": True,
        "lane_contract": {"stamp": lane.contract, "contract_digest": "",
                          "document": None, "requires": ""},
        "sm": SM,
        "empty": False,
        "targets": list(lane.targets),
        "unobserved_targets": list(lane.unobserved_targets),
        "passes": list(lane.passes),
        "graphs": [
            {
                "graph_hash": record.graph,
                "graph_specialization": "",
                "program": record.program,
                "module_path": record.target,
                "ingress_digest": record.ingress.digest(),
                "ingress": record.ingress.as_dict(),
                "status": "miss",
                "found": False,
            }
            for record in lane.graphs
        ],
        "hits": 0,
        "misses": len(lane.graphs),
    }


def test_the_serve_loop_seam_adopts_from_the_hub_and_arms_the_mint(
    binding: DeployBinding, document: GraphSetDocument, tmp_path: Path,
) -> None:
    """The production chain, minus the GPU: th#2133's answer -> HubGraphStore
    -> AdoptSession -> holes -> the mint's trigger fires.

    This is the wiring that did not exist. `ServeLoop` was constructed with
    ``compile_sink_for=None`` on every real pod, so no session was ever built
    and the work-list was always empty.
    """
    transport = StubTransport(all_miss_answer(document))
    armed: List[Any] = []
    adoption = ServeAdoption(
        "release-1", sm=SM, artifacts_dir=tmp_path / "adopted",
        cas_dir=tmp_path / "podcas", transport=transport, stack=STACK,
        loader=counting_loader([]), on_adopted=armed.append,
    )
    loaded = load_endpoint(FIXTURE_DIR)
    (model_cls,) = loaded.models
    lane = loaded.lane(model_cls, LANE)

    sink = adoption.sink_for(model_cls, lane)
    assert callable(sink), "the adopt sink is what ctx.compile arms through"
    assert transport.asks == 1
    # THE TRIGGER IS THE POST-LOAD HOOK, NOT THE SINK. Handing the sink over
    # happens before `load(ctx)` runs, so the work-list is still empty here —
    # arming the mint at this point mints nothing at all.
    assert armed == [], "the mint must not be armed before the load runs"
    adoption.loaded(model_cls, lane)
    assert armed == [adoption], "the post-load hook is the mint's trigger"
    adoption.loaded(model_cls, lane)
    assert armed == [adoption], "the trigger fires once, not once per load"
    # A second ask reuses the session — one hub round trip per boot.
    assert adoption.sink_for(model_cls, lane) is not None
    assert transport.asks == 1
    # And the store it hands the mint PUBLISHES, unlike the read-only hub one.
    assert isinstance(adoption.store, TieredGraphStore)


def test_the_production_loop_adopts_on_first_load_and_mints_what_it_missed(
    document: GraphSetDocument, tmp_path: Path,
) -> None:
    """THE production chain, driven: ``ServeLoop`` -> ``ctx.compile`` -> adopt
    -> holes -> the trigger -> the mint -> armed forwards, on one request.

    This is what a real pod runs. ``Worker`` built its ``ServeLoop`` with
    ``compile_sink_for=None``, so none of the above happened on any pod ever:
    the sink is the seam and nothing filled it. The mint's trigger fires at
    the FIRST MODEL LOAD, which under residency is inside the first request —
    the earliest instant at which the work-list exists at all.
    """
    from gen_worker.serving.residency import ResidencyManager
    from gen_worker.serving.serve_loop import ServeLoop, manifest_sizer

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "config.json").write_text(json.dumps({"seed": 3}))

    class Resolver:
        def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding:
            return DeployBinding(
                checkpoint_ref=checkpoint_ref, checkpoint_dir=tree,
                model="sdxl", defaults=dict(OVERRIDES))

        def default_pick(self, model_cls: type, slot_name: str) -> str:
            return "ckpt:tiny@1"

    armed: List[str] = []
    served: List[str] = []
    mints: List[SelfMint] = []
    compiler = FakeCompiler()

    def arm_the_mint(adoption: ServeAdoption) -> None:
        mint = a_mint(adoption.store, tmp_path, compiler, "loop")
        mints.append(mint)
        mint.arm(adoption)

    adoption = ServeAdoption(
        "release-1", sm=SM, artifacts_dir=tmp_path / "adopted",
        cas_dir=tmp_path / "podcas",
        transport=StubTransport(all_miss_answer(document)),
        stack=STACK, loader=counting_loader(armed, served),
        on_adopted=arm_the_mint,
    )
    loop = ServeLoop(
        load_endpoint(FIXTURE_DIR),
        residency=ResidencyManager(
            64 * 1024**3,
            manifest_sizer({"ckpt:tiny@1": 1024}, headroom_bytes=1024)),
        resolver=Resolver(),
        lane_contract=LANE,
        compile_sink_for=adoption.sink_for,
        on_loaded=adoption.loaded,
        output_dir=tmp_path / "out",
    )

    outcome = loop.invoke(
        "generate", {"input": {"prompt": "first ever request"}},
        request_id="first")
    assert outcome.result.model == "ckpt:tiny@1"

    # The first request adopted, found every graph missing, and armed the mint.
    assert mints, "the first model load did not trigger the mint"
    status = mints[0].join(timeout=60.0)
    assert status.state == self_mint_mod.COMPLETE, status.facts()
    expected = [r.graph for r in document.lanes[0].graphs]
    assert sorted(compiler.calls) == sorted(expected)
    assert sorted(armed) == sorted(expected)
    assert adoption.holes == ()
    # Every artifact is durable in the POD's own CAS — the fleet leg is the
    # stated pgw#1368/th#2132 hole, and it costs the fleet, never the graph.
    assert sorted(adoption.store.local_only) == sorted(expected)
    for graph in expected:
        assert adoption.store.local.has_artifact(graph, ENV)

    # A later request runs through the minted forwards, with no reboot.
    loop.invoke(
        "generate", {"input": {"prompt": "after"}}, request_id="second")
    assert served, "the second request did not use the freshly minted graphs"


def test_an_unstamped_release_is_an_eager_pod_with_a_stated_reason(
    tmp_path: Path, wire: List[tuple],
) -> None:
    """Never a boot failure — and never a silence either."""
    adoption = ServeAdoption(
        "release-1", sm=SM, artifacts_dir=tmp_path / "a",
        transport=StubTransport(None), stack=STACK)
    loaded = load_endpoint(FIXTURE_DIR)
    (model_cls,) = loaded.models
    assert adoption.sink_for(model_cls, loaded.lane(model_cls, LANE)) is None
    assert "release_not_stamped" in adoption.refusal
    assert adoption.facts() == {"adopting": False, "refusal": adoption.refusal}
    assert [e for e in wire if e[0] == "adopt_refused"]


def test_a_minted_artifact_the_fleet_cannot_take_is_still_banked_and_stated(
    tmp_path: Path, wire: List[tuple],
) -> None:
    """The upstream publish leg has no worker-side caller at HEAD, so the hub
    store refuses. That must cost the FLEET the bytes, never the graph: the
    artifact is durable on this pod and the shortfall is a counted fact."""
    local = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    upstream = HubGraphStore(StubTransport(None), "release-1", LANE, SM)
    store = TieredGraphStore(local, upstream)
    artifact = elf(tmp_path / "one.so")

    store.publish_artifact(GRAPH_HASH, ENV, artifact, manifest())

    assert local.has_artifact(GRAPH_HASH, ENV)
    assert store.local_only == (GRAPH_HASH,)
    assert store.facts() == {"tier": "local+hub", "local_only": 1}
    said = [e for e in wire if e[0] == "self_mint_publish_local_only"]
    assert len(said) == 1 and "pgw#1368" in said[0][2]


def test_a_graph_blob_with_no_route_costs_its_graph_and_names_its_owner(
    tmp_path: Path,
) -> None:
    """The mint's INPUT leg is the third unwired one, and it must fail TYPED.

    th#2133's adopt answer carries each graph's ``program`` digest and presigns
    only the compiled ARTIFACT, so a pod whose CAS has never seen the
    serialized graph has no route to it. That is a per-graph refusal naming its
    owner — never an ``AttributeError``, and never a re-trace.
    """
    from gen_worker.serving.mint_store import ProgramBlobUnreachable

    cas = LocalCAS(tmp_path / "cas")
    store = TieredGraphStore(LocalGraphStore(cas), None)
    with pytest.raises(ProgramBlobUnreachable, match="pgw#1370"):
        store.fetch_program("sha256:" + "0" * 64, tmp_path / "blob.pt2")

    # A blob this pod ALREADY holds needs no route: a digest is a digest.
    ref = cas.put_bytes(b"a serialized ExportedProgram")
    got = store.fetch_program(str(ref), tmp_path / "have.pt2")
    assert got.read_bytes() == b"a serialized ExportedProgram"


def test_the_reuse_hit_is_a_wire_fact_a_rental_can_capture(
    binding: DeployBinding, document: GraphSetDocument, tmp_path: Path,
    wire: List[tuple],
) -> None:
    """`boot_adopt` phase=`minting` on the pod that pays, `reused` on the one
    that does not — the two-line evidence a demonstration rental captures.

    A serve pod's stdout goes nowhere (pgw#760), so "this pod adopted N graphs
    and compiled none" cannot be a log line: it is the single observation the
    whole mint-and-reuse program is judged on.
    """
    from gen_worker.serving.residency import ResidencyManager
    from gen_worker.serving.serve_loop import ServeLoop, manifest_sizer

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "config.json").write_text(json.dumps({"seed": 3}))
    expected = [r.graph for r in document.lanes[0].graphs]

    class Resolver:
        def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding:
            return DeployBinding(
                checkpoint_ref=checkpoint_ref, checkpoint_dir=tree,
                model="sdxl", defaults=dict(OVERRIDES))

        def default_pick(self, model_cls: type, slot_name: str) -> str:
            return "ckpt:tiny@1"

    cas = tmp_path / "podcas"          # ONE pod CAS across both boots
    mints: List[SelfMint] = []

    def boot(tag: str, compiler: FakeCompiler) -> ServeAdoption:
        def arm(adoption: ServeAdoption) -> None:
            mint = a_mint(adoption.store, tmp_path, compiler, tag)
            mints.append(mint)
            mint.arm(adoption)

        adoption = ServeAdoption(
            "release-1", sm=SM, artifacts_dir=tmp_path / f"adopted-{tag}",
            cas_dir=cas, transport=StubTransport(all_miss_answer(document)),
            stack=STACK, loader=counting_loader([]), on_adopted=arm)
        loop = ServeLoop(
            load_endpoint(FIXTURE_DIR),
            residency=ResidencyManager(
                64 * 1024**3,
                manifest_sizer({"ckpt:tiny@1": 1024}, headroom_bytes=1024)),
            resolver=Resolver(), lane_contract=LANE,
            compile_sink_for=adoption.sink_for, on_loaded=adoption.loaded,
            output_dir=tmp_path / f"out-{tag}")
        loop.invoke("generate", {"input": {"prompt": tag}}, request_id=tag)
        return adoption

    paying = FakeCompiler()
    boot("first", paying)
    assert mints[0].join(timeout=60.0).state == self_mint_mod.COMPLETE
    assert sorted(paying.calls) == sorted(expected)

    reusing = FakeCompiler()
    boot("second", reusing)
    assert mints[1].join(timeout=60.0).state == self_mint_mod.NOTHING_TO_MINT
    assert reusing.calls == [], "the second pod compiled: that is not reuse"

    phases = [e[1] for e in wire if e[0] == "boot_adopt_summary"]
    assert phases == ["minting", "reused"], (
        f"the reuse hit must be readable off-pod, got {wire}")
    rows = [e for e in wire if e[0] == "boot_adopt_summary"]
    detail = rows[1][2]
    assert f"{len(expected)} graph(s) adopted" in detail and "0 hole(s)" in detail
    assert [e for e in wire if e[0] == self_mint_mod.KIND_SKIPPED]

    # pgw#1441: THE COUNTS ARE NUMERIC, so the reuse ratio is a query and not
    # a regex over a sentence. Paying boot 0/N, reusing boot N/N.
    assert rows[0][3]["step"] == 0
    assert rows[0][3]["total_steps"] == len(expected)
    assert rows[1][3]["step"] == len(expected)
    assert rows[1][3]["total_steps"] == len(expected)

    # And the per-BOOT roll-up never lands in the per-KEY kind, whose `phase`
    # vocabulary is hit/miss/no_export_declaration. One kind, one vocabulary.
    assert not [e for e in wire if e[0] == "boot_adopt"], (
        "the per-boot roll-up collided with the per-key event's kind")


# --- pgw#1444: the compile child must BE the parent ------------------------


def test_a_skewed_compile_child_refuses_before_it_compiles(tmp_path: Path) -> None:
    """pgw#840's hazard on the v2 mint path, which had NO protection at all.

    The child produces the bytes the parent publishes and arms, while every
    decision around it runs in the parent — sound only while both are the same
    code. `python -m gen_worker.serving.mint_child` resolves whatever the
    interpreter's path yields, so a second checkout, an inherited PYTHONPATH or
    a stale wheel silently substitutes a different compiler.

    Refused in the CHILD, before any work: a skewed child then costs nothing,
    where the v1 shape (child reports, parent judges) paid for the whole
    compile first.
    """
    from gen_worker.serving.mint_child import (
        ContractModuleMissing,
        compile_one,
        contract_digest,
    )

    mine = contract_digest()
    assert mine and mine != "", "the contract digest must have a real source"

    with pytest.raises(ContractModuleMissing, match="DIFFERENT gen_worker"):
        compile_one({"contract": "0" * 16})

    # An unstamped request (an older parent) still runs: absent is not skew,
    # and refusing it would break the mint on a mixed-version rollout.
    with pytest.raises(KeyError):  # got past the check, died on a real field
        compile_one({})


def test_a_missing_contract_module_refuses_instead_of_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE C10 REGRESSION. A guard whose digest source vanished must say so.

    `cd46c957` deleted the module v1's `_CONTRACT_MODULES` named; the digest
    became `""`; `if not CODE_DIGEST: return` turned the pgw#840 check into a
    no-op that still looked like a guard. Here an absent NAMED module is a
    typed refusal, distinguished from the frozen case where nothing is
    readable and we honestly cannot prove either way.
    """
    from gen_worker.serving import mint_child

    monkeypatch.setattr(
        mint_child, "CONTRACT_MODULES", ("mint.py", "not_a_real_module.py"))
    with pytest.raises(mint_child.ContractModuleMissing, match="partial install"):
        mint_child.contract_digest()

    # Frozen/zipimport: NOTHING is readable, so there is no claim to make.
    monkeypatch.setattr(
        mint_child, "CONTRACT_MODULES", ("nope_a.py", "nope_b.py"))
    assert mint_child.contract_digest() == mint_child.NO_SOURCE


def test_the_mint_stamps_its_contract_into_every_child_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent half: without the stamp the child has nothing to compare."""
    import subprocess

    from gen_worker.serving.mint_child import contract_digest

    seen: List[dict] = []

    def fake_run(argv: Any, **kw: Any) -> Any:
        request = json.loads(Path(argv[-1]).read_text())
        seen.append(request)
        Path(request["result"]).write_text(str(elf(tmp_path / "a.so")))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    compile_one = mint_mod._child_compiler(
        cas=tmp_path / "cas", target_arch="sm_89", toolchain={"cc": "1"})
    compile_one(
        tmp_path / "blob.pt2",
        SimpleNamespace(
            graph="g", target="unet",
            ingress=SimpleNamespace(as_dict=lambda: {})),
        tmp_path / "out.so")

    assert seen and seen[0]["contract"] == contract_digest()
