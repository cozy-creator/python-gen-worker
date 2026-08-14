"""The delegated mint child gets the parent's RESOLVED slot refs.

**The mechanism.** ``MintRequest`` carries ``snapshots`` — slot -> local
tree — and nothing else about the resolution. A path says which BYTES to
load; only a binding says which CHECKPOINT the request is for, and
``ctx.slots`` is built from bindings. The child re-runs discovery in a fresh
interpreter, so its ``spec.models`` holds only what ``@endpoint`` declared,
and a hub-catalog slot::

    models={"pipeline": Slot(StableDiffusionXLPipeline, selected_by="model")}

— has no ``default_checkpoint=``, so rediscovery yields ``spec.models == {}``
and the resolution chain has nothing to resolve; ``ctx.slots[...]`` then raises
*"no resolved model ref for this request"* 0.0 s into ``warmup_forward``. Not
an ordering problem (the parent resolves long before the spawn) and not a
serialisation gap (``ModelRef`` is already a ``msgspec.Struct``): a field the
handoff struct never had.

A harness endpoint that declares ``default_checkpoint=`` cannot see this — a
declared ref survives rediscovery. The endpoint under test is a CATALOG slot,
which is also the *quiet* half of the same defect: a catalog slot WITH a code
default resolves in the child to the DECLARED checkpoint while the parent
serves the hub's pick — a mint tracing graphs for a model this pod never runs
(``JuggleEndpoint`` below).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import msgspec
import pytest

from gen_worker import child_preflight
from gen_worker import child_contract
from gen_worker import handler_proof
from gen_worker import mint_child, mint_process, registry, warmup
from gen_worker import mint_process as mp
from gen_worker.api.binding import ModelRef, wire_ref
from gen_worker.cli import run as cli_run
from gen_worker.executor import _BackgroundMint, _hub_binding_for_wire_ref
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import CompileCell

from harness import mint_catalog_slot_pgw969 as catalog
from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready, is_result_for

CATALOG_MODULE = "harness.mint_catalog_slot_pgw969"
#: The hub's pick for this dispatch. Nothing in the endpoint's code names it.
PICK_REF = wire_ref(_hub_binding_for_wire_ref("harness/catalog-pick:prod"))
GIB = 1 << 30


# ---------------------------------------------------------------------------
# The parent half: one REAL dispatch that binds the catalog slot and serves it
# ---------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path: Path) -> Tuple[Path, ModelRef]:
    """Run one real dispatch through the real executor and the real CAS
    downloader; return the materialized tree and the binding behind it.

    This is where the parent's resolution comes from — not a hand-built
    ``ModelRef``. The endpoint declares no checkpoint, so a served response
    naming ``PICK_REF`` proves the binding is the hub's and the parent's.
    """
    cli_run._INJECTED_CACHE.clear()
    catalog.RESOLVED_REFS.clear()
    cache_dir = tmp_path / "cas"
    cache_dir.mkdir(parents=True, exist_ok=True)
    blobs = BlobHost(tmp_path)
    try:
        snap = blobs.snapshot("snap-catalog-pick", [
            blobs.file("w", b"pick-weights", path_in_snapshot="weights.txt"),
        ])
        with hub_double(
            modules=(CATALOG_MODULE,), cache_dir=cache_dir,
        ) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-969", attempt=1,
                function_name="catalog-generate",
                input_payload=msgspec.msgpack.encode(catalog.CatalogIn()),
                models=[pb.ModelBinding(slot="pipeline", ref=PICK_REF)],
                snapshots={PICK_REF: snap},
            ))
            res = conn.wait_for(is_result_for("r-969")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        served_out = msgspec.msgpack.decode(res.inline, type=catalog.CatalogOut)
    finally:
        blobs.shutdown()

    assert served_out.response == f"{PICK_REF}|pick-weights", served_out
    # Both the parent's boot warm and its served dispatch land here.
    assert set(catalog.RESOLVED_REFS) == {PICK_REF}, (
        "the PARENT must resolve the catalog slot to the hub's pick")

    trees = sorted(
        p for p in (cache_dir / "cas" / "snapshots").iterdir() if p.is_dir())
    assert len(trees) == 1, trees
    cli_run._INJECTED_CACHE.clear()
    catalog.RESOLVED_REFS.clear()
    # The binding the executor itself mints for a hub-picked slot ref — the
    # same call `_slot_dispatch_binding` made to serve the dispatch above.
    return trees[0], _hub_binding_for_wire_ref(PICK_REF)


def _request(
    tree: Path, binding: Any, tmp_path: Path, *, function: str = "catalog-generate",
) -> mp.MintRequest:
    """The child's request, built through the REAL parent chain and
    round-tripped through msgspec — the boundary IS a file.

    ``binding=None`` no longer means "a path with no ref" — pgw#974 made that
    unrepresentable. It now means the parent resolved this slot not at all, so
    the slot is ABSENT, which is the only shape a wiring gap can still take.
    """
    pending = SimpleNamespace(
        family="pgw969", arm_token="ck1-catalog", recipe="dynamo",
        cfg=CompileCell(shapes=((1024, 1024),), targets=("unet",),
                        family="pgw969", regional=False, text_len=77,
                        dynamic=(), lora_bucket=0, guidance_scales=(),
                        text_lens=()), target=tmp_path / "cell.tar.gz",
        mint_root=tmp_path)
    bg = _BackgroundMint(
        spec=SimpleNamespace(name=function), instance=object(), snapshots=None,
        pendings={}, pipes={},
        modules=(CATALOG_MODULE,),
        slots=({"pipeline": child_contract.MintSlot(ref=binding, path=str(tree))}
               if binding is not None else {}),
    )
    task = mint_process.MintTask(
        pending=pending, pipe=object(), function=function,
        modules=bg.modules, slots=dict(bg.slots),
        execution_lane="w8a8-lora64", device=-1)
    request = mint_process.build_request(
        task, workdir=tmp_path / "w")
    return msgspec.json.decode(msgspec.json.encode(request), type=mp.MintRequest)


def _child_specs(request: mp.MintRequest) -> Tuple[Any, List[Any]]:
    """Exactly what ``mint_child.mint`` does before it reads a weight:
    rediscover, select the class-scoped sibling set, install the parent's
    bindings, and refuse if they cannot resolve."""
    specs = registry.collect_endpoints(list(request.modules))
    chosen, siblings = child_preflight.select_specs(specs, request.function)
    child_preflight.bind_slots(siblings, request.slots)
    # pgw#1089 took the whole request out of this signature: the boot-trace
    # child resolves the same slots for the same reason and must get the same
    # refusal, and a second copy of the check would be a second answer to "can
    # this process trace the checkpoint the parent serves".
    child_preflight.assert_slots_resolvable(
        siblings, request.slots, what=mint_child.mint_identity(request))
    return chosen, siblings


# ---------------------------------------------------------------------------
# 1. The pod's crash, at the frame it happened at
# ---------------------------------------------------------------------------


def test_a_rediscovered_catalog_slot_carries_no_ref_at_all() -> None:
    """The whole defect in one assertion, and the reason a path is not enough.

    Green before and after the fix: rediscovery cannot invent a checkpoint
    the decorator never named. That is precisely why the binding has to
    cross.
    """
    specs = registry.collect_endpoints([CATALOG_MODULE])
    for spec in specs:
        assert spec.slots["pipeline"] is not None
        assert spec.models == {}, (
            "a catalog slot declares no ref; only the parent has one")


def test_the_production_traceback_reproduced(tmp_path: Path) -> None:
    """The pod's sentence, byte-for-byte, from an unbound warm context — the
    exact ``run_warm_job`` -> handler -> ``ctx.slots`` frame that crashed."""
    spec = next(
        s for s in registry.collect_endpoints([CATALOG_MODULE])
        if s.name == "catalog-generate")
    instance = spec.cls()
    instance.pipeline_path = str(tmp_path)
    job = next(j for j in handler_proof.warm_jobs([spec])
               if j.spec.name == spec.name)
    with pytest.raises(ValueError) as exc:
        handler_proof.run_warm_job(instance, job, {}, "w8a8-lora64")
    assert "slot 'pipeline': no resolved model ref for this request" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. The fix, end to end: parent dispatch -> wire -> child -> warm forward
# ---------------------------------------------------------------------------


def test_the_child_warms_the_checkpoint_THE_PARENT_SERVED(
    served: Tuple[Path, ModelRef], tmp_path: Path,
) -> None:
    """RED at HEAD. The child's own warm forward, through the endpoint's own
    handler, on the tree a real dispatch materialized — and it must resolve
    the SAME checkpoint the parent answered that dispatch with.

    Not "resolves something": the ref is compared against what the serving
    path recorded, because a child that traced a different checkpoint would
    pack a cell the parent's proof then misses (pgw#815's silent class).
    """
    tree, binding = served
    request = _request(tree, binding, tmp_path)
    chosen, siblings = _child_specs(request)

    instance = chosen.cls()
    loaded = cli_run.run_setup(
        instance, {s: r.path for s, r in request.slots.items()},
        arm_compile=False,
        return_loaded=True) or {}
    assert loaded["pipeline"] == str(tree)

    catalog.RESOLVED_REFS.clear()
    for job in handler_proof.warm_jobs(siblings):
        handler_proof.run_warm_job(
            instance, job, {}, request.execution_lane,
            origin=mint_child.mint_identity(request))

    assert catalog.RESOLVED_REFS, "the child ran no warm forward at all"
    assert set(catalog.RESOLVED_REFS) == {PICK_REF}, (
        "the child must trace the parent's checkpoint, not a rediscovered one")


def test_both_siblings_are_bound_so_the_warm_plan_stays_class_scoped(
    served: Tuple[Path, ModelRef], tmp_path: Path,
) -> None:
    """``instance_key`` is a live property over ``spec.models``. Binding the
    invoked function alone would move its key out from under its sibling and
    narrow a class-scoped mint to one lane — silently."""
    tree, binding = served
    request = _request(tree, binding, tmp_path)
    chosen, siblings = _child_specs(request)

    assert {s.name for s in siblings} == {
        "catalog-generate", "catalog-generate-turbo"}
    keys = {s.instance_key for s in siblings}
    assert len(keys) == 1, "the sibling set must still share ONE instance key"
    for spec in siblings:
        assert wire_ref(spec.models["pipeline"]) == PICK_REF


def test_the_binding_crosses_the_wire_whole(
    served: Tuple[Path, ModelRef], tmp_path: Path,
) -> None:
    """Identity, not just a path string: source, tag and storage dtype all
    ride the same struct, so the child cannot be handed a narrowed half of a
    resolution again."""
    tree, binding = served
    binding = msgspec.structs.replace(binding, storage_dtype="fp8")
    request = _request(tree, binding, tmp_path)
    assert request.slots["pipeline"].ref == binding
    _chosen, siblings = _child_specs(request)
    assert siblings[0].models["pipeline"].storage_dtype == "fp8"


# ---------------------------------------------------------------------------
# 3. A child that still cannot resolve a slot says WHICH mint, and refuses
#    before it reads a weight
# ---------------------------------------------------------------------------


def test_an_unbound_required_slot_refuses_by_name_before_the_load(
    served: Tuple[Path, ModelRef], tmp_path: Path,
) -> None:
    """The honest-failure half. ``ValueError: slot 'pipeline': no resolved
    model ref`` named a symptom and no mint: family, key, lane and function
    all had to be reconstructed from a truncated activity row plus a DB read.
    """
    tree, _binding = served
    request = _request(tree, None, tmp_path)
    with pytest.raises(child_preflight.PreflightRefused) as exc:
        _child_specs(request)
    text = str(exc.value)
    for fact in ("pgw969", "ck1-catalog", "w8a8-lora64", "catalog-generate",
                 "'pipeline'", "sent no resolved binding"):
        assert fact in text, f"{fact!r} missing from: {text}"


def test_the_refusal_is_the_childs_terminal_word_on_a_real_subprocess(
    served: Tuple[Path, ModelRef], tmp_path: Path,
) -> None:
    """The real entrypoint: ``python -m gen_worker.mint_child request.json``,
    spawned by the parent's real supervisor, against a request whose slot
    binding is missing.

    RED at HEAD — the child got as far as the endpoint's handler and exited
    ``1`` (a CRASH the parent retries once, burning a second pod) with a bare
    ``ValueError``. A wiring gap is deterministic, so it must be
    ``EXIT_REFUSED``: terminal on the first attempt, and the report must name
    the mint.
    """
    tree, _binding = served
    request = _request(tree, None, tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    outcome = asyncio.run(mp.run_mint(
        request, workdir=tmp_path / "w", python=sys.executable, env=env))

    assert outcome.status == mp.REFUSED, (
        f"{outcome.status}: {outcome.detail or outcome.stderr_tail}")
    assert outcome.exit_code == mp.EXIT_REFUSED
    report = outcome.report
    assert report is not None and report.status == "refused"
    for fact in ("pgw969", "ck1-catalog", "'pipeline'", "catalog-generate"):
        assert fact in report.detail, f"{fact!r} missing from: {report.detail}"
    # The refusal precedes every side effect the CHILD could have: it wrote no
    # artifact, and it never reached the toolchain probe or a weight.
    assert not Path(request.target).exists()
    assert "warmup_forward" not in report.phases, (
        "the refusal must land before the endpoint's own handler runs — "
        f"phases={report.phases}")


def test_a_bound_slot_that_still_fails_names_the_divergence(
    tmp_path: Path,
) -> None:
    """The other half of the refusal: the parent's binding DID cross and the
    child still cannot resolve it. That is real divergence between the two
    processes, never a normal path, and it must not read like the wire gap
    above."""
    request = _request(
        tmp_path, ModelRef(source="tensorhub", path="harness/pick", tag="prod"),
        tmp_path)
    request = msgspec.structs.replace(
        request, slots={"nonexistent-slot": child_contract.MintSlot(
            ref=ModelRef(source="tensorhub", path="harness/pick", tag="prod"),
            path=str(tmp_path))})
    with pytest.raises(child_preflight.PreflightRefused) as exc:
        _child_specs(request)
    assert "sent no resolved binding" in str(exc.value)
    assert "nonexistent-slot" in str(exc.value), (
        "the refusal must show what the parent DID send")


# ---------------------------------------------------------------------------
# 4. The quiet half: a code default is not a substitute for a resolution
# ---------------------------------------------------------------------------


def test_a_code_default_never_stands_in_for_the_parents_pick(
    tmp_path: Path,
) -> None:
    """``JuggleEndpoint`` is the catalog shape WITH a ``default_checkpoint=``.
    Before this fix such a child resolved — to the DECLARED repo — while the
    parent served the hub's pick, so the mint traced graphs for a checkpoint
    the pod never runs and the crash class became a silent-wrongness class.
    """
    from harness import toy_endpoints as toy

    declared = wire_ref(toy.JUGGLE_DECLARED)
    picked = ModelRef(source="tensorhub", path="harness/juggle-pick", tag="prod")
    specs = registry.collect_endpoints(["harness.toy_endpoints"])
    chosen, siblings = child_preflight.select_specs(specs, "juggle-echo")
    assert wire_ref(chosen.models["pipeline"]) == declared

    child_preflight.bind_slots(siblings, {"pipeline": child_contract.MintSlot(
        ref=picked, path=str(tmp_path))})
    with __import__("tempfile").TemporaryDirectory() as tmp:
        ctx = warmup.warm_context(
            chosen, request_id="mint-child-x", local_output_dir=tmp)
    assert wire_ref(ctx.slots["pipeline"].ref) == wire_ref(picked)
    assert wire_ref(ctx.slots["pipeline"].ref) != declared


# ---------------------------------------------------------------------------
# 5. The structural pin: the parent RECORDS what it resolved
# ---------------------------------------------------------------------------


def test_the_default_is_empty_not_absent() -> None:
    """A hub-less or slotless boot must still produce a well-formed request."""
    bg = _BackgroundMint(
        spec=SimpleNamespace(name="gen"), instance=object(), snapshots=None,
        pendings={}, pipes={})
    assert bg.slots == {}


def test_the_paths_and_the_bindings_travel_together(
    served: Tuple[Path, ModelRef], tmp_path: Path,
) -> None:
    """One resolution, two halves. Every slot the parent materialized a tree
    for must arrive with the binding that named it — a request carrying one
    without the other is the shape this issue exists to make impossible."""
    tree, binding = served
    request = _request(tree, binding, tmp_path)
    assert wire_ref(request.slots["pipeline"].ref) == PICK_REF
    assert request.slots["pipeline"].path == str(tree)


# ---------------------------------------------------------------------------
# 6. pgw#974: the assertion above is now a TYPE, and the pod's request cannot
#    be written down at all
# ---------------------------------------------------------------------------


def test_a_slot_with_bytes_and_no_identity_cannot_be_constructed() -> None:
    """The pod's request, at the constructor.

    pgw#969 made the binding travel; it could not stop a producer from
    forgetting it, because ``snapshots`` and ``slot_bindings`` were two fields
    with two defaults. ``ref`` and ``path`` now have none.
    """
    with pytest.raises(TypeError):
        child_contract.MintSlot(path="/cas/snapshots/sha256:cafe")   # type: ignore[call-arg]
    with pytest.raises(TypeError):
        child_contract.MintSlot(ref=ModelRef(                        # type: ignore[call-arg]
            source="tensorhub", path="harness/pick", tag="prod"))
    # ...and the degenerate spelling of the same lie.
    with pytest.raises(ValueError):
        child_contract.MintSlot(
            ref=ModelRef(source="tensorhub", path="harness/pick", tag="prod"),
            path="")


def test_the_pods_wire_no_longer_decodes(
    served: Tuple[Path, ModelRef], tmp_path: Path,
) -> None:
    """The decisive negative, on a REAL request off the real producer chain.

    The boundary is a JSON file the child decodes, so the constructor above is
    only half the guard: strip the identity out of a request the parent
    actually built and the file must be REFUSED, not accepted as a complete
    resolution the way the two L40S pods' was.
    """
    tree, binding = served
    doc = msgspec.json.decode(msgspec.json.encode(_request(tree, binding, tmp_path)))
    assert doc["slots"]["pipeline"]["path"] == str(tree)

    del doc["slots"]["pipeline"]["ref"]
    with pytest.raises(msgspec.ValidationError) as exc:
        msgspec.json.decode(msgspec.json.encode(doc), type=mp.MintRequest)
    assert "ref" in str(exc.value)

    # The whole slot may still be absent — that is a resolution the parent did
    # not make, and `assert_slots_resolvable` is what refuses it, by name.
    doc["slots"] = {}
    empty = msgspec.json.decode(msgspec.json.encode(doc), type=mp.MintRequest)
    assert empty.slots == {}
    with pytest.raises(child_preflight.PreflightRefused):
        _child_specs(empty)
