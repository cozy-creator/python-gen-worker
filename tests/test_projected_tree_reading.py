"""Reading a PROJECTED tree: who may, who must refuse, and what they say.

# pgw#1513: the eager bridge must refuse a projected tree, by name.

# The incident, and why it looked like a short write for two days

Two endpoints (sd15, sdxl), two volumes, 20x apart in size, identical
`SafetensorError: header too large`, hub source bytes verified intact. Every
short-write theory was dead on arrival once the write paths were read: the CAS
hashes and size-checks every object before committing it, tensorfs'
materializer re-hashes each object AND the whole file and refuses a short read,
and `project_snapshot` builds in a scratch directory and renames.

What actually happens is that a projected tree's tensor containers are ~128 B
TFSSTUB1 POINTER STUBS — the weights live in the CAS — and the stock
safetensors reader that `from_pretrained` uses knows nothing about stubs. It
reads the stub's first eight bytes as a header length and raises. The stub is a
FIXED SIZE regardless of the model behind it, which is exactly why a 3.4 GB and
a 68 GB checkpoint failed byte-identically, and that coincidence is what a
genuine truncation could never produce.

`ctx.load` is supposed to route a projected tree to the pgw#1380 streaming
engine, whose whole job is reading those stubs. The eager `from_pretrained`
bridge is documented for "a tree with no chunk store behind it — a bare
download, a local run, a fixture". Reaching the bridge WITH a projected tree
means the engine declined it, and the bridge then reports a corrupt checkpoint
for a checkpoint that is intact.

So the bridge refuses instead, naming the stub and its two sizes.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, cast

import pytest

from gen_worker._vendor.tensorfs.project import stub_bytes
from gen_worker.serving.context import (
    DeployBinding,
    _projection_declined_because,
    LoadContext,
    ProjectedTreeNotStreamable,
    _projection_artifacts,
)

# The two shapes from the field, so the fixed-size-stub signature is asserted
# rather than described.
_SD15_BYTES = 3_400_000_000
_SDXL_BYTES = 68_000_000_000


def _stubbed_tree(root: Path, named_bytes: int) -> Path:
    """A tree whose tensor container is a pointer, as `_project_entry` writes it."""
    tree = root / "snapshots" / ("sha256:" + "b7" * 32)
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(
        stub_bytes("a" * 64, named_bytes)
    )
    (tree / "model_index.json").write_text('{"_class_name": "X"}')
    return tree


class _Pipeline:
    """A loader that would happily be called — so the refusal is the guard's,
    not an accident of the fixture."""

    called: list[Any] = []

    @classmethod
    def from_pretrained(cls, path: Any, **kwargs: Any) -> "_Pipeline":
        cls.called.append(path)
        from safetensors import safe_open

        # Exactly what diffusers does: the stock reader, on the container.
        with safe_open(
            str(Path(path) / "unet" / "diffusion_pytorch_model.safetensors"),
            framework="pt",
        ):
            pass
        return cls()


@pytest.mark.parametrize("named", [_SD15_BYTES, _SDXL_BYTES])
def test_the_eager_bridge_REFUSES_a_projected_tree_by_name(
    tmp_path: Path, named: int
) -> None:
    """Both field shapes, and the refusal must carry the numbers."""
    tree = _stubbed_tree(tmp_path, named)
    _Pipeline.called.clear()

    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(checkpoint_ref="acme/m@1", checkpoint_dir=tree),
        engine=None,  # the engine declined — the state the field pods were in
    )

    with pytest.raises(ProjectedTreeNotStreamable) as caught:
        ctx.load(_Pipeline)

    assert _Pipeline.called == [], (
        "the bridge must refuse BEFORE calling the loader — reaching "
        "from_pretrained is what produced `header too large`")
    message = str(caught.value)
    assert "PROJECTED" in message
    assert "unet/diffusion_pytorch_model.safetensors" in message, (
        f"the refusal must name the member: {message}")
    assert f"{named:,}" in message, (
        f"the refusal must name the bytes the stub STANDS FOR: {message}")
    assert "header too large" in message, (
        "the refusal must name the error it is replacing, or the next reader "
        f"searching that string will not find this: {message}")


def test_the_stub_signature_is_FIXED_SIZE_across_a_20x_model(tmp_path: Path) -> None:
    """The coincidence that killed every short-write theory, asserted.

    A truncation is proportional to what was being written; a stub is not. Two
    models 20x apart produce stubs within a byte or two of each other, which is
    why the two field errors were identical.
    """
    small = _projection_artifacts(_stubbed_tree(tmp_path / "a", _SD15_BYTES))
    large = _projection_artifacts(_stubbed_tree(tmp_path / "b", _SDXL_BYTES))

    assert len(small) == 1 and len(large) == 1
    (_, small_on_disk, small_named) = small[0]
    (_, large_on_disk, large_named) = large[0]

    assert large_named // small_named >= 19, "fixture is not a 20x spread"
    assert abs(large_on_disk - small_on_disk) <= 8, (
        f"stub sizes {small_on_disk} vs {large_on_disk} should be ~equal; that "
        "near-equality IS the signature that ruled out a short write")
    assert small_on_disk < 200 and large_on_disk < 200


def test_a_MATERIALIZED_tree_still_takes_the_eager_bridge(tmp_path: Path) -> None:
    """The guard must not fire on the trees the bridge legitimately serves.

    A bare download / fixture / local run has no chunk store and no stubs, and
    it is exactly what the eager bridge exists for. If this test ever fails,
    the guard has started refusing the local development path.
    """
    tree = tmp_path / "snapshots" / "plain"
    (tree / "unet").mkdir(parents=True)
    # Real (tiny) safetensors bytes, not a pointer.
    import torch
    from safetensors.torch import save_file

    save_file(
        {"w": torch.zeros(4)},
        str(tree / "unet" / "diffusion_pytorch_model.safetensors"),
    )
    (tree / "model_index.json").write_text('{"_class_name": "X"}')

    assert _projection_artifacts(tree) == []
    _Pipeline.called.clear()
    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(checkpoint_ref="acme/m@1", checkpoint_dir=tree),
        engine=None,
    )
    ctx.load(_Pipeline)
    assert _Pipeline.called == [tree], "the eager bridge must still run here"


def test_a_tree_with_a_MISSING_PIN_is_not_answered_as_resident(tmp_path: Path) -> None:
    """The one state that is byte-perfect and still unservable.

    `resolve_projection` recovers a tree's manifest through a `snapshot:<key>`
    ref keyed on the tree's own directory name. Without it the streaming
    engine cannot bind, `ctx.load` falls to the eager bridge, and the bridge
    reads a stub. Verification alone does not catch this: every byte is
    correct.

    So `announce_resident` refuses, and the refusal sends the ref through
    `ensure_local`, which re-pins WITHOUT moving bytes.
    """
    import asyncio

    from gen_worker.models import projection
    from gen_worker.models.refs import WireRef
    from gen_worker.models.store import ModelStore
    from gen_worker.pb import worker_scheduler_pb2 as pb
    from gen_worker._vendor.tensorfs.project import stub_bytes

    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "c3" * 32)
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "x.safetensors").write_bytes(stub_bytes("a" * 64, 3_400_000_000))

    # Byte-perfect and stubbed, but nothing pinned it.
    assert projection.stub_at_any(tree) is True
    assert projection.resolve_projection(tree) is None

    async def noop(_msg: Any) -> None:
        return None

    store = ModelStore(noop, cache_dir=base)
    snap = pb.Snapshot(digest="sha256:" + "c3" * 32)
    snap.files.add(path="unet/x.safetensors", size_bytes=3_400_000_000,
                   digest="sha256:" + "a" * 64)
    ref = WireRef("acme/unpinned")

    async def run() -> bool:
        return await store.announce_resident(ref, snap)

    answered = asyncio.run(run())
    assert answered is False, (
        "a tree no engine can bind to must not be answered as resident — "
        "that is the state that reaches the eager bridge and reports "
        "`header too large` about an intact checkpoint")


def test_a_tree_whose_OBJECTS_WERE_COLLECTED_says_so_and_not_malformed(
    tmp_path: Path,
) -> None:
    """se#790's state D, measured on a real 5.6 GB `@composed-v3` tree.

    A tree's manifest pin is the ONLY GC root its objects have. Drop the pin,
    run a GC, and every object is deleted while the tree stands: the tensor
    containers are still stubs and the NON-tensor files — `model_index.json`
    among them — become dangling symlinks. `skeleton.build` then reports
    "carries no model_index.json" about a tree that has one, which is a
    reader-level fact rendered as a claim about the checkpoint.

    The refusal must name the real condition, and must not be confused with
    the cheap missing-pin case: these bytes are GONE and must be re-fetched.
    """
    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    objects = base / "objects"
    objects.mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "d4" * 32)
    tree.mkdir(parents=True)

    # A projected non-tensor entry whose object has been collected.
    (tree / "model_index.json").symlink_to(
        Path("..") / ".." / "objects" / "sha256" / "de" / "ad" / ("de" * 32)
    )
    assert (tree / "model_index.json").is_symlink()
    assert not (tree / "model_index.json").exists(), "fixture must be dangling"

    _Pipeline.called.clear()
    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(checkpoint_ref="acme/m@1", checkpoint_dir=tree),
        engine=None,
    )
    with pytest.raises(ProjectedTreeNotStreamable) as caught:
        ctx.load(_Pipeline)

    assert _Pipeline.called == []
    message = str(caught.value)
    assert "COLLECTED" in message, f"the real condition must be named: {message}"
    assert "model_index.json" in message
    assert "RE-FETCHED" in message, (
        "this must not be confused with the cheap re-pin case — the bytes are "
        f"gone: {message}")
    assert "carries no model_index.json" in message, (
        "the refusal must name the FALSE message it is pre-empting, so the "
        f"next reader searching that string lands here: {message}")


def test_collected_entries_puts_model_index_json_FIRST(tmp_path: Path) -> None:
    """se#790's refinement, and the reason it is not cosmetic.

    `model_index.json` is the entry whose absence produces the false "carries
    no model_index.json", so it is the one a reader most needs to see in the
    truncated list the refusal shows. Plain alphabetical order drops it out of
    that window on any tree with early-alphabet components — measured on a real
    `@composed-v3` tree, where it landed 2nd of 3 by luck of `dit/` sorting
    first. Everything after it keeps sorted order so the list stays diffable.
    """
    from gen_worker.models import projection

    for rel in ("model_index.json", "aaa/config.json", "dit/config.json",
                "text_encoder/config.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(Path("..") / "objects" / "sha256" / "de" / "ad" / ("de" * 32))

    got = projection.collected_entries(tmp_path)
    assert got[0] == "model_index.json", (
        f"the entry that causes the false refusal must lead the list: {got}")
    assert got[1:] == sorted(got[1:]), f"the tail must stay sorted: {got}"
    assert "model_index.json" in projection.collected_refusal(tmp_path, got)[:400], (
        "and it must survive into the truncated (shown) window of the message")


def test_the_decline_REASON_survives_the_wire_truncation(tmp_path: Path) -> None:
    """The clause this refusal exists to deliver must survive the 512-char cap.

    # pgw#1513 follow-up, from the FIRST FIELD FIRING of this refusal.

    It worked — a pod reported `ProjectedTreeNotStreamable` naming four stubs
    with bytes-on-disk against bytes-named — and then lost the one clause the
    whole mechanism exists for. `JobResult.safe_message` is sliced to 512 chars
    in `worker.py::_send_result` (deliberate: that layer declines to put an
    unbounded string on a wire it owns), and `THE ENGINE DECLINED BECAUSE: …`
    sat at the END of a longer message. It was truncated away, so the pod said
    WHAT was wrong and never WHY — and WHY is the half nobody can reconstruct
    from the logs afterwards.

    Leading with the reason is structural: it survives any cap at any layer,
    including caps nobody has told us about. This test asserts that property
    against the real limit and the real field shape, so a later edit that
    "tidies" the message ordering fails here instead of on a rental.
    """
    #: The exact slice `_send_result` applies.
    WIRE_CAP = 512

    tree = Path("/tensorhub-endpoint-cache/cas/snapshots/sha256:" + "b7" * 32)
    stubs = [
        (f"component_{i}/diffusion_pytorch_model.safetensors", 128, 3_400_000_000)
        for i in range(4)
    ]
    declined = _projection_declined_because(tmp_path / "snapshots" / "absent")

    message = str(ProjectedTreeNotStreamable(tree, stubs, declined))
    truncated = message[:WIRE_CAP]

    assert message.startswith("ENGINE DECLINED:"), (
        "the reason must LEAD — anything after the first 512 characters is not "
        f"guaranteed to reach a reader: {message[:80]!r}")
    assert declined[:120] in truncated, (
        "the decline reason must survive the wire cap intact; this is the "
        f"clause the refusal exists to deliver. Truncated form: {truncated!r}")
    assert "pointer stub(s), NOT weights" in truncated, (
        "the stub count should also fit inside the cap — it is what identifies "
        "the shape on sight")


def test_a_missing_manifest_pin_is_REPAIRED_not_refused_forever(tmp_path: Path) -> None:
    """The pod's condition 3, verbatim, and the loop it was stuck in.

    # pgw#1526: measured on an L4, same volume, pin the only variable:
    #   ENGINE DECLINED: the manifest pin `snapshot:sha256:5bd90786…` is
    #   MISSING from the store at /tmp/tensorhub-cache/cas

    `_pin_manifest` has exactly ONE caller, `ensure_snapshot`. But
    `_materialize_local`'s cached short-circuits return a tree already on disk
    without going through it, so a tree that reaches residency by any other
    route is unpinned — and `ensure_local` can never repair it, because it
    short-circuits on that same tree forever. pgw#1513 refused and bounced to
    `ensure_local` expecting a re-pin; on a pod the bounce lands right back on
    the unpinned tree. Refusing was right; refusing forever was the bug.

    So: stage a real tree, DELETE its pin the way the pod's is missing, and
    assert the store repairs it — without moving bytes — so the streaming
    engine can bind and the checkpoint actually serves.
    """
    import asyncio

    from gen_worker import activity
    from gen_worker.models import projection
    from test_weight_position import (  # type: ignore[import-not-found]
        OBJECT_BYTES,
        OBJECTS,
        _Origin,
        _Wire,
        _pb_snapshot,
        _store,
    )
    from gen_worker.models.refs import WireRef

    ref = WireRef("acme/model-a")
    origin = _Origin()
    try:
        files = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        snapshot = _pb_snapshot([(p, b, origin.put(b)) for p, b in files])
        cas = tmp_path / "cas"

        async def stage() -> Path:
            wire = _Wire()
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            store = _store(wire, cas)
            return await store.ensure_local(ref, snapshot)

        tree = asyncio.run(stage())
        assert projection.resolve_projection(tree) is not None, "fixture never pinned"

        # THE POD'S STATE: bytes fine, pin gone.
        # Refs are stored under a hashed id, so the pin is found by the NAME it
        # records rather than by its filename.
        import json as _json

        wanted = "snapshot:" + tree.name
        pins = [
            q for q in (cas / "refs").rglob("*")
            if q.is_file() and _json.loads(q.read_bytes()).get("name") == wanted
        ]
        assert len(pins) == 1, f"expected exactly one pin for {wanted}, got {pins}"
        pins[0].unlink()
        assert projection.resolve_projection(tree) is None, "fixture is not in state 3"
        before = sorted((p, p.stat().st_size) for p in tree.rglob("*") if p.is_file())

        async def reboot() -> bool:
            wire = _Wire()
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            store = _store(wire, cas)
            store.rescan_disk()
            return await store.announce_resident(ref, snapshot)

        answered = asyncio.run(reboot())

        assert projection.resolve_projection(tree) is not None, (
            "the pin must be REPAIRED — bouncing to `ensure_local` lands on the "
            "same unpinned tree forever, which is the loop the pod was in")
        assert answered is True, (
            "with the pin repaired the pod is genuinely resident and must say so")
        after = sorted((p, p.stat().st_size) for p in tree.rglob("*") if p.is_file())
        assert before == after, f"NO BYTES MAY MOVE; tree changed: {before} -> {after}"
    finally:
        origin.close()


def test_the_boot_census_NAMES_an_unpinned_stubbed_tree(
    tmp_path: Path, caplog: Any
) -> None:
    """pgw#1536: the one combination that cannot serve must be loud at boot.

    The pod incident turned on a tree existing WITHOUT its manifest pin, and
    nobody could say whether that tree predated the boot or was built during
    it — because nothing ever looked. A census at boot answers that for free
    on any run that happens anyway, instead of costing a rental.

    UNPINNED + STUBBED is the state that cannot be served, so it is the one
    that logs at ERROR rather than INFO.
    """
    import logging

    from gen_worker._vendor.tensorfs.project import stub_bytes
    from gen_worker.models.store import ModelStore

    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "e5" * 32)
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "x.safetensors").write_bytes(stub_bytes("a" * 64, 3_400_000_000))

    async def noop(_msg: Any) -> None:
        return None

    store = ModelStore(noop, cache_dir=base)
    with caplog.at_level(logging.INFO, logger="gen_worker.models.store"):
        store._census_snapshot_pins()

    census = [r for r in caplog.records if "snapshot census at boot" in r.message]
    assert census, "the boot census must emit a line per tree"
    row = census[0]
    assert "pinned=False" in row.getMessage() and "projected=True" in row.getMessage()
    assert row.levelno == logging.ERROR, (
        "unpinned + stubbed is the ONE combination that cannot serve — it must "
        f"be loud, not an INFO line nobody greps: level={row.levelname}")
    assert "UNREADABLE by the streaming engine" in row.getMessage()


def test_the_census_fingerprint_SURVIVES_TEARDOWN_as_an_activity_row(
    tmp_path: Path,
) -> None:
    """pgw#1541: a log line dies with the pod; the fingerprint must not.

    pgw#1536's census emitted logger lines ONLY, which made it invisible to
    every DB-reading harness and deleted at teardown — a rental ending in a
    teardown script silently loses the exact fact the census exists to
    preserve. Its "free on any run" value held only for a runner who knew to
    pull `cozy logs` while the pod was still alive.

    So the one state that cannot serve also lands as a
    `worker_activity_events` row. Asserted here on the REAL emit path, not a
    mock: `activity.bind_sink` is bound to a recorder exactly as `arun` binds
    it to the worker's wire.
    """
    import asyncio

    from gen_worker import activity
    from gen_worker._vendor.tensorfs.project import stub_bytes
    from gen_worker.models.store import ModelStore
    from gen_worker.pb import worker_scheduler_pb2 as pb

    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    for name in ("sha256:" + "e5" * 32, "sha256:" + "f6" * 32):
        tree = base / "snapshots" / name
        (tree / "unet").mkdir(parents=True)
        (tree / "unet" / "x.safetensors").write_bytes(
            stub_bytes("a" * 64, 3_400_000_000)
        )

    sent: list[Any] = []

    async def sink(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    async def run() -> None:
        activity.bind_sink(sink, asyncio.get_running_loop())
        store = ModelStore(sink, cache_dir=base)
        store._census_snapshot_pins()
        for _ in range(8):
            await asyncio.sleep(0)

    asyncio.run(run())

    rows = [
        m.activity_update for m in sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == activity.KIND_SNAPSHOT_CENSUS
    ]
    assert len(rows) == 1, (
        "exactly ONE fingerprint row per boot — a row per healthy tree would be "
        f"a heartbeat, not a fingerprint; got {len(rows)}")
    row = rows[0]
    assert row.phase == "unpinned_projected"
    assert row.step == 2 and row.total_steps == 2, (
        "the counts must be NUMERIC so the state is a query and not a string "
        f"search; got step={row.step} total={row.total_steps}")
    assert "unservable=2" in row.detail and "e5e5" in row.detail


def test_a_HEALTHY_census_STILL_emits_a_row_so_absence_means_one_thing(
    tmp_path: Path,
) -> None:
    """The row is unconditional, and this test is the reason why.

    My first cut fired the row only when something was unservable, reasoning
    that a healthy row is noise. That was wrong, and it broke the exact
    question the census exists to answer. The residue is PREDATE-vs-BUILT-
    THIS-BOOT, and it is read off the census being ABSENT at boot while the
    failure shows up later. Fire-on-bad-only makes absence ambiguous — healthy
    boot, crashed census, and no-trees-on-disk all render identically as "no
    row", and an absent instrument that reads as a clean bill of health is
    worse than no instrument at all.

    So: absence now means exactly one thing — the census did not run.
    """
    import asyncio

    from gen_worker import activity
    from gen_worker.models.store import ModelStore
    from gen_worker.pb import worker_scheduler_pb2 as pb

    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    # A materialized tree: real bytes, no stubs, needs no pin.
    tree = base / "snapshots" / "plain"
    tree.mkdir(parents=True)
    (tree / "weights.bin").write_bytes(b"real bytes")

    sent: list[Any] = []

    async def sink(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    async def run() -> None:
        activity.bind_sink(sink, asyncio.get_running_loop())
        ModelStore(sink, cache_dir=base)._census_snapshot_pins()
        for _ in range(8):
            await asyncio.sleep(0)

    asyncio.run(run())

    rows = [
        m.activity_update for m in sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == activity.KIND_SNAPSHOT_CENSUS
    ]
    assert len(rows) == 1, "a healthy boot must still be ON THE RECORD"
    assert rows[0].phase == "all_servable"
    assert rows[0].step == 0 and rows[0].total_steps == 1
    assert ":P-" in rows[0].detail or ":--" in rows[0].detail, (
        f"per-tree state must be packed into the row: {rows[0].detail}")


def test_an_EMPTY_store_emits_the_row_that_ANSWERS_predate_vs_this_boot(
    tmp_path: Path,
) -> None:
    """`of=0` at boot is the residue's answer, not a boring case to skip.

    An empty snapshot store at boot means any stubbed tree found later was
    BUILT THIS BOOT. That is the whole predate-vs-built-this-boot question,
    settled by one row — so the empty case is the LAST one that should have
    been an early return, and it used to be one.
    """
    import asyncio

    from gen_worker import activity
    from gen_worker.models.store import ModelStore
    from gen_worker.pb import worker_scheduler_pb2 as pb

    base = tmp_path / "cas"
    for d in ("refs", "objects", "snapshots"):
        (base / d).mkdir(parents=True)

    sent: list[Any] = []

    async def sink(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    async def run() -> None:
        activity.bind_sink(sink, asyncio.get_running_loop())
        ModelStore(sink, cache_dir=base)._census_snapshot_pins()
        for _ in range(8):
            await asyncio.sleep(0)

    asyncio.run(run())

    rows = [
        m.activity_update for m in sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == activity.KIND_SNAPSHOT_CENSUS
    ]
    assert len(rows) == 1, (
        "an EMPTY store must be on the record — it is the strongest "
        "predate-vs-built-this-boot evidence the census can produce")
    assert rows[0].total_steps == 0 and rows[0].step == 0
    assert rows[0].phase == "all_servable"


def test_the_repair_outcome_SURVIVES_THE_512_CHAR_CAP_on_a_realistic_refusal(
    tmp_path: Path,
) -> None:
    """pgw#1542: position two, because the tail gets sliced off.

    `worker.py::_send_result` slices `safe_message` at 512 chars. pgw#1513
    already lost the decline reason to that cap once, by putting it last. The
    repair outcome has the same property — per-incident, unreconstructable
    afterwards — so appending it would have repeated the identical bug.

    This asserts on a REALISTIC message: a full-length snapshot tree name and
    three long stub paths, which is what a real refusal carries. The naive
    append placement fails this test.
    """
    from gen_worker.models import projection as proj
    from gen_worker.serving.context import ProjectedTreeNotStreamable

    tree = tmp_path / "snapshots" / ("sha256:" + "5b" * 32)
    tree.mkdir(parents=True)
    proj.record_pin_outcome(tree.name, "ATTEMPTED and FAILED: PermissionError")

    stubs = [
        (f"unet/diffusion_pytorch_model-{i:05d}-of-00007.safetensors", 128,
         3_438_167_536)
        for i in range(4)
    ]
    exc = ProjectedTreeNotStreamable(tree, stubs, declined="no manifest pin")

    capped = str(exc)[:512]
    assert "repair attempted: ATTEMPTED and FAILED: PermissionError" in capped, (
        "the repair outcome must survive the 512-char wire cap — that is the "
        f"entire reason it sits in position two. Capped message: {capped!r}")
    # And the decline reason still leads, unchanged by the insertion.
    assert capped.startswith("ENGINE DECLINED: no manifest pin")


def test_an_UNATTEMPTED_repair_never_renders_as_a_clean_bill_of_health() -> None:
    """Absence must say `not attempted`, never blank.

    A blank field reads as "no repair was needed", which is the false-negative
    this whole change exists to prevent — the same missing-renders-as-fine
    failure the boot census hit one layer up.
    """
    from gen_worker.models.projection import pin_outcome

    assert pin_outcome("a-tree-nobody-touched") == "not attempted"
    assert pin_outcome("") == "not attempted"


def test_the_outcome_registry_is_PER_TREE_not_a_global_last_writer(
    tmp_path: Path,
) -> None:
    """A pod serves many trees; a global slot would misattribute the repair.

    Recording one tree's failed repair and then reading a DIFFERENT tree's
    refusal must not blame the second tree for the first tree's outcome.
    """
    from gen_worker.models import projection as proj
    from gen_worker.serving.context import ProjectedTreeNotStreamable

    good = tmp_path / "snapshots" / "tree-that-was-repaired"
    bad = tmp_path / "snapshots" / "tree-nobody-repaired"
    for d in (good, bad):
        d.mkdir(parents=True)
    proj.record_pin_outcome(good.name, "REPAIRED, pin rewritten")

    msg = str(ProjectedTreeNotStreamable(bad, [("w.safetensors", 128, 1)], "x"))
    assert "repair attempted: not attempted" in msg
    assert "REPAIRED" not in msg, (
        "one tree's repair outcome must never be attributed to another tree's "
        f"refusal: {msg}")


def test_the_registry_is_BOUNDED_so_a_long_lived_pod_cannot_grow_it() -> None:
    """A pod that sees thousands of trees must not accumulate an entry each."""
    from gen_worker.models import projection as proj

    for i in range(proj._PIN_OUTCOMES_CAP * 3):
        proj.record_pin_outcome(f"bounded-probe-{i}", "not needed: already pinned")
    assert len(proj._PIN_OUTCOMES) <= proj._PIN_OUTCOMES_CAP
    # The most RECENT survive — a live refusal is about a recent repair.
    assert proj.pin_outcome(f"bounded-probe-{proj._PIN_OUTCOMES_CAP * 3 - 1}") == (
        "not needed: already pinned")


class _PipelineCls:
    """Stands in for a real pipeline class: it HAS `from_pretrained`.

    `ctx.load` refuses a class without one BEFORE it reaches the projection
    checks, so a bare `object` never gets far enough to exercise the repair.
    Every real pipeline (`StableDiffusionPipeline`, `FluxPipeline`, …) has it.
    """

    @classmethod
    def from_pretrained(cls, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise AssertionError(
            "the eager bridge must never run for an unpinned projected tree")


def _unpinned_projected_tree(tmp_path: Path) -> Path:
    """A tree that is STUBBED and UNPINNED — the fleet's exact failing state."""
    from gen_worker._vendor.tensorfs.project import stub_bytes

    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "7c" * 32)
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(
        stub_bytes("d" * 64, 3_438_167_536)
    )
    return tree


def test_the_repair_RUNS_ON_THE_REAL_PATH_with_a_MISMATCHED_ref_spelling(
    tmp_path: Path,
) -> None:
    """pgw#1543, on a REAL `ModelStore` — no stubbed store, no stubbed pin.

    Two things are proven here, and the second is the one a mock would miss.

    1. The repair reaches the path that actually refuses. `ensure_pinned`
       shipped in pgw#1526 with two callers — `announce_resident` (boot) and
       `_materialize_local` (materialization) — and NEITHER is on `ctx.load`.
       A pod whose tree was materialized already, or whose announce ran before
       the repair existed, hits the engine decline on every request and refuses
       forever while a working repair sits two modules away. That is a ~19h
       fleet-wide serve outage that needs no log line to explain.

    2. The repair does not depend on REF SPELLING. The store banks snapshots by
       ref; the serving path holds `DeployBinding.checkpoint_ref`, which is the
       resolver's `pick.ref` and need not be the same string. A ref-keyed
       lookup would SILENTLY NO-OP on a mismatch — miss, return False, refuse
       exactly as before — and every stubbed test would still pass. So this
       banks under one spelling and serves under a deliberately DIFFERENT one,
       and still requires the pin to be written.
    """
    from gen_worker.models import projection
    from gen_worker.models import store as store_mod
    from gen_worker.models.refs import WireRef
    from gen_worker.models.store import ModelStore
    from gen_worker.pb import worker_scheduler_pb2 as pb

    digest = "sha256:" + "7c" * 32
    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    tree = base / "snapshots" / digest
    (tree / "unet").mkdir(parents=True)
    # Under 64 MiB on purpose: a real manifest for a larger file carries chunk
    # entries, and `_manifest` rightly refuses to build one without them. The
    # pin mechanism under test is identical either way, and the field's
    # multi-GB shapes are asserted by the stub-signature tests above.
    (tree / "unet" / "x.safetensors").write_bytes(stub_bytes("a" * 64, 4096))

    assert projection.stub_at_any(tree) is True
    assert projection.resolve_projection(tree) is None, "fixture must be UNPINNED"

    async def noop(_msg: Any) -> None:
        return None

    store = ModelStore(noop, cache_dir=base)
    snap = pb.Snapshot(digest=digest)
    snap.files.add(path="unet/x.safetensors", size_bytes=4096,
                   digest="sha256:" + "a" * 64)
    # Banked under the STORE's spelling ...
    store.bank_snapshot(WireRef("acme/sd15@1"), snap)
    store_mod.bind_active_store(store)
    try:
        ctx: Any = LoadContext(
            # ... and served under a DIFFERENT one. A ref-keyed repair no-ops here.
            binding=DeployBinding(
                checkpoint_ref="acme/sd15", checkpoint_dir=tree),
            engine=None,          # the engine DECLINED — the failing state
            device="cuda",
        )
        # The load cannot COMPLETE here — this fixture has no `model_index.json`
        # and no real weights, so once the repair lets the engine bind, the
        # skeleton builder rightly objects. That is fine and is not what is
        # under test: the claim is that the PIN GETS WRITTEN on this path.
        with contextlib.suppress(Exception):
            ctx.load(_Pipeline)
    finally:
        store_mod.bind_active_store(cast(Any, None))

    assert projection.resolve_projection(tree) is not None, (
        "THE PIN MUST ACTUALLY BE WRITTEN on the real path — this is the whole "
        "fix, and a repair that resolves to nothing here is the silent no-op "
        "that a ref-keyed lookup would have produced")
    assert projection.pin_outcome(tree.name) == "REPAIRED, pin rewritten", (
        "and pgw#1542's instrument must record it, so a post-mortem reader can "
        f"tell a repaired pod from an unattempted one: {projection.pin_outcome(tree.name)!r}")


def test_a_REFUSING_pod_REPAIRS_AT_THE_DECLINE_and_serves(tmp_path: Path) -> None:
    """The conversion the fleet needs: would-have-refused now returns a pipeline.

    The companion to the test above. That one proves the pin is really written
    on the real path; this one proves that once the engine can bind, the
    request SERVES instead of refusing — a currently-refusing pod becoming a
    serving pod on its next request, with no bytes moved.
    """
    from gen_worker.models import store as store_mod
    import gen_worker.serving.streaming as streaming_mod

    tree = _unpinned_projected_tree(tmp_path)
    built = object()
    repaired: list[Path] = []

    class _Engine:
        def build(self, cls: Any, *, checkpoint_dir: Path, lane: Any) -> Any:
            return built

    class _Store:
        def banked_snapshot_for_tree(self, tree_name: str) -> Any:
            return object()

        def ensure_pinned(self, ref: str, path: Path, snap: Any) -> bool:
            repaired.append(Path(path))
            return True

    store_mod.bind_active_store(cast(Any, _Store()))
    orig = streaming_mod.engine_for
    streaming_mod.engine_for = lambda *a, **k: _Engine()  # type: ignore[assignment]
    try:
        ctx: Any = LoadContext(
            binding=DeployBinding(checkpoint_ref="ep/sd15@1", checkpoint_dir=tree),
            engine=None,
            device="cuda",
        )
        out = ctx.load(_Pipeline)
    finally:
        streaming_mod.engine_for = orig  # type: ignore[assignment]
        store_mod.bind_active_store(cast(Any, None))

    assert out is built, "a repaired pod must SERVE, not refuse"
    assert repaired == [tree], "the repair must target the refusing tree"


def test_a_repair_that_FAILS_leaves_the_original_refusal_intact(
    tmp_path: Path,
) -> None:
    """A failed repair must not replace a precise diagnosis with a traceback.

    The refusal is the most valuable artifact this path produces — it is the
    one channel that survives the pod. A repair attempt that raises, or that
    re-pins and still gets no engine, must fall through to it unchanged.
    """
    from gen_worker.models import store as store_mod
    from gen_worker.serving.context import (
        DeployBinding, LoadContext, ProjectedTreeNotStreamable,
    )

    tree = _unpinned_projected_tree(tmp_path)

    class _ExplodingStore:
        def banked_snapshot_for_tree(self, tree_name: str) -> Any:
            return object()

        def ensure_pinned(self, ref: str, path: Path, snap: Any) -> bool:
            raise RuntimeError("pin write blew up")

    store_mod.bind_active_store(cast(Any, _ExplodingStore()))
    try:
        ctx: Any = LoadContext(
            binding=DeployBinding(checkpoint_ref="ep/sd15@1", checkpoint_dir=tree),
            engine=None,
            device="cuda",
        )
        with pytest.raises(ProjectedTreeNotStreamable) as caught:
            ctx.load(_PipelineCls)
    finally:
        store_mod.bind_active_store(cast(Any, None))

    assert "ENGINE DECLINED" in str(caught.value), (
        "the original refusal must survive a failed repair")


def test_repair_is_NOT_attempted_when_the_objects_were_COLLECTED(
    tmp_path: Path,
) -> None:
    """The GC'd case must NEVER be re-pinned — that would serve a lie.

    se#790 measured the worse state on a real tree: the manifest pin is a
    tree's only GC root, so a tree outliving its pin and then meeting a GC pass
    has its objects DELETED (5.6 GB there, 134 GB on H3), leaving stubs plus
    dangling symlinks. Writing a pin over absent bytes would convert an honest
    refusal into a corrupt serve, so `collected` is checked FIRST and the
    repair must not run.
    """
    from gen_worker.models import store as store_mod
    from gen_worker.models import projection as proj_mod
    from gen_worker.serving.context import (
        DeployBinding, LoadContext, ProjectedTreeNotStreamable,
    )

    tree = _unpinned_projected_tree(tmp_path)
    attempted: list[Path] = []

    class _Store:
        def banked_snapshot_for_tree(self, tree_name: str) -> Any:
            return object()

        def ensure_pinned(self, ref: str, path: Path, snap: Any) -> bool:
            attempted.append(Path(path))
            return True

    store_mod.bind_active_store(cast(Any, _Store()))
    orig = proj_mod.collected_entries
    proj_mod.collected_entries = (  # type: ignore[assignment]
        lambda root: ["unet/diffusion_pytorch_model.safetensors"])
    try:
        ctx: Any = LoadContext(
            binding=DeployBinding(checkpoint_ref="ep/sd15@1", checkpoint_dir=tree),
            engine=None,
            device="cuda",
        )
        with pytest.raises(ProjectedTreeNotStreamable):
            ctx.load(_PipelineCls)
    finally:
        proj_mod.collected_entries = orig  # type: ignore[assignment]
        store_mod.bind_active_store(cast(Any, None))

    assert attempted == [], (
        "a tree whose objects were COLLECTED must never be re-pinned — the "
        "bytes are gone and a pin over them is a corrupt serve")


# ── pgw#1544: the refusal contradicted itself, and nobody was asking ──────────
#
# THE FIELD ARTIFACT, one string, from `request_state.error_message_safe` on the
# standing stack (pod 6r92qnflyt9blk, release b98aad88…, pin e3527cab):
#
#   ENGINE DECLINED: the manifest pin `snapshot:sha256:5bd90786…` is MISSING
#   from the store at /tmp/tensorhub-cache/cas … | repair attempted: not
#   needed: already pinned | PROJECTED TREE, 4 pointer stub(s) …
#
# The pin is MISSING and the repair says already pinned, in one sentence. It was
# read as two lookups of one fact disagreeing. It is not: there was only ever
# ONE lookup. `_projection_declined_because`'s last branch was the else of two
# STRUCTURAL checks and asserted the pin was missing without ever looking, so
# the repair MEASURED and the message ASSUMED — and on the fleet the message was
# the one that was wrong. The pin was there the whole time.
#
# Which raises the question the contradiction was hiding: if the tree is pinned
# and streamable, why was the eager bridge holding it at all? Because on a POD
# nobody asks for an engine. `engine_for` has two production call sites:
# `EndpointHost` — which only `serving/__main__.py` (the local CLI) and
# `cli/daemon.py` construct — and pgw#1543's repair. The serverless worker
# builds `ServeLoop` (`worker.py:582`) with NO `engine=`, and
# `ServeLoop._backend_factory` hands `engine=self._engine` to every
# `LoadContext`. So every projected tree on every pod fell to the eager bridge
# and refused, on every request, blaming the store.
#
# That is why every local red/green passed while every pod failed: the tests
# went through `EndpointHost`, which asks, and the fleet went through
# `ServeLoop`, which does not.


def _pinned_projected_tree(tmp_path: Path) -> Path:
    """sd15's real shape, through the REAL store: 4 pointer stubs, PINNED.

    Built by `ModelStore.ensure_local` over a real CAS and a real HTTP origin,
    so the pin is the one production writes rather than one a fixture staged.
    """
    import asyncio

    from gen_worker import activity
    from gen_worker.models.refs import WireRef
    from test_weight_position import (  # type: ignore[import-not-found]
        OBJECT_BYTES,
        _Origin,
        _Wire,
        _pb_snapshot,
        _store,
    )

    members = [
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "text_encoder/model.fp16.safetensors",
        # The member the field refusal named, at its field size.
        "safety_checker/model.fp16.safetensors",
    ]
    origin = _Origin()
    try:
        files = [(p, bytes([i + 1]) * OBJECT_BYTES) for i, p in enumerate(members)]
        snapshot = _pb_snapshot([(p, b, origin.put(b)) for p, b in files])

        async def stage() -> Path:
            wire = _Wire()
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            store = _store(wire, tmp_path / "tensorhub-cache" / "cas")
            return await store.ensure_local(WireRef("tensorhub/sd15-base@prod"),
                                            snapshot)

        return asyncio.run(stage())
    finally:
        origin.close()


def test_a_PINNED_projected_tree_REACHES_THE_STREAMING_ENGINE_on_the_serve_path(
    tmp_path: Path,
) -> None:
    """pgw#1544's reproduction, and the fleet fix: the pod's own state.

    A real store, a real pinned projected tree in sd15's shape, and a
    `LoadContext` built EXACTLY as `ServeLoop._backend_factory` builds one on a
    pod — `engine=None`, no `device=`, no `io=`. Before this fix that
    combination refused every request with a message blaming a pin that was
    sitting right there.

    The load cannot COMPLETE over a fixture tree with no `model_index.json` —
    the skeleton builder rightly objects, and that objection is the proof: it
    comes from INSIDE the streaming engine, which is a place the pod's requests
    never reached.
    """
    from gen_worker.models import projection
    from gen_worker.serving.streaming.skeleton import SkeletonError

    tree = _pinned_projected_tree(tmp_path)
    stubs = _projection_artifacts(tree)
    assert len(stubs) == 4, f"fixture must be sd15-shaped: {stubs}"
    assert any(p == "safety_checker/model.fp16.safetensors" for p, _, _ in stubs)
    assert projection.resolve_projection(tree) is not None, (
        "fixture must be PINNED — the pod's actual state, and the one the "
        "refusal claimed was impossible")

    _Pipeline.called.clear()
    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(
            checkpoint_ref="tensorhub/sd15-base@prod", checkpoint_dir=tree),
        engine=None,  # ServeLoop hands this to every load on every pod
    )

    with pytest.raises(SkeletonError):
        ctx.load(_Pipeline)

    assert _Pipeline.called == [], (
        "the eager bridge must never see a projected tree — reaching "
        "from_pretrained is the `header too large` lie pgw#1513 killed")
    assert ctx._engine is not None, (
        "THE ENGINE MUST BE ASKED FOR. Nothing on the serverless worker path "
        "calls `engine_for`: `worker.py` builds ServeLoop with no engine= and "
        "ServeLoop hands that None down. A projected tree with a good pin then "
        "refused forever while the engine that reads it was never constructed")


def test_the_PRODUCTION_dispatcher_hands_its_loads_NO_ENGINE(
    tmp_path: Path,
) -> None:
    """The premise of the test above, asserted on the real object.

    `worker.py:582` constructs `ServeLoop` with no `engine=`. This builds one
    the same way and reads what its own factory manufactures. If a future
    change starts binding an engine here, this test says so — rather than
    leaving the test above passing for a reason that has quietly changed.
    """
    import sys

    from gen_worker.serving.loader import load_endpoint_module
    from gen_worker.serving.residency import ResidencyManager
    from gen_worker.serving.serve_loop import ServeLoop

    fixtures = Path(__file__).resolve().parent / "release_fixtures"

    class _Sizer:
        def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
            return 1 << 20

        def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
            return 0

    class _Resolver:
        def resolve(self, model_cls: type, checkpoint_ref: str) -> Any:
            raise AssertionError("not reached")

        def default_pick(self, model_cls: type, slot_name: str) -> str:
            return "tensorhub/sd15-base@prod"

    sys.path.insert(0, str(fixtures))
    try:
        loaded = load_endpoint_module("sd15_shaped_endpoint")
    finally:
        sys.path.remove(str(fixtures))

    loop = ServeLoop(
        loaded,
        residency=ResidencyManager(1 << 30, _Sizer()),
        resolver=_Resolver(),
        lane_contract="sd15.diffusers@1+plain.bf16@1",
    )
    model_cls = next(iter(loaded.models))
    binding = DeployBinding(
        checkpoint_ref="tensorhub/sd15-base@prod",
        checkpoint_dir=tmp_path / "tree",
    )
    key = (model_cls, "tensorhub/sd15-base@prod", "sd15.diffusers@1+plain.bf16@1")
    backend = loop._backend_factory(model_cls, binding, key)()

    assert backend.load_context._engine is None, (
        "the pod's dispatcher binds NO streaming engine — which is why "
        "`ctx.load` must ask for one itself")


def test_the_decline_reason_is_MEASURED_and_never_ASSERTED(tmp_path: Path) -> None:
    """The self-contradiction, killed at its source.

    `_projection_declined_because` returned "the manifest pin ... is MISSING"
    as the ELSE of two structural checks — a sentence about the store that
    nothing in the function had looked at. Both states are driven here on real
    trees: the claim must follow the disk, in both directions.
    """
    from gen_worker.models import projection

    pinned = _pinned_projected_tree(tmp_path / "good")
    assert projection.resolve_projection(pinned) is not None
    said = _projection_declined_because(pinned)
    assert "is MISSING" not in said, (
        "a PINNED tree must never be told its pin is missing — that sentence "
        f"sent three lanes to the store while the defect was in the wiring: {said}")
    assert "RESOLVES" in said and "WIRING" in said

    unpinned = _unpinned_projected_tree(tmp_path / "bad")
    assert projection.resolve_projection(unpinned) is None
    still = _projection_declined_because(unpinned)
    assert "is MISSING" in still, (
        f"and the genuine missing-pin diagnosis must survive intact: {still}")


def test_the_refusal_and_the_repair_OUTCOME_can_never_contradict_each_other(
    tmp_path: Path,
) -> None:
    """The field artifact, as an invariant.

    "the pin is MISSING" and "not needed: already pinned" must never appear in
    one refusal again, whatever else changes. Driven over a real pinned tree
    with the repair's own outcome recorded against it — the exact pairing the
    pod produced.
    """
    from gen_worker.models import projection

    tree = _pinned_projected_tree(tmp_path)
    projection.record_pin_outcome(tree.name, "not needed: already pinned")

    message = str(ProjectedTreeNotStreamable(
        tree,
        _projection_artifacts(tree),
        _projection_declined_because(tree),
    ))
    assert "not needed: already pinned" in message
    assert "is MISSING" not in message, (
        "ONE STRING MUST NOT CONTRADICT ITSELF. The pod said the pin was "
        f"missing and the repair said it was already pinned: {message}")


def test_the_boot_census_FIRES_ON_THE_RECONCILE_where_rows_reach_the_hub(
    tmp_path: Path,
) -> None:
    """pgw#1544: the census had a caller and still produced zero rows.

    `rescan_disk` runs inside `arun` BEFORE the transport task is created, and
    on the field pods NOTHING emitted in that window survived: the four
    verification pods of 2026-08-19/20 each produced exactly `weight_fetch` and
    `snapshot_pull` rows and not one row of any boot-window kind —
    `snapshot_census`, `process_role` and `boot_stages` are all absent, while
    pods from a day earlier carry all three. So a second boot-window caller
    would have bought a second unreadable row.

    `announce_resident` runs on the reconcile, after HelloAck, in the same
    window that delivered `snapshot_pull` — and before this pod materializes
    anything, so it still answers the predate-vs-built-this-boot question.
    Driven through the REAL store with a REAL bound activity sink: the check is
    that a ROW is EMITTED, not that a symbol exists.
    """
    import asyncio

    from gen_worker import activity
    from gen_worker.models.refs import WireRef
    from gen_worker.models.store import ModelStore
    from gen_worker.pb import worker_scheduler_pb2 as pb
    from test_weight_position import _Wire  # type: ignore[import-not-found]

    base = tmp_path / "cas"
    (base / "refs").mkdir(parents=True)
    (base / "objects").mkdir(parents=True)
    tree = base / "snapshots" / ("sha256:" + "e5" * 32)
    (tree / "unet").mkdir(parents=True)
    (tree / "unet" / "x.safetensors").write_bytes(stub_bytes("a" * 64, 3_400_000_000))

    wire = _Wire()

    async def reconcile() -> None:
        activity.bind_sink(wire.send, asyncio.get_running_loop())
        store = ModelStore(wire.send, cache_dir=base)
        snapshot = pb.Snapshot(digest="sha256:" + "e5" * 32)
        # Twice on purpose: a pod reconciles every configured ref, and the
        # census is one statement about the whole store.
        for _ in range(2):
            await store.announce_resident(WireRef("tensorhub/sd15-base@prod"),
                                          snapshot)
        # The activity sink ships on the loop; let those tasks run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(reconcile())

    rows = [u for u in wire.updates if u.kind == activity.KIND_SNAPSHOT_CENSUS]
    assert len(rows) == 1, (
        "the residency answer must emit the census EXACTLY ONCE per process — "
        f"a per-ref row is N rows for one fact: {[u.detail for u in rows]}")
    row = rows[0]
    assert "of=1" in row.detail and "unservable=1" in row.detail, (
        f"the row must carry the census, not just exist: {row.detail}")
    assert row.phase == "unpinned_projected"
