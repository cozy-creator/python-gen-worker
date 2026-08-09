"""pgw#904 — the RunAttempt cutover: exact ExecutionSpec dispatch, the
neutral driver, the ordered arm, and the death of the wall deadline.

Four surfaces, each on the path production runs:

  1. The Plan head over the REAL gRPC boundary (harness.hub_double, the one
     legal double): `run_attempt` -> PlanFactory -> PlanLedger -> the shared
     driver -> JobResult. Slot bindings materialize from the grant's
     digest-keyed transport; a pinned digest the grant does not carry is a
     typed RETRYABLE; a re-dispatched attempt under a DIFFERENT spec digest
     fails the attempt closed (one result, INVALID, naming both digests);
     an identical replay re-acks without re-executing.
  2. pgw#902 box 4 at the value+instance layer: A -> B -> A leaves BOTH
     instance keys addressable and the third dispatch REUSES A's ready
     record (same live instance object, no third setup).
  3. Part (d): no wall bound anywhere — an advancing handler outlives any
     wire `timeout_ms`; the ONLY abort authority is the registry's own
     `progress.self_diagnosis()` confession, and it produces a typed
     RETRYABLE naming the stalled counter.
  4. The ordered arm (`fleet_cells.arm_ordered`): obeys the named backend,
     refuses typed (never falls back, never self-mints) on receipt refusal,
     publisher mismatch or arm failure — and catalog discovery is GONE as a
     module, so "a fake catalog cannot affect delivery" holds by
     unreachability, not by filtering.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import pytest

import gen_worker.executor as executor_mod
from gen_worker import RequestContext, aot_delivery, cell_adopt, endpoint, fleet_cells
from gen_worker import worker_function
from gen_worker import progress as progress_mod
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.plan import PlanFactory, PlanRefusal

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_accept_for, is_ready, is_result_for
from harness.toy_endpoints import (
    BOOT_UNREACHABLE_PIPELINE,
    BOOT_UNREACHABLE_VAE,
    EchoIn,
    EchoOut,
)

_TIMEOUT = 15.0


def _payload(text: str = "marco") -> bytes:
    return msgspec.msgpack.encode(EchoIn(text=text))


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


def _spec(
    function_name: str,
    digest: str = "spec-a",
    *,
    spec_version: int = 1,
    slots: Tuple[Tuple[str, str, str], ...] = (),
) -> pb.ExecutionSpec:
    """A minimal valid eager ExecutionSpec; ``slots`` is (slot, ref, digest)."""
    spec = pb.ExecutionSpec(
        digest=digest,
        spec_version=spec_version,
        function_name=function_name,
        output_mode=pb.OUTPUT_MODE_URL,
    )
    spec.release.org = "acme"
    spec.release.endpoint = "toy"
    spec.release.release_id = "rel-1"
    spec.release.image_digest = "sha256:" + "1" * 64
    spec.release.code_closure_id = "cc-1"
    spec.numerical_lane.weights = pb.WEIGHT_LANE_BF16
    spec.arm.graph_contract_digest = "gc-1"
    spec.arm.shape = pb.ARM_SHAPE_BRANCHLESS
    spec.arm.backend = pb.STEADY_BACKEND_EAGER_ONLY
    spec.topology.accelerator = "none"
    spec.topology.execution_groups = 1
    spec.components.SetInParent()
    for slot, ref, snap_digest in slots:
        binding = spec.components.slots.add()
        binding.slot = slot
        binding.ref = ref
        binding.snapshot_digest = snap_digest
    return spec


def _attempt(
    request_id: str,
    attempt: int,
    spec: pb.ExecutionSpec,
    *,
    payload: bytes = b"",
    snapshots: Optional[Dict[str, pb.Snapshot]] = None,
) -> pb.RunAttempt:
    msg = pb.RunAttempt(input_payload=payload or _payload())
    msg.attempt.request_id = request_id
    msg.attempt.attempt = attempt
    msg.spec.CopyFrom(spec)
    msg.grant.SetInParent()
    for digest, snap in (snapshots or {}).items():
        presigned = msg.grant.snapshots[digest]
        for f in snap.files:
            row = presigned.files.add()
            row.path = f.path
            row.size_bytes = f.size_bytes
            row.digest = f.digest
            row.url = f.url
    return msg


# ---------------------------------------------------------------------------
# 1. The Plan head over the real gRPC boundary
# ---------------------------------------------------------------------------


def test_run_attempt_dispatches_and_completes() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_attempt=_attempt("ra-echo", 1, _spec("echo")))
        conn.wait_for(is_accept_for("ra-echo"))
        res = conn.wait_for(is_result_for("ra-echo")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert _decode(res.inline).response == "polo"


def test_run_attempt_unknown_spec_version_refuses_invalid() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_attempt=_attempt(
            "ra-badver", 1, _spec("echo", spec_version=99)))
        res = conn.wait_for(is_result_for("ra-badver")).job_result
        assert res.status == pb.JOB_STATUS_INVALID
        assert "spec_version" in res.safe_message


def test_run_attempt_replay_reacks_without_reexecution() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        msg = _attempt("ra-replay", 1, _spec("sleepy", digest="spec-r"))
        conn.send(run_attempt=msg)
        conn.wait_for(is_accept_for("ra-replay"))
        conn.send(run_attempt=msg)  # retransmit: identical digest
        conn.wait_for_count(is_accept_for("ra-replay"), 2)
        res = conn.wait_for(is_result_for("ra-replay")).job_result
        assert res.status == pb.JOB_STATUS_OK
        time.sleep(0.3)
        results = [
            m for m in conn.received
            if m.WhichOneof("msg") == "job_result"
            and m.job_result.request_id == "ra-replay"
        ]
        assert len(results) == 1, "an identical replay must not re-execute"


def test_run_attempt_digest_conflict_fails_the_attempt_closed() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_attempt=_attempt(
            "ra-conflict", 1, _spec("sleepy", digest="spec-first")))
        conn.wait_for(is_accept_for("ra-conflict"))
        conn.send(run_attempt=_attempt(
            "ra-conflict", 1, _spec("sleepy", digest="spec-second")))
        res = conn.wait_for(is_result_for("ra-conflict")).job_result
        assert res.status == pb.JOB_STATUS_INVALID
        assert "spec-first" in res.safe_message
        assert "spec-second" in res.safe_message
        time.sleep(0.6)  # outlive the sleepy handler: no second result
        results = [
            m for m in conn.received
            if m.WhichOneof("msg") == "job_result"
            and m.job_result.request_id == "ra-conflict"
        ]
        assert len(results) == 1, (
            "a digest conflict is ONE terminal result for the attempt")
        # The worker is not wedged: fresh work still serves.
        conn.send(run_attempt=_attempt("ra-after", 1, _spec("echo")))
        assert conn.wait_for(
            is_result_for("ra-after")).job_result.status == pb.JOB_STATUS_OK


def _slot_snapshots(
    blobs: BlobHost, payload_a: bytes, payload_vae: bytes,
    *, pipeline_name: str = "pipeline-blob", vae_name: str = "vae-blob",
) -> Tuple[pb.Snapshot, pb.Snapshot]:
    return (
        blobs.one_file_snapshot(f"snap-{pipeline_name}", pipeline_name, payload_a),
        blobs.one_file_snapshot(f"snap-{vae_name}", vae_name, payload_vae),
    )


def test_run_attempt_binds_slots_from_grant_transport(tmp_path: Path) -> None:
    """The identity/delivery re-join: refs + digests ride the spec, bytes
    ride the grant keyed BY CONTENT DIGEST, and the handler serves the
    delivered bytes."""
    blobs = BlobHost(tmp_path)
    try:
        pipe_snap, vae_snap = _slot_snapshots(
            blobs, b"exact-pipeline-bytes", b"exact-vae-bytes")
        spec = _spec(
            "slot-boot-precedence", digest="spec-slots",
            slots=(
                ("pipeline", wire_ref(BOOT_UNREACHABLE_PIPELINE), pipe_snap.digest),
                ("vae", wire_ref(BOOT_UNREACHABLE_VAE), vae_snap.digest),
            ))
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_attempt=_attempt(
                "ra-slots", 1, spec,
                snapshots={pipe_snap.digest: pipe_snap, vae_snap.digest: vae_snap}))
            res = conn.wait_for(is_result_for("ra-slots"), timeout=_TIMEOUT).job_result
            assert res.status == pb.JOB_STATUS_OK
            assert _decode(res.inline).response == "exact-pipeline-bytes"
    finally:
        blobs.shutdown()


def test_run_attempt_missing_grant_transport_is_typed_retryable(tmp_path: Path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        pipe_snap, vae_snap = _slot_snapshots(blobs, b"pipe", b"vae")
        spec = _spec(
            "slot-boot-precedence", digest="spec-nogrant",
            slots=(
                ("pipeline", wire_ref(BOOT_UNREACHABLE_PIPELINE), pipe_snap.digest),
                ("vae", wire_ref(BOOT_UNREACHABLE_VAE), vae_snap.digest),
            ))
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            # The grant carries the pipeline's transport and NOT the vae's.
            conn.send(run_attempt=_attempt(
                "ra-nogrant", 1, spec,
                snapshots={pipe_snap.digest: pipe_snap}))
            res = conn.wait_for(is_result_for("ra-nogrant")).job_result
            assert res.status == pb.JOB_STATUS_RETRYABLE
            assert vae_snap.digest in res.safe_message
    finally:
        blobs.shutdown()


def test_run_attempt_unbound_required_slot_refuses_invalid(tmp_path: Path) -> None:
    """Refuse-never-default, the Plan half: the manifest is THE exact model
    set — a required declared Slot it does not bind refuses, and the code
    default is never resurrected."""
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_attempt=_attempt(
            "ra-unbound", 1, _spec("slot-boot-precedence", digest="spec-unbound")))
        res = conn.wait_for(is_result_for("ra-unbound")).job_result
        assert res.status == pb.JOB_STATUS_INVALID
        assert "pipeline" in res.safe_message and "vae" in res.safe_message


# ---------------------------------------------------------------------------
# 2. pgw#902 box 4: A -> B -> A keeps both instance keys addressable and
#    reuses A's ready record
# ---------------------------------------------------------------------------


def test_alternating_specs_reuse_ready_instances(tmp_path: Path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        pipe_a = blobs.one_file_snapshot("snap-pipe-a", "pipe-a", b"bytes-of-A")
        pipe_b = blobs.one_file_snapshot("snap-pipe-b", "pipe-b", b"bytes-of-B")
        vae = blobs.one_file_snapshot("snap-vae-ab", "vae-ab", b"vae-bytes")
        ref_a = wire_ref(BOOT_UNREACHABLE_PIPELINE)
        # Same repo, a different tag: a fixed slot allows tag/flavor picks
        # (gw#583 gates only repo identity), and the differing binding set
        # derives a DIFFERENT instance key.
        ref_b = wire_ref(Hub("harness/boot-precedence-pipeline", tag="beta"))
        assert ref_b != ref_a
        vae_ref = wire_ref(BOOT_UNREACHABLE_VAE)
        transport = {
            pipe_a.digest: pipe_a, pipe_b.digest: pipe_b, vae.digest: vae}

        def spec_for(digest: str, pipe_ref: str, snap: pb.Snapshot) -> pb.ExecutionSpec:
            return _spec(
                "slot-boot-precedence", digest=digest,
                slots=(
                    ("pipeline", pipe_ref, snap.digest),
                    ("vae", vae_ref, vae.digest),
                ))

        with hub_double() as (scheduler, harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            def run(rid: str, digest: str, pipe_ref: str, snap: pb.Snapshot) -> str:
                conn.send(run_attempt=_attempt(
                    rid, 1, spec_for(digest, pipe_ref, snap),
                    snapshots=transport))
                res = conn.wait_for(
                    is_result_for(rid), timeout=_TIMEOUT).job_result
                assert res.status == pb.JOB_STATUS_OK, res.safe_message
                return _decode(res.inline).response

            def slot_instances(ex: Any) -> Dict[Any, Any]:
                return {
                    key: rec.instance for key, rec in ex._classes.items()
                    if rec.ready
                    and type(rec.instance).__name__ == "SlotBootPrecedenceEndpoint"
                }

            assert run("ra-a1", "spec-A", ref_a, pipe_a) == "bytes-of-A"
            ex = harness.worker.lifecycle.executor
            ready_a = slot_instances(ex)
            assert len(ready_a) == 1
            assert run("ra-b1", "spec-B", ref_b, pipe_b) == "bytes-of-B"
            assert run("ra-a2", "spec-A2", ref_a, pipe_a) == "bytes-of-A"
            ready_after = slot_instances(ex)
            # BOTH instance keys stay addressable, and A's ready record was
            # REUSED — the third dispatch served from the same live object.
            assert len(ready_after) == 2
            ((key_a, instance_a),) = ready_a.items()
            assert ready_after[key_a] is instance_a
    finally:
        blobs.shutdown()


# ---------------------------------------------------------------------------
# 3. Part (d): liveness + progress-staleness, never a clock
# ---------------------------------------------------------------------------


def test_stall_confession_aborts_typed_and_advancing_work_is_never_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two dispatches through one REAL executor: an advancing handler that
    outlives any historical bound completes; a handler running while the
    process CONFESSES a stall (`progress.self_diagnosis()`) is aborted
    RETRYABLE naming the stalled counter. No wall clock decides either."""
    from gen_worker.executor import Executor
    from gen_worker.registry import extract_specs

    @endpoint
    class Crawler:
        @worker_function()
        def crawl(self, ctx: RequestContext, payload: EchoIn) -> EchoOut:
            time.sleep(0.4)
            return EchoOut(response="finished")

    specs = extract_specs(Crawler)
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    monkeypatch.setattr(executor_mod, "_STALL_POLL_S", 0.05)

    async def _drive() -> None:
        ex = Executor(specs, _send)
        # Advancing (no confession): completes despite running far past the
        # poll interval.
        await ex.handle_run_job(pb.RunJob(
            request_id="r-advancing", attempt=1, function_name="crawl",
            input_payload=_payload(), timeout_ms=50))
        task = ex.jobs[("r-advancing", 1)].task
        assert task is not None
        await task

        # Confessed stall: the registry's own self-diagnosis is the abort
        # authority.
        stalled = progress_mod.Snapshot(
            name="infer:steps", unit="steps", done=3.0, total=0.0,
            rate_per_s=0.0, age_s=999.0, window_s=300.0, elapsed_s=999.0)
        monkeypatch.setattr(progress_mod, "self_diagnosis", lambda: stalled)
        await ex.handle_run_job(pb.RunJob(
            request_id="r-stalled", attempt=1, function_name="crawl",
            input_payload=_payload()))
        task = ex.jobs[("r-stalled", 1)].task
        assert task is not None
        await task

    asyncio.run(_drive())
    results = {
        m.job_result.request_id: m.job_result
        for m in sent if m.WhichOneof("msg") == "job_result"}
    assert results["r-advancing"].status == pb.JOB_STATUS_OK
    stalled_res = results["r-stalled"]
    assert stalled_res.status == pb.JOB_STATUS_RETRYABLE
    assert "self_stalled" in stalled_res.safe_message
    assert "infer:steps" in stalled_res.safe_message


# ---------------------------------------------------------------------------
# 4. The ordered arm: obey exactly, refuse typed, no fallbacks
# ---------------------------------------------------------------------------


@dataclass
class _Cfg:
    family: str = "sdxl"
    lora_bucket: int = 0


class _Pipe:
    pass


def _ordered(**kw: Any) -> fleet_cells.ArmOutcome:
    args: Dict[str, Any] = dict(
        backend="aot_cell", artifact=None,
        delivered_ref="root/family-sdxl#ck", delivered_digest="sha256:" + "2" * 64,
        expected=None, publisher_org="org-a")
    args.update(kw)
    return fleet_cells.arm_ordered(_Pipe(), _Cfg(), None, **args)


def _expected() -> Any:
    from gen_worker.aot_identity import ExpectedIdentity

    return ExpectedIdentity(
        cell_key="ck", toolchain_digest="t", env_seal_digest="e",
        graph_contract_digest="g", publisher_org="org-a")


def test_ordered_eager_arm_arms_nothing_and_refuses_an_artifact(tmp_path: Path) -> None:
    out = _ordered(backend="eager_only")
    assert not out.armed
    assert out.eager_reason == cell_adopt.EagerPhase.HUB_ORDERED_EAGER
    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"x")
    with pytest.raises(fleet_cells.OrderedArmError, match="artifact_on_eager_arm"):
        _ordered(backend="eager_only", artifact=art)


def test_ordered_aot_arm_requires_the_receipt_gate(tmp_path: Path) -> None:
    from gen_worker import receipts

    receipts.reset()
    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"x")
    with pytest.raises(
        fleet_cells.OrderedArmError, match="receipt_gate_unconfigured",
    ):
        _ordered(artifact=art, expected=_expected())


def _receipt_stub(publisher_org_id: str) -> Any:
    class _Receipt:
        pass

    r = _Receipt()
    r.publisher_org_id = publisher_org_id  # type: ignore[attr-defined]
    return r


def test_ordered_aot_arm_verifies_the_exact_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import receipts

    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"x")
    monkeypatch.setattr(receipts, "configured", lambda: True)
    monkeypatch.setattr(
        receipts, "verify_delivered_artifact",
        lambda artifact, family: _receipt_stub("org-b"))
    with pytest.raises(fleet_cells.OrderedArmError, match="publisher_mismatch"):
        _ordered(artifact=art, expected=_expected())

    # A receipt refusal is typed too — never a drop-to-eager.
    def _refuse(artifact: Any, family: str) -> Any:
        raise receipts.ReceiptError("signature_invalid", "not the hub's key")

    monkeypatch.setattr(receipts, "verify_delivered_artifact", _refuse)
    with pytest.raises(
        fleet_cells.OrderedArmError, match="artifact_receipt_refused",
    ):
        _ordered(artifact=art, expected=_expected())


def test_ordered_aot_arm_arms_exactly_the_named_cell_or_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker import receipts
    from gen_worker.models import provision

    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"x")
    monkeypatch.setattr(receipts, "configured", lambda: True)
    monkeypatch.setattr(
        receipts, "verify_delivered_artifact",
        lambda artifact, family: _receipt_stub("org-a"))
    seen: List[Any] = []

    def _arm(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any, bucket: int,
             *, expected: Any = None) -> AdoptOutcome:
        seen.append((artifact, expected))
        return AdoptOutcome.hit("family=sdxl key=ck")

    monkeypatch.setattr(provision, "arm_aot", _arm)
    out = _ordered(artifact=art, expected=_expected())
    assert out.armed
    # pgw#903's declared-identity expectation reached the arm, and the
    # adoption row is bound to the NAMED identity.
    assert seen == [(art, _expected())]
    (row,) = out.adoptions
    assert row.ref == "root/family-sdxl#ck" and row.armed

    # A refused arm is terminal for the attempt: no self-mint, no eager.
    monkeypatch.setattr(
        provision, "arm_aot",
        lambda *a, **k: AdoptOutcome.miss(
            "expected_identity_mismatch", "cell_key: expected ck, have other"))
    with pytest.raises(
        fleet_cells.OrderedArmError, match="expected_identity_mismatch",
    ):
        _ordered(artifact=art, expected=_expected())


def test_catalog_discovery_is_gone_by_construction() -> None:
    """Box 3/4: a fake catalog (or a reordered one) cannot affect delivery
    because nothing on the connected path LISTS a catalog any more — the
    fetch-and-filter module was deleted whole, not filtered harder."""
    with pytest.raises(ModuleNotFoundError):
        import gen_worker.aot_cells  # noqa: F401
    import inspect

    policy = inspect.getsource(fleet_cells._arming_policy)
    assert ".discover(" not in policy
    assert not hasattr(fleet_cells, "aot_cells")


# ---------------------------------------------------------------------------
# 5. Named-artifact delivery (the one legitimate byte check)
# ---------------------------------------------------------------------------


def test_named_artifact_missing_content_and_cache_hit(tmp_path: Path) -> None:
    import hashlib

    payload = b"cell-bytes"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    # No grant transport and no cache: typed missing_content.
    with pytest.raises(
        aot_delivery.NamedArtifactUnavailable, match="missing_content",
    ) as exc:
        aot_delivery.materialize_named_artifact(
            "root/family-sdxl#ck", digest, None,
            cache_dir=tmp_path, what="request r-1 attempt 1 spec d")
    assert "spec d" in str(exc.value)

    # A digest-verified cache hit needs no transport at all.
    dest = tmp_path / "aot-cells" / f"{digest.split(':', 1)[-1]}.tar.gz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(payload)
    got = aot_delivery.materialize_named_artifact(
        "root/family-sdxl#ck", digest, None,
        cache_dir=tmp_path, what="request r-1 attempt 1 spec d")
    assert got == dest

    # Stale cache bytes are refused-and-refetched, and with no transport
    # that is a typed miss — never a silent serve of the wrong bytes.
    dest.write_bytes(b"not-the-cell")
    with pytest.raises(
        aot_delivery.NamedArtifactUnavailable, match="missing_content",
    ):
        aot_delivery.materialize_named_artifact(
            "root/family-sdxl#ck", digest, None,
            cache_dir=tmp_path, what="request r-1 attempt 1 spec d")


# ---------------------------------------------------------------------------
# 6. PlanFactory refusals stay addressable on the wire head
# ---------------------------------------------------------------------------


def test_unaddressable_run_attempt_is_a_transport_fault() -> None:
    msg = pb.RunAttempt()
    with pytest.raises(PlanRefusal):
        PlanFactory.from_run_attempt(msg)
