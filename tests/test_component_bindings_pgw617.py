"""pgw#617: hierarchical slot bindings — load base composition, substitute
components from ``RunJob.ModelBinding.components`` (th#980 companion).

Real hub-double boundary (worker/lifecycle/executor real, blake3 blob host):

  * flat bindings (empty components map) behave byte-identically to today;
  * a dispatched component override loads from its OWN materialized snapshot
    and is injected into the base ``from_pretrained`` (load-then-substitute);
  * the composition (base + sorted component refs) is the instance identity —
    a component-only rebind re-runs setup, and the flat instance survives
    untouched alongside it;
  * an unknown component name refuses typed (ComponentSubstitutionError),
    never mid-denoise;
  * registry relaxation: a ``selected_by=`` slot may omit
    ``default_checkpoint`` (deploy-time bindings seed the hub mapping).
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import RequestContext, Slot, endpoint
from gen_worker.api.binding import wire_ref
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import (
    COMPOSED_DECLARED,
    COMPOSED_SETUPS,
    EchoIn,
    EchoOut,
)


def _payload(**kw: object) -> bytes:
    return msgspec.msgpack.encode(EchoIn(**kw))  # type: ignore[arg-type]


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


BASE_REF = wire_ref(COMPOSED_DECLARED)
VAE_REF = "harness/override-vae:prod"


def _base_snapshot(blobs: BlobHost) -> pb.Snapshot:
    model_index = (
        b'{"_class_name": "ToyComposedPipeline",'
        b' "vae": ["harness.toy_endpoints", "ToyVae"],'
        b' "transformer": ["harness.toy_endpoints", "ToyVae"]}'
    )
    return blobs.snapshot("snap-composed-base", [
        blobs.file("mi", model_index, path_in_snapshot="model_index.json"),
        blobs.file("tw", b"base-transformer", path_in_snapshot="transformer/weights.txt"),
        blobs.file("vw", b"base-vae", path_in_snapshot="vae/weights.txt"),
    ])


def _vae_snapshot(blobs: BlobHost) -> pb.Snapshot:
    # Override tree carries the component at its ROOT (no subfolder): the
    # loader falls back from `<root>/vae` to `<root>`.
    return blobs.snapshot("snap-override-vae", [
        blobs.file("ow", b"override-vae", path_in_snapshot="weights.txt"),
    ])


def _run(conn, rid: str, models, snapshots) -> pb.JobResult:
    conn.send(run_job=pb.RunJob(
        request_id=rid, attempt=1, function_name="composed-echo",
        input_payload=_payload(), models=models, snapshots=snapshots,
    ))
    return conn.wait_for(is_result_for(rid)).job_result


def test_flat_then_substituted_then_flat(tmp_path) -> None:
    """One connection, three dispatches: flat -> vae override -> flat.

    The override run substitutes the injected module (load-then-substitute)
    and derives a NEW instance (setup re-runs); the final flat run reuses
    the FIRST instance (no third setup) — composition identity separates
    them both ways."""
    COMPOSED_SETUPS.clear()  # module-global across same-process workers
    blobs = BlobHost(tmp_path)
    try:
        base_snap = _base_snapshot(blobs)
        vae_snap = _vae_snapshot(blobs)
        flat = [pb.ModelBinding(slot="pipeline", ref=BASE_REF)]
        subst = [pb.ModelBinding(
            slot="pipeline", ref=BASE_REF, components={"vae": VAE_REF},
        )]
        snaps = {BASE_REF: base_snap, VAE_REF: vae_snap}
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            res = _run(conn, "r-flat", flat, {BASE_REF: base_snap})
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            flat_out = _decode(res.inline).response
            assert "base=base-transformer" in flat_out
            assert "vae=base-vae" in flat_out
            assert "injected=False" in flat_out
            assert "setups=1" in flat_out

            res = _run(conn, "r-subst", subst, snaps)
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            subst_out = _decode(res.inline).response
            assert "base=base-transformer" in subst_out
            assert "vae=override-vae" in subst_out  # substituted bytes
            assert "injected=True" in subst_out      # via components= kwarg
            assert "setups=2" in subst_out           # NEW composition identity

            res = _run(conn, "r-flat-2", flat, {BASE_REF: base_snap})
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            back_out = _decode(res.inline).response
            assert "vae=base-vae" in back_out
            assert "setups=2" in back_out  # flat instance reused, no reload
    finally:
        blobs.shutdown()


def test_unknown_component_refuses_typed(tmp_path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        base_snap = _base_snapshot(blobs)
        vae_snap = _vae_snapshot(blobs)
        models = [pb.ModelBinding(
            slot="pipeline", ref=BASE_REF,
            components={"text_encoder": VAE_REF},
        )]
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            res = _run(conn, "r-unknown", models,
                       {BASE_REF: base_snap, VAE_REF: vae_snap})
            assert res.status != pb.JOB_STATUS_OK
            assert "ComponentSubstitutionError" in res.safe_message
            assert "text_encoder" in res.safe_message
    finally:
        blobs.shutdown()


# ---------------------------------------------------------------------------
# Registry relaxation: selected_by slots may be default-less (pgw#617).
# ---------------------------------------------------------------------------


class _PickIn(msgspec.Struct):
    model: str = ""


class _PickOut(msgspec.Struct):
    response: str


def test_selected_by_slot_may_omit_default_checkpoint() -> None:
    @endpoint(models={"pipeline": Slot(str, selected_by="model")})
    class DefaultlessCatalog:
        def setup(self, pipeline: str) -> None:
            self.pipeline_path = pipeline

        def pick(self, ctx: RequestContext, payload: _PickIn) -> _PickOut:
            return _PickOut(response="ok")

    specs = extract_specs(DefaultlessCatalog)
    assert [s.name for s in specs] == ["pick"]
    assert specs[0].slots["pipeline"].default_checkpoint is None


def test_selected_by_still_validates_payload_field() -> None:
    @endpoint(models={"pipeline": Slot(str, selected_by="nonexistent")})
    class BadCatalog:
        def setup(self, pipeline: str) -> None: ...

        def bad_pick(self, ctx: RequestContext, payload: _PickIn) -> _PickOut:
            return _PickOut(response="no")

    with pytest.raises(ValueError, match="names no field"):
        extract_specs(BadCatalog)
