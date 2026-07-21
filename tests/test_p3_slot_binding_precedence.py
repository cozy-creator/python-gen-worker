"""P3 (th#960/pgw#609 design table): Slot model-resolution precedence over a
real hub-double gRPC boundary — real Worker/Lifecycle/Executor, real blake3
blob host, no mocking of gen_worker internals (design principle #1: the hub
scheduler is the only legal double).

Pins two absorbed contracts:

  * pgw#606/th#938 (PR #353): boot NEVER sets up a Slot fn from the
    image-baked code default; the hub-stamped binding (delivered via Hot
    DesiredInstance) is the ONLY setup source once hub-connected, and this
    holds even when every slot default is tensorhub-sourced (not just the
    mixed Civitai+Hub live-incident shape — see the harness endpoint
    docstring for why a Hub-only fixture is used here instead of a real
    Civitai() ref: it keeps this suite hermetic with zero real network
    reachability, which the original ie#518/th#938 repro didn't need to
    prove the precedence rule itself).
  * pgw#583 (PR #307): a FIXED slot dispatched a DIFFERENT repo than
    declared refuses, naming the slot and both refs; a hub-resolved
    same-repo flavor pick still serves; a selected_by= catalog slot's
    different-repo pick is a legitimate surface, not a mismatch; an
    undeclared model slot in the dispatch map warns and does not block.

Revert-verify (per th#960 task instructions): reverting lifecycle.py's
pgw#606/th#938 fix (commit 38887c0) turns
``test_slot_boot_precedence_outranks_code_default`` red — see the th#960
tracker checkpoint for the exact repro transcript.
"""

from __future__ import annotations

import logging
import time

import msgspec
import pytest

from gen_worker.api.binding import wire_ref
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_accept_for, is_ready, is_result_for
from harness.toy_endpoints import (
    BOOT_UNREACHABLE_PIPELINE,
    BOOT_UNREACHABLE_VAE,
    CATALOG_DEFAULT_PIPELINE,
    DECLARED_PIPELINE,
    EchoIn,
    EchoOut,
)

_TIMEOUT = 15.0


def _payload(**kw: object) -> bytes:
    return msgspec.msgpack.encode(EchoIn(**kw))  # type: ignore[arg-type]


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


# ---------------------------------------------------------------------------
# pgw#606/th#938: boot precedence.
# ---------------------------------------------------------------------------


def test_slot_boot_never_advertises_ready_from_code_default() -> None:
    """Slot fns resolve JIT per-dispatch (never at boot), so — unlike a
    Hub()-sugar-bound fn — this one is available immediately with ZERO setup
    calls: neither slot default was ever independently fetched or set up
    (which would have failed loud, since no snapshot for either default is
    ever registered on this connection)."""
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        ready = conn.wait_for(is_ready)
        assert "slot-boot-precedence" in ready.state_delta.available_functions
        assert "slot-boot-precedence" not in ready.state_delta.loading_functions
        time.sleep(0.3)
        assert not any(
            m.WhichOneof("msg") == "fn_unavailable"
            and m.fn_unavailable.function_name == "slot-boot-precedence"
            for m in conn.received
        )


def test_slot_boot_never_spawns_the_th912_boot_setup_watcher() -> None:
    """pgw#606/th#938's precise pin, isolated: a Slot fn must never spawn
    the th#912 boot-setup watcher that materialized the image-baked code
    default over a hub-stamped binding. Direct state check on the REAL
    Lifecycle after a REAL boot (no mocking) — loaded in an endpoint module
    carrying ONLY this Slot fn (``_boot_setup_watch`` is one task shared by
    every awaiting function; a Hub()-sugar-bound sibling in the same module
    would legitimately spawn it too and make the assertion meaningless).

    REVERT-VERIFY (th#960 task instructions): reverting lifecycle.py's
    pgw#606/th#938 fix (commit 38887c0, `git apply -R` on that commit's
    lifecycle.py hunk) turns this red — the old watcher spawns immediately
    because both slot defaults are tensorhub-sourced and initially missing.
    Confirmed locally 2026-07-21 and restored; not committed as a revert.
    """
    with hub_double(modules=("harness.toy_endpoints_slot_only",)) as (_scheduler, harness):
        time.sleep(0.2)
        assert harness.worker.lifecycle._boot_setup_watch is None, (
            "Slot functions must not spawn a boot-setup watcher"
        )


def test_slot_boot_precedence_outranks_code_default(tmp_path) -> None:
    """The hub-stamped Hot DesiredInstance binding is the ONLY setup source:
    setup runs exactly once, serving the hub-delivered bytes — the code
    default (never delivered) could not have been used."""
    blobs = BlobHost(tmp_path)
    try:
        pipeline_payload = b"hub-stamped-pipeline-bytes"
        vae_payload = b"hub-stamped-vae-bytes"
        # A FIXED slot's identity gate (pgw#583) is keyed on repo identity,
        # not tag/flavor — the hub-stamped delivery names the SAME declared
        # repo (only the code default's TAG differs), exactly like the live
        # th#938 incident (code default `wai-illustrious` unqualified, hub
        # stamps a concrete `:prod` tag of that same repo).
        stamped_pipeline_ref = wire_ref(BOOT_UNREACHABLE_PIPELINE)
        stamped_vae_ref = wire_ref(BOOT_UNREACHABLE_VAE)
        pipeline_snap = blobs.one_file_snapshot("snap-pipeline", "pipeline", pipeline_payload)
        vae_snap = blobs.one_file_snapshot("snap-vae", "vae", vae_payload)

        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[stamped_pipeline_ref, stamped_vae_ref],
                    snapshots={stamped_pipeline_ref: pipeline_snap, stamped_vae_ref: vae_snap},
                    hot=[pb.DesiredInstance(
                        function_name="slot-boot-precedence",
                        models=[
                            pb.ModelBinding(slot="pipeline", ref=stamped_pipeline_ref),
                            pb.ModelBinding(slot="vae", ref=stamped_vae_ref),
                        ],
                    )],
                ),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "slot-boot-precedence" in m.state_delta.available_functions
                and m.state_delta.observed_residency_generation == 1
            )
            conn.send(run_job=pb.RunJob(
                request_id="r1", attempt=1, function_name="slot-boot-precedence",
                input_payload=_payload(),
                models=[
                    pb.ModelBinding(slot="pipeline", ref=stamped_pipeline_ref),
                    pb.ModelBinding(slot="vae", ref=stamped_vae_ref),
                ],
                snapshots={stamped_pipeline_ref: pipeline_snap, stamped_vae_ref: vae_snap},
            ))
            res = conn.wait_for(is_result_for("r1")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert _decode(res.inline).response == pipeline_payload.decode()
            # Companion boot-time proof lives in
            # test_slot_boot_never_advertises_ready_from_code_default: at
            # boot, before this HelloAck ever arrives, the fn is already
            # available with zero setup calls and zero fetch attempts —
            # setup only happened here, once, on THIS dispatch.
    finally:
        blobs.shutdown()


# ---------------------------------------------------------------------------
# pgw#583: model-slot identity gate.
# ---------------------------------------------------------------------------


def test_fixed_slot_wrong_repo_dispatch_refuses(tmp_path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        wrong_repo = "harness/some-other-repo:prod"
        snap = blobs.one_file_snapshot("snap-wrong", "wrong", b"irrelevant")
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-wrong", attempt=1, function_name="slot-identity-fixed",
                input_payload=_payload(),
                models=[pb.ModelBinding(slot="pipeline", ref=wrong_repo)],
                snapshots={wrong_repo: snap},
            ))
            res = conn.wait_for(is_result_for("r-wrong")).job_result
            assert res.status != pb.JOB_STATUS_OK
            assert "pipeline" in res.safe_message
            assert DECLARED_PIPELINE.path in res.safe_message
            assert "some-other-repo" in res.safe_message
    finally:
        blobs.shutdown()


def test_fixed_slot_same_repo_flavor_pick_serves(tmp_path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        same_repo_other_flavor = f"{DECLARED_PIPELINE.path}:prod#fp8"
        payload = b"fp8-flavor-bytes"
        snap = blobs.one_file_snapshot("snap-fp8", "fp8", payload)
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-flavor", attempt=1, function_name="slot-identity-fixed",
                input_payload=_payload(),
                models=[pb.ModelBinding(slot="pipeline", ref=same_repo_other_flavor)],
                snapshots={same_repo_other_flavor: snap},
            ))
            res = conn.wait_for(is_result_for("r-flavor")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            out = _decode(res.inline)
            assert out.response == f"tensorhub:{DECLARED_PIPELINE.path}:prod#fp8"
    finally:
        blobs.shutdown()


def test_catalog_slot_different_repo_pick_is_not_a_mismatch(tmp_path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        catalog_pick = "harness/slot-catalog-pick:prod"
        payload = b"catalog-pick-bytes"
        snap = blobs.one_file_snapshot("snap-catalog", "catalog", payload)
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-catalog", attempt=1, function_name="slot-identity-catalog",
                input_payload=_payload(model="catalog-pick"),
                models=[pb.ModelBinding(slot="pipeline", ref=catalog_pick)],
                snapshots={catalog_pick: snap},
            ))
            res = conn.wait_for(is_result_for("r-catalog")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            out = _decode(res.inline)
            assert out.response == "tensorhub:harness/slot-catalog-pick:prod#"
            assert CATALOG_DEFAULT_PIPELINE.path not in out.response
    finally:
        blobs.shutdown()


def test_undeclared_model_slot_warns_and_serves(tmp_path, caplog) -> None:
    blobs = BlobHost(tmp_path)
    try:
        declared_ref = f"{DECLARED_PIPELINE.path}:prod"
        lora_ref = "harness/some-lora:prod"
        snap = blobs.one_file_snapshot("snap-declared", "declared", b"declared-bytes")
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            with caplog.at_level(logging.WARNING, logger="gen_worker.executor"):
                conn.send(run_job=pb.RunJob(
                    request_id="r-undeclared", attempt=1, function_name="slot-identity-fixed",
                    input_payload=_payload(),
                    models=[
                        pb.ModelBinding(slot="pipeline", ref=declared_ref),
                        pb.ModelBinding(slot="lora", ref=lora_ref),
                    ],
                    snapshots={declared_ref: snap},
                ))
                res = conn.wait_for(is_result_for("r-undeclared")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            warnings = [
                r for r in caplog.records
                if r.levelno == logging.WARNING and "UNDECLARED_MODEL_SLOT" in r.getMessage()
            ]
            assert warnings, f"no undeclared-slot warning logged: {caplog.records}"
            assert any("lora" in r.getMessage() for r in warnings)
    finally:
        blobs.shutdown()
