"""pgw#654: the 0.58.0/0.60.0 fleet-wide false "model load failure".

Live shape (ie#544, hub a838b34a): a pod converges its v5 shadow command
(materialize SUCCEEDED, model on_disk), the first real jobs arrive, the
post-RunJob residency re-pass asks ensure_intent for the now-TERMINAL
command work, gets "" back, and reported_await("") arms guard_await's 2.0s
unreported-wait bomb while wait_idle() blocks on the in-flight job ->
UnreportedIntentWait -> WORKER_PHASE_ERROR -> the hub counts a startup
failure on a HEALTHY worker -> model_load_failure_streak -> release_broken.
3/3 releases (qwen-image, z-image, ernie) died exactly 2s after their first
dispatch. Toy gates never saw it because their jobs finish inside the grace.

The fix: ensure_intent mints a worker-local compat carrier for re-verified
command work instead of returning "" (every long await stays reportable),
and apply_command supersedes only COMMAND-BORN intents, never worker-local
job/setup/materialize obligations.
"""

from __future__ import annotations

import asyncio
import hashlib
import time

import msgspec

from gen_worker.intent_registry import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready
from harness.toy_endpoints import EchoIn

_REF_A = "harness/slow-pipeline"
_REF_B = "harness/second-ref"
_FN = "slow-slot-echo"
_RELEASE = "rel-pgw654"


def _shadow_command(
    session_id: str, desired: pb.DesiredResidency,
) -> pb.DesiredStateCommand:
    """Mirror of tensorhub buildShadowDesiredStateCommand (a838b34a):
    MATERIALIZE per disk ref + FUNCTION_READY per hot instance, all
    mandatory, command_seq = desired generation, deterministic intent ids."""
    now_ms = int(time.time() * 1000)
    cmd = pb.DesiredStateCommand(
        worker_session_id=session_id,
        command_seq=desired.generation,
        goal_id=f"goal-gen-{desired.generation}",
        release_id=_RELEASE,
        config_generation=desired.config_generation,
        issued_at_unix_ms=now_ms,
        accept_by_unix_ms=now_ms + 2_000,
        first_action_by_unix_ms=now_ms + 60_000,
        mandatory=True,
    )
    for i, ref in enumerate(desired.disk_refs):
        digest = desired.snapshots[ref].digest if ref in desired.snapshots else ""
        ident = hashlib.sha256(f"materialize|{ref}|{digest}".encode()).hexdigest()[:16]
        cmd.intents.append(pb.DesiredIntent(
            intent_id=f"intent-mat-{ident}",
            kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
            cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION,
            ref=ref,
            snapshot_digest=digest.encode(),
            desired_tier=pb.RESIDENCY_TIER_DISK,
            priority=len(desired.disk_refs) - i,
            mandatory=True,
        ))
    for i, instance in enumerate(desired.hot):
        raw = instance.SerializeToString(deterministic=True)
        bd = hashlib.sha256(raw)
        cmd.intents.append(pb.DesiredIntent(
            intent_id=f"intent-fn-{bd.hexdigest()[:16]}",
            kind=pb.DESIRED_INTENT_KIND_FUNCTION_READY,
            cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
            function_name=instance.function_name,
            desired_tier=pb.RESIDENCY_TIER_VRAM,
            binding_digest=bd.digest(),
            priority=len(desired.hot) - i,
            mandatory=True,
        ))
    return cmd


def test_first_job_on_converged_v5_worker_does_not_error(tmp_path) -> None:
    """The live killer: converged command + in-flight >2s job + re-pass."""
    blobs = BlobHost(tmp_path)
    try:
        snap_a = blobs.one_file_snapshot(
            "snap-a", "blob-a", b"weights-a",
            path_in_snapshot="transformer/weights.txt",
        )
        snap_b = blobs.one_file_snapshot(
            "snap-b", "blob-b", b"weights-b",
            path_in_snapshot="transformer/weights.txt",
        )
        with hub_double(
            modules=("harness.toy_endpoints", "harness.slow_endpoints_pgw654"),
        ) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            session_id = conn.hello.worker_session_id

            hot = [pb.DesiredInstance(
                function_name=_FN,
                models=[pb.ModelBinding(slot="pipeline", ref=_REF_A)],
            )]
            desired1 = pb.DesiredResidency(
                generation=1, release_id=_RELEASE,
                disk_refs=[_REF_A], hot=hot,
                snapshots={_REF_A: snap_a},
            )
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=desired1,
                desired_state_command=_shadow_command(session_id, desired1),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and _FN in m.state_delta.available_functions,
                timeout=30.0,
            )

            # FULL v5 convergence: the command's materialize intent must be
            # terminal SUCCEEDED — the live pods' state at first dispatch.
            def _mat_succeeded(m: pb.WorkerMessage) -> bool:
                if m.WhichOneof("msg") != "lifecycle_snapshot":
                    return False
                return any(
                    i.intent_id.startswith("intent-mat-") and i.status == 4
                    for i in m.lifecycle_snapshot.intents
                )

            conn.wait_for(_mat_succeeded, timeout=30.0)

            # First tenant job, runs well past the 2.0s grace.
            conn.send(run_job=pb.RunJob(
                request_id="r-long", attempt=1, function_name="slow-stream",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "job_accepted"
                and m.job_accepted.request_id == "r-long"
            )

            # Generation bump mid-job (live: planner adds the sibling ref).
            desired2 = pb.DesiredResidency(
                generation=2, release_id=_RELEASE,
                disk_refs=[_REF_A, _REF_B], hot=hot,
                snapshots={_REF_A: snap_a, _REF_B: snap_b},
            )
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=desired2,
                desired_state_command=_shadow_command(session_id, desired2),
            ))

            # The job must complete and the worker must NEVER report ERROR.
            result = conn.wait_for(
                lambda m: m.WhichOneof("msg") == "job_result"
                and m.job_result.request_id == "r-long",
                timeout=30.0,
            ).job_result
            assert result.status == pb.JOB_STATUS_OK

            errors = conn.count(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and m.state_delta.phase == pb.WORKER_PHASE_ERROR
            )
            assert errors == 0, (
                "worker reported WORKER_PHASE_ERROR — the pgw#654 false "
                "model_load_failure regression is back"
            )

            # And the bumped desired state still converges after the job.
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "model_event"
                and m.model_event.ref == _REF_B
                and m.model_event.state == pb.MODEL_STATE_ON_DISK,
                timeout=30.0,
            )
    finally:
        blobs.shutdown()


def test_ensure_intent_mints_carrier_for_terminal_command_work() -> None:
    """ensure_intent never returns "" for re-verified command work."""
    reg = IntentRegistry("rel-x", ["fn-a"])
    cmd = pb.DesiredStateCommand(
        worker_session_id=reg.worker_session_id, command_seq=1,
        goal_id="goal-1", release_id="rel-x", mandatory=True,
        issued_at_unix_ms=1, accept_by_unix_ms=2, first_action_by_unix_ms=3,
        intents=[pb.DesiredIntent(
            intent_id="mat-a", kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
            cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION, ref="refA",
            snapshot_digest=b"d", desired_tier=pb.RESIDENCY_TIER_DISK,
            mandatory=True,
        )],
    )
    receipt = reg.apply_command(cmd)
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    # Command-owned while active.
    assert reg.ensure_intent(
        pb.DESIRED_INTENT_KIND_MATERIALIZE, ref="refA") == "mat-a"

    reg.transition("mat-a", pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                   pb.LIFECYCLE_INTENT_STAGE_FETCHING)
    reg.transition("mat-a", pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                   pb.LIFECYCLE_INTENT_STAGE_ON_DISK)

    # Terminal command work: a re-pass gets a live compat carrier, not "".
    carrier = reg.ensure_intent(pb.DESIRED_INTENT_KIND_MATERIALIZE, ref="refA")
    assert carrier and carrier != "mat-a"
    assert reg.is_active(carrier)
    # The carrier is transitionable (reported_await can report on it).
    reg.transition(carrier, pb.LIFECYCLE_INTENT_STATUS_WAITING,
                   pb.LIFECYCLE_INTENT_STAGE_WAIT_TENANT_IDLE,
                   reason=pb.LIFECYCLE_WAIT_REASON_TENANT_WORK)


def test_generation_bump_does_not_supersede_worker_local_intents() -> None:
    """A command replaces command-owned work only; a live job/setup carrier
    must survive the bump."""
    reg = IntentRegistry("rel-x", ["fn-a"])

    def cmd(seq: int, refs: list) -> pb.DesiredStateCommand:
        c = pb.DesiredStateCommand(
            worker_session_id=reg.worker_session_id, command_seq=seq,
            goal_id=f"goal-{seq}", release_id="rel-x", mandatory=True,
            issued_at_unix_ms=1, accept_by_unix_ms=2, first_action_by_unix_ms=3,
        )
        for ref in refs:
            c.intents.append(pb.DesiredIntent(
                intent_id=f"mat-{ref}", kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
                cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION, ref=ref,
                snapshot_digest=b"d", desired_tier=pb.RESIDENCY_TIER_DISK,
                mandatory=True,
            ))
        return c

    assert reg.apply_command(cmd(1, ["refA"])).status == pb.GOAL_RECEIPT_STATUS_ACCEPTED
    job_intent = reg.ensure_local_intent("job", "r1\x001", function_name="fn-a")
    reg.transition(job_intent, pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                   pb.LIFECYCLE_INTENT_STAGE_READY, detail="executing")

    assert reg.apply_command(cmd(2, ["refB"])).status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    # Worker-local job carrier survived; the dropped command intent did not.
    assert reg.is_active(job_intent), "generation bump killed a live job intent"
    assert not reg.is_active("mat-refA"), "stale command intent not superseded"


def test_enter_error_phase_dials_cause(monkeypatch) -> None:
    """Every WORKER_PHASE_ERROR ships its cause on the gw#640 carrier."""
    from gen_worker import worker_fatal
    from gen_worker.lifecycle import Lifecycle

    dialed: list = []

    async def _fake_report(settings, detail):
        dialed.append(detail)
        return True

    monkeypatch.setattr(worker_fatal, "report_worker_error_async", _fake_report)

    lc = object.__new__(Lifecycle)  # stubbed Lifecycle (repo convention)
    lc.phase = pb.WORKER_PHASE_READY
    lc._settings = object()

    async def _run() -> None:
        lc._enter_error_phase("residency reconcile",
                              RuntimeError("unreported protocol await: x"))
        await asyncio.sleep(0)  # let the dial task run

    asyncio.run(_run())
    assert lc.phase == pb.WORKER_PHASE_ERROR
    assert len(dialed) == 1
    assert "unreported protocol await: x" in dialed[0]
    assert "residency reconcile" in dialed[0]
