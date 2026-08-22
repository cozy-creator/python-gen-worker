"""#1660 / th#2300: the worker is a COMPLETE protocol-v5 producer, or it is legacy.

Every arm here drives a REAL ``gen_worker.worker.Worker`` over a REAL gRPC socket
against the hub double, and reads the bytes the hub would read.

THE FENCE these arms exist for: the hub's typed cohort flag is not health-only.
`releaseUsesExactCapabilities` -> `placement.ExactCapabilityRequired` ->
`workerHasExactReadyCapabilityLocked`. A worker that ships
``(worker_session_id, lifecycle_snapshot)`` WITHOUT a capability set and without
answering the hub's `DesiredStateCommand` turns an inert typed path into a TOTAL
DISPATCH STARVE on its release. So `test_the_pair_never_ships_without_the_whole_producer`
asserts the implication in the only direction that can hurt anyone: *if* the pair
is on the wire, the capabilities and the receipt answering are wired too — and
the arms below it force each leg's absence and prove the pair is withheld.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import pytest

from gen_worker.lifecycle import WorkerLifecycle, snapshot_refusal
from gen_worker.lifecycle_intents import CapabilityFacts, IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double

RELEASE = "rel-pgw1660"
FUNCTIONS = ("echo", "save-a-large-file")
REF = "cozy/toy@1"


def _command(hello: pb.Hello, *, config_generation: int = 7) -> pb.DesiredStateCommand:
    """The command the real hub builds at HelloAck (`buildProtocolDesiredStateCommand`)."""
    now = int(time.time() * 1000)
    return pb.DesiredStateCommand(
        worker_session_id=hello.worker_session_id,
        command_seq=3,
        goal_id="goal-pgw1660",
        release_id=RELEASE,
        config_generation=config_generation,
        config_digest=b"config-digest",
        parameter_snapshot=b"\x80",  # msgpack {}
        issued_at_unix_ms=now,
        accept_by_unix_ms=now + 2_000,
        first_action_by_unix_ms=now + 600_000,
        intents=[
            pb.DesiredIntent(
                intent_id="intent-pgw1660-echo",
                kind=pb.DESIRED_INTENT_KIND_FUNCTION_READY,
                cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
                function_name="echo",
                binding_digest=b"the-hubs-own-binding-digest",
            ),
        ],
    )


def _ack(hello: pb.Hello) -> pb.HelloAck:
    """A HelloAck shaped like the hub's: residency, resolutions, and the command."""
    return pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        file_base_url="http://127.0.0.1:1/files",
        desired_residency=pb.DesiredResidency(
            generation=3,
            release_id=RELEASE,
            config_generation=7,
            hot=[
                pb.DesiredInstance(
                    function_name="echo",
                    models=[pb.ModelBinding(slot="data", ref=REF)],
                )
            ],
        ),
        resolutions=[
            pb.ModelResolution(ref=REF, resolved_ref=REF, lane="fp8-w8a8-dynamic+compiled")
        ],
        desired_state_command=_command(hello),
    )


def _legacy_ack(hello: pb.Hello) -> pb.HelloAck:
    return pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        file_base_url="http://127.0.0.1:1/files",
    )


def _is(kind: str) -> Callable[[pb.WorkerMessage], bool]:
    return lambda msg: msg.WhichOneof("msg") == kind


# ---------------------------------------------------------------------------
# The wire arms
# ---------------------------------------------------------------------------


def test_a_v5_hello_carries_the_pair_and_the_projection_the_hub_reads() -> None:
    """The defect's exact inverse: a Hello the hub reads as `typed`, not as
    `hello_session_id_missing_snapshot`."""
    with hub_double(release_id=RELEASE, hello_ack=_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        hello = conn.hello
        assert hello is not None

        assert hello.worker_session_id, (
            "the Hello states no worker_session_id — this worker cannot be typed "
            "at all, and the whole release falls to the legacy verdicts"
        )
        assert hello.HasField("lifecycle_snapshot"), (
            "worker_session_id with NO lifecycle_snapshot is exactly the fleet "
            "defect th#2300 measured: the hub books "
            "worker_protocol_rejected / hello_session_id_missing_snapshot and "
            "drops the typed projection"
        )
        snapshot = hello.lifecycle_snapshot
        assert snapshot.worker_session_id == hello.worker_session_id
        assert snapshot.full_replace, "the hub only ever full-replaces"
        assert snapshot.state_seq >= 1, "state_seq 0 is 'missing state_seq'"
        assert snapshot_refusal(snapshot, RELEASE) == ""

        capabilities = {c.function_name: c for c in snapshot.capabilities}
        assert set(capabilities) == set(FUNCTIONS), (
            "a typed worker with no capability for a function it serves is a "
            "dispatch starve on its whole release, not a health nuance"
        )
        for capability in capabilities.values():
            assert capability.release_id == RELEASE
            assert capability.state != pb.FUNCTION_CAPABILITY_STATE_UNSPECIFIED


def test_the_hub_command_is_answered_with_an_accepted_goal_receipt() -> None:
    with hub_double(release_id=RELEASE, hello_ack=_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        hello = conn.hello
        assert hello is not None
        command = _command(hello)

        message = conn.wait_for(_is("goal_receipt"), timeout=30.0)
        receipt = message.goal_receipt
        assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED, receipt.detail
        # Every field `workerHasExactReadyCapabilityLocked` cross-checks against
        # `info.LastDesiredStateCommand` before it will dispatch anything.
        assert receipt.worker_session_id == hello.worker_session_id
        assert receipt.command_seq == command.command_seq
        assert receipt.goal_id == command.goal_id
        assert receipt.release_id == command.release_id
        assert receipt.received_at_unix_ms > 0


def test_the_mid_stream_snapshot_advances_and_keeps_carrying_the_receipt() -> None:
    """A snapshot with no `goal_receipts` ERASES `info.LastGoalReceipt` hub-side
    and un-dispatches the worker, so every snapshot must keep carrying it."""
    with hub_double(release_id=RELEASE, hello_ack=_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        hello = conn.hello
        assert hello is not None

        message = conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "lifecycle_snapshot"
                and len(m.lifecycle_snapshot.goal_receipts) > 0
            ),
            timeout=30.0,
        )
        snapshot = message.lifecycle_snapshot
        assert snapshot.state_seq > hello.lifecycle_snapshot.state_seq, (
            "the hub SILENTLY DROPS a snapshot whose state_seq did not advance"
        )
        assert snapshot.worker_session_id == hello.worker_session_id
        assert snapshot.full_replace
        assert snapshot_refusal(snapshot, RELEASE) == ""
        accepted = [
            r for r in snapshot.goal_receipts
            if r.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED
        ]
        assert accepted, "the accepted receipt must survive every full replace"


def test_the_capability_echoes_the_hubs_own_binding_digest_and_lane() -> None:
    """`exact_binding_digest_mismatch` and `exact_lane_mismatch` are SILENT
    dispatch declines, so both come from the hub's own statements, not from a
    re-derivation this worker could get subtly wrong."""
    with hub_double(release_id=RELEASE, hello_ack=_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        message = conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "lifecycle_snapshot"
                and any(
                    c.function_name == "echo" and c.binding_digest
                    for c in m.lifecycle_snapshot.capabilities
                )
            ),
            timeout=30.0,
        )
        echo = next(
            c for c in message.lifecycle_snapshot.capabilities
            if c.function_name == "echo"
        )
        assert echo.binding_digest == b"the-hubs-own-binding-digest", (
            "the FUNCTION_READY intent the hub authored carries the digest the "
            "hub compares with bytes.Equal — echo it, never re-derive it"
        )
        assert echo.lane == "fp8-w8a8-dynamic+compiled", (
            "the lane is the HUB's resolution for this function's models"
        )


def test_the_capability_reaches_READY_at_the_commands_config_generation() -> None:
    """The dispatch-critical end state. `workerHasExactReadyCapabilityLocked`
    needs state=READY AND `config_generation == command.config_generation`; a
    config application that never converges pins every capability at APPLYING,
    which is a typed release nothing can be dispatched to."""
    with hub_double(release_id=RELEASE, hello_ack=_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        message = conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "lifecycle_snapshot"
                and any(
                    c.state == pb.FUNCTION_CAPABILITY_STATE_READY
                    for c in m.lifecycle_snapshot.capabilities
                )
            ),
            timeout=30.0,
        )
        snapshot = message.lifecycle_snapshot
        assert snapshot.config_application.state == pb.CONFIG_APPLICATION_STATE_CONVERGED, (
            f"config application stuck at {snapshot.config_application}"
        )
        ready = [
            c for c in snapshot.capabilities
            if c.state == pb.FUNCTION_CAPABILITY_STATE_READY
        ]
        assert ready
        for capability in ready:
            assert capability.config_generation == 7
            assert capability.binding_digest, (
                "a READY capability under a CONVERGED application MUST carry a "
                "binding_digest or the hub discards the whole snapshot"
            )


def test_a_release_less_worker_stays_legacy_and_ships_NEITHER_half() -> None:
    """No release id means no capability can name one, so the pair is withheld
    WHOLE. Session id and snapshot both absent is the one combination the hub
    books no `worker_protocol_rejected` row for."""
    with hub_double(hello_ack=_legacy_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        hello = conn.hello
        assert hello is not None
        assert hello.worker_session_id == ""
        assert not hello.HasField("lifecycle_snapshot")


def test_the_pair_never_ships_without_the_whole_producer() -> None:
    """THE FENCE, stated as an implication over the real wire.

    Red-armed by the three arms below, each of which removes one leg and proves
    the pair goes away with it.
    """
    with hub_double(release_id=RELEASE, hello_ack=_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        hello = conn.hello
        assert hello is not None
        if not hello.HasField("lifecycle_snapshot"):
            assert hello.worker_session_id == "", (
                "half a pair on the wire — the hub drops the projection and the "
                "release is judged legacy"
            )
            return
        # The pair IS on the wire, so both other legs must be provably live.
        assert hello.worker_session_id
        assert hello.lifecycle_snapshot.capabilities, (
            "typed with no capabilities: dispatch requires an exact READY "
            "capability row, so this release can never be dispatched to again"
        )
        receipt = conn.wait_for(_is("goal_receipt"), timeout=30.0).goal_receipt
        assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED, (
            "typed with no ACCEPTED receipt for the hub's current command: "
            "`exact_goal_receipt_not_accepted` declines every placement"
        )


# ---------------------------------------------------------------------------
# The red arms — each removes one leg of the producer
# ---------------------------------------------------------------------------


def _no_projection(_facts: CapabilityFacts) -> None:
    return None


def _lifecycle(
    *,
    route: bool = True,
    facts: Optional[CapabilityFacts] = None,
    release_id: str = RELEASE,
    project: bool = True,
) -> WorkerLifecycle:
    async def _send(_msg: pb.WorkerMessage) -> None:
        return None

    lifecycle = WorkerLifecycle(
        release_id=release_id,
        function_names=list(FUNCTIONS),
        facts=lambda: facts or CapabilityFacts(available=frozenset(FUNCTIONS)),
        send=_send,
    )
    if not project:
        lifecycle.registry.refresh_projection = _no_projection  # type: ignore[method-assign,assignment]
    if route:

        async def _body(_ack: pb.HelloAck) -> None:
            return None

        lifecycle.hello_ack_route(_body)
    return lifecycle


def test_RED_ARM_remove_the_capability_projection_and_the_pair_is_withheld() -> None:
    session_id, snapshot = _lifecycle(project=False).hello_projection()
    assert (session_id, snapshot) == ("", None), (
        "a snapshot with no capabilities must NEVER reach the hub: it flips the "
        "release to ExactCapabilityRequired with nothing to match"
    )


def test_RED_ARM_remove_the_goal_receipt_route_and_the_pair_is_withheld() -> None:
    session_id, snapshot = _lifecycle(route=False).hello_projection()
    assert (session_id, snapshot) == ("", None), (
        "without the HelloAck -> GoalReceipt answer the hub never holds an "
        "ACCEPTED receipt for its own command, and every dispatch declines"
    )


def test_RED_ARM_remove_the_release_id_and_the_pair_is_withheld() -> None:
    session_id, snapshot = _lifecycle(release_id="").hello_projection()
    assert (session_id, snapshot) == ("", None)


def test_the_complete_producer_DOES_answer_with_the_pair() -> None:
    """The control for the three red arms: with every leg wired, the pair ships.

    Without this, all three arms above would still pass on a producer that never
    ships anything at all.
    """
    session_id, snapshot = _lifecycle().hello_projection()
    assert session_id
    assert snapshot is not None
    assert {c.function_name for c in snapshot.capabilities} == set(FUNCTIONS)


# ---------------------------------------------------------------------------
# Commands the hub really sends
# ---------------------------------------------------------------------------


def _boot_ack(hello: pb.Hello) -> pb.HelloAck:
    """The FIRST HelloAck of a boot: a config generation the worker's persisted
    DesiredResidency already carries, and NO config digest, because the hub built
    the command with a nil `resolutionCfg`."""
    ack = _ack(hello)
    ack.desired_state_command.ClearField("config_digest")
    return ack


def test_a_command_with_a_generation_but_no_config_digest_is_ACCEPTED() -> None:
    """This is not a nicety — it is the shape of the first HelloAck of every
    boot. `protocolConfigDigest(resolutionCfg)` is empty whenever the hub has no
    resolution config yet (build error, discarded stale build, no provider),
    while `config_generation` comes from the persisted DesiredResidency and is
    > 0. Rejecting the pair costs the ACCEPTED receipt, and no accepted receipt
    for the CURRENT command is `exact_goal_receipt_not_accepted` — a dispatch
    starve on the whole release, on every cold boot."""
    with hub_double(release_id=RELEASE, hello_ack=_boot_ack) as (scheduler, _harness):
        conn = scheduler.wait_connection(0, timeout=30.0)
        receipt = conn.wait_for(_is("goal_receipt"), timeout=30.0).goal_receipt
        assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED, (
            f"a fresh-boot command was refused: {receipt.error_code} {receipt.detail}"
        )


def test_the_same_goal_restated_with_a_filled_in_digest_is_ACCEPTED() -> None:
    """The reconnect after a cold boot, and the third starve of the same family.

    Boot 1 gets `config_generation=N` with an EMPTY `config_digest` (the hub had
    no resolution config yet). The stream drops; the reconnect gets the SAME
    generation — so the same `command_seq`, since the hub uses the residency
    generation for it — now WITH a digest. Keyed on the command digest that reads
    as `COMMAND_SEQ_CONFLICT` and costs the accepted receipt forever after. The
    hub keys same-seq identity on `goal_id`/`release_id` itself, so this worker
    does too, and a genuine goal drift is still refused
    (`test_intent_registry_th1283`).
    """
    registry = IntentRegistry(RELEASE, list(FUNCTIONS))
    boot = pb.DesiredStateCommand(
        worker_session_id=registry.worker_session_id,
        command_seq=9,
        goal_id="goal-stable",
        release_id=RELEASE,
        config_generation=4,
        parameter_snapshot=b"\x80",
    )
    first = registry.apply_command(boot)
    assert first.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    reconnect = pb.DesiredStateCommand()
    reconnect.CopyFrom(boot)
    reconnect.config_digest = b"now-the-hub-has-one"
    receipt = registry.apply_command(reconnect)
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED, (
        f"the same goal restated was refused: {receipt.error_code} {receipt.detail}"
    )
    assert not registry.protocol_rejected
    assert receipt.SerializeToString(deterministic=True) == first.SerializeToString(
        deterministic=True
    ), (
        "and it must be the receipt the hub ALREADY HOLDS, byte for byte. The "
        "hub refuses a second receipt at the same command_seq that is not "
        "proto.Equal, and a fresh received_at_unix_ms alone differs — that is a "
        "`worker_protocol_rejected` row on a fleet whose bar is zero of them"
    )
    assert registry.snapshot().config_application.target_generation == 4, (
        "the command still APPLIED in full; only the answer was stabilised"
    )


def test_a_REBOUND_function_still_echoes_the_hubs_current_digest() -> None:
    """The echo is a fact about the COMMAND, not about intent bookkeeping.

    Re-issuing the same intent id with a different binding is "intent_id reused
    for different work" — an advisory decline that drops the intent out of the
    accepted set. Read the digest off the registry's intents and a re-bound
    function falls back to the derivation permanently, which is exactly the
    silent `exact_binding_digest_mismatch` the echo exists to prevent.
    """
    registry = IntentRegistry(RELEASE, list(FUNCTIONS))

    def _command(seq: int, goal: str, digest: bytes) -> pb.DesiredStateCommand:
        return pb.DesiredStateCommand(
            worker_session_id=registry.worker_session_id,
            command_seq=seq, goal_id=goal, release_id=RELEASE,
            intents=[pb.DesiredIntent(
                intent_id="intent-echo",
                kind=pb.DESIRED_INTENT_KIND_FUNCTION_READY,
                cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
                function_name="echo",
                binding_digest=digest,
            )],
        )

    registry.apply_command(_command(1, "goal-1", b"digest-one"))
    registry.apply_command(_command(2, "goal-2", b"digest-two"))
    registry.refresh_projection(CapabilityFacts(available=frozenset(FUNCTIONS)))
    echo = next(
        c for c in registry.snapshot().capabilities if c.function_name == "echo"
    )
    assert echo.binding_digest == b"digest-two", (
        "the capability must carry the digest the hub's CURRENT command states"
    )


def test_the_two_digests_keep_their_OPPOSITE_encodings() -> None:
    """`DesiredIntent.snapshot_digest` is the UTF-8 bytes of the digest STRING;
    `FunctionCapability.binding_digest` is raw sha256 bytes. The asymmetry is the
    hub's, and a worker that "helpfully" normalises either one breaks a
    `bytes.Equal` nobody sees fail."""
    registry = IntentRegistry(RELEASE, list(FUNCTIONS))
    digest = b"\x00\x01\x02not-utf8-at-all\xff"
    registry.apply_command(pb.DesiredStateCommand(
        worker_session_id=registry.worker_session_id,
        command_seq=1,
        goal_id="g",
        release_id=RELEASE,
        intents=[pb.DesiredIntent(
            intent_id="i-ready",
            kind=pb.DESIRED_INTENT_KIND_FUNCTION_READY,
            cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
            function_name="echo",
            binding_digest=digest,
        )],
    ))
    registry.refresh_projection(CapabilityFacts(
        available=frozenset(FUNCTIONS),
        residency={REF: pb.ModelResidency(ref=REF, snapshot_digest="blake3:abc")},
        hot={"echo": pb.DesiredInstance(
            function_name="echo", models=[pb.ModelBinding(slot="data", ref=REF)],
        )},
    ))
    echo = next(
        c for c in registry.snapshot().capabilities if c.function_name == "echo"
    )
    assert echo.binding_digest == digest, "raw bytes, echoed verbatim"
    assert echo.models[0].snapshot_digest == b"blake3:abc", (
        "the UTF-8 bytes of the digest string — never hex-decoded"
    )


# ---------------------------------------------------------------------------
# The hub's own discard rules, mirrored (tensorhub validateLifecycleSnapshot)
# ---------------------------------------------------------------------------


def _typed_snapshot() -> pb.LifecycleSnapshot:
    registry = IntentRegistry(RELEASE, list(FUNCTIONS))
    registry.refresh_projection(CapabilityFacts(available=frozenset(FUNCTIONS)))
    return registry.snapshot()


def test_the_registrys_own_snapshot_satisfies_every_hub_mandatory_rule() -> None:
    assert snapshot_refusal(_typed_snapshot(), RELEASE) == ""


@pytest.mark.parametrize(
    "break_it,expected",
    [
        (lambda s: setattr(s, "full_replace", False), "full_replace"),
        (lambda s: setattr(s, "state_seq", 0), "state_seq"),
        (lambda s: setattr(s, "worker_session_id", ""), "worker_session_id"),
    ],
)
def test_the_mirror_actually_fires(
    break_it: Callable[[pb.LifecycleSnapshot], None], expected: str,
) -> None:
    """The validator must be able to go RED, or it proves nothing about the
    snapshots it lets through."""
    snapshot = _typed_snapshot()
    break_it(snapshot)
    assert expected in snapshot_refusal(snapshot, RELEASE)


def test_a_capability_naming_another_release_is_refused() -> None:
    snapshot = _typed_snapshot()
    snapshot.capabilities[0].release_id = "some-other-release"
    assert "release_mismatch" in snapshot_refusal(snapshot, RELEASE)
