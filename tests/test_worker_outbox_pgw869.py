"""pgw#869: the worker outbox — losing the hub must not lose the evidence.

THE INCIDENT. This box rebooted mid-mint. The pod did NOT restart: the provider
reported it RUNNING throughout and the worker process was alive the whole time,
and another session had the hub back 60-90 s later. Every measurement the worker
had queued for the hub was destroyed anyway, because `SendQueue` had a durable
lane for RESULTS and none for FACTS:

* `reset_for_reconnect()` did `self._items.clear()` and restored only
  `_pending_results` — so every queued ActivityUpdate/BootPhase died on each
  disconnect;
* `_send_loop` POPS from the queue before `stream.write()`, and only results had
  a second copy (`_pending_results`) — so a fact the sender had taken when the
  stream died was gone with no queue involved at all.

These tests fail against that code and pass against the evidence lane.

WHAT "EXACTLY ONCE" MEANS HERE, and why it is a hub row rather than a wire count.
Replay after reconnect means the hub WILL sometimes see a repeat (a fact taken by
the sender, written into a socket that was already dead, then replayed). The
worker cannot know which. The hub already answers this: its upsert is

    ON CONFLICT (worker_id, kind, phase, state, self_stalled, payload_digest)
    DO UPDATE SET occurrences = occurrences + 1, ...

(tensorhub `internal/db/gen/worker_activity_events.sql.go`) over

    payload_digest = sha256(error || 0x00 || detail)

(tensorhub `internal/orchestrator/repository/worker_activity_event_store.go`,
`WorkerActivityPayloadDigest` — "the coalescing key over the VERBATIM payload").
So N identical deliveries are ONE row with `occurrences = N`. `_hub_row_key`
below is that key, verbatim; the tests assert against it rather than against a
wire count, and that is the property the acceptance actually wants.

Scope, per the issue: in-memory only. The worker process did not restart in the
motivating incident and a mint is a child of the worker, so a worker restart ends
the mint anyway. No disk persistence is built or tested here.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from typing import List, Tuple

from gen_worker import activity as activity_mod
from gen_worker import worker_credential
from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.transport import (
    _AUTH_FAILURE_EXIT_WINDOW_S,
    SendQueue,
    Transport,
)

from harness.hub_double import hub_double, is_ready

_TIMEOUT = 20.0
_RUNNING = pb.ActivityState.ACTIVITY_STATE_RUNNING
_COMPLETED = pb.ActivityState.ACTIVITY_STATE_COMPLETED


# --------------------------------------------------------------------------
# the hub's own identity, replicated verbatim so a claim about dedup is a
# claim about the row the hub actually writes
# --------------------------------------------------------------------------

def _payload_digest(error: str, detail: str) -> str:
    h = hashlib.sha256()
    h.update(error.encode())
    h.update(b"\x00")
    h.update(detail.encode())
    return h.hexdigest()


def _hub_row_key(msg: pb.WorkerMessage, worker_id: str = "w") -> Tuple:
    u = msg.activity_update
    return (
        worker_id, u.kind, u.phase, int(u.state), bool(u.self_stalled),
        _payload_digest(u.error, u.detail),
    )


def _fact(kind: str, phase: str, detail: str) -> pb.WorkerMessage:
    """A measurement: RUNNING, carrying a payload. Never coalescible."""
    return pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind=kind, phase=phase, state=_RUNNING, detail=detail))


def _beat(kind: str, phase: str, seq: int) -> pb.WorkerMessage:
    """A liveness re-report: RUNNING, no payload. Coalescible by contract."""
    return pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind=kind, phase=phase, state=_RUNNING, seq=seq))


def _activities(conn, kind: str) -> List[pb.ActivityUpdate]:
    return [
        m.activity_update for m in list(conn.received)
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == kind
    ]


def _is_activity(kind: str, phase: str = ""):
    def pred(m: pb.WorkerMessage) -> bool:
        if m.WhichOneof("msg") != "activity_update":
            return False
        u = m.activity_update
        return u.kind == kind and (not phase or u.phase == phase)
    return pred


def _await_live_reporting(harness) -> None:
    """Gate on the worker having bound its activity sink.

    `activity.bind_sink` runs inside `Executor.ensure_setup`, so a fact emitted
    before that lands on the logger instead of the stream — the test would then
    be measuring the harness, not the outbox. Gating on the worker's own first
    `warmup` ActivityUpdate was tried and is NOT a sound gate: whether a warmup
    activity is emitted at all depends on which endpoints reach setup, and runs
    were observed (`boot_phase` x18, zero `activity_update`) where it never
    was. The sink itself is the fact these tests depend on, so gate on it, and
    bound the wait by worker liveness rather than by a clock.
    """
    while activity_mod._sink is None:
        assert harness.alive, "worker exited before binding its activity sink"
        time.sleep(0.05)


# --------------------------------------------------------------------------
# THE ACCEPTANCE: kill the hub mid-activity, bring it back, assert every fact
# arrives, in order, and lands as exactly one hub row
# --------------------------------------------------------------------------

def test_hub_killed_midactivity_replays_every_fact_exactly_once() -> None:
    kind = "self_mint_compile_pgw869"
    with hub_double(worker_id="outbox-acceptance") as (scheduler, harness):
        conn0 = scheduler.wait_connection(0)
        conn0.wait_for(is_ready, _TIMEOUT)
        _await_live_reporting(harness)

        # A fact produced while the hub is up, and confirmed shipped.
        activity_mod.emit_event(kind, "entries=36 ceiling=8", phase="pool")
        conn0.wait_for(_is_activity(kind, "pool"), _TIMEOUT)

        # The hub dies. The worker process does not — exactly the incident.
        conn0.kill()

        # The "mint" keeps producing measurements into a dead connection.
        during = [
            ("trace_graph", "26 evidence @ 0.99/s"),
            ("autotune", "per_entry_device_bytes=11881591040 basis=measured"),
            ("publish", "cell ck5-aeed10f6 published"),
        ]
        for phase, detail in during:
            activity_mod.emit_event(kind, detail, phase=phase)

        # The hub comes back and the worker redials on its own.
        conn1 = scheduler.wait_connection(1, timeout=_TIMEOUT)
        for phase, _detail in during:
            conn1.wait_for(_is_activity(kind, phase), _TIMEOUT)

        replayed = [u for u in _activities(conn1, kind)]

        # (a) NOTHING WAS LOST. Every fact produced during the outage arrived.
        assert {u.phase for u in replayed} >= {p for p, _d in during}
        for phase, detail in during:
            assert any(
                u.phase == phase and u.detail == detail for u in replayed
            ), f"{phase!r} arrived without its payload: {replayed}"

        # (b) IN ORDER. Replay is FIFO in production order, not reshuffled.
        order = [u.phase for u in replayed if u.phase in {p for p, _ in during}]
        first_seen = [p for i, p in enumerate(order) if p not in order[:i]]
        assert first_seen == [p for p, _d in during]

        # (c) EXACTLY ONCE, as the hub counts it: one row per fact.
        all_msgs = [
            m for c in (conn0, conn1) for m in list(c.received)
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == kind
        ]
        rows = {_hub_row_key(m) for m in all_msgs}
        assert len(rows) == 1 + len(during), (
            "each fact must collapse to one hub row; got "
            f"{sorted(r[2] for r in rows)}"
        )


def test_terminal_produced_during_the_outage_still_lands() -> None:
    """The mint FINISHES while the hub is away. The result must still arrive.

    This is the case the abandoned-mint telemetry could not cover: it reports
    through the hub, so it shared a failure domain with the outage it exists to
    announce.
    """
    kind = "forge_terminal_pgw869"
    with hub_double(worker_id="outbox-terminal") as (scheduler, harness):
        conn0 = scheduler.wait_connection(0)
        conn0.wait_for(is_ready, _TIMEOUT)
        _await_live_reporting(harness)
        conn0.kill()

        act = activity_mod.begin(kind)
        act.phase("publishing")
        act.completed()

        conn1 = scheduler.wait_connection(1, timeout=_TIMEOUT)
        terminal = conn1.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "activity_update"
                and m.activity_update.kind == kind
                and m.activity_update.state == _COMPLETED
            ),
            _TIMEOUT,
        )
        assert terminal.activity_update.phase == "publishing"


def test_a_long_outage_coalesces_beats_and_keeps_every_measurement() -> None:
    """Getting this backwards is the failure mode the issue names.

    A queue that preserved heartbeats and discarded results would be worse than
    no queue. Beats collapse to the latest per (kind, phase); measurements do
    not collapse at all.
    """
    kind = "outbox_longoutage_pgw869"
    with hub_double(worker_id="outbox-longoutage") as (scheduler, harness):
        conn0 = scheduler.wait_connection(0)
        conn0.wait_for(is_ready, _TIMEOUT)
        _await_live_reporting(harness)
        conn0.kill()

        act = activity_mod.begin(kind)
        act.phase("compiling")
        for i in range(64):
            act.heartbeat()
            if i % 16 == 0:
                activity_mod.emit_event(
                    kind, f"measurement {i}", phase=f"m{i}")

        conn1 = scheduler.wait_connection(1, timeout=_TIMEOUT)
        for i in range(0, 64, 16):
            conn1.wait_for(_is_activity(kind, f"m{i}"), _TIMEOUT)

        got = _activities(conn1, kind)
        # Every measurement survived, with its payload.
        for i in range(0, 64, 16):
            assert any(
                u.phase == f"m{i}" and u.detail == f"measurement {i}"
                for u in got
            ), f"measurement {i} was dropped: {[u.phase for u in got]}"
        # 64 beats of one phase collapsed. They were always destined to be one
        # hub row (same conflict key, empty payload digest); the outbox simply
        # stops carrying 64 copies of it across the outage.
        beats = [u for u in got if u.phase == "compiling" and not u.detail]
        assert len(beats) <= 1, f"beats did not coalesce: {len(beats)}"


# --------------------------------------------------------------------------
# the two loss sites, at the queue, and the dedup that makes replay safe
# --------------------------------------------------------------------------

def test_reconnect_requeues_evidence_instead_of_clearing_it() -> None:
    """Loss site 1: `reset_for_reconnect` used to clear `_items` wholesale."""
    async def _run() -> None:
        q = SendQueue(maxsize=4)
        facts = [_fact("k", f"p{i}", f"d{i}") for i in range(3)]
        for f in facts:
            await q.put(f)
        await q.reset_for_reconnect()
        replayed = []
        while len(q):
            kind, msg = await q.get()
            assert kind == "evidence"
            replayed.append(msg)
        assert replayed == facts          # all three, in production order

    asyncio.run(_run())


def test_a_fact_the_sender_took_but_never_shipped_comes_back() -> None:
    """Loss site 2: the sender POPS before `write()`; a failed write lost it.

    And the duplicate this creates is safe, because both copies carry the same
    hub conflict key — one row, `occurrences = 2`.
    """
    async def _run() -> None:
        q = SendQueue()
        msg = _fact("self_mint_compile", "trace_graph", "entries=36")
        await q.put(msg)
        kind, taken = await q.get()       # sender has it; the write then FAILS
        assert kind == "evidence"
        await q.reset_for_reconnect()
        _kind, replayed = await q.get()
        assert replayed == msg
        assert _hub_row_key(taken) == _hub_row_key(replayed)

        # A fact whose write RETURNED is re-offered once more after a dirty
        # reconnect, because a returned write is only a BUFFERED write (the
        # RESHIP_WINDOW argument). It does not accumulate: one slot, one copy.
        await q.mark_event_shipped(replayed)
        await q.reset_for_reconnect()
        assert q.pending_evidence_count == 1
        _kind, again = await q.get()
        assert _hub_row_key(again) == _hub_row_key(msg)
        await q.mark_event_shipped(again)
        assert q.pending_evidence_count == 0
        assert len(q) == 0            # nothing outstanding while the stream lives

    asyncio.run(_run())


def test_beats_coalesce_in_place_and_facts_never_do() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=2)          # deliberately smaller than the input
        await q.put(_beat("k", "compiling", 1))
        await q.put(_fact("k", "pool", "K=2 binding=vram"))
        for seq in range(2, 40):
            await q.put(_beat("k", "compiling", seq))
        await q.put(_beat("k", "publishing", 99))

        await q.reset_for_reconnect()
        out = []
        while len(q):
            _kind, msg = await q.get()
            out.append(msg.activity_update)
        # one slot per (kind, phase) beat, and the LATEST beat won
        compiling = [u for u in out if u.phase == "compiling"]
        assert len(compiling) == 1 and compiling[0].seq == 39
        assert len([u for u in out if u.phase == "publishing"]) == 1
        # the measurement kept its position ahead of the later beats
        assert [u.phase for u in out] == ["compiling", "pool", "publishing"]

    asyncio.run(_run())


def test_a_result_never_enters_the_evidence_execution_lane() -> None:
    """The precondition that makes `_evidence_bytes` total.

    `_message_key` returns None for a JobResult deliberately — a result is not
    a fact, it is durable under (request_id, attempt) and is never coalesced.
    The evidence map therefore keys on something results do not have, and the
    mypy error that caught this was telling the truth. Pin the invariant so it
    is executable rather than only asserted in a docstring.
    """
    async def _run() -> None:
        q = SendQueue(maxsize=4)
        result = pb.WorkerMessage(job_result=pb.JobResult(
            request_id="r1", attempt=1))
        assert SendQueue._message_key(result) is None
        await q.put(result)
        await q.put(_fact("k", "pool", "K=2"))
        await q.put(_beat("k", "compiling", 1))
        # only the two facts are booked as evidence; the result rides its own
        # durable lane and is untouched by coalescing or shedding.
        assert q.pending_evidence_count == 2
        assert q.pending_result_keys == [("r1", 1)]

        kinds = []
        while len(q):
            kind, _msg = await q.get()
            kinds.append(kind)
        assert kinds.count("result") == 1
        assert kinds.count("evidence") == 2

    asyncio.run(_run())


def test_a_producer_never_blocks_on_a_dead_connection() -> None:
    """`maxsize` backpressure must not reach a path that just MEASURED
    something. 200 facts into a 2-slot queue, with no consumer, must return."""
    async def _run() -> None:
        q = SendQueue(maxsize=2)
        for i in range(200):
            await asyncio.wait_for(q.put(_fact("k", f"p{i}", f"d{i}")), 1.0)
        assert q.pending_evidence_count == 200

    asyncio.run(_run())


# --------------------------------------------------------------------------
# the edge that defeats everything above: an absent hub outlives the JWT
# --------------------------------------------------------------------------

def _jwt(exp_unix: float, jti: str = "jti-1") -> str:
    """An UNSIGNED JWT. The worker only ever decodes `exp`/`jti` unverified for
    its own diagnostics — nothing here is presented to a real verifier."""
    def seg(obj) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"{seg({'alg': 'none'})}.{seg({'exp': exp_unix, 'jti': jti})}.sig"


def _transport() -> Transport:
    return Transport(load_settings(worker_id="auth-probe"), object())


def test_an_expired_credential_refused_by_the_hub_is_never_fatal() -> None:
    """Wall #2 through a different door.

    A worker that outlives a hub outage ages past its 30 min JWT, redials, and
    is refused BECAUSE its token expired. Under the old ladder that is three
    strikes and death — with a full queue, in exactly the scenario the outbox
    exists for. It must not count at all.

    The hub agrees, measured: `AdmitExpiredBootToken` admits an expired token
    while `now < IssuedAtUnix + BootGraceWindow`, and
    `DefaultWorkerJWTBootGrace = 4 h` against `DefaultWorkerJWTTTL = 30 min`
    (tensorhub `config/config.go:619,675`) — ~3.5 h of admissibility past
    expiry, counted from the last rotation, with a `TokenRefresh` pushed on
    admission. Retrying is the documented recovery path; the only thing that
    was defeating it was this worker killing itself first.
    """
    worker_credential.reset()
    try:
        worker_credential.install(_jwt(time.time() - 600.0), 0.0)
        t = _transport()
        for _ in range(50):
            assert t._auth_rejection_is_fatal("invalid worker token state") is False
        # and the ladder is not merely out-run — it is UNREACHABLE.
        assert t._consecutive_auth_failures == 0
    finally:
        worker_credential.reset()


def test_an_expired_rejection_confesses_once_per_credential() -> None:
    """Patient is not the same as silent — and the confession is a hub ROW,
    which the evidence lane now holds through the very outage that caused it."""
    worker_credential.reset()
    captured: List[pb.ActivityUpdate] = []
    prior = activity_mod._sink
    activity_mod._sink = captured.append
    try:
        worker_credential.install(_jwt(time.time() - 60.0, "jti-A"), 0.0)
        t = _transport()
        for _ in range(5):
            t._auth_rejection_is_fatal("invalid worker token state")
        worker_credential.install(_jwt(time.time() - 60.0, "jti-B"), 0.0)
        for _ in range(5):
            t._auth_rejection_is_fatal("invalid worker token state")
    finally:
        activity_mod._sink = prior
        worker_credential.reset()

    rows = [u for u in captured if u.kind == "worker_credential"]
    assert len(rows) == 2, f"one confession per credential, got {len(rows)}"
    assert all(u.phase == "expired_rejected_retrying" for u in rows)
    assert "jti-A" in rows[0].detail and "jti-B" in rows[1].detail


def test_a_LIVE_credential_refused_by_an_answering_hub_still_escalates() -> None:
    """The other half must not regress: revoked/superseded/misconfigured is a
    verdict about this worker, and a pod that can never re-join must not spin
    forever. Only the expired case was exempted."""
    worker_credential.reset()
    try:
        worker_credential.install(_jwt(time.time() + 3600.0), 0.0)
        t = _transport()
        t._auth_rejection_is_fatal("invalid worker token state")
        assert t._consecutive_auth_failures == 1
        t._auth_rejection_is_fatal("invalid worker token state")
        assert t._consecutive_auth_failures == 2
        # the window is what the third strike is measured against; backdate it
        # rather than sleeping 60s for a clock this issue does not own.
        t._first_auth_failure_at -= _AUTH_FAILURE_EXIT_WINDOW_S + 1.0
        assert t._auth_rejection_is_fatal("invalid worker token state") is True
    finally:
        worker_credential.reset()


def test_a_rotation_restarts_the_evidence() -> None:
    """The streak is about a credential. A different one is a different
    attempt, so the evidence accumulated against the old one is stale."""
    worker_credential.reset()
    try:
        worker_credential.install(_jwt(time.time() + 3600.0, "jti-old"), 0.0)
        t = _transport()
        t._auth_rejection_is_fatal("invalid worker token state")
        t._auth_rejection_is_fatal("invalid worker token state")
        assert t._consecutive_auth_failures == 2
        worker_credential.install(_jwt(time.time() + 3600.0, "jti-new"), 0.0)
        t._auth_rejection_is_fatal("invalid worker token state")
        assert t._consecutive_auth_failures == 1
    finally:
        worker_credential.reset()


def test_shedding_is_bounded_and_reports_itself() -> None:
    """Growth is bounded, coalescible sheds first, and a shed emits a fact."""
    async def _run() -> None:
        q = SendQueue(maxsize=4, evidence_max=8)
        for seq in range(6):
            await q.put(_beat("k", f"phase{seq}", seq))
        for i in range(20):
            await q.put(_fact("k", f"m{i}", f"measurement {i}"))

        assert q.pending_evidence_count <= 8 + 1   # +1 = the shed report itself
        assert q.shed_total >= 18

        await q.reset_for_reconnect()
        out = []
        while len(q):
            _kind, msg = await q.get()
            out.append(msg.activity_update)
        shed = [u for u in out if u.kind == "outbox_shed"]
        assert len(shed) == 1, "a shed must never be silent"
        assert str(q.shed_total) in shed[0].detail
        # coalescible beats went first: no beat survived while facts did
        assert not [u for u in out if u.phase.startswith("phase")]
        assert [u for u in out if u.phase.startswith("m")]

    asyncio.run(_run())
