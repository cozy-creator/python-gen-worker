"""pgw#845: a cancel that lands around an acquisition's GRANT must not walk away
holding it.

Live shape (the 0.88.0 CI failure that was read as a timing flake, run
30695617252): `test_reconcile_busy_pgw654` timed out waiting for a newly desired
ref to reach ON_DISK. The message trace shows the job finishing and then 30 s of
lifecycle chatter with no model_event at all, and the captured log carries
neither a materialization warning nor a phase-error line — the convergence did
not fail, it never ran. Reproduced 6/12 under single-core contention.

Mechanism: `IntentRegistry.guard_await` wrapped the awaitable in its OWN task.
When the caller is cancelled in the window where that task has already completed,
asyncio delivers CancelledError to the caller and the completed result is
discarded. `ensure_local` therefore never set `acquired = True`, so its
`finally` never released the per-ref lock — held, forever, by nobody. Every
later materialization of that ref blocks on it, and because `_reconcile_pass` is
serialized the worker silently stops converging desired residency for good.

The canceller is ordinary: `Lifecycle.on_message` preempts the residency
reconcile on every run_job. So a tenant job arriving at the instant a reconcile
was granted the ref lock permanently wedged that worker.

Three sibling call sites had the identical shape — `_intent_lock`, the exclusive
group permits, and the per-job GPU permit (a leak there costs the worker a GPU
slot per cancelled job until it can run nothing at all).

Why this sweeps offsets instead of naming one: the defect lives in HOW MANY task
hops sit between the grant and the caller resuming, and the fix removes a hop.
A test pinned to one offset would have measured the bug before the fix and an
empty window after it. Cancelling at every scheduling offset across the grant is
deterministic (asyncio steps, not wall clock) and stays honest across the fix.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from gen_worker.executor import ModelStore
from gen_worker.lifecycle_intents import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb

# Enough scheduling steps to cover the grant and several hops past it. The
# pre-fix leak sits at the offset where the wrapper task had completed and the
# caller had not yet resumed.
_OFFSETS = range(6)


async def _noop_send(_message: pb.WorkerMessage) -> None:
    return None


def _stage(registry: IntentRegistry, intent_id: str) -> int:
    state = next(
        item for item in registry.snapshot().intents if item.intent_id == intent_id
    )
    return int(state.stage)


def test_no_cancel_offset_around_the_ref_lock_grant_can_wedge_the_ref(
    tmp_path: Path,
) -> None:
    async def preempt_at(offset: int, cache_dir: Path) -> bool:
        """Preempt a queued materialization ``offset`` scheduling steps after the
        ref lock is handed to it. Returns whether the lock was left held."""
        registry = IntentRegistry("release-1", [])
        store = ModelStore(_noop_send, cache_dir=cache_dir)
        store.bind_intent_registry(registry)
        ref = "owner/model:latest"

        lock = store._lock(ref)
        await lock.acquire()  # a sibling materialization owns the ref

        preempted = asyncio.create_task(store.ensure_local(ref))
        for _ in range(8):
            await asyncio.sleep(0)
            intent_id = store._materialize_active.get(ref)
            if intent_id and _stage(registry, intent_id) == (
                pb.LIFECYCLE_INTENT_STAGE_WAIT_REF_LOCK
            ):
                break
        else:  # pragma: no cover - the setup itself failed
            raise AssertionError(
                "the materialization never queued on the ref lock, so this "
                "sweep never reaches the grant it is about to preempt"
            )

        lock.release()  # granted to the preempted materialization
        for _ in range(offset):
            await asyncio.sleep(0)
        preempted.cancel()
        await asyncio.gather(preempted, return_exceptions=True)
        return lock.locked()

    async def run() -> None:
        wedged = [
            offset
            for offset in _OFFSETS
            if await preempt_at(offset, tmp_path / f"cache-{offset}")
        ]
        assert wedged == [], (
            f"a cancel {wedged} scheduling step(s) after the ref lock was granted "
            f"left the lock held by nobody — every later materialization of that "
            f"ref, and every desired ref behind it in the serialized reconcile "
            f"pass, is wedged for the life of the pod"
        )

    asyncio.run(run())
