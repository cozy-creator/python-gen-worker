"""A cancel that lands around an acquisition's GRANT must not walk away holding it."""

from __future__ import annotations

import asyncio
from pathlib import Path

from gen_worker.models.refs import WireRef
from gen_worker.models.store import ModelStore
from gen_worker.lifecycle_intents import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb

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
        """Preempt a queued materialization ``offset`` scheduling steps after the ref lock is handed to it."""
        registry = IntentRegistry("release-1", [])
        store = ModelStore(_noop_send, cache_dir=cache_dir)
        store.bind_intent_registry(registry)
        ref = WireRef("owner/model@latest")

        lock = store._lock(ref)
        await lock.acquire()

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

        lock.release()
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
