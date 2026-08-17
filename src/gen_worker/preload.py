"""Rotation preload: stage the NEXT checkpoint while the current job
computes — the worker half of "Rotating double-buffer serving".

The hub's desired plan (``DesiredResidency.hot``) names the instances a
worker should hold. Lifecycle's ``_reconcile_pass`` is gated on tenant idle
and cancelled by every run_job, so on its own the next checkpoint's
multi-second repo-cas pull and load never overlap inference and every hop pays
the full visible swap (~14s = 11s pull + 3s VRAM load). This driver stages
desired instances IN THE BACKGROUND while jobs run, so rotation on job
completion approaches a pointer swap.

Stage ladder per candidate instance (capacity-tier ruling):

1. **NVMe CAS** — ``store.ensure_local`` for every ref (kills the download
   term outright).
2. **True double-buffer** — when ``residency.fits()`` says the instance's
   weights fit ALONGSIDE everything resident (in-flight leases protected),
   run the full ``ensure_desired_instance`` setup in the background: the
   next dispatch finds a ready record and the visible swap is ~0.
3. **Component-first host staging** — when two don't fit: the instance's
   EXCLUSIVE components (by content digest — components already resident in
   the shared cache are, by construction, the shared TE/VAE that stay put)
   load on CPU on a dedicated low-priority thread, get eagerly PINNED
   (:func:`~gen_worker.models.pinned_swap.prestage_module`), and are seeded
   into the shared-component cache. The existing content-keyed injection
   path (``_component_share_plan`` includes any key already resident)
   consumes them at rotation, so the dispatch-time ``from_pretrained``
   skips those components' disk reads and the promote is a full-PCIe H2D
   on the copy stream. Plain-dtype bindings only — quantized flavors
   (fp8/svdq/nf4/w8a8) load through special lanes a vanilla component load
   cannot reproduce, so they stop at tier 1.

Fence carve-out: a ref whose RESIDENT identity
differs from the desired snapshot digest is never touched here — the
idle-gated reconcile owns identity moves, because a tenant job may be
re-materializing the older bytes concurrently. This driver only stages
refs that are new to the process or identity-identical, which is exactly
the rotation case (a DIFFERENT next checkpoint).

No wait-on-peer coordination, no timeouts: the driver is level-triggered
by pokes (job admitted/finished, new desired set) and every await is real
work with its own progress guarantees.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import typing
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import activity as activity_mod
from . import dispatch
from .models.refs import WireRef
from .api.binding import wire_ref
from .models import residency as residency_mod
from .models.pinned_swap import prestage_module
from pathlib import Path
import functools
from .pb import worker_scheduler_pb2 as pb

if typing.TYPE_CHECKING:  # pragma: no cover - typing only
    from .executor import Executor
    from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

_GiB = 1024 ** 3

# Components below this are config-ish (schedulers, tokenizers): staging
# them buys nothing and pollutes the shared cache. Constant, not a knob.
_MIN_STAGE_COMPONENT_BYTES = 32 * 1024 * 1024

# Dtypes whose component loads are reproducible by a plain from_pretrained.
_PLAIN_DTYPES = frozenset(
    {"", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}
)


def _nice_thread_init() -> None:
    """Background staging threads run at nice +10 (same posture as the
    background-mint warm thread): loading must not slow serving."""
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    except (OSError, AttributeError):  # pragma: no cover - platform-specific
        pass


class Preloader:
    """Background staging driver owned by the executor.

    Lifecycle hands it the accepted desired set on every HelloAck
    (:meth:`update_desired`); the executor pokes it on job admission and
    completion (rotation points). It is deliberately NOT cancelled when a
    run_job arrives — running during tenant work is its entire purpose —
    and it stops on drain.
    """

    def __init__(self, executor: "Executor") -> None:
        self._ex = executor
        self._hot: Tuple["pb.DesiredInstance", ...] = ()
        self._snapshots: Dict[WireRef, "pb.Snapshot"] = {}
        self._generation = -1
        self._wake: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._stopped = False
        self._pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Instance identities (serialized DesiredInstance) that failed a
        # background stage this generation — retried only on a new desired
        # set, never in a loop.
        self._failed: set = set()
        # Identities REFUSED deterministically (a composition the
        # tree plus the injection cannot satisfy). Never cleared by a new
        # desired set: the verdict is a function of the identity's own bytes,
        # so re-sending the same DesiredInstance cannot change it. A hub that
        # fixes the binding sends DIFFERENT bytes and is retried on merit —
        # that is the progress signal here, and it is why there is no timer.
        self._refused: set = set()

    # ---- inputs -----------------------------------------------------------

    def update_desired(
        self,
        hot: Sequence["pb.DesiredInstance"],
        snapshots: Mapping[WireRef, "pb.Snapshot"],
        generation: int,
    ) -> None:
        """Full-replacement desired state (call only with the ACCEPTED
        generation — lifecycle already filters stale acks)."""
        if generation < self._generation:
            return
        self._generation = generation
        self._hot = tuple(hot)
        self._snapshots = dict(snapshots)
        self._failed.clear()
        self.poke()

    def poke(self) -> None:
        """Signal a rotation point (job admitted/finished, new plan)."""
        if self._stopped:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (sync test construction): next poke will land
        if self._wake is None:
            self._wake = asyncio.Event()
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run(), name="rotation-preload")

    def stop(self) -> None:
        self._stopped = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    # ---- driver -----------------------------------------------------------

    async def _run(self) -> None:
        wake = self._wake
        assert wake is not None
        try:
            while not self._stopped and not self._ex.draining:
                wake.clear()
                staged_any = await self._pass()
                if self._stopped or self._ex.draining:
                    return
                if staged_any:
                    continue  # re-derive: the world moved while we staged
                await wake.wait()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("rotation preload driver crashed; parked until next poke")
            # The whole background-staging subsystem is now off
            # until the next poke — the hub's desired plan goes silently
            # unfulfilled and every rotation pays the full visible swap.
            activity_mod.emit_event(
                activity_mod.KIND_ROTATION_PRELOAD,
                f"driver crashed, parked until next poke: "
                f"{type(exc).__name__}: {exc}",
                phase="driver_crashed",
            )

    async def _pass(self) -> bool:
        """One convergence look at the desired set. Returns True when it
        performed staging work (caller re-derives candidates)."""
        for instance in self._hot:
            if self._stopped or self._ex.draining:
                return False
            ident = instance.SerializeToString(deterministic=True)
            if ident in self._failed or ident in self._refused:
                continue
            try:
                did = await self._stage_instance(instance)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failed.add(ident)
                logger.warning(
                    "rotation preload of %s failed (%s: %s); will retry on the "
                    "next desired set",
                    instance.function_name, type(exc).__name__, exc,
                )
                # The hub planned this instance hot; the stage is abandoned for
                # the whole desired-set generation, so the next rotation to it
                # pays the full visible swap.
                activity_mod.emit_event(
                    activity_mod.KIND_ROTATION_PRELOAD,
                    f"fn={instance.function_name} "
                    f"generation={self._generation}: stage failed, retried "
                    f"only on a new desired set: {type(exc).__name__}: {exc}",
                    phase="stage_failed",
                )
                continue
            if did:
                return True
        return False

    # ---- one candidate ----------------------------------------------------

    def _effective_for(
        self, instance: "pb.DesiredInstance"
    ) -> Optional[Any]:
        """The derived per-pick spec this instance resolves to (same remap
        the validated warm path applies), or None when it cannot resolve —
        resolution problems are the reconcile's to report, not ours."""

        ex = self._ex
        spec = ex.specs.get(instance.function_name)
        if spec is None or spec.cls is None:
            return None
        orders: Dict[str, dispatch.SlotOrder] = {}
        for m in instance.models:
            slot = m.slot.strip()
            ref = m.ref.strip()
            pick = ex._model_resolutions.get(ref)
            if pick is not None and pick[0]:
                ref = pick[0]
            if not slot or not ref:
                return None
            # pgw#1333: the whole binding, not just its ref — the rotation
            # preload derives the SAME effective spec the validated warm path
            # does, and a spec whose slot facts differ is a different spec.
            orders[slot] = dispatch.order_from_binding(
                m, ref=ref, owed_by=dispatch.BOOT_SENDER_OWES)
        try:
            return ex._dispatched_spec(spec, orders)
        except Exception:
            return None

    def _instance_refs(self, effective: Any) -> Dict[WireRef, Any]:
        """ref -> binding for every setup slot."""
        ex = self._ex
        out: Dict[WireRef, Any] = {}
        for slot in ex._setup_slots(effective):
            binding = effective.models[slot]
            out.setdefault(wire_ref(binding), binding)
        return out

    def _fence_conflict(self, ref: WireRef) -> bool:
        """True when the ref is RESIDENT under a different identity than the
        desired snapshot names — the case the idle-gated reconcile owns (a
        tenant job may be re-materializing the older bytes)."""
        store = self._ex.store
        wanted = store.snapshot_digest(ref, self._snapshots.get(ref))
        have = store.resident_identity(ref)[0]
        return bool(have and wanted and have != wanted)

    async def _stage_instance(self, instance: "pb.DesiredInstance") -> bool:
        ex = self._ex
        effective = self._effective_for(instance)
        if effective is None:
            return False
        rec = ex._classes.get(effective.instance_key)
        if rec is not None and rec.ready and not rec.stale:
            return False  # already hot
        refs = self._instance_refs(effective)
        if not refs:
            return False
        if any(self._fence_conflict(ref) for ref in refs):
            logger.info(
                "rotation preload skipping %s: a ref's resident identity "
                "differs from the desired snapshot (idle-gated reconcile owns "
                "identity moves)", instance.function_name,
            )
            return False

        did_work = False
        # Tier 1: bytes to the local NVMe CAS. ``local_path`` is the honest
        # "materialized on disk" signal (a banked snapshot alone is not);
        # identity moves never reach here (fence guard above).
        for ref, binding in refs.items():
            if self._stopped or self._ex.draining:
                return did_work
            if ex.store.local_path(ref) is not None:
                continue
            await ex.store.ensure_local(
                ref, self._snapshots.get(ref), binding=binding)
            did_work = True
            logger.info("rotation preload staged %s to local CAS", ref)

        # Tier 2: true double-buffer when everything fits alongside the
        # resident set (in-flight leases and pins protect the serving model
        # inside fits()/make_room, so a fitting setup never displaces it).
        res = ex.store.residency
        sizes = {
            ref: self._expected_vram_bytes(ref) for ref in refs
        }
        if res.fits(typing.cast(Mapping[str, int], sizes)):
            logger.info(
                "rotation preload: %s fits alongside the resident set — "
                "running full background setup (double-buffer)",
                instance.function_name,
            )
            await ex.ensure_desired_instance(instance, self._snapshots)
            return True

        # Tier 3: component-first pinned host staging.
        staged = await self._stage_components_host(effective)
        return did_work or staged

    def _expected_vram_bytes(self, ref: WireRef) -> int:
        res = self._ex.store.residency
        hint = res.vram_hint(ref)
        if hint > 0:
            return hint
        snap = self._snapshots.get(ref)
        if snap is not None:
            total = sum(int(f.size_bytes) for f in snap.files)
            if total > 0:
                return total
        return sum(self._ex.store.component_sizes(ref).values())

    # ---- component-first host staging -------------------------------------

    async def _stage_components_host(self, effective: Any) -> bool:
        """Load the instance's exclusive components on CPU, pin them, and
        seed the shared-component cache so dispatch-time injection consumes
        them (promote = H2D on the copy stream, from_pretrained skips their
        disk reads). Shared components already resident stay untouched —
        the component-first ruling by construction."""

        # CYCLE: models.loading is reached through executor, which imports preload.
        from .models.loading import load_component

        ex = self._ex
        res = ex.store.residency
        staged_any = False
        for slot in ex._worker_loaded_slots(effective):
            binding = effective.models.get(slot)
            if binding is None:
                continue
            ref = wire_ref(binding)
            dtype = str(getattr(binding, "dtype", "") or "").lower()
            storage = str(getattr(binding, "storage_dtype", "") or "")
            if storage or dtype not in _PLAIN_DTYPES:
                logger.info(
                    "rotation preload: %s slot %s uses a quantized lane "
                    "(dtype=%r storage=%r); staging stops at the disk tier",
                    effective.name, slot, dtype, storage,
                )
                continue
            if ex._placement_mode(effective, ref) != "auto":
                continue  # offload placements refuse shared modules anyway
            local = ex.store.local_path(ref)
            if local is None:
                continue
            digests = ex.store.component_digests(ref, local_path=Path(local))
            if not digests:
                continue
            sizes = ex.store.component_sizes(ref)
            for comp, digest in sorted(digests.items()):
                if self._stopped or self._ex.draining:
                    return staged_any
                if not comp or not digest:
                    continue
                nbytes = sizes.get(comp, 0)
                if nbytes < _MIN_STAGE_COMPONENT_BYTES:
                    continue
                key = residency_mod.LoadedComponentKey.for_component(
                    content_digest=digest, component=comp, binding=binding,
                    label=f"{ref}/{comp}",
                )
                if res.shared_obj(key) is not None:
                    continue  # already resident (the shared TE/VAE case)
                if not res.host_ram_headroom(nbytes).sufficient:
                    logger.info(
                        "rotation preload: host RAM cannot stage %s/%s "
                        "(%.2fGiB); stopping at the disk tier",
                        ref, comp, nbytes / _GiB,
                    )
                    return staged_any
                module = await self._in_stage_thread(
                    load_component, local, comp, dtype=dtype,
                )
                pinned = await self._in_stage_thread(prestage_module, module)
                # Seed-then-release: the entry becomes an ordinary LRU RAM
                # candidate until a setup consumes it (acquire_shared hit ->
                # holders>0), so host pressure can always reclaim it.
                def _loader(m: Any = module) -> Any:
                    return m

                res.acquire_shared(key, _loader)
                res.release_shared(key)
                staged_any = True
                logger.info(
                    "rotation preload staged component %s/%s (%.2fGiB, "
                    "%.2fGiB pinned) into the shared cache",
                    ref, comp, nbytes / _GiB, pinned / _GiB,
                )
        return staged_any

    async def _in_stage_thread(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run blocking staging work on the dedicated nice+10 thread (never
        the shared to_thread pool — its threads also run tenant handlers)."""
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="rotation-preload",
                initializer=_nice_thread_init,
            )
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self._pool, functools.partial(fn, *args, **kwargs))


__all__ = ["Preloader"]
