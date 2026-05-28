"""Worker-owned shared Diffusers component cache (issue #335).

Multiple model variants/bindings on one endpoint often pin the SAME immutable
base components. In the FLUX.2 Klein endpoint, ``GenerateBf16`` and
``GenerateBf16Compiled`` both bind the same HFRepo base model; loading each
binding independently builds a duplicate Python module graph and duplicate CUDA
storages for the identical VAE + text encoders. SDXL fine-tunes do the same with
a shared CLIP/VAE stack across many UNets.

This module gives the worker *process-local, per-GPU* sharing of immutable
loaded components so the bytes are loaded ONCE and shared by reference across
pipelines, with refcounting so a shared component is never freed while any
pipeline still references it.

What is and is NOT shared
-------------------------
Only IMMUTABLE base components are shared (text encoders, VAE, tokenizers — the
frozen stack). Mutable, function/request-owned state stays per-pipeline:
schedulers, LoRA overlays, compiled wrappers, adapters, and any endpoint
mutation. Callers build a *function-owned* pipeline object over the shared
modules (diffusers ``from_pipe`` / ``Pipeline(**components)``) rather than
sharing one mutable pipeline object — unless they explicitly opt into identity
sharing.

Safety boundaries enforced by the cache KEY (``LoadedComponentKey``)
-------------------------------------------------------------------
- ``device_id`` is part of the key, so multi-GPU workers never share CUDA
  storage across devices — each GPU gets its own loaded entry.
- ``dtype``, ``quantization`` + ``quant_config_digest``, ``revision`` /
  ``snapshot_digest``, ``placement`` (offload mode), ``subfolder``, and the
  ``component_set`` / ``pipeline_class`` identity are all part of the key, so a
  bf16 entry is never handed back for an fp8 request, a compiled/override
  identity never collides with a plain one, etc.
- LoRA-capable and override-capable identities fold their adapter/override
  identity into the key (``adapter_id``), so a LoRA'd or overridden wrapper can
  never alias the clean shared base another function depends on.

Integration with :class:`gen_worker.models.cache.ModelCache`
------------------------------------------------------------
Each shared entry registers in ModelCache ONCE (VRAM counted once) and holds a
refcount there for every live acquirer. ModelCache eviction/unload skip any
entry with ``refcount > 0`` (see ``ModelCache.acquire_ref`` / ``release_ref``),
so a shared component is never demoted/evicted while in use, and interplays with
the #337 pinned-base rule (pinning and refcounting both veto eviction).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _digest(value: Any) -> str:
    """Stable short digest of an arbitrary (small) value for key canonicalization.

    Used for quantization configs / revision blobs where the structure may be a
    dict. ``repr`` with sorted dict keys keeps it deterministic across runs.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.strip()
    elif isinstance(value, dict):
        raw = repr(sorted((str(k), repr(v)) for k, v in value.items()))
    else:
        raw = repr(value)
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LoadedComponentKey:
    """Canonical identity of a loadable immutable component set (#335).

    Two bindings share a single loaded entry IFF every field of their key is
    equal. The fields cover every dimension that, if different, would make the
    loaded bytes incompatible to share:

    - ``provider`` / ``ref``: which repo (tensorhub / hf / civitai + repo id).
    - ``revision``: pinned git revision OR resolved snapshot digest — different
      weights ⇒ different entry.
    - ``dtype``: torch dtype the weights were materialized at.
    - ``quantization`` + ``quant_config_digest``: quant scheme and its config.
    - ``device_id``: GPU index. Per-device so CUDA storage is never aliased
      across GPUs on a multi-GPU worker.
    - ``placement``: offload/placement mode (full / model_offload / sequential /
      group). A streamed-offload pipeline has different residency semantics.
    - ``subfolder``: which named subfolder of the repo (a text_encoder vs a vae
      living in the same SDXL repo).
    - ``component_set``: identity of the component set / pipeline class the
      entry was assembled for — incompatible pipeline classes don't share.
    - ``adapter_id``: LoRA/override identity. Empty for the clean shared base;
      non-empty isolates a LoRA'd / overridden wrapper into its own entry so it
      can never contaminate the clean base another function depends on.
    """

    provider: str = "tensorhub"
    ref: str = ""
    revision: str = ""
    dtype: str = ""
    quantization: str = ""
    quant_config_digest: str = ""
    device_id: int = 0
    placement: str = "full"
    subfolder: str = ""
    component_set: str = ""
    adapter_id: str = ""

    @classmethod
    def from_binding(
        cls,
        binding: Any,
        *,
        device_id: int = 0,
        placement: str = "full",
        component_set: str = "",
        snapshot_digest: str = "",
        quantization: str = "",
        quant_config: Any = None,
        adapter_id: str = "",
    ) -> "LoadedComponentKey":
        """Build a key from an ``@inference`` binding (Repo/HFRepo/CivitaiRepo).

        Pulls provider/ref/dtype/revision/subfolder off the binding and folds in
        the runtime-supplied device/placement/quant/adapter dimensions. Falls
        back to ``snapshot_digest`` for the revision when the binding pins no
        explicit revision (so two un-pinned bindings that resolved to the same
        on-disk snapshot still share, and two that resolved to different
        snapshots do not).
        """
        provider = str(getattr(binding, "provider", "tensorhub") or "tensorhub").strip()
        ref = str(getattr(binding, "ref", "") or "").strip()
        dtype = str(getattr(binding, "_dtype", "") or "").strip().lower()
        revision = str(getattr(binding, "_revision", "") or "").strip()
        if not revision:
            revision = str(snapshot_digest or "").strip()
        subfolder = str(getattr(binding, "_subfolder", "") or "").strip()
        return cls(
            provider=provider,
            ref=ref,
            revision=revision,
            dtype=dtype,
            quantization=str(quantization or "").strip().lower(),
            quant_config_digest=_digest(quant_config),
            device_id=int(device_id),
            placement=str(placement or "full").strip() or "full",
            subfolder=subfolder,
            component_set=str(component_set or "").strip(),
            adapter_id=str(adapter_id or "").strip(),
        )

    def cache_id(self) -> str:
        """Deterministic ModelCache model_id for this key.

        Human-readable prefix + a digest of every field so distinct keys never
        collide in the underlying ModelCache (which is keyed by string).
        """
        fields = (
            self.provider, self.ref, self.revision, self.dtype,
            self.quantization, self.quant_config_digest, str(self.device_id),
            self.placement, self.subfolder, self.component_set, self.adapter_id,
        )
        digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:16]
        readable = (self.ref or "?").replace("/", "--")[:48]
        return f"shared::{readable}::dev{self.device_id}::{digest}"


@dataclass
class _Entry:
    """A single loaded shared component set, tracked by the cache."""

    key: LoadedComponentKey
    cache_id: str
    obj: Any = None
    size_gb: float = 0.0
    refcount: int = 0
    component_ids: Dict[str, int] = field(default_factory=dict)


@dataclass
class SharedComponentStats:
    """Diagnostics snapshot for heartbeat / debugging (#335 acceptance)."""

    hits: int
    misses: int
    entries: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "entries": self.entries}


class SharedComponentCache:
    """Process-local, per-GPU cache of immutable loaded Diffusers components.

    Thread-safe. Lifecycle: ``acquire(key, loader)`` loads-once-or-reuses and
    bumps the refcount; ``release(key)`` drops a reference; ``drain()`` /
    ``shutdown()`` clear everything (drain refuses while references are live by
    default). Integrates with an optional :class:`ModelCache` so shared VRAM is
    counted once and protected from eviction while referenced.
    """

    def __init__(self, model_cache: Optional[Any] = None) -> None:
        self._model_cache = model_cache
        self._entries: Dict[LoadedComponentKey, _Entry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    # ----------------------------------------------------------------- acquire
    def acquire(
        self,
        key: LoadedComponentKey,
        loader: Callable[[], Any],
        *,
        size_gb: float = 0.0,
        pin: bool = False,
    ) -> Any:
        """Return the shared loaded object for ``key``, loading it once if absent.

        On a HIT the existing object is returned and the refcount bumped (no
        second load, no duplicate CUDA storage). On a MISS ``loader()`` is
        invoked exactly once (under the per-key load gate), the result is
        registered in the ModelCache (VRAM counted once), and a reference is
        acquired.

        ``loader`` must return the loaded component/pipeline object. ``size_gb``
        is the entry's VRAM footprint for ModelCache accounting; when 0 the
        cache estimates it from the loaded object's parameters.

        ``pin=True`` additionally pins the ModelCache entry (#337) — appropriate
        for a frozen shared base that should never leave VRAM even at refcount 0.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.obj is not None:
                self._hits += 1
                entry.refcount += 1
                if self._model_cache is not None:
                    self._model_cache.acquire_ref(entry.cache_id)
                logger.info(
                    "SharedComponentCache HIT %s refcount=%d", entry.cache_id, entry.refcount
                )
                return entry.obj
            # MISS — create the entry placeholder, then load OUTSIDE? We load
            # under the lock for correctness: concurrent acquire() of the same
            # key must not double-load. Loads are serialized per-process anyway
            # (the GPU semaphore is held by the worker setup path), so holding
            # the cache lock across the load is acceptable and avoids a second
            # per-key lock layer.
            self._misses += 1
            obj = loader()
            est = size_gb if size_gb > 0 else self._estimate_size_gb(obj)
            entry = _Entry(
                key=key,
                cache_id=key.cache_id(),
                obj=obj,
                size_gb=est,
                refcount=1,
                component_ids=self._collect_component_ids(obj),
            )
            self._entries[key] = entry
            if self._model_cache is not None:
                # Register the shared bytes in the ModelCache ONCE so its VRAM
                # accounting reflects them exactly one time, then acquire a ref
                # so eviction can't reclaim them while a pipeline holds them.
                self._model_cache.mark_loaded_to_vram(
                    entry.cache_id, obj, est, pinned=pin
                )
                self._model_cache.acquire_ref(entry.cache_id)
            logger.info(
                "SharedComponentCache MISS %s loaded size=%.2fGB refcount=1",
                entry.cache_id, est,
            )
            return obj

    # ----------------------------------------------------------------- release
    def release(self, key: LoadedComponentKey) -> int:
        """Drop one reference to ``key``; returns the new refcount.

        Releasing the last reference makes the entry an eligible eviction
        candidate in the ModelCache but does NOT immediately free it (so a
        re-acquire is a fast hit). Use :meth:`evict` / :meth:`drain` to reclaim.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            if entry.refcount > 0:
                entry.refcount -= 1
            if self._model_cache is not None:
                self._model_cache.release_ref(entry.cache_id)
            logger.info(
                "SharedComponentCache RELEASE %s refcount=%d",
                entry.cache_id, entry.refcount,
            )
            return entry.refcount

    def refcount(self, key: LoadedComponentKey) -> int:
        with self._lock:
            entry = self._entries.get(key)
            return int(entry.refcount) if entry is not None else 0

    def get(self, key: LoadedComponentKey) -> Optional[Any]:
        """Peek the shared object without changing the refcount (or None)."""
        with self._lock:
            entry = self._entries.get(key)
            return entry.obj if entry is not None else None

    def contains(self, key: LoadedComponentKey) -> bool:
        with self._lock:
            return key in self._entries

    # ------------------------------------------------------------------- evict
    def evict(self, key: LoadedComponentKey, *, force: bool = False) -> bool:
        """Free an entry. Refuses (returns False) while referenced unless ``force``.

        Drops the cache entry and unloads it from the ModelCache. Used by drain
        and by explicit unload commands.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.refcount > 0 and not force:
                logger.info(
                    "SharedComponentCache evict(%s) refused: refcount=%d",
                    entry.cache_id, entry.refcount,
                )
                return False
            self._free_entry(entry)
            del self._entries[key]
            return True

    def drain(self, *, force: bool = False) -> int:
        """Evict all entries with refcount 0 (or everything when ``force``).

        Returns the number of entries freed. Worker shutdown calls
        ``drain(force=True)``; a soft drain at idle calls ``drain()``.
        """
        with self._lock:
            freed = 0
            for key in list(self._entries.keys()):
                entry = self._entries[key]
                if entry.refcount > 0 and not force:
                    continue
                self._free_entry(entry)
                del self._entries[key]
                freed += 1
            return freed

    def shutdown(self) -> int:
        """Force-free everything (worker drain). Returns entries freed."""
        return self.drain(force=True)

    def _free_entry(self, entry: _Entry) -> None:
        # Force the ModelCache refcount to 0 so the underlying unload proceeds,
        # then unload. We only get here when this cache decided to free it.
        if self._model_cache is not None:
            try:
                # Drop any residual ModelCache refs this cache still holds.
                while self._model_cache.refcount(entry.cache_id) > 0:
                    self._model_cache.release_ref(entry.cache_id)
                self._model_cache.unload_model(entry.cache_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SharedComponentCache: ModelCache unload failed for %s: %s",
                    entry.cache_id, exc,
                )
        entry.obj = None
        entry.refcount = 0

    # ------------------------------------------------------------- diagnostics
    def stats(self) -> SharedComponentStats:
        """Snapshot of hits/misses + per-entry refcount, device, component ids
        and size — the diagnostics the #335 acceptance asks for."""
        with self._lock:
            entries = [
                {
                    "cache_id": e.cache_id,
                    "ref": e.key.ref,
                    "provider": e.key.provider,
                    "device_id": e.key.device_id,
                    "dtype": e.key.dtype,
                    "quantization": e.key.quantization,
                    "placement": e.key.placement,
                    "component_set": e.key.component_set,
                    "adapter_id": e.key.adapter_id,
                    "refcount": e.refcount,
                    "size_gb": round(e.size_gb, 3),
                    "object_id": id(e.obj) if e.obj is not None else 0,
                    "component_ids": dict(e.component_ids),
                }
                for e in self._entries.values()
            ]
            return SharedComponentStats(hits=self._hits, misses=self._misses, entries=entries)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _estimate_size_gb(obj: Any) -> float:
        try:
            from ..inference_memory import estimate_pipeline_size_gb
            return float(estimate_pipeline_size_gb(obj) or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _collect_component_ids(obj: Any) -> Dict[str, int]:
        """Map ``{component_name: id(module)}`` for storage-sharing evidence.

        ``id()`` of each sub-module is the cheap process-local proof that two
        pipelines built from one entry point at the SAME object (and therefore
        the same CUDA storages). On GPU this is corroborated by comparing
        ``next(module.parameters()).data_ptr()`` across pipelines; that check
        needs real weights and lives in the GPU validation step.
        """
        out: Dict[str, int] = {}
        comps = getattr(obj, "components", None)
        if isinstance(comps, dict):
            for name, comp in comps.items():
                if comp is not None:
                    out[str(name)] = id(comp)
        else:
            for attr in (
                "unet", "transformer", "vae", "text_encoder",
                "text_encoder_2", "text_encoder_3", "tokenizer",
            ):
                comp = getattr(obj, attr, None)
                if comp is not None:
                    out[attr] = id(comp)
        return out


def build_function_owned_pipeline(
    shared: Any,
    pipeline_cls: Optional[Any] = None,
    **extra_components: Any,
) -> Any:
    """Build a *function-owned* pipeline over SHARED immutable components.

    The default sharing mode (#335): each function gets its own pipeline object
    (own scheduler, own request-local state) whose heavy modules are the SAME
    objects as ``shared`` — so VRAM is shared but mutable state is not. Tries,
    in order:

    1. ``pipeline_cls.from_pipe(shared, **extra_components)`` — the diffusers
       supported API for re-housing one pipeline's components in another
       pipeline class without reloading weights.
    2. ``pipeline_cls(**{**shared.components, **extra_components})`` — assemble
       a fresh pipeline from the shared ``components`` dict (also a supported
       diffusers construction path) plus any per-function overrides
       (e.g. a fresh scheduler in ``extra_components``).

    When ``pipeline_cls`` is None, falls back to ``shared.__class__``. Raises if
    neither path is available so callers don't silently share mutable state.
    """
    cls = pipeline_cls or type(shared)
    from_pipe = getattr(cls, "from_pipe", None)
    if callable(from_pipe):
        return from_pipe(shared, **extra_components)
    comps = getattr(shared, "components", None)
    if isinstance(comps, dict):
        merged = {**comps, **extra_components}
        return cls(**merged)
    raise TypeError(
        f"cannot build a function-owned pipeline from {type(shared).__name__}: "
        "no from_pipe() and no .components dict to re-assemble"
    )
