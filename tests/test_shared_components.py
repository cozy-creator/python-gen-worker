"""Worker-owned shared Diffusers component cache (issue #335).

Validates the sharing + refcount + VRAM-accounting logic with lightweight REAL
objects (not deep mocks): a fake "pipeline" is a plain object carrying a
``.components`` dict so the cache's component-id collection and size estimate
exercise the same code paths a diffusers pipeline would.

What is NOT covered here (needs a GPU + real multi-variant Diffusers models):
proving two pipelines built from one shared entry alias the same CUDA storages
via ``data_ptr()``, and the before/after VRAM residency drop on flux.2-klein-4b.
Those live in the live-validation step; this file proves the framework logic
that makes them possible.
"""

from __future__ import annotations

import pytest

from gen_worker import CivitaiRepo, HFRepo, Repo
from gen_worker.models import (
    LoadedComponentKey,
    ModelCache,
    SharedComponentCache,
    build_function_owned_pipeline,
)


# --------------------------------------------------------------------------- #
# Lightweight real fakes                                                        #
# --------------------------------------------------------------------------- #


class _FakeModule:
    """A stand-in for an nn.Module component — no torch, just identity."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakePipeline:
    """A stand-in diffusers pipeline: a container of component references.

    Exposes ``.components`` (the dict diffusers pipelines carry) and a
    ``from_pipe`` classmethod / ``__init__(**components)`` so
    ``build_function_owned_pipeline`` exercises both assembly paths.
    """

    def __init__(self, **components: object) -> None:
        self.components = dict(components)
        self.scheduler = object()  # per-function mutable state, NOT shared

    @classmethod
    def from_pipe(cls, other: "_FakePipeline", **extra: object) -> "_FakePipeline":
        merged = {**other.components, **extra}
        return cls(**merged)


def _new_base() -> _FakePipeline:
    return _FakePipeline(
        vae=_FakeModule("vae"),
        text_encoder=_FakeModule("te"),
        text_encoder_2=_FakeModule("te2"),
    )


def _model_cache_no_gpu() -> ModelCache:
    # max_vram_gb fixed so accounting is deterministic without a GPU.
    return ModelCache(max_vram_gb=100.0)


# --------------------------------------------------------------------------- #
# LoadedComponentKey canonicalization                                           #
# --------------------------------------------------------------------------- #


def test_identical_bindings_produce_equal_keys() -> None:
    a = HFRepo("bfl/flux.2-klein-4b").dtype("bf16")
    b = HFRepo("bfl/flux.2-klein-4b").dtype("bf16")
    ka = LoadedComponentKey.from_binding(a, device_id=0, component_set="pkg.Flux2KleinPipeline")
    kb = LoadedComponentKey.from_binding(b, device_id=0, component_set="pkg.Flux2KleinPipeline")
    assert ka == kb
    assert ka.cache_id() == kb.cache_id()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.dtype("fp16"),            # dtype differs
        lambda b: b.revision("deadbeef"),     # revision differs
        lambda b: b.subfolder("text_encoder"),  # subfolder differs
    ],
)
def test_binding_attribute_differences_do_not_share(mutate) -> None:
    base = HFRepo("bfl/flux.2-klein-4b").dtype("bf16")
    k_base = LoadedComponentKey.from_binding(base, device_id=0, component_set="P")
    k_other = LoadedComponentKey.from_binding(mutate(base), device_id=0, component_set="P")
    assert k_base != k_other
    assert k_base.cache_id() != k_other.cache_id()


def test_device_dtype_quant_component_set_differences_do_not_share() -> None:
    b = HFRepo("bfl/flux.2-klein-4b").dtype("bf16")
    k0 = LoadedComponentKey.from_binding(b, device_id=0, component_set="P")
    # different GPU
    assert k0 != LoadedComponentKey.from_binding(b, device_id=1, component_set="P")
    # different pipeline class / component set
    assert k0 != LoadedComponentKey.from_binding(b, device_id=0, component_set="Q")
    # different quantization
    assert k0 != LoadedComponentKey.from_binding(
        b, device_id=0, component_set="P", quantization="fp8"
    )
    # different quant config (same scheme)
    assert LoadedComponentKey.from_binding(
        b, device_id=0, component_set="P", quantization="fp8", quant_config={"a": 1}
    ) != LoadedComponentKey.from_binding(
        b, device_id=0, component_set="P", quantization="fp8", quant_config={"a": 2}
    )
    # different provider (hf vs tensorhub vs civitai)
    assert LoadedComponentKey.from_binding(Repo("o/r"), device_id=0) != LoadedComponentKey.from_binding(
        CivitaiRepo("123"), device_id=0
    )


def test_unpinned_revision_falls_back_to_snapshot_digest() -> None:
    b = HFRepo("bfl/flux.2-klein-4b")  # no explicit revision
    same = LoadedComponentKey.from_binding(b, snapshot_digest="/cache/snap-A")
    same2 = LoadedComponentKey.from_binding(b, snapshot_digest="/cache/snap-A")
    diff = LoadedComponentKey.from_binding(b, snapshot_digest="/cache/snap-B")
    assert same == same2          # same on-disk snapshot -> share
    assert same != diff           # different snapshot -> do not share


# --------------------------------------------------------------------------- #
# Sharing + refcount                                                            #
# --------------------------------------------------------------------------- #


def test_identical_components_are_shared_one_instance_refcount_two() -> None:
    mc = _model_cache_no_gpu()
    cache = SharedComponentCache(model_cache=mc)
    key = LoadedComponentKey(ref="bfl/flux.2-klein-4b", dtype="bf16", component_set="P")

    base = _new_base()
    loads = {"n": 0}

    def loader() -> _FakePipeline:
        loads["n"] += 1
        return base  # always the same object; we assert it is loaded ONCE

    # GenerateBf16 acquires.
    a = cache.acquire(key, loader, size_gb=20.0)
    # GenerateBf16Compiled acquires the IDENTICAL binding.
    b = cache.acquire(key, loader, size_gb=20.0)

    assert a is b is base                 # one shared instance
    assert loads["n"] == 1                # loaded exactly once (cache hit on 2nd)
    assert cache.refcount(key) == 2       # both classes hold a reference
    stats = cache.stats()
    assert stats.hits == 1 and stats.misses == 1
    assert len(stats.entries) == 1


def test_shared_vram_counted_once() -> None:
    mc = _model_cache_no_gpu()
    cache = SharedComponentCache(model_cache=mc)
    key = LoadedComponentKey(ref="bfl/flux.2-klein-4b", dtype="bf16", component_set="P")
    base = _new_base()

    before = mc.vram_used_gb
    cache.acquire(key, lambda: base, size_gb=20.0)
    after_first = mc.vram_used_gb
    cache.acquire(key, lambda: base, size_gb=20.0)  # second holder, SAME bytes
    after_second = mc.vram_used_gb

    assert after_first - before == pytest.approx(20.0)
    # The second acquire must NOT add another 20GB — shared bytes counted once.
    assert after_second == pytest.approx(after_first)
    assert mc.refcount(key.cache_id()) == 2


def test_shared_component_not_evicted_while_referenced() -> None:
    mc = _model_cache_no_gpu()
    cache = SharedComponentCache(model_cache=mc)
    key = LoadedComponentKey(ref="shared/base", dtype="bf16", component_set="P")
    base = _new_base()
    cache.acquire(key, lambda: base, size_gb=20.0)
    cache.acquire(key, lambda: base, size_gb=20.0)  # refcount 2
    cid = key.cache_id()

    # Try to make the cache reclaim VRAM far beyond budget — the refcounted
    # shared entry must NOT be demoted/evicted.
    mc._evict_lru_for_space(mc.max_vram_gb + 50.0)
    assert mc.is_in_vram(cid)
    assert mc.refcount(cid) == 2

    # An explicit orchestrator-style unload is also refused while referenced.
    assert mc.unload_model(cid) is False
    assert mc.is_in_vram(cid)

    # The shared cache itself refuses to evict while referenced.
    assert cache.evict(key) is False
    assert cache.get(key) is base


def test_releasing_all_refs_makes_entry_evictable() -> None:
    mc = _model_cache_no_gpu()
    cache = SharedComponentCache(model_cache=mc)
    key = LoadedComponentKey(ref="shared/base", dtype="bf16", component_set="P")
    base = _new_base()
    cache.acquire(key, lambda: base, size_gb=20.0)
    cache.acquire(key, lambda: base, size_gb=20.0)
    cid = key.cache_id()

    assert cache.release(key) == 1   # one holder left
    assert mc.unload_model(cid) is False  # still referenced -> refused
    assert cache.release(key) == 0   # last holder released

    # Now the entry is reclaimable. ModelCache VRAM drops back to baseline.
    assert cache.evict(key) is True
    assert not mc.is_in_vram(cid)
    assert mc.vram_used_gb == pytest.approx(0.0)
    assert cache.get(key) is None


def test_drain_skips_referenced_then_force_clears_all() -> None:
    mc = _model_cache_no_gpu()
    cache = SharedComponentCache(model_cache=mc)
    held = LoadedComponentKey(ref="a/held", component_set="P")
    free = LoadedComponentKey(ref="b/free", component_set="P")
    cache.acquire(held, _new_base, size_gb=5.0)     # refcount 1
    cache.acquire(free, _new_base, size_gb=5.0)
    cache.release(free)                              # refcount 0

    # Soft drain frees only the unreferenced entry.
    assert cache.drain() == 1
    assert cache.contains(held)
    assert not cache.contains(free)

    # Force drain (worker shutdown) clears everything regardless of refcount.
    assert cache.shutdown() == 1
    assert not cache.contains(held)
    assert mc.vram_used_gb == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Mutable-boundary isolation (LoRA / override never alias the clean base)       #
# --------------------------------------------------------------------------- #


def test_lora_and_override_bindings_isolate_from_clean_base() -> None:
    clean = HFRepo("bfl/flux.2-klein-4b").dtype("bf16")
    lora = HFRepo("bfl/flux.2-klein-4b").dtype("bf16").allow_lora()
    override = HFRepo("bfl/flux.2-klein-4b").dtype("bf16").allow_override(_FakePipeline)

    k_clean = LoadedComponentKey.from_binding(clean, component_set="P")
    # The worker explicitly REFUSES to share LoRA/override bindings (handled by
    # _shared_component_key_for_binding returning None). Here we assert the key
    # construction itself separates them when an adapter identity is supplied,
    # so even a forced share can't alias the clean base.
    k_lora = LoadedComponentKey.from_binding(lora, component_set="P", adapter_id="lora:set-7")
    k_override = LoadedComponentKey.from_binding(override, component_set="P", adapter_id="override:Pipe")

    assert k_clean != k_lora
    assert k_clean != k_override
    assert k_lora != k_override
    assert len({k_clean.cache_id(), k_lora.cache_id(), k_override.cache_id()}) == 3


def test_adapter_identity_separates_two_lora_overlays() -> None:
    b = HFRepo("bfl/flux.2-klein-4b").dtype("bf16")
    k1 = LoadedComponentKey.from_binding(b, component_set="P", adapter_id="lora:A")
    k2 = LoadedComponentKey.from_binding(b, component_set="P", adapter_id="lora:B")
    assert k1 != k2


# --------------------------------------------------------------------------- #
# Function-owned pipeline over shared components                                #
# --------------------------------------------------------------------------- #


def test_build_function_owned_pipeline_shares_modules_not_pipeline() -> None:
    base = _new_base()
    fn_pipe = build_function_owned_pipeline(base, _FakePipeline)
    # Different pipeline object (own mutable scheduler) ...
    assert fn_pipe is not base
    assert fn_pipe.scheduler is not base.scheduler
    # ... but the heavy immutable modules are the SAME objects (shared VRAM).
    assert fn_pipe.components["vae"] is base.components["vae"]
    assert fn_pipe.components["text_encoder"] is base.components["text_encoder"]


def test_build_function_owned_pipeline_assembles_from_components_dict() -> None:
    # A pipeline class with no from_pipe — must reassemble from .components.
    class _NoFromPipe:
        def __init__(self, **components: object) -> None:
            self.components = dict(components)

    base = _new_base()
    fn = build_function_owned_pipeline(base, _NoFromPipe)
    assert isinstance(fn, _NoFromPipe)
    assert fn.components["vae"] is base.components["vae"]


def test_component_id_diagnostics_prove_object_sharing() -> None:
    mc = _model_cache_no_gpu()
    cache = SharedComponentCache(model_cache=mc)
    key = LoadedComponentKey(ref="shared/base", component_set="P")
    base = _new_base()
    cache.acquire(key, lambda: base, size_gb=3.0)
    entry = cache.stats().entries[0]
    # Diagnostics expose per-component object ids — the cheap proof that two
    # pipelines built from this entry point at the SAME module objects.
    assert entry["component_ids"]["vae"] == id(base.components["vae"])
    assert entry["refcount"] == 1
    assert entry["object_id"] == id(base)
