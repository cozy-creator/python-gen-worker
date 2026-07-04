"""Shared-component identity + single-count VRAM accounting (#335, now folded
into the models-layer Residency manager, #366).

Validates sharing + refcount + accounting with lightweight REAL objects: a fake
"pipeline" is a plain object carrying a ``.components`` dict, exercising the
same code paths a diffusers pipeline would (no GPU needed).
"""

from __future__ import annotations

import pytest

from gen_worker import HF, Civitai, Hub
from gen_worker.models import (
    LoadedComponentKey,
    Residency,
    Tier,
    build_function_owned_pipeline,
)

_GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Lightweight real fakes                                                        #
# --------------------------------------------------------------------------- #


class _FakeModule:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakePipeline:
    """A stand-in diffusers pipeline: a container of component references."""

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


def _residency_100gb() -> Residency:
    # Fixed budget so accounting is deterministic without a GPU.
    return Residency(vram_budget_bytes=100 * _GiB)


# --------------------------------------------------------------------------- #
# LoadedComponentKey canonicalization                                           #
# --------------------------------------------------------------------------- #


def test_identical_bindings_produce_equal_keys() -> None:
    a = HF("bfl/flux.2-klein-4b", dtype="bf16")
    b = HF("bfl/flux.2-klein-4b", dtype="bf16")
    ka = LoadedComponentKey.from_binding(a, device_id=0, component_set="pkg.Flux2KleinPipeline")
    kb = LoadedComponentKey.from_binding(b, device_id=0, component_set="pkg.Flux2KleinPipeline")
    assert ka == kb
    assert ka.cache_id() == kb.cache_id()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: HF(b.ref, dtype="fp16"),
        lambda b: HF(b.ref, dtype=b.dtype, revision="deadbeef"),
        lambda b: HF(b.ref, dtype=b.dtype, subfolder="text_encoder"),
    ],
)
def test_binding_attribute_differences_do_not_share(mutate) -> None:
    base = HF("bfl/flux.2-klein-4b", dtype="bf16")
    k_base = LoadedComponentKey.from_binding(base, device_id=0, component_set="P")
    k_other = LoadedComponentKey.from_binding(mutate(base), device_id=0, component_set="P")
    assert k_base != k_other
    assert k_base.cache_id() != k_other.cache_id()


def test_device_dtype_quant_component_set_differences_do_not_share() -> None:
    b = HF("bfl/flux.2-klein-4b", dtype="bf16")
    k0 = LoadedComponentKey.from_binding(b, device_id=0, component_set="P")
    assert k0 != LoadedComponentKey.from_binding(b, device_id=1, component_set="P")
    assert k0 != LoadedComponentKey.from_binding(b, device_id=0, component_set="Q")
    assert k0 != LoadedComponentKey.from_binding(
        b, device_id=0, component_set="P", quantization="fp8"
    )
    assert LoadedComponentKey.from_binding(
        b, device_id=0, component_set="P", quantization="fp8", quant_config={"a": 1}
    ) != LoadedComponentKey.from_binding(
        b, device_id=0, component_set="P", quantization="fp8", quant_config={"a": 2}
    )
    assert LoadedComponentKey.from_binding(Hub("o/r"), device_id=0) != LoadedComponentKey.from_binding(
        Civitai("123"), device_id=0
    )


def test_unpinned_revision_falls_back_to_snapshot_digest() -> None:
    b = HF("bfl/flux.2-klein-4b")  # no explicit revision
    same = LoadedComponentKey.from_binding(b, snapshot_digest="/cache/snap-A")
    same2 = LoadedComponentKey.from_binding(b, snapshot_digest="/cache/snap-A")
    diff = LoadedComponentKey.from_binding(b, snapshot_digest="/cache/snap-B")
    assert same == same2
    assert same != diff


# --------------------------------------------------------------------------- #
# Sharing + refcount + single-count accounting                                  #
# --------------------------------------------------------------------------- #


def test_identical_components_are_shared_one_instance_refcount_two() -> None:
    res = _residency_100gb()
    key = LoadedComponentKey(ref="bfl/flux.2-klein-4b", dtype="bf16", component_set="P")

    base = _new_base()
    loads = {"n": 0}

    def loader() -> _FakePipeline:
        loads["n"] += 1
        return base

    a = res.acquire_shared(key, loader, vram_bytes=20 * _GiB)
    b = res.acquire_shared(key, loader, vram_bytes=20 * _GiB)

    assert a is b is base                 # one shared instance
    assert loads["n"] == 1                # loaded exactly once (hit on 2nd)
    assert res.shared_refcount(key) == 2  # both holders counted
    stats = res.shared_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert len(stats["entries"]) == 1


def test_shared_vram_counted_once() -> None:
    res = _residency_100gb()
    key = LoadedComponentKey(ref="bfl/flux.2-klein-4b", dtype="bf16", component_set="P")
    base = _new_base()

    before = res.free_vram_bytes()
    res.acquire_shared(key, lambda: base, vram_bytes=20 * _GiB)
    after_first = res.free_vram_bytes()
    res.acquire_shared(key, lambda: base, vram_bytes=20 * _GiB)  # 2nd holder, SAME bytes
    after_second = res.free_vram_bytes()

    assert before - after_first == 20 * _GiB
    # The second acquire must NOT book another 20GB — shared bytes counted once.
    assert after_second == after_first
    assert res.shared_refcount(key) == 2


def test_shared_component_not_evicted_while_referenced() -> None:
    res = _residency_100gb()
    key = LoadedComponentKey(ref="shared/base", dtype="bf16", component_set="P")
    base = _new_base()
    res.acquire_shared(key, lambda: base, vram_bytes=20 * _GiB)
    res.acquire_shared(key, lambda: base, vram_bytes=20 * _GiB)  # refcount 2
    cid = key.cache_id()

    # Demand VRAM far beyond budget — the referenced entry must survive.
    res.make_room(150 * _GiB)
    assert res.tier(cid) is Tier.VRAM
    assert res.shared_refcount(key) == 2

    # Explicit unload-style calls are also refused while referenced.
    assert res.release_to_disk(cid) is False
    assert res.evict(cid) is False
    assert res.shared_obj(key) is base


def test_releasing_all_refs_makes_entry_evictable() -> None:
    res = _residency_100gb()
    key = LoadedComponentKey(ref="shared/base", dtype="bf16", component_set="P")
    base = _new_base()
    res.acquire_shared(key, lambda: base, vram_bytes=20 * _GiB)
    res.acquire_shared(key, lambda: base, vram_bytes=20 * _GiB)
    cid = key.cache_id()

    assert res.release_shared(key) == 1     # one holder left
    assert res.evict(cid) is False          # still referenced -> refused
    assert res.release_shared(key) == 0     # last holder released

    assert res.evict(cid) is True
    assert res.tier(cid) is None
    assert res.free_vram_bytes() == 100 * _GiB  # accounting back to baseline
    assert res.shared_obj(key) is None


def test_drain_skips_referenced_then_force_clears_all() -> None:
    res = _residency_100gb()
    held = LoadedComponentKey(ref="a/held", component_set="P")
    free = LoadedComponentKey(ref="b/free", component_set="P")
    res.acquire_shared(held, _new_base, vram_bytes=5 * _GiB)   # refcount 1
    res.acquire_shared(free, _new_base, vram_bytes=5 * _GiB)
    res.release_shared(free)                                    # refcount 0

    assert res.drain_shared() == 1          # only the unreferenced entry
    assert res.tier(held.cache_id()) is not None
    assert res.tier(free.cache_id()) is None

    assert res.drain_shared(force=True) == 1  # worker shutdown clears all
    assert res.tier(held.cache_id()) is None
    assert res.free_vram_bytes() == 100 * _GiB


# --------------------------------------------------------------------------- #
# Mutable-boundary isolation (LoRA / override never alias the clean base)       #
# --------------------------------------------------------------------------- #


def test_lora_and_override_bindings_isolate_from_clean_base() -> None:
    base = HF("bfl/flux.2-klein-4b", dtype="bf16")

    k_clean = LoadedComponentKey.from_binding(base, component_set="P")
    k_lora = LoadedComponentKey.from_binding(base, component_set="P", adapter_id="lora:set-7")
    k_override = LoadedComponentKey.from_binding(base, component_set="P", adapter_id="override:Pipe")

    assert k_clean != k_lora
    assert k_clean != k_override
    assert k_lora != k_override
    assert len({k_clean.cache_id(), k_lora.cache_id(), k_override.cache_id()}) == 3


def test_adapter_identity_separates_two_lora_overlays() -> None:
    b = HF("bfl/flux.2-klein-4b", dtype="bf16")
    k1 = LoadedComponentKey.from_binding(b, component_set="P", adapter_id="lora:A")
    k2 = LoadedComponentKey.from_binding(b, component_set="P", adapter_id="lora:B")
    assert k1 != k2


# --------------------------------------------------------------------------- #
# Function-owned pipeline over shared components                                #
# --------------------------------------------------------------------------- #


def test_build_function_owned_pipeline_shares_modules_not_pipeline() -> None:
    base = _new_base()
    fn_pipe = build_function_owned_pipeline(base, _FakePipeline)
    assert fn_pipe is not base
    assert fn_pipe.scheduler is not base.scheduler
    assert fn_pipe.components["vae"] is base.components["vae"]
    assert fn_pipe.components["text_encoder"] is base.components["text_encoder"]


def test_build_function_owned_pipeline_assembles_from_components_dict() -> None:
    class _NoFromPipe:
        def __init__(self, **components: object) -> None:
            self.components = dict(components)

    base = _new_base()
    fn = build_function_owned_pipeline(base, _NoFromPipe)
    assert isinstance(fn, _NoFromPipe)
    assert fn.components["vae"] is base.components["vae"]
