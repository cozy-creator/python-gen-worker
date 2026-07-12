"""Shared-component identity + single-count VRAM accounting (#335/#366),
re-keyed by CONTENT (gw#479): identity = the component's file digest set +
load-affecting facts. ref/revision are labels, not identity.

Validates sharing + refcount + accounting with lightweight REAL objects: a fake
"pipeline" is a plain object carrying a ``.components`` dict, exercising the
same code paths a diffusers pipeline would (no GPU needed).
"""

from __future__ import annotations

from gen_worker import HF, Hub
from gen_worker.models import (
    LoadedComponentKey,
    Residency,
    Tier,
    content_set_digest,
)

_GiB = 1024 ** 3

_FILES_A = {"model-00001.safetensors": "b3" + "a" * 62, "config.json": "b3" + "c" * 62}
_FILES_B = {"model-00001.safetensors": "b3" + "d" * 62, "config.json": "b3" + "c" * 62}


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


def _key(files=None, **kw) -> LoadedComponentKey:
    return LoadedComponentKey.for_component(
        content_digest=content_set_digest(files if files is not None else _FILES_A),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Content-keyed canonicalization (gw#479)                                       #
# --------------------------------------------------------------------------- #


def test_byte_identical_components_share_across_refs() -> None:
    """THE gw#479 property: two different Hub refs whose component files are
    byte-identical (same blake3 set) produce EQUAL keys — one loaded copy."""
    a = Hub("tensorhub/qwen-image")
    b = Hub("tensorhub/qwen-image-edit-2511")
    ka = _key(component="text_encoder", binding=a)
    kb = _key(component="text_encoder", binding=b)
    assert ka == kb
    assert ka.cache_id() == kb.cache_id()
    # The readable label differs but is non-identity (compare=False).
    assert ka.label != kb.label


def test_content_difference_does_not_share() -> None:
    b = Hub("o/r")
    assert _key(_FILES_A, component="text_encoder", binding=b) != _key(
        _FILES_B, component="text_encoder", binding=b
    )


def test_content_set_digest_is_order_invariant_and_path_sensitive() -> None:
    reordered = dict(reversed(list(_FILES_A.items())))
    assert content_set_digest(_FILES_A) == content_set_digest(reordered)
    moved = {"sub/" + p: d for p, d in _FILES_A.items()}
    assert content_set_digest(_FILES_A) != content_set_digest(moved)
    assert content_set_digest({}) == ""


def test_load_fact_differences_do_not_share() -> None:
    k0 = _key(component="text_encoder", dtype="bf16")
    assert k0 != _key(component="text_encoder", dtype="fp16")
    assert k0 != _key(component="text_encoder", dtype="bf16", device_id=1)
    assert k0 != _key(component="vae", dtype="bf16")
    assert k0 != _key(component="text_encoder", dtype="bf16", quantization="fp8")
    assert _key(
        component="text_encoder", quantization="fp8", quant_config={"a": 1}
    ) != _key(component="text_encoder", quantization="fp8", quant_config={"a": 2})
    assert k0 != _key(component="text_encoder", dtype="bf16", placement="offload")


def test_binding_dtype_and_storage_dtype_enter_identity() -> None:
    bf16 = HF("o/r", dtype="bf16")
    fp16 = HF("o/r", dtype="fp16")
    assert _key(component="te", binding=bf16) != _key(component="te", binding=fp16)
    fp8 = Hub("o/r", flavor="fp8", storage_dtype="fp8+te")
    bare = Hub("o/r")
    assert _key(component="te", binding=fp8) != _key(component="te", binding=bare)


def test_ref_is_not_identity() -> None:
    """Same bytes + same facts under different providers/refs: SHARED."""
    assert _key(component="te", binding=HF("mirror-one/repo", dtype="bf16")) == _key(
        component="te", binding=HF("mirror-two/other", dtype="bf16")
    )


# --------------------------------------------------------------------------- #
# Sharing + refcount + single-count accounting                                  #
# --------------------------------------------------------------------------- #


def test_identical_components_are_shared_one_instance_refcount_two() -> None:
    res = _residency_100gb()
    key = _key(component="P", dtype="bf16", label="bfl/flux.2-klein-4b/P")

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
    key = _key(component="P", dtype="bf16")
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
    key = _key(component="P", dtype="bf16", label="shared/base")
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
    key = _key(component="P", dtype="bf16", label="shared/base")
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
    held = _key(_FILES_A, component="P", label="a/held")
    free = _key(_FILES_B, component="P", label="b/free")
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
    k_clean = _key(component="P", dtype="bf16")
    k_lora = _key(component="P", dtype="bf16", adapter_id="lora:set-7")
    k_override = _key(component="P", dtype="bf16", adapter_id="override:Pipe")

    assert k_clean != k_lora
    assert k_clean != k_override
    assert k_lora != k_override
    assert len({k_clean.cache_id(), k_lora.cache_id(), k_override.cache_id()}) == 3


def test_adapter_identity_separates_two_lora_overlays() -> None:
    k1 = _key(component="P", adapter_id="lora:A")
    k2 = _key(component="P", adapter_id="lora:B")
    assert k1 != k2
