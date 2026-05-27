"""Model-selectable endpoints: SharedBase + per-request variant dispatch,
3-tier residency, partial readiness, tier-aware availability (issue #337).

Covers the gen-worker SDK contract for endpoints that expose a SET of
selectable fine-tunes sharing a frozen base stack:

  1. SharedBase + .variant() API shape + Variant identity.
  2. dispatch() table accepting Variant / plain Repo / LoRA-overlay Repo.
  3. Discovery-time guard: a dispatch slot named as a setup() param is a
     BUILD-TIME error; a fixed Repo slot may be a setup() param.
  4. Per-request slot resolution + HANDLER injection wiring.
  5. Shared-by-reference assembly: ONE shared component object across variants.
  6. 3-tier ModelCache: VRAM<->CPU(RAM)<->DISK promote/evict ordering + pinning.
  7. Component-compatibility mismatch warning.
  8. Partial readiness (a variant serveable as it lands; others absent).
  9. Tier-aware availability emission (debounced on transition).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal, Optional

import msgspec
import pytest

from gen_worker import (
    HFRepo,
    Repo,
    RequestContext,
    SharedBase,
    Variant,
    dispatch,
    inference,
)
from gen_worker._worker_support import _SerialWorkerSpec
from gen_worker.models.cache import ModelCache, ModelLocation
from gen_worker.worker import Worker


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakePipeline:
    """Stand-in diffusers pipeline: a container of component references."""

    def __init__(self, **components: Any) -> None:
        self.components = components
        self.device = "cpu"

    def to(self, device: str) -> "_FakePipeline":
        self.device = device
        return self

    def __call__(self, **kwargs: Any) -> Any:  # pragma: no cover - not run
        raise NotImplementedError


class _FakeModule:
    """Stand-in nn.Module-ish component (text_encoder / vae / unet)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.device = "cpu"

    def to(self, device: str) -> "_FakeModule":
        self.device = device
        return self


class _FakeCtx:
    """Minimal request-context stand-in carrying just a device for assembly."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device


class GenInput(msgspec.Struct):
    prompt: str
    model: Literal["illustrious", "animagine"] = "illustrious"


class GenOutput(msgspec.Struct):
    result: str


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    w._request_specs = {}
    w._training_specs = {}
    w._batched_specs = {}
    w._batched_instances = []
    w._serial_class_specs = {}
    w._serial_class_instances = []
    w._discovered_resources = {}
    w._function_schemas = {}
    w._batched_loop = None
    w._batched_loop_thread = None
    w._batched_inflight_lock = threading.Lock()
    w._batched_inflight = {}
    w._micro_batch_aggregators = {}
    w.scheduler_addr = ""
    w.worker_id = "test"
    w._model_cache = ModelCache(max_vram_gb=10.0)
    w._downloader = None
    w._shared_base_components = {}
    w._last_residency_tier = {}
    w._emitted = []
    # Capture availability emissions.
    w._emit_model_ready = lambda model_id, kind: w._emitted.append((model_id, kind))  # type: ignore
    return w


# --------------------------------------------------------------------------
# 1. SharedBase + .variant() API
# --------------------------------------------------------------------------


def test_sharedbase_variant_builds_variant_with_shared_and_slot_refs() -> None:
    base = SharedBase(
        _FakePipeline,
        text_encoder=HFRepo("org/sdxl").subfolder("text_encoder"),
        vae=HFRepo("madebyollin/sdxl-vae-fp16-fix"),
    )
    v = base.variant(unet=HFRepo("org/illustrious"))
    assert isinstance(v, Variant)
    assert isinstance(v, Repo)  # composes with dispatch()
    # Identity follows the primary swap slot (the unet).
    assert v.ref == "org/illustrious"
    assert v.primary_slot_name == "unet"
    assert set(v.shared_components.keys()) == {"text_encoder", "vae"}
    assert set(v.variant_slots.keys()) == {"unet"}
    assert v.pipeline_class_fqn.endswith("_FakePipeline")


def test_sharedbase_variant_requires_a_slot() -> None:
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))
    with pytest.raises(ValueError):
        base.variant()


def test_sharedbase_rejects_non_repo_component() -> None:
    with pytest.raises(TypeError):
        SharedBase(_FakePipeline, vae="not-a-repo")  # type: ignore[arg-type]


def test_hfrepo_subfolder_chainable() -> None:
    r = HFRepo("org/repo").subfolder("text_encoder")
    assert r._subfolder == "text_encoder"
    # original unchanged (immutable)
    assert HFRepo("org/repo")._subfolder == ""


# --------------------------------------------------------------------------
# 2. dispatch() table accepts Variant / plain Repo / LoRA overlay
# --------------------------------------------------------------------------


def test_discovery_manifest_records_shared_and_variant_refs() -> None:
    """The endpoint lock records each variant's shared-component refs + its
    per-variant swap-slot refs (issue #337)."""
    from gen_worker.discovery.discover import _binding_to_manifest

    base = SharedBase(
        _FakePipeline,
        text_encoder=HFRepo("org/sdxl").subfolder("text_encoder"),
        vae=HFRepo("madebyollin/sdxl-vae-fp16-fix"),
    )
    d = dispatch(field="model", table={
        "illustrious": base.variant(unet=HFRepo("org/illustrious")),
        "animagine": base.variant(unet=HFRepo("org/animagine")),
    })
    m = _binding_to_manifest(d, _FakePipeline, "pipeline")
    assert m["kind"] == "dispatch"
    illu = m["table"]["illustrious"]
    assert illu["variant_kind"] == "shared_base_variant"
    assert illu["ref"] == "org/illustrious"  # variant identity = swap slot
    assert illu["shared_components"]["vae"]["ref"] == "madebyollin/sdxl-vae-fp16-fix"
    assert illu["shared_components"]["text_encoder"]["subfolder"] == "text_encoder"
    assert illu["variant_slots"]["unet"]["ref"] == "org/illustrious"


def test_dispatch_table_accepts_variant_repo_and_lora() -> None:
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))
    d = dispatch(
        field="model",
        table={
            "illustrious": base.variant(unet=HFRepo("org/illustrious")),
            "full": HFRepo("org/full-pipeline"),
            "lora": HFRepo("org/lora").allow_lora(),
        },
    )
    assert set(d.table.keys()) == {"illustrious", "full", "lora"}
    assert isinstance(d.table["illustrious"], Variant)
    assert d.table["lora"]._allow_lora is True


# --------------------------------------------------------------------------
# 3. Discovery-time guard
# --------------------------------------------------------------------------


def test_dispatch_slot_in_setup_is_build_error() -> None:
    """The motivating crash: a per-request dispatch slot declared as a setup()
    param must be a clear BUILD-TIME error, not a runtime TypeError."""
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))
    with pytest.raises(ValueError, match="per-request dispatch slot"):

        @inference(
            models={"pipeline": dispatch(field="model", table={
                "illustrious": base.variant(unet=HFRepo("org/illustrious")),
                "animagine": base.variant(unet=HFRepo("org/animagine")),
            })}
        )
        class Bad:
            def setup(self, pipeline):  # noqa: ANN001 - dispatch slot as setup param
                self.pipeline = pipeline

            @inference.function(name="generate")
            def generate(self, ctx: RequestContext, payload: GenInput, pipeline: _FakePipeline) -> GenOutput:
                return GenOutput(result="x")


def test_dispatch_slot_with_bare_setup_is_accepted() -> None:
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))

    @inference(
        models={"pipeline": dispatch(field="model", table={
            "illustrious": base.variant(unet=HFRepo("org/illustrious")),
            "animagine": base.variant(unet=HFRepo("org/animagine")),
        })}
    )
    class Good:
        def setup(self) -> None:
            self.ready = True

        @inference.function(name="generate")
        def generate(self, ctx: RequestContext, payload: GenInput, pipeline: _FakePipeline) -> GenOutput:
            return GenOutput(result=pipeline.components.get("unet", "?"))

    spec = getattr(Good, "__gen_worker_endpoint_spec__")
    assert "pipeline" in spec.models


def test_fixed_repo_slot_still_requires_setup_param() -> None:
    """Fixed (load-once) Repo slots keep the historical contract: must match a
    setup() kwarg."""
    with pytest.raises(ValueError, match="doesn't match any setup"):

        @inference(models={"weights": HFRepo("org/repo")})
        class Bad:
            def setup(self) -> None:  # missing `weights` param
                pass

            @inference.function(name="g")
            def g(self, ctx: RequestContext, payload: GenInput) -> GenOutput:
                return GenOutput(result="x")


# --------------------------------------------------------------------------
# 4. Per-request slot resolution + handler injection wiring
# --------------------------------------------------------------------------


def _make_selectable_class():
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))

    @inference(
        models={"pipeline": dispatch(field="model", table={
            "illustrious": base.variant(unet=HFRepo("org/illustrious")),
            "animagine": base.variant(unet=HFRepo("org/animagine")),
        })}
    )
    class SDXL:
        def setup(self) -> None:
            self.ready = True

        @inference.function(name="generate")
        def generate(self, ctx: RequestContext, payload: GenInput, pipeline: _FakePipeline) -> GenOutput:
            return GenOutput(result=str(pipeline.components.get("unet")))

    return SDXL


def test_registration_records_dispatch_injection_on_serial_spec() -> None:
    cls = _make_selectable_class()
    w = _bare_worker()
    n = w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    assert n == 1
    sspec = w._serial_class_specs["generate"]
    assert isinstance(sspec, _SerialWorkerSpec)
    assert "pipeline" in sspec.dispatch_injections
    assert sspec.dispatch_injections["pipeline"].field == "model"
    # The handler-injected type is recorded for the validation check.
    assert sspec.dispatch_injection_types["pipeline"] is _FakePipeline


def test_resolve_dispatch_injection_assembles_selected_variant(monkeypatch) -> None:
    cls = _make_selectable_class()
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["generate"]

    # Stub component loading so no real download/diffusers is needed.
    def fake_load_component(ctx, repo, kind):  # noqa: ANN001
        return _FakeModule(f"{kind}:{repo.ref}")

    monkeypatch.setattr(w, "_load_component", fake_load_component)
    monkeypatch.setattr(w, "_download_repo_local", lambda ctx, repo: "/tmp/x")

    ctx = _FakeCtx()

    payload = GenInput(prompt="cat", model="animagine")
    kwargs = w._resolve_dispatch_injection_kwargs(ctx, sspec, payload)
    assert "pipeline" in kwargs
    pipe = kwargs["pipeline"]
    assert isinstance(pipe, _FakePipeline)
    # The handler receives the SELECTED variant's unet.
    assert pipe.components["unet"].tag == "unet:org/animagine"
    # And the shared vae.
    assert pipe.components["vae"].tag == "vae:org/vae"


# --------------------------------------------------------------------------
# 5. Shared-by-reference assembly
# --------------------------------------------------------------------------


def test_shared_components_are_the_same_object_across_variants(monkeypatch) -> None:
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"), text_encoder=HFRepo("org/te"))
    v1 = base.variant(unet=HFRepo("org/illustrious"))
    v2 = base.variant(unet=HFRepo("org/animagine"))

    w = _bare_worker()
    loaded = {}

    def fake_load_component(ctx, repo, kind):  # noqa: ANN001
        m = _FakeModule(f"{kind}:{repo.ref}")
        loaded.setdefault(kind, []).append(m)
        return m

    monkeypatch.setattr(w, "_load_component", fake_load_component)
    monkeypatch.setattr(w, "_download_repo_local", lambda ctx, repo: "/tmp/x")

    ctx = _FakeCtx()

    p1 = w._assemble_variant_pipeline(ctx, v1)
    p2 = w._assemble_variant_pipeline(ctx, v2)

    # The shared vae + text_encoder objects are IDENTICAL across variants
    # (loaded once, shared by reference) — only the unet differs.
    assert p1.components["vae"] is p2.components["vae"]
    assert p1.components["text_encoder"] is p2.components["text_encoder"]
    assert p1.components["unet"] is not p2.components["unet"]
    # Each shared component loaded exactly once.
    assert len(loaded["vae"]) == 1
    assert len(loaded["text_encoder"]) == 1
    assert len(loaded["unet"]) == 2


def test_shared_base_is_pinned_in_cache(monkeypatch) -> None:
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))
    v = base.variant(unet=HFRepo("org/illustrious"))
    w = _bare_worker()
    monkeypatch.setattr(w, "_load_component", lambda ctx, repo, kind: _FakeModule(kind))
    monkeypatch.setattr(w, "_download_repo_local", lambda ctx, repo: "/tmp/x")
    ctx = _FakeCtx()
    w._ensure_shared_base(ctx, v)
    key = w._shared_base_cache_key(v)
    assert w._model_cache.is_pinned(key)


# --------------------------------------------------------------------------
# 6. 3-tier ModelCache: VRAM <-> CPU(RAM) <-> DISK promote/evict + pinning
# --------------------------------------------------------------------------


def test_cache_evicts_vram_lru_to_cpu_warm_tier() -> None:
    c = ModelCache(max_vram_gb=10.0)
    c._max_ram_gb = 100.0  # generous RAM budget for the test
    c.mark_loaded_to_vram("a", _FakeModule("a"), 6.0)
    c.mark_loaded_to_vram("b", _FakeModule("b"), 6.0)  # forces eviction of "a"
    # "a" demoted to the CPU warm tier (not dropped to disk).
    assert c.residency_tier("a") == "RAM"
    assert c.residency_tier("b") == "VRAM"
    # The demoted object is moved to cpu but kept.
    assert c._models["a"].pipeline.device == "cpu"


def test_cache_promotes_warm_cpu_model_back_to_vram() -> None:
    c = ModelCache(max_vram_gb=10.0)
    c._max_ram_gb = 100.0
    c.mark_loaded_to_vram("a", _FakeModule("a"), 6.0)
    c.mark_loaded_to_vram("b", _FakeModule("b"), 6.0)  # "a" -> CPU
    assert c.residency_tier("a") == "RAM"
    ok = c._promote_from_cpu("a", device="cuda")
    assert ok
    assert c.residency_tier("a") == "VRAM"
    # Promoting "a" evicted "b" back to CPU (only one 6GB fits in 10GB).
    assert c.residency_tier("b") == "RAM"


def test_cache_pinned_shared_base_is_never_evicted() -> None:
    c = ModelCache(max_vram_gb=10.0)
    c._max_ram_gb = 100.0
    c.mark_loaded_to_vram("shared", _FakeModule("shared"), 4.0, pinned=True)
    # Loading two more 4GB models would normally evict the LRU — but the pinned
    # shared base must stay resident.
    c.mark_loaded_to_vram("v1", _FakeModule("v1"), 4.0)
    c.mark_loaded_to_vram("v2", _FakeModule("v2"), 4.0)
    assert c.residency_tier("shared") == "VRAM"
    assert c.is_pinned("shared")
    # A non-pinned victim was demoted instead.
    assert c.residency_tier("v1") in ("RAM", "VRAM")


def test_cache_residency_map_snapshot() -> None:
    c = ModelCache(max_vram_gb=10.0)
    c._max_ram_gb = 100.0
    c.mark_loaded_to_vram("a", _FakeModule("a"), 4.0)
    c.mark_cached_to_disk("d", __import__("pathlib").Path("/tmp/x"), 2.0)
    m = c.get_residency_map()
    assert m["a"] == "VRAM"
    assert m["d"] == "DISK"


# --------------------------------------------------------------------------
# 7. Component-compatibility mismatch warning
# --------------------------------------------------------------------------


def test_variant_shipping_own_component_warns(tmp_path, caplog) -> None:
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"), text_encoder=HFRepo("org/te"))
    v = base.variant(unet=HFRepo("org/illustrious"))
    w = _bare_worker()
    # The variant's local dir ships its OWN text_encoder/ that differs from base.
    variant_dir = tmp_path / "variant"
    (variant_dir / "text_encoder").mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        w._warn_component_mismatch(v, {"unet": str(variant_dir)})
    assert any("differs from the declared SharedBase" in r.message or "DIFFERS" in r.message for r in caplog.records)


def test_variant_without_own_component_does_not_warn(tmp_path, caplog) -> None:
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))
    v = base.variant(unet=HFRepo("org/illustrious"))
    w = _bare_worker()
    variant_dir = tmp_path / "variant"
    (variant_dir / "unet").mkdir(parents=True)  # only its own unet, fine
    with caplog.at_level(logging.WARNING):
        w._warn_component_mismatch(v, {"unet": str(variant_dir)})
    assert not any("DIFFERS" in r.message or "differs" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# 8. Partial readiness — serve a ready variant; another not-yet-downloaded.
# --------------------------------------------------------------------------


def test_partial_readiness_serves_ready_variant_independent_of_others(monkeypatch) -> None:
    cls = _make_selectable_class()
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["generate"]

    downloaded: list[str] = []

    def fake_download(ctx, repo):  # noqa: ANN001
        downloaded.append(repo.ref)
        return "/tmp/" + repo.ref.replace("/", "_")

    monkeypatch.setattr(w, "_download_repo_local", fake_download)
    monkeypatch.setattr(w, "_load_component", lambda ctx, repo, kind: _FakeModule(f"{kind}:{repo.ref}"))

    ctx = _FakeCtx()

    # Serve "illustrious" — only its unet (+ shared base) downloads. "animagine"
    # is never touched: the function is serveable per-variant as it lands.
    w._resolve_dispatch_injection_kwargs(ctx, sspec, GenInput(prompt="x", model="illustrious"))
    assert "org/illustrious" in downloaded
    assert "org/animagine" not in downloaded


def test_unknown_variant_key_is_a_clear_error() -> None:
    cls = _make_selectable_class()
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["generate"]
    ctx = _FakeCtx()

    class _Bad(msgspec.Struct):
        prompt: str = "x"
        model: str = "does-not-exist"

    with pytest.raises(ValueError, match="not .*in the dispatch table|dispatch table"):
        w._resolve_dispatch_injection_kwargs(ctx, sspec, _Bad())


# --------------------------------------------------------------------------
# 9. Tier-aware availability emission (debounced on transitions)
# --------------------------------------------------------------------------


def test_residency_tier_emission_is_debounced() -> None:
    w = _bare_worker()
    w._emit_residency_tier("org/x", "VRAM")
    w._emit_residency_tier("org/x", "VRAM")  # unchanged -> no second emit
    w._emit_residency_tier("org/x", "RAM")   # changed -> emits
    # VRAM -> kind READY(1); RAM -> kind CACHED(3).
    kinds = [k for (_id, k) in w._emitted]
    assert kinds == [1, 3]


def test_residency_tier_emission_skips_downloading_and_absent() -> None:
    w = _bare_worker()
    w._emit_residency_tier("org/x", "DOWNLOADING")
    w._emit_residency_tier("org/x", "ABSENT")
    assert w._emitted == []


def test_variant_modifiers_preserve_shared_and_slot_fields() -> None:
    """`.allow_lora()` / `.allow_override()` on a Variant must keep the shared
    base + variant slots (the base Repo's dataclasses.replace can't)."""
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))
    v = base.variant(unet=HFRepo("org/illustrious"))
    v2 = v.allow_lora()
    assert isinstance(v2, Variant)
    assert v2._allow_lora is True
    assert set(v2.shared_components) == {"vae"}
    assert set(v2.variant_slots) == {"unet"}
    v3 = v.allow_override(_FakePipeline)
    assert v3._allow_override is True
    assert v3._pipeline_classes


# --------------------------------------------------------------------------
# Spectrum: a dispatch table that mixes a Variant, a standalone full pipeline,
# and a LoRA overlay — all route through the same per-request resolution path.
# --------------------------------------------------------------------------


class MixedInput(msgspec.Struct):
    prompt: str
    model: Literal["variant", "full", "lora"] = "variant"


def _make_mixed_class():
    base = SharedBase(_FakePipeline, vae=HFRepo("org/vae"))

    @inference(
        models={"pipeline": dispatch(field="model", table={
            "variant": base.variant(unet=HFRepo("org/illustrious")),
            "full": HFRepo("org/full-pipeline"),
            "lora": HFRepo("org/lora").allow_lora(),
        })}
    )
    class Mixed:
        def setup(self) -> None:
            pass

        @inference.function(name="generate")
        def generate(self, ctx: RequestContext, payload: MixedInput, pipeline: Any) -> GenOutput:
            return GenOutput(result="ok")

    return Mixed


def test_standalone_full_pipeline_routes_through_injection(monkeypatch) -> None:
    Mixed = _make_mixed_class()
    w = _bare_worker()
    w._register_endpoint_class(Mixed, Mixed.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["generate"]

    seen = {}

    def fake_resolve_injected(ctx, requested_type, model_id, inj):  # noqa: ANN001
        seen["model_id"] = model_id
        seen["allow_lora"] = bool(getattr(inj.binding, "_allow_lora", False))
        return _FakePipeline()

    monkeypatch.setattr(w, "_resolve_injected_value", fake_resolve_injected)
    monkeypatch.setattr(w, "_canonical_cache_key_for_ref", lambda r: r)
    ctx = _FakeCtx()

    # Full standalone pipeline -> routes through _resolve_injected_value.
    w._resolve_dispatch_injection_kwargs(ctx, sspec, MixedInput(prompt="x", model="full"))
    assert seen["model_id"] == "org/full-pipeline"
    assert seen["allow_lora"] is False

    # LoRA overlay -> same path, allow_lora flag preserved.
    w._resolve_dispatch_injection_kwargs(ctx, sspec, MixedInput(prompt="x", model="lora"))
    assert seen["model_id"] == "org/lora"
    assert seen["allow_lora"] is True


def test_variant_resolution_emits_vram_tier(monkeypatch) -> None:
    cls = _make_selectable_class()
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    sspec = w._serial_class_specs["generate"]
    monkeypatch.setattr(w, "_download_repo_local", lambda ctx, repo: "/tmp/x")
    monkeypatch.setattr(w, "_load_component", lambda ctx, repo, kind: _FakeModule(kind))
    ctx = _FakeCtx()
    w._resolve_dispatch_injection_kwargs(ctx, sspec, GenInput(prompt="x", model="illustrious"))
    # The selected variant transitioned DOWNLOADING -> VRAM (READY kind emitted).
    assert any(k == 1 for (_id, k) in w._emitted)
