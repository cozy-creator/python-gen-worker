"""#326: SDK path for ``accelerator='none'`` (CPU-only endpoints).

Small non-AR flow-matching audio models like Kokoro-82M
(https://huggingface.co/hexgrad/Kokoro-82M) and other CPU-fast tenants need
the SDK to handle the no-accelerator case end-to-end:

  * Decoration validates ``Resources(accelerator='none')`` cleanly.
  * Discovery emits ``accelerator='none'`` (with ``gpu_count=0``-equivalent
    semantics: no GPU axis is required) into the endpoint manifest.
  * The decorator rejects the legacy oxymoronic aliases ``'cpu'`` / ``'gpu'``
    (Part 1 of #326 — CPU is the *absence* of an accelerator, not one).
  * Importing the SDK on a CPU-only machine doesn't trip ``torch.cuda.*``
    lookups during decoration or discovery — the worker has to be able to
    load a tenant module on a no-GPU host long enough to publish.

This file is the deliverable validating that the path is shippable.
"""

from __future__ import annotations

import sys
from typing import Any

import msgspec
import pytest

from gen_worker import RequestContext, Resources, inference
from gen_worker.discovery.discover import _extract_class_function_methods


# ============================================================================
# Fixtures — a minimal Kokoro-shaped CPU-only endpoint.
# ============================================================================


class _TTSInput(msgspec.Struct):
    text: str
    voice: str = "default"


class _TTSOutput(msgspec.Struct):
    audio_bytes: bytes


def _make_cpu_only_class():
    """Return a fresh CPU-only SerialWorker endpoint class.

    Modeled on Kokoro-82M: a small non-AR flow-matching TTS that runs
    real-time-or-faster on CPU. The point of the test is the
    ``accelerator='none'`` declaration — the body is intentionally trivial.
    """

    @inference(
        label="kokoro-cpu",
        description="CPU-only TTS (Kokoro-style)",
        resources=Resources(accelerator="none"),
    )
    class KokoroCPU:
        def setup(self) -> None:
            pass

        @inference.function
        def synthesize(self, ctx: RequestContext, payload: _TTSInput) -> _TTSOutput:
            return _TTSOutput(audio_bytes=b"")

        def shutdown(self) -> None:
            pass

    return KokoroCPU


# ============================================================================
# (1) Decoration discovers correctly for accelerator='none'.
# ============================================================================


def test_cpu_only_class_decoration_attaches_metadata() -> None:
    """``@inference(resources=Resources(accelerator='none'))`` must decorate
    cleanly and surface the expected archetype + kind."""
    cls = _make_cpu_only_class()

    spec = getattr(cls, "__gen_worker_endpoint_spec__")
    assert spec.kind == "inference"
    assert spec.resources.accelerator == "none"
    # accelerator='none' must NOT auto-flip requires_gpu (that's the cuda path).
    assert spec.resources.requires_gpu is None

    # Without async hooks, this is a SerialWorker class.
    assert getattr(cls, "__gen_worker_archetype__") == "SerialWorker"

    methods = getattr(cls, "__gen_worker_function_methods__")
    assert len(methods) == 1
    name, _method, fn_spec = methods[0]
    assert name == "synthesize"
    assert fn_spec.name == "synthesize"


# ============================================================================
# (2) Manifest serialization — accelerator='none', no gpu_* axes leak in.
# ============================================================================


def test_cpu_only_class_manifest_emits_accelerator_none() -> None:
    """The discovered manifest entry must carry ``accelerator='none'`` so the
    orchestrator can route the endpoint to a CPU pod (and so the build can
    pick a CPU base image)."""
    cls = _make_cpu_only_class()
    cls.__module__ = "test_no_accelerator_mod"

    entries = _extract_class_function_methods(cls, "test_no_accelerator_mod")
    assert len(entries) == 1
    entry = entries[0]

    res = entry["resources"]
    assert res.get("accelerator") == "none"
    # Resources uses omit_defaults, so a no-cuda endpoint leaves gpu_count
    # (and friends) absent from the manifest — they default to 0 / None /
    # absent on the consumer side. The key invariant: the GPU axes must NOT
    # appear with a non-zero value.
    assert int(res.get("gpu_count", 0) or 0) == 0
    assert res.get("min_compute_capability") is None or res["min_compute_capability"] == 0
    assert res.get("min_vram_gb") is None or float(res["min_vram_gb"] or 0) == 0.0
    # requires_gpu must not be auto-set for the none path.
    assert not res.get("requires_gpu", False)


def test_cpu_only_resources_wire_shape_omits_gpu_axes() -> None:
    """``msgspec.to_builtins(Resources(accelerator='none'))`` should produce
    a minimal blob — accelerator only, no spurious GPU fields."""
    r = Resources(accelerator="none")
    wire = msgspec.to_builtins(r)
    assert wire == {"accelerator": "none"}


# ============================================================================
# (3) Part 1: 'cpu' and 'gpu' aliases are rejected with a clear ValueError.
# ============================================================================


def test_resources_rejects_cpu_alias_with_pointer_to_none() -> None:
    with pytest.raises(ValueError) as exc:
        Resources(accelerator="cpu")  # type: ignore[arg-type]
    msg = str(exc.value)
    assert "accelerator='cpu' is invalid" in msg
    assert "accelerator='none'" in msg
    # Mention the canonical vocabulary so the tenant sees the fix in the message.
    assert "'cuda'" in msg and "'none'" in msg


def test_resources_rejects_gpu_alias_with_pointer_to_cuda() -> None:
    with pytest.raises(ValueError) as exc:
        Resources(accelerator="gpu")  # type: ignore[arg-type]
    msg = str(exc.value)
    assert "accelerator='gpu' is invalid" in msg
    assert "accelerator='cuda'" in msg


def test_resources_rejects_uppercase_cpu_alias() -> None:
    """The check should be case-insensitive — 'CPU' is the same typo."""
    with pytest.raises(ValueError, match="'CPU'"):
        Resources(accelerator="CPU")  # type: ignore[arg-type]


def test_resources_rejects_uppercase_gpu_alias() -> None:
    with pytest.raises(ValueError, match="'GPU'"):
        Resources(accelerator="GPU")  # type: ignore[arg-type]


def test_decorator_propagates_cpu_alias_rejection() -> None:
    """The ValueError fires at ``Resources(...)`` construction inside the
    tenant module — discovery never even sees it because the decorator
    can't be invoked with a bad Resources value."""
    with pytest.raises(ValueError, match="accelerator='cpu' is invalid"):

        @inference(resources=Resources(accelerator="cpu"))  # type: ignore[arg-type]
        class BadCPU:
            def setup(self) -> None:
                pass

            @inference.function
            def f(self, ctx: RequestContext, payload: _TTSInput) -> _TTSOutput:
                return _TTSOutput(audio_bytes=b"")

            def shutdown(self) -> None:
                pass


def test_decorator_propagates_gpu_alias_rejection() -> None:
    with pytest.raises(ValueError, match="accelerator='gpu' is invalid"):

        @inference(resources=Resources(accelerator="gpu"))  # type: ignore[arg-type]
        class BadGPU:
            def setup(self) -> None:
                pass

            @inference.function
            def f(self, ctx: RequestContext, payload: _TTSInput) -> _TTSOutput:
                return _TTSOutput(audio_bytes=b"")

            def shutdown(self) -> None:
                pass


# ============================================================================
# (4) No torch.cuda.* invocations during decoration / discovery for the
#     accelerator='none' path. A tenant publishing from a CPU-only host
#     (or our own CI on a machine without CUDA) must not be tripped up.
# ============================================================================


class _CudaTripwire:
    """Replacement for ``torch.cuda`` that records any access.

    Replacing ``torch.cuda`` with one of these gives us a paper trail: every
    attribute lookup on ``torch.cuda`` becomes a recorded entry, and we then
    assert the recorded list is empty for the no-accelerator path.
    """

    def __init__(self) -> None:
        self.accesses: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.accesses.append(name)
        # Defensive: behave like a "no CUDA" stub so any defensive call site
        # (e.g. ``torch.cuda.is_available()``) gets a benign False rather
        # than crashing — we still record the access for the assertion.
        if name == "is_available":
            return lambda: False
        if name == "device_count":
            return lambda: 0
        raise AttributeError(name)


def test_accelerator_none_path_does_not_trigger_torch_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoration + discovery for a ``Resources(accelerator='none')`` class
    must not touch ``torch.cuda``. If torch isn't installed at all, we still
    have to be able to decorate + discover — the import-time torch usage in
    ``gen_worker/__init__`` is already guarded, but this test pins the
    guarantee so a future regression flips a tripwire."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        # torch isn't installed in this environment; the no-cuda path is
        # trivially satisfied. The test still has value when torch IS
        # installed (the common case in dev), so we mark it skip-friendly.
        pytest.skip("torch not installed — no-cuda invariant is trivially true")

    tripwire = _CudaTripwire()
    monkeypatch.setattr(torch_mod, "cuda", tripwire, raising=False)

    # Decorate a fresh class (build the spec) under the tripwire.
    cls = _make_cpu_only_class()
    cls.__module__ = "test_no_accelerator_tripwire_mod"

    # Discover its manifest entry under the tripwire.
    entries = _extract_class_function_methods(cls, "test_no_accelerator_tripwire_mod")
    assert len(entries) == 1
    assert entries[0]["resources"]["accelerator"] == "none"

    # Neither decoration nor discovery should have hit torch.cuda.
    assert tripwire.accesses == [], (
        "torch.cuda was accessed during accelerator='none' decoration/discovery: "
        f"{tripwire.accesses!r}"
    )


# ============================================================================
# (5) Sanity: Resources(accelerator=None) — fully unset — is also valid.
# ============================================================================


def test_resources_accelerator_unset_stays_none() -> None:
    """``Resources()`` without an accelerator kwarg must default to None
    (the scheduler treats it the same as 'none')."""
    r = Resources()
    assert r.accelerator is None
    assert r.requires_gpu is None


def test_resources_empty_string_accelerator_normalizes_to_none() -> None:
    """An empty string is treated as 'unset' (normalizes to None)."""
    r = Resources(accelerator="")  # type: ignore[arg-type]
    assert r.accelerator is None


# ============================================================================
# (6) Self-contradiction gate: accelerator='none' + GPU resource axes
#     must raise at decoration time (#326 polish for 0.7.8). CPU-only
#     endpoints stay legal — the gate only fires when the tenant declares
#     both no-accelerator AND GPU resources in the same Resources call.
# ============================================================================


def test_accelerator_none_with_requires_gpu_raises() -> None:
    """accelerator='none' + requires_gpu=True is self-contradictory."""
    with pytest.raises(ValueError) as exc:
        Resources(accelerator="none", requires_gpu=True)
    msg = str(exc.value)
    assert "Resources(accelerator='none')" in msg
    assert "requires_gpu=" in msg
    # Error must point at both remediations.
    assert "'cuda'" in msg
    assert "drop the GPU resource axes" in msg


def test_accelerator_none_with_min_vram_gb_raises() -> None:
    """accelerator='none' + min_vram_gb=4 is self-contradictory."""
    with pytest.raises(ValueError) as exc:
        Resources(accelerator="none", min_vram_gb=4.0)
    msg = str(exc.value)
    assert "Resources(accelerator='none')" in msg
    assert "min_vram_gb=" in msg


def test_accelerator_none_with_min_compute_capability_raises() -> None:
    """accelerator='none' + min_compute_capability=8.0 is self-contradictory."""
    with pytest.raises(ValueError) as exc:
        Resources(accelerator="none", min_compute_capability=8.0)
    msg = str(exc.value)
    assert "Resources(accelerator='none')" in msg
    assert "min_compute_capability=" in msg


def test_accelerator_none_with_multiple_gpu_axes_lists_all() -> None:
    """When several GPU axes leak into a 'none' Resources, the error
    should enumerate every offender (so the tenant can fix the whole
    block in one pass instead of one-at-a-time)."""
    with pytest.raises(ValueError) as exc:
        Resources(
            accelerator="none",
            requires_gpu=True,
            min_vram_gb=8.0,
            min_compute_capability=8.0,
        )
    msg = str(exc.value)
    assert "requires_gpu=" in msg
    assert "min_vram_gb=" in msg
    assert "min_compute_capability=" in msg


def test_accelerator_none_alone_passes() -> None:
    """Regression guard: CPU-only endpoints (the whole point of #326)
    must continue to construct without raising. The gate must not block
    valid configurations."""
    r = Resources(accelerator="none")
    assert r.accelerator == "none"
    assert r.requires_gpu is None
    assert r.min_vram_gb is None
    assert r.min_compute_capability is None


def test_accelerator_cuda_with_gpu_count_via_requires_gpu_passes() -> None:
    """accelerator='cuda' + requires_gpu=True is the canonical GPU
    declaration. (Resources doesn't have a literal ``gpu_count`` field —
    the GPU-count axis is expressed via requires_gpu + min_vram_gb.)"""
    r = Resources(accelerator="cuda", requires_gpu=True)
    assert r.accelerator == "cuda"
    assert r.requires_gpu is True


def test_accelerator_cuda_with_vram_only_passes() -> None:
    """accelerator='cuda' + min_vram_gb=24 (no explicit requires_gpu) is
    fine — the accelerator='cuda' path auto-sets requires_gpu=True."""
    r = Resources(accelerator="cuda", min_vram_gb=24.0)
    assert r.accelerator == "cuda"
    # The 'cuda' path auto-flips requires_gpu when it wasn't supplied.
    assert r.requires_gpu is True
    assert r.min_vram_gb == 24.0


def test_accelerator_none_with_requires_gpu_false_passes() -> None:
    """``requires_gpu=False`` is an explicit 'no GPU' declaration and is
    consistent with accelerator='none' — the gate only fires on the
    True path."""
    r = Resources(accelerator="none", requires_gpu=False)
    assert r.accelerator == "none"
    assert r.requires_gpu is False


def test_decorator_propagates_none_with_gpu_axes_rejection() -> None:
    """The self-contradiction gate must fire under the @inference
    decorator path too — tenants see the error during import."""
    with pytest.raises(ValueError, match="Resources\\(accelerator='none'\\)"):

        @inference(resources=Resources(accelerator="none", min_vram_gb=8.0))
        class BadMixed:
            def setup(self) -> None:
                pass

            @inference.function
            def f(self, ctx: RequestContext, payload: _TTSInput) -> _TTSOutput:
                return _TTSOutput(audio_bytes=b"")

            def shutdown(self) -> None:
                pass
