"""pgw#1644 — a tensor the container never named must still LAND on the target.

The wall this closes cost a rental. `minimax-h3` 0.12.9 loaded cleanly on an
H200 — full ~105 GB fetch, quantizer swap, retie, zero meta survivors — and then
died 8 ms into compute with

    RuntimeError: Expected all tensors to be on the same device, but got mat1 is
    on cpu, different from other tensors on cuda:0 (... wrapper_CUDA_addmm)

because three RoPE buffers (`inv_freq`, `original_inv_freq`) were built by
`__init__` on the CPU. They are NON-PERSISTENT, so they are in neither the
container nor `state_dict`: the stream never named them and `meta_survivors`
never saw them.

Two defects, both covered here:

1. `_place_uninstalled` (pgw#1454) already existed to move exactly these, but
   read `pipeline.components` and fell back to the pipeline itself only when it
   was an `nn.Module`. A MODULAR pipeline is neither, so the sweep was a silent
   no-op — a fix that was present and unreachable.
2. Nothing ever asserted the result. `meta_survivors` asks "was it filled",
   which is not "did it land".

CPU-only by construction: every assertion here is about which module gets
WALKED and which tensors get NAMED, and neither needs a card.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker.serving.streaming import skeleton as sk  # noqa: E402
from gen_worker.serving.streaming.engine import StreamingLoader  # noqa: E402


class _RoPEish(torch.nn.Module):
    """A module shaped like the one that actually failed: a real parameter the
    container carries, plus a NON-PERSISTENT buffer it does not."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4))
        self.register_buffer("inv_freq", torch.arange(4.0), persistent=False)


class _ModularPipelineish:
    """Deliberately NOT an ``nn.Module`` — this is the whole point.

    ``MiniMaxH3StreamingPipeline`` is a ``ModularPipeline``/``ConfigMixin``;
    `issubclass(..., torch.nn.Module)` is False. The old root discovery could
    not reach a pipeline of this shape at all.
    """

    def __init__(self, components: dict) -> None:
        self.components = components


class _Built:
    def __init__(self, modules: dict, pipeline: object) -> None:
        self.modules = modules
        self.pipeline = pipeline


def _loader() -> StreamingLoader:
    return StreamingLoader.__new__(StreamingLoader)


def test_the_non_persistent_buffer_really_is_invisible_to_the_old_checks():
    """The premise, asserted rather than remembered: this buffer is in no
    state_dict, so nothing name-driven can see it."""
    m = _RoPEish()
    assert "inv_freq" not in set(m.state_dict())
    assert "inv_freq" in dict(m.named_buffers())
    assert sk.meta_survivors(m) == ()  # nothing on meta, yet it is misplaced


def test_roots_reach_a_modular_pipeline_that_is_not_an_nn_module():
    """DEFECT 1. A modular pipeline exposes components but is not an
    ``nn.Module``, so neither of the old arms produced a root."""
    m = _RoPEish()
    built = _Built({"text_encoder": m}, _ModularPipelineish({}))
    assert not isinstance(built.pipeline, torch.nn.Module)

    roots = _loader()._module_roots(built)
    assert m in roots, "the skeleton's own module map must be swept"


def test_roots_also_keep_passthrough_components_and_never_double_count():
    passthrough = _RoPEish()
    built_m = _RoPEish()
    built = _Built(
        {"text_encoder": built_m},
        _ModularPipelineish({"text_encoder": built_m, "vae": passthrough}),
    )
    roots = _loader()._module_roots(built)
    assert built_m in roots and passthrough in roots
    assert len(roots) == 2, "a component in both maps must be swept once"


def test_off_target_names_the_stray_and_ignores_meta():
    """DEFECT 2. `off_target` answers the question `meta_survivors` does not."""
    m = _RoPEish()
    stray = sk.off_target(m, torch.device("meta"))
    names = {n for n, _ in stray}
    assert "inv_freq" in names and "weight" in names

    with torch.device("meta"):
        on_meta = _RoPEish()
    # a tensor still on meta is meta_survivors' refusal to make, not this one's
    assert sk.off_target(on_meta, torch.device("meta")) == ()


def test_off_target_is_INDEX_TOLERANT_so_it_cannot_refuse_a_healthy_load():
    """A fence that fires on correct input is worse than the defect it guards.

    Callers pass a bare "cuda"; the stream lands tensors on "cuda:0". Comparing
    devices with `!=` would report every healthy tensor as stray.
    """
    m = _RoPEish()
    assert torch.device("cpu") != torch.device("cpu", 0)  # the trap, stated
    assert sk.off_target(m, torch.device("cpu")) == ()
    assert sk.off_target(m, torch.device("cpu", 0)) == ()


def test_the_fence_raises_when_something_did_not_land():
    m = _RoPEish()
    built = _Built({"text_encoder": m}, _ModularPipelineish({}))
    with pytest.raises(Exception) as excinfo:
        _loader()._assert_on_target(built, torch.device("meta_is_not_a_device"))
    assert "inv_freq" in str(excinfo.value) or "device" in str(excinfo.value).lower()


def test_the_fence_is_a_no_op_when_the_target_is_meta():
    """A meta target means nothing was meant to land; the survivor check owns
    that case and this one must not double-report it."""
    m = _RoPEish()
    built = _Built({"text_encoder": m}, _ModularPipelineish({}))
    _loader()._assert_on_target(built, torch.device("meta"))  # must not raise


def test_the_sweep_walks_the_skeleton_map_and_moves_the_buffer():
    """End to end on one device: the sweep must REACH the buffer through a
    modular pipeline. Placement onto a second device needs a card and is proven
    by the refire, but reachability — the actual defect — is proven here."""
    m = _RoPEish()
    built = _Built({"text_encoder": m}, _ModularPipelineish({}))
    seen = {}
    original = torch.Tensor.to

    def spy(self, *a, **k):
        seen["called"] = True
        return original(self, *a, **k)

    torch.Tensor.to = spy  # type: ignore[assignment]
    try:
        _loader()._place_uninstalled(built, torch.device("cpu", 0))
    finally:
        torch.Tensor.to = original  # type: ignore[assignment]
    assert seen.get("called"), "the sweep never reached the buffer"
    _loader()._assert_on_target(built, torch.device("cpu"))
