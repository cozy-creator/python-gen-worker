"""Folding a request's LoRA into the denoiser weights, and undoing it exactly.

The defect these tests exist for, measured on the production seams: a live
adapter on a COMPILED-ARMED denoiser changes the served tensor by exactly
``0.0``. The armed dispatcher replaced ``forward`` with the artifact's
dispatch and bound its constants from the base weights at arm time; peft (and
our own additive branch) put their work in SUBMODULES the artifact never runs.
So the request pays for an adapter and receives the base model, with no
refusal, no eager fallback and no log.

Everything below runs the real code on CPU: a real tiny diffusers
``UNet2DConditionModel``, the real ``adapter_guard`` seam, the real
``w8a8_lora`` key resolution under ``lora_fold``. The artifact stand-in is a
FROZEN DEEP COPY of the pre-fold denoiser called through its own forward —
semantically exactly what a weightless AOTI package bound once at arm time is:
the base-weight graph, its own memory, blind to later module surgery. That is
the property under test, and a stand-in that reproduces it is the honest way to
test it without a card.

Every arm here is red-armed: the drift test runs a deliberately-broken restore
(``sub_`` the delta instead of copying the original back) and asserts it is
CAUGHT, and the guard test asserts the unguarded shape serves the base model.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker.serving import adapter_guard  # noqa: E402
from gen_worker.api.errors import RefCompatibilitySurprise  # noqa: E402
from gen_worker.models import lora_fold  # noqa: E402

_RANK = 4


def _tiny_unet() -> Any:
    from diffusers import UNet2DConditionModel

    torch.manual_seed(0)
    unet: Any = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4,
        block_out_channels=(16, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=16, attention_head_dim=4, norm_num_groups=8,
    )
    unet.eval()
    return unet


class _Pipe:
    """A pipeline that is nothing but its denoiser — which is all
    ``branch_targets`` reads."""

    def __init__(self, unet: Any) -> None:
        self.unet = unet
        self.text_encoder: Any = None


def _adapter(unet: Any, seed: int, scale: float = 1.0) -> lora_fold.Adapter:
    """One adapter in the live kohya-flat diffusers grammar, over an
    attention projection (Linear) and a resnet conv (the LoCon pair class)."""
    torch.manual_seed(seed)
    sd: Dict[str, Any] = {}

    def pair(flat: str, a: Any, b: Any) -> None:
        sd[f"lora_unet_{flat}.lora_down.weight"] = a
        sd[f"lora_unet_{flat}.lora_up.weight"] = b
        sd[f"lora_unet_{flat}.alpha"] = torch.tensor(float(_RANK))

    q = unet.get_submodule(
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q")
    pair("down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
         torch.randn(_RANK, q.in_features) * 0.05,
         torch.randn(q.out_features, _RANK) * 0.05)
    r1 = unet.get_submodule("down_blocks.0.resnets.0.conv1")
    pair("down_blocks_0_resnets_0_conv1",
         torch.randn(_RANK, r1.in_channels, *r1.kernel_size) * 0.05,
         torch.randn(r1.out_channels, _RANK, 1, 1) * 0.05)
    return (sd, scale, f"cozy/probe-{seed}@v1")


def _inputs() -> Tuple[Any, Any, Any]:
    torch.manual_seed(7)
    return (torch.randn(1, 4, 8, 8), torch.tensor(5), torch.randn(1, 4, 16))


def _run(unet: Any, args: Tuple[Any, ...]) -> Any:
    with torch.inference_mode():
        return unet(*args, return_dict=False)[0].clone()


class _FrozenArtifact:
    """The AOTI package, in the one respect that matters: it is a copy of the
    denoiser taken at ARM TIME and it never sees a later module change.

    Implements the ``EntryDispatch`` surface ``wrap_module`` uses.
    """

    def __init__(self, module: Any) -> None:
        self._frozen = copy.deepcopy(module).eval()
        self.calls = 0
        self.last_selected = "probe-graph"
        self.runners: Tuple[Any, ...] = (("probe-graph", None),)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        with torch.inference_mode():
            return self._frozen(*args, **kwargs)

    def excludes(self, names: Any) -> bool:
        return False

    def remove(self, name: str, reason: str) -> None:
        self.runners = ()


class _RecordingPackage:
    """The AOTI package surface ``rearm_constants`` drives, and the only one
    it drives: ``load_constants(values, check_full_update, user_managed)``.
    Records each re-install so the test can assert the fold told the artifact
    its constants moved."""

    def __init__(self) -> None:
        self.loads: List[Dict[str, Any]] = []

    def load_constants(self, values: Any, *, check_full_update: bool,
                       user_managed: bool = False,
                       allow_h2d_copy: bool = False) -> None:
        assert check_full_update and user_managed
        self.loads.append(dict(values))


class _BoundRunner:
    """``TCGEntryRunner.runner``'s shape: the loaded package plus the constant
    table that was bound BY REFERENCE at arm time."""

    def __init__(self, module: Any) -> None:
        self._package = _RecordingPackage()
        self._bound_values = {
            name: tensor for name, tensor in module.named_parameters()}


class _Entry:
    def __init__(self, module: Any) -> None:
        self.runner = _BoundRunner(module)


class _PointerArtifact(_FrozenArtifact):
    """The other half of the truth: an artifact whose constants are the
    module's OWN tensors, bound by reference (``user_managed=True``). An
    in-place weight write IS visible to it — which is why the fold is in place
    and never a tensor swap."""

    def __init__(self, module: Any) -> None:
        self._frozen = module
        self.calls = 0
        self.last_selected = "probe-graph"
        self.entry = _Entry(module)
        self.runners = (("probe-graph", self.entry),)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        with torch.inference_mode():
            return type(self._frozen).forward(self._frozen, *args, **kwargs)


class _Dispatcher:
    """torchcg's ``_ForwardDispatcher``, in the respects the guard reads.

    pgw#1573: this file armed through ``aot_serve.wrap_module``, which has no
    production caller — the live arm is ``AdoptSession`` installing one of
    these as the module's ``forward``. Same doubles, same assertions, the seam
    a pod actually runs. ``_entries`` is torchcg's ``(record, call)`` shape and
    ``call.runner`` is the artifact's bound runner, which is what
    ``adapter_guard.rearm_constants`` re-installs.
    """

    def __init__(self, module: Any, artifact: Any) -> None:
        self.module = module
        self.eager_forward = module.forward
        self.artifact = artifact
        entry = getattr(artifact, "entry", None)
        self._entries = ((SimpleNamespace(graph="probe-graph"), entry),)

    def armed_graphs(self) -> Tuple[str, ...]:
        return ("probe-graph",)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.artifact(*args, **kwargs)


def _arm(unet: Any, artifact: Any) -> None:
    """Arm through the LIVE seam: torchcg's dispatcher, plus pgw#1573's guard.

    ``adapter_guard.install`` is what ``ctx.compile`` runs on every adopted
    module on both serving hosts, so what this file measures is what a pod
    does.
    """
    unet.forward = _Dispatcher(unet, artifact)
    assert adapter_guard.install(unet), (
        "the guard did not recognise the dispatcher, so every peft row below "
        "would pass for the wrong reason")


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_an_artifact_bound_by_value_is_blind_to_module_side_weight_surgery() -> None:
    """The defect in one assertion. An artifact whose constants are a COPY —
    which is what a bound-and-then-copied table is, and what peft's submodule
    wrapping is always up against — returns the base model bit-identically
    while an adapter is live on the module. RED ARM for the guard below."""
    unet = _tiny_unet()
    args = _inputs()
    baseline = _run(unet, args)

    artifact = _FrozenArtifact(unet)
    _arm(unet, artifact)
    # Exactly the module state peft's `inject_adapter_in_model` leaves behind
    # (verified against peft 0.19.1: absent -> {'name': cfg} -> absent on
    # unload). Set directly so the assertion does not need peft installed.
    scope = lora_fold.compute_deltas(unet, [_adapter(unet, 1)])
    held = lora_fold.apply_fold(unet, scope)
    try:
        served = _run(unet, args)
    finally:
        held.restore()

    assert artifact.calls == 1
    assert torch.equal(served, baseline), (
        "the frozen artifact must be blind to module-side weight surgery — "
        "that blindness IS the defect this issue is about"
    )


def test_the_peft_guard_routes_a_compiled_armed_module_to_eager() -> None:
    unet = _tiny_unet()
    args = _inputs()
    artifact = _FrozenArtifact(unet)
    _arm(unet, artifact)
    assert _run(unet, args) is not None and artifact.calls == 1

    setattr(unet, adapter_guard.PEFT_MARKER_ATTR, {"probe": object()})
    before = artifact.calls
    _run(unet, args)
    assert artifact.calls == before, (
        "a live peft adapter must NOT reach the artifact — the artifact "
        "cannot execute peft's submodule wrappers and would serve base"
    )

    delattr(unet, adapter_guard.PEFT_MARKER_ATTR)
    _run(unet, args)
    assert artifact.calls == before + 1, "unloading the adapter must resume compiled"


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------


def test_the_fold_reaches_a_pointer_bound_artifact() -> None:
    """An in-place fold is visible to an artifact that holds the module's own
    tensors — the property ``load_constants(user_managed=True)`` provides and
    the reason the fold must never swap a fresh tensor onto the module."""
    unet = _tiny_unet()
    args = _inputs()
    artifact = _PointerArtifact(unet)
    _arm(unet, artifact)
    baseline = _run(unet, args)

    with lora_fold.folded(_Pipe(unet), [_adapter(unet, 1)],
                          rebind=adapter_guard.rearm_constants) as stats:
        served = _run(unet, args)

    assert stats["modules"] == 2, stats
    assert not torch.equal(served, baseline), (
        "the folded adapter must change the served tensor")
    assert artifact.calls == 2, "both calls went through the artifact"
    assert torch.equal(_run(unet, args), baseline), "restore must be exact"
    assert len(artifact.entry.runner._package.loads) == 2, (
        "the artifact must be told its constants moved on the way IN and on "
        "the way OUT — AOTI folds once and never re-folds on a bare write")


def test_the_fold_matches_the_eager_adapter_it_replaces() -> None:
    """Folding is not an approximation of the adapter: ``W += B@A*s`` and an
    additive branch compute the same thing, so the folded weights must agree
    with the branch path to floating-point noise."""
    from gen_worker.models import w8a8_lora

    unet = _tiny_unet()
    args = _inputs()
    adapter = _adapter(unet, 3)

    branch = copy.deepcopy(unet)
    w8a8_lora.enable_lora_branches(branch, 16)
    w8a8_lora.apply_branch_adapters(branch, [adapter], allow_resize=True)
    branched = _run(branch, args)

    with lora_fold.folded(_Pipe(unet), [adapter]):
        folded_out = _run(unet, args)

    assert torch.allclose(folded_out, branched, atol=2e-5, rtol=2e-4), (
        f"max|delta| = {(folded_out - branched).abs().max().item():.3e}")


def test_two_adapters_fold_together_and_commute_with_the_branch_sum() -> None:
    unet = _tiny_unet()
    args = _inputs()
    a, b = _adapter(unet, 1, 0.8), _adapter(unet, 2, 0.5)
    base = _run(unet, args)

    with lora_fold.folded(_Pipe(unet), [a, b]) as stats:
        both = _run(unet, args)
    assert stats["modules"] == 2
    assert not torch.equal(both, base)

    with lora_fold.folded(_Pipe(unet), [a]):
        only_a = _run(unet, args)
    assert not torch.equal(both, only_a), "the second adapter must contribute"


# ---------------------------------------------------------------------------
# Restore: bit-exact, and the check is proven able to fail
# ---------------------------------------------------------------------------


def _serial_drift(unet: Any, pipe: Any, adapters: List[Any], args: Any,
                  *, break_restore: bool = False) -> List[Any]:
    """Ten serial requests, alternating A / none / B / none. Returns every
    bare-request output; each must equal the pre-LoRA baseline exactly."""
    bare: List[Any] = []
    for step in range(10):
        riding = [adapters[(step // 2) % len(adapters)]] if step % 2 == 0 else []
        if not riding:
            bare.append(_run(unet, args))
            continue
        deltas = lora_fold.compute_deltas(unet, riding, pipe=pipe)
        scope = lora_fold.apply_fold(unet, deltas)
        _run(unet, args)
        if break_restore:
            # THE BROKEN RESTORE the exact one exists to avoid: subtract the
            # delta instead of copying the original bytes back. Algebraically
            # the inverse; in floating point it is not, and a serial stream
            # accumulates the difference.
            with torch.no_grad():
                mods = dict(unet.named_modules())
                for path, d in deltas.items():
                    w = mods[path].weight
                    w.data.copy_((w.data.to(torch.float32) - d).to(w.dtype))
            scope.saved.clear()
        else:
            scope.restore()
    return bare


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_restore_is_bit_exact_over_ten_serial_alternating_requests(dtype: Any) -> None:
    unet = _tiny_unet().to(dtype)
    pipe = _Pipe(unet)
    args = tuple(
        t.to(dtype) if t.is_floating_point() else t for t in _inputs())
    adapters = [_adapter(unet, 1), _adapter(unet, 2)]
    before = {n: p.detach().clone() for n, p in unet.named_parameters()}
    baseline = _run(unet, args)

    bare = _serial_drift(unet, pipe, adapters, args)

    assert len(bare) == 5
    for i, out in enumerate(bare):
        assert torch.equal(out, baseline), f"bare request {i} drifted"
    for name, original in before.items():
        assert torch.equal(unet.get_parameter(name), original), (
            f"{name} did not come back byte for byte")


def test_the_drift_check_catches_a_deliberately_broken_restore() -> None:
    """RED ARM. bf16 is where the delta-subtract restore is visibly wrong; if
    this ever passes, the drift test above is proving nothing."""
    unet = _tiny_unet().to(torch.bfloat16)
    pipe = _Pipe(unet)
    args = tuple(
        t.to(torch.bfloat16) if t.is_floating_point() else t for t in _inputs())
    adapters = [_adapter(unet, 1), _adapter(unet, 2)]
    before = {n: p.detach().clone() for n, p in unet.named_parameters()}
    baseline = _run(unet, args)

    bare = _serial_drift(unet, pipe, adapters, args, break_restore=True)

    drifted_outputs = [i for i, out in enumerate(bare)
                       if not torch.equal(out, baseline)]
    drifted_weights = [n for n, original in before.items()
                       if not torch.equal(unet.get_parameter(n), original)]
    assert drifted_outputs or drifted_weights, (
        "the delta-subtract restore left no trace — the exactness check "
        "cannot go red and therefore proves nothing")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_compiled_armed_module_refuses_a_fold_with_no_rebind_seam() -> None:
    unet = _tiny_unet()
    _arm(unet, _FrozenArtifact(unet))
    with pytest.raises(RefCompatibilitySurprise, match="constant re-arm"):
        with lora_fold.folded(_Pipe(unet), [_adapter(unet, 1)]):
            pass


def test_an_eager_module_needs_no_rebind_seam() -> None:
    unet = _tiny_unet()
    args = _inputs()
    baseline = _run(unet, args)
    with lora_fold.folded(_Pipe(unet), [_adapter(unet, 1)]):
        assert not torch.equal(_run(unet, args), baseline)
    assert torch.equal(_run(unet, args), baseline)


def test_a_quantized_leaf_refuses_the_fold_by_name() -> None:
    """An fp8 grid cannot hold a low-rank delta; folding onto it would round
    the adapter away and serve a confidently weakened result."""
    from gen_worker.models.w8a8 import fp8_scaled_linear_class

    unet = _tiny_unet()
    path = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
    parent = unet.get_submodule(path.rpartition(".")[0])
    old = getattr(parent, "to_q")
    cls = fp8_scaled_linear_class()
    new = cls(old.in_features, old.out_features, bias=old.bias is not None,
              compute_dtype=torch.float32, static_input_scale=False,
              gemm_mode="pertensor")
    new.weight = old.weight.detach().to(torch.float8_e4m3fn)
    new.weight_scale = torch.ones(old.out_features, 1, dtype=torch.float32)
    setattr(parent, "to_q", new)

    with pytest.raises(RefCompatibilitySurprise, match="QUANTIZED"):
        lora_fold.compute_deltas(unet, [_adapter(_tiny_unet(), 1)])


def test_the_text_encoder_half_is_folded_too_and_not_dropped() -> None:
    """A style LoRA usually carries a text-encoder half. Folding only the
    denoiser would serve the adapter at the wrong strength with a clean log."""
    import torch.nn as nn

    class _TinyTE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)

        def forward(self, x: Any) -> Any:
            return self.q_proj(x)

    unet = _tiny_unet()
    pipe = _Pipe(unet)
    pipe.text_encoder = _TinyTE()
    before = pipe.text_encoder.q_proj.weight.detach().clone()

    torch.manual_seed(11)
    sd, weight, ref = _adapter(unet, 5)
    sd["text_encoder.q_proj.lora_A.weight"] = torch.randn(_RANK, 8) * 0.05
    sd["text_encoder.q_proj.lora_B.weight"] = torch.randn(8, _RANK) * 0.05

    with lora_fold.folded(pipe, [(sd, weight, ref)]) as stats:
        assert stats["components"] == 2, stats
        assert not torch.equal(pipe.text_encoder.q_proj.weight, before), (
            "the text-encoder half must actually land")
    assert torch.equal(pipe.text_encoder.q_proj.weight, before)


def test_a_key_that_lands_on_no_component_refuses_rather_than_being_dropped() -> None:
    unet = _tiny_unet()
    sd, weight, ref = _adapter(unet, 6)
    sd["text_encoder_2.q_proj.lora_A.weight"] = torch.randn(_RANK, 8) * 0.05
    sd["text_encoder_2.q_proj.lora_B.weight"] = torch.randn(8, _RANK) * 0.05

    with pytest.raises(RefCompatibilitySurprise, match="land on no component"):
        with lora_fold.folded(_Pipe(unet), [(sd, weight, ref)]):
            pass


def test_the_kohya_text_encoder_alias_routes_to_the_same_component() -> None:
    """`lora_te_`/`lora_te1_` are sd-scripts' own grammar and route through the
    same table as the dotted form — one alias table, checked from both sides."""
    import torch.nn as nn

    class _TinyTE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)

    unet = _tiny_unet()
    pipe = _Pipe(unet)
    pipe.text_encoder = _TinyTE()
    before = pipe.text_encoder.q_proj.weight.detach().clone()

    torch.manual_seed(13)
    sd, weight, ref = _adapter(unet, 7)
    sd["lora_te_q_proj.lora_down.weight"] = torch.randn(_RANK, 8) * 0.05
    sd["lora_te_q_proj.lora_up.weight"] = torch.randn(8, _RANK) * 0.05

    with lora_fold.folded(pipe, [(sd, weight, ref)]) as stats:
        assert stats["components"] == 2, stats
        assert not torch.equal(pipe.text_encoder.q_proj.weight, before)
    assert torch.equal(pipe.text_encoder.q_proj.weight, before)


def test_an_empty_adapter_set_is_a_no_op_that_still_yields() -> None:
    unet = _tiny_unet()
    args = _inputs()
    baseline = _run(unet, args)
    with lora_fold.folded(_Pipe(unet), []) as stats:
        assert stats == {}
        assert torch.equal(_run(unet, args), baseline)


def test_rearm_constants_still_matches_the_real_runner_surface() -> None:
    """The stand-in above defines `_package`/`_bound_values`, so it would keep
    passing if the VENDORED runner renamed them and production started
    refusing every fold. Assert the two names against the real class.

    `rearm_constants` reaches into those privates on purpose: the right home is
    a `CompiledGraphRunner.rebind()` upstream, but the vendored snapshot is
    sha256-fenced (`_vendor/VENDORED.toml`) so it cannot be edited here. This
    is the tripwire that makes the re-vendor land loudly instead of quietly.
    """
    import inspect

    from gen_worker._vendor.torchcg.runner import CompiledGraphRunner

    source = inspect.getsource(CompiledGraphRunner)
    for attribute in ("self._package", "self._bound_values"):
        assert attribute in source, (
            f"{attribute} is gone from CompiledGraphRunner — "
            "adapter_guard.rearm_constants reads it to re-install the "
            "constant "
            "table after a fold, and would now refuse every compiled fold. "
            "Give torchcg a public rebind() and call that instead"
        )


def test_the_set_digest_keys_on_refs_and_weights() -> None:
    unet = _tiny_unet()
    a, b = _adapter(unet, 1, 0.8), _adapter(unet, 2, 0.5)
    assert lora_fold.adapter_digest([a, b]) == lora_fold.adapter_digest([a, b])
    assert lora_fold.adapter_digest([a, b]) != lora_fold.adapter_digest([b, a])
    heavier = (a[0], 0.9, a[2])
    assert lora_fold.adapter_digest([a]) != lora_fold.adapter_digest([heavier])
