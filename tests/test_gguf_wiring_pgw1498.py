"""pgw#1498's SERVING PATH: a GGUF snapshot loads as ggml block bytes on our
own leaves, and everything downstream of the loader sees it.

The core module (``models/gguf_dequant`` + ``models/gguf_torch``) was correct
and unreachable — nothing constructed it, because ``load_gguf_pipeline`` handed
the decode to diffusers' ``GGUFQuantizationConfig``. What is proved here is
reachability, on the real codepath and with no mocks:

  * a composed snapshot goes in at ``loading.load_gguf_pipeline`` and a
    pipeline whose denoiser holds uint8 block buffers comes out — Linear AND
    conv, which the delegated path cannot do at all;
  * the residency walk reports the QUANTIZED size, which is the one number this
    lane exists to move;
  * the LoRA branch machinery targets the punned leaves, by the same rule and
    the same function that targets an fp8-storage leaf;
  * the ``dequant_ahead`` tier dial turns from the residency lease's surplus
    through the ordinary placement entry point, and the two ends of the dial
    serve the SAME pipeline.

Everything is synthesized locally with ``gguf.quants.quantize`` over a tiny real
``UNet2DConditionModel`` and runs on CPU: no community checkpoint is downloaded
(multi-GB weights must not transit this machine) and nothing about
decode-per-forward needs a card to be true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
diffusers = pytest.importorskip("diffusers")

from diffusers import DDIMScheduler, DiffusionPipeline, UNet2DConditionModel

from gen_worker.models import gguf_torch, loading, memory, w8a8_lora
from gen_worker.models.gguf_diffusers import (
    NormalizedTensors,
    SingleFileGguf,
    build_denoiser,
)

#: Small enough to build and run per test, wide enough that most weights are
#: block-aligned: a ggml row is the flattened per-output row and its length must
#: be a multiple of the block size (32 here), so 32 channels x a 3x3 kernel
#: quantizes and a 4-channel stem does not — which is the real mixed shape a
#: community checkpoint has.
_UNET = dict(
    sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
    block_out_channels=(32, 32),
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8,
)
_QTYPE = gguf.GGMLQuantizationType.Q4_0
_BLOCK = 32


class TinyGgufPipeline(DiffusionPipeline):
    """The smallest real ``DiffusionPipeline`` that composes a denoiser — the
    loader's ``cls`` argument, exercised for real rather than stubbed."""

    def __init__(self, unet: Any, scheduler: Any) -> None:
        super().__init__()
        cast(Any, self).register_modules(unet=unet, scheduler=scheduler)


def _unet(config: Dict[str, Any]) -> Any:
    """A real ``UNet2DConditionModel``, in eval mode."""
    return cast(Any, UNet2DConditionModel(**config)).eval()


def _quantizable(t: Any) -> bool:
    return t.dim() >= 2 and (t.numel() // int(t.shape[0])) % _BLOCK == 0


def _write_gguf(path: Path, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Pack a state dict into a real ``.gguf`` and return what it decodes to."""
    writer = gguf.GGUFWriter(str(path), arch="unet")
    decoded: Dict[str, Any] = {}
    for name, tensor in state_dict.items():
        dense = tensor.detach().float()
        if _quantizable(dense):
            rows = dense.numpy().reshape(int(dense.shape[0]), -1)
            raw = gguf.quants.quantize(rows, _QTYPE)
            writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=_QTYPE)
            decoded[name] = torch.from_numpy(
                gguf.quants.dequantize(raw, _QTYPE).astype(np.float32),
            ).reshape(dense.shape)
        else:
            writer.add_tensor(name, dense.numpy())
            decoded[name] = dense
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return decoded


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: Any) -> Any:
    """A composed GGUF snapshot: the diffusers tree a materialized checkpoint
    is, with the denoiser present only as a ``.gguf`` container."""
    root = tmp_path_factory.mktemp("gguf-snapshot")
    torch.manual_seed(1498)
    reference = _unet(_UNET)
    (root / "unet").mkdir()
    reference.save_config(root / "unet")
    cast(Any, DDIMScheduler()).save_config(root / "scheduler")
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "TinyGgufPipeline",
        "_diffusers_version": diffusers.__version__,
        "unet": ["diffusers", "UNet2DConditionModel"],
        "scheduler": ["diffusers", "DDIMScheduler"],
    }), encoding="utf-8")
    decoded = _write_gguf(root / "unet-Q4_0.gguf", reference.state_dict())
    return root, decoded


def _load(snapshot: Any, **kwargs: Any) -> Any:
    root, _ = snapshot
    found = loading.detect_gguf_snapshot(root)
    assert found is not None, "the composed snapshot must be detected as gguf"
    gguf_file, qtype = found
    assert qtype == "q4_0"
    return loading.load_gguf_pipeline(TinyGgufPipeline, root, gguf_file, **kwargs)


def _inputs() -> tuple[Any, Any, Any]:
    torch.manual_seed(7)
    return (torch.randn(1, 4, 8, 8, dtype=torch.bfloat16),
            torch.tensor([1]),
            torch.randn(1, 2, 32, dtype=torch.bfloat16))


def _denoise(pipe: Any) -> Any:
    latents, timestep, encoder = _inputs()
    with torch.no_grad():
        return pipe.unet(latents, timestep, encoder).sample.float()


# --- the loader ------------------------------------------------------------


def test_the_snapshot_loads_onto_our_leaves_and_not_diffusers_quantizer(
        snapshot: Any) -> None:
    """THE wiring claim. The pipeline that comes out of the production loader
    holds ggml block bytes on punned leaves of OURS — including convs, which
    diffusers' GGUF path (``nn.Linear`` only) cannot restructure at all."""
    pipe = _load(snapshot)

    leaves = gguf_torch.gguf_leaves(pipe.unet)
    assert leaves, "no leaf holds block bytes: the lane is unreachable again"
    kinds = {gguf_torch.structural_base(m).__name__ for m in leaves.values()}
    assert "Linear" in kinds and "Conv2d" in kinds

    # Every leaf's weight is a uint8 BUFFER, never a float parameter.
    for path, leaf in leaves.items():
        assert leaf.weight.dtype is torch.uint8, path
        assert "weight" in leaf._buffers and "weight" not in leaf._parameters

    # And the delegated path is gone: no diffusers quantizer was constructed.
    assert getattr(pipe.unet, "hf_quantizer", None) is None
    assert not any(type(m).__module__.startswith("diffusers.quantizers")
                   for m in pipe.unet.modules())


def test_the_pipeline_serves_what_the_quantized_weights_say(snapshot: Any) -> None:
    """The whole pipeline runs and its answer is the one those block bytes
    decode to — checked against an eager model holding the decoded weights."""
    root, decoded = snapshot
    pipe = _load(snapshot)

    eager = _unet(_UNET)
    eager.load_state_dict(decoded)
    eager = eager.to(torch.bfloat16)

    latents, timestep, encoder = _inputs()
    with torch.no_grad():
        served = pipe.unet(latents, timestep, encoder).sample.float()
        want = eager(latents, timestep, encoder).sample.float()
    assert torch.allclose(served, want, atol=2e-2, rtol=0), (
        (served - want).abs().max().item())


def test_residency_reports_the_quantized_size_not_the_dequantized_one(
        snapshot: Any) -> None:
    """The number the small-card ladder reads. A lying ``.shape`` would have
    over-reported this denoiser by the compression ratio — the one figure the
    whole lane exists to move."""
    pipe = _load(snapshot)
    reference = UNet2DConditionModel(**_UNET)

    walked = memory.estimate_pipeline_size_gb(pipe.unet)
    dense = memory.estimate_pipeline_size_gb(reference)
    assert walked < dense / 2, (walked, dense)

    quantized = gguf_torch.quantized_bytes(pipe.unet)
    assert 0 < quantized
    # The blocks dominate what the walk saw.
    assert quantized > 0.5 * walked * (1024 ** 3)


def test_the_weight_lane_is_stamped_and_is_a_declared_lane(snapshot: Any) -> None:
    """A GGML denoiser traces differently from a plain one, so it is its own
    compiled-graph family. The token is in the ONE vocabulary a loader may
    stamp, which is what the completeness tape checks."""
    pipe = _load(snapshot)
    assert loading.pipeline_weight_lane(pipe) == loading.EXECUTION_LANE_GGUF
    assert loading.EXECUTION_LANE_GGUF in loading.STAMPABLE_BASE_EXECUTION_LANES
    assert w8a8_lora.branch_execution_lane(pipe.unet) == "gguf"


def test_a_gguf_component_is_refused_by_the_component_loader(snapshot: Any) -> None:
    """The denoiser is built inside the PIPELINE load; there is no
    component-level loader to borrow, and that refuses by name."""
    root, _ = snapshot
    with pytest.raises(loading.ComponentExecutionLaneUnsupported, match="GGUF"):
        loading.contract_loaded_component(
            root, "unet", cls=UNet2DConditionModel, compute_dtype=torch.bfloat16)


# --- the LoRA branch route -------------------------------------------------


def test_the_lora_branch_walker_admits_the_punned_leaves(snapshot: Any) -> None:
    """``w8a8_lora.branch_modules`` selects by EXACT class over
    ``structural_base``. One shared marker is what makes a pun it has never
    heard of visible — with two spellings a GGML Linear was invisible to
    adapter targeting while every walk still looked correct."""
    pipe = _load(snapshot)
    targeted = w8a8_lora.branch_modules(pipe.unet)
    punned = gguf_torch.gguf_leaves(pipe.unet)
    linear_or_conv = {
        path for path, leaf in punned.items()
        if gguf_torch.structural_base(leaf) in (torch.nn.Linear, torch.nn.Conv2d)
    }
    assert linear_or_conv, "the fixture must punn Linear/Conv leaves"
    assert linear_or_conv <= set(targeted)


def test_a_branch_adds_onto_the_decoded_weight_and_never_touches_the_blocks(
        snapshot: Any) -> None:
    """The branch wraps the punned forward, so it adds onto the DECODED
    output. Attach-after-decode is lossless on a 4-bit grid and unpatching is
    exact — which is why the refuse-adapters-on-a-quantized-grid rule has no
    referent here."""
    pipe = _load(snapshot)
    unet = pipe.unet
    before = _denoise(pipe)
    blocks = {p: leaf.weight.clone()
              for p, leaf in gguf_torch.gguf_leaves(unet).items()}

    w8a8_lora.enable_lora_branches(unet, 16)
    assert w8a8_lora.branch_bucket(unet) == 16
    # A zeroed branch is exactly the branchless answer.
    assert torch.equal(_denoise(pipe), before)

    touched = 0
    for path, mod in w8a8_lora.branch_modules(unet).items():
        if gguf_torch.is_gguf_leaf(mod) and isinstance(mod, torch.nn.Linear):
            with torch.no_grad():
                mod.lora_b.normal_(0.0, 0.05)
                mod.lora_a.normal_(0.0, 0.05)
            touched += 1
    assert touched, "no punned Linear carried a branch"
    assert not torch.equal(_denoise(pipe), before)

    # The block bytes are read-only under an adapter — byte-identical.
    for path, leaf in gguf_torch.gguf_leaves(unet).items():
        assert torch.equal(leaf.weight, blocks[path]), path

    w8a8_lora.disable_lora_branches(unet)
    assert torch.equal(_denoise(pipe), before)


def test_the_bucketed_lane_is_a_gguf_graph_family(snapshot: Any) -> None:
    """A branch-bearing GGML denoiser is a different compiled graph from both
    the branchless GGML one and a plain-resident bucket."""
    pipe = _load(snapshot)
    w8a8_lora.enable_lora_branches(pipe.unet, 16)
    w8a8_lora.stamp_execution_lane(pipe)
    assert loading.pipeline_weight_lane(pipe) == "gguf-lora16"

    from gen_worker.compile_cache import execution_lane_bucket, execution_lane_token

    assert execution_lane_bucket("gguf-lora16") == ("gguf", 16)
    assert execution_lane_token("gguf-lora16") == "gguf-lora16"


# --- the tier dial, driven from the lease ----------------------------------


def _lease(monkeypatch: Any, free_gb: float) -> None:
    """The card's own free-VRAM reading, which the dial caps its spend by. A
    HOST FACT, and the only thing a CPU box cannot supply for real."""
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda: free_gb)


def test_a_zero_lease_leaves_the_pipeline_quantized_resident(
        snapshot: Any, monkeypatch: Any) -> None:
    """Tier 2, the normal GGUF case. No lease in scope means no surplus to
    spend, and the constrained tier is the correct serving answer — never a
    failure."""
    _lease(monkeypatch, 40.0)
    pipe = _load(snapshot)
    quantized = gguf_torch.quantized_bytes(pipe.unet)

    applied = memory.apply_low_vram_config(pipe, mode="off", stream_budget_bytes=0)
    assert "gguf_dequant_ahead" not in applied
    assert gguf_torch.quantized_bytes(pipe.unet) == quantized


def test_a_surplus_lease_decodes_ahead_through_the_ordinary_placement_call(
        snapshot: Any, monkeypatch: Any) -> None:
    """Tier 3, and it is reached from the SAME entry point every other rung is:
    ``apply_low_vram_config`` with pgw#1497's admission budget. A worker handed
    surplus memory spends it decoding once instead of every step."""
    _lease(monkeypatch, 40.0)
    pipe = _load(snapshot)
    quantized = gguf_torch.quantized_bytes(pipe.unet)
    assert quantized > 0

    applied = memory.apply_low_vram_config(
        pipe, mode="off", stream_budget_bytes=8 * (1024 ** 3))
    assert applied["gguf_dequant_ahead"] > 0
    assert applied["gguf_quantized_bytes"] == 0
    assert applied["gguf_peak_transient_bytes"] == 0
    assert gguf_torch.quantized_bytes(pipe.unet) == 0
    # One-way: nothing was re-quantized, and the leaves stay punned so an
    # adapter still attaches exactly as it did on the quantized side.
    assert gguf_torch.gguf_leaves(pipe.unet)


def test_the_two_ends_of_the_dial_serve_the_same_pipeline(
        snapshot: Any, monkeypatch: Any) -> None:
    """Paul's ruling is that these are two settings of ONE dial, not two
    lanes — so the answer may not depend on which end the lease landed on."""
    _lease(monkeypatch, 40.0)
    constrained = _load(snapshot)
    surplus = _load(snapshot)
    memory.apply_low_vram_config(
        surplus, mode="off", stream_budget_bytes=8 * (1024 ** 3))

    assert torch.equal(_denoise(constrained), _denoise(surplus))


def test_a_partial_lease_graduates_and_shrinks_the_per_forward_transient(
        snapshot: Any, monkeypatch: Any) -> None:
    """Between the ends the dial graduates per layer, LARGEST FIRST — the only
    order that shrinks the transient headroom a fit plan must reserve."""
    _lease(monkeypatch, 40.0)
    pipe = _load(snapshot)
    quantized = gguf_torch.quantized_bytes(pipe.unet)
    transient = gguf_torch.peak_transient_bytes(pipe.unet, dtype=torch.bfloat16)
    resident = int(memory.estimate_pipeline_size_gb(pipe) * (1024 ** 3))

    applied = memory.apply_low_vram_config(
        pipe, mode="off", stream_budget_bytes=resident + quantized // 2)
    assert 0 < applied["gguf_dequant_ahead"]
    assert 0 < applied["gguf_quantized_bytes"] < quantized
    # ``<=``, not ``<``: this fixture has many weights of the SAME largest
    # size, so a partial spend can graduate several and still leave one of them
    # quantized. The claim is that the transient never GROWS as the dial turns
    # up, and that it reaches zero at the far end (asserted above).
    assert applied["gguf_peak_transient_bytes"] <= transient


def test_a_card_with_no_free_headroom_keeps_the_constrained_tier(
        snapshot: Any, monkeypatch: Any) -> None:
    """A lease written when the card was emptier is not a licence to allocate
    bytes that are gone — the same cap the partial_stream rung applies."""
    _lease(monkeypatch, 0.0)
    pipe = _load(snapshot)
    quantized = gguf_torch.quantized_bytes(pipe.unet)

    applied = memory.apply_low_vram_config(
        pipe, mode="off", stream_budget_bytes=8 * (1024 ** 3))
    assert applied.get("gguf_dequant_ahead", 0) == 0
    assert gguf_torch.quantized_bytes(pipe.unet) == quantized


# --- the source seam -------------------------------------------------------


def test_the_store_source_is_one_constructor_from_the_edge_source(
        snapshot: Any, tmp_path: Path) -> None:
    """Paul's storage ruling: a ``.gguf`` FILE exists only at the edges, and the
    served path takes per-tensor block bytes out of the CAS. Both sources build
    the SAME denoiser through the same call — the swap is the constructor and
    nothing else.
    """
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.tensorfs.tensors import open_tensors

    # A cross-attention-free UNet, and the reason is a REAL constraint worth
    # naming rather than working around silently: tensorfs' `gguf-v1` planner
    # caps a tensor name at MAX_TENSOR_NAME_LEN = 63 bytes (llama.cpp's own
    # GGML_MAX_NAME), and diffusers keys blow straight through it —
    # `down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.weight`
    # is 69. So the CAS path cannot be fed a CONTAINER carrying our key layout,
    # which is exactly what the ingest half exists to stop doing: normalize to
    # per-tensor CAS regions with our own metadata and no container in the
    # middle. Recorded on the pgw#1498 tracker section.
    config = dict(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 32),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"), mid_block_type=None,
        cross_attention_dim=32, norm_num_groups=8,
    )
    torch.manual_seed(3)
    reference = _unet(config)
    config_dir = tmp_path / "unet"
    config_dir.mkdir()
    reference.save_config(config_dir)
    staged = tmp_path / "staged"
    staged.mkdir()
    gguf_file = staged / "unet-Q4_0.gguf"
    _write_gguf(gguf_file, reference.state_dict())
    assert max(len(k) for k in reference.state_dict()) <= 63

    cas = LocalCAS(tmp_path / "cas")
    manifest = cas.ingest_repository(staged)
    with open_tensors(cas, manifest) as reader:
        views = {name: reader[name] for name in reader}
        assert any(v.format == "gguf-v1" and v.block.quantized
                   for v in views.values())
        from_store = build_denoiser(
            UNet2DConditionModel, config_dir,
            NormalizedTensors(views), compute_dtype=torch.bfloat16)

    from_edge = build_denoiser(
        UNet2DConditionModel, config_dir,
        SingleFileGguf(gguf_file), compute_dtype=torch.bfloat16)

    assert set(gguf_torch.gguf_leaves(from_store)) == \
        set(gguf_torch.gguf_leaves(from_edge))
    latents, timestep, encoder = _inputs()
    with torch.no_grad():
        assert torch.equal(
            from_store(latents, timestep, encoder).sample,
            from_edge(latents, timestep, encoder).sample)


def test_a_source_that_leaves_a_weight_on_meta_refuses_by_name(
        snapshot: Any) -> None:
    """A meta weight builds, loads, ADVERTISES and then dies on the first
    request. The silence is the defect, so the build refuses instead."""
    root, _ = snapshot

    class _Partial:
        def tensors(self, model: Any, config: Any) -> Dict[str, Any]:
            full = SingleFileGguf(root / "unet-Q4_0.gguf").tensors(model, config)
            dropped = next(k for k in sorted(full) if k.endswith("conv_in.weight"))
            return {k: v for k, v in full.items() if k != dropped}

    with pytest.raises(ValueError, match="still on `meta`"):
        build_denoiser(UNet2DConditionModel, root / "unet", _Partial(),
                       compute_dtype=torch.bfloat16)


def test_bytes_that_are_not_this_weight_refuse_rather_than_reshape(
        snapshot: Any) -> None:
    """The model states the logical shape and the container states the bytes.
    Taking each from the side that holds it is only safe while the ELEMENT
    COUNTS agree; when they do not, these bytes are not this weight."""
    root, _ = snapshot

    class _Wrong:
        def tensors(self, model: Any, config: Any) -> Dict[str, Any]:
            full = SingleFileGguf(root / "unet-Q4_0.gguf").tensors(model, config)
            key = next(k for k in sorted(full)
                       if isinstance(full[k], gguf_torch.QuantizedTensor))
            packed = full[key]
            full[key] = gguf_torch.QuantizedTensor(
                packed.blocks,
                gguf_torch.QuantSpec(packed.spec.qtype, torch.Size((4, 8))))
            return full

    with pytest.raises(ValueError, match="are not this weight"):
        build_denoiser(UNet2DConditionModel, root / "unet", _Wrong(),
                       compute_dtype=torch.bfloat16)


def test_a_packer_named_container_is_routed_to_the_key_mapping(
        tmp_path: Path) -> None:
    """The edge's routing decision. Our own reader answers a container that
    already names its tensors the way the model does; one that does not is sent
    through diffusers' single-file KEY MAPPING — which is the only thing this
    lane borrows, and which refuses BY NAME when it recognizes nothing.
    """
    config = dict(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 32),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"), mid_block_type=None,
        cross_attention_dim=32, norm_num_groups=8,
    )
    torch.manual_seed(5)
    reference = _unet(config)
    config_dir = tmp_path / "unet"
    config_dir.mkdir()
    reference.save_config(config_dir)

    # Our names -> our reader, no mapping consulted, and it BUILDS.
    ours = tmp_path / "ours.gguf"
    _write_gguf(ours, reference.state_dict())
    built = build_denoiser(UNet2DConditionModel, config_dir,
                           SingleFileGguf(ours), compute_dtype=torch.bfloat16)
    assert gguf_torch.gguf_leaves(built)

    # A packer's names -> the mapping, which recognizes none of these and says so.
    foreign = tmp_path / "packer.gguf"
    _write_gguf(foreign, {"model.diffusion_model." + k: v
                          for k, v in reference.state_dict().items()})
    with pytest.raises(ValueError, match="key mapping found no weights"):
        build_denoiser(UNet2DConditionModel, config_dir,
                       SingleFileGguf(foreign), compute_dtype=torch.bfloat16)
