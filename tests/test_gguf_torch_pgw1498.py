"""A layer stack holding GGML block bytes computes what dequantized-eager does.

The claim under test is the whole lane: weights RESIDE quantized, each forward
decodes its own weight, and the answer is the one you would get from a model
whose weights had been dequantized up front — with an attached LoRA on top, and
with the block bytes still byte-identical afterwards.

Everything is synthesized here (``gguf.quants.quantize`` over random values) and
everything runs on CPU. No community checkpoint is downloaded: multi-GB weights
must not transit this machine, and nothing about cast-per-forward needs a GPU to
be true.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")

import torch.nn as nn

from gen_worker.models import gguf_torch
from gen_worker.models.gguf_torch import (
    LoraPatch,
    QuantSpec,
    QuantizedTensor,
    attach_lora,
    detach_lora,
    gguf_leaves,
    install_quantized_weights,
    is_gguf_leaf,
    quantized_bytes,
    read_gguf,
    structural_base,
)

#: The types `gguf.quants.quantize` implements. The K-quants' DECODE is pinned
#: bit-exactly in tests/test_gguf_dequant_pgw1498.py; the packing side is
#: llama-quantize's, so the stack here is built from what we can pack locally.
LINEAR_QTYPE = "Q4_0"
CONV_QTYPE = "Q5_1"
EMBED_QTYPE = "Q8_0"


class Tiny(nn.Module):
    """Every leaf kind the lane puns, in one graph."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.conv = nn.Conv2d(8, 16, kernel_size=2)
        self.norm = nn.LayerNorm(32)

    def forward(self, ids: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        h = self.fc2(torch.relu(self.fc1(h)))
        h = self.norm(h)
        return h.sum() + self.conv(img).sum()


def _pack(weight: torch.Tensor, name: str) -> tuple[QuantizedTensor, torch.Tensor]:
    """``(installable block bytes, what they decode to)`` for one weight.

    A GGML row is the flattened per-output row — that is why the logical shape
    has to travel as metadata rather than being read off the byte array.
    """
    qtype = gguf.GGMLQuantizationType[name]
    rows = weight.detach().numpy().reshape(weight.shape[0], -1).astype(np.float32)
    raw = gguf.quants.quantize(rows, qtype)
    dense = torch.from_numpy(
        gguf.quants.dequantize(raw, qtype).astype(np.float32)).reshape(weight.shape)
    blocks = torch.from_numpy(raw.reshape(-1).copy())
    return QuantizedTensor(blocks, QuantSpec(int(qtype), weight.shape)), dense


def _stack() -> tuple[Tiny, Tiny, dict[str, object]]:
    """A punned model and the dequantized-eager model it must agree with."""
    torch.manual_seed(1498)
    reference = Tiny().to(torch.float32)

    plan = {
        "embed.weight": EMBED_QTYPE,
        "fc1.weight": LINEAR_QTYPE,
        "fc2.weight": LINEAR_QTYPE,
        "conv.weight": CONV_QTYPE,
    }
    tensors: dict[str, object] = {}
    for key, qname in plan.items():
        module = reference.get_submodule(key.rsplit(".", 1)[0])
        packed, dense = _pack(module.weight, qname)
        tensors[key] = packed
        # The reference holds exactly what the blocks decode to, so any
        # difference in the outputs is the LANE's, never the quantizer's.
        with torch.no_grad():
            module.weight.copy_(dense)
    for key in ("fc1.bias", "fc2.bias", "conv.bias"):
        tensors[key] = reference.get_parameter(key).detach().clone()

    torch.manual_seed(1498)
    quantized = Tiny().to(torch.float32)
    install_quantized_weights(quantized, tensors, compute_dtype=torch.float32)
    with torch.no_grad():
        quantized.norm.weight.copy_(reference.norm.weight)
        quantized.norm.bias.copy_(reference.norm.bias)
    return quantized, reference, tensors


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return torch.randint(0, 64, (3, 5)), torch.randn(2, 8, 6, 6)


# ---------------------------------------------------------------------------


def test_cast_per_forward_matches_dequantized_eager() -> None:
    quantized, reference, _ = _stack()
    ids, img = _inputs()
    assert torch.equal(quantized(ids, img), reference(ids, img))


def test_every_quantized_leaf_kind_was_punned() -> None:
    quantized, _, _ = _stack()
    assert sorted(gguf_leaves(quantized)) == ["conv", "embed", "fc1", "fc2"]
    # The pun keeps identity: isinstance still answers for the offload rung,
    # LoRA branch targeting and dtype introspection.
    assert isinstance(quantized.fc1, nn.Linear)
    assert isinstance(quantized.conv, nn.Conv2d)
    assert structural_base(quantized.fc1) is nn.Linear
    assert not is_gguf_leaf(quantized.norm)


def test_weights_stay_quantized_in_memory() -> None:
    quantized, reference, _ = _stack()
    dense = sum(p.numel() * p.element_size()
                for n, p in reference.named_parameters()
                if n.rsplit(".", 1)[-1] == "weight" and not n.startswith("norm"))
    assert quantized_bytes(quantized) * 2 < dense, (
        f"{quantized_bytes(quantized)} block bytes vs {dense} dense — the lane "
        "must at least halve weight residency")
    for leaf in gguf_leaves(quantized).values():
        assert leaf.weight.dtype is torch.uint8


def test_a_dtype_cast_cannot_touch_the_blocks() -> None:
    """``model.to(dtype=...)`` is the fp8-storage lane's standing hazard. Here
    it is structurally impossible: ``nn.Module._apply`` casts only
    floating-point tensors, and block bytes are uint8."""
    quantized, reference, _ = _stack()
    before = {n: m.weight.clone() for n, m in gguf_leaves(quantized).items()}
    quantized.to(torch.bfloat16)
    for name, leaf in gguf_leaves(quantized).items():
        assert leaf.weight.dtype is torch.uint8
        assert torch.equal(leaf.weight, before[name])


def test_shape_does_not_lie() -> None:
    """The reference makes ``.shape`` report the DEQUANTIZED shape so ComfyUI's
    shape-sniffing model detection keeps working. We report the storage shape,
    because every residency walk in the worker reads buffer shapes and the lie
    would over-report a quantized denoiser by the compression ratio."""
    quantized, reference, _ = _stack()
    stored = quantized.fc1.weight
    logical = getattr(quantized.fc1, gguf_torch.SPEC_ATTR)["weight"].shape
    assert tuple(logical) == (64, 32)
    assert tuple(stored.shape) != (64, 32)
    assert stored.numel() < 64 * 32 * 4


def test_the_lane_runs_at_the_production_compute_dtype() -> None:
    """Everything else here pins numerics in fp32. Production serves bf16.

    RED before the fix: the punned forward cast the weight to the leaf's
    declared ``compute_dtype`` while the activation kept its own, so a bf16
    leaf fed an fp32 activation raised `RuntimeError: Input type (float) and
    bias type (c10::BFloat16) should be the same`. The activation decides;
    ``compute_dtype`` answers only for an Embedding, whose int64 indices carry
    no float dtype to read.
    """
    quantized, reference, _ = _stack()
    for leaf in gguf_leaves(quantized).values():
        leaf.compute_dtype = torch.bfloat16
    ids, img = _inputs()

    # Mixed state on purpose: bf16 leaves, fp32 activations. The op runs in the
    # activation's dtype and the embedding — whose input is int64 — in bf16.
    got = quantized(ids, img)
    assert torch.isclose(got.float(), reference(ids, img), rtol=5e-2, atol=5e-2)

    # …and a fully bf16 stack, which is what production actually feeds.
    all_bf16 = quantized(ids, img.to(torch.bfloat16))
    assert all_bf16.dtype is torch.bfloat16
    assert torch.isclose(all_bf16.float(), reference(ids, img),
                         rtol=5e-2, atol=5e-2)


# --- LoRA -----------------------------------------------------------------


def _patch(out_features: int, in_features: int, rank: int = 4) -> LoraPatch:
    torch.manual_seed(31)
    return LoraPatch(down=torch.randn(rank, in_features) * 0.1,
                     up=torch.randn(out_features, rank) * 0.1,
                     scale=0.75)


def test_attached_lora_equals_the_same_delta_merged_into_dense_weights() -> None:
    quantized, reference, _ = _stack()
    patch = _patch(64, 32)

    attach_lora(quantized.fc1, [patch])
    with torch.no_grad():
        reference.fc1.weight.add_(patch.delta(reference.fc1.weight))

    ids, img = _inputs()
    assert torch.equal(quantized(ids, img), reference(ids, img))


def test_attaching_a_lora_leaves_the_quantized_grid_byte_identical() -> None:
    """The reconciliation with the refuse-adapters-on-a-quantized-grid rule,
    asserted rather than argued: the blocks the refusal protects are not
    written. Nothing here can round a delta into a 4-bit grid, because nothing
    here writes to the grid."""
    quantized, _, _ = _stack()
    before = quantized.fc1.weight.clone()
    attach_lora(quantized.fc1, [_patch(64, 32)])
    ids, img = _inputs()
    quantized(ids, img)
    assert torch.equal(quantized.fc1.weight, before)


def test_unpatching_is_instant_and_exact() -> None:
    quantized, reference, _ = _stack()
    ids, img = _inputs()
    clean = quantized(ids, img)

    attach_lora(quantized.fc1, [_patch(64, 32)])
    assert not torch.equal(quantized(ids, img), clean)

    assert detach_lora(quantized) == 1
    assert torch.equal(quantized(ids, img), clean)


def test_a_conv_lora_flattens_into_the_rank_product() -> None:
    quantized, reference, _ = _stack()
    torch.manual_seed(5)
    patch = LoraPatch(down=torch.randn(2, 8, 2, 2) * 0.05,
                      up=torch.randn(16, 2, 1, 1) * 0.05, scale=1.5)
    attach_lora(quantized.conv, [patch])
    with torch.no_grad():
        reference.conv.weight.add_(patch.delta(reference.conv.weight))
    ids, img = _inputs()
    assert torch.equal(quantized(ids, img), reference(ids, img))


def test_a_dense_tensor_refuses_the_attach_path() -> None:
    quantized, _, _ = _stack()
    with pytest.raises(ValueError, match="never held block bytes"):
        attach_lora(quantized.fc1, [_patch(64, 32)], name="bias")
    with pytest.raises(ValueError, match="punned leaf"):
        attach_lora(quantized.norm, [_patch(32, 32)])


# --- the budget dial -------------------------------------------------------


def test_a_full_surplus_decodes_everything_once_and_answers_identically() -> None:
    """The surplus tier: same outputs, no per-forward decode left."""
    quantized, reference, _ = _stack()
    ids, img = _inputs()
    before = quantized(ids, img)

    done = gguf_torch.dequant_ahead(quantized, surplus_bytes=math.inf,
                                    dtype=torch.float32)
    assert sorted(done) == ["conv.weight", "embed.weight", "fc1.weight",
                            "fc2.weight"]
    assert gguf_torch.quantized_bytes(quantized) == 0
    assert gguf_torch.peak_transient_bytes(quantized, dtype=torch.float32) == 0
    assert torch.equal(quantized(ids, img), before)
    assert torch.equal(quantized(ids, img), reference(ids, img))
    for leaf in gguf_leaves(quantized).values():
        assert leaf.weight.dtype is torch.float32


def test_a_zero_surplus_leaves_every_weight_quantized() -> None:
    """The constrained tier is the same call with the dial at zero."""
    quantized, _, _ = _stack()
    resident = gguf_torch.quantized_bytes(quantized)
    assert gguf_torch.dequant_ahead(quantized, surplus_bytes=0,
                                    dtype=torch.float32) == []
    assert gguf_torch.quantized_bytes(quantized) == resident


def test_a_partial_surplus_graduates_largest_first_and_shrinks_the_transient() -> None:
    """The dial between the endpoints. Largest first is the ordering that also
    lowers the transient headroom a fit plan must reserve, which is the reason
    it is the ordering."""
    quantized, reference, _ = _stack()
    ids, img = _inputs()
    before = quantized(ids, img)
    assert gguf_torch.peak_transient_bytes(quantized, dtype=torch.float32) == 64 * 32 * 4

    # Buy the three 2048-element weights (embed, fc1, fc2) and not the 512-element
    # conv, so what is left is strictly smaller than what was bought.
    price = sum(64 * 32 * 4 - getattr(quantized, n).weight.numel()
                for n in ("embed", "fc1", "fc2"))
    done = gguf_torch.dequant_ahead(quantized, surplus_bytes=price,
                                    dtype=torch.float32)

    assert sorted(done) == ["embed.weight", "fc1.weight", "fc2.weight"]
    assert gguf_torch.quantized_bytes(quantized) > 0  # conv still pays per forward
    assert gguf_torch.peak_transient_bytes(quantized, dtype=torch.float32) == 16 * 8 * 2 * 2 * 4
    assert torch.equal(quantized(ids, img), before)
    assert torch.equal(quantized(ids, img), reference(ids, img))


def test_lora_semantics_are_identical_on_both_sides_of_the_dial() -> None:
    """A weight decoded at load takes the SAME attach path as one decoded per
    forward — the tier must not be observable through the adapter."""
    per_forward, reference, _ = _stack()
    at_load, _, _ = _stack()
    gguf_torch.dequant_ahead(at_load, surplus_bytes=math.inf, dtype=torch.float32)

    patch = _patch(64, 32)
    attach_lora(per_forward.fc1, [patch])
    attach_lora(at_load.fc1, [patch])
    with torch.no_grad():
        reference.fc1.weight.add_(patch.delta(reference.fc1.weight))

    ids, img = _inputs()
    assert torch.equal(at_load(ids, img), per_forward(ids, img))
    assert torch.equal(at_load(ids, img), reference(ids, img))

    assert detach_lora(at_load) == 1
    assert detach_lora(per_forward) == 1
    assert torch.equal(at_load(ids, img), per_forward(ids, img))


def test_the_dial_never_requantizes() -> None:
    """Turning the dial up is one-way by construction: there is no path back to
    block bytes, which is the rung.py rule (the ladder SELECTS artifacts)."""
    quantized, _, _ = _stack()
    gguf_torch.dequant_ahead(quantized, surplus_bytes=math.inf,
                             dtype=torch.float32)
    assert gguf_torch.dequant_ahead(quantized, surplus_bytes=math.inf,
                                    dtype=torch.float32) == []
    assert not hasattr(gguf_torch, "requantize")


def test_the_fuse_gate_refuses_a_gguf_leaf_instead_of_inventing_a_grid() -> None:
    """The other half of the reconciliation, at the site of the check.

    ``adapter_fidelity.grid_of_module`` reads the grid off the module's
    ``weight``. On a GGML leaf that is BLOCK BYTES, so the unguarded answer
    would be a "uint8 grid" — a fuse gated against a fiction. It refuses.
    """
    from gen_worker.models import adapter_fidelity

    quantized, _, _ = _stack()
    with pytest.raises(ValueError, match="no fuse into a quantized grid"):
        adapter_fidelity.grid_of_module(quantized.fc1,
                                        path=adapter_fidelity.PATH_FUSE)
    # …and an ordinary Linear is untouched by the new arm.
    plain = nn.Linear(4, 4)
    assert adapter_fidelity.grid_of_module(
        plain, path=adapter_fidelity.PATH_FUSE).dtype == "float32"


# --- install refusals ------------------------------------------------------


def test_a_leaf_with_no_decode_forward_refuses_block_bytes() -> None:
    model = Tiny()
    packed, _ = _pack(torch.randn(32, 32), LINEAR_QTYPE)
    with pytest.raises(ValueError, match="no decode-at-use-site forward"):
        install_quantized_weights(model, {"norm.weight": packed},
                                  compute_dtype=torch.float32)


def test_an_unknown_module_path_refuses() -> None:
    model = Tiny()
    packed, _ = _pack(torch.randn(32, 32), LINEAR_QTYPE)
    with pytest.raises(KeyError, match="name no module"):
        install_quantized_weights(model, {"nope.weight": packed},
                                  compute_dtype=torch.float32)


def test_an_unserveable_qtype_refuses_at_the_spec() -> None:
    with pytest.raises(NotImplementedError, match="IQ2_XXS"):
        QuantSpec(int(gguf.GGMLQuantizationType.IQ2_XXS), torch.Size((4, 256)))


def test_installed_blocks_do_not_alias_the_source() -> None:
    """The mmap-release property: an installed buffer must own its bytes."""
    model = Tiny()
    packed, _ = _pack(torch.randn(64, 32), LINEAR_QTYPE)
    install_quantized_weights(model, {"fc1.weight": packed},
                              compute_dtype=torch.float32)
    assert model.fc1.weight.data_ptr() != packed.blocks.data_ptr()


# --- the .gguf edge --------------------------------------------------------


def _write_gguf(path, raw, qtype) -> None:
    writer = gguf.GGUFWriter(str(path), arch="llama")
    # `raw_shape` is the BYTE shape; the writer derives the logical shape from
    # the block geometry and stores it in GGUF's reversed `ne` order.
    writer.add_tensor("fc1.weight", raw, raw_shape=raw.shape, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _packed_weight():
    torch.manual_seed(11)
    weight = torch.randn(64, 32)
    qtype = gguf.GGMLQuantizationType[LINEAR_QTYPE]
    raw = gguf.quants.quantize(weight.numpy().astype(np.float32), qtype)
    dense = torch.from_numpy(gguf.quants.dequantize(raw, qtype).astype(np.float32))
    return raw, qtype, dense


def test_a_written_gguf_round_trips_through_the_edge_reader(tmp_path) -> None:
    """A real container, written and read back — the community-ingest edge,
    proved without downloading anything."""
    raw, qtype, dense = _packed_weight()
    path = tmp_path / "tiny.gguf"
    _write_gguf(path, raw, qtype)

    read = read_gguf(path)
    assert read.architecture == "llama"
    got = read.tensors["fc1.weight"]
    assert isinstance(got, QuantizedTensor)
    assert tuple(got.shape) == (64, 32)
    assert got.qtype == int(qtype)

    model = Tiny()
    install_quantized_weights(model, {"fc1.weight": got},
                              compute_dtype=torch.float32)
    x = torch.randn(3, 32)
    assert torch.equal(model.fc1(x),
                       torch.nn.functional.linear(x, dense, model.fc1.bias))


def test_the_served_path_reads_block_bytes_out_of_the_cas(tmp_path) -> None:
    """The NORMALIZED path, Paul 2026-08-19: the store hands back per-tensor
    block bytes and the serving side never sees a container.

    tensorfs' ``gguf-v1`` planner already cuts the file into one region per
    tensor, and a ``TensorView`` already carries the GGML type and the block
    geometry. Nothing is dequantized on the way in — the bytes that land in the
    buffer are the bytes that were ingested.
    """
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.tensorfs.tensors import open_tensors

    raw, qtype, dense = _packed_weight()
    staged = tmp_path / "staged"
    staged.mkdir()
    _write_gguf(staged / "model.gguf", raw, qtype)

    cas = LocalCAS(tmp_path / "cas")
    manifest = cas.ingest_repository(staged)
    with open_tensors(cas, manifest) as reader:
        view = reader["fc1.weight"]
        assert view.format == "gguf-v1"
        assert view.dtype == LINEAR_QTYPE
        assert view.block.quantized
        tensors = gguf_torch.quantized_tensors_from_views({"fc1.weight": view})

    packed = tensors["fc1.weight"]
    assert isinstance(packed, QuantizedTensor)
    assert tuple(packed.shape) == (64, 32)
    assert torch.equal(packed.blocks,
                       torch.from_numpy(raw.reshape(-1).copy()))

    model = Tiny()
    install_quantized_weights(model, tensors, compute_dtype=torch.float32)
    x = torch.randn(3, 32)
    assert torch.equal(model.fc1(x),
                       torch.nn.functional.linear(x, dense, model.fc1.bias))
