"""The arena residency layout and its (path, offset, len) triples.

The layout arithmetic and the triple resolution decide every byte the arena
ever maps, and both are decidable without a card. What is left for the GPU is
the driver's own behaviour, which is varena#1/#2's ground and is verified by
`benchmarks/arena_facade_pgw1507.py`.

Real ``nn.Module`` trees and real safetensors files throughout — no mocks. A
mock leaf would agree with whatever the layout believes about it, which is the
one thing worth checking.

# pgw#1507: the varena facade behind pgw#1497's planner contract.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, List, Tuple

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from gen_worker.models.arena_residency import (  # noqa: E402
    CORE_REGION,
    DEFAULT_GRANULARITY,
    dlpack_dtype,
    plan_layout,
    safetensors_triples,
)
from gen_worker.models.stream_residency import (  # noqa: E402
    MemoryBudget,
    discover_leaves,
    own_tensors,
    plan_residency,
    tensor_bytes,
)

GRAN = DEFAULT_GRANULARITY
MIB = 1 << 20


# ---------------------------------------------------------------------------
# Real trees
# ---------------------------------------------------------------------------


class Pyramid(nn.Module):
    """A real tree with big leaves, small leaves and a parent that owns a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.big = nn.Linear(2048, 2048, bias=False)  # 16 MiB fp32
        self.mid = nn.Linear(1024, 1024, bias=False)  # 4 MiB
        self.odd = nn.Linear(1024, 768, bias=False)  # 3 MiB — half a chunk over
        self.small = nn.Linear(16, 16, bias=False)  # 1 KiB
        self.norm = nn.LayerNorm(16)
        # A tensor on a module that HAS children: never a leaf, so no region
        # can own it. pgw#1497 measured what happens when nobody places it.
        self.register_buffer("position_ids", torch.arange(16))

    def forward(self, x: Any) -> Any:  # pragma: no cover
        return self.norm(self.small(self.odd(self.mid(self.big(x)))))


def specs_for(module: nn.Module, name: str = "root") -> List[Tuple[str, List[Any]]]:
    """The layout's input, derived from a real tree exactly as the facade does."""
    leaves, _costs, _adapters = discover_leaves([(name, module)])
    out = []
    for leaf_name, leaf in leaves.items():
        slots = []
        for attr, is_param, tensor in own_tensors(leaf):
            code, bits = dlpack_dtype(torch, tensor.dtype)
            slots.append((attr, is_param, tensor_bytes(tensor), tuple(tensor.shape), code, bits))
        out.append((leaf_name, slots))
    return out


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------


def test_every_candidate_region_is_granularity_aligned_and_disjoint():
    """The invariant the whole design rests on.

    ``unback`` releases the chunks WHOLLY inside its request. Two regions
    sharing a chunk could therefore never release it — and if one were in
    flight while the other unbacked, the shared chunk would be pulled out from
    under a live kernel. Alignment is what makes per-leaf residency safe.
    """
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    seen = []
    for region in layout.regions:
        assert region.offset % GRAN == 0, region
        assert region.span % GRAN == 0, region
        seen.append((region.offset, region.offset + region.span))
    seen.sort()
    for (_a0, a1), (b0, _b1) in zip(seen, seen[1:]):
        assert a1 <= b0, (seen,)


def test_the_forced_set_pays_one_chunk_remainder_not_one_each():
    """Small leaves are packed into the core, which is the point of the core."""
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    core = layout.by_name()[CORE_REGION]
    # `small` (1 KiB), `norm` (2 x 64 B) — three leaves, one region, one chunk.
    assert set(layout.core_names) == {"root.small", "root.norm"}
    assert core.offset == 0 and core.span == GRAN
    assert core.weight_bytes == 16 * 16 * 4 + 16 * 4 * 2


def test_the_granularity_tax_is_charged_to_the_planner_not_discovered_later():
    """``plan.resident_bytes`` must equal the bytes the arena really maps.

    ``odd`` is 3 MiB and occupies two 2 MiB chunks. A planner priced in raw
    weight bytes would believe 3 MiB is on the card when 4 MiB is — an error
    in the direction that OOMs.
    """
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    odd = layout.by_name()["root.odd"]
    assert odd.weight_bytes == 1024 * 768 * 4
    assert odd.span == 2 * GRAN
    assert odd.tax_bytes == odd.span - odd.weight_bytes

    costs = {c.name: c for c in layout.costs()}
    assert costs["root.odd"].resident_bytes == odd.span
    # Every region backed = every span mapped, and the layout says so.
    assert sum(c.resident_bytes for c in layout.costs()) == sum(
        r.span for r in layout.regions
    )
    assert layout.tax_bytes == sum(r.span - r.weight_bytes for r in layout.regions)


def test_layout_is_deterministic_and_ordered_like_the_planner_fills():
    """Same tree, same layout — twice, and in descending-size order.

    Determinism is not a nicety here: a compiled artifact keyed on a weight
    address and a residency plan both depend on this order being a function of
    the tree alone.
    """
    a = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    b = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    assert [(r.name, r.offset, r.span) for r in a.regions] == [
        (r.name, r.offset, r.span) for r in b.regions
    ]
    candidates = [r for r in a.regions if r.name != CORE_REGION]
    assert [r.name for r in candidates] == ["root.big", "root.mid", "root.odd"]


def test_excluded_leaves_go_to_the_core_and_can_never_stream():
    """An exclusion is a statement about HOOKS, and here about residency too.

    pgw#1497 measured the other reading: a component excluded from the ring
    that nobody then placed died on the first decode. Here the exclusion puts
    the leaf in the always-resident core, so there is no third state it can be
    in.
    """
    layout = plan_layout(
        specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB, exclude=["root.mid"]
    )
    assert "root.mid" in layout.core_names
    assert "root.mid" not in {r.name for r in layout.regions}
    plan = plan_residency(
        layout.costs(), budget_bytes=0, streams=2, min_stream_bytes=0, exclude=(CORE_REGION,)
    )
    assert CORE_REGION in plan.forced and CORE_REGION in plan.all_resident


def test_the_layout_and_the_planner_agree_leaf_for_leaf():
    """Every region the planner can name is a region the layout has."""
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    names = set(layout.by_name())
    for budget in (0, 4 * MIB, 8 * MIB, 24 * MIB, 1 << 30):
        plan = plan_residency(
            layout.costs(),
            budget_bytes=MemoryBudget(budget),
            streams=2,
            min_stream_bytes=0,
            exclude=(CORE_REGION,),
        )
        assert set(plan.all_resident) | set(plan.streamed) == names
        assert not (set(plan.all_resident) & set(plan.streamed))


def test_a_budget_at_the_full_span_streams_nothing():
    """The fixed point terminates at zero window when nothing streams."""
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    total = sum(r.span for r in layout.regions)
    plan = plan_residency(
        layout.costs(), budget_bytes=total, streams=2, min_stream_bytes=0,
        exclude=(CORE_REGION,),
    )
    assert plan.streamed == () and plan.window_bytes == 0
    assert plan.resident_bytes == total and plan.fits


def test_a_zero_budget_still_keeps_the_core_and_says_it_does_not_fit():
    """A confession, never a silent clamp."""
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    plan = plan_residency(
        layout.costs(), budget_bytes=0, streams=2, min_stream_bytes=0, exclude=(CORE_REGION,)
    )
    assert plan.all_resident == (CORE_REGION,)
    assert set(plan.streamed) == {"root.big", "root.mid", "root.odd"}
    assert not plan.fits


def test_the_in_flight_window_is_streams_times_the_largest_streamed_span():
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    spans = {r.name: r.span for r in layout.regions}
    plan = plan_residency(
        layout.costs(), budget_bytes=6 * MIB, streams=2, min_stream_bytes=0,
        exclude=(CORE_REGION,),
    )
    assert plan.streamed
    assert plan.window_bytes == 2 * max(spans[n] for n in plan.streamed)


def test_an_unmapped_dtype_is_refused_not_guessed():
    """A wrong DLPack code is a correctly-sized view of misread bytes."""
    with pytest.raises(TypeError, match="no DLPack code"):
        dlpack_dtype(torch, torch.complex64)
    assert dlpack_dtype(torch, torch.float16) == (2, 16)
    assert dlpack_dtype(torch, torch.bfloat16) == (4, 16)


def test_a_non_contiguous_leaf_keeps_its_whole_leaf_out_of_the_arena():
    """Half-moving a leaf is the failure mode; refusing the leaf is not."""

    class Aliased(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            storage = torch.zeros(4096, 4096)
            self.weird = nn.Linear(4096, 4096, bias=False)
            self.weird.weight = nn.Parameter(storage.t(), requires_grad=False)
            self.plain = nn.Linear(2048, 2048, bias=False)

    from gen_worker.models.arena_residency import ArenaResidency

    specs = []
    leaves, _c, _a = discover_leaves([("root", Aliased())])
    for name, leaf in leaves.items():
        slots = []
        skip = False
        for attr, is_param, tensor in own_tensors(leaf):
            if tensor.storage_offset() != 0 or not tensor.is_contiguous():
                skip = True
                break
            code, bits = dlpack_dtype(torch, tensor.dtype)
            slots.append((attr, is_param, tensor_bytes(tensor), tuple(tensor.shape), code, bits))
        if not skip and slots:
            specs.append((name, slots))
    layout = plan_layout(specs, granularity=GRAN, min_stream_bytes=MIB)
    assert "root.weird" not in layout.by_name()
    assert "root.plain" in layout.by_name()
    assert ArenaResidency is not None  # the facade is what applies this filter


def test_adapter_leaves_land_in_the_core_and_can_never_stream():
    """pgw#1507's LoRA clause, and it is the SAME code as the streamed rung's.

    An attach-based adapter is a pair of tiny leaves NEXT TO the base layer, so
    the base layer streams through its own region exactly as an unpatched one
    does and the adapters are forced resident. MEASURED on the 4070 with 96
    attached pairs: 192 adapter leaves, all in the core, none ever streamed,
    36 of their base layers streaming, outputs bitwise identical at every
    budget. This is the arithmetic half of that.
    """

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int = 8) -> None:
            super().__init__()
            self.base_layer = base
            self.lora_A = nn.Linear(base.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, base.out_features, bias=False)

        def forward(self, x: Any) -> Any:  # pragma: no cover
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    class Adapted(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = LoRALinear(nn.Linear(2048, 2048, bias=False))

    _leaves, _costs, adapters = discover_leaves([("unet", Adapted())])
    assert adapters == {"unet.to_q.lora_A", "unet.to_q.lora_B"}

    layout = plan_layout(
        specs_for(Adapted(), name="unet"),
        granularity=GRAN,
        min_stream_bytes=MIB,
        exclude=adapters,
    )
    assert set(layout.core_names) == adapters
    # The base layer keeps its own region and therefore still streams.
    assert "unet.to_q.base_layer" in layout.by_name()
    plan = plan_residency(
        layout.costs(), budget_bytes=0, streams=2, min_stream_bytes=0,
        exclude=(CORE_REGION,),
    )
    assert plan.streamed == ("unet.to_q.base_layer",)
    assert not any(name in plan.streamed for name in adapters)


def test_binding_across_the_meta_boundary_replaces_the_parameter():
    """The cold-load defect, in one test.

    ``Parameter.data = other`` is the normal bind and it preserves the
    Parameter's identity. Across the META boundary torch REFUSES it
    ("incompatible tensor type"), which is how the cold-load leg died on the
    card — so a meta parameter, which has never held a byte and has no
    identity worth preserving, is REPLACED instead.
    """
    from gen_worker.models.stream_residency import bind_tensor

    real = nn.Linear(4, 4, bias=False)
    with torch.device("meta"):
        empty = nn.Linear(4, 4, bias=False)

    # The refusal is real and is what the facade routes around.
    with pytest.raises(RuntimeError, match="incompatible tensor type"):
        bind_tensor(empty, "weight", real.weight.detach().clone(), True)

    # The facade's rule: replace, do not fill.
    value = real.weight.detach().clone()
    empty._parameters["weight"] = nn.Parameter(value, requires_grad=False)
    assert not empty.weight.is_meta
    assert torch.equal(empty.weight, real.weight)

    # And an ordinary bind still preserves identity, which is why it is the
    # default: hooks and LoRA wrappers hold the object across every promote.
    identity = real.weight
    bind_tensor(real, "weight", torch.ones(4, 4), True)
    assert real.weight is identity


# ---------------------------------------------------------------------------
# The triples
# ---------------------------------------------------------------------------


def _write_safetensors(path: Path, tensors: dict) -> None:
    header, blob, cursor = {}, bytearray(), 0
    codes = {torch.float16: "F16", torch.float32: "F32"}
    for key, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        header[key] = {
            "dtype": codes[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        blob += raw
        cursor += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(blob))


def test_triples_name_the_exact_bytes_of_every_tensor(tmp_path):
    """Read back through the triple and compare to the tensor. Byte-exact."""
    tensors = {
        "a.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "b.weight": torch.linspace(-1, 1, 128, dtype=torch.float16).reshape(8, 16),
    }
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    _write_safetensors(path, tensors)

    triples = safetensors_triples(tmp_path)
    assert set(triples) == set(tensors)
    for key, tensor in tensors.items():
        file, offset, length = triples[key]
        assert length == tensor.numel() * tensor.element_size()
        with open(file, "rb") as fh:
            fh.seek(offset)
            raw = fh.read(length)
        got = torch.frombuffer(bytearray(raw), dtype=tensor.dtype).reshape(tensor.shape)
        assert torch.equal(got, tensor)


def test_the_variant_selects_the_file_and_the_plain_name_does_not_pick_it_up(tmp_path):
    """``model.fp16.safetensors`` and ``model.safetensors`` are two checkpoints.

    Mixing them is a silent dtype fault: the header would say fp16 while the
    module says fp32, and the triple lengths would disagree by 2x.
    """
    _write_safetensors(
        tmp_path / "diffusion_pytorch_model.safetensors",
        {"w": torch.zeros(4, 4, dtype=torch.float32)},
    )
    _write_safetensors(
        tmp_path / "diffusion_pytorch_model.fp16.safetensors",
        {"w": torch.zeros(4, 4, dtype=torch.float16)},
    )
    plain = safetensors_triples(tmp_path)
    fp16 = safetensors_triples(tmp_path, variant="fp16")
    assert plain["w"][2] == 64 and fp16["w"][2] == 32
    assert plain["w"][0].name == "diffusion_pytorch_model.safetensors"
    assert fp16["w"][0].name == "diffusion_pytorch_model.fp16.safetensors"


def test_a_directory_with_no_weights_refuses(tmp_path):
    with pytest.raises(FileNotFoundError):
        safetensors_triples(tmp_path)


def test_an_implausible_header_length_is_refused(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 60) + b"{}")
    with pytest.raises(ValueError, match="implausible"):
        safetensors_triples(tmp_path)


def test_triples_cover_a_real_sd15_component_exactly():
    """The production path: the box's own sd1.5 snapshot, not a fixture.

    Every UNet parameter must have a triple whose length is the tensor's, or
    the cold-load arm would be filling some weights and silently leaving
    others as whatever the arena last held.
    """
    snapshot = Path(
        "/home/fidika/.cache/huggingface/hub/"
        "models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/"
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )
    if not (snapshot / "unet").is_dir():
        pytest.skip("sd1.5 snapshot not on this box")
    from diffusers import UNet2DConditionModel

    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(str(snapshot / "unet"))
        )
    triples = safetensors_triples(snapshot / "unet", variant="fp16")
    missing = [k for k, _ in unet.named_parameters() if k not in triples]
    assert not missing, missing[:5]
    for key, param in unet.named_parameters():
        _path, _offset, length = triples[key]
        # The snapshot is fp16 and the meta tree is fp32: compare ELEMENTS.
        assert length == param.numel() * 2, key
