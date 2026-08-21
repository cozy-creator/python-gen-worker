"""The arena residency layout and its (path, offset, len) triples."""

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


class Pyramid(nn.Module):
    """A real tree with big leaves, small leaves and a parent that owns a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.big = nn.Linear(2048, 2048, bias=False)
        self.mid = nn.Linear(1024, 1024, bias=False)
        self.odd = nn.Linear(1024, 768, bias=False)
        self.small = nn.Linear(16, 16, bias=False)
        self.norm = nn.LayerNorm(16)
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


def test_every_candidate_region_is_granularity_aligned_and_disjoint():
    """The invariant the whole design rests on."""
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
    assert set(layout.core_names) == {"root.small", "root.norm"}
    assert core.offset == 0 and core.span == GRAN
    assert core.weight_bytes == 16 * 16 * 4 + 16 * 4 * 2


def test_the_granularity_tax_is_charged_to_the_planner_not_discovered_later():
    """``plan.resident_bytes`` must equal the bytes the arena really maps."""
    layout = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    odd = layout.by_name()["root.odd"]
    assert odd.weight_bytes == 1024 * 768 * 4
    assert odd.span == 2 * GRAN
    assert odd.tax_bytes == odd.span - odd.weight_bytes

    costs = {c.name: c for c in layout.costs()}
    assert costs["root.odd"].resident_bytes == odd.span
    assert sum(c.resident_bytes for c in layout.costs()) == sum(
        r.span for r in layout.regions
    )
    assert layout.tax_bytes == sum(r.span - r.weight_bytes for r in layout.regions)


def test_layout_is_deterministic_and_ordered_like_the_planner_fills():
    """Same tree, same layout — twice, and in descending-size order."""
    a = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    b = plan_layout(specs_for(Pyramid()), granularity=GRAN, min_stream_bytes=MIB)
    assert [(r.name, r.offset, r.span) for r in a.regions] == [
        (r.name, r.offset, r.span) for r in b.regions
    ]
    candidates = [r for r in a.regions if r.name != CORE_REGION]
    assert [r.name for r in candidates] == ["root.big", "root.mid", "root.odd"]


def test_excluded_leaves_go_to_the_core_and_can_never_stream():
    """An exclusion is a statement about HOOKS, and here about residency too."""
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
    assert ArenaResidency is not None


def test_adapter_leaves_land_in_the_core_and_can_never_stream():

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
    assert "unet.to_q.base_layer" in layout.by_name()
    plan = plan_residency(
        layout.costs(), budget_bytes=0, streams=2, min_stream_bytes=0,
        exclude=(CORE_REGION,),
    )
    assert plan.streamed == ("unet.to_q.base_layer",)
    assert not any(name in plan.streamed for name in adapters)


def test_binding_across_the_meta_boundary_replaces_the_parameter():
    """The cold-load defect, in one test."""
    from gen_worker.models.stream_residency import bind_tensor

    real = nn.Linear(4, 4, bias=False)
    with torch.device("meta"):
        empty = nn.Linear(4, 4, bias=False)

    with pytest.raises(RuntimeError, match="incompatible tensor type"):
        bind_tensor(empty, "weight", real.weight.detach().clone(), True)

    value = real.weight.detach().clone()
    empty._parameters["weight"] = nn.Parameter(value, requires_grad=False)
    assert not empty.weight.is_meta
    assert torch.equal(empty.weight, real.weight)

    identity = real.weight
    bind_tensor(real, "weight", torch.ones(4, 4), True)
    assert real.weight is identity


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
    """Read back through the triple and compare to the tensor."""
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
    """``model.fp16.safetensors`` and ``model.safetensors`` are two checkpoints."""
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

