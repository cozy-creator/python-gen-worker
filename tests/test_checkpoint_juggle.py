"""The checkpoint juggle's card-free half: admission, images, validity."""

from __future__ import annotations

import json
import struct
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from gen_worker.models.arena_residency import (  # noqa: E402
    DEFAULT_GRANULARITY,
    dlpack_dtype,
    plan_layout,
)
from gen_worker.models.checkpoint_juggle import (  # noqa: E402
    CheckpointCatalog,
    CheckpointImage,
    CheckpointJuggler,
    JuggleRefusal,
    RegionInvalid,
    RegionValidity,
    ValidityLedger,
    admission_refusal,
    read_manifest,
)
from gen_worker.models.stream_residency import (  # noqa: E402
    discover_leaves,
    own_tensors,
    tensor_bytes,
)

GRAN = DEFAULT_GRANULARITY
MIB = 1 << 20

_SAFETENSORS_SPELLING = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
}


class Net(nn.Module):
    """A lane template: one streaming-sized leaf, small leaves, a buffer."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.big = nn.Linear(1024, 1024, bias=False)
        self.small = nn.Linear(16, 16, bias=False)
        self.norm = nn.LayerNorm(16)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g))

    def forward(self, x: Any) -> Any:  # pragma: no cover
        return self.norm(self.small(self.big(x)))


def write_safetensors(path: Path, tensors: Dict[str, Any]) -> None:
    """A real safetensors file, written the way the format states it."""
    header: Dict[str, Any] = {}
    blobs: List[bytes] = []
    offset = 0
    for key in sorted(tensors):
        t = tensors[key].detach().contiguous()
        raw = t.view(torch.uint8).numpy().tobytes() if t.dtype is torch.bfloat16 else t.numpy().tobytes()
        header[key] = {
            "dtype": _SAFETENSORS_SPELLING[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for blob in blobs:
            fh.write(blob)


def checkpoint_dir(tmp_path: Path, name: str, module: nn.Module, *, dtype: Any = None) -> Path:
    """One component directory holding the module's weights as a checkpoint."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key, tensor in module.state_dict().items():
        tensors[key] = tensor.to(dtype) if dtype is not None else tensor
    write_safetensors(directory / "weights.safetensors", tensors)
    return directory


def specs_for(module: nn.Module, name: str = "root") -> List[Tuple[str, List[Any]]]:
    leaves, _costs, _adapters = discover_leaves([(name, module)])
    out = []
    for leaf_name, leaf in leaves.items():
        slots = []
        for attr, is_param, tensor in own_tensors(leaf):
            code, bits = dlpack_dtype(torch, tensor.dtype)
            slots.append((attr, is_param, tensor_bytes(tensor), tuple(tensor.shape), code, bits))
        out.append((leaf_name, slots))
    return out


def layout_for(module: nn.Module) -> Any:
    return plan_layout(specs_for(module), granularity=GRAN, min_stream_bytes=MIB)


def test_a_checkpoint_of_the_same_architecture_is_admitted(tmp_path: Path) -> None:
    template = Net(seed=0)
    other = Net(seed=1)
    manifest = read_manifest(checkpoint_dir(tmp_path, "b", other))
    assert admission_refusal(layout_for(template), manifest) is None


def test_a_shape_divergence_is_refused_and_names_the_key(tmp_path: Path) -> None:
    template = Net(seed=0)

    class Wider(Net):
        def __init__(self) -> None:
            super().__init__()
            self.big = nn.Linear(1024, 2048, bias=False)

    manifest = read_manifest(checkpoint_dir(tmp_path, "wide", Wider()))
    refusal = admission_refusal(layout_for(template), manifest)
    assert refusal is not None
    assert "big.weight" in refusal and "cross-lane" in refusal


def test_a_missing_tensor_is_refused(tmp_path: Path) -> None:
    template = Net(seed=0)
    directory = checkpoint_dir(tmp_path, "partial", Net(seed=2))
    manifest = read_manifest(directory)
    del manifest["norm.bias"]
    refusal = admission_refusal(layout_for(template), manifest)
    assert refusal is not None and "norm.bias" in refusal


def test_a_float_cast_is_a_representation_choice_not_a_refusal(tmp_path: Path) -> None:
    """fp16-on-disk in an fp32 lane: admitted; ingest casts once."""
    template = Net(seed=0)
    manifest = read_manifest(
        checkpoint_dir(tmp_path, "half", Net(seed=3), dtype=torch.float16)
    )
    assert admission_refusal(layout_for(template), manifest) is None


def test_an_integer_dtype_divergence_is_a_different_artifact(tmp_path: Path) -> None:
    template = Net(seed=0)
    directory = tmp_path / "intly"
    directory.mkdir()
    tensors = {k: t for k, t in Net(seed=4).state_dict().items()}
    tensors["norm.bias"] = torch.arange(16, dtype=torch.int64)
    write_safetensors(directory / "weights.safetensors", tensors)
    refusal = admission_refusal(layout_for(template), read_manifest(directory))
    assert refusal is not None and "different artifact" in refusal


def test_manifest_refuses_unknown_dtype_spellings(tmp_path: Path) -> None:
    directory = tmp_path / "exotic"
    directory.mkdir()
    header = json.dumps(
        {"x": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]}}
    ).encode()
    with (directory / "weights.safetensors").open("wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(b"\x00" * 4)
    with pytest.raises(JuggleRefusal, match="F8_E4M3"):
        read_manifest(directory)


def build_image(
    tmp_path: Path, name: str, template: nn.Module, module: nn.Module, **kw: Any
) -> Tuple[Any, CheckpointImage]:
    layout = layout_for(template)
    manifest = read_manifest(checkpoint_dir(tmp_path, name, module, **kw))
    return layout, CheckpointImage(
        name, layout, manifest, torch_mod=torch, varena_mod=None
    )


def test_image_slots_are_bit_exact_with_the_source_tree(tmp_path: Path) -> None:
    template, other = Net(seed=0), Net(seed=5)
    layout, image = build_image(tmp_path, "b", template, other)
    state = other.state_dict()
    checked = 0
    for region in layout.regions:
        for slot in region.slots:
            key = f"{slot.leaf.partition('.')[2]}.{slot.attr}" if "." in slot.leaf else slot.attr
            assert torch.equal(image.slot_view(slot), state[key])
            checked += 1
    assert checked == len(list(template.state_dict()))


def test_image_digests_are_deterministic_and_checkpoint_distinct(tmp_path: Path) -> None:
    template = Net(seed=0)
    _, image_b1 = build_image(tmp_path, "b1", template, Net(seed=6))
    _, image_b2 = build_image(tmp_path, "b2", template, Net(seed=6))
    _, image_c = build_image(tmp_path, "c", template, Net(seed=7))
    assert image_b1.region_digests == image_b2.region_digests
    assert image_b1.region_digests != image_c.region_digests


def test_ingest_casts_once_and_values_match(tmp_path: Path) -> None:
    template = Net(seed=0)
    other = Net(seed=8)
    layout, image = build_image(tmp_path, "half", template, other, dtype=torch.float16)
    assert image.casts == len(list(other.state_dict()))
    state = other.state_dict()
    for region in layout.regions:
        for slot in region.slots:
            key = f"{slot.leaf.partition('.')[2]}.{slot.attr}" if "." in slot.leaf else slot.attr
            expected = state[key].to(torch.float16).to(torch.float32)
            assert torch.equal(image.slot_view(slot), expected)


def test_the_image_pays_the_layouts_virtual_bytes(tmp_path: Path) -> None:
    """The image is the ARENA's shape, alignment tax included — that is what makes the swap one contiguous copy per region."""
    template = Net(seed=0)
    layout, image = build_image(tmp_path, "b", template, Net(seed=9))
    assert image.nbytes == layout.virtual_bytes


def make_catalog(
    tmp_path: Path, template: nn.Module, mem: Dict[str, int], floor: int
) -> Tuple[Any, CheckpointCatalog]:
    layout = layout_for(template)
    catalog = CheckpointCatalog(
        layout,
        torch_mod=torch,
        varena_mod=None,
        host_floor_bytes=floor,
        mem_available=lambda: mem["available"],
    )
    return layout, catalog


def admit_real(catalog: CheckpointCatalog, tmp_path: Path, name: str, seed: int) -> None:
    catalog.admit(name, read_manifest(checkpoint_dir(tmp_path, name, Net(seed=seed))))


def test_pressure_evicts_lru_first_and_never_the_protected(tmp_path: Path) -> None:
    template = Net(seed=0)
    mem = {"available": 10 << 30}
    layout, catalog = make_catalog(tmp_path, template, mem, floor=4 << 30)
    for i, name in enumerate(("a", "b", "c")):
        admit_real(catalog, tmp_path, name, seed=10 + i)
        assert catalog.ensure_warm(name) is not None
    catalog.protected.add("a")
    mem["available"] = (4 << 30) + layout.virtual_bytes // 2
    admit_real(catalog, tmp_path, "d", seed=13)
    catalog.ensure_warm("d")
    assert catalog.warm("a") is not None
    assert catalog.images.get("b") is None
    assert catalog.evictions >= 1


def test_hysteresis_a_pressure_evicted_image_stays_cold_this_epoch(tmp_path: Path) -> None:
    template = Net(seed=0)
    mem = {"available": 2 << 30}
    _layout, catalog = make_catalog(tmp_path, template, mem, floor=4 << 30)
    admit_real(catalog, tmp_path, "a", seed=20)
    assert catalog.ensure_warm("a") is None
    epoch = catalog.pressure_epoch
    mem["available"] = 64 << 30
    assert catalog.ensure_warm("a") is None
    assert catalog.pressure_epoch == epoch
    admit_real(catalog, tmp_path, "b", seed=21)
    assert catalog.ensure_warm("b") is not None
    catalog.pressure_epoch += 1
    assert catalog.ensure_warm("a") is not None


def test_an_unadmitted_checkpoint_cannot_be_warmed(tmp_path: Path) -> None:
    template = Net(seed=0)
    _layout, catalog = make_catalog(
        tmp_path, template, {"available": 64 << 30}, floor=4 << 30
    )
    with pytest.raises(JuggleRefusal, match="never admitted"):
        catalog.ensure_warm("ghost")


def test_ledger_happy_path_and_all_three_red_arms(tmp_path: Path) -> None:
    layout = layout_for(Net(seed=0))
    ledger = ValidityLedger(layout, "a")
    ledger.assert_servable("a")

    first = layout.regions[0].name
    ledger.begin(first, "b")
    with pytest.raises(RegionInvalid, match="refilling"):
        ledger.assert_servable("a")
    ledger.poison(first)
    with pytest.raises(RegionInvalid, match="idempotent"):
        ledger.assert_servable("a")
    assert ledger.of(first) == (RegionValidity.INVALID, "b")
    ledger.begin(first, "b")
    ledger.complete(first)
    if len(layout.regions) > 1:
        with pytest.raises(RegionInvalid, match="mixed"):
            ledger.assert_servable("b")
    for region in layout.regions[1:]:
        ledger.begin(region.name, "b")
        ledger.complete(region.name)
    ledger.assert_servable("b")


def test_a_partial_switch_is_never_servable_under_either_identity(tmp_path: Path) -> None:
    """The exact franken state the coordinator's requirement names: some regions swapped to B, one died mid-refill."""
    layout = layout_for(Net(seed=0))
    if len(layout.regions) < 2:
        pytest.skip("needs two regions to interleave")
    ledger = ValidityLedger(layout, "a")
    ledger.begin(layout.regions[0].name, "b")
    ledger.complete(layout.regions[0].name)
    ledger.begin(layout.regions[1].name, "b")
    ledger.poison(layout.regions[1].name)
    for identity in ("a", "b"):
        with pytest.raises(RegionInvalid):
            ledger.assert_servable(identity)


class _FillCapture:
    staging = "tensorfs-pinned"

    def __init__(self) -> None:
        self.addresses: List[Tuple[Any, Any]] = []
        self.files: List[Tuple[Any, Any]] = []

    def fill_address(self, source: Any, destination: Any) -> Any:
        self.addresses.append((source, destination))
        return SimpleNamespace(destination_bytes=destination.capacity)

    def fill_files(self, sources: Any, destination: Any) -> Any:
        self.files.append((tuple(sources), destination))
        return SimpleNamespace(destination_bytes=destination.capacity)


def _fake_residency(layout: Any, compile_calls: List[int]) -> Any:
    fake_torch = SimpleNamespace(
        no_grad=nullcontext,
        cuda=SimpleNamespace(synchronize=lambda _device: None),
        compile=lambda value: compile_calls.append(1) or value,
    )
    return SimpleNamespace(
        adopted=True,
        layout=layout,
        _torch=fake_torch,
        _varena=None,
        device=SimpleNamespace(index=0),
        reservation=SimpleNamespace(base_ptr=0x40000000),
        ring=SimpleNamespace(drain=lambda: None),
        is_resident=lambda _name: True,
        _host={},
        _triples={},
        _roots=[],
    )


def test_warm_swaps_reuse_tensorfs_at_stable_addresses_without_recompile(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import gen_worker.models.checkpoint_juggle as checkpoint_juggle

    template = Net(seed=0)
    mem = {"available": 64 << 30}
    layout, catalog = make_catalog(tmp_path, template, mem, floor=4 << 30)
    manifests = {}
    for seed, name in ((30, "a"), (31, "b")):
        manifest = read_manifest(checkpoint_dir(tmp_path, name, Net(seed=seed)))
        manifests[name] = manifest
        catalog.admit(name, manifest)
        assert catalog.ensure_warm(name) is not None

    fills = _FillCapture()
    monkeypatch.setattr(checkpoint_juggle, "CudaFillClient", lambda *_args: fills)
    compile_calls: List[int] = []
    juggler = CheckpointJuggler(
        _fake_residency(layout, compile_calls),
        "a",
        manifests["a"],
        catalog=catalog,
    )

    juggler.switch_to("b")
    juggler.switch_to("a")

    region_count = len(layout.regions)
    first = [destination.pointer for _, destination in fills.addresses[:region_count]]
    second = [destination.pointer for _, destination in fills.addresses[region_count:]]
    assert first == second
    assert first == [0x40000000 + region.offset for region in layout.regions]
    assert compile_calls == []
    assert all(isinstance(source.pointer, int) for source, _ in fills.addresses)
    assert all(isinstance(destination.shape, tuple) for _, destination in fills.addresses)


def test_cold_swap_is_file_records_through_the_same_tensorfs_client(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import gen_worker.models.checkpoint_juggle as checkpoint_juggle

    template = Net(seed=0)
    layout, catalog = make_catalog(
        tmp_path, template, {"available": 0}, floor=4 << 30
    )
    manifests = {
        name: read_manifest(checkpoint_dir(tmp_path, name, Net(seed=seed)))
        for seed, name in ((40, "a"), (41, "b"))
    }
    catalog.admit("b", manifests["b"])

    fills = _FillCapture()
    monkeypatch.setattr(checkpoint_juggle, "CudaFillClient", lambda *_args: fills)
    juggler = CheckpointJuggler(
        _fake_residency(layout, []),
        "a",
        manifests["a"],
        catalog=catalog,
    )

    report = juggler.switch_to("b")

    assert fills.addresses == []
    assert len(fills.files) == len(layout.regions)
    for sources, destination in fills.files:
        assert sum(source.length for source in sources) == destination.capacity
        assert all(
            source.path is None or isinstance(source.path, str) for source in sources
        )
    assert report.tier == "disk-cold"
    assert report.bytes_moved == sum(region.span for region in layout.regions)
