"""gw#395/#396 — per-tensor streaming dtype cast + fp8-E4M3 storage flavor.

Covers: shard layout + index + __metadata__ preservation, determinism and
semantic equality vs an in-memory reference, fp8 eligibility/clamping, the
snapshot-level helpers, and the peak anonymous-RSS bound (< 2x the largest
tensor — the property that lets a 100 GB model cast on a 32 GB pod).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
from safetensors import safe_open
from safetensors.torch import save_file

from gen_worker.convert.writer import (
    fp8_cast_eligible,
    plan_shards,
    streaming_cast_snapshot,
    streaming_dtype_cast,
    streaming_fp8_snapshot,
    streaming_fp8_storage_cast,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_sharded_fixture(root: Path, *, seed: int = 7) -> Path:
    """3-shard fp32 fixture with an index, mixed float/int tensors and
    __metadata__ on every shard. Returns the index path."""
    g = torch.Generator().manual_seed(seed)
    shards = {
        "model-00001-of-00003.safetensors": {
            "blocks.0.attn.to_q.weight": torch.randn(64, 64, generator=g),
            "blocks.0.attn.to_q.bias": torch.randn(64, generator=g),
            "blocks.0.norm1.weight": torch.randn(64, generator=g),
        },
        "model-00002-of-00003.safetensors": {
            "blocks.1.ff.net.0.proj.weight": torch.randn(128, 64, generator=g),
            "blocks.1.step_counter": torch.arange(10, dtype=torch.int64),
        },
        "model-00003-of-00003.safetensors": {
            "pos_embed.proj.weight": torch.randn(32, 32, generator=g),
            "blocks.2.attn.to_v.weight": torch.randn(64, 64, generator=g),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total = 0
    for shard_name, tensors in shards.items():
        save_file(tensors, str(root / shard_name), metadata={"format": "pt", "origin": "fixture"})
        for name, t in tensors.items():
            weight_map[name] = shard_name
            total += t.numel() * t.element_size()
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}))
    return index


def _all_tensors(entry: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if entry.name.endswith(".index.json"):
        weight_map = json.loads(entry.read_text())["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = [entry.name]
    for shard in shard_files:
        with safe_open(str(entry.parent / shard), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def _tree_digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Streaming cast — correctness
# ---------------------------------------------------------------------------

def test_cast_semantic_equality_vs_in_memory_reference(tmp_path: Path) -> None:
    """Streaming output == the straightforward load-everything reference,
    tensor for tensor, plus dtype rules (floats cast, ints untouched)."""
    index = _write_sharded_fixture(tmp_path / "src")
    result = streaming_dtype_cast(index, tmp_path / "out", target_dtype=torch.bfloat16)

    src = _all_tensors(index)
    reference = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                 for k, v in src.items()}
    got = _all_tensors(result["index_path"] or result["output_paths"][0])

    assert set(got) == set(reference)
    for k in reference:
        assert got[k].dtype == reference[k].dtype, k
        assert torch.equal(got[k], reference[k]), k
    assert got["blocks.1.step_counter"].dtype == torch.int64
    assert result["tensor_count"] == len(src)


def test_cast_is_byte_deterministic_across_runs(tmp_path: Path) -> None:
    index = _write_sharded_fixture(tmp_path / "src")
    r1 = streaming_dtype_cast(index, tmp_path / "out1", target_dtype=torch.float16)
    r2 = streaming_dtype_cast(index, tmp_path / "out2", target_dtype=torch.float16)
    files1 = list(r1["output_paths"]) + ([r1["index_path"]] if r1["index_path"] else [])
    files2 = list(r2["output_paths"]) + ([r2["index_path"]] if r2["index_path"] else [])
    assert [p.name for p in files1] == [p.name for p in files2]
    assert _tree_digest(files1) == _tree_digest(files2)


def test_cast_preserves_source_metadata(tmp_path: Path) -> None:
    index = _write_sharded_fixture(tmp_path / "src")
    result = streaming_dtype_cast(index, tmp_path / "out", target_dtype=torch.bfloat16)
    assert result["metadata"].get("origin") == "fixture"
    for shard in result["output_paths"]:
        with safe_open(str(shard), framework="pt") as f:
            md = f.metadata()
            assert md and md.get("format") == "pt" and md.get("origin") == "fixture"


def test_cast_sharded_output_layout_and_index(tmp_path: Path) -> None:
    """A small shard threshold forces a multi-shard output whose index the
    stock loaders (and plan_shards) agree on."""
    index = _write_sharded_fixture(tmp_path / "src")
    result = streaming_dtype_cast(
        index, tmp_path / "out", target_dtype=torch.float32, shard_threshold=36_000)
    assert result["index_path"] is not None
    payload = json.loads(result["index_path"].read_text())
    sizes = {k: t.numel() * 4 if t.is_floating_point() else t.numel() * t.element_size()
             for k, t in _all_tensors(index).items()}
    expected_plan = plan_shards(sizes, max_shard_bytes=36_000, shard_prefix="model")
    assert payload["weight_map"] == expected_plan.weight_map
    assert payload["metadata"]["total_size"] == expected_plan.total_size
    got = _all_tensors(result["index_path"])
    assert set(got) == set(sizes)


def test_cast_single_file_input(tmp_path: Path) -> None:
    src = tmp_path / "one.safetensors"
    save_file({"w": torch.randn(8, 8)}, str(src), metadata={"k": "v"})
    result = streaming_dtype_cast(src, tmp_path / "out", target_dtype=torch.float16)
    assert result["index_path"] is None
    got = _all_tensors(result["output_paths"][0])
    assert got["w"].dtype == torch.float16


# ---------------------------------------------------------------------------
# fp8-E4M3 storage cast
# ---------------------------------------------------------------------------

def test_fp8_eligibility_rules() -> None:
    assert fp8_cast_eligible("blocks.0.attn.to_q.weight", "F32", [64, 64])
    assert fp8_cast_eligible("down_blocks.0.resnets.0.conv1.weight", "F16", [4, 4, 3, 3])
    # skip patterns
    assert not fp8_cast_eligible("blocks.0.norm1.weight", "F32", [64, 64])
    assert not fp8_cast_eligible("pos_embed.proj.weight", "F32", [32, 32])
    assert not fp8_cast_eligible("time_embedding.linear_1.weight", "F32", [64, 64])
    assert not fp8_cast_eligible("proj_out.weight", "F32", [64, 64])
    # 1-D / biases / non-weight / non-float stay
    assert not fp8_cast_eligible("blocks.0.attn.to_q.bias", "F32", [64])
    assert not fp8_cast_eligible("blocks.0.attn.to_q.weight", "I64", [64, 64])
    assert not fp8_cast_eligible("blocks.1.step_counter", "I64", [10])
    # already-fp8 input is never re-quantized
    assert not fp8_cast_eligible("blocks.0.attn.to_q.weight", "F8_E4M3", [64, 64])


def test_fp8_cast_values_clamp_and_selectivity(tmp_path: Path) -> None:
    index = _write_sharded_fixture(tmp_path / "src")
    # Inject an out-of-range weight to prove clamping (fp8-e4m3 has no inf;
    # an unclamped 1000.0 becomes NaN).
    hot = torch.full((16, 16), 1000.0)
    save_file(
        {"blocks.3.attn.to_k.weight": hot},
        str(tmp_path / "src" / "model-00004-of-00004.safetensors"),
        metadata={"format": "pt"},
    )
    payload = json.loads((tmp_path / "src" / "model.safetensors.index.json").read_text())
    payload["weight_map"]["blocks.3.attn.to_k.weight"] = "model-00004-of-00004.safetensors"
    (tmp_path / "src" / "model.safetensors.index.json").write_text(json.dumps(payload))

    index = tmp_path / "src" / "model.safetensors.index.json"
    result = streaming_fp8_storage_cast(index, tmp_path / "out")
    got = _all_tensors(result["index_path"] or result["output_paths"][0])

    assert got["blocks.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn
    assert got["blocks.1.ff.net.0.proj.weight"].dtype == torch.float8_e4m3fn
    # skip patterns / 1-D / int keep source dtype
    assert got["blocks.0.norm1.weight"].dtype == torch.float32
    assert got["pos_embed.proj.weight"].dtype == torch.float32
    assert got["blocks.0.attn.to_q.bias"].dtype == torch.float32
    assert got["blocks.1.step_counter"].dtype == torch.int64
    # clamped, not NaN
    hot_out = got["blocks.3.attn.to_k.weight"].to(torch.float32)
    assert not torch.isnan(hot_out).any()
    assert hot_out.max().item() == pytest.approx(448.0)
    # values within fp8 range round-trip through the same rounding the
    # runtime storage lane applies
    src = _all_tensors(index)
    expected = src["blocks.0.attn.to_q.weight"].clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(
        got["blocks.0.attn.to_q.weight"].view(torch.uint8),
        expected.view(torch.uint8))


def test_fp8_majority_dtype_detectable(tmp_path: Path) -> None:
    """The serve-side sniffer keys on majority F8_E4M3 headers; a denoiser
    where big weights dominate must sniff as fp8."""
    src = tmp_path / "d.safetensors"
    save_file({
        f"blocks.{i}.attn.to_q.weight": torch.randn(64, 64) for i in range(4)
    } | {"blocks.0.norm1.weight": torch.randn(64)}, str(src))
    result = streaming_fp8_storage_cast(src, tmp_path / "out")
    counts: dict[str, int] = {}
    with safe_open(str(result["output_paths"][0]), framework="pt") as f:
        for k in f.keys():
            d = str(f.get_slice(k).get_dtype())
            counts[d] = counts.get(d, 0) + 1
    assert max(counts, key=counts.get) == "F8_E4M3"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Snapshot-level helpers (diffusers tree)
# ---------------------------------------------------------------------------

def _write_diffusers_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "model_index.json").write_text('{"_class_name": "FakePipeline"}')
    (root / "unet").mkdir()
    save_file(
        {"down.0.attn.to_q.weight": torch.randn(64, 64),
         "down.0.norm.weight": torch.randn(64)},
        str(root / "unet" / "diffusion_pytorch_model.safetensors"))
    (root / "unet" / "config.json").write_text("{}")
    (root / "vae").mkdir()
    save_file({"encoder.conv_in.weight": torch.randn(4, 4, 3, 3)},
              str(root / "vae" / "diffusion_pytorch_model.safetensors"))
    (root / "vae" / "config.json").write_text("{}")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text("{}")
    return root


def test_streaming_cast_snapshot_full_tree(tmp_path: Path) -> None:
    src = _write_diffusers_tree(tmp_path / "src")
    out = tmp_path / "out"
    result = streaming_cast_snapshot(
        src, out, file_layout="diffusers", target_dtype=torch.float16)
    assert sorted(result["components"]) == ["unet", "vae"]
    assert (out / "model_index.json").is_file()
    assert (out / "scheduler" / "scheduler_config.json").is_file()
    assert (out / "unet" / "config.json").is_file()
    unet = _all_tensors(out / "unet" / "diffusion_pytorch_model.safetensors")
    vae = _all_tensors(out / "vae" / "diffusion_pytorch_model.safetensors")
    assert all(t.dtype == torch.float16 for t in unet.values())
    assert all(t.dtype == torch.float16 for t in vae.values())


def test_streaming_fp8_snapshot_denoiser_only(tmp_path: Path) -> None:
    src = _write_diffusers_tree(tmp_path / "src")
    out = tmp_path / "out"
    streaming_fp8_snapshot(src, out, file_layout="diffusers")
    unet = _all_tensors(out / "unet" / "diffusion_pytorch_model.safetensors")
    assert unet["down.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn
    assert unet["down.0.norm.weight"].dtype == torch.float32
    # vae passes through byte-identical
    src_vae = (src / "vae" / "diffusion_pytorch_model.safetensors").read_bytes()
    out_vae = (out / "vae" / "diffusion_pytorch_model.safetensors").read_bytes()
    assert src_vae == out_vae
    assert (out / "model_index.json").is_file()


def test_streaming_fp8_snapshot_refuses_singlefile(tmp_path: Path) -> None:
    from gen_worker.convert.writer import ConversionImplementationError

    (tmp_path / "src").mkdir()
    save_file({"w": torch.randn(4, 4)}, str(tmp_path / "src" / "model.safetensors"))
    with pytest.raises(ConversionImplementationError):
        streaming_fp8_snapshot(tmp_path / "src", tmp_path / "out", file_layout="singlefile")


def test_clone_normalizes_fp8_spellings() -> None:
    from gen_worker.convert.clone import normalize_outputs

    specs = normalize_outputs([
        {"dtype": "fp8:e4m3"}, {"dtype": "fp8-e4m3"}, {"dtype": "fp8"},
    ])
    assert [s.dtype for s in specs] == ["fp8"]  # all collapse + dedupe


def test_inline_conversion_fp8_route(tmp_path: Path) -> None:
    from gen_worker.convert.convert import run_inline_conversion

    src = tmp_path / "model.safetensors"
    save_file({"blocks.0.attn.to_q.weight": torch.randn(64, 64)}, str(src))
    result = run_inline_conversion(
        source_path=src, out_dir=tmp_path / "out", target_dtype="fp8")
    assert result.target_dtype == "fp8"
    assert result.attributes["conversion_strategy"] == "inline_fp8_storage_cast"
    got = _all_tensors(result.output_paths[0])
    assert got["blocks.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Peak anonymous-RSS bound — the property gw#395/#396 exist for
# ---------------------------------------------------------------------------

def _rss_anon_kb() -> int:
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon"):
            return int(line.split()[1])
    return 0


class _AnonPeakSampler:
    """Samples RssAnon on a thread; mmap'd file pages (RssFile) are
    deliberately excluded — they are reclaimable and the kernel evicts them
    under a memory cap, so anonymous memory is what OOMs a pod."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_anon_kb())
            time.sleep(0.002)

    def __enter__(self) -> "_AnonPeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *a: object) -> None:
        self._stop.set()
        self._thread.join()


def _write_big_fixture(root: Path, *, n_shards: int = 3, tensors_per_shard: int = 3,
                       largest_elems: int = 12_000_000) -> tuple[Path, int]:
    """3 shards x 3 fp32 tensors; largest tensor 48 MB. Total ~410 MB."""
    root.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total = 0
    largest_bytes = 0
    for s in range(n_shards):
        shard = f"model-{s + 1:05d}-of-{n_shards:05d}.safetensors"
        tensors = {}
        for i in range(tensors_per_shard):
            elems = largest_elems if i == 0 else largest_elems // 2
            name = f"blocks.{s * tensors_per_shard + i}.attn.to_q.weight"
            tensors[name] = torch.rand(elems // 1000, 1000)
            weight_map[name] = shard
            nbytes = tensors[name].numel() * 4
            total += nbytes
            largest_bytes = max(largest_bytes, nbytes)
        save_file(tensors, str(root / shard), metadata={"format": "pt"})
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}))
    return index, largest_bytes


@pytest.mark.parametrize("op", ["cast_bf16", "fp8"])
def test_peak_anon_rss_below_2x_largest_tensor(tmp_path: Path, op: str) -> None:
    index, largest = _write_big_fixture(tmp_path / "src")
    baseline = _rss_anon_kb()
    with _AnonPeakSampler() as sampler:
        if op == "cast_bf16":
            streaming_dtype_cast(index, tmp_path / "out", target_dtype=torch.bfloat16)
        else:
            streaming_fp8_storage_cast(index, tmp_path / "out")
    delta_kb = sampler.peak - baseline
    limit_kb = (2 * largest) // 1024
    assert delta_kb < limit_kb, (
        f"{op}: peak anon RSS delta {delta_kb} KB >= 2x largest tensor "
        f"({limit_kb} KB) — streaming bound broken")
