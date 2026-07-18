"""Root-layout w8a8 lane (gw#562) — the DiffSynth/singlefile families.

Contract round-trip against a tiny DiffSynth-class pipeline: a root shard
set (no model_index.json), a pipeline class that constructs its own model
from the shards (sanitize at read), and the worker's post-construction
Linear swap. CPU lane proves producer -> detect -> sanitize-construct ->
dequant numerics and the swap's module surgery; the sm_89+ lane proves
scaled_mm forward parity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch.nn as nn  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from gen_worker.convert.writer import (  # noqa: E402
    ConversionImplementationError,
    streaming_w8a8_snapshot,
    verify_w8a8_snapshot,
)
from gen_worker.models import w8a8  # noqa: E402
from gen_worker.models.loading import (  # noqa: E402
    load_from_pretrained,
    pipeline_weight_lane,
)
from gen_worker.models.w8a8 import (  # noqa: E402
    detect_w8a8_artifact,
    fp8_scaled_linear_class,
    sanitize_w8a8_state_dict,
    swap_w8a8_linears,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_q = nn.Linear(32, 32)
        self.mlp_in = nn.Linear(32, 64)
        self.mlp_out = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.attn_q(h)
        h = self.mlp_out(torch.nn.functional.gelu(self.mlp_in(h)))
        return x + h


class _TinyDiT(nn.Module):
    """Root-keyed module: ``blocks.N.*`` Linears quantize, the rest stay."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(16, 32)
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])
        self.norm_out = nn.LayerNorm(32)
        self.proj_out = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x)
        for b in self.blocks:
            h = b(h)
        return self.proj_out(self.norm_out(h))


_QUANTIZED_LAYERS = tuple(sorted(
    f"blocks.{i}.{n}" for i in range(2) for n in ("attn_q", "mlp_in", "mlp_out")
))


class _TinyRootPipeline:
    """DiffSynth-shaped: reads root shards itself, sanitizes, aliases the
    denoiser at ``.transformer`` (the worker-facing convention)."""

    def __init__(self) -> None:
        self.transformer: nn.Module | None = None

    @classmethod
    def from_pretrained(cls, path: str, torch_dtype=None, **_) -> "_TinyRootPipeline":
        dtype = torch_dtype or torch.bfloat16
        sd: dict[str, torch.Tensor] = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            sd.update(load_file(str(f)))
        sd = sanitize_w8a8_state_dict(sd, dtype)
        sd = {k: v.to(dtype) for k, v in sd.items()}
        model = _TinyDiT()
        model.load_state_dict(sd, assign=True)
        model.eval()
        pipe = cls()
        pipe.transformer = model
        return pipe

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert self.transformer is not None
        return self.transformer(x)


@pytest.fixture(scope="module")
def source_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(7)
    root = tmp_path_factory.mktemp("w8a8root") / "src"
    root.mkdir()
    model = _TinyDiT()
    sd = {k: v.detach().to(torch.bfloat16).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({"model_type": "tiny_dit"}))
    return root


@pytest.fixture(scope="module")
def w8a8_root(source_root: Path) -> Path:
    out = source_root.parent / "w8a8"
    result = streaming_w8a8_snapshot(source_root, out, file_layout="singlefile")
    assert result["components"] == ["model.safetensors"]
    return out


def test_producer_quantizes_block_linears_only(
    source_root: Path, w8a8_root: Path,
) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(w8a8_root.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
    scales = sorted(
        k[: -len(".weight_scale")] for k in tensors if k.endswith(".weight_scale"))
    assert tuple(scales) == _QUANTIZED_LAYERS
    for layer in _QUANTIZED_LAYERS:
        assert tensors[f"{layer}.weight"].dtype == torch.float8_e4m3fn
    assert tensors["patch_embed.weight"].dtype == torch.bfloat16
    assert tensors["proj_out.weight"].dtype == torch.bfloat16
    assert (w8a8_root / "config.json").exists()  # passthrough
    verify_w8a8_snapshot(source_root, w8a8_root)


def test_producer_refuses_te_components_on_root_layout(
    source_root: Path, tmp_path: Path,
) -> None:
    with pytest.raises(ConversionImplementationError, match="te_components"):
        streaming_w8a8_snapshot(
            source_root, tmp_path / "o", file_layout="singlefile",
            te_components=("text_encoder",))


def test_detects_root_artifact(w8a8_root: Path, source_root: Path) -> None:
    art = detect_w8a8_artifact(w8a8_root)
    assert art is not None
    assert art.component == ""
    assert art.quantized == _QUANTIZED_LAYERS
    assert not art.static_input_scales
    assert detect_w8a8_artifact(source_root) is None  # plain bf16 root


def test_scale_free_root_fp8_never_detects(source_root: Path, tmp_path: Path) -> None:
    cast = tmp_path / "cast"
    cast.mkdir()
    tensors = {
        k: (v.to(torch.float8_e4m3fn) if v.ndim == 2 else v)
        for k, v in load_file(str(source_root / "model.safetensors")).items()
    }
    save_file(tensors, str(cast / "model.safetensors"))
    assert detect_w8a8_artifact(cast) is None


def test_dequant_lane_constructs_correct_weights(
    source_root: Path, w8a8_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "")
    pipe = load_from_pretrained(_TinyRootPipeline, w8a8_root)
    assert pipe._cozy_weight_lane == "bf16-resident"
    assert pipeline_weight_lane(pipe) == ""
    assert pipe.transformer._cozy_w8a8_mode == "dequant"

    src = load_file(str(source_root / "model.safetensors"))
    got = pipe.transformer.state_dict()
    for layer in _QUANTIZED_LAYERS:
        a = src[f"{layer}.weight"].float()
        b = got[f"{layer}.weight"].float()
        rel = ((a - b).abs() / a.abs().clamp(min=1e-3)).max().item()
        assert rel < 0.13, layer  # e4m3 rounding only
        assert got[f"{layer}.weight"].dtype == torch.bfloat16
    out = pipe(torch.randn(2, 16, dtype=torch.bfloat16))
    assert out.shape == (2, 16)


def test_swap_puts_block_linears_on_fp8(
    w8a8_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scaled_mm host: worker swap replaces exactly the quantized Linears
    (module surgery is device-agnostic; forward needs the GPU lane below)."""
    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "rowwise")
    pipe = load_from_pretrained(_TinyRootPipeline, w8a8_root)
    assert pipe._cozy_weight_lane == "w8a8"
    assert pipeline_weight_lane(pipe) == "w8a8"
    lin_cls = fp8_scaled_linear_class()
    model = pipe.transformer
    for layer in _QUANTIZED_LAYERS:
        mod = model.get_submodule(layer)
        assert isinstance(mod, lin_cls), layer
        assert mod.weight.dtype == torch.float8_e4m3fn
        assert mod.weight_scale.shape == (mod.out_features, 1)
        assert mod.bias is not None and mod.bias.dtype == torch.bfloat16
    assert type(model.patch_embed) is nn.Linear
    assert type(model.proj_out) is nn.Linear


def test_swap_with_key_map_reaches_renamed_modules(
    source_root: Path, tmp_path: Path,
) -> None:
    """Converter-renamed checkpoints (anima's ``net.`` strip): the artifact
    keys carry the prefix, the module tree does not — key_map bridges."""
    prefixed_src = tmp_path / "src"
    prefixed_src.mkdir()
    sd = load_file(str(source_root / "model.safetensors"))
    save_file({f"net.{k}": v for k, v in sd.items()},
              str(prefixed_src / "model.safetensors"))
    out = tmp_path / "w8a8"
    streaming_w8a8_snapshot(prefixed_src, out, file_layout="singlefile")
    art = detect_w8a8_artifact(out)
    assert art is not None and art.component == ""
    assert all(n.startswith("net.") for n in art.quantized)

    raw: dict[str, torch.Tensor] = {}
    for f in art.files:
        raw.update(load_file(str(f)))
    clean = sanitize_w8a8_state_dict(raw, torch.bfloat16)
    model = _TinyDiT()
    model.load_state_dict(
        {k[len("net."):]: v.to(torch.bfloat16) for k, v in clean.items()},
        assign=True)
    swapped = swap_w8a8_linears(
        model, art, compute_dtype=torch.bfloat16,
        key_map=lambda k: k[len("net."):])
    assert swapped == len(_QUANTIZED_LAYERS)


class _TinyNestedPipeline:
    """Split-checkpoint-shaped (anima class): DiT under a nested dir with a
    converter-renamed (``net.``-prefixed) key space, a second passthrough
    weight set beside it. Declares the swap key_map on the class."""

    _cozy_w8a8_key_map = staticmethod(lambda k: k[len("net."):])

    def __init__(self) -> None:
        self.transformer: nn.Module | None = None

    @classmethod
    def from_pretrained(cls, path: str, torch_dtype=None, **_) -> "_TinyNestedPipeline":
        dtype = torch_dtype or torch.bfloat16
        dit_file = next(Path(path).rglob("dit.safetensors"))
        sd = sanitize_w8a8_state_dict(load_file(str(dit_file)), dtype)
        sd = {k[len("net."):]: v.to(dtype) for k, v in sd.items()}
        model = _TinyDiT()
        model.load_state_dict(sd, assign=True)
        model.eval()
        pipe = cls()
        pipe.transformer = model
        return pipe


def _nested_source(tmp_path: Path) -> Path:
    torch.manual_seed(11)
    src = tmp_path / "nested-src"
    (src / "models" / "dit").mkdir(parents=True)
    (src / "models" / "te").mkdir(parents=True)
    dit_sd = {f"net.{k}": v.detach().to(torch.bfloat16).contiguous()
              for k, v in _TinyDiT().state_dict().items()}
    save_file(dit_sd, str(src / "models" / "dit" / "dit.safetensors"))
    # The passthrough set ALSO carries w8a8-eligible-looking tensors
    # (repeated-block 2-D 16-aligned) — the ambiguity the selector exists for.
    te_sd = {
        "model.layers.0.mlp.up.weight":
            torch.randn(64, 32, dtype=torch.bfloat16),
        "model.layers.1.mlp.up.weight":
            torch.randn(64, 32, dtype=torch.bfloat16),
    }
    save_file(te_sd, str(src / "models" / "te" / "te.safetensors"))
    (src / "config.json").write_text(json.dumps({"model_type": "tiny_split"}))
    return src


def test_nested_multiset_needs_selector_and_passes_rest_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = _nested_source(tmp_path)
    with pytest.raises(ConversionImplementationError, match="weight_set_patterns"):
        streaming_w8a8_snapshot(src, tmp_path / "o1", file_layout="singlefile")
    with pytest.raises(ConversionImplementationError, match="match none"):
        streaming_w8a8_snapshot(
            src, tmp_path / "o2", file_layout="singlefile",
            weight_set_patterns=("nope/*",))

    out = tmp_path / "w8a8"
    result = streaming_w8a8_snapshot(
        src, out, file_layout="singlefile",
        weight_set_patterns=("models/dit/*",))
    assert result["components"] == ["models/dit/dit.safetensors"]

    # Selected set requanted in place (rel dir preserved), TE byte-identical.
    te_src = (src / "models" / "te" / "te.safetensors").read_bytes()
    te_out = (out / "models" / "te" / "te.safetensors").read_bytes()
    assert te_out == te_src
    assert (out / "config.json").exists()

    art = detect_w8a8_artifact(out)
    assert art is not None and art.component == ""
    assert art.quantized == tuple(f"net.{n}" for n in _QUANTIZED_LAYERS)
    verify_w8a8_snapshot(src, out)

    # Serve: worker constructs via the class loader, swaps through the
    # class-declared key_map hook (module surgery is device-agnostic).
    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "rowwise")
    pipe = load_from_pretrained(_TinyNestedPipeline, out)
    assert pipe._cozy_weight_lane == "w8a8"
    lin_cls = fp8_scaled_linear_class()
    for layer in _QUANTIZED_LAYERS:
        assert isinstance(pipe.transformer.get_submodule(layer), lin_cls), layer
    assert type(pipe.transformer.patch_embed) is nn.Linear


def test_diffusers_layout_refuses_weight_set_patterns(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}")
    with pytest.raises(ConversionImplementationError, match="non-diffusers"):
        streaming_w8a8_snapshot(
            tmp_path, tmp_path / "o", file_layout="diffusers",
            weight_set_patterns=("x/*",))


def test_sanitize_passes_scale_free_dicts_through(source_root: Path) -> None:
    sd = load_file(str(source_root / "model.safetensors"))
    assert sanitize_w8a8_state_dict(dict(sd), torch.bfloat16).keys() == sd.keys()


# ---------------------------------------------------------------------------
# GPU lane (sm_89+)
# ---------------------------------------------------------------------------


def _cuda_sm89() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


gpu = pytest.mark.skipif(not _cuda_sm89(), reason="needs CUDA sm_89+ (fp8 tensor cores)")


@gpu
def test_root_pipeline_serves_on_scaled_mm_lane(
    source_root: Path, w8a8_root: Path,
) -> None:
    w8a8.w8a8_gemm_mode.cache_clear()
    pipe = load_from_pretrained(_TinyRootPipeline, w8a8_root)
    assert pipe._cozy_weight_lane == "w8a8"
    model = pipe.transformer.to("cuda")
    x = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    y = model(x)

    ref_pipe = _TinyRootPipeline.from_pretrained(str(source_root))
    ref = ref_pipe.transformer.to("cuda")(x)
    rel = ((y - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
    assert rel < 0.25  # fp8 weight rounding + dynamic activation quant
