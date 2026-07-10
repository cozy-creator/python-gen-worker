"""gw#463: storage-side fp8 for text encoders mirrors the gw#460 loader.

The contract under test: ``streaming_fp8_snapshot(te_components=...)``
produces, for a transformers text encoder, EXACTLY the tensors the
``storage_dtype="fp8+te"`` loader would cast — same key set (derived from
the loader's own ``_fp8_block_windows`` on a meta-instantiated module) and
byte-identical fp8 payloads; everything else passes through untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gen_worker.convert import (  # noqa: E402
    FP8_TE_COMPONENTS,
    streaming_fp8_snapshot,
    te_fp8_castable_keys,
)
from gen_worker.models import loading as loading_mod  # noqa: E402


def _tiny_t5_encoder(tmp_path: Path) -> Path:
    cfg = transformers.T5Config(
        d_model=32, d_ff=64, d_kv=8, num_heads=4,
        num_layers=2, vocab_size=128, decoder_start_token_id=0,
    )
    model = transformers.T5EncoderModel(cfg).to(torch.bfloat16)
    comp = tmp_path / "src" / "text_encoder"
    model.save_pretrained(comp, safe_serialization=True)
    return comp


def _load_all_tensors(component_dir: Path) -> dict[str, "torch.Tensor"]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    files = sorted(component_dir.glob("*.safetensors"))
    idx = component_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        files = sorted({component_dir / v for v in wm.values()})
    for f in files:
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                out[k] = fh.get_tensor(k)
    return out


def test_te_components_mirror_loader() -> None:
    assert FP8_TE_COMPONENTS == loading_mod._FP8_TEXT_ENCODER_COMPONENTS


def test_te_castable_keys_match_loader_windows(tmp_path: Path) -> None:
    comp = _tiny_t5_encoder(tmp_path)
    keys = te_fp8_castable_keys(comp)

    # The authority: the loader's block-window walk on the REAL module.
    model = transformers.T5EncoderModel.from_pretrained(comp)
    windows = loading_mod._fp8_block_windows(model)
    castable = {id(p) for _, _, params in windows for p in params}
    loader_keys = {n for n, p in model.named_parameters() if id(p) in castable}

    assert keys == frozenset(loader_keys)
    assert keys, "tiny T5 must have castable block weights"
    # Embeddings and norms never cast (gw#460).
    assert not any("embed" in k or "layer_norm" in k for k in keys)


def test_snapshot_te_cast_is_loader_byte_identical(tmp_path: Path) -> None:
    comp = _tiny_t5_encoder(tmp_path)
    src = comp.parent
    out = tmp_path / "out"

    stats = streaming_fp8_snapshot(
        src, out, file_layout="diffusers",
        components=(), te_components=("text_encoder",),
    )
    assert "text_encoder" in stats["components"]

    keys = te_fp8_castable_keys(comp)
    src_t = _load_all_tensors(comp)
    out_t = _load_all_tensors(out / "text_encoder")
    assert set(src_t) == set(out_t)

    n_cast = 0
    for name, s in src_t.items():
        o = out_t[name]
        if name in keys:
            assert o.dtype == torch.float8_e4m3fn, name
            # Loader-equivalent bytes: bare .to() == clamp().to() for real
            # weights (torch fp8 cast doesn't saturate; clamp only guards
            # |w|>448 outliers, absent here and in practice).
            expect = s.to(torch.float8_e4m3fn)
            assert torch.equal(o.view(torch.uint8), expect.view(torch.uint8)), name
            n_cast += 1
        else:
            assert o.dtype == s.dtype, name
            assert torch.equal(o.view(torch.uint8), s.view(torch.uint8)), name
    assert n_cast == len(keys)
