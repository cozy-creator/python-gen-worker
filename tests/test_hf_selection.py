"""The small HF variant selector (#366, replaces the 522-LOC planner).

``select_hf_files`` decides which repo files ``snapshot_download`` pulls:
one weight set per component directory (flavor > bf16 > fp16 > untagged,
safetensors preferred), all configs/tokenizers, root-weights repos supported,
unknown layouts fall back to the whole repo (returns None).
"""

from __future__ import annotations

from gen_worker.models.download import select_hf_files

_DIFFUSERS_REPO = [
    "model_index.json",
    "README.md",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.bf16.safetensors",
    "transformer/diffusion_pytorch_model.fp16.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
]


def test_diffusers_repo_picks_one_weight_set_per_component() -> None:
    sel = select_hf_files(_DIFFUSERS_REPO)
    assert sel is not None
    # All non-weight files included.
    assert "model_index.json" in sel
    assert "scheduler/scheduler_config.json" in sel
    assert "tokenizer/tokenizer.json" in sel
    # transformer: bf16 preferred over fp16/untagged.
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" in sel
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" not in sel
    assert "transformer/diffusion_pytorch_model.safetensors" not in sel
    # vae/text_encoder have no bf16 -> fp16 wins over untagged.
    assert "vae/diffusion_pytorch_model.fp16.safetensors" in sel
    assert "vae/diffusion_pytorch_model.safetensors" not in sel


def test_flavor_overrides_the_default_preference() -> None:
    sel = select_hf_files(_DIFFUSERS_REPO, flavor="fp16")
    assert sel is not None
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" in sel
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" not in sel


def test_safetensors_preferred_over_bin() -> None:
    files = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.safetensors",
    ]
    sel = select_hf_files(files)
    assert sel is not None
    assert "unet/diffusion_pytorch_model.safetensors" in sel
    assert "unet/diffusion_pytorch_model.bin" not in sel


def test_diffusers_repo_excludes_root_monolithic_checkpoints() -> None:
    # sd1.5 shape (e2e #105 J4): untagged fp32 root checkpoints coexist with
    # fp16 component weights. The root single-file distributions are redundant
    # — picking them pulled 12GB of fp32 where ~2GB of fp16 suffices.
    files = [
        "model_index.json",
        "README.md",
        "v1-5-pruned.ckpt",
        "v1-5-pruned.safetensors",
        "v1-5-pruned-emaonly.ckpt",
        "v1-5-pruned-emaonly.safetensors",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.non_ema.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ]
    sel = select_hf_files(files)
    assert sel is not None
    assert "v1-5-pruned.safetensors" not in sel
    assert "v1-5-pruned-emaonly.safetensors" not in sel
    assert "v1-5-pruned.ckpt" not in sel
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in sel
    assert "vae/diffusion_pytorch_model.fp16.safetensors" in sel
    assert "text_encoder/model.fp16.safetensors" in sel
    assert "model_index.json" in sel


def test_root_weights_repo_selects_weights_and_sidecars() -> None:
    files = [
        "config.json",
        "tokenizer.json",
        "model.bf16.safetensors",
        "model.fp16.safetensors",
        "README.md",
    ]
    sel = select_hf_files(files)
    assert sel is not None
    assert "model.bf16.safetensors" in sel
    assert "model.fp16.safetensors" not in sel
    assert "config.json" in sel and "tokenizer.json" in sel


def test_unrecognized_layout_downloads_whole_repo() -> None:
    # Weights nested under non-diffusers dirs (ComfyUI split checkpoints).
    files = [
        "split_files/diffusion_models/model.safetensors",
        "split_files/vae/ae.safetensors",
        "notes.txt",
    ]
    assert select_hf_files(files) is None


def test_no_weights_at_all_downloads_whole_repo() -> None:
    assert select_hf_files(["README.md", "config.json"]) is None
    assert select_hf_files([]) is None


def test_sharded_single_file_checkpoint_reassembles(tmp_path):
    """A >5GB single-file checkpoint arrives from the mirror as byte-offset
    shards + index; the loader must reassemble it for from_single_file."""
    import json
    import struct

    from gen_worker.models.loading import _single_file_checkpoint

    def write_st(path, tensors):  # minimal safetensors writer
        header, blob, off = {}, b"", 0
        for name, (dtype, shape, data) in tensors.items():
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
            blob += data
            off += len(data)
        hb = json.dumps(header, separators=(",", ":")).encode()
        path.write_bytes(struct.pack("<Q", len(hb)) + hb + blob)

    write_st(tmp_path / "ckpt-00001-of-00002.safetensors",
             {"alpha": ("F32", [2], b"\x00\x00\x80?\x00\x00\x00@")})  # [1.0, 2.0]
    write_st(tmp_path / "ckpt-00002-of-00002.safetensors",
             {"beta": ("F16", [1], b"\x00<")})  # [1.0]
    (tmp_path / "ckpt.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 10},
        "weight_map": {
            "alpha": "ckpt-00001-of-00002.safetensors",
            "beta": "ckpt-00002-of-00002.safetensors",
        },
    }))

    merged = _single_file_checkpoint(tmp_path)
    assert merged is not None and merged.name == "ckpt.safetensors"
    raw = merged.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    assert set(header) == {"alpha", "beta"}
    body = raw[8 + n:]
    a0, a1 = header["alpha"]["data_offsets"]
    b0, b1 = header["beta"]["data_offsets"]
    assert body[a0:a1] == b"\x00\x00\x80?\x00\x00\x00@"
    assert body[b0:b1] == b"\x00<"
    assert header["beta"]["dtype"] == "F16"
    # Idempotent: second call reuses the cached merge.
    assert _single_file_checkpoint(tmp_path) == merged


def test_single_file_checkpoint_plain_and_layouts(tmp_path):
    from gen_worker.models.loading import _single_file_checkpoint

    single = tmp_path / "model.safetensors"
    single.write_bytes(b"x")
    assert _single_file_checkpoint(tmp_path) == single
    # A pretrained layout is never treated as single-file.
    (tmp_path / "model_index.json").write_text("{}")
    assert _single_file_checkpoint(tmp_path) is None
