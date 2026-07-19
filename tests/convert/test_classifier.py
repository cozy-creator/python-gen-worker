"""Unit tests for the small HF repo classifier (gen_worker.convert.classifier)."""

from __future__ import annotations

import json
import struct

import pytest

from gen_worker.convert.classifier import RepoRefusal, classify_repo

_DIFFUSERS = [
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
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.bf16.safetensors",
    "transformer/diffusion_pytorch_model.fp16.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
    "unet/demo.png",
]


def test_diffusers_one_weight_set_per_component() -> None:
    c = classify_repo(_DIFFUSERS)
    assert c.strategy == "diffusers"
    assert c.runtime_library == "diffusers"
    allow = set(c.allow_patterns)
    # bf16 preferred where present; next preferred tag (fp16) beats untagged;
    # untagged is the fallback when no preferred tag exists.
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" in allow
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" not in allow
    assert "transformer/diffusion_pytorch_model.safetensors" not in allow
    assert "text_encoder/model.fp16.safetensors" in allow
    assert "text_encoder/model.safetensors" not in allow
    assert "vae/diffusion_pytorch_model.safetensors" in allow
    # configs + model_index always ride along; demo media never.
    assert "model_index.json" in allow
    assert "scheduler/scheduler_config.json" in allow
    assert "unet/demo.png" not in allow


def test_diffusers_dtype_preference_fp16() -> None:
    c = classify_repo(_DIFFUSERS, dtype_pref=("fp16",))
    allow = set(c.allow_patterns)
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" in allow
    assert "text_encoder/model.fp16.safetensors" in allow
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" not in allow


def test_diffusers_pipeline_selects_complete_official_variant_shard_set() -> None:
    files = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "unet/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
        "unet/diffusion_pytorch_model.safetensors.index.fp16.json",
        "unet/diffusion_pytorch_model.bf16-00001-of-00002.safetensors",
        "unet/diffusion_pytorch_model.bf16-00002-of-00002.safetensors",
        "unet/diffusion_pytorch_model.safetensors.index.bf16.json",
    ]
    c = classify_repo(files, dtype_pref=("fp16",))

    assert c.strategy == "diffusers"
    assert set(c.allow_patterns) == {
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "unet/diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
        "unet/diffusion_pytorch_model.safetensors.index.fp16.json",
    }


def test_diffusers_skips_root_allinone_checkpoints() -> None:
    # SD1.5 shape: the repo ships all-in-one root checkpoints (12GB) on top
    # of the component tree — a diffusers-layout ingest must never pull them
    # (found live in e2e J7: 14.7GB selected instead of 2.75GB, ENOSPC on
    # the ingest pod).
    files = _DIFFUSERS + [
        "v1-5-pruned.safetensors",
        "v1-5-pruned-emaonly.safetensors",
    ]
    c = classify_repo(files, dtype_pref=("fp16",))
    assert c.strategy == "diffusers"
    allow = set(c.allow_patterns)
    assert "v1-5-pruned.safetensors" not in allow
    assert "v1-5-pruned-emaonly.safetensors" not in allow
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" in allow


def test_standalone_diffusers_vae_selects_only_canonical_weight() -> None:
    """gw#426: madebyollin's SDXL VAE is a Diffusers component, not a
    Transformers model, and its A1111 aliases are not extra logical weights."""
    files = [
        ".gitattributes",
        "README.md",
        "config.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
        "images/vae-fix.jpg",
        "sdxl.vae.safetensors",
        "sdxl_vae.safetensors",
    ]
    c = classify_repo(
        files,
        config_json={
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.18.0.dev0",
        },
    )

    assert c.strategy == "diffusers_component"
    assert c.runtime_library == "diffusers"
    assert c.attrs == {
        "file_layout": "singlefile",
        "architecture": "AutoencoderKL",
    }
    assert set(c.allow_patterns) == {
        "README.md",
        "config.json",
        "diffusion_pytorch_model.safetensors",
    }


def test_standalone_diffusers_component_selects_complete_variant_shard_set() -> None:
    files = [
        "config.json",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
        "diffusion_pytorch_model.safetensors.index.fp16.json",
        "diffusion_pytorch_model.bf16-00001-of-00002.safetensors",
        "diffusion_pytorch_model.bf16-00002-of-00002.safetensors",
        "diffusion_pytorch_model.safetensors.index.bf16.json",
        "vae-copy.safetensors",
    ]
    c = classify_repo(
        files,
        config_json={"_class_name": "AutoencoderKL"},
        dtype_pref=("fp16",),
    )

    assert c.strategy == "diffusers_component"
    assert c.attrs["dtype"] == "fp16"
    assert set(c.allow_patterns) == {
        "config.json",
        "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        "diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
        "diffusion_pytorch_model.safetensors.index.fp16.json",
    }


def test_transformers_with_sharded_index_and_onnx_excluded() -> None:
    files = [
        "config.json", "generation_config.json", "tokenizer.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "onnx/model.onnx",
        "pytorch_model.bin",
    ]
    c = classify_repo(files, config_json={"architectures": ["LlamaForCausalLM"]})
    assert c.strategy == "transformers"
    allow = set(c.allow_patterns)
    assert "model-00001-of-00002.safetensors" in allow
    assert "model.safetensors.index.json" in allow
    assert "onnx/model.onnx" not in allow
    assert "pytorch_model.bin" not in allow
    assert c.attrs.get("architecture") == "LlamaForCausalLM"


def test_peft_adapter() -> None:
    c = classify_repo(["adapter_config.json", "adapter_model.safetensors", "README.md"])
    assert c.strategy == "peft"
    assert c.runtime_library == "peft"
    assert "adapter_model.safetensors" in c.allow_patterns


def test_sentence_transformers() -> None:
    c = classify_repo([
        "modules.json", "config.json", "model.safetensors",
        "1_Pooling/config.json", "tokenizer.json",
    ])
    assert c.strategy == "sentence_transformers"


def test_gguf_quant_preference_and_explicit_pick() -> None:
    files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "README.md"]
    c = classify_repo(files)
    assert c.strategy == "gguf"
    assert [p for p in c.allow_patterns if p.endswith(".gguf")] == ["model.Q8_0.gguf"]

    c2 = classify_repo(files, gguf_quant="q4_k_m")
    assert [p for p in c2.allow_patterns if p.endswith(".gguf")] == ["model.Q4_K_M.gguf"]

    with pytest.raises(RepoRefusal) as exc:
        classify_repo(files, gguf_quant="q2_k")
    assert exc.value.reason == "gguf_quant_not_found"


def test_native_lora_via_kohya_metadata() -> None:
    c = classify_repo(
        ["my_lora.safetensors", "README.md"],
        sizes={"my_lora.safetensors": 40 * 1024 * 1024},
        safetensors_metadata={"ss_network_module": "networks.lora"},
    )
    assert c.strategy == "native_lora"
    assert c.runtime_library == "diffusers-lora"


def test_native_lora_via_readme_tags() -> None:
    c = classify_repo(
        ["style.safetensors"],
        sizes={"style.safetensors": 2 * 1024 * 1024 * 1024},
        readme_tags=["lora", "text-to-image"],
    )
    assert c.strategy == "native_lora"


def test_aio_singlefile() -> None:
    c = classify_repo(
        ["juggernaut-xl.safetensors"],
        sizes={"juggernaut-xl.safetensors": 6 * 1024 * 1024 * 1024},
    )
    assert c.strategy == "aio_singlefile"
    assert c.runtime_library == "diffusers-single-file"


@pytest.mark.parametrize("files,reason", [
    (["pytorch_model.bin"], "pickle_only"),
    (["model.onnx"], "onnx_only"),
    (["tf_model.h5"], "tf_only"),
    (["flax_model.msgpack"], "flax_only"),
    (["weights.mlpackage"], "coreml_only"),
    (["model.engine"], "tensorrt_only"),
    (["README.md"], "unknown_shape"),
])
def test_refusals(files: list[str], reason: str) -> None:
    with pytest.raises(RepoRefusal) as exc:
        classify_repo(files)
    assert exc.value.reason == reason


def test_too_large_refused() -> None:
    with pytest.raises(RepoRefusal) as exc:
        classify_repo(
            ["model_index.json", "transformer/x.safetensors"],
            sizes={"transformer/x.safetensors": 200 * 1024 * 1024 * 1024},
        )
    assert exc.value.reason == "too_large"


def test_hf_multi_weight_bundle_opts_out_of_layout_contract(tmp_path, monkeypatch) -> None:
    """chatterbox regression (ie#368): HF multi-component single-file repos
    (t3/s3gen/ve) must publish with library_name unset, mirroring the civitai
    branch (e2e #112) — tensorhub finalize rejects diffusers/single-file
    manifests carrying multiple distinct weights."""
    from pathlib import Path

    from gen_worker.convert import ingest as ing
    from gen_worker.convert.classifier import classify_repo

    paths = ["t3_cfg.safetensors", "s3gen.safetensors", "ve.safetensors", "tokenizer.json"]
    sizes = {p: 2 * 1024**3 for p in paths if p.endswith(".safetensors")}
    classification = classify_repo(paths, sizes=sizes)
    assert classification.strategy == "aio_singlefile"
    plan = ing.HFSourcePlan(
        repo_id="ResembleAI/chatterbox", revision="deadbeef", paths=paths,
        sizes=sizes, side={}, classification=classification, content_ids={})

    def fake_dl(repo_id, rev, dest, *, allow_patterns, **kw):
        for p in allow_patterns:
            target = Path(dest) / p
            if p.endswith(".safetensors"):
                header = json.dumps({
                    "weight": {
                        "dtype": "BF16",
                        "shape": [1],
                        "data_offsets": [0, 2],
                    },
                }, separators=(",", ":")).encode()
                target.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            else:
                target.write_bytes(b"x")

    monkeypatch.setattr(ing, "_snapshot_download_with_retries", fake_dl)
    monkeypatch.setattr(ing, "install_hf_http_timeouts", lambda: None)
    src = ing.ingest_huggingface("ResembleAI/chatterbox", tmp_path / "d", plan=plan)
    assert src.repo_spec["library_name"] == ""
    assert src.metadata["multi_weight_bundle"] == "true"
    assert src.attrs["dtype"] == "bf16"

    from gen_worker.convert.clone import OutputSpec, build_flavor_tree

    tree, attrs = build_flavor_tree(
        src,
        OutputSpec(dtype="source", file_layout="singlefile", file_type="safetensors"),
        tmp_path / "source-flavor",
    )
    assert attrs["dtype"] == "bf16"
    assert len(list(tree.glob("*.safetensors"))) == 3


def test_gguf_multiquant_repo_size_gate_on_selected_set() -> None:
    """gw#483: the too_large gate applies to the SELECTED quant, not the
    whole multi-quant repo (unsloth repos total 100s of GB; one quant ~18GB)."""
    gib = 1024 * 1024 * 1024
    files = [f"m-{q}.gguf" for q in (
        "Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0",
        "UD-Q4_K_XL", "UD-Q8_K_XL",
    )] + ["mmproj-F16.gguf", "README.md"]
    sizes = {p: 20 * gib for p in files if p.endswith(".gguf")}
    c = classify_repo(files, sizes=sizes, gguf_quant="UD-Q4_K_XL")
    assert c.strategy == "gguf"
    assert [p for p in c.allow_patterns if p.endswith(".gguf")] == ["m-UD-Q4_K_XL.gguf"]
    # 20GiB selected << 160GiB repo total: must NOT refuse.
    assert c.attrs["dtype"] == "gguf:ud-q4_k_xl"

    # A single selected gguf above the threshold still refuses.
    with pytest.raises(RepoRefusal) as exc:
        classify_repo(
            ["huge-Q8_0.gguf"], sizes={"huge-Q8_0.gguf": 200 * gib})
    assert exc.value.reason == "too_large"


def test_gguf_ud_and_iquant_tokens_labeled() -> None:
    c = classify_repo(["Qwen3.6-27B-UD-Q4_K_XL.gguf"])
    assert c.attrs["dtype"] == "gguf:ud-q4_k_xl"
    c2 = classify_repo(["model-IQ4_XS.gguf"])
    assert c2.attrs["dtype"] == "gguf:iq4_xs"


def test_pipeline_tree_trellis_shape() -> None:
    files = [
        "pipeline.json",
        "README.md",
        "assets/teaser.png",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.json",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    ]
    c = classify_repo(files)
    assert c.strategy == "pipeline_tree"
    assert c.runtime_library == "trellis2"
    allow = set(c.allow_patterns)
    # EVERY safetensors rides — mixed per-model dtypes are intentional, no
    # dtype-variant pick.
    for p in files:
        if p.endswith((".safetensors", ".json")):
            assert p in allow, p
    assert "assets/teaser.png" not in allow
    assert c.attrs["file_layout"] == "singlefile"


def test_pipeline_tree_without_weights_falls_through() -> None:
    with pytest.raises(RepoRefusal):
        classify_repo(["pipeline.json", "README.md"])


def test_variant_tag_ignores_embedded_version_numbers() -> None:
    """gw#593: a real HF repo's root safetensors bundle, each name carrying
    its own dotted version number (not a diffusers dtype-variant suffix).
    Real Lightricks/LTX-2.3 filenames + sizes (2026-07-19 tree API). Before
    the fix, _variant_tag misread "2.3"/"1.0"/"1.1" as variant tags, split
    every file into its own bogus group, and the alphabetically-first
    fallback group happened to be the 3 upscaler files ONLY — SILENTLY
    excluding the actual 22B DiT checkpoint entirely (found live: e2e#185's
    clone published a mirror with zero base-model weights, gw#593).

    After the fix every file lands untagged (no false dtype match), so they
    group into ONE ~147GB bundle spanning dev+distilled+lora+upscaler
    checkpoints — a real refusal (`too_large`, over the 100GB gate) instead
    of a silent wrong-file publish. This is the correct interim behavior:
    fail loud, not fail silently wrong. Actually selecting the ONE intended
    checkpoint (`ltx-2.3-22b-dev.safetensors`) needs an explicit
    caller-supplied selector — gw#593 item 2, deliberately not attempted
    here (a real API-surface change, not a regex fix)."""
    files_sizes = {
        ".gitattributes": 1571,
        "LICENSE": 21399,
        "README.md": 6570,
        "ltx-2.3-22b-dev.safetensors": 46149344974,
        "ltx-2.3-22b-distilled-1.1.safetensors": 46149345334,
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors": 7605507256,
        "ltx-2.3-22b-distilled-lora-384.safetensors": 7605507256,
        "ltx-2.3-22b-distilled.safetensors": 46149345038,
        "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors": 1090125794,
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": 995743504,
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": 995743504,
        "ltx-2.3-temporal-upscaler-x2-1.0.safetensors": 995743504,
        "ltx2.3-open.png": 1000,
    }
    with pytest.raises(RepoRefusal) as exc:
        classify_repo(list(files_sizes), sizes=files_sizes)
    assert exc.value.reason == "too_large"


def test_variant_tag_no_longer_silently_drops_base_checkpoint() -> None:
    """Narrower proof of the gw#593 false-positive fix in isolation, without
    the too_large refusal: a small subset (dev + one upscaler only, sizes
    that fit under the gate) must group TOGETHER (both untagged), never
    excluding the dev checkpoint the way the buggy regex did."""
    files_sizes = {
        "ltx-2.3-22b-dev.safetensors": 5_000_000_000,
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": 995_743_504,
    }
    c = classify_repo(list(files_sizes), sizes=files_sizes)
    assert c.strategy == "aio_singlefile"
    assert set(c.allow_patterns) == set(files_sizes)


def test_variant_tag_still_recognizes_real_dtype_suffixes() -> None:
    """Guardrail: gw#593's fix must not break the genuine diffusers
    dtype-variant convention it was designed for."""
    from gen_worker.convert.classifier import _variant_tag

    assert _variant_tag("diffusion_pytorch_model.fp16.safetensors") == "fp16"
    assert _variant_tag("diffusion_pytorch_model.bf16.safetensors") == "bf16"
    assert _variant_tag("model.fp8_e4m3fn.safetensors") == "fp8_e4m3fn"
    # The gw#593 false positives: version numbers are NOT dtype tags.
    assert _variant_tag("ltx-2.3-22b-dev.safetensors") == ""
    assert _variant_tag("ltx-2.3-22b-distilled-1.1.safetensors") == ""
    assert _variant_tag("model-v1.0.safetensors") == ""
