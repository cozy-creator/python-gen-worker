from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .writer import ConversionImplementationError

if TYPE_CHECKING:
    import torch


class NotImplementedFamilyError(ConversionImplementationError):
    pass


_SINGLEFILE_PIPELINE_CONFIGS: dict[str, tuple[str, ...]] = {
    "StableDiffusionPipeline": ("stable-diffusion-v1-5/stable-diffusion-v1-5",),
    "StableDiffusionXLPipeline": ("stabilityai/stable-diffusion-xl-base-1.0",),
    "FluxPipeline": ("black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"),
    "Flux2Pipeline": ("black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B"),
}


def _normalize_family(family: str | None) -> str:
    raw = str(family or "").strip().lower()
    if raw in {"sd15", "sd2", "sd1", "sd15_sd2"}:
        return "sd15_sd2"
    if raw in {"sdxl"}:
        return "sdxl"
    if raw in {"flux", "flux1", "flux2", "flex2"}:
        return "flux"
    if raw in {"auraflow"}:
        return "auraflow"
    if raw in {"wan", "wan21", "wan22"}:
        return "wan"
    return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text("utf-8")))


def _torch() -> Any:
    import torch  # type: ignore

    return torch


def _st_load(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file as st_load_file  # type: ignore

    return cast(dict[str, Any], st_load_file(str(path), device="cpu"))


def _st_save(state_dict: dict[str, Any], output_path: Path) -> None:
    from safetensors.torch import save_file as st_save_file  # type: ignore

    st_save_file(state_dict, str(output_path))


def _load_sharded_safetensors(index_json: Path) -> dict[str, torch.Tensor]:
    idx = _read_json(index_json)
    weight_map = idx.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("invalid_safetensors_index")
    # Load each shard once, merge dicts. This is memory heavy but conversion needs full weights anyway.
    shard_names: list[str] = sorted({str(v) for v in weight_map.values() if isinstance(v, str) and v.strip()})
    out: dict[str, torch.Tensor] = {}
    for name in shard_names:
        shard_path = index_json.parent / name
        out.update(cast(dict[str, Any], _st_load(shard_path)))
    return out


def _load_component_state_dict(
    component_dir: Path,
    *,
    safetensors_bases: list[str],
    bin_base: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Loads a diffusers component state_dict from common weight naming patterns.

    Supported safetensors names:
    - <base>.safetensors
    - <base>.safetensors.index.json (sharded)
    - <base>.bin.safetensors (common in some HF repos)
    - <base>.bin.safetensors.index.json (rare; but cheap to support)
    - <base>.<variant>.safetensors and its sharded index (HF dtype-variant
      downloads keep the suffix — e.g. diffusion_pytorch_model.fp16.safetensors
      in repo-cas mirrors cloned with a dtype preference)

    Optional legacy fallback (unsafe for untrusted repos):
    - <bin_base>.bin
    """
    for base in safetensors_bases:
        base = str(base or "").strip()
        if not base:
            continue
        st_path = component_dir / f"{base}.safetensors"
        st_index = component_dir / f"{base}.safetensors.index.json"
        if st_path.exists():
            return cast(dict[str, Any], _st_load(st_path))
        if st_index.exists():
            return _load_sharded_safetensors(st_index)

        st_path = component_dir / f"{base}.bin.safetensors"
        st_index = component_dir / f"{base}.bin.safetensors.index.json"
        if st_path.exists():
            return cast(dict[str, Any], _st_load(st_path))
        if st_index.exists():
            return _load_sharded_safetensors(st_index)

        # Dtype-variant names, exact-name misses only.
        for st in sorted(component_dir.glob(f"{base}.*.safetensors")):
            return cast(dict[str, Any], _st_load(st))
        for idx in sorted(component_dir.glob(f"{base}.safetensors.*.index.json")) + sorted(
            component_dir.glob(f"{base}.*.safetensors.index.json")
        ):
            return _load_sharded_safetensors(idx)

    if bin_base:
        bin_path = component_dir / f"{bin_base}.bin"
        if bin_path.exists():
            torch_mod = _torch()
            return cast(dict[str, Any], torch_mod.load(str(bin_path), map_location="cpu"))
    raise FileNotFoundError(f"missing weights for {component_dir.name}")


def _maybe_cast_tensor(t: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    torch_mod = _torch()
    if not isinstance(t, torch_mod.Tensor):
        return t
    if t.is_floating_point():
        return t.to(dtype=out_dtype)
    return t


def _cast_state_dict(sd: dict[str, torch.Tensor], out_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {k: _maybe_cast_tensor(v, out_dtype) for k, v in sd.items()}


def detect_diffusers_family(model_dir: Path) -> str:
    """
    Detect a model family from a diffusers layout.
    v1 only needs to distinguish sd1/2 vs sdxl vs unsupported.
    """
    model_index = model_dir / "model_index.json"
    cls = ""
    if model_index.exists():
        try:
            mi = _read_json(model_index)
            cls = str(mi.get("_class_name") or "").strip()
        except Exception:
            cls = ""

    # Prefer component-set detection (more robust than class_name).
    if (model_dir / "text_encoder_2").exists() or (model_dir / "tokenizer_2").exists():
        return "sd15_sd2"
    if (model_dir / "transformer").exists():
        # Transformer-first pipelines (Flux/SD3/etc.) are not supported for checkpoint exports (v1).
        return "flux"
    if (model_dir / "unet").exists() and (model_dir / "vae").exists() and (model_dir / "text_encoder").exists():
        # Stable Diffusion 1.x/2.x family (includes many fine-tunes).
        return "sd15_sd2"

    # Class-name fallback.
    if "StableDiffusionXL" in cls:
        return "sdxl"
    if "Flux" in cls:
        return "flux"
    if "StableDiffusion" in cls:
        return "sd15_sd2"
    return "unknown"


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _singlefile_attempts_for_family(model_family: str) -> list[tuple[str, str | None]]:
    family = _normalize_family(model_family)
    attempts: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str]] = set()

    def add(pipeline_class: str, config: str | None = None) -> None:
        cls = str(pipeline_class or "").strip()
        cfg = str(config or "").strip()
        if cls == "":
            return
        key = (cls, cfg)
        if key in seen:
            return
        seen.add(key)
        attempts.append((cls, cfg or None))

    if family == "sd15_sd2":
        add("StableDiffusionPipeline")
        for cfg in _SINGLEFILE_PIPELINE_CONFIGS.get("StableDiffusionPipeline", ()):
            add("StableDiffusionPipeline", cfg)
        return attempts
    if family == "sdxl":
        add("StableDiffusionXLPipeline")
        for cfg in _SINGLEFILE_PIPELINE_CONFIGS.get("StableDiffusionXLPipeline", ()):
            add("StableDiffusionXLPipeline", cfg)
        return attempts
    if family == "flux":
        for cls in ("Flux2Pipeline", "FluxPipeline"):
            add(cls)
            for cfg in _SINGLEFILE_PIPELINE_CONFIGS.get(cls, ()):
                add(cls, cfg)
        return attempts
    return attempts


def _load_singlefile_pipeline(*, input_path: Path, pipeline_class: str, config: str | None) -> Any:
    import diffusers

    cls = getattr(diffusers, pipeline_class, None)
    if cls is None:
        raise ValueError(f"pipeline_class_unavailable:{pipeline_class}")

    kwargs: dict[str, Any] = {"torch_dtype": _torch().bfloat16}
    if config:
        kwargs["config"] = config

    if hasattr(cls, "from_single_file"):
        return cls.from_single_file(str(input_path), **kwargs)

    from diffusers.loaders.single_file import FromSingleFileMixin

    fn = getattr(FromSingleFileMixin.from_single_file, "__func__", None)
    if fn is None:
        raise ValueError(f"pipeline_not_implemented:{pipeline_class}")
    return fn(cls, str(input_path), **kwargs)


def singlefile_to_diffusers(input_path: Path, output_dir: Path, *, model_family: str) -> dict[str, str]:
    family = _normalize_family(model_family)
    attempts = _singlefile_attempts_for_family(family)
    if not attempts:
        raise ConversionImplementationError(
            f"unsupported_family_for_layout_conversion:{family}:singlefile_to_diffusers"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for pipeline_class, config in attempts:
        try:
            pipe = _load_singlefile_pipeline(
                input_path=input_path,
                pipeline_class=pipeline_class,
                config=config,
            )
            pipe.save_pretrained(str(output_dir), safe_serialization=True)
            return {
                "pipeline_class": pipeline_class,
                "pipeline_config": str(config or ""),
                "output_dir": str(output_dir),
            }
        except Exception as exc:  # noqa: BLE001
            cfg = f" config={config}" if config else ""
            errors.append(f"{pipeline_class}{cfg}: {exc}")
            continue

    raise ConversionImplementationError(
        "singlefile_to_diffusers_failed:" + "; ".join(errors[:5])
    )


def convert_diffusers_to_singlefile(
    model_dir: Path,
    output_path: Path,
    *,
    family: str | None = None,
    output_dtype: str = "fp16",
) -> None:
    fam = _normalize_family(family)
    if fam == "unknown":
        fam = detect_diffusers_family(model_dir)
    if fam not in {"sd15_sd2", "sdxl"}:
        raise NotImplementedFamilyError(f"diffusers_to_singlefile not implemented for family={fam}")

    dtype_name = str(output_dtype or "").strip().lower()
    if dtype_name in {"fp16", "f16", "float16"}:
        out_dtype = _torch().float16
    elif dtype_name in {"bf16", "bfloat16"}:
        out_dtype = _torch().bfloat16
    elif dtype_name in {"fp32", "f32", "float32"}:
        out_dtype = _torch().float32
    else:
        raise ValueError("invalid_output_dtype")

    if fam == "sdxl":
        state_dict = _convert_sdxl(model_dir)
    else:
        state_dict = _convert_sd15_sd2(model_dir)

    state_dict = _cast_state_dict(state_dict, out_dtype)
    _ensure_parent(output_path)
    _st_save(cast(dict[str, Any], state_dict), output_path)


# ---- SD 1.x / 2.x (based on diffusers' conversion script) ----

def _convert_sd15_sd2(model_path: Path) -> dict[str, torch.Tensor]:
    unet_path = model_path / "unet"
    vae_path = model_path / "vae"
    text_enc_path = model_path / "text_encoder"

    unet_state_dict = _load_component_state_dict(
        unet_path,
        safetensors_bases=["diffusion_pytorch_model"],
        bin_base="diffusion_pytorch_model",
    )
    vae_state_dict = _load_component_state_dict(
        vae_path,
        safetensors_bases=["diffusion_pytorch_model"],
        bin_base="diffusion_pytorch_model",
    )
    text_enc_dict = _load_component_state_dict(
        text_enc_path,
        safetensors_bases=["model", "pytorch_model"],
        bin_base="pytorch_model",
    )

    # Convert.
    unet_state_dict = _convert_sd_unet_state_dict(unet_state_dict)
    vae_state_dict = _convert_sd_vae_state_dict(vae_state_dict)

    is_v20_model = "text_model.embeddings.position_ids" in text_enc_dict
    if is_v20_model:
        text_enc_dict = _convert_sd_text_enc_state_dict_v20(text_enc_dict)
        text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
    else:
        # v1 already matches expected keyspace.
        text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}

    # Final keyspace prefixes.
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    out: dict[str, torch.Tensor] = {}
    out.update(text_enc_dict)
    out.update(unet_state_dict)
    out.update(vae_state_dict)
    return out


def _convert_sd_unet_state_dict(unet_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Ported from diffusers' convert_diffusers_to_original_stable_diffusion.py (sd 1/2).
    """
    unet_conversion_map = [
        ("time_embed.0.weight", "time_embed.0.weight"),
        ("time_embed.0.bias", "time_embed.0.bias"),
        ("time_embed.2.weight", "time_embed.2.weight"),
        ("time_embed.2.bias", "time_embed.2.bias"),
        ("conv_in.weight", "input_blocks.0.0.weight"),
        ("conv_in.bias", "input_blocks.0.0.bias"),
        ("conv_norm_out.weight", "out.0.weight"),
        ("conv_norm_out.bias", "out.0.bias"),
        ("conv_out.weight", "out.2.weight"),
        ("conv_out.bias", "out.2.bias"),
    ]
    unet_conversion_map_resnet = [
        ("norm1", "in_layers.0"),
        ("conv1", "in_layers.2"),
        ("norm2", "out_layers.0"),
        ("conv2", "out_layers.3"),
        ("time_emb_proj", "emb_layers.1"),
        ("conv_shortcut", "skip_connection"),
    ]

    # Build layer maps (matches upstream script structure).
    for i in range(4):
        for j in range(2):
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map += [
                (hf_down_res_prefix, sd_down_res_prefix),
            ]
            if i < 3:
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map += [
                    (hf_down_atn_prefix, sd_down_atn_prefix),
                ]

        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map += [(hf_downsample_prefix, sd_downsample_prefix)]

    hf_mid_res_prefix = "mid_block.resnets.0."
    sd_mid_res_prefix = "middle_block.0."
    unet_conversion_map += [(hf_mid_res_prefix, sd_mid_res_prefix)]
    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map += [(hf_mid_atn_prefix, sd_mid_atn_prefix)]
    hf_mid_res_prefix = "mid_block.resnets.1."
    sd_mid_res_prefix = "middle_block.2."
    unet_conversion_map += [(hf_mid_res_prefix, sd_mid_res_prefix)]

    for i in range(4):
        for j in range(3):
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map += [(hf_up_res_prefix, sd_up_res_prefix)]
            if i > 0:
                hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
                unet_conversion_map += [(hf_up_atn_prefix, sd_up_atn_prefix)]

        if i < 3:
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.1."
            unet_conversion_map += [(hf_upsample_prefix, sd_upsample_prefix)]

    sd_state_dict: dict[str, torch.Tensor] = {}
    for k, v in unet_state_dict.items():
        new_key = k
        for hf_prefix, sd_prefix in unet_conversion_map:
            if new_key.startswith(hf_prefix):
                new_key = new_key.replace(hf_prefix, sd_prefix)

        # Resnet key shims.
        for hf_part, sd_part in unet_conversion_map_resnet:
            new_key = new_key.replace(hf_part, sd_part)

        sd_state_dict[new_key] = v
    return sd_state_dict


def _convert_sd_vae_state_dict(vae_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    vae_conversion_map = [
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid_block.attentions.0.", "mid.attn_1."),
    ]
    for i in range(4):
        for j in range(2):
            hf_down_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map += [(hf_down_prefix, sd_down_prefix)]

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map += [(hf_downsample_prefix, sd_downsample_prefix)]

            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"up.{3-i}.upsample."
            vae_conversion_map += [(hf_upsample_prefix, sd_upsample_prefix)]

        for j in range(3):
            hf_up_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
            vae_conversion_map += [(hf_up_prefix, sd_up_prefix)]

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"mid.block_{j+1}."
        vae_conversion_map += [(hf_mid_res_prefix, sd_mid_res_prefix)]

    vae_conversion_map_attn = [
        ("to_q", "q"),
        ("to_k", "k"),
        ("to_v", "v"),
        ("to_out.0", "proj_out"),
    ]

    sd_state_dict: dict[str, torch.Tensor] = {}
    for k, v in vae_state_dict.items():
        new_key = k
        for hf_prefix, sd_prefix in vae_conversion_map:
            if new_key.startswith(hf_prefix):
                new_key = new_key.replace(hf_prefix, sd_prefix)
        for hf_part, sd_part in vae_conversion_map_attn:
            new_key = new_key.replace(hf_part, sd_part)
        sd_state_dict[new_key] = v
    return sd_state_dict


def _convert_sd_text_enc_state_dict_v20(text_enc_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Ported from upstream script; minimal regex mapping.
    textenc_conversion_lst = [
        ("text_model.encoder.layers.", "resblocks."),
        ("self_attn.q_proj.", "attn.in_proj_"),
        ("self_attn.k_proj.", "attn.in_proj_"),
        ("self_attn.v_proj.", "attn.in_proj_"),
        ("self_attn.out_proj.", "attn.out_proj."),
        ("layer_norm1.", "ln_1."),
        ("layer_norm2.", "ln_2."),
        ("mlp.fc1.", "mlp.c_fc."),
        ("mlp.fc2.", "mlp.c_proj."),
        ("final_layer_norm.", "ln_final."),
        ("text_model.embeddings.position_embedding.weight", "positional_embedding"),
        ("text_model.embeddings.token_embedding.weight", "token_embedding.weight"),
        ("text_model.embeddings.", ""),
        ("text_model.final_layer_norm.", "ln_final."),
        ("text_model.encoder.", ""),
        ("text_projection.weight", "text_projection"),
    ]

    import re

    protected = {re.escape(src): dst for src, dst in textenc_conversion_lst}
    pattern = re.compile("|".join(protected.keys()))

    def _repl(m: re.Match[str]) -> str:
        return protected[re.escape(m.group(0))]

    out: dict[str, torch.Tensor] = {}
    for k, v in text_enc_dict.items():
        k2 = pattern.sub(_repl, k)
        out[k2] = v
    return out


# ---- SDXL (based on diffusers' conversion script) ----


def _convert_sdxl(model_path: Path) -> dict[str, torch.Tensor]:
    unet_path = model_path / "unet"
    vae_path = model_path / "vae"
    text_enc_path = model_path / "text_encoder"
    text_enc2_path = model_path / "text_encoder_2"

    unet_state_dict = _load_component_state_dict(
        unet_path,
        safetensors_bases=["diffusion_pytorch_model"],
        bin_base="diffusion_pytorch_model",
    )
    vae_state_dict = _load_component_state_dict(
        vae_path,
        safetensors_bases=["diffusion_pytorch_model"],
        bin_base="diffusion_pytorch_model",
    )
    text_enc_dict = _load_component_state_dict(
        text_enc_path,
        safetensors_bases=["model", "pytorch_model"],
        bin_base="pytorch_model",
    )
    text_enc2_dict = _load_component_state_dict(
        text_enc2_path,
        safetensors_bases=["model", "pytorch_model"],
        bin_base="pytorch_model",
    )

    unet_state_dict = _convert_sdxl_unet_state_dict(unet_state_dict)
    vae_state_dict = _convert_sdxl_vae_state_dict(vae_state_dict)

    # The SDXL script treats text_encoder (OpenAI CLIP) as already in expected keyspace.
    text_enc_dict = {"conditioner.embedders.0.transformer." + k: v for k, v in text_enc_dict.items()}
    text_enc2_dict = _convert_sdxl_text_enc_state_dict(text_enc2_dict)
    text_enc2_dict = {"conditioner.embedders.1.model." + k: v for k, v in text_enc2_dict.items()}

    # SDXL uses a transposed text projection.
    if "conditioner.embedders.1.model.text_projection" in text_enc2_dict:
        text_enc2_dict["conditioner.embedders.1.model.text_projection"] = text_enc2_dict[
            "conditioner.embedders.1.model.text_projection"
        ].t().contiguous()

    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    out: dict[str, torch.Tensor] = {}
    out.update(text_enc_dict)
    out.update(text_enc2_dict)
    out.update(unet_state_dict)
    out.update(vae_state_dict)
    return out


def _convert_sdxl_unet_state_dict(unet_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    unet_conversion_map = [
        ("time_embedding.linear_1.weight", "time_embed.0.weight"),
        ("time_embedding.linear_1.bias", "time_embed.0.bias"),
        ("time_embedding.linear_2.weight", "time_embed.2.weight"),
        ("time_embedding.linear_2.bias", "time_embed.2.bias"),
        ("conv_in.weight", "input_blocks.0.0.weight"),
        ("conv_in.bias", "input_blocks.0.0.bias"),
        ("conv_norm_out.weight", "out.0.weight"),
        ("conv_norm_out.bias", "out.0.bias"),
        ("conv_out.weight", "out.2.weight"),
        ("conv_out.bias", "out.2.bias"),
        ("add_embedding.linear_1.weight", "label_emb.0.0.weight"),
        ("add_embedding.linear_1.bias", "label_emb.0.0.bias"),
        ("add_embedding.linear_2.weight", "label_emb.0.2.weight"),
        ("add_embedding.linear_2.bias", "label_emb.0.2.bias"),
    ]

    unet_conversion_map_resnet = [
        ("norm1", "in_layers.0"),
        ("conv1", "in_layers.2"),
        ("norm2", "out_layers.0"),
        ("conv2", "out_layers.3"),
        ("time_emb_proj", "emb_layers.1"),
        ("conv_shortcut", "skip_connection"),
    ]

    unet_conversion_map_layer = [
        ("proj_in", "proj_in"),
        ("proj_out", "proj_out"),
        ("transformer_blocks.0.attn1.to_q", "transformer_blocks.0.attn1.to_q"),
        ("transformer_blocks.0.attn1.to_k", "transformer_blocks.0.attn1.to_k"),
        ("transformer_blocks.0.attn1.to_v", "transformer_blocks.0.attn1.to_v"),
        ("transformer_blocks.0.attn1.to_out.0", "transformer_blocks.0.attn1.to_out.0"),
        ("transformer_blocks.0.attn2.to_q", "transformer_blocks.0.attn2.to_q"),
        ("transformer_blocks.0.attn2.to_k", "transformer_blocks.0.attn2.to_k"),
        ("transformer_blocks.0.attn2.to_v", "transformer_blocks.0.attn2.to_v"),
        ("transformer_blocks.0.attn2.to_out.0", "transformer_blocks.0.attn2.to_out.0"),
        ("transformer_blocks.0.ff.net.0.proj", "transformer_blocks.0.ff.net.0.proj"),
        ("transformer_blocks.0.ff.net.2", "transformer_blocks.0.ff.net.2"),
        ("transformer_blocks.0.norm1", "transformer_blocks.0.norm1"),
        ("transformer_blocks.0.norm2", "transformer_blocks.0.norm2"),
        ("transformer_blocks.0.norm3", "transformer_blocks.0.norm3"),
    ]

    for i in range(3):
        for j in range(2):
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map += [(hf_down_res_prefix, sd_down_res_prefix)]
            if i < 3:
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map += [(hf_down_atn_prefix, sd_down_atn_prefix)]

        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map += [(hf_downsample_prefix, sd_downsample_prefix)]

    for j in range(2):
        hf_down_res_prefix = f"down_blocks.3.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*3 + j + 1}.0."
        unet_conversion_map += [(hf_down_res_prefix, sd_down_res_prefix)]

    hf_mid_res_prefix = "mid_block.resnets.0."
    sd_mid_res_prefix = "middle_block.0."
    unet_conversion_map += [(hf_mid_res_prefix, sd_mid_res_prefix)]
    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map += [(hf_mid_atn_prefix, sd_mid_atn_prefix)]
    hf_mid_res_prefix = "mid_block.resnets.1."
    sd_mid_res_prefix = "middle_block.2."
    unet_conversion_map += [(hf_mid_res_prefix, sd_mid_res_prefix)]

    for i in range(4):
        for j in range(3):
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map += [(hf_up_res_prefix, sd_up_res_prefix)]
            if i > 0:
                hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
                unet_conversion_map += [(hf_up_atn_prefix, sd_up_atn_prefix)]
        if i < 3:
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.1."
            unet_conversion_map += [(hf_upsample_prefix, sd_upsample_prefix)]

    sd_state_dict: dict[str, torch.Tensor] = {}
    for k, v in unet_state_dict.items():
        new_key = k
        for hf_prefix, sd_prefix in unet_conversion_map:
            if new_key.startswith(hf_prefix):
                new_key = new_key.replace(hf_prefix, sd_prefix)
        for hf_part, sd_part in unet_conversion_map_resnet:
            new_key = new_key.replace(hf_part, sd_part)
        for hf_part, sd_part in unet_conversion_map_layer:
            new_key = new_key.replace(hf_part, sd_part)
        sd_state_dict[new_key] = v
    return sd_state_dict


def _convert_sdxl_vae_state_dict(vae_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    vae_conversion_map = [
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid_block.attentions.0.", "mid.attn_1."),
    ]
    for i in range(4):
        for j in range(2):
            hf_down_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map += [(hf_down_prefix, sd_down_prefix)]
        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map += [(hf_downsample_prefix, sd_downsample_prefix)]
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"up.{3-i}.upsample."
            vae_conversion_map += [(hf_upsample_prefix, sd_upsample_prefix)]
        for j in range(3):
            hf_up_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
            vae_conversion_map += [(hf_up_prefix, sd_up_prefix)]
    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"mid.block_{j+1}."
        vae_conversion_map += [(hf_mid_res_prefix, sd_mid_res_prefix)]

    sd_state_dict: dict[str, torch.Tensor] = {}
    for k, v in vae_state_dict.items():
        new_key = k
        for hf_prefix, sd_prefix in vae_conversion_map:
            if new_key.startswith(hf_prefix):
                new_key = new_key.replace(hf_prefix, sd_prefix)
        sd_state_dict[new_key] = v
    return sd_state_dict


def _convert_sdxl_text_enc_state_dict(text_enc_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Ported from upstream SDXL script; regex based mapping.
    textenc_conversion_lst = [
        ("transformer.resblocks.", "resblocks."),
        ("ln_1", "ln_1"),
        ("ln_2", "ln_2"),
        (".c_fc.", ".mlp.c_fc."),
        (".c_proj.", ".mlp.c_proj."),
        (".attn", ".attn"),
        ("ln_final.", "ln_final."),
        ("token_embedding.weight", "token_embedding.weight"),
        ("positional_embedding", "positional_embedding"),
        ("text_projection", "text_projection"),
    ]

    import re

    protected = {re.escape(src): dst for src, dst in textenc_conversion_lst}
    pattern = re.compile("|".join(protected.keys()))

    def _repl(m: re.Match[str]) -> str:
        return protected[re.escape(m.group(0))]

    out: dict[str, torch.Tensor] = {}
    for k, v in text_enc_dict.items():
        k2 = pattern.sub(_repl, k)
        out[k2] = v
    return out


def diffusers_to_singlefile(input_dir: Path, output_path: Path, *, model_family: str) -> dict[str, str]:
    convert_diffusers_to_singlefile(
        input_dir,
        output_path,
        family=model_family,
        output_dtype="bf16",
    )
    return {
        "model_family": _normalize_family(model_family),
        "output_path": str(output_path),
    }


__all__ = [
    "singlefile_to_diffusers",
    "diffusers_to_singlefile",
    "detect_diffusers_family",
    "convert_diffusers_to_singlefile",
]
