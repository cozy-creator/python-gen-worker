from __future__ import annotations

import re
import tokenize
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gen_worker.component_vocab import (
    component_vocabulary,
    declare_components,
    reset_component_vocabulary,
)
from gen_worker.convert import base_model_families, layout, registry, repackage
from gen_worker.convert.layout_spec import DirMatch, HintMatch, LayoutDeclaration
from gen_worker.convert.repack_spec import (
    ComponentRepack,
    DeclarationError,
    LayoutSignature,
    RenameRule,
    RepackVariant,
    RepackageFamily,
    SinglefileTarget,
)


@pytest.fixture(autouse=True)
def _clean_registries():
    registry.reset_registries()
    base_model_families.reset_foreign_family_maps()
    reset_component_vocabulary()
    yield
    registry.reset_registries()
    base_model_families.reset_foreign_family_maps()
    reset_component_vocabulary()


_DENOISER = ComponentRepack(
    name="unet",
    safetensors_bases=("diffusion_pytorch_model",),
    variants=(
        RepackVariant(
            out_prefix="model.diffusion_model.",
            rules=(
                RenameRule(kind="prefix", pairs=(("conv_in.", "input_blocks.0.0."),)),
                RenameRule(kind="substring", pairs=(("norm1", "in_layers.0"),)),
            ),
        ),
    ),
)

_ENCODER = ComponentRepack(
    name="text_encoder",
    safetensors_bases=("model", "pytorch_model"),
    variants=(
        RepackVariant(
            probe_key="text_model.embeddings.position_ids",
            out_prefix="cond_stage_model.model.",
            rules=(RenameRule(kind="alternation", pairs=(("text_model.encoder.", ""),)),),
        ),
        RepackVariant(out_prefix="cond_stage_model.transformer."),
    ),
)

_ENCODER_2 = ComponentRepack(
    name="text_encoder_2",
    safetensors_bases=("model",),
    variants=(
        RepackVariant(
            out_prefix="conditioner.embedders.1.model.",
            transpose_keys=("text_projection",),
        ),
    ),
)

ALPHA = RepackageFamily(
    family="alpha",
    aliases=("alpha-legacy",),
    alias_prefixes=("alpha-",),
    signature=LayoutSignature(requires_all=("unet", "text_encoder"), forbids=("text_encoder_2",)),
    to_singlefile=(_DENOISER, _ENCODER),
    to_diffusers=(SinglefileTarget("AlphaPipeline", config_repos=("acme/alpha-base",)),),
)

BETA = RepackageFamily(
    family="beta",
    signature=LayoutSignature(requires_all=("unet",), requires_any=(("text_encoder_2", "tokenizer_2"),)),
    to_singlefile=(_DENOISER, _ENCODER, _ENCODER_2),
    to_diffusers=(SinglefileTarget("BetaPipeline"),),
)

GAMMA = RepackageFamily(
    family="gamma",
    to_diffusers=(SinglefileTarget("GammaPipeline"),),
)


def _write(component_dir: Path, base: str, tensors: dict[str, torch.Tensor]) -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(component_dir / f"{base}.safetensors"))


def _alpha_tree(root: Path, *, v2_probe: bool = False) -> Path:
    _write(root / "unet", "diffusion_pytorch_model", {
        "conv_in.weight": torch.ones(2, 2, dtype=torch.float32),
        "down.norm1.bias": torch.zeros(2, dtype=torch.float32),
    })
    encoder = {"text_model.encoder.layers.0.weight": torch.full((2, 2), 0.5, dtype=torch.float32)}
    if v2_probe:
        encoder["text_model.embeddings.position_ids"] = torch.zeros(2, dtype=torch.float32)
    _write(root / "text_encoder", "model", encoder)
    return root


def _beta_tree(root: Path) -> Path:
    _alpha_tree(root)
    _write(root / "text_encoder_2", "model", {
        "text_projection": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    })
    return root


def test_declared_family_round_trips_through_real_safetensors(tmp_path: Path) -> None:
    registry.register_repackage_family(ALPHA)
    tree = _alpha_tree(tmp_path / "src")
    out = tmp_path / "out" / "model.safetensors"

    repackage.convert_diffusers_to_singlefile(tree, out, family="alpha", output_dtype="fp16")

    produced = load_file(str(out))
    assert set(produced) == {
        "model.diffusion_model.input_blocks.0.0.weight",
        "model.diffusion_model.down.in_layers.0.bias",
        "cond_stage_model.transformer.text_model.encoder.layers.0.weight",
    }
    assert produced["model.diffusion_model.input_blocks.0.0.weight"].dtype == torch.float16
    assert torch.allclose(
        produced["cond_stage_model.transformer.text_model.encoder.layers.0.weight"].float(),
        torch.full((2, 2), 0.5),
    )


def test_probe_key_selects_the_declared_variant(tmp_path: Path) -> None:
    registry.register_repackage_family(ALPHA)
    tree = _alpha_tree(tmp_path / "src", v2_probe=True)
    out = tmp_path / "out.safetensors"

    repackage.convert_diffusers_to_singlefile(tree, out, family="alpha", output_dtype="bf16")

    produced = load_file(str(out))
    assert "cond_stage_model.model.layers.0.weight" in produced
    assert not any(k.startswith("cond_stage_model.transformer.") for k in produced)


def test_declared_transpose_is_applied(tmp_path: Path) -> None:
    registry.register_repackage_family(BETA)
    out = tmp_path / "out.safetensors"

    repackage.convert_diffusers_to_singlefile(
        _beta_tree(tmp_path / "src"), out, family="beta", output_dtype="fp32")

    produced = load_file(str(out))
    assert torch.equal(
        produced["conditioner.embedders.1.model.text_projection"],
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )


def test_onboarding_a_family_needs_no_sdk_change(tmp_path: Path) -> None:
    """The acceptance test, stated directly: a family the SDK has never heard of converts correctly the moment its declaration is registered — and is refused by name before that."""
    tree = _alpha_tree(tmp_path / "src")
    out = tmp_path / "out.safetensors"

    with pytest.raises(registry.UnknownFamilyError) as exc:
        repackage.convert_diffusers_to_singlefile(tree, out, family="alpha")
    assert "alpha" in str(exc.value)

    registry.register_repackage_family(ALPHA)
    repackage.convert_diffusers_to_singlefile(tree, out, family="alpha")
    assert out.exists()


def test_dual_encoder_tree_is_refused_by_the_single_encoder_converter(tmp_path: Path) -> None:
    registry.register_repackage_family(ALPHA)
    registry.register_repackage_family(BETA)

    with pytest.raises(repackage.ConverterSignatureError) as exc:
        repackage.convert_diffusers_to_singlefile(
            _beta_tree(tmp_path / "src"), tmp_path / "o.safetensors", family="alpha")
    message = str(exc.value)
    assert "alpha" in message and "text_encoder_2" in message


def test_single_encoder_tree_is_refused_by_the_dual_encoder_converter(tmp_path: Path) -> None:
    registry.register_repackage_family(ALPHA)
    registry.register_repackage_family(BETA)

    with pytest.raises(repackage.ConverterSignatureError):
        repackage.convert_diffusers_to_singlefile(
            _alpha_tree(tmp_path / "src"), tmp_path / "o.safetensors", family="beta")


def test_detection_and_the_guard_read_the_same_field(tmp_path: Path) -> None:
    registry.register_repackage_family(ALPHA)
    registry.register_repackage_family(BETA)

    assert repackage.detect_diffusers_family(_beta_tree(tmp_path / "b")) == "beta"
    assert repackage.detect_diffusers_family(_alpha_tree(tmp_path / "a")) == "alpha"

    out = tmp_path / "auto.safetensors"
    repackage.convert_diffusers_to_singlefile(tmp_path / "b", out)
    assert "conditioner.embedders.1.model.text_projection" in load_file(str(out))


def test_two_families_claiming_one_tree_is_refused_not_guessed(tmp_path: Path) -> None:
    registry.register_repackage_family(ALPHA)
    registry.register_repackage_family(
        RepackageFamily(
            family="alpha_twin",
            signature=LayoutSignature(requires_all=("unet",)),
            to_singlefile=(_DENOISER,),
        )
    )
    with pytest.raises(repackage.AmbiguousFamilyError) as exc:
        repackage.detect_diffusers_family(_alpha_tree(tmp_path / "src"))
    assert "alpha" in str(exc.value) and "alpha_twin" in str(exc.value)


def test_a_converter_without_a_signature_is_refused_at_declaration_time() -> None:
    with pytest.raises(DeclarationError) as exc:
        RepackageFamily(family="nosig", to_singlefile=(_DENOISER,))
    assert "EMPTY layout signature" in str(exc.value)


def test_a_component_whose_variants_never_match_is_refused_at_declaration_time() -> None:
    with pytest.raises(DeclarationError):
        ComponentRepack(
            name="unet",
            variants=(RepackVariant(probe_key="only.if.present"),),
        )


def test_missing_required_component_refuses_by_name(tmp_path: Path) -> None:
    registry.register_repackage_family(ALPHA)
    root = tmp_path / "src"
    _write(root / "unet", "diffusion_pytorch_model", {"conv_in.weight": torch.ones(1)})
    with pytest.raises(repackage.ConverterSignatureError) as exc:
        repackage.convert_diffusers_to_singlefile(root, tmp_path / "o.safetensors", family="alpha")
    assert "text_encoder" in str(exc.value)


def test_prefix_rule_only_rewrites_a_leading_match() -> None:
    rule = RenameRule(kind="prefix", pairs=(("conv_in.", "input_blocks.0.0."),))
    assert rule.apply("conv_in.weight") == "input_blocks.0.0.weight"
    assert rule.apply("mid.conv_in.weight") == "mid.conv_in.weight"


def test_substring_rule_rewrites_every_occurrence() -> None:
    rule = RenameRule(kind="substring", pairs=(("norm1", "in_layers.0"),))
    assert rule.apply("a.norm1.b.norm1") == "a.in_layers.0.b.in_layers.0"


def test_alternation_rule_is_single_pass_so_output_is_never_rewritten() -> None:
    """A sequential pass would rewrite its own output; the CLIP keyspaces depend on this not happening."""
    rule = RenameRule(kind="alternation", pairs=(("a.", "b."), ("b.", "c.")))
    assert rule.apply("a.x") == "b.x"


def test_alternation_prefers_the_longest_source() -> None:
    rule = RenameRule(kind="alternation", pairs=(("ln_", "LN"), ("ln_final.", "final.")))
    assert rule.apply("ln_final.weight") == "final.weight"


def test_empty_rule_is_refused() -> None:
    with pytest.raises(DeclarationError):
        RenameRule(kind="prefix", pairs=())
    with pytest.raises(DeclarationError):
        RenameRule(kind="nonsense", pairs=(("a", "b"),))  # type: ignore[arg-type]


def test_alias_and_prefix_normalization_replaces_the_ladder() -> None:
    registry.register_repackage_family(ALPHA)
    assert registry.normalize_family("alpha") == "alpha"
    assert registry.normalize_family("alpha-legacy") == "alpha"
    assert registry.normalize_family("alpha-illustrious-v3") == "alpha"
    assert registry.normalize_family("nobody") == "unknown"
    assert registry.normalize_family(None) == "unknown"


def test_unknown_family_names_what_is_registered() -> None:
    registry.register_repackage_family(ALPHA)
    with pytest.raises(registry.UnknownFamilyError) as exc:
        registry.require_repackage_family("delta")
    assert "delta" in str(exc.value) and "alpha" in str(exc.value)


def test_direction_is_declared_not_assumed(tmp_path: Path) -> None:
    registry.register_repackage_family(GAMMA)
    assert not GAMMA.supports_diffusers_to_singlefile
    assert GAMMA.supports_singlefile_to_diffusers
    with pytest.raises(repackage.NotImplementedFamilyError):
        repackage.convert_diffusers_to_singlefile(
            _alpha_tree(tmp_path / "src"), tmp_path / "o.safetensors", family="gamma")


def test_conflicting_registration_is_refused() -> None:
    registry.register_repackage_family(ALPHA)
    with pytest.raises(DeclarationError):
        registry.register_repackage_family(
            RepackageFamily(
                family="alpha",
                signature=LayoutSignature(requires_all=("unet",)),
                to_singlefile=(_DENOISER,),
            )
        )
    registry.register_repackage_family(ALPHA)


def test_alias_collision_between_families_is_refused() -> None:
    registry.register_repackage_family(ALPHA)
    with pytest.raises(DeclarationError):
        registry.register_repackage_family(
            RepackageFamily(family="other", aliases=("alpha-legacy",))
        )


_ALPHA_LAYOUT = LayoutDeclaration(
    variant="alpha1",
    family="alpha",
    order=20,
    hints=(HintMatch(any_tokens=("alpha1",)),),
    class_match=(HintMatch(any_tokens=("alphapipeline",)),),
    dirs=(DirMatch(requires=("unet", "text_encoder"), forbids=("text_encoder_2",)),),
)
_BETA_LAYOUT = LayoutDeclaration(
    variant="beta2",
    family="beta",
    order=10,
    hints=(HintMatch(any_tokens=("beta2",)),),
    dirs=(DirMatch(requires=("unet", "text_encoder_2"),),),
    root_sentinels=("video_vae_encoder.safetensors", "text_encoder_post_modules.safetensors"),
)


def test_variant_and_family_come_from_declarations() -> None:
    registry.register_layout(_ALPHA_LAYOUT)
    registry.register_layout(_BETA_LAYOUT)
    assert layout.infer_model_family_variant_from_hint("acme/Alpha-1_repackage") == "alpha1"
    assert layout.infer_model_family_from_hint("acme/Alpha-1") == "alpha"
    assert layout.canonical_model_family_from_variant("beta2") == "beta"
    assert layout.canonical_model_family_from_variant("nothing") == "unknown"


def test_punctuation_does_not_change_a_hint_match() -> None:
    registry.register_layout(_BETA_LAYOUT)
    for spelling in ("beta2", "beta-2", "BETA_2", "beta.2"):
        assert layout.infer_model_family_variant_from_hint(f"vendor/{spelling}-dev") == "beta2"


def test_root_sentinels_detect_a_layout_with_no_family_token(tmp_path: Path) -> None:
    registry.register_layout(_ALPHA_LAYOUT)
    registry.register_layout(_BETA_LAYOUT)
    info = layout.detect_huggingface_source_layout(
        repo_dir=tmp_path,
        files=[
            "transformer.safetensors",
            "video_vae_encoder.safetensors",
            "text_encoder_post_modules.safetensors",
        ],
    )
    assert (info.model_family_variant, info.model_family) == ("beta2", "beta")
    assert info.source_layout == "single-file"


def test_component_dirs_detect_the_variant(tmp_path: Path) -> None:
    registry.register_layout(_ALPHA_LAYOUT)
    registry.register_layout(_BETA_LAYOUT)
    info = layout.detect_huggingface_source_layout(
        repo_dir=tmp_path,
        files=["model_index.json", "unet/config.json", "text_encoder/config.json"],
    )
    assert info.model_family_variant == "alpha1"
    assert info.source_layout == "multi-file"

    dual = layout.detect_huggingface_source_layout(
        repo_dir=tmp_path,
        files=["unet/config.json", "text_encoder/config.json", "text_encoder_2/config.json"],
    )
    assert dual.model_family_variant == "beta2"


def test_order_keeps_a_specific_variant_ahead_of_a_broad_one() -> None:
    registry.register_layout(
        LayoutDeclaration(variant="broad", family="b", order=90,
                          hints=(HintMatch(any_tokens=("acme",)),)))
    registry.register_layout(
        LayoutDeclaration(variant="specific", family="s", order=10,
                          hints=(HintMatch(all_tokens=("acme", "xl")),)))
    assert layout.infer_model_family_variant_from_hint("acme-xl-v2") == "specific"
    assert layout.infer_model_family_variant_from_hint("acme-v2") == "broad"


def test_a_matcher_with_no_condition_is_refused() -> None:
    with pytest.raises(DeclarationError):
        HintMatch()
    with pytest.raises(DeclarationError):
        LayoutDeclaration(variant="v", family="f")


def test_nothing_is_detected_when_nothing_is_declared(tmp_path: Path) -> None:
    info = layout.detect_huggingface_source_layout(
        repo_dir=tmp_path, files=["unet/config.json", "model_index.json"])
    assert (info.model_family, info.model_family_variant) == ("unknown", "unknown")
    assert info.source_layout == "multi-file"


def test_civitai_map_is_injected_not_shipped() -> None:
    assert base_model_families.civitai_to_family("SDXL 1.0") is None
    base_model_families.declare_foreign_family_map(
        base_model_families.CIVITAI, {"SDXL 1.0": "sdxl", "Pony": "sdxl-pony"})
    assert base_model_families.civitai_to_family("SDXL 1.0") == "sdxl"
    assert base_model_families.civitai_to_family("Pony") == "sdxl-pony"
    assert base_model_families.civitai_to_family("Nothing Known") is None


def test_foreign_maps_are_namespaced_per_catalog() -> None:
    base_model_families.declare_foreign_family_map("othersite", {"X": "alpha"})
    assert base_model_families.foreign_to_family("othersite", "X") == "alpha"
    assert base_model_families.civitai_to_family("X") is None


def test_the_generic_vocabulary_covers_the_components_that_were_being_missed() -> None:
    vocab = component_vocabulary()
    assert vocab.is_denoiser("transformer_2")
    assert vocab.is_denoiser("transformer")
    assert vocab.is_denoiser("unet")
    assert not vocab.is_denoiser("vae")
    assert vocab.role_of("vae.decode") == "vae"
    assert vocab.role_of("text_encoder_2") == "text_encoder"
    assert vocab.role_of("connectors") == "unknown"


def test_an_endpoint_declares_its_own_components() -> None:
    declare_components(extras=("connectors",))
    vocab = component_vocabulary()
    assert "connectors" in vocab.all
    assert vocab.role_of("connectors") == "extra"
    before = component_vocabulary().denoisers
    declare_components(denoisers=("unet",))
    assert component_vocabulary().denoisers == before


