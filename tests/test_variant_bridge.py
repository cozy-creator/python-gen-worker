from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.serving.variants import VariantAmbiguous, detect_variant


def _tree(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    (root / "model_index.json").write_text("{}")
    return root


def test_a_variant_only_tree_names_its_variant(tmp_path: Path) -> None:
    """The sd1.5 shape, exactly as the mirror ships it."""

    tree = _tree(
        tmp_path / "sd15",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "text_encoder/model.fp16.safetensors",
    )
    assert detect_variant(tree) == "fp16"


def test_a_SHARDED_variant_only_tree_names_its_variant(tmp_path: Path) -> None:
    """The sdxl shape: the variant is glued to the shard suffix, and the tree carries BOTH index spellings at once — `_add_variant` writes one and older diffusers wrote the other."""

    tree = _tree(
        tmp_path / "sdxl",
        "unet/diffusion_pytorch_model.fp16-00001-of-00003.safetensors",
        "unet/diffusion_pytorch_model.fp16-00002-of-00003.safetensors",
        "unet/diffusion_pytorch_model.fp16-00003-of-00003.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors.index.json",
        "unet/diffusion_pytorch_model.safetensors.index.fp16.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/model.fp16.safetensors",
    )
    assert detect_variant(tree) == "fp16"


def test_a_PLAIN_tree_is_left_alone(tmp_path: Path) -> None:
    """Every published/converted checkpoint."""

    tree = _tree(
        tmp_path / "converted",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
    )
    assert detect_variant(tree) is None


def test_a_tree_carrying_BOTH_names_is_left_alone(tmp_path: Path) -> None:
    """diffusers' ordinary ladder already resolves this one, so the variant is an extra rather than the only way in — and choosing it would silently change which bytes serve."""

    tree = _tree(
        tmp_path / "both",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
    )
    assert detect_variant(tree) is None


def test_TWO_variants_and_no_plain_name_REFUSES_by_name(tmp_path: Path) -> None:
    """Picking would be the worker guessing about PRECISION on bytes the publisher already labelled."""

    tree = _tree(
        tmp_path / "ambiguous",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.bf16.safetensors",
    )
    with pytest.raises(VariantAmbiguous) as caught:
        detect_variant(tree)
    assert "bf16" in str(caught.value) and "fp16" in str(caught.value)


def test_an_index_json_is_not_mistaken_for_a_variant(tmp_path: Path) -> None:
    """A greedy pattern reads `model.safetensors.index.json` as variant `index`, silently, and then asks diffusers for files that do not exist."""

    tree = _tree(
        tmp_path / "sharded-plain",
        "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
        "unet/diffusion_pytorch_model-00002-of-00002.safetensors",
        "unet/diffusion_pytorch_model.safetensors.index.json",
    )
    assert detect_variant(tree) is None


def test_a_missing_or_empty_tree_answers_None_rather_than_raising(
    tmp_path: Path,
) -> None:
    assert detect_variant(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert detect_variant(tmp_path / "empty") is None


def test_the_bridge_PASSES_the_variant_it_detected(tmp_path: Path) -> None:

    from gen_worker.serving.context import DeployBinding, LoadContext

    tree = _tree(
        tmp_path / "variant-only",
        "unet/diffusion_pytorch_model.fp16.safetensors",
    )
    seen: dict = {}

    class Pipe:
        @classmethod
        def from_pretrained(cls, path, **kwargs):  # type: ignore[no-untyped-def]
            seen.update(kwargs)
            seen["path"] = path
            return cls()

        def to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

    ctx: LoadContext = LoadContext(
        binding=DeployBinding(checkpoint_ref="x/y@1", checkpoint_dir=tree))
    ctx.load(Pipe)  # type: ignore[arg-type]

    assert seen["variant"] == "fp16", (
        "the bridge detected the variant and then did not pass it — the "
        "pgw#1460 shape")
    assert seen["path"] == tree


def test_the_bridge_passes_NO_variant_for_a_plain_tree(tmp_path: Path) -> None:
    """The counter-case, without which the assertion above could be a constant."""

    from gen_worker.serving.context import DeployBinding, LoadContext

    tree = _tree(tmp_path / "plain", "unet/diffusion_pytorch_model.safetensors")
    seen: dict = {}

    class Pipe:
        @classmethod
        def from_pretrained(cls, path, **kwargs):  # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return cls()

        def to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

    ctx: LoadContext = LoadContext(
        binding=DeployBinding(checkpoint_ref="x/y@1", checkpoint_dir=tree))
    ctx.load(Pipe)  # type: ignore[arg-type]
    assert "variant" not in seen
