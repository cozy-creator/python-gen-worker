"""pgw#1633 — the skeleton-conformance suite: no GPU, no weights, whole fleet.

pgw#1626 was found by a rented pod. `skeleton.build` never ran `tie_weights()`,
so every tied-encoder pipeline failed 100% of its invokes on a CORRECT
checkpoint — the tie a checkpoint relies on to omit an alias was broken by the
meta build and never re-established, and `meta_survivors` (rightly) refused.
It was fixed for the two classes that had already been hit (T5, Qwen3).

The class is not "T5 and Qwen3". It is "a checkpoint omits a parameter because
the class ties it, and the loader must put the tie back". That is a property of
every architecture the fleet will ever serve, and the only honest way to know
whether the NEXT one has it is to ask each one, from its configs, before it is
ever bound.

So this suite asks. For every fleet pipeline tree in
`tests/fixtures/checkpoint-configs`:

1. meta-build every weight-bearing component (`skeleton.build_modules`);
2. work out what a saved checkpoint would CARRY — every parameter except the
   ones that alias another after `retie()`, which is exactly what
   `save_pretrained` drops;
3. fill those, and only those, the way `StreamingLoader._install` does — a
   rebind of `_parameters[leaf]`, not a `copy_`;
4. `retie()`;
5. assert `meta_survivors == ()` and that every alias IS its source by
   IDENTITY (a copy would double the VRAM and serve stale bytes).

Step 2 derives the tie structure from the MODEL, never from the
`_tied_weights_keys` class attribute. That attribute lists names a class MIGHT
tie; whether the tie is live is a config question (`tie_word_embeddings`), and
reading the attribute as an answer scores `Qwen2_5_VLForConditionalGeneration`
— which declares `lm_head.weight` and does not tie it — as a failure on a
checkpoint that carries the tensor perfectly well.

NO WEIGHTS AND NO GPU. Every fill is a zero-storage `expand` of a scalar, so a
66 GB DiT costs the same as a tiny one; the whole suite is configs and shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from gen_worker.serving.streaming import skeleton  # noqa: E402

CORPUS = Path(__file__).parent / "fixtures" / "checkpoint-configs"

# THE ROSTER IS DATA, AND IT IS CHECKED. Coverage that can quietly shrink is
# not coverage: deleting a fixture must turn this suite red, not green with
# fewer cases. Each entry is the endpoint directory and the pipeline class its
# model_index.json names.
FLEET: Mapping[str, str] = {
    "ernie": "ErnieImagePipeline",
    "flux.1-dev": "FluxPipeline",
    "flux.1-schnell": "FluxPipeline",
    "flux.2-klein-4b": "Flux2KleinPipeline",
    "flux.2-klein-9b": "Flux2KleinPipeline",
    "foundation-1": "StableAudioPipeline",
    "qwen-image": "QwenImagePipeline",
    "sd15": "StableDiffusionPipeline",
    "sdxl": "StableDiffusionXLPipeline",
    "stable-audio-open": "StableAudioPipeline",
    "wan-2.2": "WanPipeline",
    "z-image": "ZImagePipeline",
}

# The tie exposures this corpus is here to hold down, named per (endpoint,
# component) so that losing one is a red test rather than a quieter suite.
# `flux.2-klein-*`'s Qwen3 and `wan-2.2`'s UMT5 are the two pgw#1626 recorded
# as forward exposure on their held 2.0.0 majors; they are pre-cleared here.
EXPECTED_TIES: Mapping[Tuple[str, str], str] = {
    ("ernie", "pe"): "Ministral3ForCausalLM",
    ("flux.1-dev", "text_encoder_2"): "T5EncoderModel",
    ("flux.1-schnell", "text_encoder_2"): "T5EncoderModel",
    ("flux.2-klein-4b", "text_encoder"): "Qwen3ForCausalLM",
    ("flux.2-klein-9b", "text_encoder"): "Qwen3ForCausalLM",
    ("foundation-1", "text_encoder"): "T5EncoderModel",
    ("stable-audio-open", "text_encoder"): "T5EncoderModel",
    ("wan-2.2", "text_encoder"): "UMT5EncoderModel",
}


def _alias_map(module: Any) -> Dict[str, str]:
    """`alias name -> source name` for every tie this module actually has.

    Derived from the module after `retie()`, by object IDENTITY. A saved
    checkpoint carries the source and omits the alias, so this is precisely
    the set the store will not be able to fill.
    """
    skeleton.retie(module)
    seen: Dict[int, str] = {}
    aliases: Dict[str, str] = {}
    for name, tensor in module.named_parameters(remove_duplicate=False):
        source = seen.get(id(tensor))
        if source is None:
            seen[id(tensor)] = name
        else:
            aliases[name] = source
    return aliases


def _fill(module: Any, omitted: Mapping[str, str]) -> int:
    """Bind every parameter the checkpoint WOULD carry to a real tensor.

    The rebind is `_parameters[leaf] = Parameter(...)`, which is what
    `StreamingLoader._install` does — an in-place `copy_` cannot be used
    against meta and would also hide the very thing this suite measures, since
    a tie broken by the meta build would silently survive a copy into the
    source.

    The tensor is a zero-strided expand of a scalar: correct dtype, correct
    shape, no storage. Shapes are not what is under test here; presence is.
    """
    filled = 0
    for name, tensor in list(module.named_parameters(remove_duplicate=False)):
        if tensor.device.type != "meta" or name in omitted:
            continue
        parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
        leaf = name.rsplit(".", 1)[-1]
        real = torch.zeros((), dtype=tensor.dtype).expand(tensor.shape)
        parent._parameters[leaf] = torch.nn.Parameter(real, requires_grad=False)
        filled += 1
    return filled


def _tree(endpoint: str) -> Path:
    return CORPUS / endpoint


def test_the_corpus_is_exactly_the_declared_fleet_pgw1633() -> None:
    """A fixture that disappears must be a failure, never a smaller run."""
    on_disk = {p.name for p in CORPUS.iterdir() if p.is_dir()}
    assert on_disk == set(FLEET), (
        "the vendored corpus and the declared roster disagree; "
        f"on disk only: {sorted(on_disk - set(FLEET))}, "
        f"declared only: {sorted(set(FLEET) - on_disk)}"
    )


@pytest.mark.parametrize("endpoint", sorted(FLEET))
def test_the_index_names_a_pipeline_class_this_image_can_build_pgw1633(endpoint: str) -> None:
    """`model_index.json` is the CLASS MAP, and a class map must map.

    A `_class_name` this image cannot resolve is a tree no worker on this
    build can load — the same answer, three weeks earlier and for free.
    """
    index_path, index = skeleton.read_index(_tree(endpoint))
    class_name = index.get("_class_name")
    assert class_name == FLEET[endpoint], f"{index_path} names {class_name!r}"
    import diffusers

    assert hasattr(diffusers, str(class_name)), (
        f"{index_path} names pipeline class {class_name!r}, which this image's "
        f"diffusers does not carry"
    )


@pytest.mark.parametrize("endpoint", sorted(FLEET))
def test_every_declared_module_comes_off_meta_pgw1633(endpoint: str) -> None:
    """Fill by the checkpoint's key set, retie, and leave nothing on meta."""
    tree = _tree(endpoint)
    ties = {name: _alias_map(module) for name, module in skeleton.build_modules(tree).items()}

    modules = skeleton.build_modules(tree)
    assert modules, f"{endpoint} declares no weight-bearing component"
    for name, module in modules.items():
        aliases = ties[name]
        assert _fill(module, aliases) > 0, f"{endpoint}/{name} has no parameters to fill"
        skeleton.retie(module)
        survivors = skeleton.meta_survivors(module)
        assert survivors == (), (
            f"{endpoint}/{name} ({type(module).__name__}): {len(survivors)} parameter(s) "
            f"stay on meta after a checkpoint-shaped fill and retie: {survivors[:8]}"
        )
        for alias, source in aliases.items():
            assert module.get_parameter(alias) is module.get_parameter(source), (
                f"{endpoint}/{name}: {alias} is a COPY of {source}, not a tie — "
                f"that doubles the resident bytes and serves whichever one the "
                f"stream wrote last"
            )


@pytest.mark.parametrize("endpoint", sorted({ep for ep, _ in EXPECTED_TIES}))
def test_the_retie_is_what_closes_it_pgw1633(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The red arm, per endpoint: without `retie`, the incident returns.

    A green suite whose fill happens to cover everything proves nothing. For
    every tied component in the corpus, the same fill with `retie` neutered
    must leave exactly the alias on meta — which is pgw#1626's pod failure,
    reproduced at $0.
    """
    tree = _tree(endpoint)
    ties = {name: _alias_map(module) for name, module in skeleton.build_modules(tree).items()}
    tied = {name: aliases for name, aliases in ties.items() if aliases}
    assert tied, f"{endpoint} was listed as a tie exposure but declares no tie"

    monkeypatch.setattr(skeleton, "retie", lambda module: False)
    modules = skeleton.build_modules(tree)
    for name, aliases in tied.items():
        module = modules[name]
        _fill(module, aliases)
        survivors = skeleton.meta_survivors(module)
        assert set(survivors) == set(aliases), (
            f"{endpoint}/{name}: with retie disabled the survivors are {survivors}, "
            f"expected exactly the aliases {sorted(aliases)} — this arm no longer "
            f"proves the assertion has teeth"
        )


def test_the_tie_exposures_are_the_ones_recorded_pgw1633() -> None:
    """The tie roster is measured, not asserted from memory.

    pgw#1626 recorded wan-2.2 (UMT5) and the Qwen3 family as forward exposure
    on their held 2.0.0 majors. This is that pre-clearance as a checkmark: the
    classes must still be here, still tied, and no NEW tied component may
    appear in the corpus without being written down.
    """
    found: Dict[Tuple[str, str], str] = {}
    for endpoint in sorted(FLEET):
        for name, module in skeleton.build_modules(_tree(endpoint)).items():
            if _alias_map(module):
                found[(endpoint, name)] = type(module).__name__
    assert found == dict(EXPECTED_TIES), (
        f"the corpus's tie structure moved.\n  new: "
        f"{sorted(set(found) - set(EXPECTED_TIES))}\n  gone: "
        f"{sorted(set(EXPECTED_TIES) - set(found))}"
    )


def test_a_component_the_index_declares_but_the_tree_lacks_is_refused_pgw1633(
    tmp_path: Path,
) -> None:
    """th#2265's shape, from the loader's side.

    A declared module component with no directory is a tree contradicting its
    own index. `build_modules` names it rather than quietly returning a
    smaller module set — the same refusal `_plan()` makes, one step earlier.
    """
    index: Dict[str, object] = {
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
    }
    src = _tree("sdxl")
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "config.json").write_text((src / "unet" / "config.json").read_text())
    (tmp_path / skeleton.MODEL_INDEX).write_text(json.dumps(index))
    assert skeleton.build_modules(tmp_path)

    index["vae"] = ["diffusers", "AutoencoderKL"]
    (tmp_path / skeleton.MODEL_INDEX).write_text(json.dumps(index))
    with pytest.raises(skeleton.SkeletonError, match="vae"):
        skeleton.build_modules(tmp_path)


def test_a_passthrough_component_needs_no_files_to_answer_the_modules_pgw1633(
    tmp_path: Path,
) -> None:
    """A tokenizer's bytes are not the meta question.

    Every checkpoint-config fixture on the fleet omits `vocab.json`/`merges.txt`
    deliberately (megabytes of real bytes, no weights), and one of them —
    wan-2.2, a held major this suite exists to pre-clear — ships no
    `tokenizer/` directory at all. If the module answer depended on the
    passthrough half, the classes that most need checking would be the ones
    that never get checked.
    """
    src = _tree("sdxl")
    index = {
        "_class_name": "StableDiffusionXLPipeline",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "config.json").write_text((src / "vae" / "config.json").read_text())
    (tmp_path / skeleton.MODEL_INDEX).write_text(json.dumps(index))

    modules = skeleton.build_modules(tmp_path)
    assert sorted(modules) == ["vae"]


def test_build_and_build_modules_read_one_index_pgw1633() -> None:
    """The two entry points must not become two parsers.

    `build` is what serving calls; `build_modules` is what this suite and the
    release-build fence call. They share `component_specs`, and this pins the
    consequence: the module sets agree, component for component.
    """
    import diffusers

    tree = _tree("sdxl")
    full = skeleton.build(getattr(diffusers, FLEET["sdxl"]), tree)
    modules_only = skeleton.build_modules(tree)
    assert sorted(full.modules) == sorted(modules_only)


def test_the_corpus_carries_no_weights_pgw1633() -> None:
    """The weights-locality rule, as a test.

    This corpus is configs. A `.safetensors` (or any tensor container) landing
    in it is a mistake that must fail here, not in a review.
    """
    offenders: List[str] = []
    for path in CORPUS.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".index.json"):
            continue
        if any(name.endswith(ext) for ext in (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt")):
            offenders.append(str(path.relative_to(CORPUS)))
    assert not offenders, f"tensor containers in a config-only corpus: {offenders}"
