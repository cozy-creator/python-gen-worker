"""pgw#1080 increment 2: the component-level structure-only builder.

The property under test is the one ie#638 needs: the process that EXPORTS a
family's compile target never holds that family's checkpoint values. It is
proved here on the micro family's real tree, through the real SDK seams —
``run_setup`` -> ``provision.load_slot`` -> the pipeline class's own
``from_pretrained`` — not against a stub, because the failure this replaces
was precisely a load path that looked right and loaded weights anyway.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from gen_worker import meta_instantiation as mi  # noqa: E402
from gen_worker.models import structure_only as so  # noqa: E402


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from micro_diffusion.weights import SEED, materialize

    root = tmp_path_factory.mktemp("micro-tree")
    return materialize(root, seed=SEED)


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def test_a_compile_target_is_built_from_config_and_holds_no_weights(
    tree: Path,
) -> None:
    module, facts = so.build_component(tree, "transformer", device="cpu")

    assert facts.component == "transformer"
    assert facts.cls_name == "MicroDenoiser"
    assert facts.parameters > 0
    assert facts.virtual_param_bytes > 0, "the facts must price what was SKIPPED"
    for name, param in module.named_parameters():
        assert mi.is_virtual(param), f"{name} holds real storage"


def test_the_structure_claims_the_TARGET_device_not_meta(tree: Path) -> None:
    """AOTInductor codegens for the device the traced tensors report, so a
    structure that claimed ``meta`` would compile a cell for no card at all —
    and a meta parameter cannot even be exported beside a real input
    (measured: ``Tensor device mismatch … cpu and meta``)."""
    module, _facts = so.build_component(tree, "transformer", device="cpu")
    devices = {str(p.device) for p in module.parameters()}
    assert devices == {"cpu"}


def test_buffers_stay_REAL_because_literals_ship_from_them(tree: Path) -> None:
    """A config-derived table is what a literal-bearing cell packs. Faking it
    would make ``aot_package.literal_constants`` unpackable — and keeping it
    real is also the folding fence: a literal derived from a PARAMETER stays
    fake and fails loudly instead of baking one checkpoint into a shared
    cell."""
    module, facts = so.build_component(tree, "transformer", device="cpu")
    buffers = dict(module.named_buffers())
    assert buffers, "the micro denoiser registers a rope/frequency table"
    assert all(not mi.is_virtual(b) for b in buffers.values())
    assert facts.real_buffer_bytes > 0
    assert facts.real_buffer_bytes < facts.virtual_param_bytes


def test_a_class_without_the_config_surface_REFUSES_by_name(
    tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoConfigSurface:
        pass

    monkeypatch.setattr(so, "_component_class",
                        lambda *_a, **_k: NoConfigSurface)
    with pytest.raises(so.StructureOnlyUnsupported) as caught:
        so.build_component(tree, "transformer", device="cpu")
    message = str(caught.value)
    assert "NoConfigSurface" in message
    assert "load_config" in message and "from_config" in message
    assert caught.value.component == "transformer"


def test_a_component_the_tree_does_not_declare_REFUSES_by_name(
    tree: Path,
) -> None:
    with pytest.raises(so.StructureOnlyUnsupported) as caught:
        so.build_component(tree, "text_encoder", device="cpu")
    assert "model_index.json" in str(caught.value)
    assert "transformer" in str(caught.value), "it names what IS declared"


# ---------------------------------------------------------------------------
# Export + compile on the structure — the measurement that chose fake over meta
# ---------------------------------------------------------------------------


def test_the_structure_EXPORTS_and_AOT_COMPILES(tree: Path) -> None:
    from gen_worker import aot_mint

    module, _facts = so.build_component(tree, "decoder", device="cpu")
    mode = so.fake_mode_of(module)
    assert mode is not None

    with so.under(mode):
        latent = torch.randn(1, 8, 16)
        program = aot_mint.export_program(module, (latent,), {})
    assert so.fake_mode_of_program(program) is mode, (
        "the compile has to find the program's own mode — `aot_compile` "
        "asserts every input belongs to ONE mode")
    files = aot_mint.compile_entry_files(program, "decoder")
    assert files, "AOTInductor produced no loose files for the fake structure"


# ---------------------------------------------------------------------------
# The pgw#984 warm proof runs on RANDOM values, and gives them back
# ---------------------------------------------------------------------------


def test_random_values_are_defined_and_are_released_again(tree: Path) -> None:
    module, facts = so.build_component(tree, "transformer", device="cpu")

    materialized = so.materialize_random(module, device="cpu")
    assert materialized == facts.virtual_param_bytes
    values = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    assert not mi.is_virtual(next(module.parameters()))
    assert torch.isfinite(values).all(), (
        "a NaN/inf init would make every downstream cosine undefined")
    assert float(values.abs().max()) > 0.0

    so.restore_virtual(module, device="cpu")
    assert all(mi.is_virtual(p) for p in module.parameters())


def test_the_random_values_are_REPRODUCIBLE(tree: Path) -> None:
    """A proof that behaves differently on two runs of the same cell is not a
    proof, so the seed is fixed rather than clock-derived."""
    first, _ = so.build_component(tree, "decoder", device="cpu")
    second, _ = so.build_component(tree, "decoder", device="cpu")
    so.materialize_random(first, device="cpu")
    so.materialize_random(second, device="cpu")
    for (name, a), (_n, b) in zip(first.named_parameters(),
                                  second.named_parameters()):
        assert torch.equal(a, b), name


# ---------------------------------------------------------------------------
# The SDK seam — the composed pipeline, and the silent-ignore gate
# ---------------------------------------------------------------------------


def test_run_setup_composes_a_pipeline_whose_TARGETS_hold_no_weights(
    tree: Path,
) -> None:
    from gen_worker.cli.run import run_setup

    from micro_diffusion.main import Generate

    instance = Generate()
    loaded = run_setup(
        instance, {"pipeline": str(tree)}, device="cpu", arm_compile=False,
        return_loaded=True, structure_only=("transformer", "decoder")) or {}
    pipe = loaded["pipeline"]

    assert so.structure_only_components(pipe) == ("decoder", "transformer")
    for param in pipe.transformer.parameters():
        assert mi.is_virtual(param)
    for param in pipe.decoder.parameters():
        assert mi.is_virtual(param)
    facts = so.facts_of(pipe)
    assert len(facts) == 2
    assert sum(f.virtual_param_bytes for f in facts) > 0


def test_a_pipeline_that_IGNORES_the_injection_is_refused_not_trusted(
    tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``components=`` is a REQUEST. A class free to swallow the keyword and
    load the checkpoint itself would hand a mint every weight it exists to
    avoid, silently — so the composed object is checked, and the refusal names
    what it built instead."""
    from gen_worker.cli import run as run_mod

    class Ignores:
        transformer = "a real module, loaded from the checkpoint"

    with pytest.raises(so.StructureOnlyUnsupported) as caught:
        run_mod._assert_structure_honored(
            Ignores(), {"transformer": object()}, slot="pipeline")
    assert "does not carry the injected structure-only module" in str(
        caught.value)


def test_the_micro_tree_names_its_component_classes(tree: Path) -> None:
    """Without a component class map there is nothing to build from config;
    this is the packaging half of the contract and it is part of the tree."""
    from gen_worker.models.loading import model_index_entry

    assert model_index_entry(tree, "transformer") == (
        "micro_diffusion.model", "MicroDenoiser")
    assert model_index_entry(tree, "decoder") == (
        "micro_diffusion.model", "MicroDecoder")
