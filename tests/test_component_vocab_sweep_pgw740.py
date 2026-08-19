"""pgw#740 B5/B6: one ``declare_components`` call reaches every consumer.

B5 landed the vocabulary module; the 20+ copies of ``("transformer","unet",…)``
across ``models/`` and ``convert/`` were repointed to it. These tests pin the
OUTCOME the sweep bought: a declared component reaches every consumer, and it
reaches consumers whose modules were imported BEFORE the declaration ran —
the import-time-freeze hazard that dropped Wan's ``transformer_2`` and LTX's
``connectors``.
"""

from __future__ import annotations

import pytest

from gen_worker.component_vocab import (
    component_vocabulary,
    declare_components,
    quant_candidate_components,
    reset_component_vocabulary,
    weight_components,
)


@pytest.fixture(autouse=True)
def _clean_vocabulary():
    reset_component_vocabulary()
    yield
    reset_component_vocabulary()


def test_a_declared_component_reaches_every_swept_consumer() -> None:
    """One declaration, every consumer — the point of the sweep.

    LTX's ``connectors`` is the real case: it carries weights, so it must be
    walked, sized, offloaded and quantized, and it is not a diffusers name.
    """
    from gen_worker.convert import size_walk, source
    from gen_worker.convert.layout_spec import LayoutSignals
    from gen_worker.models import memory

    assert "connectors" not in weight_components()

    declare_components(auxiliaries=("connectors",))

    assert component_vocabulary().role_of("connectors") == "auxiliary"
    assert "connectors" in weight_components()
    assert "connectors" in quant_candidate_components()
    # ...and in each swept consumer's own derived view
    assert "connectors" in size_walk._diffusers_weight_component_dirs()
    assert "connectors" in source._weight_component_dirs()
    assert "connectors" in source._diffusers_component_dirs()
    assert "connectors" in LayoutSignals().component_dirs
    assert "connectors" in memory._component_order_hint()


def test_a_declaration_after_import_is_still_seen() -> None:
    """The freeze hazard, directly. These modules are imported at the top of
    this test session; the declaration happens now. A module-level tuple would
    have snapshotted the vocabulary before this line and missed it."""
    from gen_worker.convert import size_walk

    before = size_walk._diffusers_weight_component_dirs()
    declare_components(auxiliaries=("motion_adapter",))
    after = size_walk._diffusers_weight_component_dirs()

    assert "motion_adapter" not in before
    assert "motion_adapter" in after


def test_a_declared_component_is_sized_rather_than_counted_as_zero(tmp_path) -> None:
    """End of the line for the bug: an undeclared component contributes zero
    bytes to the size facts the orchestrator gates VRAM placement on."""
    from gen_worker.convert.size_walk import compute_size_facts

    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "m.safetensors").write_bytes(b"x" * 2048)
    (tmp_path / "connectors").mkdir()
    (tmp_path / "connectors" / "m.safetensors").write_bytes(b"y" * 1024)

    undeclared = compute_size_facts(tmp_path)
    assert "connectors" not in undeclared["components"]
    assert undeclared["full_model_bytes"] == 2048

    declare_components(auxiliaries=("connectors",))
    declared = compute_size_facts(tmp_path)
    assert declared["components"]["connectors"]["total_bytes"] == 1024
    assert declared["full_model_bytes"] == 3072


def test_both_moe_experts_are_lora_branch_targets() -> None:
    """gw#679's defect restated as a vocabulary test: a dual-expert pipeline
    must offer BOTH experts as branch targets, or the low expert serves
    undistilled weights on a distilled ladder."""
    from gen_worker.models.w8a8_lora import branch_targets

    class _Mod:
        def named_modules(self):
            return iter(())

    class _WanMoE:
        transformer = _Mod()
        transformer_2 = _Mod()

    assert set(branch_targets(_WanMoE())) == {"transformer", "transformer_2"}


def test_stream_residency_reads_no_component_vocabulary_at_def_time() -> None:
    """The rung that replaced the block-window offload (pgw#1497) takes no
    component vocabulary at all — it discovers leaves from the tree it is
    handed — so the def-time-frozen-default hazard this test was written for
    cannot recur here. Assert the shape, not the deleted signature."""
    import inspect

    from gen_worker.models.stream_residency import StreamedResidency

    params = inspect.signature(StreamedResidency.__init__).parameters
    assert "components" not in params
    for name, param in params.items():
        assert not isinstance(param.default, (tuple, list)) or not param.default, (
            f"{name}: a non-empty sequence default re-freezes a vocabulary at "
            "def time"
        )
