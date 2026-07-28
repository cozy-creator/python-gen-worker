"""pgw#740 B5/B6: the call-site sweep onto the ONE component vocabulary.

B5 landed the vocabulary module; the 20+ copies of ``("transformer","unet",…)``
across ``models/`` and ``convert/`` are repointed here. Two things are tested:

1. **The sweep holds** — an extension of #740's tokenizer-based acceptance grep
   to the swept files. A component-name string literal in LIVE code (comments
   and docstrings are provenance and stay) is a new copy, and a new copy is how
   Wan's ``transformer_2`` and LTX's ``connectors`` got silently dropped.

2. **B6, the correctness gap the copies caused** — one ``declare_components``
   call reaches every consumer, and it reaches them even though the consumer
   modules were imported BEFORE the declaration ran. That ordering is the whole
   hazard: a module-level tuple snapshots the vocabulary at import time, and
   endpoint declarations run at endpoint-module import, which is later.
"""

from __future__ import annotations

import re
import tokenize
from pathlib import Path

import pytest

from gen_worker.component_vocab import (
    component_vocabulary,
    declare_components,
    denoiser_components,
    quant_candidate_components,
    reset_component_vocabulary,
    weight_components,
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "gen_worker"


@pytest.fixture(autouse=True)
def _clean_vocabulary():
    reset_component_vocabulary()
    yield
    reset_component_vocabulary()


# ---------------------------------------------------------------------------
# 1. the sweep, as a permanent test
# ---------------------------------------------------------------------------

#: Files the B5 sweep repointed. A component literal in any of these is a
#: regression unless it is allow-listed below with a reason.
_SWEPT_FILES = (
    "models/loading.py",
    "models/memory.py",
    "models/download.py",
    "models/w8a8.py",
    "models/w4a4.py",
    "models/w8a8_lora.py",
    "models/svdq.py",
    "models/gguf_local.py",
    "models/provision.py",
    "utils/lora.py",
    "convert/convert.py",
    "convert/source.py",
    "convert/writer.py",
    "convert/clone.py",
    "convert/size_walk.py",
    "convert/svdq.py",
    "convert/layout_spec.py",
    "cli/serve.py",
)

#: (file, literal) -> why this one is NOT a vocabulary copy. Each names a
#: single specific thing, never an enumeration of "the components".
_ALLOWED = {
    ("models/memory.py", "vae"):
        "diffusers' VAE slicing/tiling APIs hang off the .vae attribute by "
        "name; this addresses that one attribute, it does not enumerate.",
    ("models/w8a8_lora.py", "unet"):
        "diffusers' lora_state_dict(unet_config=...) keyword — upstream's "
        "parameter name, not our component vocabulary.",
    ("convert/writer.py", "decoder"):
        "a layer-name pattern matched INSIDE a model; collides with the "
        "component name by coincidence.",
    ("convert/writer.py", "scheduler"):
        "one specific path, out_dir/scheduler/config.json, for the distilled "
        "scheduler overrides.",
    ("utils/lora.py", "text_encoder"):
        "right-hand side of the kohya alias table: sd-scripts' fixed grammar "
        "(lora_te1_/lora_te_) mapped ONTO a vocabulary name. Upstream decides "
        "which aliases exist, so the table is correctly literal.",
    ("utils/lora.py", "text_encoder_2"): "kohya alias target — see above.",
    ("utils/lora.py", "text_encoder_3"): "kohya alias target — see above.",
}


def _code_literals(path: Path) -> list[tuple[int, str]]:
    """(lineno, value) for every string literal in LIVE code.

    Comments and docstrings are stripped: a component name surviving in prose
    is provenance and is allowed — exactly the rule #740's acceptance grep uses.
    """
    out: list[tuple[int, str]] = []
    with open(path, "rb") as handle:
        for tok in tokenize.tokenize(handle.readline):
            if tok.type != tokenize.STRING:
                continue
            raw = tok.string.lstrip("rbufRBUF")
            if raw.startswith(('"""', "'''")):
                continue  # docstring
            out.append((tok.start[0], raw[1:-1]))
    return out


def test_no_swept_file_repeats_the_component_vocabulary() -> None:
    names = set(component_vocabulary().all)
    offenders: list[str] = []
    for rel in _SWEPT_FILES:
        for lineno, value in _code_literals(_SRC / rel):
            if value in names and (rel, value) not in _ALLOWED:
                offenders.append(f"{rel}:{lineno}: {value!r}")
    assert offenders == [], (
        "component-name literals in live code — these are new copies of the "
        "vocabulary. Read it from gen_worker.component_vocab instead, or "
        "allow-list with a reason if it names one specific thing:\n"
        + "\n".join(offenders)
    )


def test_the_sweep_left_no_module_level_snapshot() -> None:
    """The copies were module-level tuples; a replacement that is still a
    module-level tuple has fixed nothing. Every swept module must read the
    vocabulary through a call."""
    # Anchored at column 0: an indented `components = denoiser_components()`
    # inside a function IS the correct call-time pattern. Binding the RESULT
    # at module scope is the bug; binding the FUNCTION (no parens) is fine.
    snapshot = re.compile(
        r"^[A-Za-z_]\w*(?:\s*:[^=\n]+)?\s*=\s*"
        r"(?:denoiser|weight|quant_candidate|pipeline_component|text_encoder)"
        r"[a-z_]*\(\)",
        re.MULTILINE,
    )
    offenders: list[str] = []
    for rel in _SWEPT_FILES:
        for hit in snapshot.finditer((_SRC / rel).read_text()):
            offenders.append(f"{rel}: module-level snapshot {hit.group(0)!r}")
    assert offenders == [], (
        "these bind the vocabulary at import time, before an endpoint's "
        "declare_components() runs:\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. B6 — the correctness gap: transformer_2 and connectors
# ---------------------------------------------------------------------------

def test_wan_moe_second_expert_is_a_denoiser_everywhere() -> None:
    """``transformer_2`` (Wan 2.2 A14B's low-noise expert) was absent from
    several copies, so it was skipped by the offload and quant loops."""
    assert "transformer_2" in denoiser_components()
    assert "transformer_2" in weight_components()
    assert "transformer_2" in quant_candidate_components()


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
    assert "connectors" in source._default_quant_candidate_components()
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


def test_block_window_offload_defaults_to_every_denoiser() -> None:
    """``apply_block_window_offload``'s default was a literal
    ``("transformer","unet")`` in the signature — a default argument, so even
    repointing it would have frozen the vocabulary at def time."""
    import inspect

    from gen_worker.models.loading import apply_block_window_offload

    default = inspect.signature(apply_block_window_offload).parameters["components"].default
    assert default is None, "a non-None default re-freezes the vocabulary at def time"
