"""Discovery emission for Slot-declared endpoints (pgw#520 / th#767): the
`slots` manifest block, and that Slot endpoints never emit the legacy
ModelChoice `model.choices[]` list (no first-party curated list, th#767
second refinement)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_slot_emits_slots_block_not_model_choices(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_slot"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import HF, RequestContext, Resources, Slot, endpoint
        from gen_worker.families import SdxlDefaults

        class In_(msgspec.Struct):
            prompt: str = ""
            model: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(
            models={
                "pipeline": Slot(
                    object, selected_by="model",
                    default_checkpoint=HF("stabilityai/stable-diffusion-xl-base-1.0"),
                    default_config=SdxlDefaults(steps=28, guidance=6.0),
                ),
                "vae": HF("madebyollin/sdxl-vae-fp16-fix"),
            },
            resources=Resources(vram_gb=12),
        )
        class Gen:
            def setup(self, pipeline: object, vae: object) -> None: ...
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_slot.main")
    (fn,) = fns
    assert "model" not in fn  # no ModelChoice/model.choices[] surface

    (slot,) = fn["slots"]
    assert slot["name"] == "pipeline"
    assert slot["pipeline_class"] == "builtins.object"
    assert slot["selected_by"] == "model"
    assert slot["default_checkpoint"] == {
        "source": "huggingface",
        "path": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    assert slot["family"] == "sdxl"
    assert slot["default_config"]["steps"] == 28
    assert slot["default_config"]["guidance"] == 6.0

    # The plain vae binding still emits through the ordinary bindings block
    # (back-compat: bare ModelRef contributes no `slots` entry).
    assert "vae" not in {s["name"] for s in fn["slots"]}
    assert fn["bindings"]["vae"]["provider"] == "huggingface"


def test_slot_with_no_selected_by_or_default_emits_minimal_block(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_slot_bare"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import RequestContext, Slot, endpoint
        from gen_worker.families import SdxlDefaults

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(model=Slot(object, default_config=SdxlDefaults(steps=20)))
        class Gen:
            def setup(self, model: object) -> None: ...
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_slot_bare.main")
    (fn,) = fns
    (slot,) = fn["slots"]
    assert slot["name"] == "model"
    assert "selected_by" not in slot
    assert "default_checkpoint" not in slot
    assert slot["family"] == "sdxl"  # from the default_config preset's registration


def test_slot_family_from_fallback_when_no_compile(tmp_pkg: Path) -> None:
    """pgw#520 reconciliation of pgw#519's family stamping (pgw#523: the
    stamp function is now unconditional-when-known, not allow_lora-
    triggered): a Slot-declared binding with no Compile(family=...)
    resolves its family stamp from the Slot's own fallback-preset
    registration."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_slot_lora"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import Hub, RequestContext, Slot, endpoint
        from gen_worker.families import SdxlDefaults

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(model=Slot(
            object,
            default_checkpoint=Hub("o/base"),
            default_config=SdxlDefaults(steps=28),
        ))
        class Gen:
            def setup(self, model: object) -> None: ...
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_slot_lora.main")
    (fn,) = fns
    assert "allow_lora" not in fn["bindings"]["model"]
    assert fn["bindings"]["model"]["family"] == "sdxl"


def test_compile_family_wins_over_slot_fallback_family(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_slot_compile_wins"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        import gen_worker
        from gen_worker import Compile, Hub, RequestContext, Slot, endpoint
        from gen_worker.families import SdxlDefaults

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(
            model=Slot(object, default_checkpoint=Hub("o/base"),
                       default_config=SdxlDefaults(steps=28)),
            compile=Compile(family="explicit-family", shapes=((512, 512),)),
        )
        class Gen:
            def setup(self, model: object) -> None:
                gen_worker.arm_compile(model)  # pgw#517: self-loaded slot
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_slot_compile_wins.main")
    (fn,) = fns
    assert fn["bindings"]["model"]["family"] == "explicit-family"
    (slot,) = fn["slots"]
    assert slot["family"] == "explicit-family"
