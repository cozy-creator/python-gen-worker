"""pgw#1143 steps 1-3 (§1.33): the per-slot layout DEMAND, end to end on the
SDK side.

What is proven here, in the order the design puts it:

1. a declared slot's demand reaches the PUBLISHED MANIFEST, per component
   path, in declaration order — that is the whole point, since the manifest is
   the only thing the hub reads;
2. absence is UNDECLARED — no key at all, not an empty one, so the hub cannot
   read it as "accepts everything";
3. an invalid declaration is refused WHERE IT IS WRITTEN — an unknown handle
   at the Slot constructor, a component key that matches nothing at
   decoration, a non-literal declaration at the lint;
4. the census cross reports an unbacked handle and does NOT refuse it.

Run: uv run pytest tests/test_slot_layout_demand_pgw1143.py
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import msgspec
import pytest

from gen_worker import RequestContext, Slot, endpoint
from gen_worker.discovery.discover import (
    _census_unbacked_layouts,
    _slot_to_manifest,
)
from gen_worker.discovery.execution_lanes import (
    DerivedContract,
    DerivedExecutionLanes,
)
from gen_worker.families.base import GenerationDefaults
from gen_worker.models.tensor_layout_contract import (
    CONTRACT_HF_FP8_BLOCKWISE,
    CONTRACT_PLAIN_BF16,
    LayoutDeclarationError,
)
from gen_worker.registry import extract_specs

REPO = Path(__file__).resolve().parents[1]


class _Vae:
    pass


class _TextEncoder:
    pass


class _Unet:
    pass


class FakePipeline:
    """An introspectable pipeline: the derived tree is {vae, text_encoder,
    unet}, through the same `_get_signature_keys` hook diffusers exposes."""

    def __init__(self, vae: _Vae, text_encoder: _TextEncoder, unet: _Unet):
        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet

    @classmethod
    def _get_signature_keys(cls, _obj: object) -> tuple:
        return {"vae", "text_encoder", "unet"}, set()


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


class _Defaults(GenerationDefaults):
    pass


def _components() -> dict:
    return {"vae": "weights", "text_encoder": "weights", "unet": "weights"}


# ── 1. the declaration reaches the manifest, ordered, per component ──────────


def test_a_declared_slot_publishes_its_demand_per_component_in_order() -> None:
    slot = Slot(
        FakePipeline,
        selected_by="model",
        layouts={
            "*": (CONTRACT_PLAIN_BF16,),
            "text_encoder": (CONTRACT_HF_FP8_BLOCKWISE, CONTRACT_PLAIN_BF16),
        },
    )
    block = _slot_to_manifest(
        "pipeline", slot, family="", components=_components())
    assert block["layouts"] == {
        "*": ["plain.bf16@1"],
        "text_encoder": ["hf.fp8-blockwise@1", "plain.bf16@1"],
    }
    # Order IS preference (§1.33 pt 2) — the fp8 encoder is declared FIRST and
    # must not be sorted into second place by anything on the way out.
    assert block["layouts"]["text_encoder"][0] == "hf.fp8-blockwise@1"


def test_an_undeclared_slot_emits_no_layouts_key_at_all() -> None:
    block = _slot_to_manifest(
        "pipeline", Slot(FakePipeline), family="", components=_components())
    assert "layouts" not in block, (
        "UNDECLARED must be ABSENT. An empty mapping on the wire is a "
        "declaration of nothing, and the hub cannot tell it from a slot whose "
        "author has not spoken."
    )


# ── 2. refused where written ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "layouts, fragment",
    [
        ({}, "declares nothing"),
        ({"*": ()}, "is empty"),
        ({"*": "plain.bf16@1"}, "ORDERED tuple"),
        ({"*": ("plain.bf16",)}, "not a contract handle"),
        ({"*": ("cozy.not-registered@1",)}, "is not registered"),
        ({"*": ("plain.bf16@1", "plain.bf16@1")}, "repeats"),
        ({"": ("plain.bf16@1",)}, "must be a non-empty component path"),
        (
            {"*": ("diffusers.multifile@1+plain.bf16@1",)},
            "topology axis has no registry yet",
        ),
    ],
)
def test_an_invalid_declaration_is_refused_at_the_declaration_site(
    layouts: object, fragment: str,
) -> None:
    with pytest.raises(LayoutDeclarationError) as excinfo:
        Slot(FakePipeline, layouts=layouts)  # type: ignore[arg-type]
    assert fragment in str(excinfo.value)


def test_a_component_key_matching_nothing_is_a_decoration_error() -> None:
    """The key vocabulary is the DERIVED tree. A key that matches no component
    reads as a declaration and gates nothing, which is worse than silence."""

    @endpoint(models={
        "pipeline": Slot(
            FakePipeline,
            layouts={"txt_encoder": (CONTRACT_PLAIN_BF16,)},
        ),
    })
    class Bad:
        def setup(self, pipeline: FakePipeline) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext[_Defaults], p: _In) -> _Out:
            return _Out()

    with pytest.raises(ValueError) as excinfo:
        extract_specs(Bad)
    message = str(excinfo.value)
    assert "txt_encoder" in message
    # The refusal names the tree, because the author's next action is to pick
    # a real path out of it.
    assert "text_encoder" in message and "vae" in message


def test_a_real_component_key_is_accepted_at_decoration() -> None:
    @endpoint(models={
        "pipeline": Slot(
            FakePipeline,
            layouts={
                "*": (CONTRACT_PLAIN_BF16,),
                "text_encoder": (CONTRACT_HF_FP8_BLOCKWISE,),
            },
        ),
    })
    class Good:
        def setup(self, pipeline: FakePipeline) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext[_Defaults], p: _In) -> _Out:
            return _Out()

    specs = extract_specs(Good)
    slot = specs[0].slots["pipeline"]
    assert slot.layouts == {
        "*": ("plain.bf16@1",),
        "text_encoder": ("hf.fp8-blockwise@1",),
    }


# ── 3. the lint refuses what the AST sweep cannot read ───────────────────────


def _run_lint(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_layout_declarations.py"),
         str(target)],
        capture_output=True, text=True, timeout=120,
    )


def test_the_lint_refuses_a_computed_declaration(tmp_path: Path) -> None:
    module = tmp_path / "computed.py"
    module.write_text(textwrap.dedent(
        """
        from gen_worker import Slot

        HANDLES = ["plain.bf16@1"]

        def build():
            return Slot(object, layouts={"*": tuple(h for h in HANDLES)})
        """
    ), encoding="utf-8")
    result = _run_lint(module)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "layouts['*']" in result.stderr


def test_the_lint_accepts_literals_and_vocabulary_constants(
    tmp_path: Path,
) -> None:
    module = tmp_path / "literal.py"
    module.write_text(textwrap.dedent(
        """
        from gen_worker import Slot
        from gen_worker.models.tensor_layout_contract import CONTRACT_PLAIN_BF16

        SLOT = Slot(object, layouts={
            "*": (CONTRACT_PLAIN_BF16,),
            "text_encoder": ("hf.fp8-blockwise@1", CONTRACT_PLAIN_BF16),
        })
        """
    ), encoding="utf-8")
    result = _run_lint(module)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_shipped_tree_passes_the_lint() -> None:
    result = _run_lint(REPO / "src")
    assert result.returncode == 0, result.stderr


# ── 4. the census is a LOWER bound: it reports, it does not refuse ───────────


def test_a_declared_handle_no_decoder_backs_is_reported_not_refused() -> None:
    derived = DerivedExecutionLanes(
        derivation="gen_worker.discovery.execution_lanes@1",
        execution_lanes=("bf16-w16a16+eager",),
        contracts=(DerivedContract(
            contract=CONTRACT_PLAIN_BF16,
            decoder="gen_worker.models.loading:load_from_pretrained",
            execution_lanes=("bf16-w16a16+eager",),
            composes_lora=True,
        ),),
        excluded_modules=(),
    )
    fn = {
        "name": "generate",
        "slots": [_slot_to_manifest(
            "pipeline",
            Slot(FakePipeline, layouts={
                "*": (CONTRACT_PLAIN_BF16,),
                "text_encoder": (CONTRACT_HF_FP8_BLOCKWISE,),
            }),
            family="", components=_components(),
        )],
    }
    # `hf.fp8-blockwise@1` is decoded natively by transformers through
    # `quantization_config`, with zero cozy markers. Refusing it would make
    # §1.33's own worked example illegal.
    assert _census_unbacked_layouts(fn, derived) == ["hf.fp8-blockwise@1"]


def test_the_census_is_silent_for_an_undeclared_slot() -> None:
    derived = DerivedExecutionLanes(
        derivation="gen_worker.discovery.execution_lanes@1",
        execution_lanes=(), contracts=(), excluded_modules=(),
    )
    fn = {
        "name": "generate",
        "slots": [_slot_to_manifest(
            "pipeline", Slot(FakePipeline), family="",
            components=_components())],
    }
    assert _census_unbacked_layouts(fn, derived) == []


# ── 5. the wire the hub actually reads ───────────────────────────────────────


def test_the_demand_survives_the_endpoint_lock_toml_round_trip() -> None:
    """endpoint.lock is TOML and the hub parses JSON, so the declaration
    crosses two encoders before any gate sees it. A nested table inside an
    array-of-tables is exactly where a TOML encoder's key ordering can bite —
    and the hub's `manifestSlotLayoutDoc` decodes `map[string][]string`, so a
    shape change here is a silent UNDECLARED on the other side."""
    block = _slot_to_manifest(
        "pipeline",
        Slot(FakePipeline, selected_by="model", layouts={
            "*": (CONTRACT_PLAIN_BF16,),
            "text_encoder": (CONTRACT_HF_FP8_BLOCKWISE, CONTRACT_PLAIN_BF16),
        }),
        family="sdxl", components=_components(),
    )
    doc = {"functions": [{"name": "generate", "slots": [block]}]}
    decoded = msgspec.toml.decode(msgspec.toml.encode(doc))
    assert decoded["functions"][0]["slots"][0]["layouts"] == {
        "*": ["plain.bf16@1"],
        "text_encoder": ["hf.fp8-blockwise@1", "plain.bf16@1"],
    }
