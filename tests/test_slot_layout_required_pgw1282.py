"""pgw#1282 (HARDCUT A19) — a model slot that declares no consumed
tensor-layout contract does not ship.

Master publishes it SILENTLY: `layouts=` was optional, absence was the
UNDECLARED tri-state, and the hub's gate then fell back to the image-wide
decoder census. Measured fleet-wide, absence was never a considered choice —
it was the default. These tests drive the REAL discovery entry point the
Dockerfile runs (`discover_manifest` + `validate_endpoint_lock`), so the
refusal is a property of the built image, not of a unit seam.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from gen_worker.discovery import (
    refuse_undeclared_slot_layouts,
    validate_endpoint_lock,
)
from gen_worker.models.tensor_layout_contract import (
    LayoutDeclarationError,
    LayoutRequirements,
    UndeclaredSlotLayoutError,
    parse_layout_id,
    parse_layout_requirements,
    validate_layout_handle,
)

# The DECLARED demand this fixture states, and the one a hub-side variant of
# the same family derives. `plain.bf16@1` is the handle every dense
# safetensors tree in the fleet derives to (th#1937), which is why the
# fixture's declaration and a live variant's derived contract are comparable
# at all — one vocabulary, two sides.
DECLARED = ("cozy.fp8-rowwise@1", "plain.bf16@1")


def _endpoint_tree(tmp_path: Path, slot_kwargs: str, pkg: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(textwrap.dedent(f"""
        [project]
        name = "{pkg}"

        [tool.gen_worker]
        main = "{pkg}.main"
    """))
    src = tmp_path / pkg
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(textwrap.dedent(f"""
        import msgspec
        from gen_worker import RequestContext, Resources, Slot, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str = ""

        class Pipe:
            pass

        @endpoint(
            models={{"pipeline": Slot(Pipe{slot_kwargs})}},
            resources=Resources(gpu_count=1),
        )
        class Gen:
            def setup(self, pipeline: Pipe) -> None:
                self.pipeline = pipeline

            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_()
    """))
    return tmp_path


_NEXT_PKG = itertools.count()


def _manifest(tmp_path: Path, slot_kwargs: str,
              monkeypatch: pytest.MonkeyPatch) -> dict:
    """One fresh package name per call — the interpreter caches an imported
    endpoint module, so reusing one name would hand the second fixture the
    first fixture's slots and quietly assert nothing."""
    from gen_worker.discovery.discover import discover_manifest

    pkg = f"ep1282_{next(_NEXT_PKG)}"
    root = _endpoint_tree(tmp_path, slot_kwargs, pkg)
    monkeypatch.syspath_prepend(str(root))
    return discover_manifest(root)


def _slot(manifest: dict) -> dict:
    return manifest["functions"][0]["slots"][0]


# ---------------------------------------------------------------------------
# 1. RED on master: an undeclared model slot publishes silently
# ---------------------------------------------------------------------------


def test_an_undeclared_model_slot_refuses_at_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path, "", monkeypatch)

    # The emission is honest — there is no invented default on the manifest.
    assert "layouts" not in _slot(manifest)
    assert "layouts_undeclarable" not in _slot(manifest)

    # And the build refuses it. This is the assertion that is RED on master,
    # where `validate_endpoint_lock` reported ok on exactly this manifest.
    result = validate_endpoint_lock(manifest)
    assert not result.ok
    joined = "\n".join(result.errors)
    assert "'pipeline'" in joined            # the slot, by name
    assert "'generate'" in joined            # the function, by name
    assert 'layouts={"*": ("plain.bf16@1",)}' in joined   # the syntax
    assert "layouts_undeclarable=" in joined              # and the escape

    with pytest.raises(UndeclaredSlotLayoutError) as excinfo:
        refuse_undeclared_slot_layouts(manifest)
    assert "pipeline" in str(excinfo.value)


def test_every_offender_is_named_in_one_run() -> None:
    """A build names every undeclared slot at once — an author fixing them
    one refusal per image build is the reason nobody fixed them."""
    manifest = {"functions": [
        {"name": "generate", "slots": [{"name": "pipeline"},
                                       {"name": "refiner"}]},
        {"name": "edit", "slots": [{"name": "pipeline"}]},
    ]}
    with pytest.raises(UndeclaredSlotLayoutError) as excinfo:
        refuse_undeclared_slot_layouts(manifest)
    message = str(excinfo.value)
    assert message.count("declares no consumed") == 3
    assert "'refiner'" in message
    assert "'edit'" in message


# ---------------------------------------------------------------------------
# 2. a declared contract travels into the manifest VERBATIM
# ---------------------------------------------------------------------------


def test_a_declared_contract_travels_verbatim_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(
        tmp_path,
        ', layouts={"*": ("plain.bf16@1", "cozy.fp8-rowwise@1")}',
        monkeypatch)

    slot = _slot(manifest)
    # CANONICAL order, not as written: the set is a filter and its order
    # carries no preference, so two spellings of one set emit one manifest.
    assert slot["layouts"] == {"*": list(DECLARED)}

    validate_endpoint_lock(manifest)
    assert validate_endpoint_lock(manifest).ok

    # The emitted handles parse under the SHARED vocabulary — the same
    # grammar tensorhub's contractspec speaks, and the same one a live
    # variant's DERIVED contract is expressed in. A handle that only this
    # side could read would make the two sides incomparable.
    for handle in slot["layouts"]["*"]:
        assert validate_layout_handle(handle, where="pgw1282") == handle
        assert parse_layout_id(handle, where="pgw1282").quant == handle

    # And the declared set MEETS a live variant's derived contract: every
    # dense safetensors checkpoint in the catalog derives `plain.bf16@1`
    # (th#1937), so this endpoint can serve at least one variant of its own
    # repo rather than declaring a set nothing satisfies.
    derived_of_a_live_variant = "plain.bf16@1"
    assert derived_of_a_live_variant in slot["layouts"]["*"]


def test_partial_declaration_checks_only_its_declared_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per the tensor-layout ruling: partial declaration = partial checking.

    A slot naming ONE component key states nothing about the others, and
    discovery adds no whole-tree default to cover them. ABSENT is what A19
    refuses; PARTIAL is legal and stays exactly as narrow as it was written.
    """
    manifest = _manifest(
        tmp_path, ', layouts={"*": ("plain.bf16@1",)}', monkeypatch)
    assert _slot(manifest)["layouts"] == {"*": ["plain.bf16@1"]}
    assert validate_endpoint_lock(manifest).ok


# ---------------------------------------------------------------------------
# 3. the explicit third rung — declared UNDECLARABLE, with a reason
# ---------------------------------------------------------------------------


def test_undeclarable_carries_its_reason_onto_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "gguf: the quant axis has no registered handle (th#1809 T3)"
    manifest = _manifest(
        tmp_path, f', layouts_undeclarable="{reason}"', monkeypatch)

    slot = _slot(manifest)
    assert slot["layouts_undeclarable"] == reason
    assert "layouts" not in slot
    assert validate_endpoint_lock(manifest).ok
    refuse_undeclared_slot_layouts(manifest)  # does not raise


def test_the_escape_is_a_declaration_and_not_a_default() -> None:
    from gen_worker import Slot

    class Pipe:
        pass

    # A reason that was WRITTEN and says nothing is the silence A19 removed.
    # `""` is the parameter default — not written at all, and refused by
    # discovery instead, with the syntax in the message.
    for blank in ("   ", "\n", "\t "):
        with pytest.raises(LayoutDeclarationError, match="needs a REASON"):
            Slot(Pipe, layouts_undeclarable=blank)
    assert Slot(Pipe, layouts_undeclarable="").layouts_undeclarable == ""

    # and it cannot be worn alongside a real declaration
    with pytest.raises(LayoutDeclarationError, match="mutually exclusive"):
        Slot(Pipe, layouts={"*": ("plain.bf16@1",)},
             layouts_undeclarable="because")

    assert Slot(Pipe, layouts_undeclarable=" why  ").layouts_undeclarable == "why"


# ---------------------------------------------------------------------------
# 4. the same refusal at PR time — the AST fence an endpoint repo runs
# ---------------------------------------------------------------------------


def _run_lint(target: Path) -> subprocess.CompletedProcess:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "lint_layout_declarations.py"),
         str(target)],
        capture_output=True, text=True, timeout=120,
    )


def test_the_fence_refuses_an_undeclared_slot_in_source(tmp_path: Path) -> None:
    """Discovery refuses at image build; this refuses in the diff, which is
    where a layout demand is actually judged. An endpoint repo runs the same
    script against its own tree."""
    module = tmp_path / "bare.py"
    module.write_text('x = Slot(Pipe, selected_by="model")\n', encoding="utf-8")
    result = _run_lint(module)
    assert result.returncode == 1
    assert "declares no consumed tensor-layout contract" in result.stderr
    assert 'layouts={"*": ("plain.bf16@1",)}' in result.stderr


def test_the_fence_wants_a_literal_reason_on_the_escape(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    empty.write_text('x = Slot(Pipe, layouts_undeclarable="")\n', encoding="utf-8")
    assert _run_lint(empty).returncode == 1

    computed = tmp_path / "computed.py"
    computed.write_text(
        'x = Slot(Pipe, layouts_undeclarable=REASON)\n', encoding="utf-8")
    assert _run_lint(computed).returncode == 1

    good = tmp_path / "good.py"
    good.write_text(
        'x = Slot(Pipe, layouts_undeclarable="gguf: the quant axis has no '
        'registered handle (th#1809 T3)")\n', encoding="utf-8")
    assert _run_lint(good).returncode == 0, _run_lint(good).stderr


# ---------------------------------------------------------------------------
# 5. the REQUIREMENTS axis (Paul, 2026-08-15)
# ---------------------------------------------------------------------------


def test_the_two_forms_of_a_requirement_are_one_declaration() -> None:
    """Dual form, per the standing input ruling: the compact string an author
    writes and the structured object are the same value, and `render()` puts
    the compact form back so a round trip is stable."""
    compact = parse_layout_requirements("sm100+", where="t")
    structured = parse_layout_requirements(
        LayoutRequirements(minimum="sm100+"), where="t")
    mapping = parse_layout_requirements({"min_sm": 100}, where="t")
    assert compact == structured == mapping
    assert compact.render() == "sm100+"
    assert parse_layout_requirements(compact.render(), where="t") == compact


def test_an_unbuilt_requirement_term_is_refused_not_ignored() -> None:
    """`kernels` is named in the ruling and NOT built — there is no runtime
    kernel-capability probe in this worker, so it would be a floor with no
    fact behind it. An ignored requirement is one that silently does not
    hold, so the grammar refuses it by name and stays extensible."""
    for term in ("torch>=2.13", "kernels", "sm100", "sm_100+", ""):
        with pytest.raises(LayoutDeclarationError):
            parse_layout_requirements(term, where="t")
    with pytest.raises(LayoutDeclarationError, match="not.*built"):
        parse_layout_requirements({"kernels": ["sa2"]}, where="t")
    # and there is no "no floor" value — absence is the undeclared axis
    for bad in (0, -1, True, "  "):
        with pytest.raises(LayoutDeclarationError):
            parse_layout_requirements({"min_sm": bad}, where="t")


def test_a_requirement_guarding_nothing_is_refused() -> None:
    from gen_worker import Slot

    class Pipe:
        pass

    with pytest.raises(LayoutDeclarationError, match="does not accept"):
        Slot(Pipe, layouts={"*": ("plain.bf16@1",)},
             layout_requirements={"cozy.svdq-nvfp4-lr8@1": "sm100+"})
    with pytest.raises(LayoutDeclarationError, match="without layouts"):
        Slot(Pipe, layout_requirements={"plain.bf16@1": "sm90+"})
    with pytest.raises(LayoutDeclarationError, match="declares nothing"):
        Slot(Pipe, layouts={"*": ("plain.bf16@1",)}, layout_requirements={})


def test_a_declared_requirement_travels_and_an_undeclared_one_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial checking, on this axis too: the guarded handle carries its
    floor onto the manifest and the unguarded one carries NO key — never a
    zero, which `contractspec.DecodeEntry.MinSM` reads as 'no floor'."""
    manifest = _manifest(
        tmp_path,
        ', layouts={"*": ("plain.bf16@1", "cozy.svdq-nvfp4-lr8@1")}'
        ', layout_requirements={"cozy.svdq-nvfp4-lr8@1": "sm100+"}',
        monkeypatch)

    slot = _slot(manifest)
    assert slot["layout_requirements"] == {
        "cozy.svdq-nvfp4-lr8@1": {"min_sm": 100}}
    assert "plain.bf16@1" not in slot["layout_requirements"]
    assert validate_endpoint_lock(manifest).ok


def test_a_slot_with_no_requirements_emits_no_requirements_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(
        tmp_path, ', layouts={"*": ("plain.bf16@1",)}', monkeypatch)
    assert "layout_requirements" not in _slot(manifest)
