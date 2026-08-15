"""pgw#996 — the AOT mint's STATIC preconditions are refused at IMAGE BUILD.

Two sides, both proven here:

* the BUILD side — a real ``discover_manifest`` walk over a toy endpoint tree
  that declares an export, run through the real ``validate_endpoint_lock``.
  An image missing a static precondition FAILS, naming it.
* the MINT side — the same condition is no longer reachable as a mint-time
  fallback. ``mint_recipe`` on a toolchain-less, floor-violating environment
  returns the AOT recipe and declines nothing, because the only image that can
  reach a pod already proved both.

No mocks of the thing under test: the discovery walk, the declaration
registry, the validation gate and ``mint_recipe`` are the shipped ones.
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import pytest

from gen_worker import child_preflight
from gen_worker import aot_preconditions as pre
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells
from gen_worker.api import export_contract as ec
from gen_worker.discovery.discover import discover_manifest
from gen_worker.discovery.validation import validate_endpoint_lock


# ---------------------------------------------------------------------------
# A toy endpoint tree that DECLARES an AOT export
# ---------------------------------------------------------------------------

_DECLARATION = """
    from gen_worker import (
        Compile, Dim, GraphClass, Input, register_export_declaration,
    )

    FAMILY = "ep996"

    DECL = register_export_declaration(Compile(
        family=FAMILY, targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
        shape_strategy="static-rows", warm_changes_key=False))
"""

_MAIN = """
    import msgspec
    from gen_worker import (
        Compile, RequestContext, Resources, Slot, endpoint,
        import_export_declaration,
    )

    import_export_declaration("{decl_module}")

    class In_(msgspec.Struct):
        prompt: str = ""
        model: str = ""

    class Out_(msgspec.Struct):
        y: str = ""

    class Pipe:
        @classmethod
        def lora_state_dict(cls, *a, **k):
            return {{}}

    @endpoint(
        models={{"pipeline": Slot(Pipe, selected_by="model")}},
        resources=Resources(gpu=True),
        compile=Compile(family="ep996", shapes=((1024, 1024),), text_len=77),
        lora_bucket={bucket},
    )
    class Gen:
        def setup(self, pipeline: Pipe) -> None: ...

        def generate(self, ctx: RequestContext, data: In_) -> Out_:
            return Out_()
"""


def _tree(root: Path, pkg: str, declaration: str, *, bucket: int = 0,
          external: bool = False) -> Path:
    (root / "pyproject.toml").write_text(textwrap.dedent(f"""
        [project]
        name = "{pkg}"

        [tool.gen_worker]
        main = "{pkg}.main"
    """))
    src = root / pkg
    src.mkdir()
    (src / "__init__.py").write_text("")
    # `external` puts the declaration OUTSIDE the walked package, where the
    # discovery walk never imports it and `import_export_declaration`'s
    # swallow is the only thing standing between a broken declaration and a
    # published image.
    decl_module = f"{pkg}_decl" if external else f"{pkg}.aot_declaration"
    target = (root / f"{pkg}_decl.py") if external else (src / "aot_declaration.py")
    target.write_text(textwrap.dedent(declaration))
    (src / "main.py").write_text(
        textwrap.dedent(_MAIN).format(
            pkg=pkg, bucket=bucket, decl_module=decl_module))
    return root


@pytest.fixture()
def clean_registry(monkeypatch: pytest.MonkeyPatch):
    """The declaration registry is process-global; a build gate that inherits
    another test's registrations is testing nothing."""
    ec.reset_export_declarations()
    yield
    ec.reset_export_declarations()


@pytest.fixture()
def toolchain(monkeypatch: pytest.MonkeyPatch):
    """The image's C++ toolchain, under test control."""

    def _set(path: str) -> None:
        monkeypatch.setattr(cc, "cxx_compiler", lambda: path)

    _set("/usr/bin/g++")
    return _set


@pytest.fixture(autouse=True)
def cuda_root(monkeypatch: pytest.MonkeyPatch):
    """The image's CUDA root, under test control (pgw#1017 GAP A).

    Autouse and healthy by default, for the same reason `toolchain` stubs g++:
    these rows are about a hypothetical endpoint IMAGE, and the box running the
    suite is not one. Without it every row here would answer about the CI
    runner — which carries a CUDA torch wheel and no `/usr/local/cuda`, so the
    honest verdict is `refused` and it has nothing to do with the toy endpoint
    under test.

    Returns a setter so a row that wants the refusal can ask for it.
    """
    from gen_worker import cuda_root as cr

    def _set(home: str, gaps: tuple[str, ...] = ()) -> None:
        monkeypatch.setattr(cr, "torch_cuda_home", lambda: home)
        monkeypatch.setattr(cr, "missing_parts", lambda _root: list(gaps))

    _set("/usr/local/cuda")
    return _set


def _build(root: Path, pkg: str, declaration: str, *, bucket: int = 0,
           external: bool = False):
    _tree(root, pkg, declaration, bucket=bucket, external=external)
    manifest = discover_manifest(root)
    return manifest, validate_endpoint_lock(manifest)


# ---------------------------------------------------------------------------
# BUILD side — the refusals
# ---------------------------------------------------------------------------


def test_a_declaring_image_with_no_cxx_compiler_FAILS_THE_BUILD(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_registry, toolchain,
) -> None:
    """pgw#823's 336 s of L4 time, refused for free in the image that caused
    it. The endpoint declares an AOT export and the image cannot link one."""
    monkeypatch.syspath_prepend(str(tmp_path))
    toolchain("")  # no g++ anywhere

    manifest, result = _build(tmp_path, "ep996_nocxx", _DECLARATION)

    assert result.ok is False
    (err,) = [e for e in result.errors if "cxx_toolchain" in e]
    assert "InvalidCxxCompiler" in err
    assert "install g++" in err
    assert "ep996" in err  # names the family that declared the export
    row = next(r for r in manifest["aot_preconditions"]
               if r["check"] == pre.CHECK_CXX_TOOLCHAIN)
    assert row["verdict"] == pre.REFUSED


def test_a_declaration_module_that_cannot_IMPORT_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_registry, toolchain,
) -> None:
    """``import_export_declaration`` swallows so the endpoint still BOOTS.
    On a pod the result is indistinguishable from "declares no export"; at
    build time it is a broken image, and the swallow stops being correct."""
    monkeypatch.syspath_prepend(str(tmp_path))

    manifest, result = _build(
        tmp_path, "ep996_badimport",
        'raise RuntimeError("the pinned lib moved")\n', external=True)

    assert result.ok is False
    (err,) = [e for e in result.errors
              if pre.CHECK_DECLARATION_IMPORT in e]
    assert "the pinned lib moved" in err
    assert "ep996_badimport_decl" in err


def test_a_declaration_below_the_lifted_lora_torch_floor_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_registry, toolchain,
) -> None:
    """The pinned torch wheel is an image property. A bucket-bearing endpoint
    on torch 2.9 cannot trace its own lifted fork — decided at build."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(pre, "_torch_version", lambda: "2.9.0+cu121")

    _manifest, result = _build(
        tmp_path, "ep996_oldtorch", _DECLARATION, bucket=64)

    assert result.ok is False
    (err,) = [e for e in result.errors
              if pre.CHECK_LIFTED_LORA_TORCH_FLOOR in e]
    assert "torch >= 2.13" in err and "2.9.0+cu121" in err


def test_the_torch_floor_is_not_asked_of_a_bucketless_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_registry, toolchain,
) -> None:
    """No lifted fork is declared, so the floor is not this image's problem —
    the gate must not refuse a build for a lane it does not run."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(pre, "_torch_version", lambda: "2.9.0+cu121")

    manifest, result = _build(tmp_path, "ep996_nobucket", _DECLARATION)

    assert result.ok is True, result.errors
    assert not [r for r in manifest["aot_preconditions"]
                if r["check"] == pre.CHECK_LIFTED_LORA_TORCH_FLOOR]


def test_a_healthy_declaring_image_builds_and_records_its_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_registry, toolchain,
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(pre, "_torch_version", lambda: "2.13.0+cu130")
    # a bucket-bearing image also owes an adapter backend, and the
    # CI box has no `peft` — that check has its own suite
    # (`test_adapter_backend_preflight_pgw501`), so state the image's answer
    # here rather than letting the box decide what "healthy" means.
    monkeypatch.setattr(pre, "adapter_backend_present", lambda: True)

    manifest, result = _build(tmp_path, "ep996_ok", _DECLARATION, bucket=64)

    assert result.ok is True, result.errors
    verdicts = {r["check"]: r["verdict"] for r in manifest["aot_preconditions"]}
    assert verdicts == {
        pre.CHECK_DECLARATION_EVALUATES: pre.OK,
        pre.CHECK_CXX_TOOLCHAIN: pre.OK,
        # The two the custom-Dockerfile audit added. An exhaustive
        # equality, deliberately — a row that stops being stamped is the exact
        # defect this whole issue is about, and `in` would not notice.
        pre.CHECK_CUDA_ROOT: pre.OK,
        pre.CHECK_TORCH_SINGLETON: pre.OK,
        pre.CHECK_LIFTED_LORA_TORCH_FLOOR: pre.OK,
        pre.CHECK_ADAPTER_BACKEND: pre.OK,
    }


def test_an_endpoint_that_declares_NO_export_owes_the_AOT_lane_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_registry,
) -> None:
    """JIT is intake mode (ruled). No declaration, no toolchain obligation —
    a g++-less image carrying an undeclared endpoint still builds."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(cc, "cxx_compiler", lambda: "")

    manifest, result = _build(
        tmp_path, "ep996_intake", "# this family declares no export\n")

    assert result.ok is True, result.errors
    assert "aot_preconditions" not in manifest


# ---------------------------------------------------------------------------
# BUILD side — what is NOT a broken build
# ---------------------------------------------------------------------------


# pgw#1107 retired the thunk, and with it the two cases that lived here:
#
# * a declaration that says "blocked" — a THUNK raising `MintRefused` became
#   `Compile(blockers=...)`, and the BLOCKED verdict off that declared form is
#   `test_declared_blockers_pgw1115.py`
#   (`test_the_image_BUILD_says_a_family_is_blocked_without_renting_a_pod`);
# * a declaration that is simply BROKEN — with no factory to evaluate, a
#   declaration module that throws fails at IMPORT, which is
#   `test_a_declaration_module_that_cannot_IMPORT_fails_the_build` above.
#
# `static_mint_preconditions` no longer catches anything around the registry
# read, because the read cannot raise.


def test_a_torchless_discovery_ABSTAINS_out_loud(clean_registry) -> None:
    """A manifest build without torch cannot decide an image's preconditions.
    It says so — an unrecorded abstention is how a gate becomes decorative."""
    ec.register_export_declaration(
        _example_declaration(), family="ep996-abstain")
    rows = pre.static_mint_preconditions(
        {"ep996-abstain": 64}, torch_available=False)

    (row,) = rows
    assert row.verdict == pre.ABSTAINED
    assert "without torch" in row.detail
    lock = {"functions": [], "aot_preconditions": [r.manifest_row() for r in rows]}
    result = validate_endpoint_lock(lock)
    assert result.ok is True
    assert any("abstained" in w for w in result.warnings)


def _example_declaration():
    from gen_worker.api.decorators import Compile
    from gen_worker.api.export_contract import Dim, GraphClass, Input

    return Compile(
        family="ep996-abstain", targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
        shape_strategy="static-rows", warm_changes_key=False)


# ---------------------------------------------------------------------------
# MINT side — the same conditions are no longer reachable as a fallback
# ---------------------------------------------------------------------------


def _mint_recipe_on(monkeypatch: pytest.MonkeyPatch, *, bucket: int) -> tuple:
    """Run the real ``mint_recipe`` with a declaration and a captured event
    stream. Returns (recipe, events)."""
    decl = _example_declaration()
    monkeypatch.setattr(ec, "export_declaration", lambda _f: decl)
    monkeypatch.setattr(fleet_cells, "export_declaration", lambda _f: decl)
    monkeypatch.setattr(fleet_cells, "aot_export_spec", lambda *a, **k: object())
    from gen_worker import aot_mint

    monkeypatch.setattr(aot_mint, "declaration_module_gaps", lambda *a, **k: [])

    events: list = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, **kw: events.append((kind, detail, kw)))

    cfg = types.SimpleNamespace(
        family="ep996-abstain", lora_bucket=bucket, shapes=(), text_lens=(),
        guidance_scales=(), targets=("unet",), regional=False)
    return fleet_cells.mint_recipe(object(), cfg, delegate=True), events


def test_the_mint_no_longer_declines_for_a_missing_cxx_toolchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before pgw#996: this returned RECIPE_DYNAMO under the
    ``no_cxx_toolchain`` phase. The image that could produce it now fails to
    build (see the build-side proof above), so the pod stops second-guessing
    a fact it cannot change."""
    monkeypatch.setattr(cc, "cxx_compiler", lambda: "")
    assert cc.cxx_toolchain_present() is False

    recipe, events = _mint_recipe_on(monkeypatch, bucket=0)

    assert recipe == fleet_cells.RECIPE_AOT
    assert [e for e in events if e[2].get("phase") == "no_cxx_toolchain"] == []


def test_the_mint_no_longer_declines_below_the_lifted_lora_torch_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before pgw#996: ``aot_lifted_torch_gap``. The torch wheel is baked
    into the image the build gate cleared."""
    monkeypatch.setattr(pre, "torch_version_gap", lambda _v: "torch is ancient")

    recipe, events = _mint_recipe_on(monkeypatch, bucket=64)

    assert recipe == fleet_cells.RECIPE_AOT
    assert [e for e in events
            if e[2].get("phase") == "aot_lifted_torch_gap"] == []


def test_the_retired_phases_are_gone_from_the_mint_path() -> None:
    """The tokens themselves: a decline phase that still exists in the source
    is a decline a fleet can still be told, whatever the tests say."""
    source = Path(fleet_cells.__file__).read_text()
    assert "no_cxx_toolchain" not in source
    assert "aot_lifted_torch_gap" not in source
    assert "cxx_toolchain_present" not in source


def test_the_CHILD_still_refuses_a_toolchainless_AOT_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What is deleted is the parent's silent DOWNGRADE, not the child's typed
    refusal. A refusal is loud and terminal; a downgrade bills the fleet for
    eager serving and says nothing."""
    from gen_worker import mint_child

    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(cc, "cxx_toolchain_present", lambda: False)

    req = types.SimpleNamespace(
        slots={}, recipe=mint_child.RECIPE_AOT,
        target="/tmp/x", capture="/tmp/y", device=0,
        modules=[], function="generate", cfg=None, configs={},
        execution_lane="", report="/tmp/r", arm_token="")
    with pytest.raises(child_preflight.PreflightRefused, match="no C\\+\\+ compiler"):
        mint_child.mint(req)


def test_ONE_floor_AND_THE_MINT_NO_LONGER_SPELLS_IT(clean_registry) -> None:
    """pgw#914 RED: two spellings of a precondition is how a build proves one
    thing and a pod discovers another, so the mint keeps NONE. The floor is an
    image fact, decided once by the build gate that ships `endpoint.lock`; a
    per-mint re-decision could only ever disagree with it, and the wrapper's
    stated reason to exist (the mint-request CLI) died with the forge
    (DESIGN-RULINGS §4.28)."""
    from gen_worker import aot_mint

    for retired in (
        "LIFTED_LORA_TORCH_FLOOR",
        "lifted_torch_gap",
        "torch_version_gap",
    ):
        assert not hasattr(aot_mint, retired)
    source = Path(aot_mint.__file__).read_text()
    assert "LIFTED_LORA_TORCH_FLOOR" not in source
    assert "__version__" not in source

    # ...and the ONE surviving spelling still refuses below the floor.
    ec.register_export_declaration(_example_declaration())
    rows = pre.static_mint_preconditions(
        {"ep996-abstain": 64}, torch_available=True,
        torch_version="2.9.1+cu126")
    floor = [r for r in rows
             if r.check == pre.CHECK_LIFTED_LORA_TORCH_FLOOR]
    assert floor and floor[0].verdict == pre.REFUSED
    assert "2.13" in floor[0].detail and "bind_views" in floor[0].detail
