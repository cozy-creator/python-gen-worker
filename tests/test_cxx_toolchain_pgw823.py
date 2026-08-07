"""pgw#823 — AOTInductor needs a C++ compiler; the endpoint images have none.

Measured on a real L4 (gen-worker 0.84.0, sdxl 0.2.102, release `39ac3726`,
pod `d0l6455n9nifo3`): the mint loaded the pipeline, exported the
adapter-bearing graph class, reached the linker and refused —

    entry 'unet/adapter=true,cfg=true/B=2,H_lat=80,T_txt=77,W_lat=192':
    aot_compile failed: InductorError: InvalidCxxCompiler: No working C++
    compiler found in torch._inductor.config.cpp.cxx: (None, 'g++')

    aot_mint_phases: status=refused total_s=336.58 — no cell produced

336 seconds of L4 time to learn something `shutil.which` answers instantly.
The pre-flight guard that exists for exactly this (`toolchain_present`, gated
in `mint_child`) PASSED, because it is `any()` over a mixed C/C++ list and the
image does carry a C compiler.

The fix is a SEPARATE predicate, not a tightening: on CUDA the dynamo lane
emits Triton kernels behind a Python wrapper and compiles with no C++
compiler at all (leg 2's 24-47 minute mints are the proof), so tightening
`toolchain_present` would refuse the only mint lane the fleet has working.
"""

from __future__ import annotations

import types

import pytest

from gen_worker import compile_cache as cc
from gen_worker import fleet_cells


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def test_a_C_only_image_passes_the_dynamo_guard_and_fails_the_AOT_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured pod, exactly: `cc`/`gcc` present, `g++` absent."""
    present = {"cc": "/usr/bin/cc", "gcc": "/usr/bin/gcc"}
    monkeypatch.setattr(cc.shutil, "which", lambda n: present.get(n))
    monkeypatch.delenv("CXX", raising=False)
    assert cc.toolchain_present() is True        # the dynamo lane still mints
    assert cc.cxx_toolchain_present() is False   # the AOT lane cannot link


def test_a_cxx_bearing_image_satisfies_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    present = {"cc": "/usr/bin/cc", "gcc": "/usr/bin/gcc", "g++": "/usr/bin/g++"}
    monkeypatch.setattr(cc.shutil, "which", lambda n: present.get(n))
    monkeypatch.delenv("CXX", raising=False)
    assert cc.toolchain_present() is True
    assert cc.cxx_toolchain_present() is True
    assert cc.cxx_compiler() == "/usr/bin/g++"


def test_CXX_wins_because_inductor_honours_it_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cc.shutil, "which",
        lambda n: "/opt/toolchain/clang++" if n == "myclang++" else None)
    monkeypatch.setenv("CXX", "myclang++")
    # Force the PATH branch: torch's config is the first source, and a real
    # torch here would name its own candidates.
    monkeypatch.setattr(cc, "_CXX_CANDIDATES", ("g++",))
    import sys
    monkeypatch.setitem(sys.modules, "torch._inductor", None)
    assert cc.cxx_compiler() == "/opt/toolchain/clang++"


def test_the_predicate_does_not_contradict_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When torch names candidates and NONE resolve, that IS the
    InvalidCxxCompiler the linker raises — the guard must agree with it and
    not fall through to a broader PATH guess."""
    import sys

    cfg = types.SimpleNamespace(cpp=types.SimpleNamespace(cxx=(None, "g++")))
    monkeypatch.setitem(
        sys.modules, "torch._inductor",
        types.SimpleNamespace(config=cfg))
    # `c++` exists on PATH but torch will not use it — it asked for g++.
    monkeypatch.setattr(
        cc.shutil, "which", lambda n: "/usr/bin/c++" if n == "c++" else None)
    assert cc.cxx_toolchain_present() is False


# ---------------------------------------------------------------------------
# The refusal — pgw#996 moved it to the BUILD
#
# pgw#823 asked this question on the pod, where the only available answer was
# a silent downgrade: decline the AOT recipe, serve eager forever, bill the
# fleet. The toolchain is a property of the IMAGE, so the question now belongs
# to the image build (`aot_preconditions` / `validate_endpoint_lock`, proven in
# test_mint_preconditions_pgw996.py) and a g++-less image that declares an
# export cannot be published at all. What survives on the mint path is the
# CHILD's typed refusal, below: loud and terminal, never a downgrade.
# ---------------------------------------------------------------------------


def test_the_parent_no_longer_second_guesses_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 336 s is bought back by the build gate, not by a pod-side branch.
    A pod that reaches this code is running an image whose build PROVED the
    toolchain, so the parent asks nothing and mints AOT."""
    monkeypatch.setattr(fleet_cells.cc, "cxx_toolchain_present", lambda: False)
    from gen_worker import aot_mint

    monkeypatch.setattr(aot_mint, "lifted_torch_gap", lambda *a, **k: "")
    monkeypatch.setattr(
        aot_mint, "declaration_module_gaps", lambda *a, **k: [])
    monkeypatch.setattr(
        fleet_cells, "aot_export_spec", lambda *a, **k: object())

    import gen_worker.api.export_contract as ec
    from gen_worker.api.decorators import Compile
    from gen_worker.api.export_contract import Dim, GraphClass, Input

    decl = Compile(
        family="sdxl", targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", 4)),),
        shape_strategy="static-rows", warm_changes_key=False)
    # Patch the caller's binding too: fleet_cells imports the name at module
    # scope (pgw#976), so patching only export_contract leaves the real one bound.
    monkeypatch.setattr(ec, "export_declaration", lambda _f: decl)
    monkeypatch.setattr(fleet_cells, "export_declaration", lambda _f: decl)

    events: list = []
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, **kw: events.append((kind, detail, kw)))

    cfg = types.SimpleNamespace(
        family="sdxl", lora_bucket=64, shapes=(), text_lens=(),
        guidance_scales=(), targets=("unet",), regional=False)
    recipe = fleet_cells.mint_recipe(object(), cfg, delegate=True)

    assert recipe == fleet_cells.RECIPE_AOT
    assert [e for e in events if e[2].get("phase") == "no_cxx_toolchain"] == []


def test_the_child_refuses_the_AOT_recipe_before_reading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child's own belt-and-braces: the parent may be an older SDK."""
    from gen_worker import mint_child

    monkeypatch.setattr(mint_child, "assert_composable", lambda *a, **k: None)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(cc, "cxx_toolchain_present", lambda: False)

    req = types.SimpleNamespace(
        slots={},
        target="/tmp/x", work_root="/tmp/y", device=0, vram_cap_bytes=0,
        modules=[], function="generate", cfg=None, configs={}, execution_lane="",
        report="/tmp/r", cell_key="")
    with pytest.raises(mint_child.MintChildRefused, match="no C\\+\\+ compiler"):
        mint_child.mint(req)


# pgw#1010: `test_the_dynamo_recipe_is_NOT_refused_by_the_cxx_gate` stood here.
# The child mints ONE artifact kind now, and AOTInductor links a shared object,
# so the C++ compiler is an unconditional precondition rather than a per-recipe
# one. The finding it recorded (leg 2's dynamo mints ran 24-47 minutes on an
# image with no C++ compiler) is exactly why the gate fires before a weight is
# read, which the test above still asserts.
