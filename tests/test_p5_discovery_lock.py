"""P5 (th#960/pgw#609 design table): endpoint.lock contract — a real
``discover`` walk over a toy on-disk package, matching SDK declarations:
slots, kinds, reserved-method/duplicate-slug rejection, payload Meta bounds,
accelerator + cuda-floor fields present (producer half of th#904's 422 gate
— pgw emits ``resources.gpu``/``resources.compute_capability``, tensorhub's
T8 validates them). Real discovery/registry code, no mocking.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _write(pkg: Path, main_src: str) -> None:
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent(main_src))


def test_slot_endpoint_emits_slots_block(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_p5_slot"
    _write(pkg, """
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
            },
            resources=Resources(vram_gb=12, compute_capability=8.9),
        )
        class Gen:
            def setup(self, pipeline: object) -> None: ...
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """)

    (fn,) = discover_functions(tmp_pkg, main_module="ep_p5_slot.main")
    assert fn["kind"] == "inference"
    (slot,) = fn["slots"]
    assert slot["name"] == "pipeline"
    assert slot["selected_by"] == "model"
    assert slot["family"] == "sdxl"
    # Accelerator + cuda-floor fields (th#904 producer half): gpu implied by
    # vram_gb/compute_capability; compute_capability IS the cuda-floor.
    assert fn["resources"]["gpu"] is True
    assert fn["resources"]["compute_capability"] == pytest.approx(8.9)
    assert fn["resources"]["vram_gb"] == pytest.approx(12.0)


def test_cpu_endpoint_never_carries_gpu_or_cuda_floor(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_p5_cpu"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            x: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """)

    (fn,) = discover_functions(tmp_pkg, main_module="ep_p5_cpu.main")
    assert fn["kind"] == "inference"
    assert fn["resources"].get("gpu") in (None, False)
    assert "compute_capability" not in fn["resources"]


def test_duplicate_wire_route_within_a_class_fails_validation() -> None:
    """Two class-shape entries slugifying to the SAME wire route on ONE
    class would silently shadow one of them at dispatch — the bake-time
    validator (#328) must reject it. (The discover WALK itself hard-fails
    even earlier via ``_assert_unique_function_names`` for this exact case
    — ``discover_functions`` raises ValueError before ``validate_endpoint_lock``
    would ever see it. This exercises validate_endpoint_lock's OWN guard
    directly, the surface a hand-assembled or legacy lock dict would hit.)"""
    from gen_worker.discovery.validation import validate_endpoint_lock

    functions = [
        {"name": "generate-now", "class_name": "Gen", "python_name": "generate_now", "kind": "inference"},
        {"name": "generate-now", "class_name": "Gen", "python_name": "generate__now", "kind": "inference"},
    ]
    result = validate_endpoint_lock({"functions": functions})
    assert not result.ok
    assert any("slugify" in e for e in result.errors)


def test_discover_walk_hard_fails_on_duplicate_wire_route(tmp_pkg: Path) -> None:
    """The discovery walk itself refuses the exact same shape even earlier
    (``_assert_unique_function_names``) — belt and suspenders with the
    validator above."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_p5_dup"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            x: str = ""

        class OutA(msgspec.Struct):
            y: str

        class OutB(msgspec.Struct):
            z: str

        @endpoint
        class Gen:
            def generate_now(self, ctx: RequestContext, data: In_) -> OutA:
                return OutA(y="a")

            def generate__now(self, ctx: RequestContext, data: In_) -> OutB:
                return OutB(z="b")
    """)
    with pytest.raises(ValueError, match="defined 2x"):
        discover_functions(tmp_pkg, main_module="ep_p5_dup.main")


def test_missing_kind_fails_validation(tmp_pkg: Path) -> None:
    from gen_worker.discovery.validation import validate_endpoint_lock

    result = validate_endpoint_lock({"functions": [
        {"name": "generate", "class_name": "Gen", "python_name": "generate", "kind": ""},
    ]})
    assert not result.ok
    assert any("kind must be one of" in e for e in result.errors)


def test_payload_meta_bounds_compile_into_the_input_schema(tmp_pkg: Path) -> None:
    """msgspec.Meta bounds on a payload field survive into the discovered
    JSON schema — the hub validates requests against this schema, so a
    bound that silently vanished at discovery would be a validation hole."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_p5_meta"
    _write(pkg, """
        from typing import Annotated
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            steps: Annotated[int, msgspec.Meta(ge=1, le=150)] = 20

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """)
    (fn,) = discover_functions(tmp_pkg, main_module="ep_p5_meta.main")
    steps_schema = fn["input_schema"]["$defs"]["In_"]["properties"]["steps"]
    assert steps_schema["minimum"] == 1
    assert steps_schema["maximum"] == 150


def test_discovery_stubs_missing_heavy_deps_but_fails_loud_on_touch(tmp_pkg: Path) -> None:
    """Absorbed from test_discovery_heavy_deps.py (pgw#506): a module-top
    ``import torch`` must be free during discovery (build-time walks never
    have torch installed for CPU-only build images) — imports as a stub —
    but any ACTUAL attribute use at module scope fails loud with an
    actionable error, not a silent wrong schema."""
    from gen_worker.discovery.discover import discover_functions

    fake_heavy_dep = "gw_p5_fake_heavy_dep"
    pkg = tmp_pkg / "ep_p5_heavy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(
        f"import {fake_heavy_dep}\nfrom {fake_heavy_dep} import nn\n" + textwrap.dedent("""
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            text: str = ""

        class Out_(msgspec.Struct):
            reply: str

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(reply=data.text)
    """))

    fns = discover_functions(
        tmp_pkg, main_module="ep_p5_heavy.main", extra_heavy_deps=(fake_heavy_dep,),
    )
    assert [f["name"] for f in fns] == ["generate"]
    assert fns[0]["input_schema"]  # schemas still build against the stub

    # A DIFFERENT package (import caching would otherwise mask a re-walk of
    # the same module name): module-scope USE of the missing dep fails loud.
    pkg2 = tmp_pkg / "ep_p5_heavy_use"
    pkg2.mkdir()
    (pkg2 / "__init__.py").write_text("")
    (pkg2 / "main.py").write_text(
        f"import {fake_heavy_dep}\nDTYPE = {fake_heavy_dep}.bfloat16\n" + textwrap.dedent("""
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            text: str = ""

        class Out_(msgspec.Struct):
            reply: str

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(reply=data.text)
    """))
    with pytest.raises(ValueError, match=f"{fake_heavy_dep}.bfloat16"):
        discover_functions(
            tmp_pkg, main_module="ep_p5_heavy_use.main", extra_heavy_deps=(fake_heavy_dep,),
        )
