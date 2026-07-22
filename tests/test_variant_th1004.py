"""th#1004: ``@variant_of`` — function-level variant annotation round-trips
through a real discovery walk into the manifest (``variant_of`` /
``variant``), with build-time validation of dangling/self/chained targets.
Real discovery/registry code, no mocking (same idiom as test_p5).
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


def test_method_variant_round_trips_into_manifest(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_v_method"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, variant_of

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="base")

            @variant_of("generate")
            def generate_turbo(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="turbo")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_v_method.main")}
    assert set(fns) == {"generate", "generate-turbo"}
    assert "variant_of" not in fns["generate"]
    assert fns["generate-turbo"]["variant_of"] == "generate"
    assert fns["generate-turbo"]["variant"] == "turbo"


def test_function_shaped_variant_and_custom_kind(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_v_fn"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, variant_of

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        def render(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="base")

        @endpoint
        @variant_of("render", kind="draft")
        def render_draft(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="draft")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_v_fn.main")}
    assert fns["render-draft"]["variant_of"] == "render"
    assert fns["render-draft"]["variant"] == "draft"


def test_dangling_variant_target_fails_discovery(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_v_dangling"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, variant_of

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class Gen:
            @variant_of("nonexistent")
            def generate_turbo(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="turbo")
    """)

    with pytest.raises(ValueError, match="unknown function 'nonexistent'"):
        discover_functions(tmp_pkg, main_module="ep_v_dangling.main")


def test_chained_variant_fails_discovery(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_v_chain"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, variant_of

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="base")

            @variant_of("generate")
            def generate_turbo(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="turbo")

            @variant_of("generate_turbo")
            def generate_hyper(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="hyper")
    """)

    with pytest.raises(ValueError, match="itself a variant"):
        discover_functions(tmp_pkg, main_module="ep_v_chain.main")


# Module scope: typing.get_type_hints resolves handler annotations from the
# declaring module's globals, so spec-time structs cannot be test-local.
import msgspec

from gen_worker import RequestContext, endpoint, variant_of


class SelfIn(msgspec.Struct):
    prompt: str = ""


class SelfOut(msgspec.Struct):
    y: str


def test_self_variant_fails_at_spec_time() -> None:
    from gen_worker.registry import extract_specs

    @endpoint
    class Gen:
        @variant_of("generate_turbo")
        def generate_turbo(self, ctx: RequestContext, data: SelfIn) -> SelfOut:
            return SelfOut(y="turbo")

    with pytest.raises(ValueError, match="cannot target itself"):
        extract_specs(Gen)


def test_variant_decl_rejects_empty_target() -> None:
    from gen_worker import variant_of

    with pytest.raises(ValueError, match="target function name"):
        variant_of("")
    with pytest.raises(ValueError, match="non-empty kind"):
        variant_of("generate", kind="  ")
