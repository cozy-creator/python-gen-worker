"""@inference.stage decorator hardening tests (#325).

Covers the SerialWorker 3D pattern — TRELLIS-style pipelines where one
class declares multiple ``@inference.stage`` methods that the wire-route
``@inference.function`` chains together.

Today every stage is invoked in-process as a normal Python method call.
The decorator + manifest plumbing is forward-compatible with future
disaggregated inference (gen-orchestrator routes per-stage to remote
workers on the requested ``gpu_class``).

Scope:

  1. Decoration discovery: multiple stages on a class land on
     ``__gen_worker_stage_methods__`` with their specs intact.
  2. Stage ``name`` is slug-normalized at decoration; an unslugifiable
     name (e.g. ``'!!!'``) raises ``ValueError``.
  3. ``gpu_class`` is validated against the ``Literal['small','large']``
     Domain — msgspec doesn't enforce Literal at construction, so the
     decorator does it explicitly.
  4. Duplicate stage names within a class are rejected at class-decoration
     time (would clash in the manifest's ``stages`` list).
  5. Stages remain ordinary callable methods today (in-process invocation
     for any future cross-process routing must preserve this contract).
  6. ``@inference.function`` + ``@inference.stage`` mix on the same class.
  7. Manifest plumbing: the per-function manifest entry carries a
     ``stages`` list with ``{python_name, name, gpu_class}`` for every
     stage on the class — required for future orchestrator-side remote
     stage routing.
"""

from __future__ import annotations

from typing import Iterator

import msgspec
import pytest

from gen_worker import RequestContext, inference
from gen_worker.api.decorators import _StageSpec


# -------- Shared payload / output structs ---------------------------------


class Image(msgspec.Struct):
    url: str


class Embeds(msgspec.Struct):
    text: str = "embeds"


class Structure(msgspec.Struct):
    data: str = "structure"


class Textured(msgspec.Struct):
    data: str = "textured"


class Mesh(msgspec.Struct):
    glb_url: str


# -------- Test class factories --------------------------------------------


def _make_trellis_class():
    """The canonical SerialWorker 3D pattern from issue #325."""

    @inference()
    class Trellis2Generate:
        def setup(self):
            self.ready = True

        @inference.stage(name="encode", gpu_class="small")
        def encode(self, image_url: str) -> Embeds:
            return Embeds(text=f"embeds:{image_url}")

        @inference.stage(name="structure", gpu_class="large")
        def make_structure(self, embeds: Embeds) -> Structure:
            return Structure(data=f"struct:{embeds.text}")

        @inference.stage(name="texture", gpu_class="large")
        def make_texture(self, structure: Structure) -> Textured:
            return Textured(data=f"tex:{structure.data}")

        @inference.stage(name="mesh_extract", gpu_class="small")
        def extract_mesh(self, textured: Textured) -> Mesh:
            return Mesh(glb_url=f"glb://{textured.data}")

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            embeds = self.encode(payload.url)
            structure = self.make_structure(embeds)
            textured = self.make_texture(structure)
            return self.extract_mesh(textured)

        def shutdown(self):
            pass

    return Trellis2Generate


# -------- 1. Discovery: stages on __gen_worker_stage_methods__ ------------


def test_multiple_stages_discovered_on_class() -> None:
    cls = _make_trellis_class()
    stages = getattr(cls, "__gen_worker_stage_methods__")
    assert isinstance(stages, list)
    assert len(stages) == 4

    by_name = {spec.name: (attr_name, spec) for attr_name, _m, spec in stages}
    assert set(by_name.keys()) == {"encode", "structure", "texture", "mesh-extract"}

    # The decorator slug-normalizes 'mesh_extract' -> 'mesh-extract'.
    attr_name, spec = by_name["mesh-extract"]
    assert attr_name == "extract_mesh"
    assert isinstance(spec, _StageSpec)
    assert spec.gpu_class == "small"

    # encode -> 'small', structure / texture -> 'large'.
    assert by_name["encode"][1].gpu_class == "small"
    assert by_name["structure"][1].gpu_class == "large"
    assert by_name["texture"][1].gpu_class == "large"


def test_stage_default_gpu_class_is_large() -> None:
    @inference()
    class C:
        def setup(self):
            pass

        @inference.stage(name="solo")
        def my_stage(self) -> None:
            pass

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            return Mesh(glb_url="x")

        def shutdown(self):
            pass

    stages = C.__gen_worker_stage_methods__
    assert len(stages) == 1
    _attr, _method, spec = stages[0]
    assert spec.gpu_class == "large"


def test_stage_without_explicit_name_uses_method_name_slug() -> None:
    @inference()
    class C:
        def setup(self):
            pass

        @inference.stage
        def my_StageMethod(self) -> None:
            pass

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            return Mesh(glb_url="x")

        def shutdown(self):
            pass

    stages = C.__gen_worker_stage_methods__
    assert len(stages) == 1
    _attr, _method, spec = stages[0]
    # slugify_name: 'my_StageMethod' -> 'my-stagemethod'.
    assert spec.name == "my-stagemethod"
    assert spec.gpu_class == "large"


# -------- 2. Stage name slug validation -----------------------------------


def test_stage_name_unslugifiable_raises() -> None:
    """A name that slugifies to empty (e.g. '!!!') is rejected at decoration."""
    with pytest.raises(ValueError, match="empty slug"):
        @inference.stage(name="!!!")
        def bad(self):
            pass


def test_stage_name_with_punctuation_normalized() -> None:
    """Names with underscores, spaces, mixed case are slug-normalized to the
    wire-friendly form (matches @inference.function slug rules)."""

    @inference()
    class C:
        def setup(self):
            pass

        @inference.stage(name="Make Structure")
        def stage1(self):
            pass

        @inference.stage(name="extract_mesh.v2")
        def stage2(self):
            pass

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            return Mesh(glb_url="x")

        def shutdown(self):
            pass

    names = {spec.name for _a, _m, spec in C.__gen_worker_stage_methods__}
    assert names == {"make-structure", "extract-mesh.v2"}


# -------- 3. gpu_class validation ----------------------------------------


def test_stage_invalid_gpu_class_raises() -> None:
    """Literal['small','large'] is NOT enforced by msgspec at construction,
    so the decorator validates explicitly. Typos must fail fast.
    """
    with pytest.raises(ValueError, match="gpu_class"):
        @inference.stage(name="x", gpu_class="medium")  # type: ignore[arg-type]
        def fn(self):
            pass

    with pytest.raises(ValueError, match="gpu_class"):
        @inference.stage(name="x", gpu_class="huge")  # type: ignore[arg-type]
        def fn2(self):
            pass

    with pytest.raises(ValueError, match="gpu_class"):
        @inference.stage(name="x", gpu_class="")  # type: ignore[arg-type]
        def fn3(self):
            pass


def test_stage_gpu_class_small_and_large_both_accepted() -> None:
    """Sanity-check happy path for both valid members."""

    @inference.stage(name="a", gpu_class="small")
    def a(self):
        pass

    @inference.stage(name="b", gpu_class="large")
    def b(self):
        pass

    assert a.__gen_worker_stage_spec__.gpu_class == "small"
    assert b.__gen_worker_stage_spec__.gpu_class == "large"


# -------- 4. Duplicate name rejection ------------------------------------


def test_duplicate_stage_name_within_class_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate stage name"):
        @inference()
        class _Dup:
            def setup(self):
                pass

            @inference.stage(name="encode")
            def encode_a(self):
                pass

            @inference.stage(name="encode")
            def encode_b(self):
                pass

            @inference.function
            def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
                return Mesh(glb_url="x")

            def shutdown(self):
                pass


def test_duplicate_stage_name_after_slug_normalization_rejected() -> None:
    """Two stages with names that slugify to the same value must also fail.
    'encode-x' and 'encode_x' both slugify to 'encode-x'.
    """
    with pytest.raises(ValueError, match="duplicate stage name"):
        @inference()
        class _Dup:
            def setup(self):
                pass

            @inference.stage(name="encode-x")
            def a(self):
                pass

            @inference.stage(name="encode_x")
            def b(self):
                pass

            @inference.function
            def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
                return Mesh(glb_url="x")

            def shutdown(self):
                pass


def test_same_stage_name_across_different_classes_is_fine() -> None:
    """Stage names are scoped to a class — two classes can each have
    'encode' without clashing.
    """

    @inference()
    class A:
        def setup(self):
            pass

        @inference.stage(name="encode")
        def encode(self):
            pass

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            return Mesh(glb_url="a")

        def shutdown(self):
            pass

    @inference()
    class B:
        def setup(self):
            pass

        @inference.stage(name="encode")
        def encode(self):
            pass

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            return Mesh(glb_url="b")

        def shutdown(self):
            pass

    assert {s.name for _a, _m, s in A.__gen_worker_stage_methods__} == {"encode"}
    assert {s.name for _a, _m, s in B.__gen_worker_stage_methods__} == {"encode"}


# -------- 5. Stages are still ordinary methods (in-process) --------------


def test_stage_method_is_invocable_in_process() -> None:
    """Today (no remote dispatch) each stage is a normal Python method.
    The wire-route ``@inference.function`` chains them together; this must
    keep working without any SDK-level magic.
    """
    cls = _make_trellis_class()
    inst = cls()
    inst.setup()

    # Each stage is callable as a plain method.
    embeds = inst.encode("https://example.com/cat.png")
    assert embeds.text == "embeds:https://example.com/cat.png"

    struct = inst.make_structure(embeds)
    assert struct.data.startswith("struct:")

    tex = inst.make_texture(struct)
    assert tex.data.startswith("tex:")

    mesh = inst.extract_mesh(tex)
    assert mesh.glb_url.startswith("glb://")


def test_decorated_stages_preserve_function_identity() -> None:
    """The decorator returns the original function with metadata attached —
    no wrapper. Important for ``inspect.signature`` / type hints / etc.
    """

    @inference.stage(name="encode", gpu_class="small")
    def my_stage(self, x: int) -> int:
        return x + 1

    assert my_stage.__name__ == "my_stage"
    assert hasattr(my_stage, "__gen_worker_stage_spec__")
    spec = my_stage.__gen_worker_stage_spec__
    assert isinstance(spec, _StageSpec)
    assert spec.name == "encode"
    assert spec.gpu_class == "small"


# -------- 6. @inference.function + @inference.stage mix -------------------


def test_function_and_stage_on_same_class_coexist() -> None:
    """The same class can declare both wire-routable functions
    (``@inference.function``) and pipeline stages (``@inference.stage``)
    without interfering with each other.
    """
    cls = _make_trellis_class()
    fns = cls.__gen_worker_function_methods__
    stages = cls.__gen_worker_stage_methods__
    assert len(fns) == 1
    assert len(stages) == 4

    # Different attribute lists — no overlap.
    fn_attrs = {a for a, _m, _s in fns}
    stage_attrs = {a for a, _m, _s in stages}
    assert fn_attrs.isdisjoint(stage_attrs)

    # The function can call its sibling stages end-to-end.
    inst = cls()
    inst.setup()

    class _Ctx:
        pass

    mesh = inst.generate(_Ctx(), Image(url="https://example.com/x.png"))
    assert mesh.glb_url.startswith("glb://")


def test_class_with_only_function_no_stages_works() -> None:
    """A class with NO @inference.stage methods is valid — stages are
    optional. This guards against accidental coupling of the two systems.
    """

    @inference()
    class NoStages:
        def setup(self):
            pass

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            return Mesh(glb_url="x")

        def shutdown(self):
            pass

    assert NoStages.__gen_worker_stage_methods__ == []
    assert len(NoStages.__gen_worker_function_methods__) == 1


# -------- 7. Manifest plumbing -------------------------------------------


def test_manifest_emits_stages_with_name_and_gpu_class(monkeypatch) -> None:
    """The discovery path that builds the per-function manifest entry must
    surface the stage list so future versions of gen-orchestrator can route
    stages remotely. The shape must be:

        functions[i]["stages"] = [
            {"python_name": ..., "name": ..., "gpu_class": "small" | "large"},
            ...
        ]

    Today nothing reads this; tomorrow the orchestrator will. The contract
    is what enables forward-compat without re-baking endpoints.
    """
    from gen_worker.discovery.discover import _extract_class_function_methods

    cls = _make_trellis_class()
    entries = _extract_class_function_methods(cls, module_name="test_module")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["class_name"] == "Trellis2Generate"
    assert entry["archetype"] == "SerialWorker"

    stages = entry.get("stages")
    assert isinstance(stages, list)
    assert len(stages) == 4

    # Each stage entry has the three required fields.
    for s in stages:
        assert set(s.keys()) >= {"python_name", "name", "gpu_class"}
        assert s["gpu_class"] in ("small", "large")

    by_name = {s["name"]: s for s in stages}
    assert set(by_name.keys()) == {"encode", "structure", "texture", "mesh-extract"}
    assert by_name["encode"]["gpu_class"] == "small"
    assert by_name["encode"]["python_name"] == "encode"
    assert by_name["mesh-extract"]["python_name"] == "extract_mesh"
    assert by_name["structure"]["gpu_class"] == "large"
    assert by_name["texture"]["gpu_class"] == "large"


def test_manifest_omits_stages_key_when_no_stages_declared() -> None:
    """A class without any @inference.stage methods must not surface an
    empty ``stages`` key — keeps the manifest tight and avoids the
    orchestrator mistaking ``[]`` for "explicitly zero stages".
    """
    from gen_worker.discovery.discover import _extract_class_function_methods

    @inference()
    class NoStages:
        def setup(self):
            pass

        @inference.function
        def generate(self, ctx: RequestContext, payload: Image) -> Mesh:
            return Mesh(glb_url="x")

        def shutdown(self):
            pass

    entries = _extract_class_function_methods(NoStages, module_name="test_module")
    assert len(entries) == 1
    assert "stages" not in entries[0]
