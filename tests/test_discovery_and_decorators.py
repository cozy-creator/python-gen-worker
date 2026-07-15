"""@endpoint authoring surface + discovery — integration suite.

One test per distinct behavior:

  1. Plain-function endpoint: discovered + dispatched with no class/setup.
  2. Class endpoint: public methods are routable; helpers must be private.
  3. model= placement key: a ModelChoice payload field is a curated closed
     enum whose picks carry typed defaults; `Choice | ModelRef` opens BYOM;
     ONE handler stays ONE function (no fan-out).
  4. Optional shutdown/setup; async-generator = streaming.
  5. Resources(vram_gb=..) implies gpu.
  6. msgspec.Meta bounds compile into the discovered endpoint.lock schema.
  7. Duplicate routable names raise at collection.
  8. Discovery walks submodules across package layouts, dedups re-exports,
     and skips third-party leaks.
"""

from __future__ import annotations

import enum
import logging
import textwrap
from pathlib import Path
from typing import Annotated, AsyncIterator, Union

import msgspec
import pytest

from gen_worker import (
    HF, Hub, Model, ModelChoice, ModelDefaults, ModelRef, RequestContext,
    Resources, endpoint,
)
from gen_worker.registry import collect_from_namespace, extract_specs


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    result: str


class _Defaults(ModelDefaults, frozen=True):
    steps: int
    guidance: float = 5.0


class _Model(ModelChoice[_Defaults], enum.Enum):
    SMALL = Model("small", Hub("o/small"), _Defaults(20), hot=True)
    LARGE = Model("large", HF("o/large"), _Defaults(30, 7.0), price=3.0)


# --------------------------------------------------------------------------- #
# 1. plain-function endpoint                                                    #
# --------------------------------------------------------------------------- #


def test_function_endpoint_discovered_and_shaped() -> None:
    @endpoint
    def hello(ctx: RequestContext, data: _In) -> _Out:
        return _Out(result=data.prompt)

    specs = extract_specs(hello)
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "hello"
    assert s.cls is None
    assert s.output_mode == "single"
    assert s.kind == "inference"


def test_function_endpoint_kwargs() -> None:
    @endpoint(kind="dataset", name="make-rows", resources=Resources(gpu=True))
    def build(ctx: RequestContext, data: _In) -> _Out:
        return _Out(result="x")

    (s,) = extract_specs(build)
    assert s.name == "make-rows"
    assert s.kind == "dataset"
    assert s.resources.gpu is True


def test_function_endpoint_rejects_bad_shapes() -> None:
    with pytest.raises(TypeError, match=r"\(ctx, payload\)"):
        @endpoint
        def nope(ctx: RequestContext) -> _Out:  # missing payload
            return _Out(result="")

    with pytest.raises(ValueError, match="runtime="):
        @endpoint(runtime="vllm")
        def nope2(ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")


# --------------------------------------------------------------------------- #
# 2. class endpoint: public methods route; setup optional                       #
# --------------------------------------------------------------------------- #


def test_class_methods_route_and_helpers_stay_private() -> None:
    @endpoint
    class Multi:
        def alpha(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result=self._suffix(data.prompt))

        def beta(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="b")

        def _suffix(self, s: str) -> str:  # private helper: not routable
            return s + "!"

    specs = extract_specs(Multi)
    assert sorted(s.name for s in specs) == ["alpha", "beta"]


def test_public_non_handler_method_is_rejected() -> None:
    with pytest.raises(TypeError, match="underscore"):
        @endpoint
        class Bad:
            def handler(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(result="")

            def helper(self):  # public non-handler
                pass


def test_model_slot_must_match_setup_param() -> None:
    with pytest.raises(ValueError, match="pipe"):
        @endpoint(models={"pipe": HF("o/r")})
        class Bad:
            def setup(self, other_name: str) -> None:
                pass

            def gen(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(result="")

    # model= shorthand: slot name comes from the setup parameter.
    @endpoint(model=HF("o/r", dtype="bf16"))
    class Good:
        def setup(self, pipe: str) -> None:
            self.pipe = pipe

        def gen(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (s,) = extract_specs(Good)
    assert list(s.models) == ["pipe"]
    assert s.models["pipe"].dtype == "bf16"


# --------------------------------------------------------------------------- #
# 3. model= placement key (pgw#509)                                             #
# --------------------------------------------------------------------------- #


class _CuratedIn(msgspec.Struct):
    prompt: str = ""
    model: _Model = _Model.SMALL


class _ByomIn(msgspec.Struct):
    prompt: str = ""
    model: Union[_Model, ModelRef] = _Model.SMALL


def test_model_choice_is_a_closed_enum_with_typed_defaults() -> None:
    # Wire form is the id string; the pick decodes back to a member whose
    # defaults are read as typed data (no ctx.models string-sniffing).
    raw = msgspec.msgpack.encode(_CuratedIn(model=_Model.LARGE))
    assert msgspec.msgpack.decode(raw) == {"prompt": "", "model": "large"}
    back = msgspec.msgpack.decode(raw, type=_CuratedIn)
    assert back.model is _Model.LARGE
    assert back.model.defaults.steps == 30
    assert back.model.ref.source == "huggingface" and back.model.ref.path == "o/large"
    assert back.model.hot is False and back.model.price == 3.0
    # JSON schema is a closed enum — the curated allowlist.
    schema = msgspec.json.schema(_CuratedIn)
    assert schema["$defs"]["_Model"]["enum"] == ["large", "small"]


def test_model_choice_byom_union_accepts_client_modelref() -> None:
    # A curated pick and an arbitrary client ModelRef decode through the same
    # field, distinguished on the wire by JSON type (string vs object).
    cur = msgspec.json.encode(_ByomIn(model=_Model.SMALL))
    byo = msgspec.json.encode(
        _ByomIn(model=ModelRef(source="civitai", path="12345"))
    )
    assert msgspec.json.decode(cur, type=_ByomIn).model is _Model.SMALL
    picked = msgspec.json.decode(byo, type=_ByomIn).model
    assert isinstance(picked, ModelRef) and picked.path == "12345"


def test_one_handler_is_one_function_not_a_fan_out() -> None:
    # 16 near-identical checkpoints used to be 16 variant functions; now the
    # class exposes ONE routable `generate` and selection is the payload arg.
    @endpoint(
        models={"pipeline": Hub("o/base"), "vae": Hub("o/vae")},
        resources=Resources(vram_gb=12),
    )
    class Gen:
        def setup(self, pipeline: object, vae: object) -> None: ...

        def generate(self, ctx: RequestContext, data: _CuratedIn) -> _Out:
            return _Out(result=str(data.model.defaults.steps))

    specs = extract_specs(Gen)
    assert [s.name for s in specs] == ["generate"]
    assert list(specs[0].models) == ["pipeline", "vae"]


def test_model_row_rejects_non_modelref_and_bad_defaults() -> None:
    with pytest.raises(TypeError, match="ModelRef"):
        Model("x", "not-a-ref", _Defaults(10))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ModelDefaults"):
        Model("x", Hub("o/r"), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        Model("", Hub("o/r"), _Defaults(10))


# --------------------------------------------------------------------------- #
# 4. optional setup/shutdown; async-generator = streaming                       #
# --------------------------------------------------------------------------- #


def test_streaming_detected_from_async_generator() -> None:
    @endpoint
    class Streamer:
        async def stream(self, ctx: RequestContext, data: _In) -> AsyncIterator[_Out]:
            yield _Out(result="x")

    (s,) = extract_specs(Streamer)
    assert s.output_mode == "stream"
    assert s.is_async_gen


def test_no_setup_no_shutdown_class_is_valid() -> None:
    @endpoint
    class Bare:
        def run(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="ok")

    (s,) = extract_specs(Bare)
    assert s.name == "run"
    assert not hasattr(Bare, "setup")


# --------------------------------------------------------------------------- #
# 5. Resources                                                                  #
# --------------------------------------------------------------------------- #


def test_resources_vram_implies_gpu() -> None:
    assert Resources().gpu is False
    assert Resources(vram_gb=12).gpu is True
    assert Resources(compute_capability=8.0).gpu is True
    assert Resources(gpu=True).vram_gb is None
    with pytest.raises(ValueError):
        Resources(vram_gb=0)
    with pytest.raises(ValueError):
        Resources(compute_capability=-1)


def test_resources_host_ask_gw490() -> None:
    r = Resources(ram_gb=64, vcpus=16)
    assert r.ram_gb == 64.0 and r.vcpus == 16
    assert r.gpu is False  # host asks never imply a GPU
    raw = msgspec.to_builtins(r)
    assert raw["ram_gb"] == 64.0 and raw["vcpus"] == 16
    assert "ram_gb" not in msgspec.to_builtins(Resources())  # omit_defaults
    with pytest.raises(ValueError):
        Resources(ram_gb=0)
    with pytest.raises(ValueError):
        Resources(vcpus=-2)


# --------------------------------------------------------------------------- #
# 6. msgspec.Meta bounds compile into the discovered schema                     #
# --------------------------------------------------------------------------- #


class BoundedInput(msgspec.Struct):
    steps: Annotated[int, msgspec.Meta(ge=1, le=50)] = 4
    name: Annotated[str, msgspec.Meta(min_length=1, max_length=64)] = "x"


def test_meta_bounds_compiled_into_discovered_input_schema() -> None:
    from gen_worker.discovery.discover import _extract_entries

    @endpoint
    class Bounded:
        def gen(self, ctx: RequestContext, data: BoundedInput) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Bounded, "testmod")
    schema = entry["input_schema"]
    defs = schema.get("$defs", {})
    props = (defs.get("BoundedInput") or {}).get("properties", {})
    assert props["steps"]["minimum"] == 1
    assert props["steps"]["maximum"] == 50
    assert props["name"]["minLength"] == 1
    assert props["name"]["maxLength"] == 64


# --------------------------------------------------------------------------- #
# 7. duplicate routable names                                                   #
# --------------------------------------------------------------------------- #


def test_duplicate_function_names_raise_at_collection() -> None:
    import types

    @endpoint
    class A:
        def gen(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="a")

    @endpoint
    class B:
        def gen(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="b")

    mod = types.ModuleType("dupes")
    mod.A, mod.B = A, B
    with pytest.raises(ValueError, match="duplicate"):
        collect_from_namespace(mod)


# --------------------------------------------------------------------------- #
# 8. package walking                                                            #
# --------------------------------------------------------------------------- #


def _endpoint_src(cls_name: str, fn_name: str) -> str:
    return textwrap.dedent(f"""
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            x: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class {cls_name}:
            def {fn_name}(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """)


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_discovery_walks_layouts_dedups_and_skips_third_party(tmp_pkg: Path, caplog) -> None:
    from gen_worker.discovery.walk import find_endpoints

    # (a) re-exported submodule class is returned exactly once.
    reexport = tmp_pkg / "ep_reexport"
    reexport.mkdir()
    (reexport / "clone.py").write_text(_endpoint_src("CloneHF", "clone"))
    (reexport / "main.py").write_text("from .clone import CloneHF\n")
    (reexport / "__init__.py").write_text("from .main import *  # noqa: F403\n")

    # (b) un-reexported submodule class found via the pkgutil walk.
    walk = tmp_pkg / "ep_walk"
    walk.mkdir()
    (walk / "synth.py").write_text(_endpoint_src("SynthEp", "synth"))
    (walk / "__init__.py").write_text("")

    # (c) third-party class re-exported in is SKIPPED with a diagnostic.
    other = tmp_pkg / "other_pkg"
    other.mkdir()
    (other / "__init__.py").write_text(_endpoint_src("StrayClass", "stray"))
    stray = tmp_pkg / "ep_stray"
    stray.mkdir()
    (stray / "__init__.py").write_text("from other_pkg import StrayClass\n")

    with caplog.at_level(logging.INFO, logger="gen_worker.discovery.walk"):
        found = find_endpoints(["ep_reexport", "ep_walk", "ep_stray"])

    qualnames = [f.qualname for f in found]
    assert "CloneHF" in qualnames
    assert qualnames.count("CloneHF") == 1  # re-export dedup
    assert "SynthEp" in qualnames           # submodule walk fallback
    assert "StrayClass" not in qualnames     # third-party leak rejected
    assert any(
        "outside the walked package" in r.message for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# 9. producer kinds publish explicitly — generator handlers are rejected       #
# --------------------------------------------------------------------------- #


def test_producer_kind_rejects_sync_generator_class_handler() -> None:
    with pytest.raises(TypeError, match="publish_flavors"):
        @endpoint(kind="conversion")
        class BadConversion:
            def run(self, ctx: RequestContext, data: _In):
                yield _Out(result="x")


def test_producer_kind_rejects_async_generator_class_handler() -> None:
    with pytest.raises(TypeError, match="inference-only"):
        @endpoint(kind="training")
        class BadTraining:
            async def run(self, ctx: RequestContext, data: _In):
                yield _Out(result="x")


def test_producer_kind_rejects_generator_function() -> None:
    with pytest.raises(TypeError, match="must not be a generator"):
        @endpoint(kind="dataset")
        def bad_dataset(ctx: RequestContext, data: _In):
            yield _Out(result="x")


def test_from_scratch_example_uses_publish_contract() -> None:
    """Discovery smoke for examples/from-scratch: imports, one conversion-kind
    non-generator handler, result struct output (the explicit-publish shape)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "from-scratch" / "from_scratch.py"
    spec = importlib.util.spec_from_file_location("from_scratch_example", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    (s,) = extract_specs(mod.FromScratch)
    assert s.kind == "conversion"
    assert s.output_mode == "single"
    assert not s.is_async_gen
    assert s.output_type is mod.FromScratchResult


def test_manifest_emits_model_placement_key(tmp_pkg: Path) -> None:
    """One handler -> one function; a ModelChoice payload field emits the
    `model` block (field + byom + slot + curated choices carrying structured
    ModelRef bindings, typed defaults, and hot/price hints) — the pgw#509
    SDK->tensorhub (th#761) contract. Every choice binding gets the
    endpoint's Compile(family=...) stamp (pgw#519), unconditionally when
    known (pgw#523: family stamping is no longer allow_lora-triggered)."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_model"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import enum
        from typing import Union
        import msgspec
        import gen_worker
        from gen_worker import (Compile, Hub, HF, Model, ModelChoice, ModelDefaults,
                                ModelRef, RequestContext, Resources, endpoint)

        class D(ModelDefaults, frozen=True):
            steps: int
            guidance: float = 5.0

        class M(ModelChoice[D], enum.Enum):
            WAI = Model("wai", Hub("o/wai"), D(28, 6.0), hot=True)
            PONY = Model("pony", HF("o/pony"), D(30), price=2.0)

        class In_(msgspec.Struct):
            prompt: str = ""
            model: Union[M, ModelRef] = M.WAI

        class Out_(msgspec.Struct):
            y: str

        @endpoint(models={"pipeline": Hub("o/base"), "vae": Hub("o/vae")},
                  resources=Resources(vram_gb=12),
                  compile=Compile(family="wai-arch", shapes=((512, 512),)))
        class Gen:
            def setup(self, pipeline: object, vae: object) -> None:
                gen_worker.arm_compile(pipeline)  # pgw#517: self-loaded slots
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_model.main")
    assert [f["name"] for f in fns] == ["generate"]
    block = fns[0]["model"]
    assert block["field"] == "model"
    assert block["byom"] is True
    assert block["slot"] == "pipeline"
    by_id = {c["id"]: c for c in block["choices"]}
    assert set(by_id) == {"wai", "pony"}
    assert by_id["wai"]["binding"]["provider"] == "tensorhub"
    assert by_id["wai"]["binding"]["ref"] == "o/wai"
    assert by_id["wai"]["binding"]["family"] == "wai-arch"
    assert by_id["wai"]["defaults"] == {"steps": 28, "guidance": 6.0}
    assert by_id["wai"]["hot"] is True
    assert by_id["pony"]["binding"]["provider"] == "huggingface"
    assert by_id["pony"]["binding"]["family"] == "wai-arch"
    assert by_id["pony"]["price"] == 2.0
    assert "hot" not in by_id["pony"]      # false hint omitted


def test_model_shorthand_skips_server_handle_setup_param() -> None:
    """runtime= endpoints inject a ServerHandle into setup(); the model=
    shorthand must resolve the slot from the remaining parameter."""
    from gen_worker.runtimes.server import ServerHandle, VLLMRuntime

    runtime = VLLMRuntime(max_model_len=16384, gpu_memory_utilization=0.94)

    @endpoint(model=HF("o/llm"), resources=Resources(vram_gb=40), runtime=runtime)
    class Chat:
        def setup(self, model: str, server: ServerHandle) -> None:
            self.base_url = server.base_url

        def complete(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (s,) = extract_specs(Chat)
    assert s.runtime is runtime
    assert list(s.models) == ["model"]
    assert s.models["model"].path == "o/llm"


# --------------------------------------------------------------------------- #
# 8. binding family stamping (ie#358 / pgw#523)                                #
# --------------------------------------------------------------------------- #


def test_compile_block_emits_video_shapes_and_storage_dtype() -> None:
    """ie#381: the lock's compile block carries (w, h, frames) rows verbatim
    and the primary binding's weight-storage lane, so the hub's cell producer
    builds from an identically-loaded (fp8) pipeline."""
    import gen_worker
    from gen_worker import Compile, Hub
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(
        model=Hub("tensorhub/ltx-2.3-distilled", storage_dtype="fp8"),
        resources=Resources(vram_gb=78),
        compile=Compile(
            family="ltx-2.3",
            shapes=((960, 544, 241), (1920, 1088, 241), (1280, 704, 121)),
            targets=("transformer",),
        ),
    )
    class Gen:
        def setup(self, model: str) -> None:
            # self-loading (str) slot: arms compile explicitly (pgw#517).
            self.model = model
            gen_worker.arm_compile(self.model)

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Gen, "testmod")
    assert entry["compile"]["shapes"] == [[960, 544, 241], [1920, 1088, 241], [1280, 704, 121]]
    assert entry["compile"]["storage_dtype"] == "fp8"
    assert entry["compile"]["targets"] == ["transformer"]


def test_compile_block_omits_storage_dtype_for_bf16_bindings() -> None:
    from gen_worker import Compile, Hub
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(
        model=Hub("cozy/sdxl-base"),
        resources=Resources(vram_gb=12),
        compile=Compile(family="sdxl", shapes=((1024, 1024),)),
    )
    class Gen:
        def setup(self, model: str) -> None:
            # self-loading (str) slot: arms compile explicitly (pgw#517).
            self.model = model
            gen_worker.arm_compile(self.model)

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Gen, "testmod")
    assert "storage_dtype" not in entry["compile"]


def test_binding_emits_family_stamp_from_compile() -> None:
    """pgw#523: family stamping is unconditional-when-known — no allow_lora
    flag gates it, and ModelRef carries no such flag any more (identity !=
    permission; overlay permission lives on the slot-policy loras axis,
    th#772)."""
    from gen_worker import Compile, Hub
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(
        model=Hub("cozy/sdxl-base"),
        resources=Resources(vram_gb=12),
        compile=Compile(family="sdxl", shapes=((1024, 1024),)),
    )
    class Gen:
        def setup(self, model: str) -> None:
            # self-loading (str) slot: arms compile explicitly (pgw#517).
            self.model = model
            gen_worker.arm_compile(self.model)

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Gen, "testmod")
    (block,) = entry["bindings"].values()
    assert "allow_lora" not in block
    assert block["family"] == "sdxl"


def test_binding_emits_no_family_when_none_declared() -> None:
    """pgw#523: with no Compile(family=...) and no fallback-preset family,
    a binding simply carries no `family` key — this used to hard-fail when
    allow_lora=True lacked a family (th#586's gate rekeyed off the binding/
    slot family directly, not that flag, so the co-occurrence requirement
    is gone too)."""
    from gen_worker import Hub
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(model=Hub("cozy/sdxl-base"), resources=Resources(vram_gb=12))
    class Gen:
        def setup(self, model: str) -> None:
            self.model = model

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Gen, "testmod")
    (block,) = entry["bindings"].values()
    assert "family" not in block


def test_components_binding_emits_in_manifest() -> None:
    """pgw#505: a declared components= subset surfaces on the manifest
    binding block for both tensorhub and huggingface sources — the hub
    reads it to scope its ModelOp DOWNLOAD resolve; the worker's own
    download layer reads it off the binding object directly (not the
    manifest) on the hub-less/local paths."""
    from gen_worker import Hub
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(
        models={
            "pipeline": Hub("o/sdxl-full", components=("vae",)),
            "extra": HF("o/hf-repo", components=("unet", "text_encoder")),
        },
        resources=Resources(vram_gb=12),
    )
    class Gen:
        def setup(self, pipeline: str, extra: str) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Gen, "testmod")
    bindings = entry["bindings"]
    assert bindings["pipeline"]["components"] == ["vae"]
    assert bindings["extra"]["components"] == ["unet", "text_encoder"]


def test_no_components_binding_omits_manifest_key() -> None:
    from gen_worker import Hub
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(model=Hub("o/whole-repo"), resources=Resources(vram_gb=12))
    class Gen:
        def setup(self, model: str) -> None:
            self.model = model

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    (entry,) = _extract_entries(Gen, "testmod")
    (block,) = entry["bindings"].values()
    assert "components" not in block


def test_model_choice_binding_family_matches_top_level_binding(tmp_pkg: Path) -> None:
    """pgw#519: model.choices[].binding gets the SAME family stamp that a
    top-level bindings block gets from Compile(family=...) — tensorhub's
    th#586 architecture gate polices LoRA targets against it on both
    surfaces identically. pgw#523: the stamp is unconditional-when-known,
    so EVERY binding under the endpoint (including "vae", which carries no
    permission flag of any kind any more) gets it, not just some subset."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_choice_family"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import enum
        from typing import Union
        import msgspec
        import gen_worker
        from gen_worker import (Compile, Hub, HF, Model, ModelChoice, ModelDefaults,
                                ModelRef, RequestContext, Resources, endpoint)

        class D(ModelDefaults, frozen=True):
            steps: int = 28

        class M(ModelChoice[D], enum.Enum):
            A = Model("a", Hub("o/a"), D(28))
            B = Model("b", Hub("o/b"), D(30))

        class In_(msgspec.Struct):
            prompt: str = ""
            model: Union[M, ModelRef] = M.A

        class Out_(msgspec.Struct):
            y: str

        @endpoint(models={"pipeline": Hub("o/base"), "vae": Hub("o/vae")},
                  resources=Resources(vram_gb=12),
                  compile=Compile(family="sdxl", shapes=((1024, 1024),)))
        class Gen:
            def setup(self, pipeline: object, vae: object) -> None:
                gen_worker.arm_compile(pipeline)  # pgw#517: self-loaded slots
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_choice_family.main")
    (fn,) = fns

    top_level_family = fn["bindings"]["pipeline"]["family"]
    assert top_level_family == "sdxl"
    assert fn["bindings"]["vae"]["family"] == "sdxl"  # stamped unconditionally now

    by_id = {c["id"]: c for c in fn["model"]["choices"]}
    assert set(by_id) == {"a", "b"}
    for choice in by_id.values():
        assert "allow_lora" not in choice["binding"]
        # Choice-binding emission equals top-level emission w.r.t. family.
        assert choice["binding"]["family"] == top_level_family


def test_model_choice_binding_emits_no_family_when_none_declared(tmp_pkg: Path) -> None:
    """Mirrors test_binding_emits_no_family_when_none_declared for the
    choices[].binding surface: with no Compile(family=...) a choice binding
    simply carries no `family` key — discovery no longer hard-fails here
    (pgw#523 retired the allow_lora-requires-family co-occurrence check)."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_choice_family_missing"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import enum
        from typing import Union
        import msgspec
        from gen_worker import (Hub, Model, ModelChoice, ModelDefaults,
                                ModelRef, RequestContext, Resources, endpoint)

        class D(ModelDefaults, frozen=True):
            steps: int = 28

        class M(ModelChoice[D], enum.Enum):
            A = Model("a", Hub("o/a"), D(28))

        class In_(msgspec.Struct):
            prompt: str = ""
            model: Union[M, ModelRef] = M.A

        class Out_(msgspec.Struct):
            y: str

        @endpoint(models={"pipeline": Hub("o/base")}, resources=Resources(vram_gb=12))
        class Gen:
            def setup(self, pipeline: object) -> None: ...
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="ok")
    """))

    fns = discover_functions(tmp_pkg, main_module="ep_choice_family_missing.main")
    (fn,) = fns
    (choice,) = fn["model"]["choices"]
    assert "family" not in choice["binding"]
