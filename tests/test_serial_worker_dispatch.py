"""SerialWorker class-shape dispatch wiring tests (#322 / #328).

Mirrors test_batched_worker_discovery.py but for the synchronous archetype:
the sync `@inference` class. The smoke test confirms:

  1. `@inference` on a sync class attaches archetype="SerialWorker".
  2. `_register_endpoint_class` routes to `_register_endpoint_class_serial`
     based on archetype + runtime=None, producing a `_SerialWorkerSpec`
     keyed by `slugify_name(method_name)`.
  3. `_ensure_serial_class_started` resolves model bindings to local paths
     and calls `instance.setup(**models)`.
  4. The `warming` startup phase signal is emitted around warmup().
  5. `_shutdown_serial_workers` calls `instance.shutdown()` exactly once.
  6. `TORCHINDUCTOR_CACHE_DIR` is set before tenant setup() runs.
  7. `validate_endpoint_lock` accepts a clean class-shape lock and rejects
     a stale function-shape one.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from typing import AsyncIterator, Iterator

import msgspec
import pytest

from gen_worker import HFRepo, Repo, RequestContext, inference
from gen_worker._worker_support import _SerialWorkerSpec
from gen_worker.discovery import validate_endpoint_lock
from gen_worker.worker import Worker


class GenerateInput(msgspec.Struct):
    prompt: str
    steps: int = 4


class GenerateOutput(msgspec.Struct):
    result: str


class TokenDelta(msgspec.Struct):
    delta_text: str = ""
    finished: bool = False
    item_id: str = "item-0"


class FakeSetupPipeline:
    calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):  # noqa: ANN001
        cls.calls.append((source, kwargs))
        return cls()


def _make_serial_class(setup_calls, warmup_calls, shutdown_calls):
    """Build a fresh SerialWorker class for each test."""

    @inference(models={"pipe": Repo("test-org/test-repo").flavor("bf16")})
    class TestSerial:
        def setup(self, pipe):
            setup_calls.append({"pipe": pipe})
            self.pipe = pipe

        def warmup(self):
            warmup_calls.append(True)

        @inference.function
        def generate(self, ctx: RequestContext, payload: GenerateInput) -> GenerateOutput:
            return GenerateOutput(result=f"echo:{payload.prompt}:{payload.steps}")

        def shutdown(self):
            shutdown_calls.append(True)

    return TestSerial


def _make_serial_streaming_class():
    """Build a SerialWorker class with an incremental-output method."""

    @inference()
    class TestSerialStreaming:
        def setup(self):
            pass

        @inference.function
        def chat(self, ctx: RequestContext, payload: GenerateInput) -> Iterator[TokenDelta]:
            for tok in payload.prompt.split():
                yield TokenDelta(delta_text=tok, item_id="item-0")
            yield TokenDelta(delta_text="", finished=True, item_id="item-0")

        def shutdown(self):
            pass

    return TestSerialStreaming


def _bare_worker() -> Worker:
    """Build a Worker instance with only the dispatch dicts populated.

    Avoids the heavyweight gRPC init the real `__init__` does; tests against
    discovery + dispatch routing don't need any of it.
    """
    w = Worker.__new__(Worker)
    w._request_specs = {}
    w._training_specs = {}
    w._batched_specs = {}
    w._batched_instances = []
    w._serial_class_specs = {}
    w._serial_class_instances = []
    w._discovered_resources = {}
    w._function_schemas = {}
    w._batched_loop = None
    w._batched_loop_thread = None
    w._batched_inflight_lock = threading.Lock()
    w._batched_inflight = {}
    w._micro_batch_aggregators = {}
    w.scheduler_addr = ""
    w.worker_id = "test"
    return w


def test_inference_class_sync_attaches_serial_worker_archetype() -> None:
    setup_calls: list = []
    warmup_calls: list = []
    shutdown_calls: list = []
    cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
    spec = getattr(cls, "__gen_worker_endpoint_spec__")
    assert spec.kind == "inference"
    assert spec.runtime is None
    assert getattr(cls, "__gen_worker_archetype__") == "SerialWorker"
    methods = getattr(cls, "__gen_worker_function_methods__")
    assert len(methods) == 1
    name, _method, fn_spec = methods[0]
    assert name == "generate"


def test_register_endpoint_class_routes_serial_archetype_to_serial_registry() -> None:
    setup_calls: list = []
    warmup_calls: list = []
    shutdown_calls: list = []
    cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
    w = _bare_worker()
    n = w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    assert n == 1
    assert "generate" in w._serial_class_specs
    assert "generate" not in w._batched_specs
    sspec = w._serial_class_specs["generate"]
    assert isinstance(sspec, _SerialWorkerSpec)
    assert sspec.payload_type is GenerateInput
    assert sspec.output_type is GenerateOutput
    assert sspec.output_mode == "single"
    assert sspec.delta_type is None
    # Per-class record carries the singleton instance + endpoint_spec.
    assert len(w._serial_class_instances) == 1
    rec = w._serial_class_instances[0]
    assert rec["cls_name"] == "TestSerial"
    assert rec["instance"] is sspec.instance
    assert rec["started"] is False
    assert rec["shutdown_done"] is False


def test_register_endpoint_class_serial_picks_up_incremental_return() -> None:
    cls = _make_serial_streaming_class()
    w = _bare_worker()
    n = w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    assert n == 1
    sspec = w._serial_class_specs["chat"]
    assert sspec.output_mode == "incremental"
    assert sspec.delta_type is TokenDelta


def test_register_endpoint_class_serial_accepts_async_method() -> None:
    """#345 Improvement B: an `async def` @inference.function method WITHOUT
    runtime= is now a first-class SerialWorker async handler (not BatchedWorker,
    not rejected). `_inspect_serial_method` returns is_async=True for a properly
    annotated async coroutine handler.
    """
    w = _bare_worker()

    class _Mock:
        async def fn(self, ctx: RequestContext, payload: GenerateInput) -> GenerateOutput:  # noqa: ANN001
            return GenerateOutput(result="ok")

    (
        payload_type,
        output_type,
        delta_type,
        output_mode,
        ctx_param,
        payload_param,
        is_async,
    ) = w._inspect_serial_method(_Mock, _Mock.fn)
    assert is_async is True
    assert output_mode == "single"
    assert payload_type is GenerateInput
    assert output_type is GenerateOutput
    assert delta_type is None


def test_inspect_serial_method_async_incremental_requires_async_iterator() -> None:
    """#345 Improvement B validation: an `async def` generator handler must
    annotate its return as AsyncIterator[Delta]; a sync Iterator annotation on
    an async generator is rejected.
    """
    w = _bare_worker()

    class _Bad:
        async def fn(self, ctx: RequestContext, payload: GenerateInput) -> Iterator[TokenDelta]:  # noqa: ANN001
            yield TokenDelta(delta_text="x")

    with pytest.raises(ValueError, match="AsyncIterator"):
        w._inspect_serial_method(_Bad, _Bad.fn)


def test_inspect_serial_method_async_iterator_yields_delta() -> None:
    """#345 Improvement B: a correctly-annotated async-generator handler
    (AsyncIterator[Delta]) registers as incremental + async.
    """
    w = _bare_worker()

    class _Good:
        async def fn(self, ctx: RequestContext, payload: GenerateInput) -> AsyncIterator[TokenDelta]:  # noqa: ANN001
            yield TokenDelta(delta_text="x")

    (
        payload_type,
        output_type,
        delta_type,
        output_mode,
        ctx_param,
        payload_param,
        is_async,
    ) = w._inspect_serial_method(_Good, _Good.fn)
    assert is_async is True
    assert output_mode == "incremental"
    assert delta_type is TokenDelta


def test_discovery_emits_is_async_for_serial_methods() -> None:
    """#345 Improvement B: the discovery manifest surfaces `is_async` per
    function — True for an `async def` handler, False for a sync one.
    """
    from gen_worker.discovery.discover import _extract_class_function_methods

    @inference()
    class MixedSerial:
        def setup(self):
            pass

        @inference.function
        def sync_fn(self, ctx: RequestContext, payload: GenerateInput) -> GenerateOutput:
            return GenerateOutput(result="s")

    @inference()
    class AsyncSerial:
        def setup(self):
            pass

        @inference.function
        async def async_fn(self, ctx: RequestContext, payload: GenerateInput) -> GenerateOutput:
            return GenerateOutput(result="a")

    sync_fns = _extract_class_function_methods(MixedSerial, "m")
    async_fns = _extract_class_function_methods(AsyncSerial, "m")
    assert sync_fns[0]["is_async"] is False
    assert async_fns[0]["is_async"] is True
    # Async-without-runtime stays SerialWorker (not BatchedWorker).
    assert getattr(AsyncSerial, "__gen_worker_archetype__") == "SerialWorker"


def test_torchinductor_cache_dir_set_on_serial_registration() -> None:
    """SerialWorker registration must set TORCHINDUCTOR_CACHE_DIR before any
    tenant setup() runs. The cross-cutting hook is the whole point of
    #322's cache-directory plumbing.
    """
    setup_calls: list = []
    warmup_calls: list = []
    shutdown_calls: list = []
    cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
    # Clear any prior value to test that registration sets it from scratch.
    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR"), \
        "TORCHINDUCTOR_CACHE_DIR must be set during SerialWorker registration"


def test_torchinductor_cache_dir_honors_env_override() -> None:
    """Operators can point at a Runpod volume mount via the env var.
    Registration must not clobber an existing value.
    """
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/test-override-cache"
    try:
        setup_calls: list = []
        warmup_calls: list = []
        shutdown_calls: list = []
        cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
        w = _bare_worker()
        w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/test-override-cache"
    finally:
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)


def test_resolve_serial_model_paths_returns_kwargs_for_static_refs() -> None:
    """`_resolve_serial_model_paths` returns {kwarg_name: ref_string} for
    every binding with a static .ref. Used by setup(**models).
    """
    setup_calls: list = []
    warmup_calls: list = []
    shutdown_calls: list = []
    cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
    w = _bare_worker()
    paths = w._resolve_serial_model_paths(cls.__gen_worker_endpoint_spec__)
    assert "pipe" in paths
    assert paths["pipe"]
    # Empty bindings → empty dict (not None).
    assert w._resolve_serial_model_paths(None) == {}


def test_resolve_serial_model_paths_materializes_hfrepo_bindings(tmp_path, monkeypatch) -> None:
    """Static HFRepo setup kwargs are downloaded by the worker boundary."""

    @inference(models={"pipe": HFRepo("black-forest-labs/FLUX.2-klein-base-4b-fp8")})
    class TestHFSerial:
        def setup(self, pipe):  # pragma: no cover - not invoked here
            self.pipe = pipe

        @inference.function
        def generate(self, ctx: RequestContext, payload: GenerateInput) -> GenerateOutput:
            return GenerateOutput(result="ok")

    class FakeDownloader:
        def __init__(self) -> None:
            self.calls = []

        def download(self, model_ref, dest_dir, filename=None):  # noqa: ANN001
            self.calls.append((model_ref, dest_dir, filename))
            local = tmp_path / "hf-snapshot"
            local.mkdir()
            return str(local)

    fake = FakeDownloader()
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path / "cas")
    w = _bare_worker()
    w._downloader = fake

    paths = w._resolve_serial_model_paths(TestHFSerial.__gen_worker_endpoint_spec__)

    assert paths == {"pipe": str(tmp_path / "hf-snapshot")}
    assert fake.calls == [
        ("black-forest-labs/FLUX.2-klein-base-4b-fp8", str(tmp_path / "cas"), None)
    ]


def test_resolve_serial_setup_kwargs_loads_typed_hfrepo_binding(tmp_path, monkeypatch) -> None:
    @inference(models={"pipe": HFRepo("acme/model").dtype("bf16")})
    class TestTypedHFSerial:
        def setup(self, pipe: FakeSetupPipeline):
            self.pipe = pipe

        @inference.function
        def generate(self, ctx: RequestContext, payload: GenerateInput) -> GenerateOutput:
            return GenerateOutput(result="ok")

    class FakeDownloader:
        def download(self, model_ref, dest_dir, filename=None):  # noqa: ANN001
            local = tmp_path / "hf-snapshot"
            local.mkdir(exist_ok=True)
            return str(local)

    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path / "cas")
    w = _bare_worker()
    w._downloader = FakeDownloader()
    FakeSetupPipeline.calls.clear()

    kwargs = w._resolve_serial_setup_kwargs(
        TestTypedHFSerial.__gen_worker_endpoint_spec__,
        TestTypedHFSerial().setup,
    )

    assert isinstance(kwargs["pipe"], FakeSetupPipeline)
    assert FakeSetupPipeline.calls == [(str(tmp_path / "hf-snapshot"), {})]


def test_flux2_klein_modelopt_root_safetensors_loaded_by_worker(tmp_path, monkeypatch) -> None:
    loaded: dict[str, object] = {}
    quant_dir = tmp_path / "quant"
    base_dir = tmp_path / "base"
    quant_dir.mkdir()
    base_dir.mkdir()
    checkpoint = quant_dir / "flux-2-klein-base-4b-fp8.safetensors"
    checkpoint.write_bytes(b"fake")
    # The materialized base snapshot carries the transformer config that
    # diffusers' single-file `config=<local dir>` path reads; create it so the
    # loader keeps the local-dir config source (vs falling back to the repo id).
    (base_dir / "transformer").mkdir()
    (base_dir / "transformer" / "config.json").write_text("{}")

    class FakeFlux2Transformer2DModel:
        @staticmethod
        def from_single_file(path, **kwargs):  # noqa: ANN001
            loaded["single_file_path"] = path
            loaded["single_file_kwargs"] = kwargs
            return "transformer"

    class FakeQuantConfig:
        def __init__(self, quant_type, **kwargs):  # noqa: ANN001
            loaded["quant_type"] = quant_type
            loaded["quant_kwargs"] = kwargs

    class Flux2KleinPipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):  # noqa: ANN001
            loaded["pipeline_source"] = source
            loaded["pipeline_kwargs"] = kwargs
            return cls()

    Flux2KleinPipeline.__module__ = "diffusers.pipelines.flux2"

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.Flux2Transformer2DModel = FakeFlux2Transformer2DModel
    fake_quantizers = types.ModuleType("diffusers.quantizers")
    fake_quant_config = types.ModuleType("diffusers.quantizers.quantization_config")
    fake_quant_config.NVIDIAModelOptConfig = FakeQuantConfig
    fake_modelopt = types.ModuleType("modelopt")
    fake_modelopt_torch = types.ModuleType("modelopt.torch")
    fake_modelopt_opt = types.ModuleType("modelopt.torch.opt")
    fake_modelopt_opt.enable_huggingface_checkpointing = lambda: loaded.setdefault("checkpointing", True)

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", fake_quantizers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.quantization_config", fake_quant_config)
    monkeypatch.setitem(sys.modules, "modelopt", fake_modelopt)
    monkeypatch.setitem(sys.modules, "modelopt.torch", fake_modelopt_torch)
    monkeypatch.setitem(sys.modules, "modelopt.torch.opt", fake_modelopt_opt)
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path / "cas")

    class FakeDownloader:
        def download(self, model_ref, dest_dir, filename=None):  # noqa: ANN001
            assert model_ref == "black-forest-labs/FLUX.2-klein-base-4B"
            return str(base_dir)

    w = _bare_worker()
    w._downloader = FakeDownloader()

    pipe = w._try_load_flux2_klein_modelopt_pipeline(
        requested_type=Flux2KleinPipeline,
        model_source=str(quant_dir),
        original_ref="black-forest-labs/FLUX.2-klein-base-4b-fp8",
        torch_dtype=object(),
    )

    assert isinstance(pipe, Flux2KleinPipeline)
    assert loaded["checkpointing"] is True
    assert loaded["quant_type"] == "FP8"
    assert loaded["single_file_path"] == str(checkpoint)
    assert loaded["single_file_kwargs"]["config"] == str(base_dir)
    assert loaded["single_file_kwargs"]["subfolder"] == "transformer"
    assert loaded["pipeline_source"] == str(base_dir)
    assert loaded["pipeline_kwargs"]["transformer"] == "transformer"


def test_flux2_klein_modelopt_falls_back_to_repo_id_config_when_transformer_missing(
    tmp_path, monkeypatch
) -> None:
    """When the materialized base snapshot lacks ``transformer/config.json``,
    `from_single_file` must receive the base repo id as ``config=`` (diffusers'
    Hub-resolved path) instead of the local dir, whose local config-resolution
    branch is the one that raises a bare AttributeError on this diffusers
    commit. The pipeline still loads from the local dir.
    """
    loaded: dict[str, object] = {}
    quant_dir = tmp_path / "quant"
    base_dir = tmp_path / "base"
    quant_dir.mkdir()
    base_dir.mkdir()  # no transformer/ subfolder -> triggers the fallback
    checkpoint = quant_dir / "flux-2-klein-base-4b-nvfp4.safetensors"
    checkpoint.write_bytes(b"fake")

    class FakeFlux2Transformer2DModel:
        @staticmethod
        def from_single_file(path, **kwargs):  # noqa: ANN001
            loaded["single_file_kwargs"] = kwargs
            return "transformer"

    class FakeQuantConfig:
        def __init__(self, quant_type, **kwargs):  # noqa: ANN001
            loaded["quant_type"] = quant_type

    class Flux2KleinPipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):  # noqa: ANN001
            loaded["pipeline_source"] = source
            return cls()

    Flux2KleinPipeline.__module__ = "diffusers.pipelines.flux2"

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.Flux2Transformer2DModel = FakeFlux2Transformer2DModel
    fake_quantizers = types.ModuleType("diffusers.quantizers")
    fake_quant_config = types.ModuleType("diffusers.quantizers.quantization_config")
    fake_quant_config.NVIDIAModelOptConfig = FakeQuantConfig
    fake_modelopt = types.ModuleType("modelopt")
    fake_modelopt_torch = types.ModuleType("modelopt.torch")
    fake_modelopt_opt = types.ModuleType("modelopt.torch.opt")
    fake_modelopt_opt.enable_huggingface_checkpointing = lambda: None

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", fake_quantizers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.quantization_config", fake_quant_config)
    monkeypatch.setitem(sys.modules, "modelopt", fake_modelopt)
    monkeypatch.setitem(sys.modules, "modelopt.torch", fake_modelopt_torch)
    monkeypatch.setitem(sys.modules, "modelopt.torch.opt", fake_modelopt_opt)
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path / "cas")

    class FakeDownloader:
        def download(self, model_ref, dest_dir, filename=None):  # noqa: ANN001
            return str(base_dir)

    w = _bare_worker()
    w._downloader = FakeDownloader()

    pipe = w._try_load_flux2_klein_modelopt_pipeline(
        requested_type=Flux2KleinPipeline,
        model_source=str(quant_dir),
        original_ref="black-forest-labs/FLUX.2-klein-base-4b-nvfp4",
        torch_dtype=object(),
    )

    assert isinstance(pipe, Flux2KleinPipeline)
    assert loaded["quant_type"] == "NVFP4"
    # Fallback: from_single_file gets the repo id, not the (incomplete) local dir.
    assert loaded["single_file_kwargs"]["config"] == "black-forest-labs/FLUX.2-klein-base-4B"
    # Pipeline still assembles from the locally materialized snapshot.
    assert loaded["pipeline_source"] == str(base_dir)


def test_ensure_serial_class_started_calls_setup_with_resolved_models() -> None:
    """`_ensure_serial_class_started` resolves models and passes them to
    setup() as kwargs (`def setup(self, **models)` or named params).
    """
    setup_calls: list = []
    warmup_calls: list = []
    shutdown_calls: list = []
    cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    rec = w._serial_class_instances[0]
    assert not rec["started"]
    w._ensure_serial_class_started(rec)
    assert rec["started"]
    assert len(setup_calls) == 1
    assert "pipe" in setup_calls[0], setup_calls
    assert len(warmup_calls) == 1
    # Idempotent: second call is a no-op.
    w._ensure_serial_class_started(rec)
    assert len(setup_calls) == 1
    assert len(warmup_calls) == 1


def test_shutdown_serial_workers_invokes_instance_shutdown_once() -> None:
    """Drain path: shutdown() should be called exactly once per registered
    class. Subsequent calls are no-ops thanks to the shutdown_done flag.
    """
    setup_calls: list = []
    warmup_calls: list = []
    shutdown_calls: list = []
    cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    w._shutdown_serial_workers()
    assert shutdown_calls == [True]
    # Idempotent: second call is a no-op.
    w._shutdown_serial_workers()
    assert shutdown_calls == [True]


def test_validate_endpoint_lock_accepts_clean_class_shape() -> None:
    """A lock dict with class-shape entries (post-#322) validates clean."""
    lock = {
        "functions": [
            {
                "name": "generate",
                "python_name": "generate",
                "class_name": "FluxKlein",
                "archetype": "SerialWorker",
                "kind": "inference",
            },
            {
                "name": "caption",
                "python_name": "caption",
                "class_name": "JoyCaption",
                "archetype": "BatchedWorker",
                "kind": "inference",
                "runtime": "sglang",
            },
        ]
    }
    r = validate_endpoint_lock(lock)
    assert r.ok, r.errors
    assert not r.errors


def test_validate_endpoint_lock_rejects_stale_function_shape() -> None:
    """An entry missing class_name is the pre-#322 function-shape — bake
    must fail with a pointer to the migration guide.
    """
    lock = {"functions": [{"name": "generate", "python_name": "generate"}]}
    r = validate_endpoint_lock(lock)
    assert not r.ok
    assert any("class_name" in e or "class-shape" in e for e in r.errors)
    assert any("#328" in e for e in r.errors), \
        "Migration error must point at #328 for tenants to find the guide"


def test_validate_endpoint_lock_rejects_same_class_slug_collision() -> None:
    """Two methods on the same class slugifying to the same wire route
    silently shadow each other at dispatch — bake must fail loud.
    """
    lock = {
        "functions": [
            {
                "name": "do-thing",
                "python_name": "do_thing",
                "class_name": "C",
                "archetype": "SerialWorker",
                "kind": "inference",
            },
            {
                "name": "do-thing",
                "python_name": "do_other",
                "class_name": "C",
                "archetype": "SerialWorker",
                "kind": "inference",
            },
        ]
    }
    r = validate_endpoint_lock(lock)
    assert not r.ok
    assert any("slugify" in e or "wire route" in e for e in r.errors)


def test_validate_endpoint_lock_warns_on_runtime_on_serial_worker() -> None:
    """`runtime=` only applies to BatchedWorker. On a SerialWorker it's an
    advisory warning (the dispatch path ignores it anyway).
    """
    lock = {
        "functions": [
            {
                "name": "g", "python_name": "g",
                "class_name": "C", "archetype": "SerialWorker", "kind": "inference",
                "runtime": "sglang",
            },
        ]
    }
    r = validate_endpoint_lock(lock)
    assert r.ok
    assert len(r.warnings) >= 1
    assert any("runtime" in w for w in r.warnings)


def test_validate_endpoint_lock_rejects_missing_functions_key() -> None:
    """Defensive check: an empty / wrongly-shaped lock dict fails."""
    r = validate_endpoint_lock({})
    assert not r.ok
    assert any("functions" in e for e in r.errors)


def test_handle_job_request_lookup_finds_serial_spec() -> None:
    """The dispatch-lookup leg in `_handle_job_request` resolves a
    SerialWorker function_name to its `_serial_class_specs` entry.

    Doesn't run the full request (that needs a connected scheduler stream);
    just confirms the dispatch table lookup is wired in.
    """
    setup_calls: list = []
    warmup_calls: list = []
    shutdown_calls: list = []
    cls = _make_serial_class(setup_calls, warmup_calls, shutdown_calls)
    w = _bare_worker()
    w._register_endpoint_class(cls, cls.__gen_worker_endpoint_spec__)
    # The bare worker doesn't have a scheduler / outgoing queue, so we
    # can't call `_handle_job_request` directly. But we can verify the
    # lookup conditions match what `_handle_job_request` would see.
    assert w._request_specs.get("generate") is None
    assert w._training_specs.get("generate") is None
    assert w._batched_specs.get("generate") is None
    assert w._serial_class_specs.get("generate") is not None
