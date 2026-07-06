"""setup() receives LOADED models (not path strings) + fatal errors carry class+detail."""
import gen_worker.cli.run as cli_run
from gen_worker.executor import _map_exception
from gen_worker.pb import worker_scheduler_pb2 as pb


class _FakePipe:
    loaded_from = None

    @classmethod
    def from_pretrained(cls, path, **kw):
        obj = cls(); obj.loaded_from = path; return obj

    @classmethod
    def from_single_file(cls, path, **kw):
        obj = cls(); obj.loaded_from = path; return obj


def test_run_setup_loads_annotated_slot(tmp_path, monkeypatch):
    (tmp_path / "model_index.json").write_text("{}")
    seen = {}

    class EP:
        def setup(self, pipeline: _FakePipe) -> None:
            seen["pipeline"] = pipeline

    monkeypatch.setattr(cli_run, "_INJECTED_CACHE", {})
    cli_run.run_setup(EP(), {"pipeline": str(tmp_path)})
    assert isinstance(seen["pipeline"], _FakePipe)
    assert seen["pipeline"].loaded_from == str(tmp_path)


def test_run_setup_loads_module_layout_slot(tmp_path, monkeypatch):
    # Module-layout repo (root config.json, no model_index.json): e.g. a bare
    # AutoencoderKL repo like madebyollin/sdxl-vae-fp16-fix. Must route through
    # from_pretrained, not the single-file path.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    seen = {}

    class EP:
        def setup(self, vae: _FakePipe) -> None:
            seen["vae"] = vae

    monkeypatch.setattr(cli_run, "_INJECTED_CACHE", {})
    cli_run.run_setup(EP(), {"vae": str(tmp_path)})
    assert isinstance(seen["vae"], _FakePipe)
    assert seen["vae"].loaded_from == str(tmp_path)


def test_map_exception_fatal_carries_class_and_detail():
    status, msg = _map_exception(TypeError("'str' object is not callable"))
    assert status == pb.JOB_STATUS_FATAL
    assert msg.startswith("TypeError:")
    assert "not callable" in msg


def test_map_exception_fatal_sanitizes_secrets():
    status, msg = _map_exception(RuntimeError("boom Bearer abc123"))
    assert status == pb.JOB_STATUS_FATAL
    assert "abc123" not in msg and msg.startswith("RuntimeError")


def test_select_function_base_fn_wins_exact_match_over_variant_attr_matches():
    # Base fn + variants share one attr name; --method <base> must select the
    # base spec instead of erroring ambiguous (ie#348).
    import types

    import msgspec

    from gen_worker import HF, RequestContext, endpoint

    class _In(msgspec.Struct):
        prompt: str

    class _Out(msgspec.Struct):
        result: str

    @endpoint(
        model=HF("o/base", dtype="fp16"),
        variants={"generate_alt": HF("o/alt")},
    )
    class Gen:
        def setup(self, pipeline: str) -> None: ...

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(result="")

    mod = types.SimpleNamespace(Gen=Gen)
    candidates = cli_run._collect_class_methods(mod)
    assert len(candidates) == 2

    base = cli_run._select_function(candidates, cls_name=None, method_name="generate")
    assert base.fn_name == "generate"
    variant = cli_run._select_function(candidates, cls_name=None, method_name="generate_alt")
    assert variant.fn_name == "generate-alt"


def test_map_exception_redacts_presigned_url_but_keeps_context():
    # Presigned-URL download failures used to collapse to "internal error";
    # the message must survive with only the credential material redacted.
    exc = RuntimeError(
        "download failed for https://r2.example.com/cas/ab12?"
        "X-Amz-Credential=AKIAEXAMPLE%2F20260706&X-Amz-Signature=deadbeef123"
    )
    status, msg = _map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert msg.startswith("RuntimeError: download failed")
    assert "r2.example.com" in msg
    assert "AKIAEXAMPLE" not in msg and "deadbeef123" not in msg
    assert "[redacted]" in msg


def test_map_exception_redacts_bare_signature_param():
    status, msg = _map_exception(RuntimeError("PUT 403: Signature=abc123 rejected"))
    assert status == pb.JOB_STATUS_FATAL
    assert "abc123" not in msg and "rejected" in msg
