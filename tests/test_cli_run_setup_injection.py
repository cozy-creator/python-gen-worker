"""setup() receives LOADED models (not path strings) + fatal errors carry class+detail.

GEN_WORKER_FORBID_CPU_OFFLOAD: this box exports =1. The injection tests drive
`_load_injected_model` through the real `place_pipeline` call with a ~0-byte
`_FakePipe` stub (no real weights, no CUDA touch beyond the veto check itself),
so they scope the veto OFF explicitly (test_oom_degraded_ladder.py pattern).
"""
import gen_worker.cli.run as cli_run
from gen_worker.executor import _map_exception
from gen_worker.pb import worker_scheduler_pb2 as pb


class _FakePipe:
    loaded_from = None

    @classmethod
    def from_pretrained(cls, path, **kw):
        obj = cls()
        obj.loaded_from = path
        return obj

    @classmethod
    def from_single_file(cls, path, **kw):
        obj = cls()
        obj.loaded_from = path
        return obj


def test_run_setup_loads_annotated_slot(tmp_path, monkeypatch):
    (tmp_path / "model_index.json").write_text("{}")
    seen = {}

    class EP:
        def setup(self, pipeline: _FakePipe) -> None:
            seen["pipeline"] = pipeline

    monkeypatch.setattr(cli_run, "_INJECTED_CACHE", {})
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "0")  # see module docstring
    cli_run.run_setup(EP(), {"pipeline": str(tmp_path)})
    assert isinstance(seen["pipeline"], _FakePipe)
    assert seen["pipeline"].loaded_from == str(tmp_path)


def test_run_setup_str_slot_receives_snapshot_path(tmp_path, monkeypatch):
    # gw#416: a str-typed slot gets the snapshot PATH (executor contract) —
    # never a loaded DiffusionPipeline.
    from pathlib import Path

    (tmp_path / "model_index.json").write_text("{}")
    seen = {}

    class EP:
        def setup(self, pipeline: str, aux: Path) -> None:
            seen["pipeline"] = pipeline
            seen["aux"] = aux

    monkeypatch.setattr(cli_run, "_INJECTED_CACHE", {})
    cli_run.run_setup(EP(), {"pipeline": str(tmp_path), "aux": str(tmp_path)})
    assert seen["pipeline"] == str(tmp_path)
    assert isinstance(seen["pipeline"], str)
    assert seen["aux"] == Path(tmp_path)
    assert isinstance(seen["aux"], Path)


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
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "0")  # see module docstring
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


def test_map_exception_bare_valueerror_is_fatal_not_invalid():
    # pgw#514/P9: PIL/numpy/tenant code raise ValueError for internal bugs;
    # blaming the client (INVALID, never retried) hid worker-side failures.
    status, msg = _map_exception(ValueError("could not broadcast input array"))
    assert status == pb.JOB_STATUS_FATAL
    assert msg.startswith("ValueError:")
    # Typed validation errors keep INVALID.
    from gen_worker.api.errors import ValidationError

    status, _ = _map_exception(ValidationError("bad field"))
    assert status == pb.JOB_STATUS_INVALID
    import msgspec

    status, _ = _map_exception(msgspec.ValidationError("bad payload"))
    assert status == pb.JOB_STATUS_INVALID


def test_map_exception_redacts_worker_paths_in_fatal():
    # pgw#514/P8: FATAL "ExcClass: first-line" must not leak pod filesystem
    # layout (FileNotFoundError carries absolute paths).
    exc = FileNotFoundError(
        "[Errno 2] No such file or directory: '/tmp/tensorhub-cache/cas/blobs/ab/cd/abcd'")
    status, msg = _map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert msg.startswith("FileNotFoundError")
    assert "/tmp/tensorhub-cache" not in msg
    assert "[redacted]" in msg
    # URL paths and owner/repo refs survive (diagnosability).
    status, msg = _map_exception(RuntimeError(
        "download failed for https://r2.example.com/cas/ab12 (ref acme/model:prod)"))
    assert "r2.example.com/cas/ab12" in msg
    assert "acme/model:prod" in msg
