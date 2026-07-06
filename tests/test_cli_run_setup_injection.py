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
