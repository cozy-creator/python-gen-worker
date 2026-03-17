from __future__ import annotations

from gen_worker.injection import InjectionSpec, ModelRef, ModelRefSource as Src
from gen_worker.worker import RequestContext, Worker


class _DummyModel:
    last_source: str = ""
    last_kwargs: dict = {}

    @classmethod
    def from_pretrained(cls, source: str, **kwargs):
        cls.last_source = str(source)
        cls.last_kwargs = dict(kwargs)
        return cls()


class _DownloaderStub:
    def __init__(self, local_path: str) -> None:
        self.local_path = local_path
        self.calls: list[tuple[str, str]] = []

    def download(self, model_ref: str, cache_dir: str) -> str:
        self.calls.append((model_ref, cache_dir))
        return self.local_path


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    w._model_manager = None
    w._model_cache = None
    w._custom_runtime_locks = {}
    w._custom_runtime_cache = {}
    w._downloader = None
    return w


def test_non_diffusers_hf_ref_normalizes_without_downloader() -> None:
    w = _bare_worker()
    ctx = RequestContext("run-hf-no-downloader")
    inj = InjectionSpec(param_name="model", param_type=_DummyModel, model_ref=ModelRef(Src.FIXED, "joycaption"))

    _ = Worker._resolve_injected_value(w, ctx, _DummyModel, "hf:owner/repo@main", inj)  # type: ignore[arg-type]

    assert _DummyModel.last_source == "owner/repo"
    assert _DummyModel.last_kwargs.get("revision") == "main"


def test_non_diffusers_hf_ref_uses_downloader_path_when_available() -> None:
    w = _bare_worker()
    dl = _DownloaderStub("/tmp/cozy-model-cache/hf-owner-repo-main")
    w._downloader = dl
    ctx = RequestContext("run-hf-downloader")
    inj = InjectionSpec(param_name="model", param_type=_DummyModel, model_ref=ModelRef(Src.FIXED, "joycaption"))

    _ = Worker._resolve_injected_value(w, ctx, _DummyModel, "hf:owner/repo@main", inj)  # type: ignore[arg-type]

    assert dl.calls
    assert dl.calls[0][0] == "hf:owner/repo@main"
    assert _DummyModel.last_source == "/tmp/cozy-model-cache/hf-owner-repo-main"
