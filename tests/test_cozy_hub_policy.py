import sys
from types import SimpleNamespace

from gen_worker.cozy_hub_policy import default_resolve_preferences, detect_worker_capabilities


def test_default_preferences_order() -> None:
    prefs = default_resolve_preferences()
    assert prefs["file_type_preference"] == ["flashpack", "safetensors"]
    assert prefs["quantization_preference"][:2] == ["fp8", "bf16"]
    assert prefs["file_layout_preference"] == ["diffusers"]


def test_detect_worker_capabilities_without_torch(monkeypatch) -> None:
    # Ensure torch import fails.
    monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
    caps = detect_worker_capabilities()
    assert "installed_libs" in caps.to_dict()


def test_detect_worker_capabilities_with_fake_torch(monkeypatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability():
            return (8, 9)

    fake_torch = SimpleNamespace(
        __version__="2.10.0",
        version=SimpleNamespace(cuda="12.8"),
        cuda=_FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    caps = detect_worker_capabilities()
    assert caps.cuda_version == "12.8"
    assert caps.gpu_sm == 89
    assert caps.torch_version == "2.10.0"
