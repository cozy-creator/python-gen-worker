"""pgw#1518: a model_index.json library entry may be a diffusers PIPELINE
SUBMODULE, not an importable module. Every sd15 checkpoint on the hub names
one, and the streaming skeleton refused all of them."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from gen_worker.serving.streaming.skeleton import SkeletonError, _resolve


def test_pipeline_submodule_library_resolves() -> None:
    """`stable_diffusion` is not importable as a top-level module; it is a
    diffusers pipeline submodule. This is the exact entry every sd15
    model_index.json carries, and the boot that found it died here."""
    cls = _resolve("stable_diffusion", "StableDiffusionSafetyChecker")
    assert cls.__name__ == "StableDiffusionSafetyChecker"
    with pytest.raises(ImportError):
        __import__("stable_diffusion")


def test_ordinary_module_libraries_still_resolve() -> None:
    assert _resolve("transformers", "CLIPTextModel").__name__ == "CLIPTextModel"
    assert _resolve("diffusers", "UNet2DConditionModel").__name__ == "UNet2DConditionModel"


def test_unknown_library_still_refuses_by_name() -> None:
    """A genuinely absent library must still be a typed refusal — the fallback
    widens what resolves, it must not turn a miss into a silent None."""
    with pytest.raises(SkeletonError) as caught:
        _resolve("no_such_library_anywhere", "Thing")
    assert "no_such_library_anywhere" in str(caught.value)


def test_known_library_unknown_class_refuses_by_name() -> None:
    with pytest.raises(SkeletonError) as caught:
        _resolve("diffusers", "NoSuchClassInThisVersion")
    assert "NoSuchClassInThisVersion" in str(caught.value)
