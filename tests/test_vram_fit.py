"""The offload ladder is consulted BEFORE placement, and its answer is obeyed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import StableDiffusionPipeline  # noqa: E402

from gen_worker.models import memory as gwmem  # noqa: E402
from gen_worker.serving.context import DeployBinding, LoadContext  # noqa: E402

FIXTURE = "hf-internal-testing/tiny-stable-diffusion-pipe"


def _local_snapshot() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = "models--" + FIXTURE.replace("/", "--")
    for snapshot in sorted(Path(HF_HUB_CACHE).glob(f"{repo}/snapshots/*")):
        if (snapshot / "model_index.json").is_file():
            return snapshot
    pytest.skip(f"{FIXTURE} is not in the local HF cache ({HF_HUB_CACHE})")


def _ctx(device: str) -> "LoadContext[Any]":
    binding = DeployBinding(
        checkpoint_ref="ckpt:tiny@fixture", checkpoint_dir=_local_snapshot()
    )
    return LoadContext(binding=binding, device=device)


def _recording_to(moved: List[Any]) -> Any:
    def to(self: Any, *args: Any, **kwargs: Any) -> Any:
        moved.append(args)
        return self

    return to


def test_the_ladder_is_asked_while_the_pipeline_is_still_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE RED ARM."""
    asked_with_resident: List[float] = []
    real = gwmem.apply_low_vram_config

    def recording(pipeline: Any, **kwargs: Any) -> Any:
        asked_with_resident.append(gwmem.estimate_cuda_resident_gb(pipeline))
        return real(pipeline, **kwargs)

    monkeypatch.setattr(gwmem, "apply_low_vram_config", recording)
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    monkeypatch.setattr(type(pipe), "to", lambda self, *a, **k: self)

    _ctx("cuda")._placed(pipe)

    assert asked_with_resident, (
        "the offload ladder was never consulted on the eager bridge — this is "
        "the pgw#1486 defect: 1785 lines of correct ladder with no caller, and "
        "an SDXL pipeline that OOMs on a card it fits with one rung engaged"
    )
    assert asked_with_resident[0] == 0.0, (
        f"the ladder was consulted with {asked_with_resident[0]:.2f} GiB "
        f"already resident on the card. `select_auto_mode` nets the "
        f"requirement against what is already there (pgw#1025), so asking it "
        f"after placement makes every pipeline 'fit' and selects the most "
        f"memory-hungry rung — measured as model_offload -> vae_only on SDXL"
    )


def test_an_offload_rung_places_its_own_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the ladder answers an offload rung, `_placed` must NOT then move the pipeline to the device."""
    armed: List[str] = []
    def arming(pipeline: Any, **kwargs: Any) -> Any:
        armed.append("model_offload")
        return {"mode": "model_offload"}

    monkeypatch.setattr(gwmem, "apply_low_vram_config", arming)
    moved: List[Any] = []
    ctx = _ctx("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    monkeypatch.setattr(type(pipe), "to", _recording_to(moved))

    ctx._placed(pipe)

    assert armed == ["model_offload"], "the rung was not engaged"
    assert not moved, (
        f"an offload rung was engaged and the bridge moved the pipeline to the "
        f"device anyway ({moved}) — that re-lands the very weights the rung "
        f"just evicted, which is the OOM pgw#1486 exists to remove"
    )


def test_a_resident_rung_still_places_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gwmem, "select_auto_mode", lambda **_: "off")
    moved: List[Any] = []
    ctx = _ctx("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    monkeypatch.setattr(type(pipe), "to", _recording_to(moved))

    ctx._placed(pipe)

    assert moved == [("cuda",)], (
        f"the ladder said this pipeline fits, so the worker's placement "
        f"decision must still be applied; moves seen: {moved}"
    )


def test_a_ladder_that_cannot_size_the_pipeline_does_not_block_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-diffusers `ctx.load` caller reaches this path too."""
    def explode(pipeline: Any, **kwargs: Any) -> Any:
        raise TypeError("this is not a diffusers pipeline")

    monkeypatch.setattr(gwmem, "apply_low_vram_config", explode)
    moved: List[Any] = []
    ctx = _ctx("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    monkeypatch.setattr(type(pipe), "to", _recording_to(moved))

    ctx._placed(pipe)

    assert moved == [("cuda",)]


def test_compile_will_not_arm_over_hook_managed_weights() -> None:
    """The ADMISSION half."""
    sink_calls: List[Any] = []
    def sink(target: Any) -> Any:
        sink_calls.append(target)
        return "ARMED"

    ctx: "LoadContext[Any]" = LoadContext(
        binding=DeployBinding(checkpoint_ref="r", checkpoint_dir=Path(".")),
        compile_sink=sink,
    )
    module = torch.nn.Linear(2, 2)

    ctx._engaged_rung = ""
    assert ctx.compile(module) == "ARMED", (
        "with no offload rung engaged, marking must reach torchcg unchanged — "
        "otherwise this guard has silently disabled compilation everywhere"
    )
    assert sink_calls == [module]

    ctx._engaged_rung = "model_offload"
    assert ctx.compile(module) is module, (
        "an offload rung is engaged and `ctx.compile` still armed a compiled "
        "graph over weights accelerate relocates every forward"
    )
    assert len(sink_calls) == 1, "the sink must not be reached under a rung"


def test_pipeline_device_never_answers_meta_to_endpoint_code() -> None:
    """The bottom rung's own defect."""
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    assert gwmem.install_execution_device_fallback()

    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.to("meta")

    assert pipe.device.type == "meta"

    class _Hook:
        execution_device = torch.device("cpu")

    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component._hf_hook = _Hook()
    assert pipe.device.type == "cpu", (
        "a hook names where the next forward runs, so `pipeline.device` must "
        "report that and not `meta` — this is what the sdxl endpoint's "
        "`torch.Generator(device=model.pipe.device)` builds on"
    )
