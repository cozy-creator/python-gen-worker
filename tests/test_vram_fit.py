"""The offload ladder is consulted BEFORE placement, and its answer is obeyed.

pgw#1486, measured on a real 7.62 GiB RTX 4070 against the real sdxl endpoint at
1024^2 CFG batch-2. The do-nothing baseline OOMs in 1.74 s inside the unet;
`enable_model_cpu_offload()` alone serves the same request in 26.8 s at a
6.41 GiB peak with an image out. The ladder that picks exactly that rung has
shipped in `models/memory.py` for months and had ZERO callers on the v2 serve
path — and calling it in the wrong ORDER does not help, it inverts the answer:

    select_auto_mode  PRE-placement -> "model_offload"   (serves)
    select_auto_mode POST-placement -> "vae_only"        (OOMs)

because `select_auto_mode` nets the requirement against
`estimate_cuda_resident_gb(pipeline)` (pgw#1025, deliberate and correct: a
pipeline must not be charged twice for bytes already on the card), and
`_placed` had already made every byte resident.

So the property under test is an ORDERING one, and it is written that way: not
"the ladder was called" but "the ladder was called while nothing was resident".
That is checkable with no GPU at all, which is why it runs on every runner
rather than only where the defect was found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import StableDiffusionPipeline  # noqa: E402

from gen_worker.models import memory as gwmem  # noqa: E402
from gen_worker.serving.context import DeployBinding, LoadContext  # noqa: E402

#: The same real tiny pipeline `test_bridge_placement.py` uses. The defect was
#: in what happens to the object `from_pretrained` returns, so a hand-built
#: double that never calls it could not have caught it.
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
    """A `.to` that records where it was asked to go and moves nothing.

    The contract under test is "the bridge does not move an offloaded
    pipeline", and that is observable as a CALL — so the assertions run on
    runners with no CUDA device, which is where this suite mostly lives.
    """
    def to(self: Any, *args: Any, **kwargs: Any) -> Any:
        moved.append(args)
        return self

    return to


# pgw#1486: the ladder is consulted before placement, not after.
def test_the_ladder_is_asked_while_the_pipeline_is_still_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE RED ARM. At master `apply_low_vram_config` is never called at all,
    so `asked_with_resident` stays empty and this fails on the first assert.

    The second assert is the half that a "did we call the ladder?" test would
    miss entirely: calling it one line later still compiles, still logs, and
    still returns a mode — and that mode is the one that OOMs.
    """
    asked_with_resident: List[float] = []
    real = gwmem.apply_low_vram_config

    def recording(pipeline: Any, **kwargs: Any) -> Any:
        asked_with_resident.append(gwmem.estimate_cuda_resident_gb(pipeline))
        return real(pipeline, **kwargs)

    monkeypatch.setattr(gwmem, "apply_low_vram_config", recording)
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    # The real `.to` is stubbed so the ORDERING question is answerable on a
    # runner with no CUDA device — which is most of them, and the property is
    # not a CUDA property.
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


# pgw#1486: an offload rung owns placement; the bridge must not re-move it.
def test_an_offload_rung_places_its_own_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the ladder answers an offload rung, `_placed` must NOT then move
    the pipeline to the device.

    The `.to(device)` is the whole defect: it re-lands exactly the bytes the
    rung just moved off the card. Asserted on the CALL rather than on resulting
    device placement so it runs without a GPU — the contract is "the bridge
    does not move it", and that is what is observable everywhere.
    """
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


# pgw#1452: a pipeline that fits is still placed where the worker said.
def test_a_resident_rung_still_places_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counter-case, so the assertion above cannot pass by never placing
    anything: a pipeline that FITS is still placed on the device, exactly as
    pgw#1452 requires."""
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


# pgw#1486: a ladder that cannot size an object never refuses the load.
def test_a_ladder_that_cannot_size_the_pipeline_does_not_block_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-diffusers `ctx.load` caller reaches this path too. A ladder
    that raises on an object it cannot read must degrade to the pre-pgw#1486
    behaviour — place it whole — never refuse the load."""
    def explode(pipeline: Any, **kwargs: Any) -> Any:
        raise TypeError("this is not a diffusers pipeline")

    monkeypatch.setattr(gwmem, "apply_low_vram_config", explode)
    moved: List[Any] = []
    ctx = _ctx("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    monkeypatch.setattr(type(pipe), "to", _recording_to(moved))

    ctx._placed(pipe)

    assert moved == [("cuda",)]


# pgw#1486: admission check — no compiled graph over relocating weights.
def test_compile_will_not_arm_over_hook_managed_weights() -> None:
    """The ADMISSION half. Under an offload rung accelerate moves a module's
    weights on and off the device per forward, so a compiled graph's bound
    constants dangle — a use-after-free, which on the compiled path is the
    uncatchable SIGSEGV (pgw#1255 leg 2), not an OOM anyone can retry."""
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


# pgw#1486: the bottom rung parked modules on `meta` and broke `pipe.device`.
def test_pipeline_device_never_answers_meta_to_endpoint_code() -> None:
    """The bottom rung's own defect. `enable_sequential_cpu_offload` parks
    modules on `meta`, so `pipeline.device` — the public property endpoint code
    is told to ask, because `ctx.load`'s contract is that authors never name a
    device — answered `meta`, and `torch.Generator(device=pipe.device)` died
    with "META device type not an accelerator" before any image.

    Both directions are asserted, because the fallback must not INVENT a device
    for a pipeline that is genuinely on meta with nothing to onload to.
    """
    pipe = StableDiffusionPipeline.from_pretrained(_local_snapshot())
    assert gwmem.install_execution_device_fallback()

    # EVERY module, not just the big three: `DiffusionPipeline.device` reports
    # the first module it finds, so a leftover cpu-resident safety checker
    # would hide the meta answer and make this test pass for the wrong reason.
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.to("meta")

    # No accelerate hook anywhere: `meta` is the honest answer, and the
    # fallback must terminate rather than recurse through `_execution_device`,
    # which itself ends in `return self.device`.
    assert pipe.device.type == "meta"

    class _Hook:
        execution_device = torch.device("cpu")

    # Every module, because that is what `enable_sequential_cpu_offload` does
    # and what diffusers' `_execution_device` requires: it bails to
    # `self.device` on the FIRST component it finds without a hook.
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component._hf_hook = _Hook()
    assert pipe.device.type == "cpu", (
        "a hook names where the next forward runs, so `pipeline.device` must "
        "report that and not `meta` — this is what the sdxl endpoint's "
        "`torch.Generator(device=model.pipe.device)` builds on"
    )
