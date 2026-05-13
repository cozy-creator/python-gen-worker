"""Discovery emits the new `functions[].bindings.<param>` manifest shape.

Golden-manifest tests for:

- Fixed-pick binding (no override).
- Fixed-pick binding + .allow_override.
- Dispatch binding (no override).
- Dispatch binding + .allow_override.
- Multi-param injection (fixed + dispatch + overridable).

Reference: `progress.json` issue #9 (decorator-table-model-bindings).
"""

from __future__ import annotations

from typing import Literal

import msgspec

from gen_worker import Repo, RequestContext, Resources, dispatch, inference_function
from gen_worker.discovery.discover import _extract_function_metadata


class _PipelineA:
    pass


class _PipelineB:
    pass


# --- Payload + output structs reused across tests ---------------------------


class _BasicPayload(msgspec.Struct):
    prompt: str


class _DispatchPayload(msgspec.Struct):
    variant: Literal["nf4", "int8"]
    prompt: str
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1


class _MultiPayload(msgspec.Struct):
    variant: Literal["depth", "canny"]
    prompt: str


class _Out(msgspec.Struct):
    ok: bool


# --- Tests ------------------------------------------------------------------


def test_fixed_binding_no_override_manifest_shape() -> None:
    @inference_function(
        resources=Resources(requires_gpu=True, min_vram_gb=22.0),
        models={"pipeline": Repo("acme/flux").flavor("bf16")},
    )
    def generate_bf16(
        ctx: RequestContext, pipeline: _PipelineA, payload: _BasicPayload,
    ) -> _Out:
        return _Out(ok=True)

    md = _extract_function_metadata(generate_bf16, module_name="tests.t")
    assert md["name"] == "generate-bf16"
    assert md["resources"] == {"requires_gpu": True, "min_vram_gb": 22.0}
    assert md["bindings"] == {
        "pipeline": {
            "kind": "fixed",
            "ref": "acme/flux",
            "flavor": "bf16",
            "tag": "prod",
            "allow_override": False,
            "pipeline_classes": ["test_discovery_bindings._PipelineA"],
        },
    }


def test_fixed_binding_with_override_emits_pipeline_classes() -> None:
    @inference_function(
        resources=Resources(requires_gpu=True, min_vram_gb=22.0),
        models={
            "pipeline": Repo("acme/flux")
            .flavor("bf16")
            .allow_override(_PipelineA, _PipelineB),
        },
    )
    def generate_overridable(
        ctx: RequestContext, pipeline: _PipelineA, payload: _BasicPayload,
    ) -> _Out:
        return _Out(ok=True)

    md = _extract_function_metadata(generate_overridable, module_name="tests.t")
    binding = md["bindings"]["pipeline"]
    assert binding["kind"] == "fixed"
    assert binding["allow_override"] is True
    assert set(binding["pipeline_classes"]) == {
        "test_discovery_bindings._PipelineA",
        "test_discovery_bindings._PipelineB",
    }


def test_dispatch_binding_no_override_manifest_shape() -> None:
    @inference_function(
        resources=Resources(
            requires_gpu=True,
            min_vram_gb=14.0,
            vram_must_fit="full_model",
            vram_scales_with=("width", "height", "num_images_per_prompt"),
        ),
        models={
            "pipeline": dispatch(
                field="variant",
                table={
                    "nf4": Repo("acme/flux").flavor("nf4"),
                    "int8": Repo("acme/flux").flavor("int8"),
                },
            ),
        },
    )
    def generate_bnb(
        ctx: RequestContext, pipeline: _PipelineA, payload: _DispatchPayload,
    ) -> _Out:
        return _Out(ok=True)

    md = _extract_function_metadata(generate_bnb, module_name="tests.t")
    binding = md["bindings"]["pipeline"]
    assert binding["kind"] == "dispatch"
    assert binding["field"] == "variant"
    assert binding["allow_override"] is False
    assert set(binding["table"].keys()) == {"nf4", "int8"}
    # Each table entry has ref + flavor + tag.
    nf4 = binding["table"]["nf4"]
    assert nf4["ref"] == "acme/flux"
    assert nf4["flavor"] == "nf4"
    assert nf4["tag"] == "prod"


def test_dispatch_binding_with_override() -> None:
    @inference_function(
        resources=Resources(requires_gpu=True, min_vram_gb=12.0),
        models={
            "pipeline": dispatch(
                field="variant",
                table={
                    "depth": Repo("acme/cn-depth").flavor("bf16"),
                    "canny": Repo("acme/cn-canny").flavor("bf16"),
                },
            ).allow_override(_PipelineA),
        },
    )
    def generate(
        ctx: RequestContext, pipeline: _PipelineA, payload: _MultiPayload,
    ) -> _Out:
        return _Out(ok=True)

    md = _extract_function_metadata(generate, module_name="tests.t")
    binding = md["bindings"]["pipeline"]
    assert binding["allow_override"] is True
    assert binding["pipeline_classes"] == ["test_discovery_bindings._PipelineA"]


def test_multi_param_independent_bindings() -> None:
    """Three injected params, three independent override policies."""

    @inference_function(
        resources=Resources(requires_gpu=True, min_vram_gb=22.0),
        models={
            "pipeline": Repo("acme/flux").flavor("nf4"),
            "adapter": Repo("acme/lora-realism").allow_override(_PipelineB),
            "controlnet": dispatch(
                field="variant",
                table={
                    "depth": Repo("acme/cn-depth").flavor("bf16"),
                    "canny": Repo("acme/cn-canny").flavor("bf16"),
                },
            ).allow_override(_PipelineA),
        },
    )
    def generate_multi(
        ctx: RequestContext,
        pipeline: _PipelineA,
        adapter: _PipelineB,
        controlnet: _PipelineA,
        payload: _MultiPayload,
    ) -> _Out:
        return _Out(ok=True)

    md = _extract_function_metadata(generate_multi, module_name="tests.t")
    bindings = md["bindings"]
    assert set(bindings.keys()) == {"pipeline", "adapter", "controlnet"}
    assert bindings["pipeline"]["allow_override"] is False
    assert bindings["adapter"]["allow_override"] is True
    assert bindings["controlnet"]["allow_override"] is True
    assert bindings["controlnet"]["kind"] == "dispatch"
