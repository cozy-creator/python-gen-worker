from __future__ import annotations

import hashlib
from typing import Annotated, List

import msgspec

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelArtifacts, ModelRef, ModelRefSource as Src
from gen_worker.types import Asset


class GenerateInput(msgspec.Struct):
    prompt: str
    images: List[Asset] = msgspec.field(default_factory=list)


class GenerateOutput(msgspec.Struct):
    ok: bool
    prompt: str
    model_path: str
    input_sha256: List[str]
    output: Asset


def _sha256_file(asset: Asset, max_bytes: int = 5 * 1024 * 1024) -> str:
    data = asset.read_bytes(max_bytes=max_bytes)
    return hashlib.sha256(data).hexdigest()

@worker_function(ResourceRequirements(requires_gpu=False))
def generate(
    ctx: ActionContext,
    artifacts: Annotated[ModelArtifacts, ModelRef(Src.DEPLOYMENT, "stub-model")],
    payload: GenerateInput,
) -> GenerateOutput:
    prompt = payload.prompt
    model_path = str(artifacts.root_dir)

    # Inputs: ensure assets are materialized; the worker runtime populates local_path.
    in_sums: List[str] = []
    for a in payload.images:
        if not a.exists():
            raise ValueError("input asset missing")
        in_sums.append(_sha256_file(a))

    output_text = f"prompt={prompt}\nmodel_path={model_path}\ninputs={len(payload.images)}\n"
    out_asset = ctx.save_bytes(
        f"runs/{ctx.run_id}/outputs/result.txt",
        output_text.encode("utf-8"),
    )
    return GenerateOutput(
        ok=True,
        prompt=prompt,
        model_path=model_path,
        input_sha256=in_sums,
        output=out_asset,
    )
