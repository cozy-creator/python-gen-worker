from __future__ import annotations

from typing import Annotated

import msgspec
import torch
from transformers import AutoModelForCTC, AutoProcessor

from gen_worker import Asset, RequestContext, ResourceRequirements, inference_function
from gen_worker import io as gw_io
from gen_worker.api.injection import ModelRef, ModelRefSource as Src

_MODEL_KEY = "medasr"


class MedASRInput(msgspec.Struct):
    audio: Asset


class MedASROutput(msgspec.Struct):
    text: str


@inference_function(resources=ResourceRequirements())
def medasr_transcribe(
    ctx: RequestContext,
    model: Annotated[AutoModelForCTC, ModelRef(Src.FIXED, _MODEL_KEY)],
    processor: Annotated[AutoProcessor, ModelRef(Src.FIXED, _MODEL_KEY)],
    payload: MedASRInput,
) -> MedASROutput:
    ctx.raise_if_canceled()

    device = next(model.parameters()).device
    speech, sample_rate = gw_io.read_audio(payload.audio, target_sample_rate=16000)
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    with torch.inference_mode():
        try:
            token_ids = model.generate(**inputs)
        except Exception:
            logits = model(**inputs).logits
            token_ids = torch.argmax(logits, dim=-1)

    ctx.raise_if_canceled()

    try:
        text = processor.batch_decode(token_ids, skip_special_tokens=True)[0]
    except Exception:
        tokenizer = getattr(processor, "tokenizer", processor)
        text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]

    return MedASROutput(text=text)
