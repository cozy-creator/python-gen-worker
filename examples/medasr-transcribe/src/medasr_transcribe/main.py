from __future__ import annotations

from typing import Annotated, Any

import msgspec
import numpy as np
import soundfile as sf
import soxr
import torch
from transformers import AutoModelForCTC, AutoProcessor

from gen_worker import ActionContext, Asset, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

_MODEL_KEY = "medasr"


class MedASRInput(msgspec.Struct):
    audio: Asset


class MedASROutput(msgspec.Struct):
    text: str


@worker_function(ResourceRequirements())
def medasr_transcribe(
    ctx: ActionContext,
    model: Annotated[AutoModelForCTC, ModelRef(Src.DEPLOYMENT, _MODEL_KEY)],
    processor: Annotated[AutoProcessor, ModelRef(Src.DEPLOYMENT, _MODEL_KEY)],
    payload: MedASRInput,
) -> MedASROutput:
    if payload.audio.local_path is None:
        raise RuntimeError("audio.local_path missing")

    if ctx.is_canceled():
        raise InterruptedError("canceled")

    device = next(model.parameters()).device
    speech, sample_rate = sf.read(payload.audio.local_path, always_2d=False, dtype="float32")
    if speech.ndim > 1:
        # Mix down multi-channel audio to mono.
        speech = np.mean(speech, axis=1)
    if sample_rate != 16000:
        speech = soxr.resample(speech, sample_rate, 16000)
        sample_rate = 16000
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    with torch.inference_mode():
        try:
            token_ids = model.generate(**inputs)
        except Exception:
            logits = model(**inputs).logits
            token_ids = torch.argmax(logits, dim=-1)

    if ctx.is_canceled():
        raise InterruptedError("canceled")

    try:
        text = processor.batch_decode(token_ids, skip_special_tokens=True)[0]
    except Exception:
        tokenizer = getattr(processor, "tokenizer", processor)
        text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]

    return MedASROutput(text=text)
