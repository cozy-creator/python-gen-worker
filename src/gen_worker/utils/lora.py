from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from ..api.types import LoraSpec

logger = logging.getLogger(__name__)


@runtime_checkable
class LoraCapablePipeline(Protocol):
    def load_lora_weights(self, *args: Any, **kwargs: Any) -> None: ...
    def set_adapters(self, *args: Any, **kwargs: Any) -> None: ...
    def unload_lora_weights(self) -> None: ...


@contextmanager
def load_loras(
    pipeline: LoraCapablePipeline,
    loras: list["LoraSpec"],
    request_id: str = "",
) -> Generator[None, None, None]:
    """Context manager that loads LoRA adapters onto *pipeline* for the duration
    of the ``with`` block, then unloads them.

    Handles:
    - Missing ``alpha`` keys (injects alpha = rank so diffusers doesn't error)
    - Multi-adapter ``set_adapters`` with per-lora weights
    - Guaranteed cleanup via ``unload_lora_weights`` in the finally block

    Usage::

        with load_loras(pipeline, payload.loras, ctx.request_id):
            result = pipeline(...)
    """
    from safetensors.torch import load_file as load_safetensors
    import torch

    loaded_adapters: list[str] = []
    try:
        for i, spec in enumerate(loras):
            if spec.file.local_path is None:
                raise RuntimeError(f"LoRA {i} not materialized (local_path is None)")
            name = spec.adapter_name or f"lora_{i}"
            state_dict = load_safetensors(spec.file.local_path)
            for key in list(state_dict.keys()):
                if "lora_down.weight" in key:
                    alpha_key = key.replace("lora_down.weight", "alpha")
                    if alpha_key not in state_dict:
                        rank = state_dict[key].shape[0]
                        state_dict[alpha_key] = torch.tensor(float(rank))
                        logger.info(
                            "[request_id=%s] injected missing alpha key %r = %d",
                            request_id,
                            alpha_key,
                            rank,
                        )
            pipeline.load_lora_weights(state_dict, adapter_name=name)
            loaded_adapters.append(name)

        if loaded_adapters:
            pipeline.set_adapters(
                loaded_adapters,
                adapter_weights=[s.weight for s in loras],
            )

        yield

    finally:
        if loaded_adapters and hasattr(pipeline, "unload_lora_weights"):
            try:
                pipeline.unload_lora_weights()
            except Exception:
                logger.warning("unload_lora_weights failed; pipeline may have stale adapters", exc_info=True)
