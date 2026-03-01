from __future__ import annotations

from pathlib import Path
import json
from typing import Iterator, Mapping, Protocol


class NamedParamModule(Protocol):
    def named_parameters(self) -> Iterator[tuple[str, object]]:
        ...


class OptimizerLike(Protocol):
    def state_dict(self) -> Mapping[str, object]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, object]) -> object:
        ...


def save_trainable_module_checkpoint(
    *,
    module: NamedParamModule,
    optimizer: OptimizerLike | None,
    output_dir: str,
    step: int,
    final: bool,
    model_name_or_path: str = "",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Persist trainable module parameters + optimizer state to a checkpoint directory.

    This is intentionally low-level and runtime-agnostic so endpoint code can keep
    checkpoint hooks thin while runtime keeps upload cadence/IO ownership.
    """

    import torch

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainable: dict[str, torch.Tensor] = {}
    for name, value in module.named_parameters():
        requires_grad = bool(getattr(value, "requires_grad", False))
        if not requires_grad:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        trainable[str(name)] = value.detach().cpu()
    if not trainable:
        raise ValueError("no trainable parameters found while saving checkpoint")

    fmt = "lora_safetensors_v1"
    primary_path = out_dir / "lora.safetensors"
    try:
        from safetensors.torch import save_file as save_safetensors

        save_safetensors(trainable, str(primary_path))
    except Exception:
        fmt = "lora_torch_v1"
        primary_path = out_dir / "lora.pt"
        torch.save(trainable, primary_path)

    optimizer_path = out_dir / "optimizer.pt"
    if optimizer is not None:
        torch.save(optimizer.state_dict(), optimizer_path)

    payload: dict[str, object] = {
        "format": fmt,
        "step": int(step),
        "final": bool(final),
        "primary_path": str(primary_path),
        "model_name_or_path": str(model_name_or_path),
    }
    if optimizer is not None:
        payload["optimizer_path"] = str(optimizer_path)
    if metadata:
        payload.update({str(k): v for (k, v) in metadata.items()})
    (out_dir / "checkpoint_meta.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def load_trainable_module_checkpoint(
    *,
    module: NamedParamModule,
    optimizer: OptimizerLike | None,
    checkpoint_dir: str,
) -> None:
    import torch

    ckpt_dir = Path(checkpoint_dir)
    safetensors_path = ckpt_dir / "lora.safetensors"
    torch_path = ckpt_dir / "lora.pt"

    state: Mapping[str, torch.Tensor]
    if safetensors_path.exists():
        from safetensors.torch import load_file as load_safetensors

        raw = load_safetensors(str(safetensors_path))
        state = {str(k): v for (k, v) in raw.items()}
    elif torch_path.exists():
        raw = torch.load(torch_path, map_location="cpu")
        if isinstance(raw, dict):
            state = {str(k): v for (k, v) in raw.items() if isinstance(v, torch.Tensor)}
        else:
            state = {}
    else:
        return

    for name, value in module.named_parameters():
        requires_grad = bool(getattr(value, "requires_grad", False))
        if not requires_grad:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        src = state.get(str(name))
        if src is None:
            continue
        value.data.copy_(src.to(device=value.device, dtype=value.dtype))

    optimizer_path = ckpt_dir / "optimizer.pt"
    if optimizer is not None and optimizer_path.exists():
        raw_optimizer_state = torch.load(optimizer_path, map_location="cpu")
        if isinstance(raw_optimizer_state, dict):
            optimizer.load_state_dict(raw_optimizer_state)


__all__ = [
    "load_trainable_module_checkpoint",
    "save_trainable_module_checkpoint",
]
