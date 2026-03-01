from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OptimizerBundle:
    optimizer: Any
    scheduler: Any | None = None


def _resolve_params(model_or_params: Any) -> Any:
    if model_or_params is None:
        return None
    params_fn = getattr(model_or_params, "parameters", None)
    if callable(params_fn):
        return params_fn()
    if isinstance(model_or_params, Iterable):
        return model_or_params
    return None


def build_default_adamw_bundle(
    model_or_params: Any,
    *,
    hyperparams: Mapping[str, object] | None = None,
) -> OptimizerBundle:
    """
    Optional torch helper for endpoint authors.

    Returns empty bundle when torch is unavailable or params are unresolved.
    """
    params = _resolve_params(model_or_params)
    if params is None:
        return OptimizerBundle(optimizer=None, scheduler=None)

    hp = dict(hyperparams or {})
    lr = float(hp.get("learning_rate", hp.get("lr", 1e-4)))
    weight_decay = float(hp.get("weight_decay", 0.0))
    betas_raw = hp.get("betas", (0.9, 0.999))
    if isinstance(betas_raw, (tuple, list)) and len(betas_raw) == 2:
        betas = (float(betas_raw[0]), float(betas_raw[1]))
    else:
        betas = (0.9, 0.999)
    warmup_steps = int(hp.get("warmup_steps", 0))
    max_steps = int(hp.get("max_steps", 0))

    try:
        import torch  # type: ignore
    except Exception:
        return OptimizerBundle(optimizer=None, scheduler=None)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = None
    if max_steps > 0:
        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=max(1, warmup_steps),
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, max_steps - warmup_steps),
                    ),
                ],
                milestones=[max(1, warmup_steps)],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_steps))
    return OptimizerBundle(optimizer=optimizer, scheduler=scheduler)


__all__ = ["OptimizerBundle", "build_default_adamw_bundle"]
