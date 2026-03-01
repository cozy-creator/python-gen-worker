from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch  # type: ignore
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["seed_everything"]
