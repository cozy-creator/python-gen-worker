"""Neutral dispatch orders.

The wire-reading head ``executor.handle_run_job`` projects its message into
these values at the boundary. The driver and every shared helper read ONLY
these: ``pb.RunJob`` never crosses into shared machinery, which is what makes
the head replaceable without touching the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SlotOrder:
    """One dispatched slot binding, as far as the head resolved it.

    ``ref`` is a canonical tensorhub-CAS ref or ``""`` (unbound — the driver
    drops an optional slot, and the head may leave a required slot for the
    code-declared default).
    """

    ref: str = ""
    inference_defaults: str = ""
    objective: str = ""
    distilled: bool = False
    distilled_status: str = ""


@dataclass(frozen=True)
class AdapterOrder:
    """One LoRA overlay riding a slot's base model."""

    ref: str
    weight: float = 1.0
    inference_defaults: str = ""


__all__ = ["AdapterOrder", "SlotOrder"]
