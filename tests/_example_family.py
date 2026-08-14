"""A family vocabulary declared the way an ENDPOINT declares one.

the SDK used to ship `SdxlDefaults` / `WanDefaults`, so its own
tests reached for a real family whenever they needed a `GenerationDefaults`
subclass. Family vocabularies now live in the endpoint that owns the family, so
the SDK's tests declare their own — which is also the point being proven: the
registry works for a family the library has never heard of.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from gen_worker.families import GenerationDefaults, family

ExampleScheduler = Literal["euler_a", "dpmpp_2m_karras", "lcm"]


@family("example")
class ExampleDefaults(GenerationDefaults, frozen=True):
    """Checkpoint-kind vocabulary for the synthetic ``example`` family."""

    scheduler: ExampleScheduler = "euler_a"
    steps: int = 28
    guidance: float = 6.0
    negative: str = ""
    max_guidance: Optional[float] = None


@family("example", kind="lora")
class ExampleLoraDefaults(GenerationDefaults, frozen=True):
    """LoRA-kind overlay for the same family: every recipe field is "no opinion"."""

    trigger_words: Tuple[str, ...] = ()
    recommended_weight: Optional[float] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    max_guidance: Optional[float] = None
    scheduler: Optional[ExampleScheduler] = None


__all__ = ["ExampleDefaults", "ExampleLoraDefaults", "ExampleScheduler"]
