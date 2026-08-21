from __future__ import annotations


def refuse_unservable(fps: int) -> int:
    """The author's own refusal, raised from a SIBLING module."""

    if fps == 60:
        raise ValueError(f"fps={fps} is not servable by this checkpoint")
    return fps
