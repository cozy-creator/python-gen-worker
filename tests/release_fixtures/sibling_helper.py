"""A SIBLING top-level module beside the endpoint's package (pgw#1533).

h3's shape: `minimax_h3.main` is the endpoint, and its fps refusal raises in
`src/cozy_rife.py` — a top-level module at the same source root, not inside
the `minimax_h3` package. A shared helper beside the main package is the
common case, and it was uncovered.
"""

from __future__ import annotations


def refuse_unservable(fps: int) -> int:
    """The author's own refusal, raised from a SIBLING module."""

    if fps == 60:
        raise ValueError(f"fps={fps} is not servable by this checkpoint")
    return fps
