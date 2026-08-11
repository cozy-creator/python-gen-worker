"""th#1771 part 2: the output-integrity floor is a SERVE-path floor.

pgw#1094's whole contract is *"nothing is uploaded unlooked-at"* and *"it
cannot bank as a successful render"*. A boot-warmup / mint warm forward uploads
nothing and banks nothing — and its INPUT is the derived warm payload, which is
`WARMUP_TEXT` plus a flat mid-gray 128px PNG. For a reference-conditioned model
that input produces a legitimately flat output.

Measured live on minimax-h3 0.4.7 / gen-worker 0.101.0: with part 1's fix the
warm payload finally BUILT, the warm forward ran a full 4-minute denoise, and
`ctx.save_video` then refused its own warm output —

    OutputIntegrityError: blank: video video failed the output-integrity floor
    (integrity blank (adjacent_frame_corr 0.994, frame_std_min 0.00134 …))

— which the worker reported as `JOB_STATUS_FATAL` for the paying request that
happened to wake the pod. Same shape as part 1, one layer further down.

So: judge served outputs, never warm ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from gen_worker.api.errors import OutputIntegrityError
from gen_worker.output_integrity import check_frames, guard_frames, judged


def _blank_clip(frames: int = 8, side: int = 64) -> np.ndarray:
    """The warm payload's own output shape: a constant mid-gray clip."""
    return np.full((frames, side, side, 3), 128, dtype=np.uint8)


class _Ctx:
    def __init__(self, boot_warmup: bool) -> None:
        self.boot_warmup = boot_warmup


def test_blank_is_still_a_reject_verdict() -> None:
    """The judge is unchanged — only WHO it is applied to."""
    assert check_frames(_blank_clip()).rejected
    with pytest.raises(OutputIntegrityError):
        guard_frames(_blank_clip(), ref="video")


def test_a_warm_context_is_not_judged() -> None:
    assert judged(_Ctx(boot_warmup=True)) is False


def test_a_serve_context_is_judged() -> None:
    assert judged(_Ctx(boot_warmup=False)) is True


def test_a_context_without_the_attribute_is_judged() -> None:
    """Fail CLOSED: an unknown context is a serve context."""
    assert judged(object()) is True
