"""The serve-path output-integrity floor: nothing is uploaded unlooked-at.

WHY THIS EXISTS (ie#615 / ie#634 / pgw#1094). Production minimax-h3 0.3.8
VAE-decoded pure NOISE and uploaded it, on requests that were billed and
settled. Every check we had passed, because the "proof" was ffprobe container
metadata plus a billing row. Metadata is not pixels. This module is the one
place on the worker that looks at the PIXELS before an output can be banked as
a success.

WHAT IT IS. Two numbers over the frames the endpoint already holds in memory:

* ``adjacent_frame_corr`` — the MEDIAN Pearson correlation of a handful of
  adjacent grey frame pairs spread across the clip. Real video is highly
  self-similar frame to frame even through motion and cuts; independently
  sampled noise correlates with nothing. The floor is MEASURED, not guessed:
  production noise 0.29, real renders 0.92-0.99, so :data:`NOISE_CORR_FLOOR`
  = 0.6 sits in an empty middle. Taking the MEDIAN over a spread of pairs is
  what makes a hard CUT safe — a cut drives one pair to ~0 while the rest
  stay high.
* ``frame_std_min`` — the smallest per-frame grey standard deviation over the
  sampled frames. Zero is a constant-fill frame: blank/black output, which
  correlates with nothing and must not be reported as a correlation failure.

A non-finite pixel (NaN/Inf reaching the decoder's output) is its own verdict:
it is the observable, save-path-side form of the latent NaN the tracker asked
about, at the one seam that sees every endpoint.

WHAT IT COSTS. Single-digit milliseconds on a full 121-frame 1344x768 clip,
because only ~2*``pairs`` frames are ever touched and each is decimated to
~96 rows BEFORE the float conversion (the expensive pass then runs on ~1/s^2
of the pixels). Naive full-resolution was 555 ms on the same clip. The
decimation is free in ACCURACY because the statistic is a coarse whole-frame
correlation — cheap and blind are one fact (see the scope note).

SCOPE — READ THIS BEFORE QUOTING A PASS. This floor catches NOISE and BLANK.
It is NOT a quality gate, and it is specifically blind to the melt class:
smearing REMOVES high-frequency temporal variation, so a melted/over-smoothed
render scores HIGHER here than a clean one (cozy-eval measured 0.956 melted vs
0.916 clean). ``tests/test_output_integrity_pgw1094.py`` pins that inversion as
an executable assertion so nobody re-reads this gate as a quality verdict.

WHO IT JUDGES (th#1771). SERVED outputs, and only those — see :func:`judged`.
A boot-warmup / mint warm forward uploads nothing and banks nothing, and its
input is the derived warm payload (``WARMUP_TEXT`` + a flat mid-gray 128px
PNG), so a flat output there is the expected answer to a flat question.

REFERENCE IMPLEMENTATION. ``cozy_eval.integrity`` / ``cozy_eval.metrics.temporal``
(ce#10). The numbers, the floors and the decimation are shared with it
deliberately: the eval half judges candidates offline, this half refuses them
online, and a disagreement between the two would be a bug in one of them.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence

import msgspec

#: Adjacent ``(t, t+1)`` pairs sampled across a buffered clip.
INTEGRITY_PAIRS = 5

#: Median adjacent-frame grey correlation below this reads as NOISE. MEASURED
#: (ie#615/ie#634): VAE-decoded noise 0.29, real renders 0.92-0.99.
NOISE_CORR_FLOOR = 0.6

#: Per-frame grey standard deviation (on [0, 1] greys) below this reads as
#: BLANK — a constant-fill frame.
BLANK_STD_FLOOR = 0.01

#: Working HEIGHT the statistic is computed at, reached by integer decimation.
#: MEASURED to cost nothing in accuracy: on a 1344x768 clip panning 1 px/frame
#: the median correlation is 0.9755 at full resolution and 0.9789 at 96 rows,
#: while noise stays ~0.00 either way.
INTEGRITY_TARGET_H = 96

#: Streaming budget: how many adjacent pairs the chunk collector may compute
#: before it halves what it kept and doubles its chunk stride. Keeps the gate
#: bounded on a clip whose length is not known until the producer is done.
STREAM_PAIR_BUDGET = 2 * INTEGRITY_PAIRS

PASS = "pass"
NOISE = "noise"
BLANK = "blank"
NONFINITE = "nonfinite"
UNMEASURED = "unmeasured"

#: Appended to every PASS. A green integrity line can never be read as a
#: quality pass — the melt class scores HIGHER here than a clean render.
SCOPE_NOTE = (
    "SCOPE: this floor catches NOISE and BLANK output only. It is not a "
    "quality gate — a melted/over-smoothed render scores HIGHER on "
    "adjacent_frame_corr than a clean one."
)

_LUMA = (0.299, 0.587, 0.114)


class OutputIntegrity(msgspec.Struct, frozen=True, kw_only=True):
    """One output's integrity answer, and the numbers behind it."""

    verdict: str = UNMEASURED
    adjacent_frame_corr: float = float("nan")
    frame_std_min: float = float("nan")
    corr_series: tuple[float, ...] = ()
    frames_sampled: int = 0
    detail: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True only on an explicit PASS. UNMEASURED is not a pass."""
        return self.verdict == PASS

    @property
    def rejected(self) -> bool:
        """True when the pixels themselves failed — the request must not bank."""
        return self.verdict in (NOISE, BLANK, NONFINITE)

    def summary(self) -> str:
        return (
            f"integrity {self.verdict} (adjacent_frame_corr "
            f"{self.adjacent_frame_corr:.3f}, frame_std_min "
            f"{self.frame_std_min:.5f}, frames_sampled {self.frames_sampled}"
            + (f", {self.detail}" if self.detail else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# the numeric core — decimated grey planes, one pass
# ---------------------------------------------------------------------------


def _frame_count(source: Any) -> int:
    import numpy as np

    if isinstance(source, (list, tuple)):
        return len(source)
    return int(np.asarray(source).shape[0])


def _pair_starts(total: int, count: int) -> List[int]:
    """Uniformly spread start indices ``k`` for consecutive ``(k, k+1)`` pairs."""
    if total < 2:
        return []
    last = total - 2
    if count >= last + 1:
        return list(range(last + 1))
    if count == 1:
        return [0]
    step = last / (count - 1)
    return sorted({round(i * step) for i in range(count)})


def _grey(frame: Any, target_h: int) -> Any:
    """One frame -> a decimated grey plane in [0, 1], float32.

    Decimation happens BEFORE the float conversion, on a numpy VIEW, so the
    expensive pass runs on ~1/stride^2 of the pixels. This is the whole reason
    the gate is affordable on the serve path.
    """
    import numpy as np

    a = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[-1] != 3):
        raise ValueError(f"expected (H, W) or (H, W, 3) pixels, got shape {a.shape}")
    stride = max(1, a.shape[0] // max(target_h, 1))
    small = a[::stride, ::stride]
    f = small.astype(np.float32)
    if a.dtype == np.uint8 or (f.size and float(f.max()) > 1.5):
        f = f / 255.0
    if f.ndim == 3:
        f = f @ np.asarray(_LUMA, np.float32)
    return f


def _grey_planes(source: Any, indices: Sequence[int], target_h: int) -> List[Any]:
    """Decimated grey planes for JUST ``indices`` — nothing else is touched."""
    import numpy as np

    if isinstance(source, (list, tuple)):
        picked: List[Any] = [source[i] for i in indices]
    else:
        arr = np.asarray(source)  # a view for an ndarray; nothing is copied
        picked = [arr[i] for i in indices]
    return [_grey(p, target_h) for p in picked]


def _corr(a: Any, b: Any, sa: float, sb: float) -> float:
    if sa < 1e-6 or sb < 1e-6:
        return 0.0  # a flat frame correlates with nothing
    x, y = a.ravel(), b.ravel()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sa * sb))


def _verdict(
    corrs: Sequence[float],
    std_min: float,
    *,
    frames_sampled: int,
    nonfinite: bool,
    corr_floor: float,
    std_floor: float,
    t0: float,
    detail: str = "",
) -> OutputIntegrity:
    import numpy as np

    median = float(np.median(corrs)) if corrs else float("nan")
    series = tuple(float(c) for c in corrs)
    if nonfinite:
        verdict, why = NONFINITE, "a decoded frame carries NaN/Inf pixels"
    elif std_min < std_floor:
        verdict, why = BLANK, (
            f"a sampled frame carries no spatial signal "
            f"(std {std_min:.5f} < {std_floor})"
        )
    elif corrs and median < corr_floor:
        verdict, why = NOISE, (
            f"median adjacent-frame correlation {median:.3f} < floor "
            f"{corr_floor} — consecutive frames are unrelated, which is what "
            f"VAE-decoded noise looks like"
        )
    else:
        verdict, why = PASS, SCOPE_NOTE
    return OutputIntegrity(
        verdict=verdict,
        adjacent_frame_corr=median,
        frame_std_min=std_min,
        corr_series=series,
        frames_sampled=frames_sampled,
        detail=f"{detail}; {why}" if detail else why,
        seconds=round(time.monotonic() - t0, 5),
    )


def _unmeasured(reason: str, t0: float) -> OutputIntegrity:
    return OutputIntegrity(
        verdict=UNMEASURED,
        detail=f"integrity UNMEASURED: {reason}",
        seconds=round(time.monotonic() - t0, 5),
    )


def check_frames(
    frames: Any,
    *,
    pairs: int = INTEGRITY_PAIRS,
    corr_floor: float = NOISE_CORR_FLOOR,
    std_floor: float = BLANK_STD_FLOOR,
    target_h: int = INTEGRITY_TARGET_H,
) -> OutputIntegrity:
    """Judge a buffered clip (or a single frame). Milliseconds, no reference.

    ``frames`` is anything the encode path already holds: a ``(F, H, W, 3)``
    numpy array, a ``(H, W, 3)`` single frame, a list of PIL images or arrays,
    or a CPU torch tensor. A clip too short to have an adjacent pair is judged
    on the BLANK/NONFINITE half alone — that half is fully measured there, so
    it is an honest verdict, not an unmeasured one.
    """
    t0 = time.monotonic()
    try:
        import numpy as np

        arr = frames
        if hasattr(arr, "detach"):  # a CPU torch tensor
            arr = arr.detach().cpu().numpy()
        if not isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[None]
        total = _frame_count(arr)
        if total < 1:
            return _unmeasured("no frames", t0)
        starts = _pair_starts(total, pairs)
        needed = sorted(set(starts) | {k + 1 for k in starts} | ({0} if not starts else set()))
        greys = dict(zip(needed, _grey_planes(arr, needed, target_h)))
        stds = {i: float(g.std()) for i, g in greys.items()}
        nonfinite = any(not bool(np.isfinite(g).all()) for g in greys.values())
        corrs = [
            _corr(greys[k], greys[k + 1], stds[k], stds[k + 1]) for k in starts
        ]
        detail = "" if starts else f"{total} frame(s): no adjacent pair to correlate"
        return _verdict(
            corrs, min(stds.values()), frames_sampled=len(needed),
            nonfinite=nonfinite, corr_floor=corr_floor, std_floor=std_floor,
            t0=t0, detail=detail,
        )
    except Exception as exc:  # an unjudgeable output is UNMEASURED, never a pass
        return _unmeasured(f"{type(exc).__name__}: {exc}", t0)


def check_image(
    image: Any,
    *,
    std_floor: float = BLANK_STD_FLOOR,
    target_h: int = INTEGRITY_TARGET_H,
) -> OutputIntegrity:
    """Judge a single image: BLANK and NONFINITE only.

    The correlation half needs two frames and there is only one, so it is not
    reported — the same floor applies to images trivially, and this is what
    "trivially" means. It is fully measured, so a flat image REJECTS here.
    """
    return check_frames(image, std_floor=std_floor, target_h=target_h)


class StreamCollector:
    """Judge a clip that arrives in CHUNKS, without ever rebuffering it.

    The streaming encode seam (gw#476) hands the encoder one decoded chunk at
    a time and the total length is unknown until the producer is done, so the
    buffered "spread five pairs over the clip" sampling cannot be used. This
    collector instead takes the full buffered spread from the FIRST chunk
    (so a single-chunk stream is bit-identical to :func:`check_frames`) and
    one adjacent pair from each chunk after it — thinned by halving the kept
    series and doubling the chunk stride whenever the computed count reaches
    :data:`STREAM_PAIR_BUDGET`. The pairs therefore stay spread across the
    whole clip and the cost stays bounded no matter how long it is.

    Each chunk MUST be judged inside the producer loop: the staged host buffer
    a chunk borrows is only valid until the next iteration step.
    """

    def __init__(
        self,
        *,
        pairs: int = INTEGRITY_PAIRS,
        corr_floor: float = NOISE_CORR_FLOOR,
        std_floor: float = BLANK_STD_FLOOR,
        target_h: int = INTEGRITY_TARGET_H,
    ) -> None:
        self._pairs = pairs
        self._corr_floor = corr_floor
        self._std_floor = std_floor
        self._target_h = target_h
        self._corrs: List[float] = []
        self._std_min = float("inf")
        self._nonfinite = False
        self._frames_sampled = 0
        self._chunk_index = 0
        self._stride = 1
        self._seconds = 0.0
        self._unmeasurable: Optional[str] = None

    def observe(self, chunk: Any) -> None:
        """Fold one decoded chunk into the running statistic."""
        index = self._chunk_index
        self._chunk_index += 1
        if index % self._stride:
            return
        t0 = time.monotonic()
        try:
            self._observe(chunk, first=index == 0)
        except Exception as exc:
            if self._unmeasurable is None:
                self._unmeasurable = f"{type(exc).__name__}: {exc}"
        self._seconds += time.monotonic() - t0

    def _observe(self, chunk: Any, *, first: bool) -> None:
        import numpy as np

        arr = chunk
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        if not isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[None]
        total = _frame_count(arr)
        if total < 1:
            return
        if first:
            starts = _pair_starts(total, self._pairs)
        else:
            mid = max(0, total // 2 - 1)
            starts = [mid] if total >= 2 else []
        needed = sorted(set(starts) | {k + 1 for k in starts} | ({0} if not starts else set()))
        greys = dict(zip(needed, _grey_planes(arr, needed, self._target_h)))
        self._frames_sampled += len(needed)
        stds = {i: float(g.std()) for i, g in greys.items()}
        self._std_min = min([self._std_min, *stds.values()])
        if any(not bool(np.isfinite(g).all()) for g in greys.values()):
            self._nonfinite = True
        for k in starts:
            self._corrs.append(_corr(greys[k], greys[k + 1], stds[k], stds[k + 1]))
        if len(self._corrs) >= STREAM_PAIR_BUDGET:
            del self._corrs[1::2]  # keep every other — the spread survives
            self._stride *= 2

    def verdict(self) -> OutputIntegrity:
        t0 = time.monotonic() - self._seconds
        if self._unmeasurable is not None and not self._corrs:
            return _unmeasured(self._unmeasurable, t0)
        if self._frames_sampled == 0:
            return _unmeasured("no frames were streamed", t0)
        detail = "" if self._corrs else "no adjacent pair to correlate"
        if self._unmeasurable is not None:
            detail = (detail + "; " if detail else "") + \
                f"partial: {self._unmeasurable}"
        return _verdict(
            self._corrs, self._std_min, frames_sampled=self._frames_sampled,
            nonfinite=self._nonfinite, corr_floor=self._corr_floor,
            std_floor=self._std_floor, t0=t0, detail=detail,
        )


# ---------------------------------------------------------------------------
# enforcement — the one place a verdict becomes a refusal
# ---------------------------------------------------------------------------


def enforce(result: OutputIntegrity, *, ref: str, kind: str) -> None:
    """Bank the verdict and refuse the output if the pixels failed.

    ``kind`` is ``"video"`` / ``"image"`` — it rides the event detail so a
    reject is countable per output kind hub-side.

    REJECT raises :class:`~gen_worker.api.errors.OutputIntegrityError`, which
    maps FATAL: the request never reaches ``succeeded`` and nothing is
    uploaded. UNMEASURED serves — a screen that could not run is a confession,
    not a verdict — but it still emits its event, because a fail-soft outcome
    that a hub decision may depend on must never be only a log line (pgw#760).
    PASS is silent: one event per served request would be a row per render for
    no decision.
    """
    if result.ok:
        return
    # Deferred: `activity` pulls protobuf + psutil, and the io/save modules on
    # the `import gen_worker` path must not.
    from . import activity
    from .api.errors import OutputIntegrityError

    activity.emit_event(
        activity.KIND_OUTPUT_INTEGRITY,
        f"{kind} {ref}: {result.summary()}",
        phase=result.verdict,
        duration_ms=int(result.seconds * 1000),
    )
    if result.rejected:
        raise OutputIntegrityError(result.verdict, ref=ref, kind=kind,
                                   summary=result.summary())


def judged(ctx: Any) -> bool:
    """Whether this context's outputs are subject to the floor.

    th#1771: this is the SERVE-path floor — its whole contract is that nothing
    is UPLOADED unlooked-at and that a bad render cannot BANK as a success. A
    boot-warmup / mint warm forward uploads nothing and banks nothing: its
    output is discarded by construction, and its INPUT is the derived warm
    payload — `WARMUP_TEXT` and a flat mid-gray 128px PNG. A degenerate output
    from a degenerate input is the expected result there, not a defect, and
    refusing it turned a valid warm into a FATAL on the paying request that
    happened to wake the pod (minimax-h3 `reference-to-video`, measured).
    """
    return not bool(getattr(ctx, "boot_warmup", False))


def guard_frames(frames: Any, *, ref: str) -> OutputIntegrity:
    """:func:`check_frames` + :func:`enforce` — the video save-path one-liner."""
    result = check_frames(frames)
    enforce(result, ref=ref, kind="video")
    return result


def guard_image(image: Any, *, ref: str) -> OutputIntegrity:
    """:func:`check_image` + :func:`enforce` — the image save-path one-liner."""
    result = check_image(image)
    enforce(result, ref=ref, kind="image")
    return result


__all__ = [
    "BLANK",
    "BLANK_STD_FLOOR",
    "INTEGRITY_PAIRS",
    "INTEGRITY_TARGET_H",
    "NOISE",
    "NOISE_CORR_FLOOR",
    "NONFINITE",
    "PASS",
    "SCOPE_NOTE",
    "STREAM_PAIR_BUDGET",
    "UNMEASURED",
    "OutputIntegrity",
    "StreamCollector",
    "check_frames",
    "check_image",
    "enforce",
    "guard_frames",
    "guard_image",
    "judged",
]
