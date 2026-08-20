"""The floor judges RENDERS, and a trace produces no render (pgw#1522).

Two output paths existed and only one was trace-aware. `ctx.save_image` /
`ctx.save_video` are STUBBED under `TraceRequestContext` — they return a
`trace://` ref and upload nothing — so every endpoint that saves through the
context derived fine. The module-level `gw_io.write_video` / `write_image`
writers ran the output-integrity floor FIRST, before delegating to that same
stub, so the first endpoint to reach the output step through them died:

    OutputIntegrityError: blank: video video failed the output-integrity floor
      (adjacent_frame_corr 0.000, frame_std_min 0.00000)

The verdict is TRUE. Under a hollow session every parameter is a fake tensor
carrying no bytes, so the frames are EXACTLY zero — the signature of
virtualized weights, not a marginal render. minimax-h3 has no choice about the
path: only `gw_io.write_video` muxes the soundtrack into one `VideoAsset`.

The floor is not weakened and its threshold is untouched. `judged()` — the one
predicate that already answered "whose outputs are subject to this?" for the
boot-warmup case — now answers it for the trace case too, so the two paths
cannot disagree.
"""

from __future__ import annotations

from typing import Any

import pytest

from gen_worker.api.errors import OutputIntegrityError
from gen_worker.output_integrity import check_frames, enforce, judged
from gen_worker.release.trace_context import TraceRequestContext


def _trace_ctx() -> TraceRequestContext:
    return TraceRequestContext(lane=None, checkpoint_ref="trace:x", step_budget=1)


class _Serve:
    """A serving context as far as the floor is concerned."""

    boot_warmup = False


def test_a_trace_context_is_not_judged() -> None:
    assert judged(_trace_ctx()) is False


def test_a_serve_context_is_STILL_judged() -> None:
    """The floor stays armed where it exists to be armed."""

    assert judged(_Serve()) is True
    assert judged(object()) is True


def test_the_boot_warmup_exemption_is_untouched() -> None:
    """th#1771's exemption shares the predicate and must not regress."""

    class Warm:
        boot_warmup = True

    assert judged(Warm()) is False


def test_the_FLOOR_ITSELF_still_rejects_a_blank_render() -> None:
    """The guard is not weakened — only its audience is stated.

    A blank clip is still refused with the same verdict and the same
    threshold; nothing about this change makes a bad render bankable.
    """

    numpy = pytest.importorskip("numpy")

    blank = numpy.zeros((10, 32, 32, 3), dtype=numpy.uint8)
    result = check_frames(blank)
    assert result.rejected
    with pytest.raises(OutputIntegrityError) as refusal:
        enforce(result, ref="ref", kind="video")
    assert "blank" in str(refusal.value)


def test_the_marker_is_PRIVATE_so_no_author_can_branch_on_it() -> None:
    """There is deliberately no `ctx.is_trace` on the author surface.

    Author code branching on trace-ness corrupts compilation coverage, which
    is why that spelling was deleted. This fact is the platform's, about where
    the output goes — so it must not arrive as a public member.
    """

    ctx = _trace_ctx()
    assert ctx._outputs_discarded is True
    assert not hasattr(ctx, "is_trace")
    assert "_outputs_discarded".startswith("_")


def test_the_module_level_writer_reaches_the_ctx_stub_under_a_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H3's actual path, end to end: gw_io.write_image -> ctx.save_bytes.

    The floor used to fire BEFORE the encode and the delegation, so the trace
    stub was never reached. This drives the REAL `gw_io` writer with a blank
    image — the only thing a hollow session can produce — and asserts it lands
    on the stub instead of raising.
    """

    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    from gen_worker import io as gw_io

    ctx = _trace_ctx()
    blank = Image.fromarray(numpy.zeros((32, 32, 3), dtype=numpy.uint8))

    saved: dict[str, Any] = {}
    original = ctx.save_bytes

    def record(ref: str, data: bytes, **kwargs: Any) -> Any:
        saved["reached"] = True
        return original(ref, data, **kwargs)

    monkeypatch.setattr(ctx, "save_bytes", record)

    asset = gw_io.write_image(ctx, "clip", blank)

    assert saved.get("reached") is True
    assert str(getattr(asset, "ref", "")).startswith("trace://")


def test_the_same_writer_STILL_refuses_a_blank_render_at_SERVE(
    tmp_path: Any,
) -> None:
    """The other half: the writer must not have gone soft.

    Same call, same blank image, a context that is not a trace — the floor
    fires exactly as before. Without this the fix could have been "delete the
    guard" and every test above would still pass.
    """

    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    from gen_worker import io as gw_io

    class Serve:
        boot_warmup = False

        def save_bytes(self, ref: str, data: bytes, **_: Any) -> Any:
            raise AssertionError("the floor must refuse before the save")

    blank = Image.fromarray(numpy.zeros((32, 32, 3), dtype=numpy.uint8))

    with pytest.raises(OutputIntegrityError):
        gw_io.write_image(Serve(), "clip", blank)
