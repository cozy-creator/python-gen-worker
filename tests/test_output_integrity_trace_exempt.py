from __future__ import annotations

from typing import Any

import pytest

from gen_worker.api.errors import OutputIntegrityError
from gen_worker.output_integrity import check_frames, enforce, judged
from gen_worker.release.trace_context import TraceRequestContext


def _trace_ctx() -> TraceRequestContext:
    return TraceRequestContext(lane=None, checkpoint_ref="trace:x", step_budget=1)


class _Serve:

    boot_warmup = False


def test_a_trace_context_is_not_judged() -> None:
    assert judged(_trace_ctx()) is False


def test_a_serve_context_is_STILL_judged() -> None:
    """The floor stays armed where it exists to be armed."""

    assert judged(_Serve()) is True
    assert judged(object()) is True


def test_the_boot_warmup_exemption_is_untouched() -> None:

    class Warm:
        boot_warmup = True

    assert judged(Warm()) is False


def test_the_FLOOR_ITSELF_still_rejects_a_blank_render() -> None:
    """The guard is not weakened — only its audience is stated."""

    numpy = pytest.importorskip("numpy")

    blank = numpy.zeros((10, 32, 32, 3), dtype=numpy.uint8)
    result = check_frames(blank)
    assert result.rejected
    with pytest.raises(OutputIntegrityError) as refusal:
        enforce(result, ref="ref", kind="video")
    assert "blank" in str(refusal.value)


def test_the_marker_is_PRIVATE_so_no_author_can_branch_on_it() -> None:
    """There is deliberately no `ctx.is_trace` on the author surface."""

    ctx = _trace_ctx()
    assert ctx._outputs_discarded is True
    assert not hasattr(ctx, "is_trace")
    assert "_outputs_discarded".startswith("_")


def test_the_module_level_writer_reaches_the_ctx_stub_under_a_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H3's actual path, end to end: gw_io.write_image -> ctx.save_bytes."""

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
    """The other half: the writer must not have gone soft."""

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
