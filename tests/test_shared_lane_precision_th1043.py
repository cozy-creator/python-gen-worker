"""th#1043: joint precision fit for shared-component multi-lane records.

Real qwen-image evidence: two lanes (t2i/edit) share a ~15.5GB text
encoder+VAE, each carries an exclusive ~40GB bf16 transformer. Loading each
lane's precision reactively — one at a time, against whatever free VRAM
happens to be left at that moment — let the FIRST lane consume all headroom
at native precision, starving the second into an offload placement the
shared-component invariant then refuses outright (RetryableError,
"shared-component lanes require resident placement"). The fix decides
precision for the WHOLE group up front, against its COMBINED footprint.
"""

from __future__ import annotations

from gen_worker.executor import _shared_lanes_need_fp8

_GiB = 1024 ** 3

# Real th#1043 shapes: shared text_encoder+vae ~15.5GB, each transformer
# ~40GB bf16 (no fp8 checkpoint exists yet for this family).
_QWEN_SIZES = {
    "t2i": {"text_encoder": 15 * _GiB, "vae": int(0.5 * _GiB), "transformer": 40 * _GiB},
    "edit": {"text_encoder": 15 * _GiB, "vae": int(0.5 * _GiB), "transformer": 40 * _GiB},
}
_SHARED = ["text_encoder", "vae"]


def test_starved_group_forces_fp8_when_it_fits():
    # A100-80GB: ~79GiB usable. Native (15.5 + 40 + 40 = 95.5GiB) doesn't
    # fit; fp8'd transformers (15.5 + 20 + 20 = 55.5GiB) do.
    free = 79 * _GiB
    assert _shared_lanes_need_fp8(_QWEN_SIZES, _SHARED, free) is True


def test_group_that_fits_natively_is_left_alone():
    # A card with enough headroom for both lanes at native precision needs
    # no forcing.
    free = 200 * _GiB
    assert _shared_lanes_need_fp8(_QWEN_SIZES, _SHARED, free) is False


def test_group_that_cannot_fit_even_at_fp8_is_left_to_the_ladder():
    # Too small even halved: forcing fp8 here would just relocate the
    # failure, not fix it — leave it to the existing per-lane ladder.
    free = 30 * _GiB
    assert _shared_lanes_need_fp8(_QWEN_SIZES, _SHARED, free) is False


def test_single_lane_group_never_forces():
    assert _shared_lanes_need_fp8({"t2i": _QWEN_SIZES["t2i"]}, _SHARED, 1 * _GiB) is False


def test_no_shared_components_never_forces():
    assert _shared_lanes_need_fp8(_QWEN_SIZES, [], 1 * _GiB) is False
