"""A family's platform knob must admit its own DISTILLED checkpoint's recipe."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from gen_worker.models import (
    AnimaDefaults,
    ErnieDefaults,
    Krea2Defaults,
    Knob,
    decode_defaults,
)

FAMILIES = [
    pytest.param(
        AnimaDefaults,
        {"steps": {"default": 10}, "guidance": {"default": 1.0}},
        10,
        1.0,
        (30, 1.5),
        id="anima-turbo-10-steps-cfg-off",
    ),
    pytest.param(
        Krea2Defaults,
        {"steps": {"default": 8}, "guidance": {"default": 0.0, "hi": 0.0}},
        8,
        0.0,
        (20, 1.0),
        id="krea2-tdm-turbo-8-steps-guidance-pinned-0",
    ),
    pytest.param(
        ErnieDefaults,
        {"steps": {"default": 8}, "guidance": {"default": 1.0}},
        8,
        1.0,
        (1, 1.5),
        id="ernie-turbo-8-steps-cfg-1",
    ),
]


@pytest.mark.parametrize("cls,row,steps,guidance,naive_floors", FAMILIES)
def test_shipped_bounds_preserve_the_distilled_row(
    cls: type[Any],
    row: dict[str, object],
    steps: int,
    guidance: float,
    naive_floors: tuple[int, float],
) -> None:
    """GREEN: the shipped platform envelope round-trips the distilled recipe."""
    decoded = decode_defaults(cls, row, model_name=cls.__name__)
    assert decoded.steps.default == steps, (
        f"{cls.__name__}: the platform steps floor rewrote this family's own "
        f"distilled recipe from {steps} to {decoded.steps.default}"
    )
    assert decoded.guidance.default == guidance, (
        f"{cls.__name__}: the platform guidance floor rewrote this family's own "
        f"distilled recipe from {guidance} to {decoded.guidance.default}"
    )
    lo, hi = decoded.guidance.lo, decoded.guidance.hi
    assert lo is None or hi is None or lo <= hi, (
        f"{cls.__name__}: guidance decoded to the EMPTY range [{lo}, {hi}] — the "
        f"Flux1/schnell defect. The platform floor must admit the distilled "
        f"checkpoint's pinned value."
    )


@pytest.mark.parametrize("cls,row,steps,guidance,naive_floors", FAMILIES)
def test_the_naive_port_would_corrupt_the_distilled_row(
    cls: type[Any],
    row: dict[str, object],
    steps: int,
    guidance: float,
    naive_floors: tuple[int, float],
) -> None:
    """RED: copying the base handler's wire floors demonstrably breaks the row."""
    step_floor, guidance_floor = naive_floors
    naive = msgspec.defstruct(
        f"Naive{cls.__name__}",
        [
            (
                "steps",
                Knob[int],
                Knob(28, lo=step_floor, hi=100, name="steps"),
            ),
            (
                "guidance",
                Knob[float],
                Knob(4.0, lo=guidance_floor, hi=15.0, name="guidance"),
            ),
        ],
        frozen=True,
    )
    decoded: Any = decode_defaults(naive, row, model_name=f"naive-{cls.__name__}")
    lo, hi = decoded.guidance.lo, decoded.guidance.hi
    corrupted = (
        decoded.steps.default != steps
        or decoded.guidance.default != guidance
        or (lo is not None and hi is not None and lo > hi)
    )
    assert corrupted, (
        f"{cls.__name__}: the naive floors {naive_floors} did NOT corrupt the "
        f"distilled row, so the GREEN assertion above proves nothing for this "
        f"family. Either the row or the floors are no longer the real ones — "
        f"re-derive both from the endpoint before trusting this suite."
    )


DECLARED_LANE_BOUNDS: dict[type[Any], dict[str, list[tuple[str, float, float]]]] = {
    Krea2Defaults: {
        "steps": [("raw main.py:313", 20, 80), ("turbo main.py:340", 1, 16)],
        "guidance": [("raw main.py:316", 1.0, 10.0),
                     ("turbo pins _TURBO_GUIDANCE main.py:95", 0.0, 0.0)],
    },
    AnimaDefaults: {
        "steps": [("base main.py:313", 30, 50),
                  ("turbo pins _TURBO_STEPS main.py:133", 10, 10)],
        "guidance": [("base main.py:314", 1.5, 10.0),
                     ("turbo pins _TURBO_CFG main.py:134", 1.0, 1.0)],
    },
    ErnieDefaults: {
        "steps": [("base main.py:235", 1, 100), ("turbo main.py:254", 1, 16)],
        "guidance": [("base main.py:238", 1.5, 15.0),
                     ("turbo pins main.py:458", 1.0, 1.0)],
    },
}


@pytest.mark.parametrize(
    "cls", list(DECLARED_LANE_BOUNDS), ids=lambda c: c.__name__
)
def test_platform_knob_is_the_widest_declared_envelope(
    cls: type[Any],
) -> None:
    """The platform knob must admit EVERY lane the family declares."""
    shipped = cls()
    for knob_name, lanes in DECLARED_LANE_BOUNDS[cls].items():
        widest_lo = min(lo for _, lo, _ in lanes)
        widest_hi = max(hi for _, _, hi in lanes)
        knob = getattr(shipped, knob_name)
        assert knob.lo <= widest_lo, (
            f"{cls.__name__}.{knob_name}: platform lo={knob.lo} is ABOVE the "
            f"widest declared floor {widest_lo}, so a lane's own row will be "
            f"clamped UP. Declared lanes: {lanes}"
        )
        assert knob.hi >= widest_hi, (
            f"{cls.__name__}.{knob_name}: platform hi={knob.hi} is BELOW the "
            f"widest declared ceiling {widest_hi}, so a lane's own row will be "
            f"clamped DOWN. Declared lanes: {lanes}"
        )
