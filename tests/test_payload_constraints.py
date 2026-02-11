from __future__ import annotations

from typing import Annotated

import msgspec

from gen_worker.payload_constraints import Clamp, apply_payload_constraints


class P(msgspec.Struct):
    steps: Annotated[int | float, Clamp(20, 50, cast="int")] = 25
    cfg: Annotated[float, Clamp(0.0, 20.0, cast="float")] = 7.0


def test_apply_payload_constraints_clamps_and_rounds() -> None:
    p = P(steps=1000, cfg=-1.0)
    apply_payload_constraints(p)
    assert p.steps == 50
    assert p.cfg == 0.0

    p2 = P(steps=20.49, cfg=7.25)
    apply_payload_constraints(p2)
    assert p2.steps == 20
    assert p2.cfg == 7.25

    p3 = P(steps=20.5, cfg=7.25)
    apply_payload_constraints(p3)
    assert p3.steps == 21  # half-up

