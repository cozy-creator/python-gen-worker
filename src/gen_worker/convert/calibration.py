"""Calibration-policy metadata + enforcement helper."""

from __future__ import annotations

from typing import Literal
import logging

CalibrationPolicy = Literal["required", "beneficial", "unsupported"]

VALID_POLICIES: frozenset[str] = frozenset({"required", "beneficial", "unsupported"})


CalibrationAction = Literal["calibrate", "skip", "dummy"]
"""What the tenant should do at runtime:

- ``"calibrate"`` — run the calibration forward loop against the dataset.
- ``"skip"`` — weight-only path; no forward_loop.
- ``"dummy"`` — calibration with the built-in smoke pool. Used ONLY in
  tests / CI; production submits never reach this branch because callers
  don't set ``allow_dummy=True``.
"""


def resolve_calibration_action(
    policy: CalibrationPolicy,
    *,
    has_dataset: bool,
    skip_calibration: bool = False,
    allow_dummy: bool = False,
    scheme: str = "",
) -> CalibrationAction:
    """Decide whether to calibrate, skip, or use a dummy pool for one scheme."""

    _log = logging.getLogger(__name__)
    label = f"[{scheme}]" if scheme else ""

    if policy == "required":
        if has_dataset:
            return "calibrate"
        if allow_dummy:
            _log.warning(
                "calibration%s: allow_dummy=True — running with built-in smoke "
                "pool. DO NOT SHIP.", label,
            )
            return "dummy"
        raise ValueError(
            f"calibration{label}: scheme requires a calibration dataset "
            f"(policy='required'). Supply a calibration dataset, or set "
            f"allow_dummy_calibration=True on the spec for smoke tests. "
            f"See docs/calibration-dataset-schema.md for the dataset shape."
        )

    if policy == "beneficial":
        if has_dataset:
            if skip_calibration:
                _log.warning(
                    "calibration%s: skip_calibration=True — running weight-only "
                    "even though a dataset was supplied. Expect measurable "
                    "quality drop vs a calibrated run.", label,
                )
                return "skip"
            return "calibrate"
        if allow_dummy:
            _log.warning(
                "calibration%s: no dataset supplied but allow_dummy=True — "
                "running with built-in smoke pool. DO NOT SHIP.", label,
            )
            return "dummy"
        if skip_calibration:
            _log.warning(
                "calibration%s: skip_calibration=True — running weight-only. "
                "Expect measurable quality drop vs a calibrated run.", label,
            )
            return "skip"
        raise ValueError(
            f"calibration{label}: scheme has policy='beneficial' but no "
            f"calibration dataset was supplied. Default is calibrate — "
            f"silently falling back to weight-only would ship uncalibrated "
            f"weights to invokers who didn't realize the difference. Pass "
            f"a calibration dataset, or set skip_calibration=True on the spec "
            f"to opt out explicitly."
        )

    if policy == "unsupported":
        if has_dataset:
            raise ValueError(
                f"calibration{label}: scheme is weight-only (policy="
                f"'unsupported') — a calibration dataset is not used. Drop "
                f"the calibration dataset, or switch to a calibrated "
                f"quantization recipe (e.g. nvfp4)."
            )
        return "skip"

    raise ValueError(f"calibration{label}: unknown policy {policy!r}")


__all__ = [
    "CalibrationAction",
    "CalibrationPolicy",
    "VALID_POLICIES",
    "resolve_calibration_action",
]
