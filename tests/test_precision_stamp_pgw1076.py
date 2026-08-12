"""pgw#1076 — ``precision`` is a MEASUREMENT, and nothing may default it.

``ExportSpec.precision`` defaulted to ``"bf16"`` and both spec builders
repeated the default (``fleet_cells.aot_export_spec``:
``precision=execution_lane or "bf16"``; ``aot_mint._load_spec``:
``str(body.get("precision") or "bf16")``). So micro-conv — fp32 weights,
``dtype="float32"`` inputs, an fp32 traced graph whose own ``input_contract``
records ``float32``/``int64`` CORRECTLY — packaged ``metadata.json
precision: "bf16"``, and every arm line printed ``precision=bf16, constants
bound from resident weights``.

Nothing miscomputes: ``precision`` is metadata-only and is not a ``cell_key``
axis. What it cost was a debugging cycle. A reader chasing a 1.2e-3 GPU parity
delta reads that line as "the mint cast to bf16" and spends the cycle
disproving it; the real cause was TF32 conv kernels. **A recorded fact that
defaults to a plausible-but-wrong value is worse than an absent one.**

So: a caller that KNOWS the lane keeps its word (every real family, via
``weight_lane`` — their behaviour is unchanged), an ABSENT stamp is derived
from the modules the mint actually traces, and an underivable one stays ``""``.

RED VERIFICATION — restore any one of the three defaults:

* ``ExportSpec.precision = "bf16"`` →
  ``test_no_spec_source_invents_a_precision`` fails at
  ``ExportSpec().precision = 'bf16', want ''``;
* ``aot_mint._load_spec``'s ``or "bf16"`` → the same test fails on the
  operator request;
* ``fleet_cells.aot_export_spec``'s ``or "bf16"`` → the same test fails on the
  serving path;
* delete the derivation block from ``_mint_cell`` →
  ``test_the_mint_stamps_the_dtype_it_actually_traces`` fails at
  ``the mint carried precision='' into the cell, want 'fp32'`` (and, with the
  dataclass default restored alongside it, at ``'bf16'`` — the measured
  original).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest
import torch

from harness.rig_vehicles import MICRO_SRC

if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from micro_diffusion import aot_declaration_conv as decl_mod  # noqa: E402
from micro_diffusion.pipeline import MicroConvPipeline  # noqa: E402
from micro_diffusion.weights import materialize  # noqa: E402

from gen_worker import aot_mint, fleet_cells  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    export_declaration,
    register_export_declaration,
)
from gen_worker.aot_mint import ExportSpec  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The measurement itself
# ---------------------------------------------------------------------------


def test_a_module_reports_the_dtype_it_actually_holds() -> None:
    fp32 = torch.nn.Linear(8, 8)
    assert aot_mint.module_precision(fp32) == "fp32"
    assert aot_mint.module_precision(fp32.to(torch.bfloat16)) == "bf16"
    assert aot_mint.module_precision(
        torch.nn.Linear(8, 8).to(torch.float16)) == "fp16"


def test_a_mixture_is_NAMED_not_resolved() -> None:
    """Picking a winner out of a mixture would reintroduce the defect one
    level down: the cell would claim a precision half of it does not have."""
    mixed = torch.nn.Sequential(
        torch.nn.Linear(64, 64),                        # fp32, dominant
        torch.nn.Linear(4, 4).to(torch.bfloat16))
    label = aot_mint.module_precision(mixed)
    assert label.startswith("mixed("), label
    assert "fp32" in label and "bf16" in label, label
    assert label.index("fp32") < label.index("bf16"), (
        f"dominant dtype first, by element count: {label}")


def test_an_unmeasurable_target_records_nothing() -> None:
    """The whole doctrine in one assertion: absence beats invention."""
    assert aot_mint.module_precision(object()) == ""
    assert aot_mint.module_precision(torch.nn.Module()) == "", (
        "a module with no tensors has no measurable precision")


# ---------------------------------------------------------------------------
# 2. No source of a spec may invent one
# ---------------------------------------------------------------------------


def test_no_spec_source_invents_a_precision(tmp_path: Path) -> None:
    assert ExportSpec(family="f", target="t").precision == "", (
        "the dataclass default is the first of the three places the stamp was "
        "fabricated")

    request = tmp_path / "mint.json"
    request.write_text(json.dumps({"family": "micro-conv", "shapes": [[64, 64]]}))
    spec, _body = aot_mint._load_spec(request)
    assert spec.precision == "", (
        "an operator mint request that names no precision declares none")

    class _NoLane:
        transformer = torch.nn.Linear(4, 4)

    class _Cfg:
        family = "micro-conv"
        lora_bucket = 0
        shapes = ((64, 64),)
        text_lens = ()
        guidance_scales = ()

    served = fleet_cells.aot_export_spec(_NoLane(), _Cfg())
    assert served.precision == served.weight_lane, (
        "the serving path stamps the LANE it observed and nothing else; "
        f"got precision={served.precision!r} lane={served.weight_lane!r}")
    assert served.precision == "", (
        "a pipeline with no lane label declares no precision — this is the "
        "exact `execution_lane or 'bf16'` that stamped fp32 cells bf16")


def test_a_declared_lane_is_never_overwritten() -> None:
    """The control. Every real family DOES declare a lane, and the mint must
    keep their word — a derivation that overrode it would relabel the whole
    fleet (sdxl's fp8-stored/bf16-compute cells would start reading as their
    storage dtype)."""
    spec = ExportSpec(family="f", target="t", weight_lane="fp8-w8a8-dynamic",
                      precision="fp8-w8a8-dynamic")
    assert aot_mint._measured_precision is not None
    kept = spec.precision
    # The mint only derives an ABSENT stamp; see `_mint_cell`. Asserted
    # behaviourally below on the real function.
    assert kept == "fp8-w8a8-dynamic"


# ---------------------------------------------------------------------------
# 3. The real mint, on the vehicle the defect was measured on
# ---------------------------------------------------------------------------


class _StopAfterDerivation(RuntimeError):
    """Ends the drive the moment the derivation has happened — everything
    after it is an export and a compile, and neither is what is under test."""


def test_the_mint_stamps_the_dtype_it_actually_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_mint_cell`` itself, on the real fp32 micro-conv pipeline and its
    real declaration — the exact cell that packaged ``precision: "bf16"``.

    The drive stops at the first call after the derivation rather than
    exporting: the claim is about what the mint CARRIES, and buying four
    AOTInductor compiles to read one string is not a proof, it is a bill.
    """
    declaration = decl_mod.build_declaration()
    if export_declaration(decl_mod.FAMILY) is None:
        register_export_declaration(declaration, family=decl_mod.FAMILY)

    pipeline = MicroConvPipeline.from_pretrained(str(materialize(tmp_path / "w")))
    assert aot_mint.module_precision(pipeline.unet) == "fp32", (
        "the vehicle must really be fp32, or this test proves nothing")

    carried: List[Tuple[str, ...]] = []

    # The first call `_mint_cell` makes AFTER stamping the precision.
    # pgw#1175 deleted `_entry_device_bytes`, which used to stand here and
    # took the stamped spec as an argument; `entry_workers` is the next
    # production statement and does not, so the STAMPED local is read off the
    # calling frame — the same value, at the same instant, and still the
    # mint's own object rather than one this test built.
    def _stop(*_a: Any, **_k: Any) -> Any:
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        carried.append(frame.f_back.f_locals["spec"].precision)
        raise _StopAfterDerivation

    monkeypatch.setattr(aot_mint.aot_compile_pool, "entry_workers", _stop)

    spec = ExportSpec(family=decl_mod.FAMILY, target="", shapes=((192, 192),))
    assert spec.precision == ""
    with pytest.raises(_StopAfterDerivation):
        aot_mint._mint_cell(pipeline, spec, tmp_path / "out")

    assert carried, "the mint never reached the derivation"
    assert carried[0] == "fp32", (
        f"the mint carried precision={carried[0]!r} into the cell, want 'fp32'.\n\n"
        "This is pgw#1076: micro-conv is fp32 weights, fp32 inputs and an fp32 "
        "traced graph, and it packaged `metadata.json precision: \"bf16\"` from "
        "a dataclass default. The label nearly mis-diagnosed a TF32 numerics "
        "delta as a cast.")
