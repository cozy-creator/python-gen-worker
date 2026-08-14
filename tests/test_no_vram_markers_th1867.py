"""th#1867 (DESIGN-RULINGS §1.35): the VRAM marker vocabulary is DELETED, and
this is the fence that keeps it deleted.

Paul's ruling in one line: *"We should be able to run any model on any GPU. The
challenge is not IF it can run — it's: is it an EFFICIENT choice?"* So there is
no feasibility gate anywhere on this worker. A card that looks too small is a
card whose best rung sits further down the ladder, and which rung that is gets
MEASURED at load time (``models/memory.select_auto_mode``) rather than declared
by an author who cannot know it.

WHY A FENCE AND NOT JUST A DELETION. §2.4 ruling 4 measured what a partial cut
does: the hub builder folds an absent ``min_vram_gb`` from ``vram_gb``, so
removing one marker and leaving another silently drops a floor — the worst
outcome available. The three had to go together, and they have to STAY gone
together. Each assertion below names the specific way its marker came back
before: a field on ``Resources``, a key on the manifest, a comparison inside
``variant_fit``, a kwarg on the descent, a field on the wire.

A REFUSAL THIS SWEEP TRIED TO KEEP, AND THE TESTS REFUTED. th#1867 first kept a
``cuda_unavailable`` refusal for a GPU-declaring function on a CUDA-less pod,
re-based off the FACT rather than the deleted ``strict_vram``, citing pgw#1212's
record that the CPU serve path has never executed. This repo's own boot tests
said otherwise: every harness endpoint declares ``Resources(gpu=True)`` and
serves on CPU-only CI, and ``test_boot_span_ladder_pgw797`` boots one and runs a
warmup forward through it. Withdrawing a function that demonstrably works is the
opposite of §1.35 amendment 2's bar, so the refusal went and the CPU rung is
planned and SERVED behind a loud warning.

pgw#1212's protection is unchanged and lives where it is true — the REACTIVE
walk in ``models/rung`` still stops one rung short of CPU
(``FLOOR_CPU_RUNG_UNEXECUTABLE``). The last two tests here pin both halves so
neither drifts.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.api.decorators import Resources
from gen_worker.models import rung
from gen_worker.models.hub_policy import (
    FIT_FITS,
    FIT_INCOMPATIBLE,
    TensorhubWorkerCapabilities,
    variant_fit,
)
from gen_worker.models.serve_fit import ServePlan, plan_serve

#: Every spelling the deleted concept ever had on this SDK.
DELETED_FIELDS = ("vram_gb_hint", "min_vram_gb", "strict_vram")

_CUDA = TensorhubWorkerCapabilities(
    cuda_version="12.8", gpu_sm=89, torch_version="2.9.0", installed_libs=[],
)
_NO_CUDA = TensorhubWorkerCapabilities(
    cuda_version="", gpu_sm=0, torch_version="2.9.0", installed_libs=[],
)


# --- the declaration ---------------------------------------------------------

@pytest.mark.parametrize("field", DELETED_FIELDS)
def test_the_marker_is_not_a_field_on_resources(field: str) -> None:
    assert field not in Resources.__struct_fields__, (
        f"`Resources.{field}` is back. §2.4 ruling 4 deleted all three of "
        f"{DELETED_FIELDS} as ONE change because the hub folds an absent "
        "`min_vram_gb` from `vram_gb` — a partial restoration silently "
        "reinstates a buy floor.")


@pytest.mark.parametrize("field", DELETED_FIELDS)
def test_declaring_the_marker_is_a_hard_error_not_a_silent_drop(field: str) -> None:
    """msgspec Structs reject unknown keywords, so an endpoint still carrying
    the old declaration fails at import with the field named — never boots
    having quietly ignored it."""
    kwargs: dict[str, Any] = {field: 24}
    with pytest.raises(TypeError):
        Resources(**kwargs)


def test_nothing_vram_shaped_reaches_the_manifest() -> None:
    """The projection is what the hub actually reads. `min_vram_gb` is the
    exact key `function_requirements.go` folds into `requirement_payload_json`
    -> `req.VRAMGB` -> `MinVRAMGB` -> the GPU candidate filter, so a key
    leaking back here is a buy floor leaking back."""
    manifest = Resources(gpu=True, vcpus=8, ram_gb_hint=64,
                         min_disk_gb=100, compute_capability=8.9).manifest_dict()
    assert not [k for k in manifest if "vram" in k.lower()], manifest
    # The two floors that SURVIVE, asserted positively so this test cannot
    # pass by the projection having broken entirely.
    assert manifest["compute_capability"] == 8.9
    assert manifest["ram_gb"] == 64.0
    assert manifest["min_disk_gb"] == 100.0


# --- the fit verdict ---------------------------------------------------------

@pytest.mark.parametrize("free_gb", [0.0, 0.5, 8.0, 24.0, 80.0, 1e6])
def test_variant_fit_never_consults_the_card_size(free_gb: float) -> None:
    """The deleted comparison was `effective_vram_requirement_gb(hint) <=
    free_vram_gb`. Sweeping free VRAM from an empty card to an absurd one must
    not move the verdict: if it does, a size input has grown back."""
    verdict, _ = variant_fit(Resources(gpu=True), _CUDA, free_gb)
    assert verdict == FIT_FITS


@pytest.mark.parametrize("free_gb", [0.0, 1.0, 24.0, 80.0])
def test_plan_serve_always_serves_on_a_cuda_card(free_gb: float) -> None:
    """0.0 GB free is the real case, not a synthetic one: `gpu_free_mem` is
    genuinely 0 on a saturated card, and pgw#940 made that read as no room
    rather than as an empty card. Under §1.35 no amount of "no room" is a
    refusal — it selects a deeper rung."""
    plan = plan_serve(Resources(gpu=True), _CUDA, free_gb)
    assert plan.serveable, plan.reason
    assert plan.run_mode == "native"


def test_serveplan_carries_no_recommended_card() -> None:
    assert "recommended_vram_gb" not in ServePlan.__dataclass_fields__


# --- the descent -------------------------------------------------------------

@pytest.mark.parametrize("fn", [rung.descend, rung.descent_floor])
def test_no_declaration_argument_on_the_descent(fn: Callable[..., object]) -> None:
    """`strict_vram` truncated this walk before the first host-RAM-touching
    rung. The walk may end only where OUR ladder ends."""
    assert set(inspect.signature(fn).parameters) == {"current"}


def test_the_declaration_truncated_floor_token_is_gone() -> None:
    assert not hasattr(rung, "FLOOR_STRICT_VRAM_TRUNCATED")


# --- the wire ----------------------------------------------------------------

def test_fn_degraded_carries_no_card_size() -> None:
    """FnDegraded is OPERATOR EVIDENCE and every field on it must be measured
    or observed. `recommended_vram_gb`'s only source was the author's own
    hint, which §1.2 measured wrong in both directions on live releases —
    anima declared 8 GB against a 10.6 GiB peak, sdxl declared 20 against a
    proven 9.3 GiB run on a 16 GB A4000 — and the hub then learned a monotone
    buy floor from it (th#1720)."""
    fields = set(pb.FnDegraded.DESCRIPTOR.fields_by_name)
    assert "recommended_vram_gb" not in fields
    # What replaces it: what actually happened, and how much slower.
    assert {"wanted", "ran", "est_latency_multiplier", "reason"} <= fields


# --- no CUDA is a RUNG, not a refusal ----------------------------------------

def test_a_gpu_declaring_function_SERVES_on_a_pod_with_no_cuda() -> None:
    """The property the boot tests already prove, pinned so it cannot regress.

    `test_boot_span_ladder_pgw797` boots a `Resources(gpu=True)` endpoint on
    CPU-only CI and runs a warmup forward through it. A plan-time refusal here
    would withdraw that function — which is how th#1867 found this out, by
    reddening those tests with a refusal it had drafted.
    """
    plan = plan_serve(Resources(gpu=True), _NO_CUDA, 0.0)
    assert plan.serveable, plan.reason
    assert plan.run_mode == "cpu"
    assert plan.degraded, "a CPU serve is degraded and must say so on FnDegraded"
    assert plan.warning, "a silent CPU serve is §1.7a's blackout"


def test_nothing_in_this_planner_refuses_on_hardware_at_all() -> None:
    """Sweep the whole (caps x free VRAM) space this planner can see. The only
    non-serveable verdict left is a LIBRARY one — our image, not the card."""
    for free_gb in (0.0, 1.0, 24.0, 80.0):
        for caps in (_CUDA, _NO_CUDA):
            plan = plan_serve(Resources(gpu=True), caps, free_gb)
            assert plan.serveable, (caps, free_gb, plan.reason)
    missing_lib = plan_serve(
        Resources(gpu=True, libraries=("nunchaku",)), _CUDA, 80.0)
    assert not missing_lib.serveable
    assert missing_lib.fit == FIT_INCOMPATIBLE
    assert "librar" in missing_lib.reason.lower(), missing_lib.reason


def test_the_reactive_walk_STILL_refuses_the_cpu_rung_pgw1212() -> None:
    """DO NOT DELETE WITHOUT pgw#1212.

    Plan time is not the situation pgw#1212 describes; the reactive descent is.
    A pod that has OOMed its way down the ladder must stop one rung short of
    CPU and name OUR code, because THAT path really has never been executed.
    Deleting the plan-time refusal must not be mistaken for deleting this one.
    """
    assert rung.descend("sequential") is None
    assert rung.descent_floor("sequential") == rung.FLOOR_CPU_RUNG_UNEXECUTABLE
