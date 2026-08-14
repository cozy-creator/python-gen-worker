"""pgw#1206 A2 — THE degrade-don't-OOM ruling as ONE test on the One Rung ladder.

The ruling (Paul, 2026-07-10): a CUDA OOM is a ladder transition, never a hard
fail — descend one rung and retry, down to the terminal rung. The implementation
used to span three vocabularies in five files; this file is the ruling's single
red-verified guard on the consolidated ladder.

th#1867 removed the ruling's one exception. ``Resources(strict_vram=True)`` let
an author opt out of the host-RAM-touching rungs, which turned "runs slower"
into "does not run" — §2.4 ruling 4 read it as a card requirement in softer
words and deleted it with `vram_gb_hint`/`min_vram_gb`. The descent now has no
opt-out at all: it ends where OUR ladder ends and nowhere else.

The wire half was RED against 91df247a: ``serve_fit.demoted`` wrote
``ran="offload:<placement>"`` while tensorhub matches ``FnDegraded.ran``
EXACTLY (degradation_reschedule.go: ``case "offload","cpu","emergency_quant"``)
— so every runtime demotion silently missed the VRAM-driven-drain arm.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from gen_worker.models import rung
from gen_worker.models.memory import place_pipeline
from gen_worker.models.serve_fit import (
    RUN_CPU,
    plan_serve,
    RUN_FP8_STORAGE,
    RUN_NATIVE,
    RUN_OFFLOAD,
    replan,
)

#: tensorhub's exact-match FnDegraded.ran vocabulary
#: (profiling/degradation.go + autoscale/degradation_reschedule.go). The hub
#: still accepts "emergency_quant"; pgw#1206 D deleted the only rung that
#: produced it, so nothing here may emit it again.
GO_RAN_VOCABULARY = {RUN_NATIVE, RUN_FP8_STORAGE, RUN_OFFLOAD, RUN_CPU, "bf16", "fp8"}


# --- the ladder itself ------------------------------------------------------

def test_one_ordered_ladder() -> None:
    """One ladder, best-first, price monotonic, projections coherent."""
    names = [r.name for r in rung.LADDER]
    assert names == [
        "native", "fp8_storage",
        "model_offload", "group_offload", "sequential", "cpu",
    ]
    # No rung manufactures a quant at runtime. A quant rung is an
    # AOT artifact the ladder SELECTS; the only runtime storage transform left
    # is fp8-E4M3, which is a re-encoding of the same weights, not a quant
    # recipe with an ungated quality verdict.
    assert [r.storage for r in rung.LADDER if r.storage] == ["fp8"]
    assert "emergency_quant" not in {r.run_mode for r in rung.LADDER}
    prices = [r.latency for r in rung.LADDER]
    assert prices == sorted(prices), "price must be monotonic down the ladder"
    # Host-RAM-touching == exactly the placement tail (the pgw#1063 whole-tree
    # host-RAM charge; it was the strict_vram boundary until th#1867).
    assert rung.PLACEMENT_LADDER == ("model_offload", "group_offload", "sequential")
    for r in rung.LADDER:
        assert r.touches_host_ram == (r.name in rung.PLACEMENT_LADDER)
    # Resident flavors are not rungs: from any of them the reactive walk
    # starts at the first placement rung.
    for flavor in ("", "off", "vae_only", "auto"):
        nxt = rung.descend(flavor)
        assert nxt is not None and nxt.name == "model_offload"


def test_descend_walks_every_rung_and_terminates() -> None:
    """From any resident token the reactive walk visits each placement rung
    exactly once and ends — it never wraps, never skips to CPU."""
    seen = []
    cur = ""
    while True:
        nxt = rung.descend(cur)
        if nxt is None:
            break
        seen.append(nxt.name)
        cur = nxt.name
    assert seen == list(rung.PLACEMENT_LADDER)


def test_no_declaration_can_truncate_the_ladder_th1867() -> None:
    """th#1867 deleted ``strict_vram``, which cut this walk before the first
    host-RAM-touching rung. The walk must now end ONLY where the ladder ends —
    a fact about our code — and never on an author's say-so, so ``descend``
    takes no declaration argument at all and cannot grow one back silently."""
    import inspect
    for fn in (rung.descend, rung.descent_floor):
        params = set(inspect.signature(fn).parameters)
        assert params == {"current"}, (
            f"{fn.__name__} takes {sorted(params)}: a second parameter here is "
            "a declaration re-entering the descent (th#1867 §2.4 ruling 4)")
    first = rung.descend("")
    second = rung.descend("model_offload")
    assert first is not None and first.name == "model_offload"
    assert second is not None and second.name == "group_offload"


def test_floor_only_deepens() -> None:
    """The learned per-ref floor is a max, not a last-write."""
    assert rung.floor_of("group_offload", "model_offload") == "group_offload"
    assert rung.floor_of("model_offload", "sequential") == "sequential"
    assert rung.floor_of("", "model_offload") == "model_offload"
    assert rung.floor_of("auto", "") == "auto"  # non-ladder tokens rank equal-shallowest


# --- OOM at each rung: descend, never die -----------------------------------
# The seam is apply_low_vram_config (the established th#1043 pattern): the
# ladder WALK is the code under test; the diffusers hooks are not, and this
# box exports GEN_WORKER_FORBID_CPU_OFFLOAD, which the real applier honors.


def _arm(monkeypatch: pytest.MonkeyPatch, serves_at: str) -> list[str]:
    import torch

    from gen_worker.models import memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory, "_move_pipeline_to_cpu", lambda *_: None)
    monkeypatch.setattr(memory, "repair_device_placement", lambda *_: [])
    monkeypatch.setattr(memory, "flush_memory", lambda: None)
    attempted: list[str] = []

    def fake_apply(pipeline: object, *, mode: str, logger: object = None) -> dict:
        attempted.append(mode)
        if mode != serves_at:
            raise RuntimeError("CUDA error: out of memory")
        return {"mode": mode}

    monkeypatch.setattr(memory, "apply_low_vram_config", fake_apply)
    return attempted


@pytest.mark.parametrize("serves_at", ["model_offload", "group_offload", "sequential"])
def test_oom_descends_one_rung_at_a_time_and_serves(
    serves_at: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA OOM at each rung engages the NEXT rung — one at a time, in order,
    never a hard fail while a rung remains."""
    attempted = _arm(monkeypatch, serves_at)
    applied = place_pipeline(object(), mode="off", logger=logging.getLogger("t"))
    ladder_prefix = list(rung.PLACEMENT_LADDER[: rung.PLACEMENT_LADDER.index(serves_at) + 1])
    assert attempted == ["off"] + ladder_prefix
    assert applied["mode"] == serves_at
    assert applied.get("oom_demotions") == len(ladder_prefix)
    assert applied.get("requested_mode") == "off"


def test_oom_below_terminal_rung_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When even the terminal rung OOMs the failure surfaces (typed CUDA OOM)
    — the ladder ends, it does not wrap or invent capacity."""
    attempted = _arm(monkeypatch, serves_at="never")
    with pytest.raises(RuntimeError, match="out of memory"):
        place_pipeline(object(), mode="off", logger=logging.getLogger("t"))
    assert attempted == ["off"] + list(rung.PLACEMENT_LADDER)


def test_the_reactive_descent_has_no_declaration_opt_out_th1867(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario `strict_vram` used to refuse now DESCENDS and serves.

    Same arm as the deleted test — a pipeline that OOMs resident and serves at
    `model_offload`. It used to raise before touching a host-RAM rung. §1.35
    makes that descent the normal path, so the ladder is walked and the
    placement is applied."""
    attempted = _arm(monkeypatch, serves_at="model_offload")
    place_pipeline(object(), mode="off", logger=logging.getLogger("t"))
    assert attempted == ["off", "model_offload"]


# --- the wire: ran is the Go vocabulary, exactly ----------------------------

def test_demotion_ran_matches_go_vocabulary_exactly() -> None:
    """RED against 91df247a: a runtime demotion must report ``ran`` from the
    hub's exact-match vocabulary; placement detail rides the reason, not the
    token tensorhub switches on."""
    plan = replan(None, run_mode=RUN_OFFLOAD, detail="CUDA OOM mid-inference; model_offload")
    assert plan.ran in GO_RAN_VOCABULARY, plan.ran
    assert plan.ran == RUN_OFFLOAD
    assert plan.run_mode == RUN_OFFLOAD
    assert plan.degraded
    assert plan.est_latency_multiplier == rung.price(RUN_OFFLOAD)


def test_load_rung_and_cast_drop_share_the_one_replan() -> None:
    """The load-time rungs and the th#737 cast-drop report through the SAME
    projection — no third vocabulary, wanted/ran stay honest."""
    engaged = replan(None, run_mode=RUN_FP8_STORAGE, detail="load fit: fp8")
    assert engaged.run_mode == RUN_FP8_STORAGE and engaged.ran == RUN_FP8_STORAGE
    assert engaged.degraded
    dropped = replan(None, wanted="fp8", ran="bf16", detail="no cast surface")
    assert dropped.run_mode == RUN_NATIVE
    assert dropped.wanted == "fp8" and dropped.ran == "bf16"
    assert dropped.degraded  # ran != wanted must surface as FnDegraded


# --- pgw#1206 D: no rung manufactures a quant, and the floor still holds -----


def _resources() -> Any:
    from gen_worker.api.decorators import Resources

    # th#1867: there is nothing size-shaped left to declare. The rung a load
    # lands on is decided by `models/memory` from MEASURED bytes, which is why
    # the tests below drive `place_pipeline` rather than `plan_serve`.
    return Resources(gpu=True)


def _cuda_caps() -> Any:
    from gen_worker.models.hub_policy import TensorhubWorkerCapabilities

    return TensorhubWorkerCapabilities(
        cuda_version="13.0", gpu_sm=90, torch_version="2.13.0", installed_libs=[])


@pytest.mark.parametrize("free_gb", [40.0, 8.0])
def test_a_model_over_VRAM_SERVES_and_never_refuses(free_gb: float) -> None:
    """Degrade-don't-OOM (Paul, 2026-07-10) survives the nf4 deletion AND
    th#1867's deletion of the whole plan-time estimate.

    At 40 GB free the old ladder had exactly one answer — quantize the denoiser
    to 4 bits at load, at a quality nobody gated. That rung is gone. th#1867
    then removed the plan-time prediction itself, so what this asserts is the
    property that outlived both: an 80 GB model on an 8 GB card is SERVEABLE,
    at every free-VRAM value, and the descent that carries it is measured at
    load time rather than guessed here.
    """
    plan = plan_serve(_resources(), _cuda_caps(), free_vram_gb=free_gb)

    assert plan.serveable, plan.reason
    assert "emergency" not in (plan.warning or "").lower()
    assert "4-bit" not in (plan.warning or "")


def test_the_reactive_demotion_still_reports_the_go_vocabulary() -> None:
    """The plan-time ladder is gone; the WIRE contract it fed is not. A
    measured demotion must still project `ran` into tensorhub's exact-match
    RunMode vocabulary, or the degradation arrives unreadable."""
    demoted = replan(None, run_mode=RUN_OFFLOAD, detail="measured: 80.0 GB tree, 8.0 GB free")
    assert demoted.ran in GO_RAN_VOCABULARY
    assert demoted.degraded
    # The measured numbers survive into the warning, beside the priced story.
    assert "8.0 GB free" in demoted.warning


def test_no_runtime_quant_rung_exists_to_be_reached() -> None:
    """The mechanism, not just the decision: the fit verdict vocabulary has no
    runtime-quant member, so nothing can route to one again by adding a caller.
    """
    from gen_worker.models import hub_policy, loading

    assert not hasattr(hub_policy, "FIT_EMERGENCY")
    assert not hasattr(loading, "emergency_quantization_config")
    assert not hasattr(loading, "emergency_quant_enabled")
    assert not hasattr(loading, "bitsandbytes_available")
    assert not hasattr(loading, "EMERGENCY_FIT_FACTOR")
    # ...while the AOT path that READS a quant off disk is untouched: a
    # fetched artifact still loads through its own config.
    assert hasattr(loading, "synthesize_quantization_config")
