"""pgw#1206 A2 — THE degrade-don't-OOM ruling as ONE test on the One Rung ladder.

The ruling (Paul, 2026-07-10): a CUDA OOM is a ladder transition, never a hard
fail — descend one rung and retry, down to the terminal rung; the only opt-out
is the author's ``Resources(strict_vram=True)``. The implementation used to
span three vocabularies in five files; this file is the ruling's single
red-verified guard on the consolidated ladder.

The wire half was RED against 91df247a: ``serve_fit.demoted`` wrote
``ran="offload:<placement>"`` while tensorhub matches ``FnDegraded.ran``
EXACTLY (degradation_reschedule.go: ``case "offload","cpu","emergency_quant"``)
— so every runtime demotion silently missed the VRAM-driven-drain arm.
"""

from __future__ import annotations

import logging

import pytest

from gen_worker.models import rung
from gen_worker.models.memory import place_pipeline
from gen_worker.models.serve_fit import (
    RUN_CPU,
    RUN_EMERGENCY,
    RUN_FP8_STORAGE,
    RUN_NATIVE,
    RUN_OFFLOAD,
    replan,
)

#: tensorhub's exact-match FnDegraded.ran vocabulary
#: (profiling/degradation.go + autoscale/degradation_reschedule.go).
GO_RAN_VOCABULARY = {RUN_NATIVE, RUN_FP8_STORAGE, RUN_EMERGENCY, RUN_OFFLOAD, RUN_CPU, "bf16", "fp8"}


# --- the ladder itself ------------------------------------------------------

def test_one_ordered_ladder() -> None:
    """One ladder, best-first, price monotonic, projections coherent."""
    names = [r.name for r in rung.LADDER]
    assert names == [
        "native", "fp8_storage", "nf4",
        "model_offload", "group_offload", "sequential", "cpu",
    ]
    prices = [r.latency for r in rung.LADDER]
    assert prices == sorted(prices), "price must be monotonic down the ladder"
    # Host-RAM-touching == exactly the placement tail (the strict_vram
    # boundary and the pgw#1063 whole-tree host-RAM charge).
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


def test_strict_vram_truncates_before_host_ram() -> None:
    """The author's opt-out: no descent may reach a host-RAM-touching rung."""
    assert rung.descend("", strict_vram=True) is None
    assert rung.descend("model_offload", strict_vram=True) is None


def test_floor_only_deepens() -> None:
    """The learned per-ref floor (gw#463) is a max, not a last-write."""
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


def test_strict_vram_refuses_instead_of_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    """th#1107/th#1043: strict_vram refuses BEFORE any host-RAM rung — on the
    reactive path too, with a typed message, never a silent slow serve."""
    attempted = _arm(monkeypatch, serves_at="model_offload")
    with pytest.raises(RuntimeError, match="strict_vram"):
        place_pipeline(
            object(), mode="off", strict_vram=True, logger=logging.getLogger("t"))
    assert attempted == ["off"]


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
    engaged = replan(None, run_mode=RUN_EMERGENCY, detail="load fit: nf4")
    assert engaged.run_mode == RUN_EMERGENCY and engaged.ran == RUN_EMERGENCY
    assert engaged.degraded
    dropped = replan(None, wanted="fp8", ran="bf16", detail="no cast surface")
    assert dropped.run_mode == RUN_NATIVE
    assert dropped.wanted == "fp8" and dropped.ran == "bf16"
    assert dropped.degraded  # ran != wanted must surface as FnDegraded
