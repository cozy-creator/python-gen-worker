"""The entrypoint dispatch loop: envelope in, residency leases around, result out."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker.serving import DeployBinding, load_endpoint
from gen_worker.serving.envelope import EnvelopeError
from gen_worker.serving.host import ServeDispatchError
from gen_worker.serving.residency import NeverFits, ResidencyManager, Tier
from gen_worker.serving.serve_loop import ServeLoop, manifest_sizer

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
RT_DIR = Path(__file__).parent / "fixtures" / "serving_rt_endpoint"

GB = 1024**3

DREAM = "org/dreamshaper@2"
JUGGER = "org/juggernaut@7"
HUGE = "org/videotitan@1"

WEIGHTS = {DREAM: 3 * GB, JUGGER: 3 * GB, HUGE: 40 * GB,
           "r/right@1": 1 * GB, "l/left@1": 1 * GB}

TURBO_ROW = {
    "ref": "cozy/lightning-4step@1", "path": "/adapters/lightning.safetensors",
    "name": "lightning-4step", "distillation": True,
    "defaults": {"scheduler": "euler_trailing"},
}
STYLE_ROW = {
    "ref": "maker/ink-style@3", "path": "/adapters/ink.safetensors",
    "name": "ink-style", "scale": 0.7,
}


class LocalResolver:
    """Deploy state over local config-only trees — the BindingResolver seam."""

    def __init__(self, root: Path, defaults_by_ref: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.root = root
        self.defaults_by_ref = dict(defaults_by_ref or {})
        self.default_picks: Dict[str, str] = {}
        self.resolved: List[str] = []

    def _tree(self, ref: str) -> Path:
        tree = self.root / ref.replace("/", "_").replace("@", "_")
        if not tree.exists():
            tree.mkdir(parents=True)
            (tree / "config.json").write_text(json.dumps({"seed": len(ref)}))
        return tree

    def resolve(self, model_cls: type, checkpoint_ref: str) -> DeployBinding:
        self.resolved.append(checkpoint_ref)
        return DeployBinding(
            checkpoint_ref=checkpoint_ref,
            checkpoint_dir=self._tree(checkpoint_ref),
            model="sdxl",
            defaults=self.defaults_by_ref.get(checkpoint_ref, {}),
        )

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        return self.default_picks.get(slot_name, "")


def make_loop(
    tmp_path: Path,
    fixture: Path = FIXTURE_DIR,
    *,
    vram_gb: int = 64,
    defaults_by_ref: Dict[str, Dict[str, Any]] | None = None,
) -> tuple[ServeLoop, LocalResolver, ResidencyManager]:
    loaded = load_endpoint(fixture)
    resolver = LocalResolver(tmp_path / "trees", defaults_by_ref)
    manager = ResidencyManager(
        vram_gb * GB, manifest_sizer(WEIGHTS, headroom_bytes=1 * GB)
    )
    loop = ServeLoop(
        loaded,
        residency=manager,
        resolver=resolver,
        lane_contract="sdxl.diffusers@1+plain.bf16@1" if fixture is FIXTURE_DIR else "",
        output_dir=tmp_path / "outputs",
    )
    return loop, resolver, manager


def test_the_envelope_serves_end_to_end_with_fake_tensors(tmp_path: Path) -> None:
    loop, _, manager = make_loop(tmp_path)
    outcome = loop.invoke(
        "generate",
        {"model": DREAM, "input": {"prompt": "a lighthouse", "seed": 3}},
        request_id="req-1",
    )
    result = outcome.result
    assert result.model == DREAM
    assert result.loras == []
    saved = tmp_path / "outputs" / result.image.ref
    assert saved.is_file() and saved.stat().st_size > 0
    assert manager.tier_of(DREAM, "SdxlModel/sdxl.diffusers@1+plain.bf16@1") is Tier.VRAM

    # pgw#1404 degraded mode, end to end on the REAL serve path — and pgw#1599
    # narrowed WHAT can trigger it, which is asserted here rather than left to
    # be discovered on a pod.
    #
    # Paul, 2026-08-18: "we should be able to place any model on any machine;
    # if the machine is a poor match, it will complain and warn loudly when it
    # enters degraded mode, but still run anyway." The half that matters most
    # is intact and is everything above this line: THE REQUEST SUCCEEDED on a
    # host with no CUDA device at all.
    #
    # The warning half has ONE input now instead of two. The VRAM arm lost its
    # input with the floor STRINGS (Paul: "there is no required VRAM"); the
    # number that replaces it is COMPUTED from the lane's demand formula over
    # the advertised shape envelope (pgw#1600) and is not wired yet.
    #
    # The `min_sm` arm, though, now FIRES here — and it is right to.
    # pgw#1621: this fixture used to declare `LaneRef("sdxl.diffusers-bf16@1",
    # dtype=torch.float32)` — a lane NAMED bf16 carrying an author-typed
    # float32 — and the old `warnings == ()` was bought by exactly that
    # incoherence, not by the endpoint genuinely being an fp32 one. Under v2
    # the dtype is not the fixture's to pick: it is `declared_dtype` on the
    # ratified quant rule, and `spec/v2/rules/plain.bf16.v1.json` states
    # `capability_floor_sm: 80`. So a real bf16 lane on a host with no CUDA
    # device is a real shortfall, and the warning is the system working.
    #
    # THE HALF THAT MATTERS IS STILL EVERYTHING ABOVE THIS LINE: the request
    # SUCCEEDED. This is the whole degrade-never-refuse ruling measured on the
    # real serve path — it warns loudly, and it serves.
    (warned,) = outcome.warnings
    assert warned.startswith("DEGRADED PLACEMENT: cpu (no CUDA device)")
    assert "sdxl.diffusers@1+plain.bf16@1 needs sm_80+" in warned
    assert "Running anyway" in warned
    # ONE row, not two: the unpack above is the assertion that the VRAM arm
    # stayed quiet, and it is quiet for a stated reason rather than because the
    # instrument is dead — `min_vram_gb` is deleted and pgw#1600 has not landed
    # its computed replacement. `test_the_degraded_placement_warning_can_still
    # _go_red` holds the other polarity: an h100 that MEETS the floor reports
    # nothing, so an always-warning instrument cannot hide here either.
    #
    # And it reaches the caller through the ADJUSTMENT LEDGER — the field-less
    # row `ctx.warn` writes — rather than a second channel beside it. That was
    # asserted here as an absence while this fixture warned about nothing; it
    # is asserted as the row itself now, which is the stronger form.
    (placement,) = [row for row in outcome.adjustments if row["field"] == ""]
    assert placement["reason"] == warned
    assert (placement["requested"], placement["applied"]) == ("", "")


def test_the_degraded_placement_warning_can_still_go_red(tmp_path: Path) -> None:
    """The instrument is NARROWED, not dead (pgw#1599).

    Its VRAM arm lost its input with the floor strings and gets it back as a
    COMPUTED number in pgw#1600. Its `min_sm` arm is untouched and still
    DERIVED — pgw#1621 only moved where from: it is `capability_floor_sm` on
    the lane's ratified QUANT RULE now, rather than a lookup on the contract's
    dtype spelling. A real bf16 lane on a machine with no CUDA device still
    warns, loudly, and still serves.

    Written as its own test on purpose: "the warning did not fire" is only
    honest evidence when something else proves it CAN.
    """
    from gen_worker.serving.placement import DeviceFacts, shortfalls
    from gen_worker.serving.model import Model, model_requires
    from gen_worker import lane
    from gen_worker.demand import GiB, const
    from gen_worker.models import SDXL

    real_lane = ("sdxl.diffusers@1", "plain.bf16@1")
    real_lane_id = "sdxl.diffusers@1+plain.bf16@1"

    class RealLaneModel(Model[SDXL], lanes={real_lane: lane(request=const(GiB(1)))}):
        pass

    # The floor is DERIVED from the QUANT RULE's own document, never written.
    assert model_requires(RealLaneModel)[real_lane_id].min_terms().min_sm == 80

    cpu_only = DeviceFacts(sm=0, vram_gib=0.0, name="cpu (no CUDA device)")
    (shortfall,) = shortfalls(RealLaneModel, real_lane, facts=cpu_only)
    assert shortfall.term == "min_sm"
    assert "sdxl.diffusers@1+plain.bf16@1" in shortfall.message
    assert "Running anyway" in shortfall.message

    h100 = DeviceFacts(sm=90, vram_gib=80.0, name="NVIDIA H100 80GB HBM3")
    assert shortfalls(RealLaneModel, real_lane, facts=h100) == ()


def test_adapters_ride_the_envelope_and_the_scopes_restore(tmp_path: Path) -> None:
    loop, _, _ = make_loop(tmp_path)
    envelope = {
        "model": DREAM,
        "adapters": {"turbo": TURBO_ROW, "loras": [STYLE_ROW]},
        "input": {"prompt": "ink harbor"},
    }
    outcome = loop.invoke("generate", envelope, request_id="req-2")
    assert [(u.ref, u.scale) for u in outcome.result.loras] == [
        ("cozy/lightning-4step@1", 1.0), ("maker/ink-style@3", 0.7),
    ]
    import serving_v2_fixture.main as v2

    ((_, backend),) = loop._backends.items()
    live = backend.model
    assert isinstance(live, v2.SdxlModel)
    assert live.pipe.active_adapters == []
    assert live.pipe.loaded_loras == []

    second = loop.invoke(
        "generate", {"model": DREAM, "input": {"prompt": "plain"}},
        request_id="req-3",
    )
    assert second.result.loras == []


def test_the_distillation_slot_guard_and_signature_derived_refusals(
    tmp_path: Path,
) -> None:
    loop, resolver, _ = make_loop(tmp_path)
    def refuse(envelope: Dict[str, Any]) -> str:
        with pytest.raises(EnvelopeError) as excinfo:
            loop.invoke("generate", envelope, request_id="r")
        return str(excinfo.value)

    message = refuse({"model": DREAM, "adapters": {"turbo": STYLE_ROW},
                      "input": {"prompt": "x"}})
    assert "not distillation-marked" in message
    assert '"model", not "models"' in refuse(
        {"models": {"model": DREAM}, "input": {"prompt": "x"}})
    assert "undeclared slot" in refuse(
        {"model": DREAM, "adapters": {"stylegrid": STYLE_ROW},
         "input": {"prompt": "x"}})
    assert "no envelope pick and no deployment default" in refuse(
        {"input": {"prompt": "x"}})
    resolver.default_picks["model"] = JUGGER
    outcome = loop.invoke("generate", {"input": {"prompt": "x"}}, request_id="r2")
    assert outcome.result.model == JUGGER
    with pytest.raises(ServeDispatchError, match="no function 'enhance'"):
        loop.invoke("enhance", {"input": {}}, request_id="r3")


def test_step_distilled_checkpoint_ignores_turbo_and_warns(tmp_path: Path) -> None:
    loop, _, _ = make_loop(
        tmp_path, defaults_by_ref={DREAM: {"step_distilled": True, "cfg": False}}
    )
    outcome = loop.invoke(
        "generate",
        {"model": DREAM, "adapters": {"turbo": TURBO_ROW},
         "input": {"prompt": "x"}},
        request_id="r",
    )
    assert any("already step-distilled" in w for w in outcome.warnings)
    assert outcome.result.loras == []


def test_residency_wraps_the_loop_lru_and_never_fits(tmp_path: Path) -> None:
    loop, _, manager = make_loop(tmp_path, vram_gb=5)
    lane = "SdxlModel/sdxl.diffusers@1+plain.bf16@1"
    loop.invoke("generate", {"model": DREAM, "input": {"prompt": "a"}}, request_id="r1")
    assert manager.tier_of(DREAM, lane) is Tier.VRAM
    loop.invoke("generate", {"model": JUGGER, "input": {"prompt": "b"}}, request_id="r2")
    assert manager.tier_of(JUGGER, lane) is Tier.VRAM
    assert manager.tier_of(DREAM, lane) is Tier.ABSENT
    outcome = loop.invoke(
        "generate", {"model": DREAM, "input": {"prompt": "c"}}, request_id="r3")
    assert outcome.result.model == DREAM
    with pytest.raises(NeverFits, match="refuse at admission"):
        loop.invoke("generate", {"model": HUGE, "input": {"prompt": "d"}},
                    request_id="r4")


def test_the_instance_is_reused_across_requests_not_rebuilt(tmp_path: Path) -> None:
    loop, _, _ = make_loop(tmp_path)
    loop.invoke("generate", {"model": DREAM, "input": {"prompt": "a"}}, request_id="r1")
    ((key, backend),) = loop._backends.items()
    first_model = backend.model
    assert first_model is not None
    loop.invoke("generate", {"model": DREAM, "input": {"prompt": "b"}}, request_id="r2")
    assert loop._backends[key].model is first_model


def test_multi_model_slots_lease_in_stable_slot_name_order(tmp_path: Path) -> None:
    loop, _, manager = make_loop(tmp_path, fixture=RT_DIR)
    import serving_rt_fixture.main as rt

    rt.reset()
    lease_order: List[str] = []
    true_lease = manager.lease

    def spying_lease(
        checkpoint_ref: str, lane: str, factory: Any, **kwargs: Any
    ) -> Any:
        lease_order.append(lane)
        return true_lease(checkpoint_ref, lane, factory, **kwargs)

    manager.lease = spying_lease  # type: ignore[method-assign]
    outcome = loop.invoke(
        "pair",
        {"models": {"right": "r/right@1", "left": "l/left@1"},
         "input": {"value": 5}},
        request_id="r",
    )
    assert outcome.result.served_by == "SlowModel+OtherModel"
    assert lease_order == [
        "OtherModel/sdxl.diffusers@1+plain.bf16@1", "SlowModel/sdxl.diffusers@1+plain.bf16@1"
    ]


def test_single_flight_rides_the_lease_per_instance(tmp_path: Path) -> None:
    loop, _, _ = make_loop(tmp_path, fixture=RT_DIR)
    import serving_rt_fixture.main as rt

    rt.reset()
    done: List[str] = []

    def held() -> None:
        loop.invoke("run", {"model": "r/right@1", "input": {"value": 1, "hold": True}},
                    request_id="h")
        done.append("held")

    def chaser() -> None:
        rt.ENTERED.acquire()
        loop.invoke("run", {"model": "r/right@1", "input": {"value": 2}},
                    request_id="c")
        done.append("chaser")

    threads = [threading.Thread(target=held), threading.Thread(target=chaser)]
    for t in threads:
        t.start()
    rt.RELEASE.set()
    for t in threads:
        t.join(timeout=30)
    assert sorted(done) == ["chaser", "held"]
    assert rt.HIGH_WATER == 1
