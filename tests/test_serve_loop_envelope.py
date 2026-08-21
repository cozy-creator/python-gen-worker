"""The entrypoint dispatch loop: envelope in, residency leases around, result out.

Integration, no mocks beyond the seams the design names (LoaderEngine absent
-> eager bridge; sizer = the static local table): the main_v2-contract-exact
fixture endpoint serves REAL requests end-to-end on CPU with fake weights
from config-only checkpoints, through the signature-derived envelope and
`ResidencyManager.lease` around every invocation — the pgw#1372 dispatch
loop the sdxl migration (se#751) lands on.
"""

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
        # The deploy's lane pick (multi-lane models refuse an unnamed lane).
        lane_contract="sdxl.diffusers-bf16@1" if fixture is FIXTURE_DIR else "",
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
    # The fixture's structured evidence: the served checkpoint's pinned ref.
    assert result.model == DREAM
    assert result.loras == []
    saved = tmp_path / "outputs" / result.image.ref
    assert saved.is_file() and saved.stat().st_size > 0
    # The instance is resident under its reservation (weights + headroom).
    assert manager.tier_of(DREAM, "SdxlModel/sdxl.diffusers-bf16@1") is Tier.VRAM

    # pgw#1404 degraded mode, end to end on the REAL serve path. This test host
    # has no CUDA device at all, and the fixture's bf16 lane declares vram12g —
    # so the machine is under the floor by every measure. Paul, 2026-08-18:
    # "we should be able to place any model on any machine; if the machine is a
    # poor match, it will complain and warn loudly when it enters degraded
    # mode, but still run anyway." Both halves are asserted here: the request
    # SUCCEEDED (everything above this line), and it says why it will be slow.
    assert len(outcome.warnings) == 1
    warning = outcome.warnings[0]
    assert warning.startswith("DEGRADED PLACEMENT: ")
    assert "sdxl.diffusers-bf16@1" in warning  # the lane
    assert "12.0 GiB" in warning               # the declared floor
    assert "0.0 GiB" in warning                # what this machine actually has
    assert "cpu (no CUDA device)" in warning   # the card
    assert "Running anyway" in warning         # the consequence, not a refusal
    # It rides the adjustment ledger (JobResult.adjustments), field-less, so
    # the hub records it against the request rather than the caller inferring
    # it from a latency graph.
    assert [
        row for row in outcome.adjustments
        if row["field"] == "" and row["reason"] == warning
    ]


def test_adapters_ride_the_envelope_and_the_scopes_restore(tmp_path: Path) -> None:
    loop, _, _ = make_loop(tmp_path)
    envelope = {
        "model": DREAM,
        "adapters": {"turbo": TURBO_ROW, "loras": [STYLE_ROW]},
        "input": {"prompt": "ink harbor"},
    }
    outcome = loop.invoke("generate", envelope, request_id="req-2")
    # Application order: the distillation adapter first, then style LoRAs;
    # scale is the envelope-resolved value.
    assert [(u.ref, u.scale) for u in outcome.result.loras] == [
        ("cozy/lightning-4step@1", 1.0), ("maker/ink-style@3", 0.7),
    ]
    # The model-owned scopes restored the post-load baseline: no adapters
    # remain applied on the persistent instance.
    import serving_v2_fixture.main as v2

    ((_, backend),) = loop._backends.items()
    live = backend.model
    assert isinstance(live, v2.SdxlModel)
    assert live.pipe.active_adapters == []
    assert live.pipe.loaded_loras == []

    # A second, adapter-free request on the SAME instance serves clean.
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

    # A style LoRA cannot seize the distillation slot (typed takeover guard).
    message = refuse({"model": DREAM, "adapters": {"turbo": STYLE_ROW},
                      "input": {"prompt": "x"}})
    assert "not distillation-marked" in message
    # "models" against a single-model signature.
    assert '"model", not "models"' in refuse(
        {"models": {"model": DREAM}, "input": {"prompt": "x"}})
    # Picks for undeclared adapter slots.
    assert "undeclared slot" in refuse(
        {"model": DREAM, "adapters": {"stylegrid": STYLE_ROW},
         "input": {"prompt": "x"}})
    # No pick anywhere: the worker never guesses which bytes to serve.
    assert "no envelope pick and no deployment default" in refuse(
        {"input": {"prompt": "x"}})
    # A deployment default fills the slot; the same envelope then serves.
    resolver.default_picks["model"] = JUGGER
    outcome = loop.invoke("generate", {"input": {"prompt": "x"}}, request_id="r2")
    assert outcome.result.model == JUGGER
    # Unknown functions stay a dispatch refusal, before envelope decode.
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
    # ctx.warn rows ride the outcome (the reply envelope's warning channel).
    assert any("already step-distilled" in w for w in outcome.warnings)
    assert outcome.result.loras == []  # the adapter was ignored, not applied


def test_residency_wraps_the_loop_lru_and_never_fits(tmp_path: Path) -> None:
    loop, _, manager = make_loop(tmp_path, vram_gb=5)  # fits ONE 3G+1G instance
    lane = "SdxlModel/sdxl.diffusers-bf16@1"
    loop.invoke("generate", {"model": DREAM, "input": {"prompt": "a"}}, request_id="r1")
    assert manager.tier_of(DREAM, lane) is Tier.VRAM
    # The second checkpoint evicts the first BETWEEN requests (no host tier:
    # straight back to the chunk store) — admission before allocation.
    loop.invoke("generate", {"model": JUGGER, "input": {"prompt": "b"}}, request_id="r2")
    assert manager.tier_of(JUGGER, lane) is Tier.VRAM
    assert manager.tier_of(DREAM, lane) is Tier.ABSENT
    # Re-serving the first re-admits it (a fresh load — nothing cached lies).
    outcome = loop.invoke(
        "generate", {"model": DREAM, "input": {"prompt": "c"}}, request_id="r3")
    assert outcome.result.model == DREAM
    # A model that can NEVER fit refuses typed at admission — before its
    # instance is constructed, before any byte moves.
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
    assert loop._backends[key].model is first_model  # same author object


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
        # Envelope key is "models" (two slots), keyed by SLOT NAME.
        {"models": {"right": "r/right@1", "left": "l/left@1"},
         "input": {"value": 5}},
        request_id="r",
    )
    assert outcome.result.served_by == "SlowModel+OtherModel"
    # STABLE slot-name order: "left" before "right", whatever the signature
    # order — the multi-model deadlock rule.
    assert lease_order == ["OtherModel/eager", "SlowModel/eager"]


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
        rt.ENTERED.acquire()  # only once the first request is inside
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
    # The instrumented gauge proves one-at-a-time on the instance.
    assert rt.HIGH_WATER == 1
