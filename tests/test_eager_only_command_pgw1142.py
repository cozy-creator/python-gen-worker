"""pgw#1142 / DESIGN-RULINGS §4.32 item 4 — the operator's eager-only command.

Paul, verbatim: *"The consumer should have the ability to turn off compile
entirely, and serve-eager only, if the compile is broken or they just don't
care. I.e., they can send some sort of command to the worker to serve eager
rather than serve compiled."*

Four properties, and every one of them is a separate way the feature can be
built wrong:

1. **It suppresses.** Nothing arms, nothing mints, and an already-armed cell is
   not called.
2. **It is REVERSIBLE.** Releasing it resumes compiled serving on the next
   request with no re-arm — which is only true if the suppression is read at
   the CALL rather than by unwrapping the artifact.
3. **It is DISTINCT from §4.31's sticky de-arm.** Same eager posture, two
   triggers: releasing an operator order must not resurrect a cell that
   de-armed itself for cause, and their tokens must never collide.
4. **It is legible.** A suppressed worker's request rows say
   ``operator_eager_only`` — not ``aot_cell`` (the cell is still armed, and
   naive classification reads exactly that), and not any of the tokens that
   mean something failed.
"""

from __future__ import annotations

import argparse
import asyncio
import types
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from gen_worker import activity
from gen_worker import aot_serve
from gen_worker import boot_adopt
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells
from gen_worker import lifecycle as lifecycle_mod
from gen_worker import serve_posture
from gen_worker import serving_mode
from gen_worker.cell_adopt import EagerPhase
from gen_worker.cli import serve as serve_cli
from gen_worker.pb import worker_scheduler_pb2 as pb
from torch_compiled_graphs import CallIngress, CallInput, CompiledGraphRunner


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def default_posture() -> Any:
    """The order is process-global; no test may leak one into the next."""
    serve_posture.reset()
    yield
    serve_posture.reset()


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


class FakeTensor:
    def __init__(self, shape: Any, dtype: str = "torch.bfloat16") -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class FakePackage:
    """Stands in for TCG's bound runner; records every invocation."""

    def __init__(self) -> None:
        self.invocations: List[Any] = []
        self.bound = False
        self.calls = 0
        self.declared_fqns = ("conv_in.weight",)
        self.raises = False

    def bind(self, values: Dict[str, Any], *, device: str) -> None:
        del values, device
        self.bound = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.invocations.append((args, kwargs))
        if self.raises:
            raise RuntimeError("cuda kernel launch failed")
        self.calls += 1
        return "ARTIFACT_OUTPUT"


class FakeModule:
    def __init__(self) -> None:
        self.device = "cpu"
        self.eager_calls = 0

    def state_dict(self) -> Dict[str, Any]:
        return {"conv_in.weight": FakeTensor([1])}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.eager_calls += 1
        return "EAGER_OUTPUT"


class FakePipeline:
    def __init__(self, module: FakeModule) -> None:
        self.unet = module


class Cfg:
    family = "sdxl-base"
    lora_bucket = 0
    targets = ("unet",)
    regional = False


KEY = "cg-key-v1-" + "d" * 56
META: Dict[str, Any] = {
    "family": "sdxl-base",
    "compiled_graph_key": KEY,
}

#: The ONE graph class this fixture arms. An entry NAMES its class.
ENTRY_NAME = "unet/g"

INGRESS = CallIngress(
    parameters=("sample",),
    flat_arity=1,
    inputs=(CallInput(
        "sample", 0, "sample", 0, (), "sample", "bfloat16",
        (2, 4, "h", "w"),
    ),),
    symbols=(("h", (64, 160)), ("w", (64, 160))),
)


def _armed_module() -> tuple[FakeModule, FakePackage, FakePipeline]:
    """A module wrapped by the real TCG runner policy, one class armed."""
    module, package = FakeModule(), FakePackage()
    pipeline = FakePipeline(module)
    package.bind(module.state_dict(), device="cpu")
    runner = aot_serve.TCGEntryRunner(
        cast(CompiledGraphRunner, package),
        INGRESS,
        "unet",
        ENTRY_NAME,
        Cfg.family,
    )
    dispatch = aot_serve.EntryDispatch(declared=(ENTRY_NAME,))
    dispatch.add(ENTRY_NAME, runner)
    aot_serve.wrap_module(module, dispatch, META, target="unet")
    setattr(pipeline, "_cozy_aot", {
        "meta": META,
        "targets": {"unet": {
            "module": module, "attr": "forward",
            "state": getattr(module, "_cozy_aot")["state"]}},
        "entries": {ENTRY_NAME: {
            "compiled_graph_key": KEY,
            "target": "unet",
            "class_hash": "",
        }},
        "bound_constants": {"pools": {}, "literals": {}},
    })
    # The fixture's own claim, checked once here rather than in five rows: this
    # pipeline really is serving the class it says it is.
    assert aot_serve.armed_entries(pipeline) == {ENTRY_NAME: KEY}
    return module, package, pipeline


def _call() -> tuple[Any, ...]:
    return (FakeTensor([2, 4, 128, 128]),)


# ---------------------------------------------------------------------------
# 1. the order itself
# ---------------------------------------------------------------------------


def test_default_posture_permits_compiled_serving() -> None:
    assert serve_posture.eager_only() is False
    assert serve_posture.block() == ""


def test_the_order_engages_and_is_released() -> None:
    assert serve_posture.apply_command(
        True, actor="paul", reason="the compile is broken") is True
    assert serve_posture.eager_only() is True
    block = serve_posture.block()
    assert "paul" in block and "the compile is broken" in block

    assert serve_posture.apply_command(False, actor="paul") is True
    assert serve_posture.eager_only() is False
    assert serve_posture.block() == ""


def test_repeating_the_order_is_not_a_transition() -> None:
    """The hub replays it to a reconnecting worker; that is not an event."""
    assert serve_posture.apply_command(True, actor="hub") is True
    assert serve_posture.apply_command(True, actor="hub") is False
    assert serve_posture.order().active is True


def test_each_transition_confesses_on_the_wire(events: List[Any]) -> None:
    serve_posture.apply_command(True, actor="paul", reason="broken")
    serve_posture.apply_command(True, actor="paul", reason="broken")  # no-op
    serve_posture.apply_command(False, actor="paul")

    kinds = [(e.kind, e.phase) for e in events]
    assert kinds == [
        (activity.KIND_SERVE_POSTURE, serve_posture.PHASE_SUPPRESSED),
        (activity.KIND_SERVE_POSTURE, serve_posture.PHASE_RELEASED),
    ]
    # The operator's words reach the hub — an order nobody can explain is an
    # unexplained fleet-wide eager posture six months later.
    assert "broken" in events[0].detail
    assert serve_posture.REASON in events[0].detail


def test_the_token_is_its_own(events: List[Any]) -> None:
    """It must never be countable with a failure class or with a plan order."""
    assert serve_posture.REASON == EagerPhase.OPERATOR_EAGER_ONLY.value
    assert serve_posture.REASON == serving_mode.POSTURE_OPERATOR_EAGER_ONLY
    assert serve_posture.REASON == boot_adopt.OPERATOR_EAGER_ONLY
    assert boot_adopt.OPERATOR_EAGER_ONLY in boot_adopt.GATE_REASONS
    for other in (EagerPhase.COMPILED_DEGRADED, EagerPhase.HUB_ORDERED_EAGER,
                  EagerPhase.BOOT_ENDED_UNCOMPILED, EagerPhase.UNCOMPILED):
        assert serve_posture.REASON != other.value
    # ...and it is not the plan's `eager_only` backend token either: one is a
    # standing order about this POD, the other is one dispatch's backend.
    assert serve_posture.REASON != "eager_only"


# ---------------------------------------------------------------------------
# 2. it suppresses — arming, minting, and the ordered arm
# ---------------------------------------------------------------------------


def test_arming_block_names_the_order() -> None:
    """The ONE precondition authority answers it, so every arming route does."""
    pipeline = FakePipeline(FakeModule())
    # Whatever else this host blocks on (no CUDA in CI), it is not the order.
    assert "EAGER ONLY" not in cc.arming_block(
        pipeline, Cfg(), cache_ready=True, allow_cold=True)
    serve_posture.apply_command(True, actor="paul", reason="don't care")
    # And the order is named FIRST: it is the reason an operator asked for,
    # so it must not be masked by an environment fact behind it.
    assert cc.arming_block(
        pipeline, Cfg(), cache_ready=True, allow_cold=True
    ).startswith("paul ordered this worker to serve EAGER ONLY")


def test_the_fleet_policy_is_not_even_entered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not merely "does not arm": it must not resolve, download or mint.

    `arming_block` would refuse the arm at the far end anyway. This asserts the
    cost is not paid — a hub round trip and a materialize, and on a miss a
    self-mint child that compiles for minutes before rediscovering the refusal.
    """
    def _explode(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the arming policy ran under an eager-only order")

    monkeypatch.setattr(fleet_cells, "_arming_policy", _explode)
    serve_posture.apply_command(True, actor="paul")
    outcome = fleet_cells.enable_compiled(FakePipeline(FakeModule()), Cfg())
    assert outcome.armed is False
    assert outcome.eager_reason == EagerPhase.OPERATOR_EAGER_ONLY


def test_an_ordered_aot_arm_is_obeyed_as_eager_not_refused() -> None:
    """The hub named an exact cell; the operator said eager.

    The operator wins, and the shape of winning matters: `OrderedArmError`
    would fail the attempt typed and take the function down, which is the
    opposite of "serve eager instead".
    """
    serve_posture.apply_command(True, actor="paul", reason="broken")
    outcome = fleet_cells.arm_ordered(
        FakePipeline(FakeModule()), Cfg(), None,
        backend="aot_cell", artifact=Path("/nonexistent/cell.pt2"),
        delivered_ref="root/family-sdxl-base#cg-key-v1-abc",
        delivered_digest="sha256:abc", expected=None, publisher_org="root")
    assert outcome.armed is False
    assert outcome.eager_reason == EagerPhase.OPERATOR_EAGER_ONLY


# ---------------------------------------------------------------------------
# 3. serve-time suppression, and reversibility
# ---------------------------------------------------------------------------


def test_an_armed_cell_serves_eager_under_the_order_and_stays_armed() -> None:
    module, package, pipeline = _armed_module()

    assert module.forward(*_call()) == "ARTIFACT_OUTPUT"
    assert len(package.invocations) == 1

    serve_posture.apply_command(True, actor="paul", reason="broken")
    assert module.forward(*_call()) == "EAGER_OUTPUT"
    assert module.eager_calls == 1
    # The artifact was not reached, and it was not thrown away.
    assert len(package.invocations) == 1
    assert aot_serve.is_armed(pipeline) is True


def test_releasing_the_order_resumes_compiled_serving_with_no_rearm() -> None:
    """The property that decides the enforcement point.

    Unwrapping the artifact would have produced the same first half and a
    permanently eager worker afterwards.
    """
    module, package, pipeline = _armed_module()
    serve_posture.apply_command(True, actor="paul", reason="broken")
    assert module.forward(*_call()) == "EAGER_OUTPUT"

    serve_posture.apply_command(False, actor="paul", reason="it was the driver")
    assert module.forward(*_call()) == "ARTIFACT_OUTPUT"
    assert len(package.invocations) == 1
    assert module.eager_calls == 1


# ---------------------------------------------------------------------------
# 4. distinct from §4.31's sticky de-arm
# ---------------------------------------------------------------------------


def test_releasing_the_order_does_not_resurrect_a_de_armed_cell() -> None:
    """De-arm is EVIDENCE; the order is POLICY. Policy does not overrule it."""
    module, package, pipeline = _armed_module()
    package.raises = True

    # §4.31: a cell-attributable failure serves this request eager and de-arms
    # the cell, sticky for the boot.
    assert module.forward(*_call()) == "EAGER_OUTPUT"
    assert getattr(module, "_cozy_aot")["state"]["failed"] is True

    serve_posture.apply_command(True, actor="paul")
    serve_posture.apply_command(False, actor="paul")
    package.raises = False
    assert module.forward(*_call()) == "EAGER_OUTPUT"
    # Two forwards, two eager answers: the failing one (which the tenant still
    # got a correct result from) and the one after the order was released.
    assert module.eager_calls == 2


def test_the_two_triggers_report_different_reasons() -> None:
    de_armed = serving_mode.resolve(
        active_compile_ref="", eager_posture=EagerPhase.COMPILED_DEGRADED.value)
    assert de_armed.fallback_reason == EagerPhase.COMPILED_DEGRADED.value

    serve_posture.apply_command(True, actor="paul")
    ordered = serving_mode.resolve(
        active_compile_ref="", eager_posture=EagerPhase.COMPILED_DEGRADED.value)
    assert ordered.fallback_reason == serve_posture.REASON


# ---------------------------------------------------------------------------
# 5. telemetry: a suppressed worker is not a broken adopt path
# ---------------------------------------------------------------------------


def test_a_suppressed_request_is_not_reported_as_compiled() -> None:
    ref = f"root/family-sdxl-base#{KEY}"
    armed = serving_mode.resolve(active_compile_ref=ref, sm="89")
    assert armed.serving_mode == serving_mode.MODE_AOT_CELL

    serve_posture.apply_command(True, actor="paul", reason="broken")
    suppressed = serving_mode.resolve(active_compile_ref=ref, sm="89")
    assert suppressed.serving_mode == serving_mode.MODE_EAGER
    assert suppressed.fallback_reason == serve_posture.REASON
    # Not a FALLBACK: nothing fell back, so the compiled-vs-eager comparison
    # keeps its meaning.
    assert suppressed.served_eager_fallback is False
    # The cell is still named — an operator has to be able to see WHICH cell
    # is standing by.
    assert suppressed.served_cell_ref == ref


def test_a_real_guard_miss_still_reports_itself() -> None:
    """The order must not swallow the per-request fallback vocabulary."""
    missed = serving_mode.resolve(
        active_compile_ref=f"root/family-sdxl-base#{KEY}", guard_missed=True)
    assert missed.fallback_reason == serving_mode.FALLBACK_GUARD_MISS
    assert missed.served_eager_fallback is True


# ---------------------------------------------------------------------------
# 6. the wire: the command arrives on the existing control channel
# ---------------------------------------------------------------------------


def test_the_scheduler_command_applies_in_both_directions() -> None:
    """The real `Lifecycle.on_message` branch, over a real protobuf.

    Bound to a bare namespace on purpose: the branch must not depend on worker
    state, because a worker holding a broken cell is exactly the worker whose
    state may be unhealthy.
    """
    handler = lifecycle_mod.Lifecycle.on_message.__get__(
        types.SimpleNamespace())

    asyncio.run(handler(pb.SchedulerMessage(
        serve_posture=pb.ServePosture(
            eager_only=True, actor="admin:paul", reason="ie#657 cells crash"))))
    assert serve_posture.eager_only() is True
    assert serve_posture.order().actor == "admin:paul"

    asyncio.run(handler(pb.SchedulerMessage(
        serve_posture=pb.ServePosture(eager_only=False, actor="admin:paul"))))
    assert serve_posture.eager_only() is False


def test_the_command_rides_the_existing_oneof() -> None:
    msg = pb.SchedulerMessage(serve_posture=pb.ServePosture(eager_only=True))
    assert msg.WhichOneof("msg") == "serve_posture"
    # Round-trips: the hub and the worker share one generated contract.
    assert pb.SchedulerMessage.FromString(
        msg.SerializeToString()).serve_posture.eager_only is True


# ---------------------------------------------------------------------------
# 7. the cozy-local CLI half
# ---------------------------------------------------------------------------


def test_the_serve_socket_carries_a_posture_control_frame() -> None:
    frame = serve_cli._parse_frame(
        b'{"posture":{"eager_only":true,"reason":"broken"}}')
    assert frame == {"kind": "posture", "eager_only": True, "reason": "broken"}

    off = serve_cli._parse_frame(b'{"posture":{"eager_only":false}}')
    assert off == {"kind": "posture", "eager_only": False, "reason": ""}

    bad = serve_cli._parse_frame(b'{"posture":{"eager_only":"yes"}}')
    assert bad["kind"] == "error"


def test_eager_only_is_a_distinct_flag_from_eager() -> None:
    """`--eager` is about when setup() runs. Overloading it would be a trap."""
    parser = argparse.ArgumentParser()
    serve_cli.add_subparser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["serve", "--eager-only"])
    assert args.eager_only is True
    assert args.eager is False
