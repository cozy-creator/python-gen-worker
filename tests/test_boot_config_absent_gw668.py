"""gw#668: "no boot config was ever injected" is not "boot config generation 0"."""

from __future__ import annotations

from typing import Any, Optional

import msgspec
import pytest

from gen_worker.config import load_settings
from gen_worker.config.settings import BOOT_CONFIG_GENERATION_ABSENT
from gen_worker.lifecycle_intents import CapabilityFacts, IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb

_ENV = "WORKER_CONFIG_GENERATION"


def _command(registry: IntentRegistry, generation: int) -> pb.DesiredStateCommand:
    return pb.DesiredStateCommand(
        worker_session_id=registry.worker_session_id,
        command_seq=1,
        goal_id="goal-1",
        release_id="release-1",
        config_generation=generation,
        config_digest=b"digest",
        parameter_snapshot=msgspec.msgpack.encode({}),
        first_action_by_unix_ms=9_000_000_000_000,
        intents=[
            pb.DesiredIntent(
                intent_id=f"config-{generation}",
                kind=pb.DESIRED_INTENT_KIND_CONFIG_APPLY,
                cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
                mandatory=True,
            )
        ],
        mandatory=True,
    )


def _converge(boot_stamp: Optional[int], generation: int = 4) -> pb.ConfigApplication:
    kwargs: dict[str, Any] = {}
    if boot_stamp is not None:
        kwargs["boot_config_generation"] = boot_stamp
    registry = IntentRegistry("release-1", ["echo"], **kwargs)
    receipt = registry.apply_command(_command(registry, generation))
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED
    registry.config_snapshot_applied(generation)
    registry.bindings_applied(generation)
    return registry.snapshot().config_application


def _capability_state(boot_stamp: int) -> Any:
    registry = IntentRegistry(
        "release-1", ["echo"], boot_config_generation=boot_stamp)
    registry.apply_command(_command(registry, 1))
    registry.config_snapshot_applied(1)
    registry.bindings_applied(1)
    registry.refresh_projection(CapabilityFacts(
        config_generation=1,
        parameter_snapshot_generation=1,
        binding_ready_generation=1,
        available=frozenset({"echo"}),
        hot={"echo": pb.DesiredInstance(function_name="echo")},
    ))
    return registry.snapshot().capabilities[0].state


def test_a_pod_less_worker_converges_its_boot_class() -> None:
    application = _converge(BOOT_CONFIG_GENERATION_ABSENT)
    assert application.boot_generation == 4, (
        "there is no boot-only environment for this process to be stale "
        "against, so the class is not-applicable and converges")
    assert application.state == pb.CONFIG_APPLICATION_STATE_CONVERGED
    assert not application.pending_classes.boot


def test_the_absent_stamp_is_the_default() -> None:
    assert _converge(None).state == pb.CONFIG_APPLICATION_STATE_CONVERGED


def test_a_pod_launched_worker_with_a_stale_stamp_still_reports_boot_stale() -> None:
    for stamp in (0, 1, 3):
        application = _converge(stamp)
        assert application.boot_generation == stamp, stamp
        assert application.state == pb.CONFIG_APPLICATION_STATE_BOOT_STALE, stamp
        assert application.pending_classes.boot, stamp
    assert _converge(4).state == pb.CONFIG_APPLICATION_STATE_CONVERGED


def test_capabilities_are_ready_for_a_pod_less_worker_and_stale_for_an_old_pod() -> None:
    assert _capability_state(BOOT_CONFIG_GENERATION_ABSENT) == \
        pb.FUNCTION_CAPABILITY_STATE_READY
    assert _capability_state(0) == pb.FUNCTION_CAPABILITY_STATE_BOOT_STALE


@pytest.mark.parametrize(
    "env,expected",
    [
        (None, BOOT_CONFIG_GENERATION_ABSENT),
        ("", BOOT_CONFIG_GENERATION_ABSENT),
        ("0", 0),
        ("7", 7),
    ],
)
def test_settings_distinguishes_absent_from_zero(
    env: Optional[str], expected: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if env is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, env)
    assert load_settings().boot_config_generation == expected
