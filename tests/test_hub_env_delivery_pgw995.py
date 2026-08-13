"""pgw#995 deliverable 2 — hub-shaped env delivery reaches `Settings`.

The `GEN_WORKER_PREFER_AOT` postmortem was not a bug in any component. The flag
was declared, the compiled graph was set, the loader worked, the gate worked. What broke
was the DELIVERY between them: a release rebuild stopped declaring the name, the
hub withheld the compiled graph silently, and the worker booted without it. Three pod
attempts.

Nothing in this repo could have caught that, because the local rig constructs
its own environment — `mint_process.child_env` for the mint child,
`dict(os.environ)` for the adopting process. Both are shapes a production pod
never has. So the regression class had exactly one detector: a pod.

These tests give it a second one. They drive the REAL
`gen_worker.config.load_settings` — the same function `entrypoint._run_main`
calls — over an environment produced by the hub's resolution rule rather than by
the test's own convenience.

Run: pytest tests/test_hub_env_delivery_pgw995.py -v
"""

from __future__ import annotations

import pytest

from gen_worker import config as config_pkg
from gen_worker.config import load_settings

from harness import hub_env


def _boot(monkeypatch: pytest.MonkeyPatch, env: dict) -> None:
    """Replace the process environment with a pod's, exactly."""
    for name in list(os_environ_names()):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def os_environ_names() -> list:
    import os

    return [n for n in os.environ if n.startswith(
        ("GEN_WORKER_", "TENSORHUB_", "WORKER_", "COZY_", "HF_"))]


# ---------------------------------------------------------------------------
# The seam: a DECLARED compiled graph is delivered and reaches the typed struct
# ---------------------------------------------------------------------------


def test_a_hub_delivered_env_value_reaches_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole chain, end to end, with nothing hand-placed in the middle.

    The worker function declares `HF_TOKEN`; the operator sets it; the release
    declares it; so the hub delivers it and `Settings` carries it. `HF_TOKEN` is
    the realistic subject rather than a made-up name: it is org-settable (the
    hub reserves the `HF_` prefix but exempts this one), it is what a gated
    clone needs, and its absence is what th#1073 records breaking gated clones
    for days.

    Every hop is the production one — `resolve()` implements the hub's rule and
    `load_settings()` is the function the worker's own entrypoint calls.
    """
    declarations = hub_env.declared_by(["HF_TOKEN"])
    compiled_graphs = hub_env.EndpointEnvCompiledGraphs(
        {"HF_TOKEN": "hf_operator_set_token"})

    delivery = hub_env.resolve(declarations, compiled_graphs)
    assert delivery.env == {"HF_TOKEN": "hf_operator_set_token"}
    assert delivery.withheld == ()

    # The pod boots with image env + delivered compiled graphs, and NOTHING the test
    # process happened to be carrying.
    _boot(monkeypatch, hub_env.pod_environ({}, delivery))

    settings = load_settings()
    assert settings.hf_token == "hf_operator_set_token", (
        "a declared, delivered env did not reach Settings — the delivery chain "
        "is broken between the hub's resolve and config.loader")


# ---------------------------------------------------------------------------
# The regression: the postmortem's exact shape, now RED locally
# ---------------------------------------------------------------------------


def test_a_rebuild_that_stops_declaring_a_name_withholds_it_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GEN_WORKER_PREFER_AOT`, reproduced in milliseconds instead of three pods.

    Release N declares the name and the value arrives. Release N+1 is rebuilt
    from worker code that no longer declares it — the operator changed nothing,
    the compiled graph is untouched — and the value stops arriving. The point is not that
    it stops (that is the declaration contract working); the point is that the
    delivery SAYS SO, which is the half that was missing on both sides.
    """
    compiled_graphs = hub_env.EndpointEnvCompiledGraphs(
        {"HF_TOKEN": "hf_operator_set_token"})

    before = hub_env.resolve(
        hub_env.declared_by(["HF_TOKEN"]), compiled_graphs)
    _boot(monkeypatch, hub_env.pod_environ({}, before))
    assert load_settings().hf_token == "hf_operator_set_token"

    # The rebuild. Nobody edited the compiled graph; the worker function's env list
    # changed, so the release declares nothing.
    after = hub_env.resolve(hub_env.declared_by([]), compiled_graphs)

    assert after.env == {}, "an undeclared name must not be injected"
    assert after.withheld_names() == ["HF_TOKEN"], (
        "the rebuild dropped a configured compiled_graph and reported NOTHING — this is "
        "precisely the silence that cost three pod attempts (th#1650)")
    assert after.withheld[0].reason == hub_env.WITHHELD_UNDECLARED
    assert "0 env name(s)" in after.withheld[0].detail, (
        "a withholding must distinguish 'this release declares nothing at all' "
        "(the rebuild case) from 'this one name was removed' (the intended one)")

    _boot(monkeypatch, hub_env.pod_environ({}, after))
    assert load_settings().hf_token == "", (
        "the withheld value still reached Settings — something other than the "
        "hub is supplying it, which is the substitution this harness exists to "
        "forbid")


def test_an_ambient_export_cannot_stand_in_for_a_hub_delivered_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rig that lets the developer's own shell satisfy the assertion is
    decoration. `pod_environ(strip=...)` is what stops that, so it is tested
    rather than trusted: with the name exported ambiently and NOT declared by
    the release, `Settings` must still come up empty.
    """
    monkeypatch.setenv("HF_TOKEN", "ambient_shell_token")
    ambient = {"HF_TOKEN": "ambient_shell_token"}

    delivery = hub_env.resolve(
        hub_env.declared_by([]),
        hub_env.EndpointEnvCompiledGraphs({"HF_TOKEN": "hf_operator_set_token"}))

    env = hub_env.pod_environ(ambient, delivery, strip=["HF_TOKEN"])
    _boot(monkeypatch, env)
    assert load_settings().hf_token == ""


# ---------------------------------------------------------------------------
# The reserved namespace still wins over a declaration
# ---------------------------------------------------------------------------


def test_a_release_cannot_declare_its_way_into_the_platform_namespace(
) -> None:
    """pgw#763 delta 0: the process-split switch is platform-only. Declaring it
    must not deliver it — otherwise a release could opt its own tenant code out
    of the boundary that contains it."""
    delivery = hub_env.resolve(
        hub_env.declared_by(["GEN_WORKER_COMPUTE_CHILD"]),
        hub_env.EndpointEnvCompiledGraphs({"GEN_WORKER_COMPUTE_CHILD": "1"}))
    assert delivery.env == {}
    assert delivery.withheld[0].reason == hub_env.WITHHELD_RESERVED


def test_the_loader_is_the_only_component_this_harness_talks_to() -> None:
    """Guard against the harness growing into a second config implementation.

    `hub_env` models DELIVERY. The moment it starts constructing `Settings`
    itself it stops testing the production path and starts certifying its own —
    which is how a harness becomes the thing it was written to check.
    """
    src = (hub_env.__file__ or "")
    assert src
    text = open(src).read()
    for forbidden in ("load_settings", "Settings(", "msgspec"):
        assert forbidden not in text, (
            f"hub_env references {forbidden!r}: it must produce an ENVIRONMENT "
            f"and let the real loader turn it into config")
    assert config_pkg.load_settings is load_settings


# ---------------------------------------------------------------------------
# The rig mode itself (pgw#995 deliverable 2)
# ---------------------------------------------------------------------------


def _rig():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import micro_mint_rig  # noqa: PLC0415 - the script under test

    return micro_mint_rig


def test_rig_hub_env_mode_delivers_declared_and_strips_ambient() -> None:
    """`--hub-env` boots the mint child the way a pod is booted.

    Two properties, and the second is the one that makes the mode worth having:
    a DECLARED compiled graph is delivered, and an AMBIENT value of the same name is
    stripped rather than inherited. Without the strip, a developer with the
    variable exported in their shell gets a green rig for a release that would
    have booted without it on a pod — which is not a weaker test, it is a test
    that reports the opposite of the truth.
    """
    rig = _rig()
    env, withheld = rig.hub_delivered_env(
        {"PATH": "/usr/bin", "HF_TOKEN": "ambient-shell-value"},
        {"HF_TOKEN": "delivered-by-hub"})

    assert env["HF_TOKEN"] == "delivered-by-hub"
    assert env["PATH"] == "/usr/bin", "image env must survive"
    assert withheld == []


def test_rig_hub_env_mode_reports_an_undeclared_compiled_graph_instead_of_dropping_it(
) -> None:
    """The rig's whole reason to exist is turning a pod-only failure into a
    local one. An compiled graph the release does not declare must show up as a FACT the
    rig reports, not as a variable that quietly is not there."""
    rig = _rig()
    env, withheld = rig.hub_delivered_env(
        {"PATH": "/usr/bin"}, {"COZY_SOMETHING_UNDECLARED": "x"})

    assert "COZY_SOMETHING_UNDECLARED" not in env
    assert len(withheld) == 1
    assert withheld[0]["name"] == "COZY_SOMETHING_UNDECLARED"
    assert withheld[0]["reason"] == hub_env.WITHHELD_UNDECLARED
    assert withheld[0]["detail"], "a withholding with no detail is a shrug"


def test_rig_strips_every_name_it_claims_to_deliver() -> None:
    """A name the rig DECLARES but does not STRIP is a hole: ambient value in,
    hub value never exercised, mode silently decorative."""
    rig = _rig()
    assert set(rig.RIG_DECLARED_ENV) <= set(rig.RIG_STRIPPED_ENV), (
        "every declared name must also be stripped, or the ambient environment "
        "can satisfy the assertion the mode exists to make")
