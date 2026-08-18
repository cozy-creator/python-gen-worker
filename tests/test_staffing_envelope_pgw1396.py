"""The v2 surface can state the machine it needs (se#755 / pgw#1396).

`Resources` has carried `vcpus`, `max_gpu_count`,
`max_gpus_per_execution_group` and `parallel` all along, and tensorhub has
read every one of them at FUNCTION scope all along
(`internal/builder/function_requirements.go` -> `payload["min_vcpus"]`, and
`staffing_envelope.go` for the other three). What was missing was any syntax
for reaching them from the pgw#1382 author surface: `@entrypoint` took no
kwargs, and `entrypoints_v2._resources()` hand-built `{gpu, requires}` from
the model slots alone, so five of `Resources`' eight fields could never
appear in a v2 manifest.

These tests pin the emission. The hub-side READ is pinned in tensorhub
th#2166 against the exact rows this file produces.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

from gen_worker import Resources
from gen_worker.discovery.entrypoints_v2 import discover_entrypoints
from gen_worker.serving.entrypoints import ENTRYPOINT_ATTR, EntrypointDeclarationError

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
MODULE = "staffing_endpoint"


@pytest.fixture(scope="module")
def staffing() -> Iterator[ModuleType]:
    sys.path.insert(0, str(FIXTURES))
    try:
        yield importlib.import_module(MODULE)
    finally:
        sys.path.remove(str(FIXTURES))


@pytest.fixture(scope="module")
def rows(staffing: ModuleType) -> dict[str, dict]:
    return {row["name"]: row for row in discover_entrypoints(MODULE)}


# -- declaration ------------------------------------------------------------


def test_the_declaration_reaches_the_spec(staffing: ModuleType) -> None:
    """`@entrypoint(resources=...)` and bare `@entrypoint` both work."""

    assert getattr(staffing.generate, ENTRYPOINT_ATTR).resources is staffing.H3_STAFFING
    assert getattr(staffing.analyze, ENTRYPOINT_ATTR).resources == Resources(vcpus=4)
    # The bare form is untouched: absence is None, not an empty Resources.
    assert getattr(staffing.control, ENTRYPOINT_ATTR).resources is None


def test_resources_must_be_a_resources() -> None:
    """A typed refusal naming the author's own line, not a manifest key."""

    module = type(sys)("pgw1396_probe")
    module.__dict__["__name__"] = "pgw1396_probe"
    sys.modules["pgw1396_probe"] = module
    source = (
        "import msgspec\n"
        "from gen_worker import RequestContext, entrypoint\n"
        "class In(msgspec.Struct): text: str\n"
        "class Out(msgspec.Struct): text: str\n"
        "@entrypoint(resources={'vcpus': 4})\n"
        "def f(ctx: RequestContext, payload: In) -> Out: ...\n"
    )
    try:
        with pytest.raises(EntrypointDeclarationError) as excinfo:
            exec(source, module.__dict__)  # noqa: S102
    finally:
        del sys.modules["pgw1396_probe"]
    assert "resources=" in str(excinfo.value)
    assert "gen_worker.Resources" in str(excinfo.value)


def test_the_envelope_refusals_fire_at_declaration_time() -> None:
    """`Resources.__post_init__` mirrors the hub's `extractStaffingEnvelope`
    ingest refusals, so a contradiction costs a ValueError and not a build."""

    with pytest.raises(ValueError, match="below gpu_count"):
        Resources(gpu_count=4, max_gpu_count=2)
    with pytest.raises(ValueError, match="vcpus must be positive"):
        Resources(vcpus=0)


# -- emission ---------------------------------------------------------------


def test_all_five_facts_reach_the_manifest(rows: dict[str, dict]) -> None:
    """The five facts se#755 measured as having "no home"."""

    block = rows["generate"]["resources"]
    assert block["vcpus"] == 16                          # the CPU floor
    assert block["max_gpu_count"] == 4                   # the SHAPE (ceiling)
    assert block["max_gpus_per_execution_group"] == 4    # the degree axis
    # A tuple in memory, a JSON array on the wire — `normalizeParallelMechanisms`
    # decodes `[]any`. Asserted through `json` so the WIRE shape is the claim.
    assert json.loads(json.dumps(block))["parallel"] == ["sequence"]
    # Host RAM: a RECOMMENDATION, nested, and NEVER a minimum. Paul,
    # 2026-07-11: a RunPod GPU pod can neither select nor guarantee it, so a
    # declared floor was unenforceable theater (tensorhub
    # `TermDef.MinimumDeclarable` is false for this term alone).
    assert block["requires"]["recommended"] == {"min_host_ram_gb": 96.0}
    assert "min_host_ram_gb" not in block["requires"]


def test_host_ram_cannot_be_declared_as_a_minimum() -> None:
    """The asymmetry with `vcpus` is the RULING, not an oversight."""

    with pytest.raises(ValueError, match="RECOMMENDED only"):
        Resources(requires="ram96g")


def test_vcpus_is_not_a_requires_term() -> None:
    """ONE axis, ONE spelling. `resources.vcpus` is already the declared
    surface and the hub already normalizes it to `min_vcpus` itself; a
    `requires.min_vcpus` term would land in one payload key from two
    directions."""

    from gen_worker.models.tensor_layout_contract import KNOWN_REQUIREMENT_TERMS

    assert "min_vcpus" not in KNOWN_REQUIREMENT_TERMS
    assert "min_cpu" not in " ".join(KNOWN_REQUIREMENT_TERMS)
    with pytest.raises(ValueError):
        Resources(requires={"minimum": {"min_vcpus": 4}})


def test_a_weightless_function_can_state_its_cpu_floor(rows: dict[str, dict]) -> None:
    """se#757 blocker C cleared the SIGNATURE; this clears the MACHINE.

    `dj-utils` (vcpus=4) and `music-analysis` (2/8/8 across three functions)
    are the shipped shape, and it is why this declaration is per-FUNCTION: a
    single endpoint- or model-scoped number cannot express three values, and
    a weightless function has no `Model` class to hang one on.
    """

    row = rows["analyze"]
    assert row["slots"] == []
    assert row["resources"] == {"vcpus": 4}
    # No accelerator is manufactured for a function that declared none.
    assert "gpu" not in row["resources"]


def test_both_requires_scopes_fold_rather_than_shadow(rows: dict[str, dict]) -> None:
    """The model header's lane-keyed `requires=` and the function's own
    `Resources(requires=)` both target `functions[].resources.requires`, and
    they FOLD by `term_meets` — the strictest of everything declared for this
    function. Either one shadowing the other is a silent under-declaration
    the buy path cannot detect."""

    requires = rows["generate"]["resources"]["requires"]
    assert requires["min_vram_gb"] == 78.0                       # from the class header
    assert requires["recommended"]["min_host_ram_gb"] == 96.0    # from the function


def test_the_undeclared_control_is_unchanged(rows: dict[str, dict]) -> None:
    """An entrypoint with no `resources=` emits EXACTLY what it emitted
    before pgw#1396 — the sdxl deploy pins this shape."""

    assert rows["control"]["resources"] == {"gpu": True, "requires": {"min_vram_gb": 78.0}}


def test_absent_stays_absent(staffing: ModuleType) -> None:
    """No model slot AND no `resources=` means no claim: the key is omitted
    entirely and the hub falls back to release-level resolution. A
    present-but-EMPTY block is the hub's explicit CPU declaration, which is a
    different statement and must not be made by accident."""

    sys.path.insert(0, str(FIXTURES))
    try:
        rows = {r["name"]: r for r in discover_entrypoints("weightless_endpoint")}
    finally:
        sys.path.remove(str(FIXTURES))
    assert "resources" not in rows["transform"]
