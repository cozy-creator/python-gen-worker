"""The STAFFING ENVELOPE an entrypoint declares: which machine it is placed on."""

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


def test_the_declaration_reaches_the_spec(staffing: ModuleType) -> None:
    """`@entrypoint(resources=...)` and bare `@entrypoint` both work."""

    assert getattr(staffing.generate, ENTRYPOINT_ATTR).resources is staffing.H3_STAFFING
    assert getattr(staffing.analyze, ENTRYPOINT_ATTR).resources == Resources(vcpus=4)
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
    """`Resources.__post_init__` mirrors the hub's `extractStaffingEnvelope` ingest refusals, so a contradiction costs a ValueError and not a build."""

    with pytest.raises(ValueError, match="below gpu_count"):
        Resources(gpu_count=4, max_gpu_count=2)
    with pytest.raises(ValueError, match="vcpus must be positive"):
        Resources(vcpus=0)


def test_all_five_facts_reach_the_manifest(rows: dict[str, dict]) -> None:

    block = rows["generate"]["resources"]
    assert block["vcpus"] == 16
    assert block["max_gpu_count"] == 4
    assert block["max_gpus_per_execution_group"] == 4
    assert json.loads(json.dumps(block))["parallel"] == ["sequence"]
    assert block["requires"]["recommended"] == {"min_host_ram_gb": 96.0}
    assert "min_host_ram_gb" not in block["requires"]


def test_host_ram_cannot_be_declared_as_a_minimum() -> None:
    """The asymmetry with `vcpus` is the RULING, not an oversight."""

    with pytest.raises(ValueError, match="RECOMMENDED only"):
        Resources(requires="ram96g")


def test_vcpus_is_not_a_requires_term() -> None:
    """ONE axis, ONE spelling."""

    from gen_worker.models.tensor_layout_contract import KNOWN_REQUIREMENT_TERMS

    assert "min_vcpus" not in KNOWN_REQUIREMENT_TERMS
    assert "min_cpu" not in " ".join(KNOWN_REQUIREMENT_TERMS)
    with pytest.raises(ValueError):
        Resources(requires={"minimum": {"min_vcpus": 4}})


def test_a_weightless_function_can_state_its_cpu_floor(rows: dict[str, dict]) -> None:

    row = rows["analyze"]
    assert row["slots"] == []
    assert row["resources"] == {"vcpus": 4}
    assert "gpu" not in row["resources"]


def test_both_requires_scopes_fold_rather_than_shadow(rows: dict[str, dict]) -> None:
    """The model header's lane-derived floor and the function's own `Resources(requires=)` both target `functions[].resources.requires`, and they FOLD by `term_meets` — the strictest of everything declare..."""

    requires = rows["generate"]["resources"]["requires"]
    assert "min_vram_gb" not in requires
    assert requires["min_sm"] == 80
    assert requires["recommended"]["min_host_ram_gb"] == 96.0


def test_the_undeclared_control_is_unchanged(rows: dict[str, dict]) -> None:
    """An entrypoint with no `resources=` emits the class header's floor and nothing of its own — the sdxl deploy pins this shape."""

    assert rows["control"]["resources"] == {
        "gpu": True, "requires": {"min_sm": 80},
    }


def test_the_block_reaches_the_real_manifest(tmp_path: Path) -> None:
    """The whole `cozy build` discovery path, not just `discover_entrypoints`."""
    from gen_worker.discovery.discover import discover_manifest

    root = tmp_path / "staffing_e2e"
    package = root / "src" / "staffing_e2e"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "staffing-e2e"\nversion = "0.1.0"\n\n'
        '[tool.gen_worker]\nmain = "staffing_e2e.main"\n'
    )
    (root / "endpoint.toml").write_text(
        'schema_version = 1\nmain = "staffing_e2e.main"\n\n'
        '[[build.profiles]]\naccelerator = "cuda"\n'
    )
    (package / "__init__.py").write_text("")
    (package / "main.py").write_text((FIXTURES / f"{MODULE}.py").read_text())

    sys.path.insert(0, str(root / "src"))
    try:
        manifest = discover_manifest(root)
    finally:
        sys.path.remove(str(root / "src"))

    blocks = {row["name"]: row.get("resources") for row in manifest["entrypoints"]}
    assert blocks["generate"]["vcpus"] == 16
    assert blocks["generate"]["max_gpu_count"] == 4
    assert blocks["generate"]["max_gpus_per_execution_group"] == 4
    assert list(blocks["generate"]["parallel"]) == ["sequence"]
    assert blocks["generate"]["requires"]["recommended"]["min_host_ram_gb"] == 96.0
    assert blocks["analyze"] == {"vcpus": 4}
    assert blocks["control"] == {
        "gpu": True, "requires": {"min_sm": 80},
    }


def test_absent_stays_absent(staffing: ModuleType) -> None:
    """No model slot AND no `resources=` means no claim: the key is omitted entirely and the hub falls back to release-level resolution."""

    sys.path.insert(0, str(FIXTURES))
    try:
        rows = {r["name"]: r for r in discover_entrypoints("weightless_endpoint")}
    finally:
        sys.path.remove(str(FIXTURES))
    assert "resources" not in rows["transform"]
