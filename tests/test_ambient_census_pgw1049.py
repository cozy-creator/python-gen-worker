"""pgw#1049: the ambient-input census, enforced against the INSTALLED torch.

The th#1678 wirecontract shape, python-side: the census file classifies
every env input torch/triton consult; this test scans the installed tree and
goes RED on any input the census does not classify — including the inputs a
future torch upgrade introduces. That red is the feature.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_lint() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lint_ambient_census", REPO / "scripts" / "lint_ambient_census.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_every_installed_torch_input_is_classified() -> None:
    lint = _load_lint()
    if importlib.util.find_spec("torch") is None:
        pytest.skip("no torch installed to census")
    rows, errors = lint.load_census()
    assert errors == [], errors
    found = lint.scan_installed_tree()
    assert found, "scan found nothing — the scanner itself is broken"
    unclassified = sorted(
        f"{name}  (read at {where})" for name, where in found.items()
        if lint.classify(name, rows) is None)
    assert unclassified == [], (
        "torch consults ambient inputs the census does not classify — add "
        "rows to scripts/ambient_inputs_census.txt:\n"
        + "\n".join(unclassified))


def test_neutralized_claims_are_scrub_facts() -> None:
    """Every input classified NEUTRALIZED must actually be erased by
    scrub_env — checked against the live SCRUB_PREFIXES, not the census's
    own claim about them."""
    lint = _load_lint()
    from gen_worker import env_seal

    rows, _ = lint.load_census()
    if importlib.util.find_spec("torch") is None:
        pytest.skip("no torch installed to census")
    for name in lint.scan_installed_tree():
        if lint.classify(name, rows) == "NEUTRALIZED":
            assert name.startswith(env_seal.SCRUB_PREFIXES), (
                f"{name} is classified NEUTRALIZED but scrub_env would not "
                "erase it")


def test_unclassified_input_is_red() -> None:
    """RED-proof: a name outside every row classifies to None — the exact
    condition the completeness test fails on."""
    lint = _load_lint()
    rows, _ = lint.load_census()
    assert lint.classify("CUDA_BRAND_NEW_BEHAVIOR_KNOB", rows) is None
    assert lint.classify("TORCHINDUCTOR_MAX_AUTOTUNE", rows) == "NEUTRALIZED"
    assert lint.classify("PYTHONHASHSEED", rows) == "IMPOSED"
    assert lint.classify("CUDA_VISIBLE_DEVICES", rows) == "PLUMBING"


def test_structural_lies_are_red(tmp_path: Path) -> None:
    """A census that CLAIMS an imposition or a scrub that does not exist is
    itself red — the manifest may never outrun the tree."""
    lint = _load_lint()
    lying = tmp_path / "census.txt"
    lying.write_text(
        "TOTALLY_FAKE_VAR  IMPOSED  claims an imposition that does not exist\n"
        "SOMETHING_ELSE_*  NEUTRALIZED  claims a scrub that does not exist\n")
    problems = lint.check(census_path=lying)
    assert any("claims IMPOSED" in p for p in problems), problems
    assert any("claims NEUTRALIZED" in p for p in problems), problems
    # And an incomplete census (missing IMPOSED rows for DECLARED_ENV) is
    # itself named:
    assert any("has no IMPOSED census row" in p for p in problems), problems


def test_shipped_census_is_green() -> None:
    lint = _load_lint()
    assert lint.check() == []
