"""pgw#918 — the base-weight-lane vocabulary is ONE list, and it is complete.

``executor._SPECULATIVE_CELL_BASE_LANES`` carries a comment stating its own
invariant: *"Must cover every base lane a loader can leave a pipeline on, or a
cold worker can never pull the very cell its own boot would mint"* — the
ie#546 burst, where 9 workers re-minted a cell that was armed and published on
a lane no lookup ever speculated.  The constant violated it: ``"w4a4"`` and
``"svdq-native"`` were stampable and absent.

An authored allowlist nobody checks is the defect class (C4), so this test
does the checking mechanically: it PARSES every ``_cozy_weight_lane`` /
``_WEIGHT_LANE_ATTR`` assignment under ``gen_worker/models`` and fails if a
loader can stamp a base lane the single source of truth does not name.  Add a
lane to a loader without adding it here and this test names the file, the
line, and the lane.

No torch, no GPU, no mocks — the real source tree and the real constants.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Set, Tuple

from gen_worker import aot_mint, executor
from gen_worker.compile_cache import execution_lane_bucket
from gen_worker.models import loading
from gen_worker.models.w8a8_lora import lora_execution_lane

#: Assignment targets that mean "this pipeline's base weight lane".
_EXECUTION_LANE_TARGETS = frozenset({"_cozy_weight_lane", "_WEIGHT_LANE_ATTR"})

#: Folded by :func:`loading.pipeline_weight_lane` to ``""`` — it traces
#: identically to plain bf16 and is therefore not a distinct cell lane.
_FOLDED = frozenset({"bf16-resident"})

_MODELS_DIR = Path(loading.__file__).parent


def _string_literals(node: ast.AST) -> Set[str]:
    """Every string this value expression can EVALUATE TO.

    Deliberately not ``ast.walk``: a ternary's *test* is full of strings that
    are not lanes (``mode != "dequant"``), and counting them would make the
    check noisy enough to be turned off.  Covers the two shapes the loaders
    actually use — a bare literal (``pipe._cozy_weight_lane = "w8a8"``) and a
    conditional over literals (``"w4a4" if mode != "dequant" else
    "bf16-resident"``).  Anything else (a call such as ``lora_lane(...)``)
    yields the empty set and is reported as derived.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _string_literals(node.body) | _string_literals(node.orelse)
    return set()


def _is_execution_lane_target(target: ast.AST) -> bool:
    if isinstance(target, ast.Attribute):
        return target.attr in _EXECUTION_LANE_TARGETS
    if isinstance(target, ast.Name):
        return target.id in _EXECUTION_LANE_TARGETS
    return False


def stamped_execution_lanes() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """``(literal lanes, derived lanes)`` keyed by ``file:line``.

    ``setattr(pipe, _WEIGHT_LANE_ATTR, "fp8-hooks")`` is a Call, not an
    Assign, so it is walked too — that is the exact site
    ``loading.py`` stamps ``"fp8-hooks"`` from.
    """
    literal: Dict[str, Set[str]] = {}
    derived: Dict[str, Set[str]] = {}
    for path in sorted(_MODELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            value: ast.AST | None = None
            if isinstance(node, ast.Assign) and any(
                _is_execution_lane_target(t) for t in node.targets
            ):
                value = node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) == 3
                and _is_execution_lane_target(node.args[1])
            ):
                value = node.args[2]
            if value is None:
                continue
            # The declaration of the attribute name itself is not a stamp.
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_WEIGHT_LANE_ATTR"
                for t in node.targets
            ):
                continue
            where = f"{path.relative_to(_MODELS_DIR.parent.parent)}:{node.lineno}"
            found = _string_literals(value)
            if found:
                literal[where] = found
            else:
                derived[where] = set()
    return literal, derived


def test_every_stampable_execution_lane_is_in_the_single_source_of_truth():
    literal, _derived = stamped_execution_lanes()
    assert literal, "found no _cozy_weight_lane assignment sites to check"
    known = set(loading.STAMPABLE_BASE_EXECUTION_LANES) | _FOLDED
    unknown = {
        where: sorted(execution_lanes - known)
        for where, execution_lanes in literal.items() if execution_lanes - known
    }
    assert not unknown, (
        f"loaders stamp base lanes that loading.STAMPABLE_BASE_LANES does not "
        f"name: {unknown}. This is the ie#546 defect: a cold worker never "
        f"computes the cell key for a lane it cannot speculate, so the armed "
        f"cell is unreachable and every pod re-mints."
    )


def test_the_ie546_execution_lanes_and_the_two_pgw918_found_are_all_covered():
    """The regression this issue IS: w4a4 and svdq-native were missing."""
    for execution_lane in ("", "fp8-hooks", "w8a8", "w4a4", "svdq-native"):
        assert execution_lane in loading.STAMPABLE_BASE_EXECUTION_LANES


def test_the_executor_speculates_exactly_the_loader_vocabulary():
    """One list, not two. The pre-load pull-by-key lookup speculates every
    lane a loader can leave, because there is no second copy to drift."""
    assert (tuple(executor._SPECULATIVE_CELL_BASE_EXECUTION_LANES)
            == tuple(loading.STAMPABLE_BASE_EXECUTION_LANES))


def test_the_mint_holds_no_execution_lane_allowlist_at_all():
    """pgw#850 superseded pgw#918's second half by DELETION.

    ``PARITY_LANES`` was the surviving half of that allowlist — one member,
    ``"w8a8"`` — and it composed with tensorhub's compiled-only
    ``fp8-w8a8-dynamic`` lane into a total block: the hub withholds a
    mandatory-compile lane until a cell exists, and only a pod already on the
    lane can mint one, so the single admitted token named the single lane no
    AUTO pod could ever be on. Checking an allowlist's members for
    reachability is the wrong invariant when the right answer is that the mint
    holds no allowlist. It is GIVEN a lane and compiles it.
    """
    assert not hasattr(aot_mint, "PARITY_LANES")
    assert not hasattr(aot_mint, "lane_admitted")


def test_the_one_derived_stamp_site_decomposes_to_a_named_base():
    """``w8a8_lora.lora_lane`` is the only lane stamp the parser cannot read
    as a literal.  It is the BUCKETED form of a base lane, so the base set
    stays complete iff every base it can be handed decomposes back out of
    ``lane_bucket`` into the same list."""
    _literal, derived = stamped_execution_lanes()
    assert derived, "expected the bucketed lora_lane stamp site"
    for base in loading.STAMPABLE_BASE_EXECUTION_LANES:
        for sparse in (False, True):
            stamp = lora_execution_lane(64, sparse, base=base)
            decomposed, bucket = execution_lane_bucket(stamp)
            if sparse:
                # Sparse placement is eager-only and never produces a cell —
                # it deliberately does not parse as bucketed.
                assert bucket == 0
                continue
            assert bucket == 64
            assert decomposed in loading.STAMPABLE_BASE_EXECUTION_LANES, (
                f"lora_lane(base={base!r}) decomposes to {decomposed!r}, "
                f"which is not a named base lane")


def test_the_dead_regressed_execution_lanes_constant_is_gone():
    """It had no reader in src/ and named an unstampable lane."""
    assert not hasattr(aot_mint, "REGRESSED_LANES")
