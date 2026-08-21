from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "assert_ci_proof.py"
PUBLISH_YAML = REPO / ".github" / "workflows" / "publish.yaml"

TAG_TREE = "aaaa111122223333444455556666777788889999"
OTHER_TREE = "bbbb111122223333444455556666777788889999"
HEAD_SHA = "c0a9bbb097086d56cb2cf49a6f4222c0ab600348"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assert_ci_proof", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load()


def _run(run_id: int, event: str, head_sha: str = HEAD_SHA,
         conclusion: str = "success") -> "gate.Run":
    return gate.Run(id=run_id, head_sha=head_sha, event=event, conclusion=conclusion)


def _trees(**mapping: str):
    return lambda sha: mapping.get(sha)


def test_pull_request_run_is_not_proof_even_carrying_the_exact_tree() -> None:
    """The 0.113.0 shape: the cut PR's run names the tagged commit and is green."""
    verdict = gate.assess(
        TAG_TREE, [_run(31643923466, "pull_request")], _trees(**{HEAD_SHA: TAG_TREE}))

    assert verdict.proven is False
    assert verdict.kind == gate.ONLY_PULL_REQUEST_PROOF
    assert "refs/pull" in verdict.detail
    assert str(31643923466) in verdict.detail


def test_refusal_distinguishes_wrong_evidence_from_no_evidence() -> None:
    """Two different problems must not share one message."""
    wrong_kind = gate.assess(
        TAG_TREE, [_run(1, "pull_request")], _trees(**{HEAD_SHA: TAG_TREE}))
    nothing = gate.assess(
        TAG_TREE, [_run(2, "workflow_dispatch")], _trees(**{HEAD_SHA: OTHER_TREE}))

    assert wrong_kind.kind == gate.ONLY_PULL_REQUEST_PROOF
    assert nothing.kind == gate.NO_RUN_CARRIES_TREE


def test_an_unclassified_event_cannot_prove_a_release() -> None:
    """Admissibility is a deny-by-default allow-list, so a new event type is refused until someone classifies it."""
    verdict = gate.assess(
        TAG_TREE, [_run(3, "merge_group")], _trees(**{HEAD_SHA: TAG_TREE}))

    assert verdict.proven is False


@pytest.mark.parametrize("event", sorted(gate.ADMISSIBLE_EVENTS))
def test_a_dispatched_run_on_the_same_commit_still_publishes(event: str) -> None:
    verdict = gate.assess(
        TAG_TREE, [_run(9, event)], _trees(**{HEAD_SHA: TAG_TREE}))

    assert verdict.proven is True
    assert verdict.kind == gate.PROOF_FOUND
    assert verdict.run_id == 9


def test_the_pr_run_does_not_poison_a_legitimate_dispatch_on_the_same_commit() -> None:
    """The realistic cut: a PR run AND a dispatched run, same commit, same tree."""
    runs = [_run(1, "pull_request"), _run(2, "workflow_dispatch")]

    assert gate.assess(TAG_TREE, runs, _trees(**{HEAD_SHA: TAG_TREE})).proven is True
    assert gate.assess(TAG_TREE, list(reversed(runs)),
                       _trees(**{HEAD_SHA: TAG_TREE})).proven is True


def test_a_squash_or_cherry_pick_with_identical_CONTENT_still_proves() -> None:
    """Trees, not commit shas — the property the original gate got right."""
    other_commit = "1234567890abcdef1234567890abcdef12345678"
    verdict = gate.assess(
        TAG_TREE,
        [_run(4, "push", head_sha=other_commit)],
        _trees(**{other_commit: TAG_TREE}),
    )

    assert verdict.proven is True


def test_failed_and_unresolvable_runs_are_ignored_not_trusted() -> None:
    runs = [
        _run(5, "workflow_dispatch", conclusion="failure"),
        _run(6, "workflow_dispatch", head_sha="deadbeef"),
    ]

    assert gate.assess(TAG_TREE, runs, _trees(**{HEAD_SHA: TAG_TREE})).proven is False


def _cli(tmp_path: Path, runs: list[dict], trees: dict[str, str]) -> subprocess.CompletedProcess:
    runs_json = tmp_path / "runs.json"
    trees_json = tmp_path / "trees.json"
    runs_json.write_text(json.dumps({"workflow_runs": runs}))
    trees_json.write_text(json.dumps(trees))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--tag-tree", TAG_TREE,
         "--runs-json", str(runs_json), "--trees-json", str(trees_json)],
        capture_output=True, text=True,
    )


def test_cli_exits_nonzero_and_explains_itself_on_a_pr_only_proof(tmp_path: Path) -> None:
    proc = _cli(
        tmp_path,
        [{"id": 7, "head_sha": HEAD_SHA, "event": "pull_request", "conclusion": "success"}],
        {HEAD_SHA: TAG_TREE},
    )

    assert proc.returncode == 1
    assert "::error::" in proc.stderr
    assert gate.ONLY_PULL_REQUEST_PROOF in proc.stderr
    assert "gh workflow run ci.yaml" in proc.stderr


def test_cli_exits_zero_on_a_dispatched_proof(tmp_path: Path) -> None:
    proc = _cli(
        tmp_path,
        [{"id": 8, "head_sha": HEAD_SHA, "event": "workflow_dispatch", "conclusion": "success"}],
        {HEAD_SHA: TAG_TREE},
    )

    assert proc.returncode == 0
    assert "PROVEN" in proc.stdout


def test_publish_workflow_calls_the_gate_and_keeps_no_inline_matcher() -> None:
    """A gate that can be quietly replaced by the shell loop it fixed is not a gate."""
    text = PUBLISH_YAML.read_text()

    assert "scripts/assert_ci_proof.py" in text
    assert 'select(.conclusion=="success")' not in text, (
        "publish.yaml matches CI runs inline again — the pgw#1191 rule "
        "(a pull_request run proves the MERGE tree, not the head) is bypassed"
    )
