#!/usr/bin/env python3
"""pgw#1191: a tag publishes only against a CI run that BUILT the tagged tree.

pgw#795 closed the "published a tree no CI ever saw" hole (v0.78.0) by matching
the tag's tree against a green `ci.yml` run. The match key is the run's
`head_sha` -- and that is where the hole reopened through a different door.

`ci.yml` checks out with `actions/checkout@v4` and no `ref:`. On a
`pull_request` event GitHub therefore builds `refs/pull/<n>/merge` -- the PR
head merged with whatever `master` is at that moment -- while still recording
the PR BRANCH HEAD as the run's `head_sha`. So a green PR run names a commit it
never built, and whenever master moves under a cut (the normal case: master
moved four times during the 0.113.0 cut) the certified tree and the executed
tree differ by exactly the work that landed underneath.

Only events whose checkout IS the commit they name are admissible as proof:
`workflow_dispatch` and `push` check out the ref itself. A `pull_request` run
remains the right gate for MERGING; it is not proof for PUBLISHING.

The refusal is typed and named so a releaser reading a red publish knows the
rule rather than guessing at a flake -- `only_pull_request_proof` in particular
means "your tree is fine, your EVIDENCE is the wrong kind; dispatch CI on the
tag and re-run publish".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# A run's checkout IS the commit it names only for these events. Anything else
# is inadmissible BY DEFAULT: a new event type has to be classified here before
# it can prove a release, which is the safe direction for this gate to fail.
ADMISSIBLE_EVENTS = frozenset({"workflow_dispatch", "push"})

PROOF_FOUND = "proof_found"
ONLY_PULL_REQUEST_PROOF = "only_pull_request_proof"
NO_RUN_CARRIES_TREE = "no_run_carries_tree"


@dataclass(frozen=True)
class Run:
    """One `ci.yml` run, as the Actions API reports it."""

    id: int
    head_sha: str
    event: str
    conclusion: str

    @staticmethod
    def from_api(row: dict) -> "Run":
        return Run(
            id=int(row.get("id") or 0),
            head_sha=str(row.get("head_sha") or ""),
            event=str(row.get("event") or ""),
            conclusion=str(row.get("conclusion") or ""),
        )


@dataclass(frozen=True)
class Verdict:
    proven: bool
    kind: str
    detail: str
    run_id: int | None = None


TreeResolver = Callable[[str], str | None]


def assess(tag_tree: str, runs: Iterable[Run], tree_of: TreeResolver) -> Verdict:
    """Decide whether any run PROVES `tag_tree`.

    Two passes on purpose: the happy path resolves trees only for admissible
    runs, and the second pass exists solely to tell a releaser whose only
    evidence is a PR run what is actually wrong.
    """
    successes = [r for r in runs if r.conclusion == "success" and r.head_sha]

    seen: set[str] = set()
    for run in successes:
        if run.event not in ADMISSIBLE_EVENTS or run.head_sha in seen:
            continue
        seen.add(run.head_sha)
        if tree_of(run.head_sha) == tag_tree:
            return Verdict(
                True,
                PROOF_FOUND,
                f"{run.event} run {run.id} on {run.head_sha} built this exact tree",
                run.id,
            )

    seen.clear()
    for run in successes:
        if run.event in ADMISSIBLE_EVENTS or run.head_sha in seen:
            continue
        seen.add(run.head_sha)
        if tree_of(run.head_sha) == tag_tree:
            return Verdict(
                False,
                ONLY_PULL_REQUEST_PROOF,
                f"the only green run naming this tree is a {run.event} run "
                f"({run.id}, head {run.head_sha}), which built refs/pull/<n>/merge "
                f"-- your head merged with master -- and NOT this tree",
                run.id,
            )

    return Verdict(False, NO_RUN_CARRIES_TREE, "no green run names this tree at all")


def _gh_json(args: Sequence[str]) -> object:
    out = subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out or "null")


def _api_runs(repo: str, per_page: int) -> list[Run]:
    payload = _gh_json(
        ["api", f"repos/{repo}/actions/workflows/ci.yml/runs?per_page={per_page}"])
    rows = (payload or {}).get("workflow_runs", []) if isinstance(payload, dict) else []
    return [Run.from_api(r) for r in rows]


def _api_tree_resolver(repo: str) -> TreeResolver:
    cache: dict[str, str | None] = {}

    def resolve(sha: str) -> str | None:
        if sha not in cache:
            try:
                payload = _gh_json(["api", f"repos/{repo}/commits/{sha}"])
                commit = (payload or {}).get("commit", {}) if isinstance(payload, dict) else {}
                cache[sha] = str(commit.get("tree", {}).get("sha") or "") or None
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                cache[sha] = None  # a commit the API cannot resolve proves nothing
        return cache[sha]

    return resolve


REFUSAL_HELP = """\
This tag's content has NOT been proven AS TAGGED, so it will not be published.

Only a `workflow_dispatch` or `push` run proves a tree: those check out the ref
they name. A `pull_request` run checks out `refs/pull/<n>/merge` -- your head
merged with whatever master is at that moment -- while recording your branch
head as its `head_sha`, so it names a tree it never built (pgw#1191; the same
class of hole pgw#795 closed for v0.78.0).

The fix is one CI run:

    gh workflow run ci.yml --ref <this-tag>     # or the branch you will tag

then re-run this publish job once it is green.

Do NOT relax this check to unblock a release."""


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-tree", required=True, help="tree sha being published")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--runs-json", type=Path,
                    help="pre-fetched runs payload; skips the API (tests)")
    ap.add_argument("--trees-json", type=Path,
                    help="head_sha -> tree map; skips the API (tests)")
    args = ap.parse_args(argv)

    if args.runs_json:
        payload = json.loads(args.runs_json.read_text())
        rows = payload.get("workflow_runs", payload) if isinstance(payload, dict) else payload
        runs = [Run.from_api(r) for r in rows]
    else:
        if not args.repo:
            ap.error("--repo (or $GITHUB_REPOSITORY) is required without --runs-json")
        runs = _api_runs(args.repo, args.per_page)

    if args.trees_json:
        trees = json.loads(args.trees_json.read_text())
        tree_of: TreeResolver = lambda sha: trees.get(sha)  # noqa: E731
    else:
        if not args.repo:
            ap.error("--repo (or $GITHUB_REPOSITORY) is required without --trees-json")
        tree_of = _api_tree_resolver(args.repo)

    verdict = assess(args.tag_tree, runs, tree_of)
    if verdict.proven:
        print(f"PROVEN ({verdict.kind}): {verdict.detail}")
        return 0

    print(f"::error::publish refused [{verdict.kind}] for tree {args.tag_tree}: "
          f"{verdict.detail}", file=sys.stderr)
    print(REFUSAL_HELP, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
