"""`gen-worker lock` as a COMMITTED SOURCE artifact: deterministic, checkable.

endpoint.lock lives in git beside the endpoint, so two properties matter that
did not matter when it was scratch output:

* **determinism** — locking twice writes the same bytes, or every diff is
  noise and nobody reads them;
* **`--check`** — the freshness gate. It answers "does this committed lock
  still describe this tree" and WRITES NOTHING, so CI can run it on a
  read-only checkout. The cheap arm is the inputs digest (milliseconds); only
  when the inputs moved does it pay for a re-derive, and then it compares the
  DOCUMENT, because an input that moved without moving the output is not
  drift and reporting it as drift trains people to ignore the check.

Integration through the real CLI against a real endpoint project on disk — no
mocks. The endpoint declares NO layout contract and marks NO compile target,
which is deliberate on both counts: it is pgw#1488's traced-by-default state
(before this change the derive refused the class outright), and it keeps the
gate's own test off the export path so a lock/`--check` regression cannot hide
behind a torch failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

MAIN = '''
from __future__ import annotations

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint


class Unlaned(msgspec.Struct):
    """A model type with no canonical contract."""

    steps: int = 4


class In(msgspec.Struct):
    text: str
{extra_field}

class Out(msgspec.Struct):
    echoed: str


class UnlanedModel(
    Model[Unlaned],
    self_loading="a lock/--check gate fixture: there is no pipeline at all",
):
    """No `lanes=`, no contract, no ctx.compile — and it locks."""

    def load(self, ctx: LoadContext[Unlaned]) -> None:
        self.ready = True


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: UnlanedModel) -> Out:
    return Out(echoed=payload.text)
'''


def _write_endpoint(root: Path, *, extra_field: str = "") -> None:
    package = root / "src" / "lockcheck_fixture"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        MAIN.format(extra_field=extra_field), encoding="utf-8"
    )
    (root / "endpoint.toml").write_text(
        'schema_version = 1\nmain = "lockcheck_fixture.main"\n', encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "lockcheck-fixture"\nversion = "0"\n\n'
        '[tool.gen_worker]\nmain = "lockcheck_fixture.main"\n',
        encoding="utf-8",
    )


@pytest.fixture()
def endpoint(tmp_path: Path) -> Path:
    root = tmp_path / "endpoint"
    _write_endpoint(root)
    return root


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    tree = tmp_path / "checkpoint"
    tree.mkdir()
    (tree / "config.json").write_text(
        json.dumps({"torch_dtype": "bfloat16"}), encoding="utf-8"
    )
    return tree


def _lock(
    endpoint: Path, checkpoint: Path, graph_cas: Path, *extra: str
) -> tuple[int, dict]:
    """One `gen-worker lock` invocation — as a PROCESS, deliberately.

    In-process would import the author module once and keep it in
    `sys.modules`, so every later "the source changed" arm would be measuring
    a module that never changed. The CLI is a process in production and the
    gate has to be tested as one.
    """

    completed = subprocess.run(
        [sys.executable, "-m", "gen_worker.cli", "lock", str(endpoint),
         "--checkpoint", str(checkpoint), "--graph-cas", str(graph_cas), *extra],
        capture_output=True, text=True, check=False,
    )
    summary = json.loads(completed.stdout) if completed.stdout.strip() else {}
    return completed.returncode, summary


def test_lock_is_deterministic_and_check_gates_on_the_document(
    endpoint: Path,
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    graph_cas = tmp_path / "graph-cas"
    lock = endpoint / "endpoint.lock"

    code, summary = _lock(endpoint, checkpoint, graph_cas)
    assert code == 0
    first = lock.read_bytes()
    assert summary["derive"] == "traced"
    # pgw#1488: the trace RAN on a contract-less class and found nothing
    # marked. That is a posture with a name, not silence.
    assert summary["posture"] == "traced-no-compile-targets"
    assert summary["specializations"] == 0

    # Determinism: the same tree locks to the same BYTES. (The re-run reuses
    # the saved trace and REWRITES the file, so a non-deterministic
    # serialization shows up right here.)
    code, summary = _lock(endpoint, checkpoint, graph_cas)
    assert code == 0
    assert lock.read_bytes() == first
    assert summary["derive"] == "reused"

    # --check, cheap arm: inputs unchanged, nothing re-derived...
    code, summary = _lock(endpoint, checkpoint, graph_cas, "--check")
    assert code == 0
    assert summary["check"] == "current"
    # ...and NOTHING IS WRITTEN.
    assert lock.read_bytes() == first

    # An input moves without moving the output: a comment. --check re-derives
    # and passes, because the document is what the endpoint IS.
    source = endpoint / "src" / "lockcheck_fixture" / "main.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# a comment moves the inputs\n",
        encoding="utf-8",
    )
    code, result = _lock(endpoint, checkpoint, graph_cas, "--check")
    assert code == 0
    assert result["check"] == "current"
    assert result["committed_document_digest"] == result["derived_document_digest"]
    assert lock.read_bytes() == first


def test_check_goes_red_on_a_stale_committed_lock(
    endpoint: Path,
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    """The gate's whole job: a lock that no longer describes the endpoint."""

    graph_cas = tmp_path / "graph-cas"
    lock = endpoint / "endpoint.lock"
    assert _lock(endpoint, checkpoint, graph_cas)[0] == 0
    before = lock.read_bytes()

    # A REAL change to what this endpoint IS: the entrypoint's payload gains a
    # field, so the published envelope schema — and the document with it —
    # differs from the committed one.
    _write_endpoint(endpoint, extra_field="    drifted: bool = False\n")

    code, result = _lock(endpoint, checkpoint, graph_cas, "--check")
    assert code == 1
    assert result["check"] == "drift"
    assert result["committed_document_digest"] != result["derived_document_digest"]
    # A gate does not fix what it finds.
    assert lock.read_bytes() == before


def test_check_refuses_when_there_is_no_lock_to_check(
    endpoint: Path, checkpoint: Path, tmp_path: Path
) -> None:
    assert _lock(endpoint, checkpoint, tmp_path / "graph-cas", "--check")[0] == 2
