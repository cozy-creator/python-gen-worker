"""`compile` answers a missing exported program by DERIVING it, not by failing.

Lineage: pgw#1525. `cli/compile.py`'s module docstring calls the re-derive
load-bearing — *"when this machine's graph CAS does not hold the program for a
specialization, `compile` RE-DERIVES it locally"* — and on the one state that
sentence is about, a box that has never derived this endpoint, the branch was
unreachable.

Two implementations of one `GraphStore` protocol disagree about what a miss IS:
`torchcg.store.LocalGraphStore.fetch_program` returns `None`;
`serving.mint_store.WorkerGraphStore.fetch_program` — which is what `_store()`
actually builds — RAISES `ProgramBlobUnreachable`, its own message calling the
condition "ORDINARY". `_ensure_program` tested only for `None`, so the raise
escaped past `rederive()`.

MEASURED on a wiped store, canonical master, the real verb with no flags:
`failed=14 (of 14)`, exit 1, **6.04 s**, having never logged "re-deriving" —
and the error told the user to run `compile`, which is what they had just run.
A verb whose job is a half-hour of compilation returning in six seconds is what
a dead branch looks like.

These tests drive `_ensure_program` against BOTH spellings of a miss, because a
fix written against one store is how this happened in the first place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.cli.compile import CompileError, Spec, _ensure_program
from gen_worker.serving.mint_store import ProgramBlobUnreachable

GRAPH = "cg-graph-v1-" + "a" * 56


def spec() -> Spec:
    return Spec(contract="sd15.diffusers-bf16@1", graph=GRAPH, target="unet", ingress=None)


class _Store:
    """A store whose miss spelling and post-derive answer are both dialled in."""

    def __init__(self, *, raises: bool, lands: bool) -> None:
        self.raises = raises
        self.lands = lands
        self.derived = False
        self.looks = 0

    def fetch_program(self, graph: str, destination: Path) -> Path | None:
        self.looks += 1
        if self.derived and self.lands:
            Path(destination).write_bytes(b"an exported program")
            return Path(destination)
        if self.raises:
            raise ProgramBlobUnreachable(
                f"this box holds no serialized program for graph {graph}. That "
                f"is ORDINARY on a pod that has not derived this endpoint yet"
            )
        return None


@pytest.mark.parametrize(
    "raises", [True, False], ids=["miss raises (WorkerGraphStore)", "miss is None (LocalGraphStore)"]
)
def test_a_missing_program_triggers_the_derive_in_both_store_dialects(
    tmp_path: Path, raises: bool
) -> None:
    store = _Store(raises=raises, lands=True)

    def rederive() -> None:
        store.derived = True

    program = _ensure_program(spec(), store, rederive, tmp_path)

    assert store.derived, (
        "the re-derive never ran — this is the pgw#1525 dead branch, and it is "
        "invisible in a green test suite because the verb still exits cleanly"
    )
    assert program.exists()
    assert store.looks == 2, "look, derive, look — never a third round trip"


@pytest.mark.parametrize("raises", [True, False], ids=["raises", "None"])
def test_a_program_still_absent_after_the_derive_is_the_typed_refusal(
    tmp_path: Path, raises: bool
) -> None:
    """Degrading the miss must not swallow a genuine lock/checkout disagreement."""
    store = _Store(raises=raises, lands=False)

    def rederive() -> None:
        store.derived = True

    with pytest.raises(CompileError) as caught:
        _ensure_program(spec(), store, rederive, tmp_path)

    assert store.derived, "it must still have TRIED to derive before refusing"
    assert "lock --check" in str(caught.value), "the refusal names the remedy"


def test_a_present_program_is_returned_without_paying_for_a_derive(
    tmp_path: Path,
) -> None:
    """The cache hit is the common path and must not have gotten slower."""
    store = _Store(raises=False, lands=True)
    store.derived = True  # already on this box

    def rederive() -> None:  # pragma: no cover - must not run
        raise AssertionError("a present program must never trigger a derive")

    assert _ensure_program(spec(), store, rederive, tmp_path).exists()
    assert store.looks == 1


def test_a_malformed_graph_key_refuses_before_paying_for_a_derive() -> None:
    """A wiring bug is not a miss, and no derive can satisfy it.

    Checked by `is_graph_hash` UP FRONT rather than by recognising the store's
    refusal wording — matching on message text would leave this one prose edit
    away from silently paying a two-minute derive per specialization.
    """
    bad = Spec(contract="c", graph="not-a-graph-identity", target="unet", ingress=None)

    def rederive() -> None:  # pragma: no cover - must not run
        raise AssertionError("a malformed key must never trigger a derive")

    with pytest.raises(CompileError) as caught:
        _ensure_program(bad, _Store(raises=True, lands=True), rederive, Path("/tmp"))

    assert "cg-graph-v1" in str(caught.value)
