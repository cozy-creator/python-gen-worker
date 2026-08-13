"""pgw#1111: the META round-trip that lets a WEIGHT-FREE program cross the
entry-compile pool's process boundary.

Why this exists, measured rather than asserted
----------------------------------------------
``fc77b923`` made every production mint weight-free, and
``aot_mint`` then FORCED ``parallel = False`` for exactly those mints because a
fake-parameter ``ExportedProgram`` could not survive ``torch.export.save`` ->
``load``. So the entry pool became dead code fleet-wide and every mint ran K=1,
in-process, with export and compile strictly sequential.

The cost is on the record, hub-side: the sdxl A40 mint
(release ``6ee9b4d4df2697a53da6f43a``, pod ``bgmdxhazxsugmk``, gen-worker
0.112.0) reported ``export_s`` 1378.52 + ``compile_s`` 2065.36 = 3443.88 s
against ``total_s`` 3463.84 s — **0.6 % apart**, which is what "no overlap at
all" looks like. Its ``pool`` row said ``entry_workers=3`` and carried NO
ledger (no ``pool_wall_s``, no ``pool_efficiency``, no ``peak_concurrency``),
because ``progress.width`` is recorded whether or not the width is used: the
3 was the width the override threw away.

The tests below prove the premise the override rested on is fixable, not that
it was imaginary — ``test_a_FAKE_param_program_CANNOT_round_trip`` reproduces
the original ``RuntimeError`` verbatim, and is the RED half of every other test
in this file.

Everything here exports only. No inductor, no autotune, no card.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Iterator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"
if str(MICRO_SRC) not in sys.path:
    sys.path.insert(0, str(MICRO_SRC))

from gen_worker import aot_mint  # noqa: E402
from gen_worker.models import structure_only as so  # noqa: E402


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from micro_diffusion.weights import SEED, materialize

    root = tmp_path_factory.mktemp("micro-tree")
    return materialize(root, seed=SEED)


def _weightless_program(tree: Path):
    """A real structure-only export through the real seams — the same
    ``build_component`` -> ``export_program`` pair the mint runs."""
    module, _facts = so.build_component(tree, "decoder", device="cpu")
    mode = so.fake_mode_of(module)
    assert mode is not None
    with so.under(mode):
        latent = torch.randn(1, 8, 16)
        program = aot_mint.export_program(module, (latent,), {})
    return program


@contextlib.contextmanager
def _quiet_serde() -> Iterator[None]:
    """torch's deserializer logs ``"...type %s after initial failure: %s"``
    with ONE argument, so emitting it raises ``TypeError`` — and pytest's
    capture handler re-raises formatting errors instead of swallowing them.
    It fires on any program whose EXAMPLE INPUTS are fake, which is every
    exported program, weight-free or not; a real-weight mint has taken this
    same path through the pool since pgw#809. Upstream's bug, muted here so
    the assertions below are about this change.
    """
    log = logging.getLogger("torch._export.serde.serialize")
    was, log.disabled = log.disabled, True
    try:
        yield
    finally:
        log.disabled = was


def _load(path: Path):
    with _quiet_serde():
        return torch.export.load(str(path))


def _graph_text(program) -> str:
    return str(program.graph_module.graph)


def _modes(program):
    """``(state-dict modes, example-input modes)`` as identity sets."""
    tables = list(program.state_dict.values())
    tables += list(getattr(program, "constants", {}).values())
    sd = {id(m) for m in (getattr(t, "fake_mode", None) for t in tables)
          if m is not None}
    args = program.example_inputs[0] or ()
    ins = {id(m) for m in (getattr(t, "fake_mode", None) for t in args)
           if m is not None}
    return sd, ins


# ---------------------------------------------------------------------------
# RED: the premise the serial override rested on
# ---------------------------------------------------------------------------


def test_a_FAKE_param_program_CANNOT_round_trip(tree: Path, tmp_path: Path) -> None:
    """The original failure, reproduced verbatim on the pin. Every other test
    in this file is only interesting because this one holds."""
    program = _weightless_program(tree)
    path = tmp_path / "fake.pt2"
    torch.export.save(program, str(path))
    with pytest.raises(RuntimeError) as caught, _quiet_serde():
        torch.export.load(str(path))
    assert "deserializing the saved file" in str(caught.value)


# ---------------------------------------------------------------------------
# GREEN: META crosses, and crosses IDENTICALLY
# ---------------------------------------------------------------------------


def test_the_META_round_trip_preserves_the_GRAPH(tree: Path, tmp_path: Path) -> None:
    """Parallelism is time-only. A round-tripped program whose graph differed
    from the serial one would publish a DIFFERENT artifact under the SAME cell
    key (the key digests the traced graph, sm, toolchain and env seal — none of
    which move here), which is this codebase's worst failure class."""
    program = _weightless_program(tree)
    before = _graph_text(program)

    path = tmp_path / "meta.pt2"
    with so.as_meta_for_save(program) as moved:
        assert moved > 0, "a weight-free program must have fake params to move"
        torch.export.save(program, str(path))
    loaded = _load(path)

    assert so.has_meta_params(loaded)
    assert so.revirtualize_from_meta(loaded) is not None
    assert not so.has_meta_params(loaded)
    assert _graph_text(loaded) == before


def test_the_round_tripped_program_shares_ONE_fake_mode(
    tree: Path, tmp_path: Path,
) -> None:
    """``aot_compile`` asserts every input belongs to one mode, and
    ``compile_entry_files`` reads that mode off the program itself. Params
    re-virtualized into a mode of their own would fail that assertion — or
    worse, pass it while ``fake_mode_of_program`` returned None."""
    program = _weightless_program(tree)
    path = tmp_path / "meta.pt2"
    with so.as_meta_for_save(program):
        torch.export.save(program, str(path))
    loaded = _load(path)
    so.revirtualize_from_meta(loaded)

    sd, ins = _modes(loaded)
    assert sd and ins, "neither side may be mode-less"
    assert sd == ins, "params and example inputs must share ONE fake mode"
    assert so.fake_mode_of_program(loaded) is not None


def test_a_meta_program_that_was_NOT_revirtualized_flips_the_INDUCTOR_CONFIG(
    tree: Path, tmp_path: Path,
) -> None:
    """The severance that says why the child's call site is load-bearing.

    ``compile_entry_files`` selects its inductor options with
    ``weightless=fake_mode_of_program(program) is not None``. A program left on
    META answers None, so skipping the re-virtualization does not merely fail —
    it compiles the *weight-bearing* config for a weight-free graph and
    publishes it under the weight-free cell key. Silent, and wrong.
    """
    program = _weightless_program(tree)
    path = tmp_path / "meta.pt2"
    with so.as_meta_for_save(program):
        torch.export.save(program, str(path))

    loaded = _load(path)
    assert so.fake_mode_of_program(loaded) is None, (
        "un-revirtualized: the config selector would read `weightless=False`")

    so.revirtualize_from_meta(loaded)
    assert so.fake_mode_of_program(loaded) is not None, (
        "revirtualized: the selector reads `weightless=True`, as the serial "
        "path does")


# ---------------------------------------------------------------------------
# The PRODUCTION call sites — the pool's stage and the child's load
# ---------------------------------------------------------------------------


def test_the_POOLS_OWN_STAGE_writes_a_program_the_CHILDS_OWN_LOAD_can_read(
    tree: Path, tmp_path: Path,
) -> None:
    """End to end across the process boundary, through the real
    ``EntryCompilePool._stage`` and the real ``aot_compile_child.load_program``
    — no re-implementation of either. Everything short of the inductor compile,
    which is not a cheap test on any machine.

    Delete either call site and this goes red: without ``as_meta_for_save`` the
    stage's own ``torch.export.save``/child load raises the deserialize error;
    without ``revirtualize_from_meta`` the loaded program answers
    ``fake_mode_of_program() is None`` and the last assertion fails.
    """
    from gen_worker import aot_compile_child, aot_compile_pool

    program = _weightless_program(tree)
    expected = _graph_text(program)
    original = dict(program.state_dict)

    width = aot_compile_pool.entry_workers(
        1, available_bytes=1 << 36, vcpus=8,
        device_lock=True)
    pool = aot_compile_pool.EntryCompilePool(tmp_path / "pool", width=width)
    job, _job_path = pool._stage("decoder", program, 0)

    assert pool.meta_staged_entries == 1, "the stage must have cast to META"
    assert Path(job.program).exists()
    # The parent's copy survives the cast, objects and all.
    for name, tensor in original.items():
        assert program.state_dict[name] is tensor

    with _quiet_serde():
        loaded = aot_compile_child.load_program(job)

    assert not so.has_meta_params(loaded)
    assert _graph_text(loaded) == expected
    assert so.fake_mode_of_program(loaded) is not None, (
        "the child must hand `compile_entry_files` a program whose own fake "
        "mode it can find — that is what selects the weight-free config")


# ---------------------------------------------------------------------------
# The cast must not outlive the save
# ---------------------------------------------------------------------------


def test_the_save_cast_gives_back_the_EXACT_original_tensors(
    tree: Path, tmp_path: Path,
) -> None:
    """The parent keeps using its programs after staging them (dispatch-class
    canonicalization, the resident release's weight aliases). Restoring
    equivalent new fakes would satisfy every value check and still break an
    identity comparison, so the contract is object identity."""
    program = _weightless_program(tree)
    original = dict(program.state_dict)
    assert original, "the fixture must have a state dict to protect"

    with so.as_meta_for_save(program):
        assert all(str(t.device) == "meta" for t in program.state_dict.values())
        torch.export.save(program, str(tmp_path / "meta.pt2"))

    assert list(program.state_dict) == list(original)
    for name, tensor in original.items():
        assert program.state_dict[name] is tensor, f"{name} was not the original"


def test_a_REAL_weight_program_is_untouched_by_the_cast() -> None:
    """The context is a no-op on the path it does not own. `moved == 0` is the
    fence: a real-weight mint must serialize exactly as it always has."""

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.lin(x)

    program = torch.export.export(Tiny(), (torch.randn(2, 4),))
    before = dict(program.state_dict)
    with so.as_meta_for_save(program) as moved:
        assert moved == 0
        assert all(str(t.device) != "meta" for t in program.state_dict.values())
    assert dict(program.state_dict) == before


# ---------------------------------------------------------------------------
# The override is gone: a weight-free mint may now run the width it computed
# ---------------------------------------------------------------------------


def test_aot_mint_no_longer_FORCES_a_structure_only_mint_serial() -> None:
    """Severance-style: the deleted branch was ``if parallel and
    is_structure_only(pipeline): parallel = False``. Nothing else in the module
    may reintroduce a structure-only test on the width decision — if it does,
    every weight-free mint silently returns to K=1 and no test in the tree
    noticed the first time (there was none).
    """
    source = (REPO / "src" / "gen_worker" / "aot_mint.py").read_text()
    head, _, _ = source.partition("    minted = progress.minted")
    _, _, decision = head.rpartition("    parallel = width.workers > 1")
    assert "is_structure_only" not in decision, (
        "the width decision must not re-acquire a structure-only override")
    assert "compiles SERIALLY" not in source
