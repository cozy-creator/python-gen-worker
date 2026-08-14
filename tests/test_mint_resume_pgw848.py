"""pgw#848 item 5: crash-only mint — driven for real, off-pod.

The loss this closes, measured: a mint is ~74 min of serial export (2.06 and
2.07 min/row across two independent pods, 36 sdxl entries) and then ~626 s of
AOTI compile PER ENTRY. A crash at entry 30 of 36 used to discard ~5.2 h of
paid compile, because ``build_cell`` hands every attempt a fresh ``child-<n>``
workdir and the pool's inductor cache lives inside it.

Everything here runs the REAL :class:`EntryCompilePool` — real
``torch.export`` programs, real ``aot_compile`` children, real g++ — on CPU.
The crash is a real ``SIGKILL`` of a real process, not a raised exception, so
the bank has to survive with no ``finally`` and no ``atexit`` (which is exactly
the shape a pod's OOM killer and a drain both take).

The refusal tests are the ones that matter. A resume that trusts a path is a
way to pack a stale artifact into a cell that then verifies, arms and is wrong
— the worst failure pgw#846 leaves available — so a banked entry is re-admitted
ONLY when its identity re-derives from the program THIS attempt exported.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import aot_resume
from harness.progress_wait import Cadence, await_count

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

#: Small enough that the suite stays affordable, wide enough that the compile
#: is real work (real codegen, a real g++ invocation).
_HIDDEN = 64


def _program(seed: int) -> Any:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(_HIDDEN, _HIDDEN)
            self.b = torch.nn.Linear(_HIDDEN, _HIDDEN)

        def forward(self, x: Any) -> Any:
            y = torch.relu(self.a(x)) * (1.0 + seed)
            return torch.tanh(self.b(y)) + y

    return torch.export.export(Tiny(), (torch.randn(4, _HIDDEN),))


def _name(index: int) -> str:
    # The real shape: entry names carry '/' and '=', which is why the bank
    # stores by digest and records the name.
    return f"unet/adapter=true/dim={index}"


def _entries(count: int) -> List[Tuple[str, Any]]:
    return [(_name(i), _program(i)) for i in range(count)]


def _width(entries: int, workers: int) -> pool.PoolWidth:
    # STATED, not derived: a 4-vCPU runner honestly derives K=1 and these
    # scenarios would then pass while exercising no pool at all.
    return pool.entry_workers(
        entries, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)


def _pool_with_root(workdir: Path, count: int, root: str) -> pool.EntryCompilePool:
    """Construct a pool with the resume root installed the production way
    (`aot_resume.set_root`, the process global — pgw#1030 deleted the
    redundant `resume_dir` constructor param). The root only needs to span
    construction: `open_bank` runs in `__init__`."""
    aot_resume.set_root(root)
    try:
        return pool.EntryCompilePool(
            workdir, width=_width(count, 1),
            inductor_configs={"compile_threads": 2})
    finally:
        aot_resume.set_root("")


# ---------------------------------------------------------------------------
# The refusals. These are the tests that matter.
# ---------------------------------------------------------------------------


def _bank_one(tmp_path: Path, index: int = 0) -> Tuple[Path, str, Any]:
    """Bank one entry's compiled files. Returns ``(root, entry, program)``.

    Driven straight at :class:`aot_resume.EntryBank` since pgw#1215. It used to
    run a real ``EntryCompilePool`` over a real four-linear program, because
    the pool was the bank's production driver — it is not any more: a compile
    child builds and traces its own share, so the parent holds no
    ExportedProgram to re-derive a graph hash from and can neither admit nor
    put. ⚠️ THE BANK CURRENTLY HAS NO PRODUCTION DRIVER AT ALL (see
    ``EntryCompilePool.__init__``); re-homing crash-resume at the packed
    graph-class artifact is owed by pgw#1215 step 3/4. The refusals below are
    the bank's own and are exercised unchanged — they are what a re-homed
    driver has to keep.
    """
    from gen_worker import graph_hash as graph_hash_mod

    resume = tmp_path / "resume"
    entry, program = _name(index), _program(index)
    bank = aot_resume.open_bank(
        str(resume), inductor_configs={"compile_threads": 2})
    assert bank is not None
    assert bank.admit(entry, program).reason == aot_resume.MISS, (
        "a fresh bank must MISS, or the rows below prove nothing")
    loose = tmp_path / f"wrapper-{index}.so"
    loose.write_bytes(b"compiled bytes for " + entry.encode())
    bank.put(entry, graph_hash_mod.graph_hash(program), [str(loose)])
    assert bank.banked_bytes > 0
    return resume, entry, program


def _reopen(resume: Path, **kw: Any) -> Any:
    bank = aot_resume.open_bank(
        str(resume), inductor_configs=kw.pop("inductor_configs",
                                             {"compile_threads": 2}))
    assert bank is not None
    return bank


def test_a_tampered_banked_artifact_is_refused(tmp_path: Path) -> None:
    """The banked bytes are re-hashed, so an edited artifact cannot serve."""
    resume, entry, program = _bank_one(tmp_path)
    files = sorted((resume / aot_resume.ENTRIES_DIR).glob(
        f"*/{aot_resume.FILES_DIR}/*"))
    assert files, "the bank must hold the compiled files it claims to"
    victim = max(files, key=lambda p: p.stat().st_size)
    before = victim.stat().st_size
    with victim.open("r+b") as handle:      # one byte, in place: size unchanged
        handle.seek(0)
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))
    assert victim.stat().st_size == before, (
        "the tamper must not change the SIZE — otherwise this test proves "
        "only that a size check works, and the sha256 is never exercised")

    second = _reopen(resume)
    admission = second.admit(entry, program)
    assert not admission.ok, "a tampered artifact must NOT be admitted"
    assert admission.reason == aot_resume.REFUSE_FILE_CONTENT
    assert second.resumed == []
    assert second.facts()["resume_refused"] == {
        aot_resume.REFUSE_FILE_CONTENT: 1}, (
        "a refusal is hub-visible on the pool row; a bank that silently "
        "refuses every entry looks exactly like a slow mint")


def test_a_stale_entry_is_refused_by_a_re_derived_graph_hash(
    tmp_path: Path,
) -> None:
    """THE safety property: same entry NAME, different graph — refused.

    This is the pgw#846 failure the whole design exists to prevent. The name,
    the path, the file digests and every context axis all still match; the only
    thing that moved is the graph the entry is FOR. If the check were on the
    path — or on a hash read back out of the artifact — this cell would pack a
    stale compile, verify, arm, and be wrong.
    """
    from gen_worker import graph_hash as graph_hash_mod

    resume, entry, original = _bank_one(tmp_path)
    other = _program(99)        # a different graph under the same entry name
    assert graph_hash_mod.graph_hash(other) != graph_hash_mod.graph_hash(
        original), (
        "the two programs must genuinely differ, or this test cannot fail")

    second = _reopen(resume)
    admission = second.admit(entry, other)
    assert not admission.ok and admission.reason == aot_resume.REFUSE_GRAPH
    assert second.resumed == []
    # ...and the ORIGINAL graph still admits, so the refusal is about the
    # graph and not about the bank being unusable.
    assert _reopen(resume).admit(entry, original).ok


def test_the_context_axes_are_fail_closed(tmp_path: Path) -> None:
    """A moved runtime refuses, and an UNSTATABLE one refuses too.

    "Unchanged" and "unknown" are different facts. An artifact whose toolchain,
    seal or sm this runtime cannot state is not admissible — the axis being
    empty is precisely when a wrong answer is undetectable.
    """
    resume, entry, program = _bank_one(tmp_path)

    configs = {"compile_threads": 2}      # the ones it was banked under
    moved = aot_resume.open_bank(str(resume), inductor_configs=configs)
    assert moved is not None
    moved.context["toolchain"] = "a-toolchain-that-was-not-there-before"
    assert moved.admit(entry, program).reason == aot_resume.REFUSE_CONTEXT

    unstated = aot_resume.open_bank(str(resume), inductor_configs=configs)
    assert unstated is not None
    unstated.context["sm"] = aot_resume.UNSTATED
    assert unstated.admit(entry, program).reason == aot_resume.REFUSE_UNSTATED

    # The inductor configs are an axis too: they are compared WHOLE, including
    # ones pgw#757 measured as non-identity (`compile_threads`). Refusing too
    # often costs a recompile; admitting too often costs a cell.
    reconfigured = aot_resume.open_bank(str(resume), inductor_configs={})
    assert reconfigured is not None
    assert reconfigured.admit(entry, program).reason == aot_resume.REFUSE_CONTEXT

    # ...and the same bank, unmodified, still admits: a refusal test that
    # cannot also show the positive proves only that the check is broken.
    good = aot_resume.open_bank(
        str(resume), inductor_configs={"compile_threads": 2})
    assert good is not None
    assert good.admit(entry, program).ok, good.outcomes


def test_a_bank_torn_mid_write_is_a_cold_miss_not_a_short_entry(
    tmp_path: Path,
) -> None:
    """The ledger is written LAST, and atomically.

    A crash between copying the files and recording them must leave nothing
    admissible. The failure mode this forbids is a slot that admits with fewer
    files than the entry has.
    """
    resume, entry, program = _bank_one(tmp_path)
    slot = next((resume / aot_resume.ENTRIES_DIR).iterdir())
    ledger = json.loads((slot / aot_resume.LEDGER_NAME).read_text())
    assert ledger["v"] == aot_resume.BANK_V and ledger["entry"] == entry
    assert ledger["graph_hash"] and ledger["files"]

    (slot / aot_resume.LEDGER_NAME).unlink()    # the crash-mid-copy shape
    torn = aot_resume.open_bank(
        str(resume), inductor_configs={"compile_threads": 2})
    assert torn is not None
    assert torn.admit(entry, program).reason == aot_resume.MISS

    # A ledger from another bank format is refused rather than partly read.
    (slot / aot_resume.LEDGER_NAME).write_text(
        json.dumps({**ledger, "v": aot_resume.BANK_V + 1}))
    other = aot_resume.open_bank(
        str(resume), inductor_configs={"compile_threads": 2})
    assert other is not None
    assert other.admit(entry, program).reason == aot_resume.REFUSE_FORMAT


def test_the_bank_outlives_an_abandoned_mint(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Abandonment is how a CRASHED mint ends, so the bank must not be under
    the thing abandonment deletes.

    ``fleet_cells.abandon_self_mint`` rmtree's ``mint_root``. A bank sited
    there would be destroyed on its way out of the one case it exists for —
    everything else here would recover ~5.2 h and the cleanup would delete it.
    Driven through the REAL abandon path, not an assertion about a path string.
    """
    from types import SimpleNamespace

    from gen_worker import fleet_cells, local_cell_store, mint_process

    monkeypatch.setenv(
        local_cell_store.ENV_STORE_DIR, str(tmp_path / "store"))
    mint_root = tmp_path / "selfmint-abc"
    (mint_root / "capture").mkdir(parents=True)
    key = "ck1:sdxl:deadbeef"

    # 1. The request the parent actually builds points OUTSIDE the mint tree.
    request = mint_process.build_request(
        mint_process.MintTask(
            pending=SimpleNamespace(
                family="sdxl", arm_token=key, mint_root=mint_root,
                cfg=SimpleNamespace(shapes=(), targets=(), family="sdxl")),
            pipe=None, function="generate", modules=("m",)),
        workdir=mint_root / "child-1")
    assert request.resume == str(aot_resume.bank_root(key))
    assert not Path(request.resume).is_relative_to(mint_root), (
        "a bank under mint_root is deleted by the abandon path — which is the "
        "path a crashed mint takes")

    # 2. Bank something real-shaped there, then abandon for real.
    bank = aot_resume.open_bank(request.resume, inductor_configs={})
    assert bank is not None
    loose = tmp_path / "wrapper.so"
    loose.write_bytes(b"compiled bytes")
    program = _program(0)
    from gen_worker import graph_hash as graph_hash_mod

    bank.put(_name(0), graph_hash_mod.graph_hash(program), [str(loose)])

    pending = fleet_cells.PendingSelfMint(
        family="sdxl", arm_token=key, ref=f"repo#{key}",
        cfg=SimpleNamespace(family="sdxl"), target=mint_root / "cell.tar.gz", mint_root=mint_root,
        publisher=None)
    fleet_cells.abandon_self_mint(pending)

    assert not mint_root.exists(), (
        "the abandon path must still clean the mint tree up — this test would "
        "be vacuous if it did not")
    after = aot_resume.open_bank(request.resume, inductor_configs={})
    assert after is not None
    admission = after.admit(_name(0), program)
    assert admission.ok, (
        f"the banked entry did not survive abandonment: {after.outcomes!r}")

    # 3. ...and the ADOPTED terminus is the one that drops it.
    aot_resume.discard(key)
    assert not Path(request.resume).exists()


def test_the_resume_area_is_capacity_bounded(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """A bank that survives every failure is a bank that grows without bound
    on an unhealthy pod, so the area is capped and swept oldest scope first."""
    monkeypatch.setenv(local_cells_env(), str(tmp_path / "store"))
    monkeypatch.setenv(aot_resume.ENV_MAX_BYTES, "4096")
    keep = aot_resume.bank_root("keep-me")
    for scope, size in (("old", 4096), ("older", 4096)):
        root = aot_resume.bank_root(scope)
        (root / aot_resume.ENTRIES_DIR).mkdir(parents=True)
        (root / "blob").write_bytes(b"x" * size)
        os.utime(root, (1, 1))          # deliberately the oldest scopes
    (keep / aot_resume.ENTRIES_DIR).mkdir(parents=True)
    (keep / "blob").write_bytes(b"x" * 4096)

    assert aot_resume.sweep(keep) == 2
    assert keep.exists(), "the scope being opened is never the one swept"
    assert not aot_resume.bank_root("old").exists()


def local_cells_env() -> str:
    from gen_worker import local_cell_store

    return local_cell_store.ENV_STORE_DIR


def test_no_resume_root_means_no_bank_and_no_behaviour_change() -> None:
    """The off path is the untouched path: no admission pass, no hashing, no
    copies, and the inductor cache stays where it was."""
    assert aot_resume.open_bank("") is None
    assert aot_resume.root() == ""
