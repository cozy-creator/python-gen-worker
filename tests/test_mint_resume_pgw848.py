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
# The crash, for real: a child process running a real pool, SIGKILLed mid-run
# ---------------------------------------------------------------------------

#: Both attempts run as SEPARATE PROCESSES, which is the production shape (two
#: mint children across a crash) and not merely convenient. It is also
#: MEASURED to matter: ``env_seal``'s ``loaded_libs`` fact is frozen at boot
#: from the process's own loaded shared objects, so a pytest process and a bare
#: driver produce different seal digests (observed: 5fbe7ce945c21b80 vs
#: 5f7c63e14ef3ec36) and the bank correctly refuses across them. Two identical
#: bare processes agree — verified by running the same seal twice. Attempt 2
#: therefore has to be a peer of attempt 1, not the test runner.
_DRIVER = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    sys.path.insert(0, {tests!r})
    from test_mint_resume_pgw848 import _entries, _width
    from gen_worker import aot_compile_pool as pool, aot_resume, env_seal

    cfg = json.loads(Path(sys.argv[1]).read_text())
    landed = Path(cfg["landed"])
    count = int(cfg["count"])
    # What `mint_child.mint` does before it exports anything: freeze the seal
    # at the same point in the same code path, in every attempt.
    env_seal.establish()

    # The production wiring (mint_child._mint_aot): the bank root is a
    # process global, not a constructor argument (pgw#1030).
    aot_resume.set_root(cfg["resume"])
    box = pool.EntryCompilePool(
        Path(cfg["workdir"]), width=_width(count, 1),
        inductor_configs={{"compile_threads": 2}})

    def tick(name, done, total):
        with landed.open("a") as fh:
            fh.write(json.dumps({{"entry": name, "done": done}}) + "\\n")
            fh.flush()

    out = box.compile(_entries(count), on_entry=tick)
    Path(cfg["summary"]).write_text(json.dumps({{
        "files": {{name: list(files) for name, files in out.items()}},
        "compiled": sorted(box.entry_seconds),
        "resumed": list(box.bank.resumed) if box.bank else [],
        "outcomes": dict(box.bank.outcomes) if box.bank else {{}},
        "cache_dir": str(box.cache_dir),
        "ledger": box.ledger.facts(),
    }}, sort_keys=True))
    """
)


def _driver(tmp_path: Path, name: str, *, count: int, resume_dir: str) -> Path:
    cfg = tmp_path / f"{name}.json"
    cfg.write_text(json.dumps({
        "landed": str(tmp_path / f"{name}-landed.jsonl"),
        "workdir": str(tmp_path / name / "entry-pool"),
        "summary": str(tmp_path / f"{name}-summary.json"),
        "resume": resume_dir,
        "count": count,
    }))
    (tmp_path / f"{name}-landed.jsonl").touch()
    driver = tmp_path / "driver.py"
    if not driver.exists():
        driver.write_text(_DRIVER.format(
            tests=str(Path(__file__).resolve().parent)))
    return cfg


def _spawn(cfg: Path) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(
        [sys.executable, str(cfg.parent / "driver.py"), str(cfg)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True)


def _landed(cfg: Path) -> List[str]:
    path = Path(json.loads(cfg.read_text())["landed"])
    return [
        str(json.loads(line)["entry"])
        for line in path.read_text().splitlines() if line.strip()]


def _run_until_killed(
    tmp_path: Path, *, count: int, kill_after: int, resume_dir: str,
) -> List[str]:
    """Run a real pool in a child process and SIGKILL it after N entries land.

    Returns the entry names that finished before the kill. ``SIGKILL`` because
    a mint that dies on a pod dies that way: the OOM killer and a drain do not
    deliver a handler, so anything a bank did at exit is exactly what the crash
    would take with it.
    """
    cfg = _driver(tmp_path, "attempt-1", count=count, resume_dir=resume_dir)
    proc = _spawn(cfg)

    def gone() -> Any:
        if proc.poll() is None:
            return None
        return (
            f"the pool process exited {proc.returncode} before {kill_after} "
            f"entries landed: {(proc.stderr.read() or b'')[-2000:]!r}")

    try:
        await_count(
            lambda: len(_landed(cfg)), kill_after, what="compiled entries",
            cadence=Cadence(floor_s=300.0), gone=gone, poll_s=0.25)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
    return _landed(cfg)


def _run_to_completion(
    tmp_path: Path, name: str, *, count: int, resume_dir: str,
) -> dict:
    cfg = _driver(tmp_path, name, count=count, resume_dir=resume_dir)
    proc = _spawn(cfg)
    _, err = proc.communicate()
    assert proc.returncode == 0, (err or b"")[-4000:].decode(errors="replace")
    return json.loads(
        Path(json.loads(cfg.read_text())["summary"]).read_text())


# ---------------------------------------------------------------------------
# RED: what a crash costs with no bank
# ---------------------------------------------------------------------------


def test_without_a_bank_a_killed_mint_shares_nothing_with_the_next_attempt(
    tmp_path: Path,
) -> None:
    """The loss, reproduced. This is the pre-fix behaviour, pinned.

    Attempt 1 finishes an entry and is killed. Attempt 2 — a fresh
    ``child-<n>`` workdir, exactly as ``mint_delegate.build_cell`` builds one —
    recompiles EVERY entry, including the finished one, and cannot even get an
    inductor cache hit because the cache lived inside the dead attempt's
    directory.
    """
    finished = _run_until_killed(tmp_path, count=2, kill_after=1, resume_dir="")
    assert finished, "attempt 1 must finish at least one entry to have a loss"

    summary = _run_to_completion(tmp_path, "attempt-2", count=2, resume_dir="")

    assert set(summary["files"]) == {_name(0), _name(1)}
    assert summary["resumed"] == [] and summary["outcomes"] == {}, (
        "no resume root was given, so the pool must behave exactly as it did "
        "before pgw#848 item 5 — no bank at all")
    assert summary["compiled"] == sorted(summary["files"]), (
        "every entry was recompiled — including the one attempt 1 had already "
        "finished, which is the ~626 s/entry this issue exists to stop paying "
        "twice")
    assert not Path(summary["cache_dir"]).is_relative_to(
        tmp_path / "attempt-1"), (
        "attempt 2's inductor cache is a different directory from attempt 1's, "
        "so there is not even an incidental cache hit to recover on")
    assert "resumed" not in summary["ledger"], (
        "a pod with no resume root reads exactly as it did before on the wire")


# ---------------------------------------------------------------------------
# GREEN: the same crash, with the bank
# ---------------------------------------------------------------------------


def test_a_killed_mint_re_admits_its_finished_entries(tmp_path: Path) -> None:
    """The recovery, and the proof that it is content-verified.

    Same real SIGKILL. Attempt 2 gets a fresh workdir and the STABLE resume
    root, re-exports (which is what produces the independent graph hash), and
    the finished entry comes back without a child being spawned for it — with
    the same bytes.
    """
    resume = tmp_path / "resume"
    finished = _run_until_killed(
        tmp_path, count=3, kill_after=1, resume_dir=str(resume))
    assert finished, "attempt 1 must finish at least one entry"
    assert len(finished) < 3, "the kill must land before the pool completes"

    # The bank survived a SIGKILL: nothing here ran a `finally`.
    slots = list((resume / aot_resume.ENTRIES_DIR).iterdir())
    assert len(slots) == len(finished), (
        "every entry that FINISHED must be banked before the crash, and "
        "nothing else — a slot per finished entry, written as it landed")

    summary = _run_to_completion(
        tmp_path, "attempt-2", count=3, resume_dir=str(resume))

    assert set(summary["files"]) == {_name(i) for i in range(3)}
    assert set(summary["resumed"]) == set(finished), (
        f"the finished entries {finished!r} must be re-admitted; "
        f"outcomes={summary['outcomes']!r}")
    assert set(summary["compiled"]) == set(summary["files"]) - set(finished), (
        "a re-admitted entry must not spawn a compile child (the recovered "
        "compile is the whole point), and every entry the bank did NOT hold "
        "must still be compiled normally")
    for banked in finished:
        for path in summary["files"][banked]:
            assert Path(path).is_file()
            assert Path(path).is_relative_to(resume), (
                "a re-admitted entry is served from the bank's own copy — the "
                "copy whose sha256 was just re-verified")

    facts = summary["ledger"]
    assert facts["resumed"] == len(finished), facts
    assert facts["resume_cold"] == 3 - len(finished), facts
    assert facts["resume_root"] == str(resume)
    assert facts.get("resume_banked_bytes", 0) > 0, facts


# ---------------------------------------------------------------------------
# The refusals. These are the tests that matter.
# ---------------------------------------------------------------------------


def _bank_one(tmp_path: Path, index: int = 0) -> Tuple[Path, str, Any]:
    """Compile one entry for real and bank it. Returns (root, entry, program)."""
    resume = tmp_path / "resume"
    entry, program = _name(index), _program(index)
    box = _pool_with_root(tmp_path / "first" / "entry-pool", 1, str(resume))
    box.compile([(entry, program)])
    assert box.bank is not None and box.bank.banked_bytes > 0
    return resume, entry, program


def test_a_tampered_banked_artifact_is_refused_and_recompiled(
    tmp_path: Path,
) -> None:
    """The banked bytes are re-hashed, so an edited artifact cannot serve.

    Driven through the real pool: the refusal must reach the COMPILE, not just
    the bank's return value — a refusal that still handed back the tampered
    path would be worse than no bank at all.
    """
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

    second = _pool_with_root(tmp_path / "second" / "entry-pool", 1, str(resume))
    out = second.compile([(entry, program)])

    assert second.bank is not None
    assert second.bank.resumed == [], "a tampered artifact must NOT be admitted"
    assert second.bank.outcomes[entry] == aot_resume.REFUSE_FILE_CONTENT
    assert entry in second.entry_seconds, (
        "the refusal must fall through to a real compile — a bank may cost a "
        "mint time, never a cell")
    assert out[entry], "the entry is served, from freshly compiled files"
    assert second.ledger.facts()["resume_refused"] == {
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

    second = _pool_with_root(tmp_path / "second" / "entry-pool", 1, str(resume))
    out = second.compile([(entry, other)])

    assert second.bank is not None
    assert second.bank.resumed == []
    assert second.bank.outcomes[entry] == aot_resume.REFUSE_GRAPH
    assert entry in second.entry_seconds, "it must be compiled for real instead"
    assert out[entry]


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

    from gen_worker import fleet_cells, local_cell_store, mint_delegate

    monkeypatch.setenv(
        local_cell_store.ENV_STORE_DIR, str(tmp_path / "store"))
    mint_root = tmp_path / "selfmint-abc"
    (mint_root / "capture").mkdir(parents=True)
    key = "ck1:sdxl:deadbeef"

    # 1. The request the parent actually builds points OUTSIDE the mint tree.
    request = mint_delegate.build_request(
        mint_delegate.MintTask(
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
