"""Pooled entry children do not re-pay the toolchain hash.

``env_seal.establish()`` SHA-256s every toolchain ``.so`` the image ships
(36 files, 3.96 GB, 8.13 s). A per-PROCESS memo cannot help: the pool's unit of
parallelism is a process that compiles one entry and exits, so a 72-entry mint
re-pays the pass 72 times, K-wide (28 % of per-entry compile_s).

The memo is therefore parent-seeded and on disk, keyed by
``(path, mtime_ns, size)``. This file holds it to two standards:

* **The seal value may never move.** The memo changes WHERE a digest comes
  from, never what it is. The equivalence tests below prove memo-served and
  full-rehash seals byte-identical on the same tree, and prove that any
  detectable mutation forces the fallback rehash. The one UNDETECTABLE case
  — content rewritten at the same size with mtime_ns restored — is asserted
  explicitly as stale, because that is the same trust boundary the
  in-process ``lru_cache`` (keyed on the same triple) has always had, and
  claiming more would be a lie.
* **The drop is measured on the real pool** (real children, real
  ``aot_compile``), against a cold-pass reference measured in the same run —
  not a magic number.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator

import pytest
from torch_compiled_graphs import spans

from gen_worker import aot_compile_pool as pool
from gen_worker import env_seal

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_HIDDEN = 96


# ---------------------------------------------------------------------------
# Seal equivalence: the memo may change the SOURCE of a digest, never the value
# ---------------------------------------------------------------------------


@pytest.fixture()
def seal_state() -> Iterator[None]:
    """Save/restore every process-global the memo machinery touches, so these
    tests can simulate fresh child processes without poisoning the suite."""
    snapshot = env_seal._LIB_SNAPSHOT
    disk = env_seal._DISK_MEMO
    override = env_seal._TOOLCHAIN_LIB_DIRS_OVERRIDE
    memo_env = os.environ.get(env_seal.SEAL_LIB_MEMO_ENV)
    try:
        yield
    finally:
        env_seal._LIB_SNAPSHOT = snapshot
        env_seal._DISK_MEMO = disk
        env_seal._TOOLCHAIN_LIB_DIRS_OVERRIDE = override
        env_seal._lib_digest.cache_clear()
        if memo_env is None:
            os.environ.pop(env_seal.SEAL_LIB_MEMO_ENV, None)
        else:
            os.environ[env_seal.SEAL_LIB_MEMO_ENV] = memo_env


def _fresh_process(memo_path: str = "") -> None:
    """Reset the per-process state exactly as a fresh child would have it."""
    env_seal._LIB_SNAPSHOT = None
    env_seal._DISK_MEMO = None
    env_seal._lib_digest.cache_clear()
    if memo_path:
        os.environ[env_seal.SEAL_LIB_MEMO_ENV] = memo_path
    else:
        os.environ.pop(env_seal.SEAL_LIB_MEMO_ENV, None)


def _fake_toolchain(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "libtorch_fake.so": b"A" * 4096,
        "libtriton_fake.so.1": b"B" * 2048,
        "libcudnn_fake.so.9": b"C" * 1024,
    }
    out: Dict[str, Path] = {}
    for name, content in files.items():
        path = root / name
        path.write_bytes(content)
        out[name] = path
    env_seal._TOOLCHAIN_LIB_DIRS_OVERRIDE = (root,)
    return out


def test_memo_served_seal_is_byte_identical_to_full_rehash(
    tmp_path: Path, seal_state: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_toolchain(tmp_path / "libs")
    memo_path = tmp_path / "memo.json"

    _fresh_process()
    rehash = env_seal.toolchain_library_digests()
    rehash_seal = env_seal.seal_digest(env_seal.effective_seal())
    assert env_seal.write_library_memo(memo_path) == 3

    # A fresh "child": memo present, hashing FORBIDDEN — if any digest were
    # recomputed rather than memo-served, this run would fail loudly, so
    # equality here proves both the source and the value.
    _fresh_process(str(memo_path))

    def _refuse(path: str, mtime_ns: int, size: int) -> str:
        raise AssertionError(
            f"full rehash of {path!r} ran despite a complete, current memo")

    monkeypatch.setattr(env_seal, "_lib_digest", _refuse)
    assert env_seal.toolchain_library_digests() == rehash
    assert env_seal.seal_digest(env_seal.effective_seal()) == rehash_seal


def test_any_detectable_mutation_forces_the_fallback_rehash(
    tmp_path: Path, seal_state: None,
) -> None:
    files = _fake_toolchain(tmp_path / "libs")
    memo_path = tmp_path / "memo.json"

    _fresh_process()
    stale = dict(env_seal.toolchain_library_digests())
    env_seal.write_library_memo(memo_path)

    # Content changed, mtime moves with it: the triple mismatches, so the
    # child must rehash and see the NEW content — a stale digest here would
    # be a corrupted cell key, the worst defect class in this program.
    files["libtorch_fake.so"].write_bytes(b"MUTATED" * 512)
    _fresh_process(str(memo_path))
    with_memo = dict(env_seal.toolchain_library_digests())
    _fresh_process()
    ground_truth = dict(env_seal.toolchain_library_digests())
    assert with_memo == ground_truth
    assert with_memo["libtorch_fake.so"] != stale["libtorch_fake.so"]

    # A file the memo has never seen is hashed fresh; a removed file drops
    # out: enumeration and stat are always the child's own, so the memo can
    # never resurrect a file or hide one.
    (tmp_path / "libs" / "libnccl_fake.so.2").write_bytes(b"N" * 512)
    files["libcudnn_fake.so.9"].unlink()
    _fresh_process(str(memo_path))
    with_memo = dict(env_seal.toolchain_library_digests())
    _fresh_process()
    assert with_memo == dict(env_seal.toolchain_library_digests())
    assert "libnccl_fake.so.2" in with_memo
    assert "libcudnn_fake.so.9" not in with_memo

    # An unreadable/corrupt memo is ignored wholesale: full rehash, no error.
    memo_path.write_text("{not json")
    _fresh_process(str(memo_path))
    assert dict(env_seal.toolchain_library_digests()) == with_memo


def test_same_size_mtime_preserved_mutation_is_the_documented_blind_spot(
    tmp_path: Path, seal_state: None,
) -> None:
    """Content rewritten at the SAME size with mtime_ns restored is served
    stale from the memo. Asserted, not hidden: ``(path, mtime_ns, size)``
    cannot distinguish this case, and it is the exact trust boundary the
    per-process ``lru_cache`` (keyed on the same triple) already had — the
    memo widens WHO shares that boundary, it does not weaken it. Detecting
    this case would require hashing, which is the cost being removed."""
    files = _fake_toolchain(tmp_path / "libs")
    memo_path = tmp_path / "memo.json"

    _fresh_process()
    stale = dict(env_seal.toolchain_library_digests())
    env_seal.write_library_memo(memo_path)

    target = files["libtriton_fake.so.1"]
    st = target.stat()
    target.write_bytes(b"X" * st.st_size)          # same size, new content
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))  # mtime restored

    _fresh_process(str(memo_path))
    served = dict(env_seal.toolchain_library_digests())
    assert served["libtriton_fake.so.1"] == stale["libtriton_fake.so.1"], (
        "this documents the blind spot; if it starts failing, the memo "
        "gained content verification and this test (and the code comment) "
        "should be rewritten to claim the stronger property")


def test_memo_file_is_versioned_and_keyed_by_the_triple(
    tmp_path: Path, seal_state: None,
) -> None:
    files = _fake_toolchain(tmp_path / "libs")
    memo_path = tmp_path / "memo.json"
    _fresh_process()
    env_seal.write_library_memo(memo_path)
    doc = json.loads(memo_path.read_text())
    assert doc["memo_v"] == env_seal._MEMO_V
    st = files["libtorch_fake.so"].stat()
    key = f"{files['libtorch_fake.so']}\x00{st.st_mtime_ns}\x00{st.st_size}"
    assert key in doc["digests"]
    assert all(len(d) == 16 for d in doc["digests"].values())


# ---------------------------------------------------------------------------
# The measured drop, on the real pool (the pgw#830 acceptance, tightened)
# ---------------------------------------------------------------------------


def test_pooled_children_stop_repaying_the_toolchain_hash(
    tmp_path: Path,
) -> None:
    """Every compile child's ``seal_libhash_s`` collapses to metadata cost.

    pgw#830's attribution test asserts the span EXISTS; this is the sibling
    that asserts the MAGNITUDE dropped. REAL children, spawned by the real
    pool through the real ``child_argv``/``child_env`` — the process boundary
    is the thing being fixed, so a probe of the memo function would test
    nothing.

    The children REFUSE (their job names a module this tree does not have), and
    that is deliberate rather than a compromise: ``env_seal.establish()`` is the
    child's FIRST act, before it touches the declaration, so the seal span is
    already recorded and complete by the time the preflight refuses. A real
    compile would add ~7 minutes and could not make the seal reading any more
    real — and pgw#1215's child can no longer be handed an already-exported
    program, so a "cheap real compile" here is not available at any price.

    The threshold is derived, not conjured: ``cold_s`` below is this box's
    measured full SHA-256 pass over the real toolchain (the cost every child
    used to pay — 8.13 s at 0.49 GB/s when pgw#830 measured it). A memo-served
    pass does no hashing at all — ~36 stats plus one small JSON read, measured
    at ~1 % of the cold pass — so ``0.25 * cold_s`` splits the two regimes with
    >4x margin on either side, and the 0.75 s absolute floor absorbs scheduler
    jitter under a loaded suite while staying far below any real rehash.
    """
    import msgspec

    # The reference: measure THIS box's cold identity pass. Clearing the
    # process caches only makes the next pass recompute identical values, so
    # the suite's state is unchanged afterwards.
    env_seal._LIB_SNAPSHOT = None
    env_seal._lib_digest.cache_clear()
    t0 = time.monotonic()
    manifest = env_seal.frozen_library_digests()
    cold_s = time.monotonic() - t0
    assert manifest, "no toolchain libs found — the reference is meaningless"
    threshold = max(0.25 * cold_s, 0.75)

    width = pool.entry_workers(
        3, limit=2, vcpus=16, available_bytes=64 * 1024**3, device_lock=True)
    assert width.workers == 2
    box = pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"))
    template = pool.EntryJob(
        function="nope",
        modules=("gen_worker_no_such_endpoint_module",),
        out_dir=str(tmp_path / "artifacts"))
    with pytest.raises(pool.EntryCompileFailed):
        box.compile(template)

    # The parent's one-time cost is named, never silent: it rides the pool
    # ledger as `seal_seed_s`, outside the capacity identity (it is paid
    # before the pool wall starts).
    facts = box.ledger.facts()
    assert facts["seal_seed_s"] == box.seal_seed_s
    assert Path(box.seal_memo).is_file()

    reports = sorted((tmp_path / "pool").glob("share-*/report.json"))
    assert reports, "no child wrote a report — nothing was measured"
    for path in reports:
        report = msgspec.json.decode(
            path.read_bytes(), type=pool.EntryReport)
        assert report.status == pool.REFUSED, report.detail
        seal = dict(report.overlays)
        assert "seal_libhash_s" in seal, seal
        # EVERY child, not just children after the first: the parent seeds the
        # memo before any child spawns, so no child ever pays the pass.
        assert seal["seal_libhash_s"] <= threshold, (
            f"{report.entry}: seal_libhash_s={seal['seal_libhash_s']:.2f}s "
            f"exceeds {threshold:.2f}s (25 % of this box's measured "
            f"{cold_s:.2f}s cold pass) — the child re-paid the toolchain "
            f"hash the pgw#832 memo exists to remove")
        # And the pgw#830 invariant still closes on a refusing child: the seal
        # split (memo or not) must keep covering child_seal_s.
        assert not spans.check(report.spans), report.spans
