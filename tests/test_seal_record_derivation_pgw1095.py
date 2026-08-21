"""The seal's library identity is DERIVED from the installing wheel's RECORD, and re-hashed only where no RECORD covers the file."""

from __future__ import annotations

import base64
import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pytest

from gen_worker import boot_phases, dist_records, env_seal


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _record_field(data: bytes) -> str:
    raw = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + raw.decode()


def _install(
    site: Path,
    dist: str,
    version: str,
    files: Dict[str, bytes],
    *,
    self_row: bool = True,
    lie_about_size: Optional[str] = None,
) -> Dict[str, Path]:
    info = site / f"{dist}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n")
    out: Dict[str, Path] = {}
    rows: List[str] = []
    for rel, content in files.items():
        path = site / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        out[rel] = path
        size = len(content) + (7 if rel == lie_about_size else 0)
        rows.append(f"{rel},{_record_field(content)},{size}")
    rows.append(f"{info.name}/METADATA,{_record_field(b'')},0")
    if self_row:
        rows.append(f"{info.name}/RECORD,,")
    (info / "RECORD").write_text("\n".join(rows) + "\n")
    return out


@pytest.fixture()
def env(tmp_path: Path) -> Iterator[Path]:
    """A throwaway site-packages on `sys.path`, with every process-global the identity pass touches saved and restored."""
    site = tmp_path / "site-packages"
    (site / "libs").mkdir(parents=True)
    snapshot = (
        env_seal._LIB_SNAPSHOT, env_seal._DISK_MEMO,
        env_seal._TOOLCHAIN_LIB_DIRS_OVERRIDE, env_seal._SOURCES,
    )
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    _fresh_process(site)
    try:
        yield site
    finally:
        sys.path.remove(str(site))
        importlib.invalidate_caches()
        (env_seal._LIB_SNAPSHOT, env_seal._DISK_MEMO,
         env_seal._TOOLCHAIN_LIB_DIRS_OVERRIDE, env_seal._SOURCES) = snapshot
        env_seal._lib_digest.cache_clear()
        dist_records.reset_cache()
        os.environ.pop(env_seal.SEAL_LIB_MEMO_ENV, None)


def _fresh_process(site: Path) -> None:
    env_seal._LIB_SNAPSHOT = None
    env_seal._DISK_MEMO = None
    env_seal._SOURCES = env_seal.DigestSources(0, 0, 0)
    env_seal._TOOLCHAIN_LIB_DIRS_OVERRIDE = (site / "libs",)
    env_seal._lib_digest.cache_clear()
    dist_records.reset_cache()
    importlib.invalidate_caches()
    os.environ.pop(env_seal.SEAL_LIB_MEMO_ENV, None)


_LIBS = {
    "libs/libtorch_fake.so": b"A" * 4096,
    "libs/libtriton_fake.so.1": b"B" * 2048,
    "libs/libcudnn_fake.so.9": b"C" * 1024,
}


def _forbid_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(path: str, mtime_ns: int, size: int) -> str:
        raise AssertionError(
            f"full rehash of {path!r} ran despite an intact RECORD")

    monkeypatch.setattr(env_seal, "_lib_digest", _refuse)


def test_the_recorded_digest_is_the_hash_it_replaces(env: Path) -> None:
    """RECORD's urlsafe-b64 sha256, decoded to hex and truncated, IS `sha256(content).hexdigest()[:16]`."""
    files = _install(env, "fakewheel", "1.0", _LIBS)
    for rel, content in _LIBS.items():
        st = files[rel].stat()
        derived = dist_records.digest_for(
            str(files[rel]), st.st_mtime_ns, st.st_size)
        assert derived == _sha256(content), rel


def test_the_seal_is_byte_identical_derived_or_hashed(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(env, "fakewheel", "1.0", _LIBS)

    monkeypatch.setattr(dist_records, "digest_for",
                        lambda path, mtime_ns, size: None)
    hashed = env_seal.toolchain_library_digests()
    assert env_seal.digest_sources().hashed == len(_LIBS)
    monkeypatch.undo()

    _fresh_process(env)
    _forbid_hashing(monkeypatch)
    derived = env_seal.toolchain_library_digests()
    assert derived == hashed
    src = env_seal.digest_sources()
    assert (src.record, src.memo, src.hashed) == (len(_LIBS), 0, 0)


def test_the_phase_row_reports_hit_and_names_the_source(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(env, "fakewheel", "1.0", _LIBS)
    boot_phases.reset_for_tests()
    _forbid_hashing(monkeypatch)
    env_seal.frozen_library_digests()
    rows = [r for r in boot_phases.recorded_rows()
            if r.phase == boot_phases.PHASE_LIB_MEMO and r.terminal]
    assert len(rows) == 1
    assert rows[0].reason == "hit"
    assert f"record={len(_LIBS)}" in rows[0].detail
    assert "hashed=0" in rows[0].detail
    boot_phases.reset_for_tests()


def test_a_library_no_record_covers_is_hashed_never_trusted(
    env: Path,
) -> None:
    _install(env, "fakewheel", "1.0", _LIBS)
    stray = env / "libs" / "libnccl_stray.so.2"
    stray.write_bytes(b"S" * 777)
    _fresh_process(env)

    digests = dict(env_seal.toolchain_library_digests())
    assert digests["libnccl_stray.so.2"] == _sha256(b"S" * 777)
    src = env_seal.digest_sources()
    assert (src.record, src.hashed) == (len(_LIBS), 1)


def test_a_record_that_lies_about_size_is_not_trusted(env: Path) -> None:
    files = _install(env, "fakewheel", "1.0", _LIBS,
                     lie_about_size="libs/libtorch_fake.so")
    st = files["libs/libtorch_fake.so"].stat()
    assert dist_records.digest_for(
        str(files["libs/libtorch_fake.so"]), st.st_mtime_ns, st.st_size) is None


def test_an_unanchored_record_is_not_trusted(env: Path) -> None:
    """A RECORD that does not describe itself gives the staleness guard no anchor, so the whole distribution falls back to hashing."""
    files = _install(env, "fakewheel", "1.0", _LIBS, self_row=False)
    st = files["libs/libtorch_fake.so"].stat()
    assert dist_records.digest_for(
        str(files["libs/libtorch_fake.so"]), st.st_mtime_ns, st.st_size) is None
    _fresh_process(env)
    assert dict(env_seal.toolchain_library_digests()) == {
        Path(rel).name: _sha256(content) for rel, content in _LIBS.items()}
    assert env_seal.digest_sources().hashed == len(_LIBS)


def test_tamper_that_changes_the_size_is_caught(env: Path) -> None:
    files = _install(env, "fakewheel", "1.0", _LIBS)
    _fresh_process(env)
    before = dict(env_seal.toolchain_library_digests())

    files["libs/libtorch_fake.so"].write_bytes(b"MUTATED" * 999)
    _fresh_process(env)
    after = dict(env_seal.toolchain_library_digests())
    assert after["libtorch_fake.so"] == _sha256(b"MUTATED" * 999)
    assert after["libtorch_fake.so"] != before["libtorch_fake.so"], (
        "a size-changing in-place tamper was served from RECORD: the seal "
        "would name content this host is not running")
    assert env_seal.digest_sources().hashed == 1


def test_tamper_that_leaves_a_newer_mtime_is_caught(env: Path) -> None:
    """Same size, but written after the RECORD that describes it."""
    files = _install(env, "fakewheel", "1.0", _LIBS)
    _fresh_process(env)
    before = dict(env_seal.toolchain_library_digests())

    target = files["libs/libtriton_fake.so.1"]
    replacement = b"X" * target.stat().st_size
    target.write_bytes(replacement)
    os.utime(target, ns=(target.stat().st_mtime_ns + 10**9,
                         target.stat().st_mtime_ns + 10**9))
    _fresh_process(env)
    after = dict(env_seal.toolchain_library_digests())
    assert after["libtriton_fake.so.1"] == _sha256(replacement)
    assert after["libtriton_fake.so.1"] != before["libtriton_fake.so.1"]
    assert env_seal.digest_sources().hashed == 1


def test_same_size_mtime_restored_tamper_is_the_documented_blind_spot(
    env: Path,
) -> None:
    """THE TRADE, asserted so it is ruled on rather than discovered."""
    files = _install(env, "fakewheel", "1.0", _LIBS)
    _fresh_process(env)
    before = dict(env_seal.toolchain_library_digests())

    target = files["libs/libcudnn_fake.so.9"]
    st = target.stat()
    target.write_bytes(b"Z" * st.st_size)
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    _fresh_process(env)
    after = dict(env_seal.toolchain_library_digests())
    assert after["libcudnn_fake.so.9"] == before["libcudnn_fake.so.9"]
    assert after["libcudnn_fake.so.9"] != _sha256(b"Z" * st.st_size)


def test_one_enumeration_serves_both_readers(env: Path) -> None:
    _install(env, "fakewheel", "1.0", _LIBS)
    texts = dist_records.record_texts()
    assert "fakewheel" in texts
    assert "libs/libtorch_fake.so" in texts["fakewheel"]
    dists, indexed = dist_records.coverage()
    assert dists >= 1 and indexed >= len(_LIBS)
