"""pgw#1310: the vendored snapshots are what VENDORED.toml says they are.

Two fences, deliberately different in kind, because either one alone passes on
a tree the other refuses:

1. CONTENT — every vendored file hashes to the digest recorded beside its rev.
   Catches a hand-patch, a partial re-vendor, and a rev bump whose manifest was
   not regenerated.
2. BEHAVIOUR — a digest fence alone would happily certify a correctly-recorded
   snapshot of a broken tree, so the pgw#1287 progress fix is asserted by
   RUNNING it. pgw#1308 moved that code first-party (`gen_worker.transfer`),
   which is a strengthening rather than a loss: the fence now guards pgw's own
   source instead of fidelity to an upstream rev, and a third fence refuses
   the transfer plane's return to the vendored snapshot.

Plus the reason the vendoring exists at all: the built distribution must not
require either deleted project from an index (ie#738).
"""

from __future__ import annotations

import hashlib
import threading
import tomllib
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "src" / "gen_worker" / "_vendor"
MANIFEST = tomllib.loads((VENDOR / "VENDORED.toml").read_text())


def test_every_vendored_file_matches_its_recorded_digest() -> None:
    for package, spec in MANIFEST["packages"].items():
        recorded = spec["files"]
        root = VENDOR / package
        present = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        assert present == set(recorded), (
            f"{package}: vendored file set differs from VENDORED.toml — "
            f"only on disk {sorted(present - set(recorded))}, "
            f"only recorded {sorted(set(recorded) - present)}"
        )
        for name, digest in recorded.items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            assert actual == digest, (
                f"{package}/{name} was edited in place. A vendored snapshot is "
                f"fixed upstream and re-vendored, never patched here — see "
                f"VENDORED.toml."
            )


def test_the_vendored_tensorfs_carries_no_transfer_plane() -> None:
    """pgw#1308: the transfer plane is FIRST-PARTY, and the fence says so.

    The two modules were pinned to a lineage upstream had abandoned -- current
    tensorfs has no Python transfer plane at all -- so a digest fence over
    them certified fidelity to a rev nobody would ever fix. They now live at
    `gen_worker.transfer`, where their behaviour fence (pgw#1287) went with
    them. Re-vendoring them here would silently restore two implementations of
    one wire.
    """
    import gen_worker._vendor.tensorfs as vendored

    for retired in ("transfer", "journal", "daemon"):
        assert retired not in MANIFEST["packages"]["tensorfs"]["files"], (
            f"tensorfs/{retired}.py is back in the vendored snapshot. It is "
            f"first-party at gen_worker/transfer/ (pgw#1308)."
        )
        assert not (VENDOR / "tensorfs" / f"{retired}.py").exists()
    for symbol in ("TransferGrant", "TransferReport", "download", "upload", "MountedPath"):
        assert not hasattr(vendored, symbol), (
            f"{symbol} is re-exported from the vendored storage package again"
        )


def test_the_read_plane_is_recorded_at_its_own_rev() -> None:
    """pgw#1330: the snapshot is SPLIT at two revs, and the split is declared.

    `project.py`, `tensors.py` and `gguf.py` come from a rev the storage half
    is deliberately NOT at: the newer lineage deleted three `LocalCAS` methods
    this repo still calls (ingest, GC, and the whole-tree copy the chokepoint
    uses — VENDORED.toml names them). A split that is only in a comment is a
    split nobody can audit, so the rev is a field and the file list is a
    field, and the digest fence above covers them like any other vendored file.
    """

    spec = MANIFEST["packages"]["tensorfs"]
    assert spec["read_plane_rev"] != spec["rev"]
    assert set(spec["read_plane_files"]) == {"gguf.py", "project.py", "tensors.py"}
    for name in spec["read_plane_files"]:
        assert name in spec["files"], f"{name} is not digest-fenced"


def test_the_read_plane_runs_without_the_compiled_extension() -> None:
    """The reason the split is expressible at all: this half is PURE PYTHON.

    `Layout::project` is Rust and cannot travel into a source-vendored wheel,
    so a stub that only the extension could render or parse would leave every
    consumer here unable to read a projected tree. Proved by EXECUTING the
    render/parse/read path with `tensorfs._tensorfs` made unimportable, not by
    reading the import list.
    """

    import os
    import subprocess
    import sys

    program = """
import sys


class Refuse:
    # find_spec, not find_module: the latter was REMOVED in 3.12, so a finder
    # defining it is silently ignored and the guard proves nothing.
    def find_spec(self, name, path=None, target=None):
        if name.rpartition(".")[2] == "_tensorfs":
            raise AssertionError("the read plane reached the compiled extension")
        return None


sys.meta_path.insert(0, Refuse())

# The guard must be LIVE, or everything below is vacuous.
try:
    __import__("_tensorfs")
except AssertionError:
    pass
else:
    raise SystemExit("the meta_path guard never fired -- this test proves nothing")

from gen_worker._vendor.tensorfs import parse_stub, stub_bytes

body = "ab" * 32
stub = parse_stub(stub_bytes(body, 4096))
assert stub is not None and stub.body_sha256 == body and stub.size == 4096
print("ok")
"""
    root = VENDOR.parents[1]
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_the_transfer_plane_emits_progress_as_each_object_lands() -> None:
    """pgw#1287, now asserted against pgw's OWN code rather than a pinned rev.

    The second object cannot finish until the FIRST object's callback has been
    observed. Post-drain emission (the defect: a 31.6 GB download reporting
    `0 / total` until its last byte landed, so the hub's 6-minute freshness
    window condemned a healthy pod) cannot satisfy that, at any timeout.
    """
    from gen_worker._vendor.tensorfs import CASRef
    from gen_worker.transfer.grants import TransferGrant, _run_parallel

    grants = [
        TransferGrant(
            digest=CASRef.parse(f"sha256:{i:064x}"),
            size_bytes=1024,
            url="https://example/x",
        )
        for i in range(1, 3)
    ]
    first_reported = threading.Event()
    seen: list[int] = []

    def worker(grant: TransferGrant) -> bool:
        if grant.digest == grants[1].digest:
            assert first_reported.wait(timeout=10.0), (
                "the first object's progress callback never fired while a second "
                "transfer was still running: this rev emits progress only after "
                "the whole batch drains (pgw#1287 is NOT in the vendored tree)"
            )
        return False

    def progress(digest: object, size: int) -> None:
        seen.append(size)
        first_reported.set()

    report = _run_parallel(grants, worker, parallel=2, progress=progress)
    # The worker runs on a pool thread, so its assertion arrives as a recorded
    # failure rather than as this test's traceback. Surface it verbatim.
    assert report.failures == [], report.failures[0][1]
    assert report.succeeded == 2
    assert seen == [1024, 1024]


def test_no_deleted_project_is_required_from_an_index() -> None:
    """ie#738: the break this whole cut exists to close.

    Both PyPI projects are permanently deleted, so ANY index requirement on them
    makes the published wheel unresolvable for every consumer.
    """
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    project = tomllib.loads(text)["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    # This fence's denylist must NAME the dead projects in order to refuse
    # them; respelling it would delete the refusal.
    dead = [r for r in requirements if r.split()[0].split(">")[0].split("=")[0].strip()
            # retired-name: the denylist above.
            in {"hashrepo", "tensorfs", "torch-compiled-graphs", "torchcg"}]
    assert dead == [], f"deleted PyPI projects required from an index: {dead}"
    assert "[tool.uv.sources]" in text
    # retired-name: same denylist, same reason.
    for name in ("hashrepo =", "torch-compiled-graphs =", "tensorfs =", "torchcg ="):
        assert name not in text, (
            f"a `{name}` source pin is back in pyproject.toml. A source pin is "
            f"workspace-only metadata — it is stripped from the published wheel "
            f"— so it cannot be the delivery mechanism (ie#738)."
        )


@pytest.mark.parametrize(
    "module",
    ["gen_worker._vendor.tensorfs", "gen_worker._vendor.torchcg"],
)
def test_the_vendored_packages_import_with_no_third_party_present(module: str) -> None:
    __import__(module)
