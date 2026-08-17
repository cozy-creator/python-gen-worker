"""pgw#1310: the vendored snapshots are what VENDORED.toml says they are.

Two fences, deliberately different in kind, because either one alone passes on
a tree the other refuses:

1. CONTENT — every vendored file hashes to the digest recorded beside its rev.
   Catches a hand-patch, a partial re-vendor, and a rev bump whose manifest was
   not regenerated.
2. BEHAVIOUR — the pinned `tensorfs` rev is pinned FOR the pgw#1287 progress
   fix, so the fence asserts the fix runs, not that a string says it should.
   A digest fence alone would happily certify a correctly-recorded snapshot of
   the broken tree.

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


def test_the_pinned_tensorfs_rev_emits_progress_as_each_object_lands() -> None:
    """pgw#1287: the fix this rev is pinned for, asserted by running it.

    The second object cannot finish until the FIRST object's callback has been
    observed. Post-drain emission (the defect: a 31.6 GB download reporting
    `0 / total` until its last byte landed, so the hub's 6-minute freshness
    window condemned a healthy pod) cannot satisfy that, at any timeout.
    """
    from gen_worker._vendor.tensorfs import CASRef, TransferGrant
    from gen_worker._vendor.tensorfs.transfer import _run_parallel

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
