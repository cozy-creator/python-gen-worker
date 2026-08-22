from __future__ import annotations

import dataclasses
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


def test_the_tensorfs_snapshot_is_one_rev() -> None:

    spec = MANIFEST["packages"]["tensorfs"]
    extra = sorted(k for k in spec if k.endswith("_rev") and k != "rev")
    assert not extra, (
        f"the tensorfs snapshot grew a second rev ({extra}). One rev, per "
        f"pgw#1575. If a plane needs a file the pinned rev lacks, bump the "
        f"rev; if it needs a file NO upstream rev has, it is first-party and "
        f"belongs in gen_worker/, not under _vendor/."
    )
    assert not any(k.endswith("_files") for k in spec), (
        "a per-plane file list is back. The one inventory is "
        "[packages.tensorfs.files], enforced as a set above."
    )


def test_the_write_plane_is_first_party_and_never_returns() -> None:

    from gen_worker._vendor.tensorfs import LocalCAS

    for retired in (
        "ingest_file",
        "ingest_repository",
        "collect_garbage",
        "materialize",
        "materialize" + "_repository",
    ):
        assert not hasattr(LocalCAS, retired), (
            f"LocalCAS.{retired} is back on the vendored snapshot. It exists "
            f"at NO tensorfs master ancestor; a `local.py` carrying it came "
            f"from the pre-genesis `8bafdfbb` lineage. Admission and retention "
            f"are first-party at gen_worker/cas/; the single-file hatch is "
            f"`TensorReader.materialize`."
        )

    assert "planner.py" not in MANIFEST["packages"]["tensorfs"]["files"]
    assert not (VENDOR / "tensorfs" / "planner.py").exists()

    from gen_worker._vendor.tensorfs import TensorReader
    from gen_worker.cas import collect_garbage, ingest_file, plan_chunks

    for successor in (collect_garbage, ingest_file, plan_chunks):
        assert callable(successor)
    assert callable(TensorReader.materialize)


def test_the_torchcg_snapshot_is_one_rev() -> None:

    spec = MANIFEST["packages"]["torchcg"]
    extra = sorted(k for k in spec if k.endswith("_rev") and k != "rev")
    assert extra == [], (
        f"the torchcg snapshot grew a second rev ({extra}). One rev, per "
        f"tcg#39 — a graft forks the compiled-graph store's identity."
    )
    assert spec["subdir"] == "src/torchcg"


def test_the_vendored_store_uses_the_frozen_ref_prefix() -> None:
    """The ref namespace, asserted where the fleet can see it.

    tcg#90 moved the artifact store to `torchcg/v1/graphs` and pgw's own
    document/program store kept `torchcg/v2`. Two stores share one CAS, so the
    prefixes must stay distinct AND stated.
    """

    from gen_worker._vendor.torchcg import store as artifact_store
    from gen_worker.graphs import store as pgw_store

    assert artifact_store._REF_PREFIX == "torchcg/v1/graphs"
    assert pgw_store._REF_PREFIX == "torchcg/v2"
    assert not artifact_store._REF_PREFIX.startswith(pgw_store._REF_PREFIX + "/")
    assert tuple(f.name for f in dataclasses.fields(artifact_store.StoredArtifact)) == (
        "key", "directory", "metadata", "ref")


def test_the_vendored_identity_is_TORCH_FREE() -> None:
    """The property that made the graft safe, EXECUTED rather than asserted.

    It used to be asserted of `selection`, which is deleted. It matters more
    now, not less: `torchcg.identity` resolves the layout handle every artifact
    declares, and `resolve`/cold-reuse run in processes that never import
    torch. If importing identity — or reading the identity morphism — drags
    torch in, that path stops existing and nothing else notices.
    """
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    proof = subprocess.run(
        [sys.executable, "-c",
         "import sys;"
         f"sys.path.insert(0, {src!r});"
         "from gen_worker._vendor.torchcg import identity;"
         "assert 'torch' not in sys.modules, 'importing identity pulled torch in';"
         "h = identity.contiguous_handle();"
         "assert h, 'no identity layout handle';"
         "assert 'torch' not in sys.modules, 'resolving the identity layout pulled torch in';"
         "print(h)"],
        capture_output=True, text=True, check=False)
    assert proof.returncode == 0, proof.stderr
    # Read from the VENDORED corpus, so this also proves the corpus shipped.
    assert proof.stdout.strip().startswith("torch."), proof.stdout


def test_the_vendored_identity_and_ingress_fold_a_real_key() -> None:
    """The vendored identity is exercised, not merely imported.

    `recipe.py`'s vocabulary is gone with it; what survives is the thing it was
    folding — an ingress built here reaches a key through the vendored
    identity, using the vendored layout corpus.
    """
    from gen_worker._vendor.torchcg.identity import (
        CallIngress,
        CallInput,
        artifact_key,
        contiguous_handle,
        is_artifact_key,
    )

    ingress = CallIngress(
        parameters=("latents",),
        flat_arity=1,
        inputs=(
            CallInput(
                name="latents",
                position=0,
                param="latents",
                param_position=0,
                path=(),
                exported_name="latents",
                dtype="bfloat16",
                shape=(1, 4, 8, 8),
            ),
        ),
    )
    assert len(ingress.digest()) == 32
    key = artifact_key(
        "cg-graph-v1-" + "0" * 56,
        sm="sm_89",
        env={"torch": "2.13.0"},
        policy={"always_keep_tensor_constants": True},
        layout=contiguous_handle(),
    )
    assert is_artifact_key(key.value)
    # A different card is a different artifact — the axis that used to be
    # checked by folding a `_Runtime` through `recipe.variant.key`.
    other = artifact_key(
        "cg-graph-v1-" + "0" * 56,
        sm="sm_86",
        env={"torch": "2.13.0"},
        policy={"always_keep_tensor_constants": True},
        layout=contiguous_handle(),
    )
    assert other.value != key.value


def test_the_read_plane_runs_without_the_compiled_extension() -> None:
    """The reason the snapshot is usable at all: it is PURE PYTHON."""

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
    assert report.failures == [], report.failures[0][1]
    assert report.succeeded == 2
    assert seen == [1024, 1024]


def test_no_deleted_project_is_required_from_an_index() -> None:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    project = tomllib.loads(text)["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    dead = [r for r in requirements if r.split()[0].split(">")[0].split("=")[0].strip()
            in {"hashrepo", "tensorfs", "torch-compiled-graphs", "torchcg"}]
    assert dead == [], f"deleted PyPI projects required from an index: {dead}"
    assert "[tool.uv.sources]" in text
    for name in ("hashrepo =", "torch-compiled-graphs =", "tensorfs ="):
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


def test_no_vendored_module_imports_its_sibling_by_its_INDEX_name() -> None:
    offenders = []
    for package in MANIFEST["packages"]:
        root = VENDOR / package
        siblings = {name for name in MANIFEST["packages"] if name != package}
        for path in sorted(root.rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                for sibling in siblings:
                    if stripped.startswith((f"from {sibling} ", f"import {sibling}")):
                        offenders.append(f"{package}/{path.relative_to(root)}:{number}: {stripped}")
    assert not offenders, (
        "a vendored module imports a sibling by its INDEX name, which resolves "
        "to a deleted PyPI project (or, worse, to a DIFFERENT installed copy) "
        "instead of the snapshot beside it. Re-vendoring rewrites these to "
        "relative imports:\n  " + "\n  ".join(offenders)
    )


def test_the_dev_torchcg_pin_is_the_rev_the_snapshot_was_taken_from() -> None:
    import re

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    match = re.search(
        r'^torchcg = \{ git = "[^"]*torchcg", rev = "([0-9a-f]+)"', pyproject, re.M
    )
    assert match, "the dev torchcg source pin is gone; this fence needs updating with it"
    pinned = match.group(1)
    vendored = str(MANIFEST["packages"]["torchcg"]["rev"])
    assert pinned.startswith(vendored) or vendored.startswith(pinned), (
        f"the derive's torchcg ({pinned[:12]}) is not the rev the vendored "
        f"snapshot was taken from ({vendored}). Both halves move together or "
        f"the derive and the mint disagree about the same library — bump the "
        f"pin in [tool.uv.sources], re-lock, and re-vendor in one change."
    )
