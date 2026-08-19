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

    The consumption plane comes from a rev the storage half is deliberately NOT
    at: the newer lineage deleted three `LocalCAS` methods this repo still
    calls (ingest, GC, and the whole-tree copy the chokepoint uses —
    VENDORED.toml names them). A split that is only in a comment is a split
    nobody can audit, so the rev is a field and the file list is a field, and
    the digest fence above covers them like any other vendored file.

    pgw#1344 widened the plane rather than the split: `writer.py` joins it (the
    GGUF/safetensors composer a conversion writes through) and `__init__.py`
    comes from the same rev, verbatim, now that upstream no longer re-exports a
    Python transfer plane. The membership is spelled out so a file JOINING the
    plane is a decision somebody makes on purpose.
    """

    spec = MANIFEST["packages"]["tensorfs"]
    assert spec["read_plane_rev"] != spec["rev"]
    assert set(spec["read_plane_files"]) == {
        "__init__.py", "gguf.py", "project.py", "tensors.py", "writer.py",
    }
    for name in spec["read_plane_files"]:
        assert name in spec["files"], f"{name} is not digest-fenced"


def test_the_storage_half_still_provides_what_the_ownership_ruling_keeps() -> None:
    """pgw#1308 step ⑥: the split is the STEADY STATE, and this says why.

    The ruling (VENDORED.toml, and the issue's own section) is that admission
    and GC are TENSORFS'S — upstream ported them to Rust rather than
    abandoning them, and a source-vendored wheel cannot load Rust (pgw#1310).
    So they are consumed from the pinned rev indefinitely, and a bump that
    "finishes the migration" deletes them with no reachable successor.

    A comment saying that goes stale silently. This goes red instead, and it
    is deliberately NOT a digest check: the digest fence above would refuse
    the bump too, but only with "a file changed", which is exactly the message
    that gets a fence regenerated rather than read.
    """

    from gen_worker._vendor.tensorfs import LocalCAS

    for kept in ("ingest_file", "ingest_repository", "collect_garbage", "materialize"):
        assert callable(getattr(LocalCAS, kept, None)), (
            f"LocalCAS.{kept} is gone from the vendored storage snapshot. If "
            f"this is a `rev` bump: upstream's successor is the compiled Rust "
            f"extension, which pgw cannot load (pgw#1310). Read the ownership "
            f"ruling in VENDORED.toml before bumping."
        )

    # And the one that DID die with the chokepoint flip: still defined by the
    # pinned upstream snapshot, called by nothing here. The caller census is
    # `scripts/lint_materialization_hatch.py`, in the required `fast gates`.
    assert hasattr(LocalCAS, "materialize" + "_repository")


def test_the_torchcg_snapshot_is_one_rev() -> None:
    """pgw#1342/tcg#39: the graft fields are GONE, and nothing may re-grow one.

    Two lanes grafted a newer file over an older storage half — pgw#1328's
    `selection_rev`, pgw#1332's `recipe_rev` — each because the tip carried a
    storage re-key its own lane was not entitled to take. Paul ruled the re-key
    accepted (tcg#39) and this snapshot took the tip whole, so a second rev in
    this table is now a fork of a storage identity with no ruling behind it.
    The digest fence above cannot see that: a graft records perfectly well.
    """

    spec = MANIFEST["packages"]["torchcg"]
    extra = sorted(k for k in spec if k.endswith("_rev") and k != "rev")
    assert extra == [], (
        f"the torchcg snapshot grew a second rev ({extra}). One rev, per "
        f"tcg#39 — a graft forks the compiled-graph store's identity."
    )
    # The rev itself is NOT restated here. VENDORED.toml says it is "the single
    # home of the fact", and a copy of a rev in a test is a second home that
    # drifts. What this asserts is the SHAPE the ruling fixed: one rev, and the
    # post-rename layout it must have been taken from.
    assert spec["subdir"] == "src/torchcg"


def test_the_vendored_store_uses_the_frozen_ref_prefix() -> None:
    """The re-key this bump paid for, asserted where the fleet can see it.

    `torchcg/v1` is the address every compiled graph on every pod is filed
    under. Moving it again is not a rename: it orphans every stored artifact at
    once and costs another fleet-wide re-mint. Upstream froze it (torchcg
    PR #41); this is the consumer-side half, so a rev bump that quietly moved
    it could not reach a pod through this repo.
    """

    from gen_worker._vendor.torchcg import storage

    key = "cg-key-v1-" + "0" * 56
    assert storage._REF_PREFIX == "torchcg/v1"
    assert storage._graph_ref(key) == f"torchcg/v1/graphs/{key}"
    assert tuple(f.name for f in dataclasses.fields(storage.StoreResult)) == (
        "outcome", "key", "artifact")


def test_the_selection_contract_is_torch_free_and_registered() -> None:
    """The property that made the graft safe, executed rather than asserted.

    tcg#37's implementation reads four facts per feed (dtype, shape,
    contiguity, pointer alignment), so it needs no accelerator — which is why
    it can be a JSON corpus at all, and why an adopt-only pod can rank a call
    before anything is loaded. It arrived as a graft (pgw#1328) and is now part
    of the ordinary one-rev snapshot (pgw#1342); the property that mattered
    then still has to hold, because an adopt-only boot imports it eagerly.
    """
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    proof = subprocess.run(
        [sys.executable, "-c",
         "import sys;"
         f"sys.path.insert(0, {src!r});"
         "from gen_worker._vendor.torchcg import selection;"
         "from gen_worker._vendor.torchcg.contracts import CONTRACT_FILES;"
         "assert selection.SELECTION_CONTRACT_FILE in CONTRACT_FILES;"
         "assert selection.selection_vectors();"
         "assert 'torch' not in sys.modules;"
         "print('ok')"],
        capture_output=True, text=True, check=False)
    assert proof.returncode == 0, proof.stderr
    assert proof.stdout.strip() == "ok"


def test_the_recipe_vocabulary_runs_against_the_VENDORED_identity_and_ingress() -> None:
    """`recipe.py`'s only siblings are `identity` and `ingress`, and it folds
    a real key through them.

    pgw#1332 grafted this file at its own `recipe_rev` and this test guarded the
    graft's compatibility claim. pgw#1342 took the whole package to one rev, so
    the pairing is no longer a claim — but the EXECUTION still is, and it is the
    half worth keeping: `recipe_v1` is the vocabulary the typed ModelSpec SDK
    generates against, so a snapshot where the fold silently stopped producing a
    key would be a codegen defect nothing else here notices.

    Proved by EXECUTING the pairing — fold a pinned class through
    `GraphClassVariant.key()` (which reaches `identity.from_axes` and
    `toolchain_axis_digest`) and project an ingress onto a call signature —
    rather than by reading the import list, which would pass on a tree where
    the shared symbols had changed shape.
    """
    from gen_worker._vendor.torchcg.identity import is_compiled_graph_key
    from gen_worker._vendor.torchcg.ingress import CallIngress, CallInput
    from gen_worker._vendor.torchcg.recipe import (
        GraphClassHash,
        GraphClassVariant,
        IngressDigest,
        LayoutContract,
        ParameterKind,
        call_signature,
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
    variant = GraphClassVariant(
        class_hash=GraphClassHash("0123456789abcdef"),
        ingress_digest=IngressDigest(ingress.digest()),
        ingress=ingress,
        layout=LayoutContract("bf16"),
    )
    signature = call_signature(ingress)
    assert signature.flat_arity == 1
    assert signature.parameters[0].kind is ParameterKind.TENSOR

    class _Runtime:
        sm = "sm_86"
        toolchain = {"torch": "2.13.0"}

    key = variant.key(_Runtime())
    assert is_compiled_graph_key(str(key)), str(key)


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
    # pgw#1370 carve-out: `torchcg` may appear as a PEP 735 DEPENDENCY GROUP
    # requirement + a git source pin. Groups are NOT published in wheel
    # metadata, so no index consumer ever resolves it — the property this
    # fence exists for holds — while the release-derive tests get the real
    # library (the derive imports torchcg from the env it runs in; the
    # standalone-package decision is tcg#41 open question 3, Paul's).
    # The [project] tables above stay fenced for it unchanged.
    # retired-name: the source-pin denylist itself, minus the carved-out row.
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
    """pgw#1457 incident: the re-vendor's one rewrite is easy to forget.

    Vendoring `torchcg` applies exactly one transformation — `from tensorfs
    import ...` becomes `from ..tensorfs import ...` — because upstream
    torchcg depends on tensorfs as a normal package and here there is no such
    package to depend on. Nothing wrote that rule down, so a straight
    `git archive | tar -x` re-vendor dropped it and shipped four modules
    importing a PyPI project that is PERMANENTLY DELETED (ie#738).

    It passed every gate: the digest fence recorded the wrong bytes faithfully,
    the tests passed because the dev venv happens to have a real `tensorfs`
    installed, and only mypy noticed — as a type mismatch between two classes
    with the same name, which reads as a nuisance rather than as "this wheel
    cannot be installed". This fence states the rule so the next re-vendor
    cannot silently skip it.
    """
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
    """pgw#1457 incident #2: pgw runs TWO torchcg copies, by design.

    `_vendor/torchcg` serves the mint and the serving path; the DERIVE imports
    torchcg from the env it runs in, against the `[tool.uv.sources]` dev pin
    (in production, the release env's own pin). That is deliberate — but it
    means a re-vendor that moves only the vendored half leaves two versions
    live in one tree, and the failure lands two tools away from the cause:
    the derive died on `TypeError: unexpected keyword 'session'` while the
    signature was verifiably present in the vendored tree. Both facts true at
    once is the worst kind of bug report.

    They are a MATCHED PAIR, so this asserts it rather than trusting a comment.
    """
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
