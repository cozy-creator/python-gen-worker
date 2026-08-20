"""pgw#1471: the runtime mint must be able to PUBLISH what the compiler returns.

`Engine.compile` resolves a minted key into `destination` and leaves an UNPACKED
DIRECTORY there. `_mint_one` then asked two things of it that a directory cannot
be — `publish_artifact` requires a file, and `needed_libraries` requires an ELF
object — so every runtime mint compiled successfully and died at publish.

**These drive the REAL `LocalGraphStore`, deliberately.** `test_runtime_mint.py`
has a `_Store` double whose `publish_artifact` appends to a list and returns, so
it accepts a directory that the real store refuses by precondition. That is why
this defect shipped with tests passing: the double did not model the one
precondition that mattered. Nothing here substitutes for the store.

The ELF used is a real shared object taken off this process's own memory map,
not a synthesized header — the whole point is reading genuine `DT_NEEDED`
entries out of a real package.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, cast

import pytest

from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker.serving import mint

#: The real env identity, not a SimpleNamespace: the store's `_artifact_ref`
#: and its manifest/sm cross-check both read it, and a double that satisfies
#: attribute access without satisfying the type is how this seam went wrong in
#: the first place.
ENV = EnvIdentity(stack=(("torch", "2.13.0"),), sm="sm_89")

#: A REAL cg-graph-v1 hash, taken from the sd1.5 mint on this box. The store
#: addresses rows by these and refuses anything else (`_require_graph`) — the
#: `_Store` double in test_runtime_mint.py accepts any string, which is one
#: more way it diverged from the store it stands in for.
GRAPH = "cg-graph-v1-9715a0114f7aef25b359294fea1c1b0ca33c3d3e7e17cccabaaa942d"


def _a_real_shared_object() -> bytes:
    """Bytes of a real ELF this process actually has mapped.

    Beats a synthesized 64-byte header: that would exercise the parser's
    "names no libraries" branch only, and the bug being guarded is about
    reaching the object at all and reading its real link table.
    """
    for line in Path("/proc/self/maps").read_text().splitlines():
        candidate = line.rsplit(" ", 1)[-1]
        if candidate.startswith("/") and ".so" in candidate:
            raw = Path(candidate).read_bytes()
            if raw[:4] == b"\x7fELF" and raw[4] == 2:
                return raw
    pytest.skip("no 64-bit ELF mapped into this process to read")


def _unpacked_artifact(root: Path, *, with_object: bool = True) -> Path:
    """What `Engine.compile` leaves behind: a DIRECTORY, not a file.

    Layout measured from a real sd1.5 mint on an RTX 4070: `metadata.json`
    beside `model.pt2`, the package holding the compiled object under
    `model/data/aotinductor/<graph>/<hash>.wrapper.so`.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps({"graph": GRAPH}))
    package = root / "model.pt2"
    with zipfile.ZipFile(package, "w") as bundle:
        bundle.writestr("archive_format", "pt2")
        if with_object:
            bundle.writestr(
                f"{mint._AOTI_PREFIX}{GRAPH}/abc123.wrapper.so",
                _a_real_shared_object(),
            )
    return root


# ---------------------------------------------------------------- the red arm

def test_the_REAL_store_refuses_the_directory_the_compiler_returns(
    tmp_path: Path,
) -> None:
    """RED ARM. Publishing the compiler's return value directly is the defect.

    An artifact position addresses ONE set of bytes by digest, so `is_file()`
    is not a formality the store could relax — a directory has no digest to
    address it by. This asserts the precondition is real, which is what makes
    the fix necessary rather than cosmetic.
    """
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.torchcg.store import LocalGraphStore, StoreError

    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    directory = _unpacked_artifact(tmp_path / "artifact")

    with pytest.raises(StoreError) as refusal:
        # The manifest is never reached: `is_file()` is the FIRST thing the
        # store checks, which is the whole point.
        store.publish_artifact(GRAPH, ENV, directory, cast(Any, None))
    assert "not a file" in str(refusal.value)


def test_needed_libraries_on_the_directory_USED_to_be_unreadable(
    tmp_path: Path,
) -> None:
    """The second half of the same defect, and the reason the assert stays.

    Before pgw#1471 this raised `ArtifactUnreadable` — the directory's bytes
    are not an ELF object. The fix reaches INTO the package rather than
    relaxing the assertion; relaxing it would have turned a loud failure into
    a manifest that silently constrains nothing.
    """
    directory = _unpacked_artifact(tmp_path / "artifact")

    sonames = mint.needed_libraries(directory)

    assert sonames, (
        "the real shared object inside the package links against something; "
        "an empty answer here means the reader never reached it"
    )
    assert all(isinstance(name, str) and name for name in sonames)


# ------------------------------------------------------------------- the fix

def test_the_package_is_the_publishable_file_and_the_object_is_inside_it(
    tmp_path: Path,
) -> None:
    """The two consumers want DIFFERENT objects. That is the whole fix."""
    directory = _unpacked_artifact(tmp_path / "artifact")

    package = mint.artifact_package(directory)
    assert package.is_file() and package.name == "model.pt2"

    obj = mint.compiled_object_bytes(directory)
    assert obj[:4] == b"\x7fELF"
    # Reached the object INSIDE the zip, not the zip itself.
    assert obj != package.read_bytes()

    # Idempotent: handed the package directly, the answer is unchanged.
    assert mint.needed_libraries(package) == mint.needed_libraries(directory)


def test_an_embedded_kernel_package_names_no_libraries_without_erroring(
    tmp_path: Path,
) -> None:
    """An artifact whose kernels are all embedded Triton/SASS carries no shared
    object. `needed_libraries` already treats "constrains nothing" as a correct
    answer, and unwrapping must not turn that into a failure."""
    directory = _unpacked_artifact(tmp_path / "artifact", with_object=False)

    assert mint.needed_libraries(directory) == ()
    assert mint.artifact_constraints(directory) == ()


def test_a_directory_carrying_no_package_refuses_BY_NAME(tmp_path: Path) -> None:
    """Absence must say what was missing, not KeyError somewhere downstream."""
    empty = tmp_path / "artifact"
    empty.mkdir()
    (empty / "metadata.json").write_text("{}")

    with pytest.raises(mint.ArtifactUnreadable) as refusal:
        mint.artifact_package(empty)
    assert "no .pt2 package" in str(refusal.value)
    assert "metadata.json" in str(refusal.value), (
        "the refusal should show what the directory DOES hold, so the reader "
        "can tell an empty mint from a differently-shaped one"
    )


# --------------------------------------------------- the end-to-end mint path

def _run_one_mint(tmp_path: Path, compiler: Any = None) -> Any:
    """One `_mint_one` against the REAL `LocalGraphStore`.

    Nothing is stubbed but the compiler, which stands in for inductor and
    returns exactly what `Engine.compile` returns: the unpacked directory.
    """
    from gen_worker import compile_posture
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.torchcg.store import LocalGraphStore

    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    armed: List[str] = []

    class _Adoption:
        env = ENV

        def arm(self, record: Any, artifact: Path) -> None:
            armed.append(record.graph)

    record = SimpleNamespace(graph=GRAPH, target="unet")
    host = SimpleNamespace(holes=(SimpleNamespace(record=record),),
                           adoption=_Adoption())

    def _compiler(blob: Path, rec: Any, destination: Path) -> Path:
        # What Engine.compile does: resolve the key INTO destination, leaving
        # an unpacked directory. A CONFORMING one (pgw#1561): the publish now
        # repacks the envelope and torchcg validates metadata against the
        # package on the way, so the loose `_unpacked_artifact` shape that the
        # direct-reader tests keep using no longer publishes anywhere.
        import tcg_artifacts

        return tcg_artifacts.unpacked(destination, graph_specialization=GRAPH)

    def _program_source(digest: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"exported-program")
        return destination

    box = mint.BackgroundMint(
        host=host, store=store, artifacts_dir=tmp_path / "artifacts",
        posture=compile_posture.FLEET, vcpus=4,
        compiler=compiler or _compiler, program_source=_program_source,
    )

    minted = box._mint_one(record, __import__("threading").Lock())
    return SimpleNamespace(store=store, minted=minted, armed=armed)


def test_mint_one_compiles_publishes_and_manifests_through_the_REAL_store(
    tmp_path: Path,
) -> None:
    """The path that was broken, end to end."""
    run = _run_one_mint(tmp_path)

    assert run.minted.published, "the mint compiled and then published nothing"
    assert run.minted.armed and run.armed == [GRAPH]

    # The manifest was derived from the ELF INSIDE the package, so it carries
    # the real link table rather than an empty set.
    manifest = run.store.get_manifest(GRAPH, ENV)
    assert manifest is not None
    assert manifest.sm_compiled == "sm_89"


def test_what_the_mint_PUBLISHED_is_opened_by_the_REAL_boot_loader(
    tmp_path: Path,
) -> None:
    """pgw#1561's missing round-trip: mint → store → fetch → MATERIALIZE.

    `_mint_one` arms the unpacked directory the compiler just wrote, so
    mint→arm never touches the published bytes at all. Boot adoption is their
    only reader, and no test had ever been one — which is precisely how the
    band held bare `model.pt2` ZIPs from pgw#1471 until va#3 arm 2 fetched
    them and holed 14/14 on `cannot decompress`. Every assertion this seam
    had (published ref non-empty, manifest present, armed) was TRUE over
    those unloadable bytes.

    So this drives the published object back OUT through the store adoption
    reads and opens it with `torchcg.serve.materialize` — the exact function
    boot-time arming calls — and states the two facts the loader needs:
    the blob is the gzip envelope, and it carries its own identity.
    """
    from gen_worker._vendor.torchcg.serve import materialize

    run = _run_one_mint(tmp_path)
    assert run.store.artifact_skew(GRAPH, ENV) is None, (
        "the published position is shaped like something no loader opens")

    fetched = run.store.fetch_artifact(GRAPH, ENV, tmp_path / "fetched")
    assert fetched is not None, "boot adoption fetches a MISS at a position it minted"
    with fetched.open("rb") as handle:
        assert handle.read(2) == b"\x1f\x8b", (
            f"{fetched} is not the tar+gzip envelope the boot loader reads")

    graph = materialize(fetched, tmp_path / "unpacked")
    assert graph.key, "the materialized artifact states no compiled_graph_key"
    assert graph.package.is_file(), "no model.pt2 inside the envelope"
    # metadata.json — the self-stated identity the bare-ZIP publish DISCARDED.
    assert graph.metadata.get("compiled_graph_key") == graph.key


def test_a_compiler_that_hands_back_the_bare_package_publishes_NOTHING(
    tmp_path: Path,
) -> None:
    """RED ARM. The pre-pgw#1561 world, reconstructed at the seam that made it.

    Every publisher since pgw#1471 banked `artifact_package` — the bare
    `.pt2` ZIP — and the store took it, so the defect was invisible from here.
    Handing the publish a package FILE must now be a typed refusal at the
    mint, not fourteen adoption holes an hour later on a pod.
    """
    def _package_only(blob: Path, rec: Any, destination: Path) -> Path:
        import tcg_artifacts

        destination.parent.mkdir(parents=True, exist_ok=True)
        return tcg_artifacts.aoti_package(
            destination, graph_specialization=GRAPH)

    with pytest.raises(mint.ArtifactUnreadable) as refusal:
        _run_one_mint(tmp_path, compiler=_package_only)
    assert "bare package cannot be published" in str(refusal.value)
