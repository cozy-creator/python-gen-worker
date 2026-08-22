from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, cast

import pytest

from gen_worker.graphs.env import ArtifactEnv as EnvIdentity
from gen_worker.serving import mint

ENV = EnvIdentity(stack=(("torch", "2.13.0"),), sm="sm_89")

GRAPH = "cg-graph-v1-9715a0114f7aef25b359294fea1c1b0ca33c3d3e7e17cccabaaa942d"


def _a_real_shared_object() -> bytes:
    for line in Path("/proc/self/maps").read_text().splitlines():
        candidate = line.rsplit(" ", 1)[-1]
        if candidate.startswith("/") and ".so" in candidate:
            raw = Path(candidate).read_bytes()
            if raw[:4] == b"\x7fELF" and raw[4] == 2:
                return raw
    pytest.skip("no 64-bit ELF mapped into this process to read")


def _unpacked_artifact(root: Path, *, with_object: bool = True) -> Path:
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


def test_the_REAL_store_refuses_the_directory_the_compiler_returns(
    tmp_path: Path,
) -> None:
    """RED ARM."""
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker.graphs.store import LocalGraphStore, StoreError

    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    directory = _unpacked_artifact(tmp_path / "artifact")

    with pytest.raises(StoreError) as refusal:
        store.publish_artifact(GRAPH, ENV, directory, cast(Any, None))
    assert "not a file" in str(refusal.value)


def test_needed_libraries_on_the_directory_USED_to_be_unreadable(
    tmp_path: Path,
) -> None:
    """The second half of the same defect, and the reason the assert stays."""
    directory = _unpacked_artifact(tmp_path / "artifact")

    sonames = mint.needed_libraries(directory)

    assert sonames, (
        "the real shared object inside the package links against something; "
        "an empty answer here means the reader never reached it"
    )
    assert all(isinstance(name, str) and name for name in sonames)


def test_the_package_is_the_publishable_file_and_the_object_is_inside_it(
    tmp_path: Path,
) -> None:
    """The two consumers want DIFFERENT objects."""
    directory = _unpacked_artifact(tmp_path / "artifact")

    package = mint.artifact_package(directory)
    assert package.is_file() and package.name == "model.pt2"

    obj = mint.compiled_object_bytes(directory)
    assert obj[:4] == b"\x7fELF"
    assert obj != package.read_bytes()

    assert mint.needed_libraries(package) == mint.needed_libraries(directory)


def test_an_embedded_kernel_package_names_no_libraries_without_erroring(
    tmp_path: Path,
) -> None:
    """An artifact whose kernels are all embedded Triton/SASS carries no shared object."""
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


def _run_one_mint(tmp_path: Path, compiler: Any = None) -> Any:
    from gen_worker import compile_posture
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker.graphs.store import LocalGraphStore

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

    manifest = run.store.get_manifest(GRAPH, ENV)
    assert manifest is not None
    assert manifest.sm_compiled == "sm_89"


def test_what_the_mint_PUBLISHED_is_opened_by_the_REAL_boot_loader(
    tmp_path: Path,
) -> None:
    from gen_worker._vendor.torchcg.store import open_artifact

    run = _run_one_mint(tmp_path)
    assert run.store.artifact_skew(GRAPH, ENV) is None, (
        "the published position is shaped like something no loader opens")

    fetched = run.store.fetch_artifact(GRAPH, ENV, tmp_path / "fetched")
    assert fetched is not None, "boot adoption fetches a MISS at a position it minted"
    with fetched.open("rb") as handle:
        assert handle.read(2) == b"\x1f\x8b", (
            f"{fetched} is not the tar+gzip envelope the boot loader reads")

    graph = open_artifact(fetched, tmp_path / "unpacked")
    assert graph.key, "the materialized artifact states no compiled_graph_key"
    assert graph.package.is_file(), "no model.pt2 inside the envelope"
    assert graph.metadata.get("compiled_graph_key") == graph.key


def test_a_compiler_that_hands_back_the_bare_package_publishes_NOTHING(
    tmp_path: Path,
) -> None:
    """RED ARM."""
    def _package_only(blob: Path, rec: Any, destination: Path) -> Path:
        import tcg_artifacts

        destination.parent.mkdir(parents=True, exist_ok=True)
        return tcg_artifacts.aoti_package(
            destination, graph_specialization=GRAPH)

    with pytest.raises(mint.ArtifactUnreadable) as refusal:
        _run_one_mint(tmp_path, compiler=_package_only)
    assert "bare package cannot be published" in str(refusal.value)
