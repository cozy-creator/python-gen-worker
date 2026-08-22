"""The runtime background mint tells the HUB what happened to it, not a log file."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker import compile_posture
from gen_worker.graphs.env import ArtifactEnv as EnvIdentity
from gen_worker._vendor.torchcg.store import open_artifact
from gen_worker.serving import mint
from gen_worker.serving.mint_store import graph_store


def _artifact(destination: Path, graph: str) -> Path:
    import tcg_artifacts

    return tcg_artifacts.unpacked(destination, graph_specialization=graph)


ENV = EnvIdentity(stack=(("torch", "2.13.0"),), sm="sm_89")


def graph_of(index: int) -> str:
    """A REAL `cg-graph-v1` identity, deterministic per index."""
    return "cg-graph-v1-" + hashlib.sha256(
        f"pgw1573/{index}".encode()).hexdigest()[:56]


class _Adoption:

    def __init__(self, arm_raises: BaseException | None = None) -> None:
        self.env = ENV
        self.armed: List[str] = []
        self.armed_paths: List[Path] = []
        self.arm_raises = arm_raises

    def arm(self, record: Any, artifact: Path) -> None:
        if self.arm_raises is not None:
            raise self.arm_raises
        open_artifact(Path(artifact), Path(artifact).parent / "unpacked")
        self.armed.append(record.graph)
        self.armed_paths.append(Path(artifact))


def _host(graphs: int, *, arm_raises: BaseException | None = None) -> Any:
    holes = tuple(
        SimpleNamespace(record=SimpleNamespace(
            graph=graph_of(i), target="unet"))
        for i in range(graphs))
    return SimpleNamespace(holes=holes, adoption=_Adoption(arm_raises))


def _store(tmp_path: Path) -> Any:
    return graph_store(tmp_path / "cas", None, tmp_path / "no-baked")


def _programs(tmp_path: Path) -> "Callable[[str, Path], Path]":
    def fetch(graph: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"exported-program")
        return destination

    return fetch


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    seen: List[tuple] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append((kind, phase, detail)))
    return seen


def _mint(host: Any, store: Any, tmp_path: Path, **kw: Any) -> mint.BackgroundMint:
    kw.setdefault("program_source", _programs(tmp_path))
    return mint.BackgroundMint(
        host=host, store=store, artifacts_dir=tmp_path / "artifacts",
        posture=compile_posture.FLEET, vcpus=4, **kw)


def test_a_condemned_mint_reports_itself_on_the_wire(
    tmp_path: Path, wire: List[tuple],
) -> None:
    """THE regression."""
    wedged = threading.Event()

    def _compiler(blob: Path, record: Any, destination: Path) -> Path:
        wedged.wait(timeout=10.0)
        return destination

    box = _mint(_host(2), _store(tmp_path), tmp_path, compiler=_compiler,
                window_s=0.3)
    try:
        outcome = box.run()
    finally:
        wedged.set()

    assert outcome.condemned, "the guard did not condemn a wedged mint"
    events = [e for e in wire if e[0] == mint.KIND_MINT_WEDGED]
    assert len(events) == 1, (
        f"a condemned mint must be a wire fact, not a pod-log line: {wire}")
    _kind, phase, detail = events[0]
    assert phase == "no_measured_progress", phase
    assert "hole(s)" in detail and "landed" in detail, detail
    assert "no measured progress" in detail, detail


def test_a_healthy_mint_is_never_called_wedged(
    tmp_path: Path, wire: List[tuple],
) -> None:
    """The polarity guard."""
    host = _host(3)
    store = _store(tmp_path)

    def _compiler(blob: Path, record: Any, destination: Path) -> Path:
        return _artifact(destination, record.graph)

    outcome = _mint(host, store, tmp_path, compiler=_compiler).run()

    assert outcome.landed == 3 and not outcome.condemned
    assert sorted(host.adoption.armed) == sorted(graph_of(i) for i in range(3))
    assert not [e for e in wire if e[0] == mint.KIND_MINT_WEDGED], wire


def test_the_mint_arms_the_STORE_copy_and_never_the_compilers(
    tmp_path: Path, wire: List[tuple],
) -> None:
    host = _host(1)
    store = _store(tmp_path)

    def _compiler(blob: Path, record: Any, destination: Path) -> Path:
        return _artifact(destination, record.graph)

    outcome = _mint(host, store, tmp_path, compiler=_compiler).run()

    assert outcome.landed == 1, outcome.failed
    armed = host.adoption.armed_paths[0]
    assert armed.is_file(), (
        f"{armed} is not a file — the mint armed the compiler's unpacked "
        f"directory again, so a publish/load format skew would be invisible "
        f"from here exactly as it was in pgw#1561")
    with armed.open("rb") as handle:
        assert handle.read(2) == b"\x1f\x8b", (
            f"{armed} is not the tar+gzip envelope boot adoption reads")
    assert armed == tmp_path / "artifacts" / ENV.value / f"{graph_of(0)}.so"
    assert outcome.entries[0].artifact == armed


def test_a_published_graph_that_does_not_arm_is_a_wire_fact(
    tmp_path: Path, wire: List[tuple],
) -> None:
    """`published but did not arm live` is why the pod keeps serving eager for a graph it just paid to compile."""
    host = _host(1, arm_raises=RuntimeError("no module to arm onto"))
    store = _store(tmp_path)

    def _compiler(blob: Path, record: Any, destination: Path) -> Path:
        return _artifact(destination, record.graph)

    outcome = _mint(host, store, tmp_path, compiler=_compiler).run()

    assert outcome.landed == 1
    assert store.has_artifact(graph_of(0), ENV), (
        "the graph did not reach the band boot adoption reads")
    assert not outcome.entries[0].armed
    missed = [e for e in wire if e[0] == mint.KIND_ARM_MISSED]
    assert len(missed) == 1, (
        f"a minted-and-published graph this pod cannot serve must say so: "
        f"{wire}")
    assert "minted and in the" in missed[0][2], missed[0][2]
    assert missed[0][1] == "RuntimeError", missed[0][1]
