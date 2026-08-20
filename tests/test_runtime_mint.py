"""The runtime background mint tells the HUB what happened to it, not a log file.

A serve pod exposes no stdout anybody can read (pgw#760), so a mint that ends
in `logger.error` has, from the hub's side, simply stopped emitting.

# pgw#1383: that is exactly what cost `j56tate13oav13` thirty billed minutes and
# a manual `DELETE /v1/admin/pods` — the hub's stall detector fired four times
# and was right every time; it had no worker-side fact to attribute the silence
# to. The condemnation and the missed arm are now typed wire events.
"""

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
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker._vendor.torchcg.serve import materialize
from gen_worker.serving import mint
from gen_worker.serving.mint_store import graph_store


def _artifact(destination: Path, graph: str) -> Path:
    """A real UNPACKED artifact directory — what `Engine.compile` leaves.

    This used to be a bare 64-byte ELF file, and the `_Store` double accepted
    it — which is the exact modeling gap pgw#1561 measured in the field: the
    REAL publisher now repacks the ENVELOPE from the unpacked directory
    (validating metadata against the package on the way), so a compiler seam
    handing back anything less no longer publishes anywhere.
    """
    import tcg_artifacts

    return tcg_artifacts.unpacked(destination, graph_specialization=graph)


#: The real env identity, not a namespace: the store's `_artifact_ref` and its
#: manifest/sm cross-check both read it.
ENV = EnvIdentity(stack=(("torch", "2.13.0"),), sm="sm_89")


def graph_of(index: int) -> str:
    """A REAL `cg-graph-v1` identity, deterministic per index.

    pgw#1573: these used to be `unet/class=0`. That string is not an identity
    any store addresses rows by — `LocalGraphStore._require_graph` refuses it —
    and the `_Store` double this module carried accepted it, along with a
    directory the real store refuses and a `publish_artifact` that returned a
    literal. Three preconditions unmodelled, which is why this file was green
    through pgw#1471 AND pgw#1561.
    """
    return "cg-graph-v1-" + hashlib.sha256(
        f"pgw1573/{index}".encode()).hexdigest()[:56]


class _Adoption:
    """The adopt session's arm seam — the ONE thing here that is not real.

    torchcg's `AdoptSession.arm` needs a live `nn.Module` with resident
    constants on a device to build a callable, which is a GPU fact. Everything
    up to it is the production object: the real store, the real envelope, the
    real fetch. What is recorded is the PATH the mint armed, which is the whole
    subject of pgw#1573 — it must be the store's copy, never the compiler's.
    """

    def __init__(self, arm_raises: BaseException | None = None) -> None:
        self.env = ENV
        self.armed: List[str] = []
        self.armed_paths: List[Path] = []
        self.arm_raises = arm_raises

    def arm(self, record: Any, artifact: Path) -> None:
        if self.arm_raises is not None:
            raise self.arm_raises
        # The arm READS the bytes it was handed, through the exact function
        # boot adoption's loader calls. An arm that only records a path cannot
        # tell a loadable envelope from a directory, which is the distinction
        # this module now exists to hold.
        materialize(Path(artifact), Path(artifact).parent / "unpacked")
        self.armed.append(record.graph)
        self.armed_paths.append(Path(artifact))


def _host(graphs: int, *, arm_raises: BaseException | None = None) -> Any:
    holes = tuple(
        SimpleNamespace(record=SimpleNamespace(
            graph=graph_of(i), target="unet"))
        for i in range(graphs))
    return SimpleNamespace(holes=holes, adoption=_Adoption(arm_raises))


def _store(tmp_path: Path) -> Any:
    """THE store — `graph_store`, the one constructor every entry point uses.

    `baked_root` is stated so the fixture never reaches the box's real settings
    for a directory it does not need.
    """
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
    """THE regression. The guard already condemns correctly; before this the
    condemnation went to a pod log, so from the hub's side the mint just
    stopped emitting — indistinguishable from the pod being busy and healthy."""
    wedged = threading.Event()

    def _compiler(blob: Path, record: Any, destination: Path) -> Path:
        wedged.wait(timeout=10.0)      # a compile that will never come back
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
    """The polarity guard. A mint that lands its graphs emits no defect —
    otherwise the event is noise the hub learns to ignore."""
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
    """pgw#1573 — the invariant that collapses the two load paths into one.

    `_mint_one` used to arm ``artifact``: the unpacked DIRECTORY the compile
    child had just written, while publishing an ENVELOPE the arm never opened.
    Two producers, two formats, and no run of the mint ever read its own
    published bytes — which is precisely how pgw#1471's bare-``.pt2`` publish
    survived from the first publisher that ever existed until va#3 arm 2
    fetched one on a pod and holed 14/14 on ``cannot decompress``.

    So the assertion is about WHICH BYTES: the armed path must be the store's
    fetch position, it must be a FILE, and it must carry the gzip envelope
    magic the boot loader reads. Arming the compiler's output again fails all
    three.
    """
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
    # And it is THE position boot adoption fetches to, not a scratch copy:
    # torchcg's `AdoptSession._adopt_module` addresses exactly this path.
    assert armed == tmp_path / "artifacts" / ENV.value / f"{graph_of(0)}.so"
    assert outcome.entries[0].artifact == armed


def test_a_published_graph_that_does_not_arm_is_a_wire_fact(
    tmp_path: Path, wire: List[tuple],
) -> None:
    """`published but did not arm live` is why the pod keeps serving eager for
    a graph it just paid to compile. It was a `logger.warning`."""
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
