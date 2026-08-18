"""The runtime background mint tells the HUB what happened to it, not a log file.

A serve pod exposes no stdout anybody can read (pgw#760), so a mint that ends
in `logger.error` has, from the hub's side, simply stopped emitting.

# pgw#1383: that is exactly what cost `j56tate13oav13` thirty billed minutes and
# a manual `DELETE /v1/admin/pods` — the hub's stall detector fired four times
# and was right every time; it had no worker-side fact to attribute the silence
# to. The condemnation and the missed arm are now typed wire events.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker import compile_posture
from gen_worker.serving import mint


def _elf(path: Path) -> Path:
    """A minimal 64-bit ELF with no sections.

    `needed_libraries` reads DT_NEEDED off the artifact itself, and an
    artifact naming no libraries is a legitimate answer (all-embedded
    Triton/SASS) rather than a recording bug — so this is the honest smallest
    fixture, not a stub of the reader.
    """
    raw = bytearray(64)
    raw[:4] = b"\x7fELF"
    raw[4:7] = bytes((2, 1, 1))          # ELFCLASS64, little-endian, v1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(raw))
    return path


class _Adoption:
    def __init__(self, arm_raises: BaseException | None = None) -> None:
        self.env = SimpleNamespace(value="lane-a", sm="sm_89")
        self.armed: List[str] = []
        self.arm_raises = arm_raises

    def arm(self, record: Any, artifact: Path) -> None:
        if self.arm_raises is not None:
            raise self.arm_raises
        self.armed.append(record.graph)


def _host(graphs: int, *, arm_raises: BaseException | None = None) -> Any:
    holes = tuple(
        SimpleNamespace(record=SimpleNamespace(
            graph=f"unet/class={i}", target="unet", program=f"sha256:{i:064x}"))
        for i in range(graphs))
    return SimpleNamespace(holes=holes, adoption=_Adoption(arm_raises))


class _Store:
    """The fleet sink, minus the hub."""

    def __init__(self) -> None:
        self.published: List[str] = []

    def fetch_program(self, digest: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"program")
        return destination

    def publish_artifact(
        self, graph: str, env: Any, artifact: Path, manifest: Any,
    ) -> str:
        self.published.append(graph)
        return "checkpoint-1"


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    seen: List[tuple] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append((kind, phase, detail)))
    return seen


def _mint(host: Any, store: Any, tmp_path: Path, **kw: Any) -> mint.BackgroundMint:
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

    box = _mint(_host(2), _Store(), tmp_path, compiler=_compiler, window_s=0.3)
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

    def _compiler(blob: Path, record: Any, destination: Path) -> Path:
        return _elf(destination)

    outcome = _mint(host, _Store(), tmp_path, compiler=_compiler).run()

    assert outcome.landed == 3 and not outcome.condemned
    assert host.adoption.armed == [f"unet/class={i}" for i in range(3)] or \
        sorted(host.adoption.armed) == sorted(
            f"unet/class={i}" for i in range(3))
    assert not [e for e in wire if e[0] == mint.KIND_MINT_WEDGED], wire


def test_a_published_graph_that_does_not_arm_is_a_wire_fact(
    tmp_path: Path, wire: List[tuple],
) -> None:
    """`published but did not arm live` is why the pod keeps serving eager for
    a graph it just paid to compile. It was a `logger.warning`."""
    host = _host(1, arm_raises=RuntimeError("no module to arm onto"))
    store = _Store()

    def _compiler(blob: Path, record: Any, destination: Path) -> Path:
        return _elf(destination)

    outcome = _mint(host, store, tmp_path, compiler=_compiler).run()

    assert outcome.landed == 1 and store.published == ["unet/class=0"]
    assert not outcome.entries[0].armed
    missed = [e for e in wire if e[0] == mint.KIND_ARM_MISSED]
    assert len(missed) == 1, (
        f"a minted-and-published graph this pod cannot serve must say so: "
        f"{wire}")
    assert "minted and in the" in missed[0][2], missed[0][2]
    assert missed[0][1] == "RuntimeError", missed[0][1]
