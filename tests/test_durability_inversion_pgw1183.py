"""pgw#1183 / §4.33 step 3, §1.5 — durability is INVERTED, and it is why a
1 h 37 m mint was destroyed.

The ordering §1.5 calls non-negotiable is **pack -> local CAS -> verify -> arm
-> publish**. What the tree did instead, on every fleet path:

* ``local_cell_store.store`` was gated on ``local_keep_reason`` — a fact about
  the SINK — so a *trusted* pod, the only kind that ever mints for the fleet,
  wrote **no durable copy at all**;
* ``_publish_async``'s ``finally`` rmtree'd the mint root on every exit, so a
  transport hiccup destroyed the sole copy of a completed mint;
* the publish ran on a ``daemon`` thread whose own event text conceded the
  consequence — *"this pod must survive the upload or the cell is lost"*;
* an arm refusal, a withheld publish and a sinkless pod each rmtree'd the sole
  copy too.

Every row here asserts the TARGET ordering and was RED on unmodified
``origin/master`` (@ ``6af16cb8``) — see the issue for the run.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import fleet_cells, local_cell_store
from gen_worker.cell_adopt import AdoptOutcome

KEY_A = "cg-key-v1-" + "a" * 56
ARM_A = fleet_cells.ARM_SCHEME + "-" + "1" * fleet_cells.ARM_DIGEST_HEX


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cozy-cells"
    monkeypatch.setenv(local_cell_store.ENV_STORE_DIR, str(root))
    return root


def _artifact(tmp_path: Path, *, key: str = KEY_A, name: str = "mint") -> Path:
    """A cell with readable metadata — every gate here reads the stamp."""
    p = tmp_path / name / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"kind": "aot-inductor", "cell_key": key, "family": "micro-diffusion"}
    ).encode()
    with tarfile.open(p, mode="w:gz") as tar:
        info = tarfile.TarInfo("metadata.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return p


class _Pipe:
    pass


@dataclass
class _Cfg:
    family: str = "micro-diffusion"
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, int], ...] = ((64, 64),)
    targets: Tuple[str, ...] = ("transformer",)
    text_lens: Tuple[int, ...] = (16,)
    guidance_scales: Tuple[float, ...] = (1.0,)
    regional: bool = False


class _Arm:
    def __init__(self, token: str = ARM_A) -> None:
        self.token = token

    def facts_dict(self) -> Dict[str, str]:
        return {}


class _Sink:
    """A LIVE publish sink — the trusted fleet pod, the case that wrote nothing."""

    def __init__(self, on: bool = True) -> None:
        self._on = on
        self.published: List[Path] = []

    def enabled(self) -> bool:
        return self._on

    def publish(self, family: str, art: Path, meta: dict,
                mint_duration_ms: int = 0) -> str:
        self.published.append(Path(art))
        return "ckpt-1"


def _pending(tmp_path: Path, publisher: Any) -> fleet_cells.PendingSelfMint:
    root = tmp_path / "mint"
    return fleet_cells.PendingSelfMint(
        family="micro-diffusion", arm_token=ARM_A, ref="r#x", cfg=_Cfg(),
        target=root / "cell.tar.gz", mint_root=root, publisher=publisher,
        arm_key=_Arm(),  # type: ignore[arg-type]
    )


@pytest.fixture()
def quiet(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
    events: List[Tuple[str, str]] = []

    def _emit(kind: str, detail: str = "", **kw: Any) -> None:
        events.append((kind, str(kw.get("phase") or "")))

    monkeypatch.setattr(fleet_cells.activity_mod, "emit_event", _emit)
    monkeypatch.setattr(fleet_cells, "_note_durable", lambda *a, **k: None)
    monkeypatch.setattr(fleet_cells, "arm_axis_divergence",
                        lambda arm, meta, **_kw: "")
    monkeypatch.setattr(fleet_cells, "_FINALIZED", {})
    return events


def _arming(monkeypatch: pytest.MonkeyPatch, *, ok: bool,
            observed: Optional[List[str]] = None) -> None:
    """Stand `provision.arm_aot` up, recording the CAS's view of the artifact
    AT THE MOMENT OF THE ARM — which is what proves the ORDER."""

    def _arm(pipe: Any, cfg: Any, cache_dir: Any, artifact: Path,
             bucket: int, meta: Any = None, **_kw: Any) -> AdoptOutcome:
        if observed is not None:
            observed.extend(
                c.verdict for c in local_cell_store.stored_cells()
                if c.key == KEY_A and c.artifact.is_file())
        if ok:
            return AdoptOutcome.hit(KEY_A)
        return AdoptOutcome.miss("compile_cell_failed", "the card said no")

    monkeypatch.setattr(fleet_cells.provision, "arm_aot", _arm)


# ---------------------------------------------------------------------------
# 1. The inversion itself: a TRUSTED pod wrote no durable copy
# ---------------------------------------------------------------------------


def test_a_trusted_pod_writes_the_cell_to_local_cas(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    quiet: List[Tuple[str, str]],
) -> None:
    """RED on master: ``local_cell_store.store`` sat behind
    ``local_keep_reason``, which is "" for a pod with a live sink — so the one
    machine class that mints for the fleet kept nothing, and the artifact's
    only copy lived in a temp dir owned by the process most likely to die."""
    _arming(monkeypatch, ok=True)
    sink = _Sink()
    pending = _pending(tmp_path, sink)
    art = _artifact(tmp_path)

    assert fleet_cells.adopt_delegated_mint(_Pipe(), pending, [art]) is not None

    kept = local_cell_store.lookup(KEY_A)
    assert kept is not None, (
        "a trusted pod finished a mint and wrote NO durable copy — the "
        "artifact exists only under the mint root every terminus rmtrees")


def test_the_cas_copy_exists_BEFORE_the_arm_runs(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    quiet: List[Tuple[str, str]],
) -> None:
    """§1.5's ordering is the whole point: durable BEFORE anything downstream
    can destroy it. Storing after a successful arm would still lose every mint
    whose arm crashes the process.

    And it is durable as UNVERIFIED, not as admitted: the bytes exist before
    anything has proven them, so the store must not hand them to a later
    boot's arm until the gate has answered."""
    seen: List[str] = []
    _arming(monkeypatch, ok=True, observed=seen)
    fleet_cells.adopt_delegated_mint(
        _Pipe(), _pending(tmp_path, _Sink()), [_artifact(tmp_path)])
    assert seen == [local_cell_store.VERDICT_UNVERIFIED], (
        "the arm ran before the artifact was durable")
    assert local_cell_store.lookup(KEY_A) is not None, (
        "the gate passed and the cell was never promoted to admitted")


def test_an_arm_refusal_QUARANTINES_the_bytes_instead_of_deleting_them(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    quiet: List[Tuple[str, str]],
) -> None:
    """§1.3.4: a refused entry is *kept quarantined-local for forensics*. On
    master the refusal path rmtree'd the mint root, so the one artifact that
    could explain the refusal was destroyed by the code reporting it."""
    _arming(monkeypatch, ok=False)
    assert fleet_cells.adopt_delegated_mint(
        _Pipe(), _pending(tmp_path, _Sink()), [_artifact(tmp_path)]) is None

    assert local_cell_store.lookup(KEY_A) is None, (
        "a cell that failed its arm must never be armable from the store")
    quarantined = local_cell_store.quarantined_cells()
    assert [c.key for c in quarantined] == [KEY_A], (
        "the refused artifact was deleted, not quarantined")


# ---------------------------------------------------------------------------
# 2. The publish thread: no rmtree, no daemon, and a failure is retryable
# ---------------------------------------------------------------------------


def test_a_failed_publish_does_not_destroy_the_bytes(
    store: Path, tmp_path: Path, quiet: List[Tuple[str, str]],
) -> None:
    """THE MONEY BUG. ``finally: shutil.rmtree(artifact.parent)`` ran on every
    exit path, so one ``connection reset`` deleted a completed mint."""
    art = _artifact(tmp_path)
    local_cell_store.store(art, key=KEY_A, family="micro-diffusion",
                           arm_token=ARM_A, sink=local_cell_store.SINK_OWED)

    class _Broken:
        def publish(self, *a: Any, **k: Any) -> str:
            raise RuntimeError("connection reset")

    fleet_cells._publish_async(
        _Broken(), "micro-diffusion",  # type: ignore[arg-type]
        local_cell_store.lookup(KEY_A).artifact,  # type: ignore[union-attr]
        {"cell_key": KEY_A}, cell_key_digest=KEY_A, arm_token=ARM_A,
    ).join(timeout=30)

    kept = local_cell_store.lookup(KEY_A)
    assert kept is not None and kept.artifact.is_file(), (
        "a transport failure destroyed the sole copy of a completed mint")
    assert kept.sink == local_cell_store.SINK_OWED, (
        "a failed publish must stay PENDING so the next boot retries it")


def test_the_publish_thread_is_not_a_daemon(
    store: Path, tmp_path: Path, quiet: List[Tuple[str, str]],
) -> None:
    """Its own event text stated the defect: *"this pod must survive the upload
    or the cell is lost"*. A daemon thread is killed at interpreter exit with
    no unwinding at all."""
    art = _artifact(tmp_path)
    local_cell_store.store(art, key=KEY_A, family="f", arm_token=ARM_A,
                           sink=local_cell_store.SINK_OWED)
    t = fleet_cells._publish_async(
        _Sink(), "f",  # type: ignore[arg-type]
        local_cell_store.lookup(KEY_A).artifact,  # type: ignore[union-attr]
        {"cell_key": KEY_A}, cell_key_digest=KEY_A, arm_token=ARM_A)
    assert t.daemon is False
    t.join(timeout=30)


def test_a_pending_publish_is_retried_on_the_NEXT_boot(
    store: Path, tmp_path: Path, quiet: List[Tuple[str, str]],
) -> None:
    """Cross-boot retry is what makes publish failure survivable rather than
    terminal. On master a publish that never completed left nothing behind at
    all — the bytes were gone and no record said an upload was owed."""
    local_cell_store.store(_artifact(tmp_path), key=KEY_A, family="f",
                           arm_token=ARM_A, sink=local_cell_store.SINK_OWED)
    sink = _Sink()

    for t in fleet_cells.resume_owed_publishes(sink):  # type: ignore[arg-type]
        t.join(timeout=30)

    assert [p.name for p in sink.published] == ["cell.tar.gz"]
    kept = local_cell_store.lookup(KEY_A)
    assert kept is not None
    assert kept.sink == local_cell_store.SINK_DELIVERED, (
        "a completed publish must stop being owed, or every boot re-uploads")


def test_a_published_cell_is_not_re_uploaded_next_boot(
    store: Path, tmp_path: Path, quiet: List[Tuple[str, str]],
) -> None:
    local_cell_store.store(_artifact(tmp_path), key=KEY_A, family="f",
                           arm_token=ARM_A, sink=local_cell_store.SINK_DELIVERED)
    sink = _Sink()
    assert list(fleet_cells.resume_owed_publishes(sink)) == []  # type: ignore[arg-type]
    assert sink.published == []


def test_a_cell_with_no_sink_by_design_owes_no_publish(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    quiet: List[Tuple[str, str]],
) -> None:
    """cozy-local (§4.28) never has a sink, so its cells must be durable AND
    must not accumulate an upload obligation nothing will ever discharge."""
    _arming(monkeypatch, ok=True)
    fleet_cells.adopt_delegated_mint(
        _Pipe(), _pending(tmp_path, None), [_artifact(tmp_path)])
    kept = local_cell_store.lookup(KEY_A)
    assert kept is not None
    assert kept.sink == local_cell_store.SINK_NONE
    assert list(fleet_cells.resume_owed_publishes(None)) == []


# ---------------------------------------------------------------------------
# 3. Structural: no rmtree may reach a sole copy
# ---------------------------------------------------------------------------


def test_no_terminus_deletes_a_cell_the_store_does_not_hold(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    quiet: List[Tuple[str, str]],
) -> None:
    """The mint root is SCRATCH once the CAS write lands, and every terminus
    is free to clean it. This pins the precondition rather than the rmtrees:
    a terminus that runs on a pending whose bytes never reached the store is
    the destruction bug rebuilt."""
    _arming(monkeypatch, ok=True)
    pending = _pending(tmp_path, _Sink())
    fleet_cells.adopt_delegated_mint(_Pipe(), pending, [_artifact(tmp_path)])
    fleet_cells.publish_self_mint(pending)

    kept = local_cell_store.lookup(KEY_A)
    assert kept is not None and kept.artifact.read_bytes(), (
        "the publish terminus destroyed the durable copy")


def test_local_keep_reason_is_gone(
    store: Path,
) -> None:
    """The gating predicate itself, not just its call site: a durability
    decision that reads a fact about the SINK is the inversion. What survives
    is ``no_publish_sink_reason``, which decides only whether an upload is
    OWED."""
    assert not hasattr(fleet_cells, "local_keep_reason")
    assert fleet_cells.no_publish_sink_reason(None) == \
        fleet_cells.KEEP_NO_PUBLISHER
    assert fleet_cells.no_publish_sink_reason(_Sink()) == ""  # type: ignore[arg-type]
