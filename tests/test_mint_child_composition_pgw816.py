"""pgw#816: the delegated mint child must load the pipeline the parent SERVES.

The first delegated AOT mint ever to run in production (0.81.0, real L4, sdxl
w8a8) crashed twice in ``phase=load``, 8.5 s each::

    OSError: Error no file named config.json found in directory
      /tmp/tensorhub-cache/cas/snapshots/sha256:32fa2ba6…__x76b2ae62d32f

...while the SERVING process in the same pod was loading that exact directory
fine and answering requests from it throughout.

The ``__x`` suffix is the whole diagnosis, and it is written by our own code:
``snapshot_dir_key`` stamps it when a tree is materialized with an overridden
component EXCLUDED from the fetch (th#1330 B2, worth 167-335 MB/pod). Such a
tree is loadable only TOGETHER with the override trees it was narrowed for.
The parent holds those (``_setup_locked_inner`` resolved them, and injects
them through ``from_pretrained(components=…)``); the child was handed a
``Dict[slot, path]`` and nothing else, so it re-loaded a composition that is
missing a component by construction — and diffusers reported that as a
missing ``config.json`` at the tree's ROOT, naming neither the component nor
the cause.

So the boundary was the bug: a directory path does not describe a
composition. The fix carries the parent's RESOLVED component overrides across
it and loads through the same ``run_setup`` -> ``load_slot`` ->
``from_pretrained(components=…)`` seam the executor uses.

The trees here are produced by a REAL dispatch through the real executor and
the real blake3 CAS downloader (the th#1330 boundary) — a narrowed tree
nobody hand-built, keyed by the code that keys the production one.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import msgspec
import pytest

from gen_worker import fleet_cells, mint_child, mint_delegate
from gen_worker import mint_process as mp
from gen_worker.api.binding import ModelRef, wire_ref
from gen_worker.cli import run as cli_run
from gen_worker.executor import _BackgroundMint
from gen_worker.models import provision
from gen_worker.models.cozy_snapshot import (
    dir_key_excludes_components,
    snapshot_dir_key,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import CompileCell

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import (
    COMPOSED_DECLARED,
    COMPOSED_SETUPS,
    ComposedEndpoint,
    EchoIn,
    EchoOut,
    ToyComposedPipeline,
)

GIB = 1 << 30
BASE_REF = wire_ref(COMPOSED_DECLARED)
VAE_REF = "harness/override-vae:prod"
STUB_MODULE = "harness.mint_child_stub"


# ---------------------------------------------------------------------------
# The trees: produced by a real override dispatch, exactly as a pod's are
# ---------------------------------------------------------------------------


def _base_snapshot(blobs: BlobHost) -> pb.Snapshot:
    model_index = (
        b'{"_class_name": "ToyComposedPipeline",'
        b' "vae": ["harness.toy_endpoints", "ToyVae"],'
        b' "transformer": ["harness.toy_endpoints", "ToyVae"]}'
    )
    return blobs.snapshot("snap-composed-base", [
        blobs.file("mi", model_index, path_in_snapshot="model_index.json"),
        blobs.file("tw", b"base-transformer",
                   path_in_snapshot="transformer/weights.txt"),
        blobs.file("vw", b"base-vae", path_in_snapshot="vae/weights.txt"),
    ])


def _vae_snapshot(blobs: BlobHost) -> pb.Snapshot:
    return blobs.snapshot("snap-override-vae", [
        blobs.file("ow", b"override-vae", path_in_snapshot="weights.txt"),
    ])


def _serve_one_override_dispatch(tmp_path: Path) -> Tuple[Path, Path]:
    """Run one real dispatch carrying a component override; return the
    (narrowed base tree, override tree) the pod ended up serving from."""
    COMPOSED_SETUPS.clear()
    cache_dir = tmp_path / "cas"
    cache_dir.mkdir(parents=True, exist_ok=True)
    blobs = BlobHost(tmp_path)
    try:
        snaps = {BASE_REF: _base_snapshot(blobs), VAE_REF: _vae_snapshot(blobs)}
        models = [pb.ModelBinding(
            slot="pipeline", ref=BASE_REF, components={"vae": VAE_REF})]
        with hub_double(cache_dir=cache_dir) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-816", attempt=1, function_name="composed-echo",
                input_payload=msgspec.msgpack.encode(EchoIn()),
                models=models, snapshots=snaps,
            ))
            res = conn.wait_for(is_result_for("r-816")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        served = msgspec.msgpack.decode(res.inline, type=EchoOut).response
        # The parent SERVED this composition: base bytes + injected override.
        assert "base=base-transformer" in served and "vae=override-vae" in served
        assert "injected=True" in served
    finally:
        blobs.shutdown()

    trees = sorted(p for p in (cache_dir / "cas" / "snapshots").iterdir()
                   if p.is_dir())
    base = [t for t in trees if t.name.startswith("snap-composed-base")]
    override = [t for t in trees if t.name.startswith("snap-override-vae")]
    assert len(base) == 1 and len(override) == 1, trees
    assert dir_key_excludes_components(base[0]), (
        f"the pod's base tree must be override-narrowed: {base[0].name}")
    assert not (base[0] / "vae").exists()
    return base[0], override[0]


@pytest.fixture
def trees(tmp_path: Path) -> Tuple[Path, Path]:
    cli_run._INJECTED_CACHE.clear()
    yield _serve_one_override_dispatch(tmp_path)
    cli_run._INJECTED_CACHE.clear()


def _endpoint_snapshots(base: Path) -> Dict[str, str]:
    return {"pipeline": str(base)}


def _overrides(vae: Path) -> Dict[str, Dict[str, str]]:
    return {"pipeline": {"vae": str(vae)}}


# ---------------------------------------------------------------------------
# 1. The defect, pinned on the real loader
# ---------------------------------------------------------------------------


def test_the_narrowed_tree_alone_is_not_a_loadable_composition(
    trees: Tuple[Path, Path],
) -> None:
    """What the child did: a directory path, no overrides, the real loader.

    This is the production crash reproduced at the seam it happened at. It
    fails the same way before and after the fix — that is the point: the tree
    is not loadable alone, so nothing about the LOADER can be the fix, and
    re-fetching the excluded component (167-335 MB/pod) is the wrong lever.
    """
    base, _vae = trees
    with pytest.raises(Exception) as exc:
        cli_run.run_setup(
            ComposedEndpoint(), _endpoint_snapshots(base),
            arm_compile=False, return_loaded=True)
    # The loader reaches for the component that was deliberately not fetched.
    assert "vae" in str(exc.value), exc.value


# ---------------------------------------------------------------------------
# 2. The fix, on the child's own load path
# ---------------------------------------------------------------------------


def test_the_child_composes_the_pipeline_the_parent_serves(
    trees: Tuple[Path, Path],
) -> None:
    """The child's exact call, with the parent's resolved overrides in hand.

    RED at HEAD (``run_setup`` had no way to be told about the overrides, so
    this is the crash above); GREEN with the fix, and green on the load-bearing
    axis — ``injected=True`` means the pipeline was assembled through the SAME
    ``from_pretrained(components=…)`` seam the serving process used, not
    approximated from a directory.
    """
    base, vae = trees
    instance = ComposedEndpoint()
    loaded = cli_run.run_setup(
        instance, _endpoint_snapshots(base), arm_compile=False,
        return_loaded=True, component_paths=_overrides(vae)) or {}

    pipe = loaded["pipeline"]
    assert isinstance(pipe, ToyComposedPipeline)
    assert pipe.base_weights == "base-transformer"
    assert pipe.vae.content == "override-vae"
    assert pipe.vae_injected is True, (
        "the child must inject the override component, not load the base's")
    assert instance.pipe is pipe, "setup() received the composed pipeline"


def test_the_override_trees_are_part_of_the_slot_identity(
    trees: Tuple[Path, Path],
) -> None:
    """The CLI's warm slot cache is keyed by (annotation, path). Two loads of
    the same base path with DIFFERENT overrides are different pipelines, so a
    path-only key would hand the second one the first one's composition."""
    base, vae = trees
    other = base.parent / "snap-override-vae-2"
    other.mkdir()
    (other / "weights.txt").write_text("second-override-vae")

    def _load(tree: Path) -> Any:
        return (cli_run.run_setup(
            ComposedEndpoint(), _endpoint_snapshots(base), arm_compile=False,
            return_loaded=True,
            component_paths={"pipeline": {"vae": str(tree)}}) or {})["pipeline"]

    first, again, second = _load(vae), _load(vae), _load(other)
    assert again is first, "the same composition must still hit the cache"
    assert second is not first, (
        "a different override tree is a different pipeline")
    assert second.vae.content == "second-override-vae"


# ---------------------------------------------------------------------------
# 3. The boundary: a field nobody fills is the same bug
# ---------------------------------------------------------------------------


def test_the_parent_hands_its_resolved_overrides_across_the_wire(
    tmp_path: Path,
) -> None:
    """``_BackgroundMint`` -> ``MintTask`` -> ``build_request`` -> JSON -> the
    child. Round-tripped through the real msgspec encode/decode, because the
    boundary IS a file."""
    pending = SimpleNamespace(
        family="sdxl", cell_key="ck1-abc",
        cfg=CompileCell(shapes=((1024, 1024),), targets=("unet",),
                        family="sdxl", regional=False, text_len=77,
                        dynamic=(), lora_bucket=0, guidance_scales=(),
                        text_lens=()), target=tmp_path / "cell.tar.gz",
        mint_root=tmp_path, recipe="aot")
    resolved = mp.MintSlot(
        ref=ModelRef(source="tensorhub", path="harness/composed", tag="prod"),
        path="/cas/snapshots/sha256:cafe__xdeadbeef",
        component_paths={"vae": "/cas/snapshots/sha256:beef"})

    bg = _BackgroundMint(
        spec=SimpleNamespace(name="gen"), instance=object(), snapshots=None,
        pendings={}, pipes={},
        modules=("app",),
        slots={"pipeline": resolved},
    )
    task = mint_delegate.MintTask(
        pending=pending, pipe=object(), function="gen",
        modules=bg.modules, slots=dict(bg.slots),
        execution_lane="fp8-w8a16", device=0)
    request = mint_delegate.build_request(
        task, workdir=tmp_path / "w", cap_bytes=7 * GIB)

    wire = msgspec.json.decode(msgspec.json.encode(request), type=mp.MintRequest)
    assert wire.slots == {"pipeline": resolved}, (
        "the child cannot rediscover an override it was never told about")


def test_the_executor_records_what_it_resolved() -> None:
    """The default must be empty, not absent: a flat binding has no overrides
    and must still produce a well-formed request."""
    bg = _BackgroundMint(
        spec=SimpleNamespace(name="gen"), instance=object(), snapshots=None,
        pendings={}, pipes={})
    assert bg.slots == {}


# ---------------------------------------------------------------------------
# 4. A request the child cannot honor refuses BY NAME, and never retries
# ---------------------------------------------------------------------------


def test_a_narrowed_tree_with_no_override_refuses_by_name(
    trees: Tuple[Path, Path], tmp_path: Path,
) -> None:
    """Belt and braces for the wiring, and the difference between a diagnosis
    and a stack trace: the refusal names the SLOT and the reason, before a
    single weight is read."""
    base, _vae = trees
    with pytest.raises(mint_child.MintChildRefused) as exc:
        mint_child.assert_composable({"pipeline": mp.MintSlot(
            ref=ModelRef(source="tensorhub", path="harness/composed",
                         tag="prod"),
            path=str(base))})
    assert "pipeline" in str(exc.value)
    assert "override-narrowed" in str(exc.value)
    assert base.name in str(exc.value)

    # ...and it is the FIRST thing mint() does, so a mis-wired parent costs
    # no discovery, no toolchain probe and no weights read.
    request = mp.MintRequest(
        function="composed-echo", modules=("harness.toy_endpoints",),
        family="sdxl", cell_key="ck1-abc",
        target=str(tmp_path / "cell.tar.gz"),
        work_root=str(tmp_path / "capture"),
        report=str(tmp_path / mp.REPORT_NAME),
        cfg=mp.CompileCellSpec(family="sdxl", shapes=((1024, 1024),),
                               targets=("unet",)),
        slots={"pipeline": mp.MintSlot(
            ref=ModelRef(source="tensorhub", path="harness/composed",
                         tag="prod"),
            path=str(base))},
    )
    with pytest.raises(mint_child.MintChildRefused):
        mint_child.mint(request)
    assert not (tmp_path / "capture").exists(), (
        "the refusal must precede every side effect")


def test_a_complete_tree_never_demands_overrides(tmp_path: Path) -> None:
    """The guard keys on the narrowing marker our own materializer writes —
    a bare-digest (complete) tree is loadable alone and must stay that way."""
    ref = ModelRef(source="tensorhub", path="harness/composed", tag="prod")

    def _one(key: str, **comps: str) -> Dict[str, mp.MintSlot]:
        return {"pipeline": mp.MintSlot(
            ref=ref, path=f"/cas/{key}", component_paths=dict(comps))}

    complete = snapshot_dir_key("sha256:cafe")
    subset = snapshot_dir_key("sha256:cafe", components=("vae",))
    narrowed = snapshot_dir_key("sha256:cafe", exclude=("vae",))
    assert complete == "sha256:cafe"
    mint_child.assert_composable(_one(complete))
    # A component-SCOPED subset is what the caller asked to fetch; loadable.
    mint_child.assert_composable(_one(subset))
    with pytest.raises(mint_child.MintChildRefused):
        mint_child.assert_composable(_one(narrowed))
    # With the override in hand, the same tree is composable.
    mint_child.assert_composable(_one(narrowed, vae="/cas/x"))


# ---------------------------------------------------------------------------
# 5. The OTHER snapshot shape a mint meets: a single-file CAS checkpoint
# ---------------------------------------------------------------------------


class _SingleFileToyPipeline:
    """Diffusers-shaped: a folder layout loads through ``from_pretrained``, a
    loose checkpoint through ``from_single_file``."""

    def __init__(self, source: str, route: str) -> None:
        self.source = source
        self.route = route

    @classmethod
    def from_pretrained(cls, path: str, **_kw: object) -> "_SingleFileToyPipeline":
        raise OSError(
            f"Error no file named config.json found in directory {path}.")

    @classmethod
    def from_single_file(cls, path: str, **_kw: object) -> "_SingleFileToyPipeline":
        return cls(source=str(path), route="single_file")

    def to(self, device: str) -> "_SingleFileToyPipeline":
        return self


def test_a_single_file_checkpoint_routes_around_from_pretrained(
    tmp_path: Path,
) -> None:
    """The control that discriminates the cause.

    A single-file CAS checkpoint (civitai-shaped: one loose ``.safetensors``,
    no ``model_index.json``/``config.json``) has ALWAYS routed through
    ``from_single_file`` in the shared core — so "the child used bare
    diffusers on a single-file artifact" is not what happened, and a fix aimed
    at that shape would have changed nothing. Pinned so the narrowed-tree fix
    cannot regress it.
    """
    snap = tmp_path / "sha256:deadbeef"
    snap.mkdir()
    (snap / "cyberrealistic-xl.safetensors").write_bytes(b"\x00" * 16)

    sl = provision.load_slot(
        _SingleFileToyPipeline, str(snap), slot="pipeline", device="cpu")
    assert sl.is_pipeline
    assert sl.obj.route == "single_file"
    assert sl.obj.source.endswith("cyberrealistic-xl.safetensors")


# ---------------------------------------------------------------------------
# 6. A crash the child CLASSIFIED is deterministic — one attempt, not two
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_child(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(mp, "MINT_CHILD_MODULE", STUB_MODULE)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(root / "src"), str(root / "tests")]))


def _fake_card(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    total, resident = 80 * GIB, 6 * GIB
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev=0: (total - resident, total))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev=0: resident)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=0: resident)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda dev=0: resident + GIB)


def _task(tmp_path: Path) -> mint_delegate.MintTask:
    pending = fleet_cells.PendingSelfMint(
        family="sdxl", cell_key="ck1-abc", ref="root/family-sdxl#ck1-abc",
        cfg=CompileCell(shapes=((1024, 1024),), targets=("unet",),
                        family="sdxl", regional=False, text_len=77,
                        dynamic=(), lora_bucket=0, guidance_scales=(),
                        text_lens=()),
        target=tmp_path / "cell.tar.gz", mint_root=tmp_path / "root", publisher=None, cache_dir=tmp_path,
        delegated=True)
    return mint_delegate.MintTask(
        pending=pending, pipe=SimpleNamespace(), function="gen",
        modules=("harness.toy_endpoints",), weight_lane="fp8", device=0)


class _Act:
    def __init__(self) -> None:
        self.phases: List[str] = []

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self.phases.append(phase)

    def note(self, detail: str) -> None:
        pass


def test_a_classified_child_crash_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _stub_child: None,
) -> None:
    """Two identical 8.5 s crashes bought nothing on the real L4.

    A child that wrote a ``status="failed"`` report caught its own exception,
    named it, and did so from the same request file against the same on-disk
    inputs — attempt 2 re-runs it exactly. Only a death the child could NOT
    classify (no report: signal, OOM-killer, stall kill) can differ next time,
    and a genuine resource shortfall exits ``EXIT_RESOURCE``, which still
    retries.
    """
    _fake_card(monkeypatch)
    seen: List[tuple] = []
    monkeypatch.setattr(
        mint_delegate.activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append((kind, phase, detail)))
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append((kind, phase, detail)))

    monkeypatch.setenv("MINT_STUB_MODE", "failed")
    result = asyncio.run(mint_delegate.build_cell(
        _task(tmp_path), act=_Act(), max_attempts=3))
    assert result.status == mint_delegate.FAILED
    assert result.attempts == 1, (
        "a load failure the child already classified must not buy a second "
        "billed compile of the same failure")

    aborts = [e for e in seen if e[0] == "self_mint_abort"]
    assert aborts and aborts[0][1] == "delegated_crashed"
    assert "deterministic" in aborts[0][2], (
        "the wire must say WHY there was no second attempt")
    assert "kept serving eager" in aborts[0][2]


def test_an_unclassified_death_still_gets_its_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _stub_child: None,
) -> None:
    """The complement, so the fix narrows the retry rather than removing it:
    a child that died without classifying itself is still worth one more."""
    _fake_card(monkeypatch)
    monkeypatch.setattr(
        mint_delegate.activity_mod, "emit_event", lambda *a, **k: None)
    monkeypatch.setattr(
        fleet_cells.activity_mod, "emit_event", lambda *a, **k: None)
    monkeypatch.setenv("MINT_STUB_MODE", "crash")
    result = asyncio.run(mint_delegate.build_cell(
        _task(tmp_path), act=_Act(), max_attempts=2))
    assert result.status == mint_delegate.FAILED
    assert result.attempts == 2


def test_the_retry_classification_reads_the_child_report() -> None:
    """Stated directly, because it is the whole policy."""
    failed = mp.MintReport(status="failed", detail="OSError: no config.json",
                           phase="load")
    classified = mp.MintOutcome(
        status=mp.CRASHED, exit_code=1, report=failed, last_phase="load")
    unclassified = mp.MintOutcome(
        status=mp.CRASHED, exit_code=1, report=None, last_phase="load")
    shortfall = mp.MintOutcome(
        status=mp.RESOURCE, exit_code=3,
        report=mp.MintReport(status="resource"), last_phase="warmup_forward")

    assert classified.retryable is False
    assert unclassified.retryable is True
    assert shortfall.retryable is True
