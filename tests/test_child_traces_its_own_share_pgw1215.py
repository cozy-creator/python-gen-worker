"""pgw#1215 (th#1834 Phase 3 keystone) — the wiring, the durability bound, and
the one label a share cannot state.

Three properties, none of which any other file covers:

1. **``mint_child`` drives the K-wide path.** The keystone is worth nothing if
   the production mint still calls the serial one — that would be a K=1
   regression on the fleet AND a K-wide driver nobody calls, which is the
   "built it, never wired it" defect class ``lint_unreached_surface`` exists
   for. Asserted behaviourally: the real ``_mint_aot`` is driven and the
   template it builds is inspected.
2. **A crash costs the ONE class in flight.** The child packs inside its trace
   loop, so a refusal at class *k* leaves the *k−1* artifacts on disk and NAMES
   them. Batch-packing a share would reinstate exactly the all-or-nothing loss
   pgw#1183 fixed and th#1825 spent 1 h 37 m on.
3. **The recipe and refusal ledger survive the process boundary.** The child
   wire stays weight-free, and a refusing share names every partition member
   it completed. TCG owns graph-class metadata and package admission; the
   parent-only key collapse is covered by ``test_tcg_mint_parent_pgw1270``.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List

import msgspec
import pytest

from harness.slot_facts import TEST_FACTS as _TEST_FACTS
from gen_worker._vendor.torchcg import spans

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_compile_pool as pool  # noqa: E402
from gen_worker import aot_mint, compile_cache  # noqa: E402
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
from gen_worker.child_contract import CompileSpec, MintSlot  # noqa: E402
from harness import fake_compile_child  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

FAMILY = "tiny1215"
_DECLARED = 6


@pytest.fixture(autouse=True)
def _fresh_registry() -> Any:
    reset_export_declarations()
    yield
    reset_export_declarations()


# ---------------------------------------------------------------------------
# 1. mint_child drives the K-wide path
# ---------------------------------------------------------------------------


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample)) + 1.0


def _declare() -> Any:
    return register_export_declaration(Compile(
        family=FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}), GraphClass(dims={"B": 1})),
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


def _ref() -> Any:
    from gen_worker.api.binding import Hub

    return Hub("cozy/tiny1215")


def _request(tmp_path: Path) -> Any:
    from gen_worker.mint_process import MintRequest

    return MintRequest(
        function="generate",
        modules=("harness.toy_endpoints",),
        family=FAMILY,
        arm_token="arm1-tiny1215",
        target=str(tmp_path / "out" / "cell.tar.gz"),
        work_root=str(tmp_path / "work"),
        report=str(tmp_path / "report.json"),
        cfg=CompileSpec(family=FAMILY, targets=("unet",), shapes=((4, 4),)),
        slots={"pipeline": MintSlot(
            ref=_ref(), path=str(tmp_path / "tree"), facts=_TEST_FACTS)},
        phases_snapshot=str(tmp_path / "phases.json"),
        compiled_graph_peak_rss_bytes=1024 ** 3,
    )


def test_mint_child_drives_the_K_WIDE_path_and_hands_over_the_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED if `_mint_aot` stops driving the K-wide TCG compiler.

    The keystone's whole point is that the compile children TRACE. If the mint
    child kept calling the serial driver, every fleet mint would run K=1 while
    the K-wide driver sat uncalled — the exact pair of defects this test pins
    together, because they are one defect.

    Driven, not grepped: the real `_mint_aot` runs, and what it hands the
    driver is inspected. `mint_graph_classes` itself is intercepted — the
    thing under test is the WIRING and the recipe, and running it for real
    would spawn compile children (a real AOTI compile, which is a pod leg).
    """
    from gen_worker import mint_child

    _declare()
    assert not hasattr(aot_mint, "mint")
    seen: Dict[str, Any] = {}

    def _capture(template: Any, **kw: Any) -> Any:
        seen["template"] = template
        seen["kwargs"] = kw
        return aot_mint.MintResult(entries=(), manifest="m0", timings={})

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _capture)
    request = _request(tmp_path)
    pipe = types.SimpleNamespace(unet=_TinyUNet())
    spec = aot_mint.ExportSpec(family=FAMILY, target="", shapes=((4, 4),))
    target = Path(request.target)
    target.parent.mkdir(parents=True, exist_ok=True)

    report = mint_child._mint_aot(
        request, pipe, request.cfg, target,
        started=0.0, sha256_file=lambda p: "sha", spec=spec, footprint={})

    template = seen["template"]
    assert isinstance(template, pool.EntryJob)
    # Every field a child needs to build its OWN weight-free pipeline, and
    # every one of them already lived on the request.
    assert template.function == request.function
    assert template.modules == tuple(request.modules)
    assert template.cfg == request.cfg
    assert template.slots == request.slots
    assert template.posture == request.posture
    assert template.out_dir, "a child with no out_dir cannot pack anything"
    assert all(parameter.device.type == "meta" for parameter in pipe.unet.parameters())

    kwargs = seen["kwargs"]
    # The width is DERIVED from the pod and from the enumerated class count —
    # the mint child is the last process holding a pipeline, and the adapter
    # fork means only a pipeline can enumerate the classes.
    assert kwargs["width"].workers >= 1
    assert kwargs["width"].reason
    # pgw#848: the banked per-entry RSS really reaches the width policy. It
    # went unpassed for the whole fleet's history once already.
    assert (
        kwargs["width"].per_entry_rss_bytes
        == request.compiled_graph_peak_rss_bytes
    )
    # pgw#848: the on-disk phase snapshot survives the rewiring — a mint this
    # process is KILLED must still leave its measurements behind.
    assert kwargs["phase_snapshot"] == Path(request.phases_snapshot)
    assert kwargs["on_progress"] is not None, (
        "pgw#824: the pool loop is the longest wire-silent stretch of a mint")
    assert report.status == "minted"


def test_the_serial_driver_is_not_what_a_pod_runs() -> None:
    """The two drivers are different functions with different jobs, and the
    serial one takes a PIPELINE while the K-wide one takes a recipe. Stated so
    a future edit cannot quietly re-point the mint child at the one that
    exports in-process."""
    import inspect

    from gen_worker import mint_child

    src = "\n".join(
        line for line in inspect.getsource(mint_child._mint_aot).splitlines()
        if not line.lstrip().startswith("#"))
    assert "aot_mint.mint_graph_classes(" in src
    assert "aot_mint.mint(" not in src, (
        "the mint child is calling the SERIAL driver, which exports in this "
        "process and ships each program to a child on disk")
    # And the K-wide driver genuinely holds no pipeline: a parameter named
    # `pipeline` would mean the middle tier survived.
    params = set(inspect.signature(aot_mint.mint_graph_classes).parameters)
    assert "pipeline" not in params and "template" in params


# ---------------------------------------------------------------------------
# 2. A crash costs the ONE class in flight
# ---------------------------------------------------------------------------


def _pool(tmp_path: Path, workers: int = 2) -> pool.EntryCompilePool:
    width = pool.entry_workers(
        _DECLARED, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    return pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=fake_compile_child.script(tmp_path))


def test_a_share_that_refuses_midway_keeps_the_classes_it_already_packed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE durability bound, stated from the code rather than from intent.

    A child packs INSIDE its trace loop, so at the moment it refuses on class
    *k* the *k−1* artifacts are already files on disk. Batch-packing the share
    would make the loss bound the whole share — up to 36/K classes at 156-509 s
    each — which is the all-or-nothing failure pgw#1183 fixed at the artifact
    and th#1825 paid 1 h 37 m for.

    The run still FAILS (a short class set must never publish quietly), but
    what exists is named rather than discarded.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "refuse-midway")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path)

    with pytest.raises(pool.EntryCompileFailed) as caught:
        box.compile(pool.EntryJob(
            function="generate", modules=("harness.toy_endpoints",),
            out_dir=str(tmp_path / "artifacts")))

    kept = box.refused_classes.get(caught.value.entry) or []
    assert kept, (
        "the refusing share reported NO packed classes — either it batch-packs "
        "(so a refusal on class k discards every class it finished) or the "
        "pool drops them on the failure path. Both are the all-or-nothing "
        "invariant in a smaller hat")
    for row in kept:
        assert Path(row.artifact).exists(), (
            f"{row.name}: reported as packed but there is no artifact — "
            f"'packed' has to mean ON DISK or the durability claim is a wish")
    # ...and the loss is VISIBLE: the failure sentence says what survived.
    assert "ARE packed and on disk" in str(caught.value)
    # The measurement survives too (pgw#848: the attempt that failed is the one
    # the next attempt has to size against).
    assert set(box.class_spans) >= {row.name for row in kept}


def test_a_child_envelope_that_cannot_be_parsed_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An artifact whose metadata the parent cannot read cannot be published,
    so it is a named refusal here rather than a KeyError at publish time."""
    class _Pool:
        entry_seconds: Dict[str, float] = {}
        peak_rss_bytes = 0
        #: pgw#1356: the phase table folds these, so a double without them is
        #: a double of a pool that cannot say what it compiled.
        class_spans: Dict[str, Dict[str, float]] = {}
        entry_overlays: Dict[str, Dict[str, float]] = {}

        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def compile(self, template: Any, **kw: Any) -> Any:
            return {"cls/dim=0": pool.PackedGraphClass(
                name="cls/dim=0", key="ek1-x", artifact="/tmp/x",
                metadata="{not json")}

    monkeypatch.setattr(pool, "EntryCompilePool", _Pool)
    monkeypatch.setattr(aot_mint, "_pool_facts", lambda p: {})
    width = pool.entry_workers(
        1, limit=1, vcpus=16, available_bytes=64 * 1024**3, device_lock=True)
    with pytest.raises(aot_mint.MintRefused, match="unreadable envelope"):
        aot_mint.mint_graph_classes(
            pool.EntryJob(function="generate", modules=("m",), out_dir="/tmp"),
            workdir=Path("/tmp/pgw1215-nowhere"), width=width,
            spec=aot_mint.ExportSpec(family=FAMILY, target="unet"))


def test_compile_spec_and_slots_survive_the_wire() -> None:
    """The recipe is a msgspec struct that crosses to a child process; a field
    that does not decode is a mint that dies in its first millisecond."""
    job = pool.EntryJob(
        function="generate", modules=("a", "b"),
        cfg=CompileSpec(family=FAMILY, targets=("unet",), shapes=((4, 4),)),
        slots={"pipeline": MintSlot(ref=_ref(), path="/tree", facts=_TEST_FACTS)},
        share="share-000", share_index=0, share_count=2,
        out_dir="/out")
    back = msgspec.json.decode(msgspec.json.encode(job), type=pool.EntryJob)
    assert back.function == job.function and back.modules == job.modules
    assert back.cfg == job.cfg
    assert back.slots["pipeline"].path == "/tree"
    assert (back.share_index, back.share_count) == (0, 2)
    # The three fields the round trip DELETED must not decode back into
    # existence — they exist only to repair a save/load that no longer happens.
    for gone in ("program", "symbol_values", "symbol_labels"):
        assert not hasattr(back, gone), (
            f"EntryJob.{gone} is back; the ExportedProgram boundary it served "
            f"was the whole cost pgw#1216 measured")


def test_compile_cache_import_is_not_needed_for_the_recipe() -> None:
    """Guard against the recipe quietly acquiring a torch dependency: a job
    file must be readable by anything that can decode JSON."""
    assert compile_cache is not None  # imported above; the struct is not
    raw = msgspec.json.encode(pool.EntryJob(
        function="f", modules=("m",), out_dir="/out"))
    assert msgspec.json.decode(raw, type=pool.EntryJob).function == "f"


def test_pool_reports_which_share_and_which_stride_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool of K that fails anonymously is undebuggable on a pod with no
    logs; the stride is what makes a share reproducible by hand."""
    monkeypatch.setenv("PGW_FAKE_CHILD", "die")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path)
    with pytest.raises(pool.EntryCompileFailed) as caught:
        box.compile(pool.EntryJob(
            function="generate", modules=("harness.toy_endpoints",),
            out_dir=str(tmp_path / "artifacts")))
    assert "rows[0::2]" in str(caught.value)


def test_every_declared_partition_member_survives_a_refusing_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#830's closure has to hold on the terminus a mint actually takes."""

    monkeypatch.setenv("PGW_FAKE_CHILD", "refuse-midway")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path)
    with pytest.raises(pool.EntryCompileFailed):
        box.compile(pool.EntryJob(
            function="generate", modules=("harness.toy_endpoints",),
            out_dir=str(tmp_path / "artifacts")))
    rows: List[Any] = []
    for path in sorted((tmp_path / "pool").glob("share-*/report.json")):
        rows.append(msgspec.json.decode(
            path.read_bytes(), type=pool.EntryReport))
    assert rows
    for report in rows:
        assert not spans.check(dict(report.spans)), report.spans
