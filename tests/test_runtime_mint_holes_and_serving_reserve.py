"""pgw#1371: the runtime mint — holes-only scope, per-class adopt/publish,
and the serving reserve.

The final-mandate shape (issue text, and pgw#1372's handoff contract): a
serving worker that boots onto an unminted (lane x sm) serves EAGER, mints
ONLY the holes — the graph classes with no artifact anywhere it could see —
and ARMS AND PUBLISHES EACH ONE as it lands, so a killed pod keeps every
completed graph and its successor mints only the remainder. Never
all-or-nothing, at any step.

Three seams, each driven for real with only the compile/arm interior faked
(the local-testing rule — no local inductor, no local .so load):

* the POOL takes an ordered hole list, children intersect their shares with
  it before the ``have`` filter, and coverage is proven from the children's
  own ``targeted_classes`` reports;
* ``adopt_minted_class`` is the per-class half of ``adopt_delegated_mint`` —
  durable, §4.32 arm+gate, publish, the moment a class lands — and the
  terminus FOLDS those records instead of arming or uploading twice;
* the FLEET entry-child tree nices itself (``FLEET_MINT_NICE``): e2e#1892
  run 7 measured the core reserves failing to protect the serving process
  (a 15m50s invocation never returned in 65 minutes beside a 2-on-7 mint),
  and priority is the mechanism `compile_posture`'s own doctrine names as
  the one that works.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import aot_compile_pool as pool_mod
from gen_worker import compile_posture
from harness import fake_compile_child

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_mint, mint_supervisor  # noqa: E402
from gen_worker import fleet_compiled_graphs as fleet  # noqa: E402
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
import tcg_artifacts  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_DECLARED = 6


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in ("PGW_FAKE_CHILD", "PGW_FAKE_DECLARED",
                 "PGW_FAKE_STREAM_HANG_AFTER", "PGW_FAKE_OMIT_TARGETED"):
        monkeypatch.delenv(name, raising=False)
    reset_export_declarations()
    yield
    reset_export_declarations()


# ======================================================================
# 1. The pool mints only the named holes
# ======================================================================

def _pool(tmp_path: Path, *, workers: int = 2) -> pool_mod.EntryCompilePool:
    os.environ["PGW_FAKE_CHILD"] = "ok"
    os.environ["PGW_FAKE_DECLARED"] = str(_DECLARED)
    width = pool_mod.entry_workers(
        _DECLARED, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    return pool_mod.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=fake_compile_child.script(tmp_path))


def _job(tmp_path: Path, **fields: Any) -> pool_mod.EntryJob:
    return pool_mod.EntryJob(
        function="generate", modules=("harness.toy_endpoints",),
        out_dir=str(tmp_path / "artifacts"), **fields)


def test_the_pool_mints_exactly_the_named_holes(tmp_path: Path) -> None:
    """Six declared, two holes named — two classes packed, coverage proven
    from the children's own targeted counts, not from the declaration."""
    packed = _pool(tmp_path).compile(_job(
        tmp_path, hole_classes=("cls/dim=1", "cls/dim=3")))
    assert sorted(packed) == ["cls/dim=1", "cls/dim=3"]


def test_a_hole_this_pod_already_holds_is_not_re_minted(
    tmp_path: Path,
) -> None:
    packed = _pool(tmp_path).compile(_job(
        tmp_path,
        hole_classes=("cls/dim=1", "cls/dim=3"),
        have_classes=("cls/dim=1",)))
    assert sorted(packed) == ["cls/dim=3"], (
        "a hole already packed on disk must cost neither a trace nor a "
        "compile — and the coverage proof must still close over the skip")


def test_a_stale_hole_name_is_loud_and_never_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hole list can outlive a declaration (the release moved). The matched
    classes mint; the unmatched name is a wire-visible confession, not a dead
    mint — coverage accretes and a smaller mint is not a short publish."""
    from gen_worker import activity as activity_mod

    said: List[Tuple[str, str]] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: said.append(
            (str(kw.get("phase", "")), detail)))
    packed = _pool(tmp_path).compile(_job(
        tmp_path, hole_classes=("cls/dim=2", "cls/dim=99")))
    assert sorted(packed) == ["cls/dim=2"]
    stale = [d for p, d in said if p == "stale_holes"]
    assert stale and "1 of 2" in stale[0], (
        "an unmatched hole name must be confessed on the wire — silently "
        "minting fewer classes than asked is the subset-drop defect class")


def test_a_child_that_cannot_prove_hole_coverage_is_refused(
    tmp_path: Path,
) -> None:
    """Without the targeted counts there is no evidence the holes were
    covered rather than silently dropped — the exact claim the whole-set
    proof exists to make unfalsifiable."""
    os.environ["PGW_FAKE_OMIT_TARGETED"] = "1"
    with pytest.raises(pool_mod.EntryCompileFailed) as caught:
        _pool(tmp_path).compile(_job(
            tmp_path, hole_classes=("cls/dim=1", "cls/dim=3")))
    assert "no targeted count" in str(caught.value)


# ======================================================================
# 2. The REAL trace loop filters to the holes
# ======================================================================

FAMILY = "tiny1371"


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample))


def _declare() -> Any:
    return register_export_declaration(Compile(
        family=FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(
            GraphClass(dims={"B": 1}),
            GraphClass(dims={"B": 2}),
            GraphClass(dims={"B": 4}),
        ),
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


def test_trace_for_key_holes_filter_and_share_targeted_agree() -> None:
    """The REAL export loop over the REAL declaration: the hole filter yields
    exactly the named classes, `share_targeted` counts what the filter will
    match (before any have filter), and both read the ONE row order — so the
    parent's proof `sum(targeted) == |declaration ∩ holes|` is true by
    construction, not by convention."""
    decl = _declare()
    pipe = SimpleNamespace(unet=_TinyUNet().eval())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")

    names = []
    for row in aot_mint.trace_for_key(pipe, spec, decl):
        names.append(row.name)
        row.release()
    assert len(names) == 3

    holes = (names[1], "no/such/class")
    minted = []
    for row in aot_mint.trace_for_key(pipe, spec, decl, hole_classes=holes):
        minted.append(row.name)
        assert row.declared == 3, "declared stays the WHOLE declaration"
        row.release()
    assert minted == [names[1]]

    assert aot_mint.share_targeted(
        pipe, spec, decl, hole_classes=holes) == 1
    # Over a 2-way partition the shard-local counts sum to the intersection.
    assert sum(
        aot_mint.share_targeted(
            pipe, spec, decl, share_index=i, share_count=2,
            hole_classes=tuple(names[:2]))
        for i in range(2)) == 2
    # And the have filter runs AFTER the targeted count, so a held hole is
    # counted as targeted (the parent subtracts it knowingly).
    held_minted = [
        r.name for r in aot_mint.trace_for_key(
            pipe, spec, decl, hole_classes=holes, have_classes=(names[1],))]
    assert held_minted == []


# ======================================================================
# 3. Per-class adopt/publish, and the terminus fold
# ======================================================================


class _Publisher:
    def __init__(self) -> None:
        self.published: List[Tuple[str, str]] = []

    def enabled(self) -> bool:
        return True


def _pending(tmp_path: Path, publisher: Any = None, **fields: Any) -> Any:
    cfg = SimpleNamespace(
        family="sdxl", lora_bucket=0, targets=("unet",), shapes=((4, 4),),
        guidance_scales=(), text_lens=())
    pending = fleet.PendingSelfMint(
        family="sdxl", arm_token="arm2-tiny1371",
        ref="root/family-sdxl#arm2-tiny1371", cfg=cfg,
        target=tmp_path / "mint-root" / "cell.tar.gz",
        mint_root=tmp_path / "mint-root", publisher=publisher,
        cache_dir=tmp_path / "cas", **fields)
    pending.mint_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cas").mkdir(parents=True, exist_ok=True)
    return pending


def _patch_arm(
    monkeypatch: pytest.MonkeyPatch, *, refuse: set | None = None,
) -> List[Path]:
    """Fake ONLY the arm/parity interior (a real .so load is a pod leg); the
    metadata it answers with is the artifact's own stamped envelope."""
    from gen_worker import artifact_meta

    calls: List[Path] = []
    refuse = refuse or set()

    def _arm(pipe: Any, cfg: Any, cache_dir: Any, bucket: int,
             artifact: Path, arm_key: Any, verify_numerics: bool = False,
             ) -> Tuple[bool, Any, Tuple[str, str]]:
        calls.append(Path(artifact))
        meta = dict(artifact_meta.read_metadata(Path(artifact)))
        name = str((meta.get("graph_class") or {}).get("name") or "")
        if name in refuse:
            return False, None, ("adopt_probe_failed", f"{name} refused")
        return True, meta, ("", "")

    monkeypatch.setattr(fleet, "_arm_exported_compiled_graph", _arm)
    return calls


def _patch_publish(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
    """Intercept the upload seam — the transport itself cannot run locally.
    Everything before it (durable staging, admit, dedupe) stays real."""
    shipped: List[Tuple[str, str]] = []

    def _pub(publisher: Any, family: str, artifact: Path, meta: dict,
             provenance: Any, compiled_graph_key_digest: str = "",
             mint_duration_ms: int = 0, arm_token: str = "") -> Any:
        shipped.append((compiled_graph_key_digest, str(artifact)))
        return SimpleNamespace(join=lambda *a, **k: None)

    monkeypatch.setattr(fleet, "_publish_async", _pub)
    return shipped


def _two_artifacts(tmp_path: Path) -> Tuple[Path, Path]:
    out = tmp_path / "mint-root" / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    a = tcg_artifacts.build(out / "a.tar.gz", graph_class="denoiser/h=64,w=64")
    b = tcg_artifacts.build(out / "b.tar.gz", graph_class="denoiser/h=96,w=96",
                            witness="0123456789abcdef")
    return a, b


def test_a_class_is_armed_durable_and_published_the_moment_it_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = _Publisher()
    pending = _pending(tmp_path, publisher=publisher)
    arms = _patch_arm(monkeypatch)
    shipped = _patch_publish(monkeypatch)
    a, b = _two_artifacts(tmp_path)

    assert fleet.adopt_minted_class(object(), pending, a) is True
    assert len(arms) == 1 and len(shipped) == 1
    key_a = tcg_artifacts.key_of(a)
    assert shipped[0][0] == key_a
    # Durable BEFORE anything else could lose it: the shipped path is the
    # local CAS copy, not the mint-root artifact the terminus cleans.
    assert shipped[0][1] != str(a)
    assert Path(shipped[0][1]).exists()
    inc = pending._state["incremental"]
    assert list(inc) == ["denoiser/h=64,w=64"]

    # Landing the same class twice (stream + report race) uploads ONCE.
    assert fleet.adopt_minted_class(object(), pending, a) is True
    assert len(shipped) == 1

    # THE TERMINUS FOLD: the batch adopt arms only the row the incremental
    # path never saw, and the already-shipped key is not uploaded again.
    minted = fleet.adopt_delegated_mint(object(), pending, [a, b])
    assert minted is not None
    # Counted, not name-matched: the terminus renames rows[0] onto the
    # pending's canonical target, so a re-arm of `a` would wear a different
    # basename. One incremental arm of a + one terminus arm of b — exactly.
    assert len(arms) == 2, (
        f"the terminus re-armed a class `adopt_minted_class` already armed — "
        f"one arm per class, however the mint ends (arms: "
        f"{[p.name for p in arms]})")
    entries = pending._state["adopted_entries"]
    assert {k for k, _p, _m in entries} == {key_a, tcg_artifacts.key_of(b)}

    fleet.publish_self_mint(pending)
    assert [k for k, _p in shipped].count(key_a) == 1, (
        "the terminus publish re-uploaded a key that shipped when it landed")


def test_a_mid_mint_refusal_costs_that_class_and_only_that_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = _Publisher()
    pending = _pending(tmp_path, publisher=publisher)
    arms = _patch_arm(monkeypatch, refuse={"denoiser/h=64,w=64"})
    shipped = _patch_publish(monkeypatch)
    a, b = _two_artifacts(tmp_path)

    assert fleet.adopt_minted_class(object(), pending, a) is False
    assert shipped == [], "a refused class must never ship"
    assert "denoiser/h=64,w=64" in pending._state["incremental_refused"]

    minted = fleet.adopt_delegated_mint(object(), pending, [a, b])
    assert minted is not None, "the sibling still adopts"
    assert minted.compiled_graph_key == tcg_artifacts.key_of(b)
    # The refusal was folded, not re-armed: one incremental arm of a (the
    # refusal itself) + one terminus arm of b — the terminus never re-tried a.
    assert len(arms) == 2, [p.name for p in arms]


def test_supervise_scopes_the_mint_to_the_pendings_holes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handoff contract (pgw#1372 fills it): `PendingSelfMint.holes` is
    the mint's scope — the template the children get carries it, the width is
    sized to it, and the per-class adopt sink rides along."""
    monkeypatch.setattr(
        mint_supervisor, "assert_family_mintable", lambda family: None)
    monkeypatch.setattr(
        fleet, "aot_export_spec",
        lambda pipe, cfg: SimpleNamespace(
            family="sdxl", strict=True, lora_bucket=0))
    monkeypatch.setattr(
        mint_supervisor, "export_declaration", lambda family: object())
    monkeypatch.setattr(
        aot_mint, "declared_class_rows",
        lambda pipe, spec, decl: [object()] * _DECLARED)
    monkeypatch.setattr(
        fleet, "abandon_self_mint", lambda pending: None)

    seen: Dict[str, Any] = {}

    def _mint(template: Any, **kw: Any) -> Any:
        seen["template"], seen["kwargs"] = template, kw
        raise aot_mint.MintRefused("stop here — the wiring is the test")

    monkeypatch.setattr(aot_mint, "mint_graph_classes", _mint)

    pending = _pending(tmp_path, holes=("cls/a", "cls/b"))
    task = mint_supervisor.MintTask(
        pending=pending, pipe=object(), function="generate",
        modules=("harness.toy_endpoints",))

    class _Act:
        def phase(self, *a: Any, **k: Any) -> None: ...
        def note(self, *a: Any) -> None: ...
        def heartbeat(self) -> None: ...

    result = asyncio.run(mint_supervisor.supervise(
        task, act=_Act(), max_attempts=1))
    assert result.status == mint_supervisor.FAILED

    template = seen["template"]
    assert template.hole_classes == ("cls/a", "cls/b")
    assert seen["kwargs"]["width"].entries == 2, (
        "the pool is sized to the HOLES, not the declaration — a 2-hole mint "
        "on a 36-class family must not open a 36-class-wide pool")
    assert callable(seen["kwargs"]["on_landed"]), (
        "the per-class adopt/publish sink is not wired — a killed pod would "
        "lose every completed graph again")


def test_supervise_with_every_hole_held_folds_and_never_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mint_supervisor, "assert_family_mintable", lambda family: None)
    monkeypatch.setattr(
        fleet, "aot_export_spec",
        lambda pipe, cfg: SimpleNamespace(
            family="sdxl", strict=True, lora_bucket=0))
    monkeypatch.setattr(
        mint_supervisor, "export_declaration", lambda family: object())
    monkeypatch.setattr(
        aot_mint, "declared_class_rows",
        lambda pipe, spec, decl: [object()] * _DECLARED)
    held_rows = [
        SimpleNamespace(entry="cls/a", key="k1", artifact=Path("/x"),
                        metadata={}),
    ]
    monkeypatch.setattr(
        mint_supervisor, "held_graph_classes", lambda out_dir: held_rows)
    folded: List[Any] = []

    def _fold(held: Any, *, spec: Any) -> Any:
        folded.append(held)
        raise aot_mint.MintRefused("fold reached — that is the assertion")

    monkeypatch.setattr(aot_mint, "fold_held_graph_classes", _fold)
    monkeypatch.setattr(
        aot_mint, "mint_graph_classes",
        lambda *a, **k: pytest.fail(
            "a fully-held holes mint opened a compile pool"))
    monkeypatch.setattr(fleet, "abandon_self_mint", lambda pending: None)

    pending = _pending(tmp_path, holes=("cls/a",))
    task = mint_supervisor.MintTask(
        pending=pending, pipe=object(), function="generate", modules=("m",))

    class _Act:
        def phase(self, *a: Any, **k: Any) -> None: ...
        def note(self, *a: Any) -> None: ...
        def heartbeat(self) -> None: ...

    result = asyncio.run(mint_supervisor.supervise(
        task, act=_Act(), max_attempts=1))
    assert result.status == mint_supervisor.FAILED and folded


def test_enable_compiled_scopes_the_pending_to_the_callers_holes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The executor half of the handoff: a MISS opened with named holes
    produces a `PendingSelfMint` carrying them — the value `supervise` reads.
    Real `enable_compiled`, the pgw#1033 miss fixture shape."""
    from gen_worker import compile_cache as cc_mod  # noqa: F401
    from gen_worker import config as gw_config
    from gen_worker.api.decorators import Compile as _Compile
    from gen_worker.compiled_graph_adopt import AdoptOutcome

    gw_config.reload_for_test()
    monkeypatch.setattr(
        fleet.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: AdoptOutcome.miss(
            "no_compiled_graph"))
    monkeypatch.setattr(fleet.cc, "has_compile_target", lambda p, c, **_kw: True)
    monkeypatch.setattr(fleet.cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fleet.cc, "mandatory_serving", lambda p: False)
    monkeypatch.setattr(
        fleet.cc, "apply_lora_execution_lane", lambda p, b, **_kw: None)
    monkeypatch.setattr(fleet.cc, "drop_lora_execution_lane", lambda p: None)
    monkeypatch.setattr(fleet, "_cuda_ready", lambda: True)
    monkeypatch.setattr(fleet, "_PENDING", {})
    arm_token = fleet.ARM_SCHEME + "-" + "a" * fleet.ARM_DIGEST_HEX
    monkeypatch.setattr(
        fleet, "arm_identity",
        lambda *a, **k: fleet.ArmIdentity(
            facts=(("family", "sdxl"), ("token_pin", arm_token))))
    monkeypatch.setattr(
        fleet.ArmIdentity, "token", property(lambda self: arm_token))
    monkeypatch.setattr(
        fleet.loading, "pipeline_weight_lane", lambda pipe: "plain")
    register_export_declaration(_Compile(
        family="sdxl", targets=("unet",), text_len=77,
        shapes=((1024, 1024),),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", 4, 128, 128), dtype="bfloat16"),),
        shape_strategy="static-rows", warm_changes_key=False))

    cfg = SimpleNamespace(
        family="sdxl", lora_bucket=0, shapes=((1024, 1024),),
        targets=("unet",), text_lens=(77,), guidance_scales=(1.0,),
        regional=False)
    outcome = fleet.enable_compiled(
        object(), cfg, tmp_path, None, publisher=None, delegate=True,
        holes=("cls/a", "cls/b"))
    pending = outcome.self_mint
    assert isinstance(pending, fleet.PendingSelfMint)
    assert pending.holes == ("cls/a", "cls/b"), (
        "the holes died between the caller and the obligation — the mint "
        "would cover the whole declaration again")
    fleet.abandon_self_mint(pending)


# ======================================================================
# 4. The serving reserve: the FLEET entry-child tree yields
# ======================================================================

def test_fleet_entry_children_apply_the_serving_reserve_nice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """e2e#1892 run 7: 2 children on 7 vCPU starved the serving process into
    a 65-minute non-return (15m50s baseline) — the core reserve bounds the
    AVERAGE while inductor bursts `compile_threads` workers per child. The
    kernel-level reserve is priority: the ENTRY child nices itself on FLEET,
    and the serving process (never niced) wins every contended slice."""
    from gen_worker import aot_compile_child

    levels: List[int] = []

    def _nice(inc: int) -> int:
        levels.append(int(inc))
        return int(inc)

    monkeypatch.setattr(os, "nice", _nice)
    # `_install_posture` publishes the posture process-wide; restore the
    # module global so a sibling test's width arithmetic is not run under
    # this tape's USER_MACHINE.
    monkeypatch.setattr(compile_posture, "_INSTALLED", None)
    monkeypatch.setattr(
        aot_compile_child, "arm_parent_death_signal", lambda: True)

    aot_compile_child._install_posture(pool_mod.EntryJob(
        function="f", modules=("m",), posture=compile_posture.FLEET))
    assert levels == [compile_posture.FLEET_MINT_NICE] == [10]

    levels.clear()
    aot_compile_child._install_posture(pool_mod.EntryJob(
        function="f", modules=("m",), posture=compile_posture.USER_MACHINE))
    assert levels == [compile_posture.USER_MACHINE_NICE] == [19]
