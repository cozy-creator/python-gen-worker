"""pgw#1089 (§4.27 step 1) — the boot-side ``ck1`` derivation.

Every row here runs WITHOUT a compile, which is the property the issue is about:
if any of it needed AOTI, the derivation it proves would not be a derivation
from code alone. (It also keeps the suite inside Paul's 2026-08-10 rule that
mints run on remote machines only — a fake-tensor trace is not a mint, and the
rows below do not even trace: they exercise the fold, the memo and the pool
arithmetic on synthetic blocks. The end-to-end derive-and-compare against a
real mint's stamp is a POD leg by construction, because a stamp needs a mint.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from torch_compiled_graphs import is_compiled_graph_key
from torch_compiled_graphs.identity import from_axes, toolchain_axis_digest

from gen_worker import boot_key
from gen_worker.child_contract import CompileSpec


# ---------------------------------------------------------------------------
# Synthetic TCG declaration outputs. Real exported-program identity is covered
# by test_tcg_compile_child_pgw1232 and test_graph_witness_pgw1031.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _gpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the runtime key-complete on a GPU-less box.

    Patches only the PROBES: every value is a real fact of some runtime, and
    nothing about how they are combined into a key is faked. ``sm`` is a KEY
    AXIS, so a box that cannot state one cannot state a key at all — which is
    correct behaviour and not something to test around.
    """
    import torch

    from gen_worker import compile_cache

    full = {
        "sku": "l4", "sm": "sm_89", "torch": str(torch.__version__),
        "triton": "3.6.0", "cuda": "13.0",
        "image_digest": "sha256:" + "ab" * 32,
    }
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
def _block(target: str = "unet", *, dim: int = 64, fqns: Any = None) -> Dict[str, Any]:
    payload = json.dumps(
        {"target": target, "dim": dim, "fqns": fqns or ["w.weight"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {"class_hash": hashlib.sha256(payload.encode()).hexdigest()[:16]}


def _fold(blocks: Dict[str, Dict[str, Any]]):
    return boot_key.fold(blocks, family="tiny")


# ---------------------------------------------------------------------------
# THE headline property: TCG owns every key axis operation
# ---------------------------------------------------------------------------


def test_fold_calls_the_same_tcg_axis_authority_as_mint() -> None:
    from gen_worker import compile_cache as cc

    blocks = {"forward/x@64": _block(dim=64), "forward/x@128": _block(dim=128)}
    expected = {
        name: str(from_axes({
            "graph": row["class_hash"],
            "sm": "sm_89",
            "toolchain": toolchain_axis_digest(dict(cc.toolchain_digest())),
        }))
        for name, row in blocks.items()
    }
    assert _fold(blocks) == expected


def test_every_key_is_a_cg_key_v1_key_over_exactly_three_axes() -> None:
    entry_keys = _fold({"a": _block()})
    (key,) = entry_keys.values()
    assert key.startswith("cg-key-v1-")
    assert is_compiled_graph_key(key)


@pytest.mark.parametrize(
    "block",
    [
        {},
        {"class_hash": "0" * 15},
        {"class_hash": "G" * 16},
        {"class_hash": "0" * 16, "graph_witness": "0" * 16},
    ],
)
def test_fold_refuses_any_noncanonical_tcg_class_hash(block: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="class_hash|contain only"):
        _fold({"a": block})


# ---------------------------------------------------------------------------
# Parallelism is not an identity axis
# ---------------------------------------------------------------------------


def test_class_order_and_assignment_do_not_move_the_key() -> None:
    """N-wide and 1-wide must derive the identical key on the same tree.

    Two independent orderings are exercised: the ORDER the blocks arrive in
    (children finish in whatever order they finish) and the SHARE each child
    is assigned. Both are structurally incapable of moving the key —
    the fold addresses each named class independently — and "structurally
    incapable" is exactly the claim that must be pinned, because
    the compile pool's equivalent discipline (assembly by entry NAME, never by
    completion) had to be stated to be kept.
    """
    names = [f"forward/x@{d}" for d in (64, 128, 192, 256, 320)]
    blocks = {n: _block(dim=64 + 64 * i) for i, n in enumerate(names)}

    forward = _fold(blocks)
    reversed_arrival = _fold({n: blocks[n] for n in reversed(names)})
    assert forward == reversed_arrival

    # And the sharding itself: whatever K is, the shares partition the rows
    # exactly once each — the property `rows[i::K]` has and a hand-rolled
    # chunker does not.
    rows = list(range(len(names)))
    for workers in (1, 2, 3, 5, 8):
        seen: list = []
        for index, count in boot_key.shares(len(rows), workers):
            seen.extend(rows[index::count])
        assert sorted(seen) == rows


def test_sharding_is_round_robin_not_contiguous() -> None:
    """A family's rows are grouped by TARGET in the declaration, and its
    denoiser rows cost an order of magnitude more than its VAE rows.
    Contiguous chunks hand one child the whole denoiser group and the pool's
    wall becomes that child's wall."""
    rows = [f"e{i}" for i in range(8)]
    shares = boot_key.shares(8, 4)
    assert shares == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert rows[0::4] == ["e0", "e4"]
    assert rows[3::4] == ["e3", "e7"]


def test_no_child_is_spawned_with_an_empty_share() -> None:
    """K is capped at the class count upstream, but the share list must not
    depend on that: a child with nothing to trace is a process spawned to
    report a refusal."""
    assert boot_key.shares(2, 5) == [(0, 5), (1, 5)]


# ---------------------------------------------------------------------------
# The width is DERIVED, never a constant and never an env
# ---------------------------------------------------------------------------


def test_width_derives_from_the_pods_own_cgroup_quota(monkeypatch) -> None:
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 8.0)
    width = boot_key.trace_workers(36)
    assert width.workers == 7          # 8 cores minus one serving-headroom core
    assert "cpu.max=8" in width.reason


def test_width_never_exceeds_the_class_count(monkeypatch) -> None:
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 32.0)
    assert boot_key.trace_workers(3).workers == 3


def test_width_is_at_least_one_on_a_one_core_pod(monkeypatch) -> None:
    """A pod narrower than the headroom still traces — serially, which is the
    honest answer, and never zero-wide."""
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 1.0)
    assert boot_key.trace_workers(36).workers == 1


def test_an_uncapped_cgroup_falls_back_to_the_measured_core_count(
    monkeypatch,
) -> None:
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: None)
    monkeypatch.setattr(boot_key, "effective_cpu_count", lambda: 4)
    width = boot_key.trace_workers(36)
    assert width.workers == 3
    assert "uncapped cgroup" in width.reason


def test_no_env_decides_the_width(monkeypatch) -> None:
    """§1.17: an env may carry a value, never a decision. There is no env in
    this module's width path at all — proven by moving every plausible knob
    and asserting the answer does not move."""
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 4.0)
    before = boot_key.trace_workers(36).workers
    for name in ("GEN_WORKER_TRACE_WORKERS", "GEN_WORKER_BOOT_KEY_K",
                 "OMP_NUM_THREADS", "GEN_WORKER_ENTRY_WORKERS"):
        monkeypatch.setenv(name, "1")
    assert boot_key.trace_workers(36).workers == before


# ---------------------------------------------------------------------------
# The memo: TCG graph-axis outputs only
# ---------------------------------------------------------------------------


def test_the_memo_never_holds_the_folded_key(tmp_path: Path) -> None:
    """The memo may hold the GRAPH half and must not hold the key.

    An sm or toolchain that changed has to move the key on the very NEXT boot;
    a memoized key would
    answer with the previous pod's. Read off the FILE, not off the API, so a
    future field cannot smuggle one in.
    """
    digest = "closure0123"
    blocks = {"a": _block(), "b": _block(dim=128)}
    entry_keys = _fold(blocks)
    assert boot_key.write_memo(tmp_path, digest, blocks)

    doc = json.loads((tmp_path / boot_key.MEMO_FILENAME).read_text())
    blob = json.dumps(doc)
    for key in entry_keys.values():
        assert key not in blob
    assert "cg-key-v1-" not in blob
    assert set(doc["closures"][digest]["blocks"]) == {"a", "b"}
    # And not the axes that must re-derive every boot, either.
    for axis in ("toolchain", "sm_89"):
        assert axis not in blob


def test_memo_rows_are_only_closed_tcg_class_hashes(tmp_path: Path) -> None:
    blocks = {"a": _block(), "b": _block(dim=128)}
    assert boot_key.class_hashes_of(blocks) == {
        name: row["class_hash"] for name, row in blocks.items()
    }
    assert boot_key.write_memo(tmp_path, "closure", blocks)
    assert boot_key.read_memo(tmp_path, "closure") == blocks


def test_the_memo_round_trips_and_a_foreign_closure_is_a_miss(
    tmp_path: Path,
) -> None:
    blocks = {"a": _block()}
    boot_key.write_memo(tmp_path, "closureA", blocks)
    assert boot_key.read_memo(tmp_path, "closureA") == blocks
    assert boot_key.read_memo(tmp_path, "closureB") == {}
    assert boot_key.read_memo(None, "closureA") == {}


def test_one_unreadable_block_invalidates_the_WHOLE_memo_entry(
    tmp_path: Path,
) -> None:
    """A partial class set is not a narrower key, it is a wrong one
 — so a memo that can only answer for some of its classes must
    answer for none."""
    boot_key.write_memo(tmp_path, "closureA", {"a": _block(), "b": _block()})
    path = tmp_path / boot_key.MEMO_FILENAME
    doc = json.loads(path.read_text())
    doc["closures"]["closureA"]["blocks"]["b"] = "{not json"
    path.write_text(json.dumps(doc))
    assert boot_key.read_memo(tmp_path, "closureA") == {}


def test_one_open_schema_block_invalidates_the_WHOLE_memo_entry(
    tmp_path: Path,
) -> None:
    boot_key.write_memo(tmp_path, "closureA", {"a": _block(), "b": _block()})
    path = tmp_path / boot_key.MEMO_FILENAME
    doc = json.loads(path.read_text())
    doc["closures"]["closureA"]["blocks"]["b"] = json.dumps({
        "class_hash": "0" * 16,
        "graph_witness": "1" * 16,
    })
    path.write_text(json.dumps(doc))
    assert boot_key.read_memo(tmp_path, "closureA") == {}


def test_a_version_bump_reads_as_absent_never_as_a_stale_hit(
    tmp_path: Path,
) -> None:
    boot_key.write_memo(tmp_path, "closureA", {"a": _block()})
    path = tmp_path / boot_key.MEMO_FILENAME
    doc = json.loads(path.read_text())
    doc["v"] = boot_key.MEMO_VERSION + 1
    path.write_text(json.dumps(doc))
    assert boot_key.read_memo(tmp_path, "closureA") == {}


def test_the_closure_digest_moves_on_code_and_on_declaration(
    monkeypatch,
) -> None:
    """What a per-class graph hash is a pure function OF — and nothing else.

    Deliberately NOT sm/toolchain/env_seal: they are key AXES that re-derive
    in milliseconds, and folding them in here would make the memo miss on
    facts whose whole point is that they are cheap to restate.
    """
    cfg = CompileSpec(
        family="tiny", targets=("unet",), shapes=((1024, 1024),),
        text_lens=(77,), guidance_scales=(7.5,))
    base = boot_key.closure_digest("tiny", cfg)

    wider = CompileSpec(
        family="tiny", targets=("unet",),
        shapes=((1024, 1024), (768, 768)),
        text_lens=(77,), guidance_scales=(7.5,))
    assert boot_key.closure_digest("tiny", wider) != base

    from gen_worker import compile_cache as cc

    monkeypatch.setattr(cc, "content_keys", lambda: {"sdk": "deadbeef"})
    assert boot_key.closure_digest("tiny", cfg) != base


@pytest.mark.parametrize("axis", ["sm", "toolchain"])
def test_the_closure_digest_ignores_the_re_derived_axes(axis: str) -> None:
    """Stated as a property of the recorded facts rather than by monkeypatching
    a probe: the digest's inputs are enumerated in one dict, and neither the
    sm nor the toolchain is among them."""
    # The digest is over a literal fact block; assert the axis names appear
    # nowhere in the source of that block.
    import inspect

    body = inspect.getsource(boot_key.closure_digest)
    marker = f'"{axis}"'
    assert marker not in body.split('"""')[2], (
        f"{axis} must not enter the memo key — it is a key AXIS that "
        f"re-derives every boot in milliseconds")


# ---------------------------------------------------------------------------
# The axes actually move when the facts move
# ---------------------------------------------------------------------------


def test_a_different_traced_graph_moves_the_key() -> None:
    a = _fold({"e": _block(fqns=["w.weight"])})
    b = _fold({"e": _block(fqns=["w.weight", "w.bias"])})
    assert a["e"] != b["e"]


# ---------------------------------------------------------------------------
# The child contract
# ---------------------------------------------------------------------------


def test_the_child_runs_this_parents_own_code(tmp_path: Path) -> None:
    """pgw#840's failure, at the second child seam: ``-m gen_worker.…`` means
    whatever ``gen_worker`` the child's import system resolves. The package
    root leads PYTHONPATH and the cwd is removed, and the digest is the
    backstop that PROVES it rather than assuming it."""
    env = boot_key.child_env({"PYTHONPATH": "/somewhere/else"})
    assert env["PYTHONPATH"].split(":")[0] == boot_key.PACKAGE_ROOT
    assert env["PYTHONSAFEPATH"] == "1"
    assert len(boot_key.CODE_DIGEST) == 16

    argv = boot_key.child_argv(tmp_path / "job.json")
    assert argv[1:3] == ["-m", boot_key.TRACE_CHILD_MODULE]


def test_a_child_that_produced_no_hashes_refuses_the_whole_derivation(
    tmp_path: Path, monkeypatch,
) -> None:
    """A partial class set is not a narrower key, it is a WRONG key: a cell's
    identity is its whole class set. So one dead child refuses
    the derivation, and the pod mints the ordinary way."""
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 2.0)
    monkeypatch.setattr(
        boot_key, "_run_children",
        lambda jobs, python="": [boot_key.TraceReport(
            ok=False, reason="structure_unsupported",
            detail="MicroEscapeDenoiser has no from_config")])
    with pytest.raises(boot_key.BootKeyUnavailable) as err:
        boot_key.derive(
            function="fn", modules=("m",), family="tiny",
            cfg=CompileSpec(family="tiny", targets=("unet",)),
            slots={}, declared_hint=2,
            work_root=tmp_path)
    assert err.value.reason == "structure_unsupported"
    assert "MicroEscapeDenoiser" in err.value.detail


def test_a_declaration_with_no_classes_refuses(tmp_path: Path) -> None:
    with pytest.raises(boot_key.BootKeyUnavailable) as err:
        boot_key.derive(
            function="fn", modules=("m",), family="tiny",
            cfg=CompileSpec(family="tiny"), slots={}, declared_hint=0,
            work_root=tmp_path)
    assert err.value.reason == "no_classes"


def test_a_child_running_drifted_source_refuses_rather_than_being_believed(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 2.0)
    canon = json.dumps(_block(), sort_keys=True, separators=(",", ":"))
    monkeypatch.setattr(
        boot_key, "_run_children",
        lambda jobs, python="": [boot_key.TraceReport(
            ok=True, blocks={"a": canon}, declared_classes=1,
            code_digest="beefbeefbeefbeef")])
    with pytest.raises(boot_key.BootKeyUnavailable) as err:
        boot_key.derive(
            function="fn", modules=("m",), family="tiny",
            cfg=CompileSpec(family="tiny", targets=("unet",)),
            slots={}, declared_hint=1,
            work_root=tmp_path)
    assert err.value.reason == "code_drift"


def test_a_derivation_that_returns_an_incomplete_class_set_refuses(
    tmp_path: Path, monkeypatch,
) -> None:
    """The children agree the declaration produces 2 classes and return 1."""
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 2.0)
    canon = json.dumps(_block(), sort_keys=True, separators=(",", ":"))
    monkeypatch.setattr(
        boot_key, "_run_children",
        lambda jobs, python="": [boot_key.TraceReport(
            ok=True, blocks={"a": canon}, declared_classes=2,
            code_digest=boot_key.CODE_DIGEST)])
    with pytest.raises(boot_key.BootKeyUnavailable) as err:
        boot_key.derive(
            function="fn", modules=("m",), family="tiny",
            cfg=CompileSpec(family="tiny", targets=("unet",)),
            slots={}, declared_hint=2,
            work_root=tmp_path)
    assert err.value.reason == "class_set_gap"
    assert "of the 2" in err.value.detail


def test_children_that_enumerated_different_class_sets_refuse(
    tmp_path: Path, monkeypatch,
) -> None:
    """Two children that composed different pipelines traced two different
    cells' graphs. The completeness proof is what catches it, and it consults
    no parent-side guess to do so."""
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 3.0)
    canon = json.dumps(_block(), sort_keys=True, separators=(",", ":"))
    other = json.dumps(_block(dim=128), sort_keys=True, separators=(",", ":"))
    monkeypatch.setattr(
        boot_key, "_run_children",
        lambda jobs, python="": [
            boot_key.TraceReport(
                ok=True, blocks={"a": canon}, declared_classes=2,
                code_digest=boot_key.CODE_DIGEST),
            boot_key.TraceReport(
                ok=True, blocks={"b": other}, declared_classes=3,
                code_digest=boot_key.CODE_DIGEST),
        ])
    with pytest.raises(boot_key.BootKeyUnavailable) as err:
        boot_key.derive(
            function="fn", modules=("m",), family="tiny",
            cfg=CompileSpec(family="tiny", targets=("unet",)),
            slots={}, declared_hint=2,
            work_root=tmp_path)
    assert err.value.reason == "class_set_disagreement"


def test_derive_end_to_end_with_stubbed_children_memoizes_and_hits(
    tmp_path: Path, monkeypatch,
) -> None:
    """The parent's whole loop — shard, run, fold, memo — with the CHILDREN
    stubbed and nothing else.

    Proves the three memo transitions AND the property that makes the memo
    worth having: **a hit spawns no child at all.** The second derive below
    would raise if it reached ``_run_children``, because the stub is replaced
    with one that fails.
    """
    monkeypatch.setattr(boot_key, "cpu_quota_cores", lambda: 4.0)
    blocks = {"a": _block(), "b": _block(dim=128)}
    canon = {k: json.dumps(v, sort_keys=True, separators=(",", ":"))
             for k, v in blocks.items()}

    def _children(jobs, python=""):
        return [boot_key.TraceReport(
            ok=True, blocks=canon, nodes={"a": 41, "b": 41},
            trace_ms={"a": 1200, "b": 1300}, declared_classes=2,
            code_digest=boot_key.CODE_DIGEST)]

    monkeypatch.setattr(boot_key, "_run_children", _children)
    kwargs: Dict[str, Any] = dict(
        function="fn", modules=("m",), family="tiny",
        cfg=CompileSpec(family="tiny", targets=("unet",)),
        slots={}, declared_hint=2,
        work_root=tmp_path, memo_dir=tmp_path)

    first = boot_key.derive(**kwargs)
    assert first.memo == "miss"
    assert first.traced == 2
    assert first.nodes == {"a": 41, "b": 41}
    assert len(first.keys) == 2
    assert all(k.startswith("cg-key-v1-") for k in first.keys)

    # A HIT MUST NOT TRACE. Any child spawn now is a hard failure.
    def _never(jobs, python=""):
        raise AssertionError(
            "a memo hit spawned a trace child — the memo path is supposed to "
            "be milliseconds, and pgw#1089 says so")

    monkeypatch.setattr(boot_key, "_run_children", _never)
    second = boot_key.derive(**kwargs)
    assert second.memo == "hit"
    assert second.keys == first.keys
    assert second.traced == 0
    assert second.workers == 0
    assert "no trace child" in second.width_reason

    # The VERIFY posture traces anyway and rules on what the memo held.
    monkeypatch.setattr(boot_key, "_run_children", _children)
    verified = boot_key.derive(**kwargs, trust_memo=False)
    assert verified.memo == "verified"
    assert verified.keys == first.keys

    # Poison it, and the FRESH trace must win and say so.
    closure = boot_key.closure_digest("tiny", kwargs["cfg"], function="fn")
    poisoned_blocks = {"a": _block(dim=999), "b": _block(dim=998)}
    boot_key.write_memo(tmp_path, closure, poisoned_blocks)
    caught = boot_key.derive(**kwargs, trust_memo=False)
    assert caught.memo == "invalidated"
    assert caught.keys == first.keys


# ---------------------------------------------------------------------------
# priced, and it decides nothing
# ---------------------------------------------------------------------------


def test_prop_economy_reports_and_decides_nothing() -> None:
    reports = [
        boot_key.TraceReport(ok=True, export_probe_ms=4000, prop_probe_ms=400),
        boot_key.TraceReport(ok=True, export_probe_ms=6000, prop_probe_ms=800),
    ]
    econ = boot_key.prop_economy(reports)
    assert econ["measured"] is True
    assert econ["ratio"] == pytest.approx(0.12, abs=1e-6)
    assert boot_key.prop_economy([])["measured"] is False
    # The width is not a function of the economy — instrument first.
    assert "prop" not in boot_key.trace_workers(36).reason
