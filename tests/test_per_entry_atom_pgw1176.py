"""pgw#1176 — THE ATOM IS ONE GRAPH CLASS.

Three assertions, one per half of the change. Every one of them was proven
RED first, on unmodified ``origin/master`` @ ``4dfdcd60``, with the messages
quoted below — the fix lives in the modules under test, so grafting the
assertions onto the old tree is what makes the red meaningful.

1. IDENTITY. ``2 of 2 byte-identical classes re-keyed when one shape row was
   added: ['denoiser/h=16,w=16', 'denoiser/h=8,w=8']
   (ck1-c4c134dbc3a958741a33f7d4a608728ef91d98ce635f141ece289e65 ->
    ck1-48512ea33f97c8490fd5ebdfdf9d036f2d2ceffeb753dee7010e18ab)``
2. ARTIFACT. ``the artifact format cannot express ONE graph: no
   combined_graph_hash stamped``
3. ARM. ``one unarmable class un-armed every sibling:
   outcome=constants_constant_set_mismatch ...; served={}``
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import pytest
import torch

from gen_worker import aot_serve as aot
from gen_worker import compiled_graph_key

from harness import exported_compiled_graph as rig
from harness.exported_compiled_graph import declared  # noqa: F401 — fixture


TOOLCHAIN = {"torch": "abc123", "triton": "def456", "ptxas": "0f0f0f"}


def compiled_graph_meta(h: int, w: int, **over: Any) -> Dict[str, Any]:
    """One format-3 compiled graph artifact's metadata, built the mint's own way."""
    meta = aot.compiled_graph_metadata(
        family=rig.FAMILY, precision="w8a8", compiled_graph_key="",
        name=rig.compiled_graph_name(h, w), compiled_graph=rig._compiled_graph(h, w),
        strict_export=True, lora_bucket=0, **over)
    meta.update(rig.RUNTIME)
    meta["toolchain"] = dict(TOOLCHAIN)
    meta["compiled_graph_key"] = compiled_graph_key.from_compiled_graph_metadata(meta).digest
    return meta


# --- 1. IDENTITY -----------------------------------------------------------


def test_widening_the_ladder_does_not_rekey_unchanged_classes() -> None:
    """The measured disease, closed. ``envelope_facts`` digests the UNION of
    the ladder across the bundle, so on master adding ONE aspect ratio moved
    the key of every class in the compiled graph — 35 of sdxl's 36 trace identically.
    Per compiled graph, the shape facts that affect tracing are inside ``class_hash``,
    so an author who widens the ladder adds NEW keys and moves none.
    """
    narrow = {name: compiled_graph_meta(h, w) for name, (h, w) in
              ((rig.compiled_graph_name(h, w), (h, w)) for h, w in rig.ROWS)}
    # The wider declaration: same two classes, plus a third.
    wide = {name: compiled_graph_meta(h, w) for name, (h, w) in
            ((rig.compiled_graph_name(h, w), (h, w))
             for h, w in rig.ROWS + ((32, 32),))}

    assert set(wide) - set(narrow) == {rig.compiled_graph_name(32, 32)}
    moved = [n for n in narrow if narrow[n]["compiled_graph_key"] != wide[n]["compiled_graph_key"]]
    assert not moved, f"widening the ladder re-keyed {moved!r}"
    # And the new class is genuinely new, not a relabel of an old one.
    assert wide[rig.compiled_graph_name(32, 32)]["compiled_graph_key"] not in {
        m["compiled_graph_key"] for m in narrow.values()}


def test_the_envelope_is_not_a_key_axis() -> None:
    """Stated as a refusal, not only as an absence: a caller still shipping
    the dropped axis must fail here rather than silently widening the key."""
    with pytest.raises(compiled_graph_key.CompiledGraphKeyError) as exc:
        compiled_graph_key.from_axes({
            "graph": "a" * 16, "envelope": "b" * 16, "sm": "sm_89",
            "toolchain": "c" * 16})
    assert "envelope" in str(exc.value)


def test_the_manifest_digest_is_a_label_not_an_identity() -> None:
    """``combined_graph_hash``'s arithmetic survives, demoted. It is what the
    hub folds compile-health rows under — one row per
    ``(manifest_digest, sm, toolchain)`` — so it must be sm- and
    toolchain-FREE, or that tuple is degenerate."""
    hashes = ["ff" * 8, "aa" * 8]
    assert compiled_graph_key.manifest_digest(hashes) == \
        compiled_graph_key.manifest_digest(reversed(hashes))
    assert not compiled_graph_key.is_key(compiled_graph_key.manifest_digest(hashes)), (
        "a manifest digest must never have compiled-graph-key shape — nothing may "
        "resolve, download, verify or arm it")


def test_a_ck1_key_is_not_an_compiled_graph_key() -> None:
    """The re-key, enforced at the grammar. A ck1 key names a 36-compiled graph
    all-or-nothing compiled graph, which this runtime cannot arm at all; admitting it
    would let a compiled graph ref reach a per-compiled graph path and fail late.

    THE FIFTH SWEEP ERROR, and it landed in the atom's own proof: a blanket
    ``ck1-`` -> ``ek1-`` fixture sweep rewrote the REFUSAL line here, leaving
    a contradictory pair one character apart —
    ``assert not is_key("ek1-…")`` beside ``assert is_key("ek1-…")``. The row
    went red, which is the only reason it surfaced, and while it was red the
    ck1-refusal invariant had NO passing guard in this file at all.

    It is the exact class this branch's handover brief documents: **a
    mechanical sweep neuters the file whose purpose is to be an exception**,
    and the exception here is the one line that must keep naming the OLD
    scheme. Fixed, and annotated so the next sweep leaves it alone.
    """
    assert compiled_graph_key.KEY_SCHEME == "ek1"
    # fence-symbol-exempt: `ck1` is the SUPERSEDED scheme and naming it is the
    # whole assertion — a sweep that renames this line deletes the invariant.
    assert not compiled_graph_key.is_key("ck1-" + "0" * 56)
    assert compiled_graph_key.is_key("ek1-" + "0" * 56)
    # The refusal is about the PREFIX, not the length: a well-formed ck1 key
    # of exactly the right shape is still refused, which is what makes an
    # orphaned ref fail at the comparison rather than late.
    assert len("ck1-" + "0" * 56) == len("ek1-" + "0" * 56)


# --- 2. ARTIFACT -----------------------------------------------------------


def test_one_artifact_carries_exactly_one_graph(tmp_path: Path) -> None:
    meta = compiled_graph_meta(*rig.ROWS[0])
    assert meta["format"] == 3
    assert aot.verify_contract(meta) == ""
    compiled_graph = meta[compiled_graph_key.COMPILED_GRAPH_BLOCK_KEY]
    assert compiled_graph["name"] == rig.compiled_graph_name(*rig.ROWS[0])
    assert "compiled_graphs" not in meta and "combined_graph_hash" not in meta

    work = tmp_path / "work"
    work.mkdir()
    (work / aot.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
    packed = aot.pack(work, tmp_path / f"{meta['compiled_graph_key']}.tar.gz", meta)
    back, block = aot._unpack(packed, tmp_path / "out")
    assert block["class_hash"] == compiled_graph["class_hash"]
    assert compiled_graph_key.from_compiled_graph_metadata(back).digest == meta["compiled_graph_key"]


def test_a_forged_key_stamp_is_refused_on_the_staged_bytes() -> None:
    meta = compiled_graph_meta(*rig.ROWS[0])
    meta["compiled_graph_key"] = "ek1-" + "0" * 56
    reason = aot.verify_contract(meta)
    assert "!=" in reason and "recorded facts describe" in reason


def test_a_format_2_compiled_graph_cannot_restate_a_per_compiled_graph_identity() -> None:
    """Why the ck1 corpus purge is hygiene rather than a correctness
    precondition: a 36-compiled graph compiled graph records an ``compiled graphs`` MAP, so the
    recomputation raises rather than matching."""
    legacy = {
        "format": 2, "kind": aot.ARTIFACT_KIND, **rig.RUNTIME,
        "family": rig.FAMILY, "compiled_graph_key": "ek1-" + "0" * 56,
        "compiled_graphs": {rig.compiled_graph_name(h, w): rig._compiled_graph(h, w)
                    for h, w in rig.ROWS},
        "combined_graph_hash": "0" * 16,
        compiled_graph_key.EXPORT_ENVELOPE_KEY: {"shapes": [list(r) for r in rig.ROWS]},
        "toolchain": dict(TOOLCHAIN),
        "host_isa": {"machine": platform.machine(), "march": "",
                     "simdlen": 0, "level": ""},
    }
    with pytest.raises(compiled_graph_key.CompiledGraphKeyError) as exc:
        compiled_graph_key.from_compiled_graph_metadata(legacy)
    assert "compiled_graph" in str(exc.value)


# --- 3. ARM ----------------------------------------------------------------


class _Unbindable(rig.ProbePackage):  # type: ignore[misc]
    def get_constant_fqns(self) -> List[str]:
        return ["weight", "a_constant_the_manifest_never_declared"]


def _arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    packages: Dict[str, Any], *, compiled_graphs: List[Any],
    declared_names: Sequence[str] = (),
) -> Any:
    """Arm compiled graphs one artifact at a time, the way the loop does."""
    from gen_worker.models import provision

    monkeypatch.setattr(aot, "runtime_key", lambda: dict(rig.RUNTIME))
    monkeypatch.setattr(aot, "_compiled_graph_admission_drift", lambda *a, **k: None)
    monkeypatch.setattr(
        aot, "_load_package", lambda path, compiled_graph="model": packages[compiled_graph])
    module = rig.ProbeDenoiser()
    pipeline = rig.ProbePipeline(module)
    outcomes = []
    for i, (h, w) in enumerate(compiled_graphs):
        meta = compiled_graph_meta(h, w)
        work = tmp_path / f"work{i}"
        work.mkdir()
        (work / aot.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
        artifact = aot.pack(work, tmp_path / f"compiled_graph{i}.tar.gz", meta)
        outcomes.append(aot.enable(
            pipeline, rig.compiled_graph_cfg(rig.declaration()), tmp_path / "cache",
            artifact, declared=tuple(declared_names)))
    del provision  # the arm route is exercised through `enable` directly
    return pipeline, module, outcomes


def test_a_failing_compiled_graph_does_not_un_arm_its_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,
) -> None:
    """THE RED, closed. On master ``arm_compiled_graph`` bound every compiled graph before
    any wrap — "a compiled graph that cannot arm one of its graph classes arms none of
    them" — so one unbindable class cost the whole compiled graph and the pod served
    fully eager. An compiled graph is one graph, so it arms whole or not at all, and a
    sibling's failure is not its business.
    """
    good, bad = rig.ROWS[0], rig.ROWS[1]
    packages = {
        rig.compiled_graph_name(*good): rig.ProbePackage(),
        rig.compiled_graph_name(*bad): _Unbindable(),
    }
    pipeline, module, outcomes = _arm(
        tmp_path, monkeypatch, packages, compiled_graphs=[good, bad])

    assert outcomes[0].armed, outcomes[0].detail
    assert not outcomes[1].armed
    assert outcomes[1].reason == "constants_constant_set_mismatch"
    armed = aot.armed_compiled_graphs(pipeline)
    assert set(armed) == {rig.compiled_graph_name(*good)}, (
        f"the surviving class must still serve compiled; armed={armed}")
    assert aot.is_armed(pipeline)

    # And it genuinely SERVES: a call in the good class's shape runs compiled.
    out = module(torch.zeros(good), torch.tensor(1.0))
    assert out is not None
    assert packages[rig.compiled_graph_name(*good)].invocations == 1


def test_compiled_graphs_accrete_into_one_registry_and_one_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,
) -> None:
    """Coverage accretes: a second arm joins the SAME live wrap, the same
    target pool and the same dispatch. There is no complete state to wait
    for, and a subset is a legitimate steady state (pgw#1177: ~0.75 GiB of
    device memory per resident container is what a pod buys by holding one)."""
    packages = {rig.compiled_graph_name(h, w): rig.ProbePackage() for h, w in rig.ROWS}
    pipeline, module, outcomes = _arm(
        tmp_path, monkeypatch, packages, compiled_graphs=list(rig.ROWS))
    assert all(o.armed for o in outcomes)
    assert len(aot.armed_compiled_graphs(pipeline)) == 2
    marker = getattr(pipeline, aot._MARKER_ATTR)
    assert list(marker["bound_constants"]["pools"]) == [rig.TARGET], (
        "two compiled_graphs of one target must share ONE pool")
    for h, w in rig.ROWS:
        module(torch.zeros((h, w)), torch.tensor(1.0))
    assert aot.served_compiled_graph_calls(pipeline) == {
        rig.compiled_graph_name(h, w): 1 for h, w in rig.ROWS}


def test_the_pool_binds_by_reference_and_never_clones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,
) -> None:
    """§4.33 step 4, enforced. ``update_constant_buffer(user_managed=True)``
    makes no copy of its own, so the pool's ``.clone()`` was the ONLY copy in
    the system — one full duplicate of the target's weights held for the life
    of the arm (~5.1 GiB on sdxl's single ``unet`` target). Deleted; the
    contiguity normalisation it also did survives, per tensor.
    """
    packages = {rig.compiled_graph_name(*rig.ROWS[0]): rig.ProbePackage()}
    pipeline, module, _ = _arm(
        tmp_path, monkeypatch, packages, compiled_graphs=[rig.ROWS[0]])
    pool = getattr(pipeline, aot._MARKER_ATTR)["bound_constants"]["pools"]
    bound = pool[rig.TARGET]["weight"]
    assert bound.data_ptr() == module.weight.data_ptr(), (
        "the pool must hold the RESIDENT tensor, not a device duplicate")


def test_a_noncontiguous_resident_is_made_contiguous_not_left_dangling() -> None:
    """The one thing the deleted clone also did. An AOTI container takes a raw
    pointer, so a non-contiguous resident cannot be bound by reference — that
    exception is copied per tensor, never by cloning the whole pool."""
    view = torch.arange(16, dtype=torch.float32).reshape(4, 4).t()
    assert not view.is_contiguous()
    spec = aot.ConstantSpec(
        fqn="w", source=aot.SOURCE_STATE_DICT, dtype="float32", shape=(4, 4))
    pool = aot.target_constant_pool([[spec]], {"w": view})
    assert pool["w"].is_contiguous()
    assert torch.equal(pool["w"], view)


def test_a_serve_failure_de_arms_ONE_class_and_the_siblings_keep_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,
) -> None:
    """§4.31 per compiled graph. The old wrapper set ``state['failed']`` for the whole
    target on any artifact error, so one bad graph class took every sibling
    with it — the same all-or-nothing shape one layer down from the arm."""
    good, bad = rig.ROWS[0], rig.ROWS[1]
    packages = {
        rig.compiled_graph_name(*good): rig.ProbePackage(),
        rig.compiled_graph_name(*bad): rig.ProbePackage(raises="kernel exploded"),
    }
    pipeline, module, outcomes = _arm(
        tmp_path, monkeypatch, packages, compiled_graphs=[good, bad])
    assert all(o.armed for o in outcomes)

    module(torch.zeros(bad), torch.tensor(1.0))  # served EAGER, correctly
    states = aot.compiled_graph_states(pipeline)
    assert states[rig.compiled_graph_name(*bad)]["state"] == "de_armed"
    assert states[rig.compiled_graph_name(*good)]["state"] == "armed"
    assert aot.is_armed(pipeline), (
        "one de-armed class must not stop the pipeline serving compiled")

    module(torch.zeros(good), torch.tensor(1.0))
    assert packages[rig.compiled_graph_name(*good)].invocations == 1


def test_a_de_armed_class_is_not_re_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,
) -> None:
    """§4.31's de-arm is STICKY for the boot. Per compiled graph that has to be
    enforced where the registry is, or a background compile would cheerfully
    re-arm the class that just failed."""
    packages = {rig.compiled_graph_name(*rig.ROWS[0]): rig.ProbePackage()}
    pipeline, _module, _ = _arm(
        tmp_path, monkeypatch, packages, compiled_graphs=[rig.ROWS[0]])
    assert aot.disarm_compiled_graph(pipeline, rig.compiled_graph_name(*rig.ROWS[0]), "parity")
    assert not aot.armed_compiled_graphs(pipeline)
    marker = getattr(pipeline, aot._MARKER_ATTR)
    dispatch = aot._dispatch_for(marker, rig.TARGET)
    assert dispatch is not None
    with pytest.raises(Exception) as exc:
        dispatch.add(rig.compiled_graph_name(*rig.ROWS[0]), cast(Any, object()))
    assert "de-armed" in str(exc.value)


def test_a_declared_but_uncompiled_class_reports_pending_not_a_shape_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,
) -> None:
    """Under accretion the commonest reason nothing admits a call is that its
    class has not been compiled YET. Reporting that as a shape gap would ask
    the growth path to add a class the declaration already carries."""
    packages = {rig.compiled_graph_name(*rig.ROWS[0]): rig.ProbePackage()}
    names = [rig.compiled_graph_name(h, w) for h, w in rig.ROWS]
    pipeline, module, _ = _arm(
        tmp_path, monkeypatch, packages, compiled_graphs=[rig.ROWS[0]],
        declared_names=names)
    marker = getattr(pipeline, aot._MARKER_ATTR)
    dispatch = aot._dispatch_for(marker, rig.TARGET)
    assert dispatch is not None
    assert dispatch.pending == (rig.compiled_graph_name(*rig.ROWS[1]),)

    gaps: List[Any] = []
    from gen_worker import shape_growth
    monkeypatch.setattr(
        shape_growth, "report_and_submit", lambda gap: gaps.append(gap))
    module(torch.zeros(rig.ROWS[1]), torch.tensor(1.0))  # served eager
    assert not gaps, f"a pending class must not be reported as a shape gap: {gaps}"
