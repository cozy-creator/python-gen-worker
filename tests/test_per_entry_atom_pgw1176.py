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
from gen_worker import cell_key

from harness import exported_cell as rig
from harness.exported_cell import declared  # noqa: F401 — fixture


TOOLCHAIN = {"torch": "abc123", "triton": "def456", "ptxas": "0f0f0f"}


def entry_meta(h: int, w: int, **over: Any) -> Dict[str, Any]:
    """One format-3 entry artifact's metadata, built the mint's own way."""
    meta = aot.entry_metadata(
        family=rig.FAMILY, precision="w8a8", cell_key="",
        name=rig.entry_name(h, w), entry=rig._entry(h, w),
        strict_export=True, lora_bucket=0, **over)
    meta.update(rig.RUNTIME)
    meta["toolchain"] = dict(TOOLCHAIN)
    meta["cell_key"] = cell_key.from_entry_metadata(meta).digest
    return meta


# --- 1. IDENTITY -----------------------------------------------------------


def test_widening_the_ladder_does_not_rekey_unchanged_classes() -> None:
    """The measured disease, closed. ``envelope_facts`` digests the UNION of
    the ladder across the bundle, so on master adding ONE aspect ratio moved
    the key of every class in the cell — 35 of sdxl's 36 trace identically.
    Per entry, the shape facts that affect tracing are inside ``class_hash``,
    so an author who widens the ladder adds NEW keys and moves none.
    """
    narrow = {name: entry_meta(h, w) for name, (h, w) in
              ((rig.entry_name(h, w), (h, w)) for h, w in rig.ROWS)}
    # The wider declaration: same two classes, plus a third.
    wide = {name: entry_meta(h, w) for name, (h, w) in
            ((rig.entry_name(h, w), (h, w))
             for h, w in rig.ROWS + ((32, 32),))}

    assert set(wide) - set(narrow) == {rig.entry_name(32, 32)}
    moved = [n for n in narrow if narrow[n]["cell_key"] != wide[n]["cell_key"]]
    assert not moved, f"widening the ladder re-keyed {moved!r}"
    # And the new class is genuinely new, not a relabel of an old one.
    assert wide[rig.entry_name(32, 32)]["cell_key"] not in {
        m["cell_key"] for m in narrow.values()}


def test_the_envelope_is_not_a_key_axis() -> None:
    """Stated as a refusal, not only as an absence: a caller still shipping
    the dropped axis must fail here rather than silently widening the key."""
    with pytest.raises(cell_key.CellKeyError) as exc:
        cell_key.from_axes({
            "graph": "a" * 16, "envelope": "b" * 16, "sm": "sm_89",
            "toolchain": "c" * 16})
    assert "envelope" in str(exc.value)


def test_the_manifest_digest_is_a_label_not_an_identity() -> None:
    """``combined_graph_hash``'s arithmetic survives, demoted. It is what the
    hub folds compile-health rows under — one row per
    ``(manifest_digest, sm, toolchain)`` — so it must be sm- and
    toolchain-FREE, or that tuple is degenerate."""
    hashes = ["ff" * 8, "aa" * 8]
    assert cell_key.manifest_digest(hashes) == \
        cell_key.manifest_digest(reversed(hashes))
    assert not cell_key.is_key(cell_key.manifest_digest(hashes)), (
        "a manifest digest must never have entry-key shape — nothing may "
        "resolve, download, verify or arm it")


def test_a_ck1_key_is_key_shaped_and_still_names_nothing() -> None:
    """The re-key, enforced where it is actually enforceable — the AXES.

    th#1897 SUPERSEDES the grammar-level reading this row used to carry
    (``assert not is_key("ck1-…")``). The compiled-graph key grammar is the
    shared contract with tensorhub's ``IsCompiledGraphKey`` and it is
    scheme-AGNOSTIC on purpose (th#1183): it refuses SHAPE, never scheme, so
    that a newer fleet's key stays addressable by an older hub and the two can
    ship in different windows. The corpus both repos vendor states it
    outright — ``ck1-<56 hex>`` is ``valid: true``, noted as "the scheme the 3
    purged micro-diffusion rows carried".

    So a ck1 token parses, and then names nothing: no artifact of that scheme
    survives the purge, and its digest was computed over axes this runtime
    cannot restate, so it misses at the comparison. The orphan fails there,
    which is the property this row was always about — the grammar was only
    ever a convenient place to state it, and it is the wrong place because it
    is not this repo's to decide alone.
    """
    assert cell_key.KEY_SCHEME == "cg-key-v1"
    # fence-symbol-exempt: `ck1` is the SUPERSEDED scheme and naming it is the
    # whole assertion — a sweep that renames this line deletes the invariant.
    assert cell_key.is_key("ck1-" + "0" * 56)
    assert cell_key.is_key("cg-key-v1-" + "0" * 56)
    # And it is not a key this runtime can ever mint, which is the miss.
    assert not cell_key.from_axes(
        {"graph": "0" * 16, "sm": "sm_100", "toolchain": "0" * 16}
    ).digest.startswith("ck1-")


# --- 2. ARTIFACT -----------------------------------------------------------


def test_one_artifact_carries_exactly_one_graph(tmp_path: Path) -> None:
    meta = entry_meta(*rig.ROWS[0])
    assert (meta[aot.COMPILED_GRAPH_FORMAT_KEY]
            == aot.COMPILED_GRAPH_FORMAT == 1)
    assert aot.verify_contract(meta) == ""
    entry = meta[cell_key.ENTRY_BLOCK_KEY]
    assert entry["name"] == rig.entry_name(*rig.ROWS[0])
    assert "entries" not in meta and "combined_graph_hash" not in meta

    work = tmp_path / "work"
    work.mkdir()
    (work / aot.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
    packed = aot.pack(work, tmp_path / f"{meta['cell_key']}.tar.gz", meta)
    back, block = aot._unpack(packed, tmp_path / "out")
    assert block["class_hash"] == entry["class_hash"]
    assert cell_key.from_entry_metadata(back).digest == meta["cell_key"]


def test_a_forged_key_stamp_is_refused_on_the_staged_bytes() -> None:
    meta = entry_meta(*rig.ROWS[0])
    meta["cell_key"] = "cg-key-v1-" + "0" * 56
    reason = aot.verify_contract(meta)
    assert "!=" in reason and "recorded facts describe" in reason


def test_a_format_2_cell_cannot_restate_a_per_entry_identity() -> None:
    """Why the ck1 corpus purge is hygiene rather than a correctness
    precondition: a 36-entry cell records an ``entries`` MAP, so the
    recomputation raises rather than matching."""
    legacy = {
        "format": 2, "kind": aot.ARTIFACT_KIND, **rig.RUNTIME,
        "family": rig.FAMILY, "cell_key": "cg-key-v1-" + "0" * 56,
        "entries": {rig.entry_name(h, w): rig._entry(h, w)
                    for h, w in rig.ROWS},
        "combined_graph_hash": "0" * 16,
        cell_key.EXPORT_ENVELOPE_KEY: {"shapes": [list(r) for r in rig.ROWS]},
        "toolchain": dict(TOOLCHAIN),
        "host_isa": {"machine": platform.machine(), "march": "",
                     "simdlen": 0, "level": ""},
    }
    with pytest.raises(cell_key.CellKeyError) as exc:
        cell_key.from_entry_metadata(legacy)
    assert "entry" in str(exc.value)


# --- 3. ARM ----------------------------------------------------------------


class _Unbindable(rig.ProbePackage):  # type: ignore[misc]
    def get_constant_fqns(self) -> List[str]:
        return ["weight", "a_constant_the_manifest_never_declared"]


def _arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    packages: Dict[str, Any], *, entries: List[Any],
    declared_names: Sequence[str] = (),
) -> Any:
    """Arm entries one artifact at a time, the way the loop does."""
    from gen_worker.models import provision

    monkeypatch.setattr(aot, "runtime_key", lambda: dict(rig.RUNTIME))
    monkeypatch.setattr(
        aot, "_load_package", lambda path, entry="model": packages[entry])
    monkeypatch.setattr(
        aot, "_entry_admission_drift", lambda *a, **k: None, raising=False)
    module = rig.ProbeDenoiser()
    pipeline = rig.ProbePipeline(module)
    outcomes = []
    for i, (h, w) in enumerate(entries):
        meta = entry_meta(h, w)
        work = tmp_path / f"work{i}"
        work.mkdir()
        (work / aot.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
        artifact = aot.pack(work, tmp_path / f"cell{i}.tar.gz", meta)
        outcomes.append(aot.enable(
            pipeline, rig.cell_cfg(rig.declaration()), tmp_path / "cache",
            artifact, declared=tuple(declared_names)))
    del provision  # the arm route is exercised through `enable` directly
    return pipeline, module, outcomes


def test_a_failing_entry_does_not_un_arm_its_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,  # noqa: F811
) -> None:
    """THE RED, closed. On master ``arm_entry`` bound every entry before
    any wrap — "a cell that cannot arm one of its graph classes arms none of
    them" — so one unbindable class cost the whole cell and the pod served
    fully eager. An entry is one graph, so it arms whole or not at all, and a
    sibling's failure is not its business.
    """
    good, bad = rig.ROWS[0], rig.ROWS[1]
    packages = {
        rig.entry_name(*good): rig.ProbePackage(),
        rig.entry_name(*bad): _Unbindable(),
    }
    pipeline, module, outcomes = _arm(
        tmp_path, monkeypatch, packages, entries=[good, bad])

    assert outcomes[0].armed, outcomes[0].detail
    assert not outcomes[1].armed
    assert outcomes[1].reason == "constants_constant_set_mismatch"
    armed = aot.armed_entries(pipeline)
    assert set(armed) == {rig.entry_name(*good)}, (
        f"the surviving class must still serve compiled; armed={armed}")
    assert aot.is_armed(pipeline)

    # And it genuinely SERVES: a call in the good class's shape runs compiled.
    out = module(torch.zeros(good), torch.tensor(1.0))
    assert out is not None
    assert packages[rig.entry_name(*good)].invocations == 1


def test_entries_accrete_into_one_registry_and_one_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,  # noqa: F811
) -> None:
    """Coverage accretes: a second arm joins the SAME live wrap, the same
    target pool and the same dispatch. There is no complete state to wait
    for, and a subset is a legitimate steady state (pgw#1177: ~0.75 GiB of
    device memory per resident container is what a pod buys by holding one)."""
    packages = {rig.entry_name(h, w): rig.ProbePackage() for h, w in rig.ROWS}
    pipeline, module, outcomes = _arm(
        tmp_path, monkeypatch, packages, entries=list(rig.ROWS))
    assert all(o.armed for o in outcomes)
    assert len(aot.armed_entries(pipeline)) == 2
    marker = getattr(pipeline, aot._MARKER_ATTR)
    assert list(marker["bound_constants"]["pools"]) == [rig.TARGET], (
        "two entries of one target must share ONE pool")
    for h, w in rig.ROWS:
        module(torch.zeros((h, w)), torch.tensor(1.0))
    assert aot.served_entry_calls(pipeline) == {
        rig.entry_name(h, w): 1 for h, w in rig.ROWS}


def test_the_pool_binds_by_reference_and_never_clones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,  # noqa: F811
) -> None:
    """§4.33 step 4, enforced. ``update_constant_buffer(user_managed=True)``
    makes no copy of its own, so the pool's ``.clone()`` was the ONLY copy in
    the system — one full duplicate of the target's weights held for the life
    of the arm (~5.1 GiB on sdxl's single ``unet`` target). Deleted; the
    contiguity normalisation it also did survives, per tensor.
    """
    packages = {rig.entry_name(*rig.ROWS[0]): rig.ProbePackage()}
    pipeline, module, _ = _arm(
        tmp_path, monkeypatch, packages, entries=[rig.ROWS[0]])
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,  # noqa: F811
) -> None:
    """§4.31 per entry. The old wrapper set ``state['failed']`` for the whole
    target on any artifact error, so one bad graph class took every sibling
    with it — the same all-or-nothing shape one layer down from the arm."""
    good, bad = rig.ROWS[0], rig.ROWS[1]
    packages = {
        rig.entry_name(*good): rig.ProbePackage(),
        rig.entry_name(*bad): rig.ProbePackage(raises="kernel exploded"),
    }
    pipeline, module, outcomes = _arm(
        tmp_path, monkeypatch, packages, entries=[good, bad])
    assert all(o.armed for o in outcomes)

    module(torch.zeros(bad), torch.tensor(1.0))  # served EAGER, correctly
    states = aot.entry_states(pipeline)
    assert states[rig.entry_name(*bad)]["state"] == "de_armed"
    assert states[rig.entry_name(*good)]["state"] == "armed"
    assert aot.is_armed(pipeline), (
        "one de-armed class must not stop the pipeline serving compiled")

    module(torch.zeros(good), torch.tensor(1.0))
    assert packages[rig.entry_name(*good)].invocations == 1


def test_a_de_armed_class_is_not_re_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,  # noqa: F811
) -> None:
    """§4.31's de-arm is STICKY for the boot. Per entry that has to be
    enforced where the registry is, or a background compile would cheerfully
    re-arm the class that just failed."""
    packages = {rig.entry_name(*rig.ROWS[0]): rig.ProbePackage()}
    pipeline, _module, _ = _arm(
        tmp_path, monkeypatch, packages, entries=[rig.ROWS[0]])
    assert aot.disarm_entry(pipeline, rig.entry_name(*rig.ROWS[0]), "parity")
    assert not aot.armed_entries(pipeline)
    marker = getattr(pipeline, aot._MARKER_ATTR)
    dispatch = aot._dispatch_for(marker, rig.TARGET)
    assert dispatch is not None
    with pytest.raises(Exception) as exc:
        dispatch.add(rig.entry_name(*rig.ROWS[0]), cast(Any, object()))
    assert "de-armed" in str(exc.value)


def test_a_declared_but_uncompiled_class_reports_pending_not_a_shape_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: Any,  # noqa: F811
) -> None:
    """Under accretion the commonest reason nothing admits a call is that its
    class has not been compiled YET. Reporting that as a shape gap would ask
    the growth path to add a class the declaration already carries."""
    packages = {rig.entry_name(*rig.ROWS[0]): rig.ProbePackage()}
    names = [rig.entry_name(h, w) for h, w in rig.ROWS]
    pipeline, module, _ = _arm(
        tmp_path, monkeypatch, packages, entries=[rig.ROWS[0]],
        declared_names=names)
    marker = getattr(pipeline, aot._MARKER_ATTR)
    dispatch = aot._dispatch_for(marker, rig.TARGET)
    assert dispatch is not None
    assert dispatch.pending == (rig.entry_name(*rig.ROWS[1]),)

    gaps: List[Any] = []
    from gen_worker import shape_growth
    monkeypatch.setattr(
        shape_growth, "report_and_submit", lambda gap: gaps.append(gap))
    module(torch.zeros(rig.ROWS[1]), torch.tensor(1.0))  # served eager
    assert not gaps, f"a pending class must not be reported as a shape gap: {gaps}"
