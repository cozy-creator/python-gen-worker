"""pgw#917's ingress merge and its REFUSAL half, on the sharded path.

The gate has always existed — ``aot_mint.canonicalize_dispatch_classes`` —
and it can only be asked by a process holding the WHOLE declaration's
``ExportedProgram``s. th#1834 Phase 3 abolished that process: a compile child
holds ONE SHARE, and the serving parent that supervises the shares may not
trace at all (the th#1299 fence). So the sharded path lost the gate.

**The loss is SILENT, and that is worse than the duplicate-key 409 an earlier
reading predicted.** Measured against the code: a mergeable pair keys APART,
because ``aot_serve.class_hash`` folds ``class_dims`` and ``class_dims`` is
the one axis such a pair differs on. Both rows compile, both publish, both
arm, and ``EntryDispatch.select`` answers ``entry_ambiguous`` on every call
they carry — 100 % eager on those coordinates, the exact 4,200-refusal defect
pgw#917 was filed to fix. The parent-side dedupe by KEY that landed with the
keystone is a different, narrower invariant and never sees this pair.

:func:`aot_mint.canonicalize_packed_classes` is the same gate at the only seam
a supervisor can reach: the packed envelope. Both halves are asserted here —
merge when a colliding cluster differs only on the class-row coordinate,
REFUSE naming the axes when it does not.

The area-preserving arithmetic is verbatim sdxl: 112x144 = 144x112 = 16,128,
and a ``BasicTransformerBlock`` never sees ``H_lat`` and ``W_lat``, only the
flattened ``(B, H_lat*W_lat, C)``. So the two blocks below carry the SAME
ingress contract and DIFFERENT ``class_dims``, which is the whole shape of the
defect.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gen_worker import aot_mint, aot_serve


def _block(
    *, name: str, dims: List[List[Any]], witness: str = "w" * 16,
    seq: int = 16128, target: str = "unet", arm: bool = True,
) -> Dict[str, Any]:
    """One packed entry block, in the shape ``aot_mint.keying_block`` writes
    and ``aot_serve.stamp_entry`` stamps."""
    block: Dict[str, Any] = {
        "name": name,
        "target": target,
        "fork": [[aot_mint.ADAPTER_FORK, arm]],
        "class_dims": dims,
        "inputs": [
            {"name": "hidden", "position": 0, "dtype": "float16",
             "shape": [2, seq, 320], "path": []},
        ],
        "symbols": {},
        "constants": [],
        "graph": {"v": 3, "constant_fqns": [], "lifted_inputs": [],
                  "pytree": {"user_inputs": ["hidden"]},
                  "specialization": {}},
        "graph_witness": witness,
    }
    return aot_serve.stamp_entry(name, block, strict=True, lora_bucket=0)


def _meta(**over: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "precision": "fp16", "lora_bucket": 0, "strict_export": True,
        "source_digest": "blake3:src",
    }
    meta.update(over)
    return meta


def test_two_aspect_rows_the_dispatch_cannot_TELL_APART_become_ONE_entry(
) -> None:
    """The merge half. Two declared rows over one ingress contract are one
    logical class: one entry survives, the other is recorded as its alias.

    They key APART — that is the point, and it is asserted here so nobody
    re-derives the by-key dedupe as the fix for this."""
    a = _block(name="unet.denoise@112x144", dims=[["H_lat", 112], ["W_lat", 144]])
    b = _block(name="unet.denoise@144x112", dims=[["H_lat", 144], ["W_lat", 112]])
    assert a["class_hash"] != b["class_hash"], (
        "a mergeable pair keying IDENTICALLY would be caught by the by-key "
        "dedupe and this gate would be unnecessary — the premise has moved")

    aliases = aot_mint.canonicalize_packed_classes(
        {a["name"]: a, b["name"]: b},
        {a["name"]: _meta(), b["name"]: _meta()})

    assert aliases == {"unet.denoise@112x144": ["unet.denoise@144x112"]}, (
        "both rows survived, so both publish, both arm, and every call they "
        "carry serves EAGER on `entry_ambiguous` — pgw#917's original defect, "
        "returned silently through the sharded path")


def test_a_cluster_that_is_NOT_one_class_REFUSES_and_names_the_axis() -> None:
    """The refusal half, which the sharded path lost with the merge.

    A cluster that collides at ingress while differing on a named identity
    axis is not one artifact; publishing both is how a declaration ships a
    class the dispatch can never select. The refusal names the axis, which a
    bare "these two clash" never could.
    """
    a = _block(name="unet.denoise@112x144", dims=[["H_lat", 112], ["W_lat", 144]])
    b = _block(name="unet.denoise@144x112", dims=[["H_lat", 144], ["W_lat", 112]],
               witness="DIFFERENT-BODY!")

    with pytest.raises(aot_mint.MintRefused) as exc:
        aot_mint.canonicalize_packed_classes(
            {a["name"]: a, b["name"]: b},
            {a["name"]: _meta(), b["name"]: _meta()})
    assert "graph" in str(exc.value), (
        "a refusal that does not name the differing axis sends the author "
        "looking at the wrong half of the declaration")
    assert "entry_ambiguous" in str(exc.value)


def test_a_metadata_axis_the_entry_block_does_NOT_carry_still_refuses() -> None:
    """`_class_identity` folds four axes that live on the ARTIFACT rather than
    in the entry block — precision, lora_bucket, strict, source_digest — and a
    reconstruction that dropped them would MERGE two artifacts that are not the
    same thing. Asserted with the one an adopting pod cannot recover from."""
    a = _block(name="unet.denoise@112x144", dims=[["H_lat", 112], ["W_lat", 144]])
    b = _block(name="unet.denoise@144x112", dims=[["H_lat", 144], ["W_lat", 112]])

    with pytest.raises(aot_mint.MintRefused) as exc:
        aot_mint.canonicalize_packed_classes(
            {a["name"]: a, b["name"]: b},
            {a["name"]: _meta(precision="fp16"),
             b["name"]: _meta(precision="fp8")})
    assert "precision" in str(exc.value)


def test_rows_on_DIFFERENT_targets_are_never_compared() -> None:
    """Grouped exactly the way the serve path groups — target, adapter arm —
    because those are the axes dispatch resolves BEFORE ingress. Two entries
    on different targets cannot collide however alike their contracts look."""
    a = _block(name="unet.denoise", dims=[["H_lat", 112]], target="unet")
    b = _block(name="vae.decode", dims=[["H_lat", 112]], target="vae.decode")
    assert aot_mint.canonicalize_packed_classes(
        {a["name"]: a, b["name"]: b},
        {a["name"]: _meta(), b["name"]: _meta()}) == {}


def test_the_two_adapter_ARMS_of_one_class_are_never_merged() -> None:
    """pgw#790's branchless class declares the NEGATIVE half of its contract,
    so the two arms do not mutually admit — and they must not, or the attach
    lane serves eager."""
    a = _block(name="unet.denoise+lora", dims=[["H_lat", 112]], arm=True)
    b = _block(name="unet.denoise", dims=[["H_lat", 112]], arm=False)
    assert aot_mint.canonicalize_packed_classes(
        {a["name"]: a, b["name"]: b},
        {a["name"]: _meta(), b["name"]: _meta()}) == {}


def test_the_coverage_LABEL_counts_survivors_and_not_absorbed_rows() -> None:
    """Owed item (d), re-fired on the merge.

    A compile child stamps its artifact's ``manifest_digest`` EMPTY — it holds
    ONE SHARE and cannot state a declaration-wide coverage label — and the
    parent folds the real one across every share. An absorbed class
    contributes the same ``class_hash`` its survivor already contributes, so
    folding it as well would make the label depend on how the declaration
    happened to be sharded, which is exactly what a coverage label may not do.
    """
    a = _block(name="unet.denoise@112x144", dims=[["H_lat", 112], ["W_lat", 144]])
    b = _block(name="unet.denoise@144x112", dims=[["H_lat", 144], ["W_lat", 112]])
    spec = aot_mint.ExportSpec(
        family="sdxl", target="unet", strict=True, lora_bucket=0)

    survivors_only = aot_mint.class_manifest({a["name"]: a}, spec)
    both = aot_mint.class_manifest({a["name"]: a, b["name"]: b}, spec)
    assert survivors_only != both

    held = [
        aot_mint.MintedArtifact(
            key="cg-key-v1-" + "a" * 56, entry=a["name"],
            artifact=aot_mint.Path("/dev/null"),
            metadata={"entry": a, **_meta()}),
        aot_mint.MintedArtifact(
            key="cg-key-v1-" + "b" * 56, entry=b["name"],
            artifact=aot_mint.Path("/dev/null"),
            metadata={"entry": b, **_meta()}),
    ]
    result = aot_mint.fold_held_graph_classes(held, spec=spec)
    assert len(result.entries) == 1
    assert result.manifest == survivors_only, (
        "the coverage label counted an absorbed class twice — the label would "
        "then depend on the sharding rather than on the declaration")
    assert result.entries[0].metadata["manifest_digest"] == survivors_only, (
        "the parent must stamp the folded label on the result; the bytes "
        "inside each artifact keep the empty stamp its child honestly wrote")
