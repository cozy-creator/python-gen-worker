"""pgw#1277: TCG owns compiled-graph identity, and the worker reads the block
TCG actually writes.

THE DEFECT THIS FILE WAS WRITTEN TO CATCH. The worker's duplicate identity
module keyed off a metadata block named ``entry``. TCG — which has minted every
artifact since pgw#1270 — writes ``graph_class``. Nothing in ``src/`` has
written an ``entry`` block since that cut; only fixtures did. So every
production consumer of the block was reading a shape that no longer exists:

* ``fleet_cells._identity_axes`` raised ``CellPublishRefused`` on every real
  artifact — self-mint publish was broken outright;
* ``mint_supervisor`` counted every already-packed graph class as NOT held and
  recompiled it, because the ``KeyError`` landed in a broad ``except`` that
  logs "not a readable compiled graph";
* the boot-key memo silently answered "" and disabled itself.

None of it raised in CI, because every fixture built the obsolete shape. That
is the exact silent-revert hazard the shared key corpus exists to fence, in the
one place the corpus could not see: the corpus fences the KEY GRAMMAR, not
which block the axes are read FROM.

The fix is the deletion: there is no second identity module to drift.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from gen_worker._vendor.torchcg import identity as tcg_identity
from gen_worker._vendor.torchcg import is_compiled_graph_key

from gen_worker import env_seal, fleet_cells, graph_facts, local_cell_store

#: Exactly the metadata shape ``aot_mint`` records for a minted graph class:
#: TCG's ``graph_class`` block, never an ``entry`` block.
#: Split out so the toolchain block keeps its own type: indexing the
#: heterogeneous metadata dict widens it to Collection[str] under strict mypy.
TCG_TOOLCHAIN: dict[str, str] = {"torch": "c" * 16, "ptxas": "d" * 16}

TCG_METADATA: dict[str, Any] = {
    "kind": "aot-inductor",
    "sm": "sm_89",
    "graph_class": {"name": "unet", "target": "unet", "class_hash": "a" * 16},
    "toolchain": dict(TCG_TOOLCHAIN),
}

#: pgw#1341: the env-seal digest the hub's ArtifactIdentity requires by name is
#: NOT a key axis (pgw#1059 amendment 4) and NOT a metadata field — TCG's closed
#: vocabulary has none. It rides the mint provenance the local store holds, and
#: the publish path fails closed without it, which is a separate refusal from
#: the identity one this file is about.
PROVENANCE = local_cell_store.MintProvenance(
    env_seal=env_seal.seal_digest({"v": 4, "matmul_precision": "high"}),
    lane="bf16-w16a16", sku="l4", gen_worker="0.123.0")

VECTORS = pathlib.Path(__file__).parent / "testdata" / "compiled_graph_key_vectors.json"


def _vectors() -> list:
    loaded = json.loads(VECTORS.read_text())
    return loaded["vectors"] if isinstance(loaded, dict) else loaded


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: v["note"][:40])
def test_the_shared_corpus_is_answered_by_tcgs_grammar(vector: dict) -> None:
    """The vendored corpus is the three-way contract (pgw, tensorhub, TCG).

    It must be answered by the ONE surviving implementation. A worker-side
    grammar that agreed with the corpus while TCG disagreed would be a second
    authority wearing a green fence.
    """
    assert is_compiled_graph_key(vector["key"]) is vector["valid"], vector["note"]


def test_the_worker_publish_path_keys_a_real_tcg_artifact() -> None:
    """RED before pgw#1277: the publish path refused every artifact TCG minted.

    Mutate ``graph_class`` back to ``entry`` and this fails — which is the
    regression, stated as a test rather than as a comment.
    """
    axes = fleet_cells._identity_axes("toy", dict(TCG_METADATA), PROVENANCE)
    expected = tcg_identity.from_artifact_metadata(TCG_METADATA)
    assert axes["graph"] == "a" * 16
    assert axes["sm"] == "sm_89"
    assert axes["toolchain"] == tcg_identity.toolchain_axis_digest(TCG_TOOLCHAIN)
    assert expected.value.startswith("cg-key-v1-")


def test_an_artifact_that_cannot_name_its_class_has_no_identity() -> None:
    """The refusal survives the move: a missing axis is still typed and fatal.

    Deleting a duplicate must not delete its fail-closed posture — an artifact
    published under partial axes is a row the fleet can never arm (pgw#1046).
    """
    hollow = json.loads(json.dumps(TCG_METADATA))
    hollow["graph_class"]["class_hash"] = ""
    with pytest.raises(tcg_identity.IdentityError):
        tcg_identity.from_artifact_metadata(hollow)


@pytest.mark.parametrize("axis", ["sm", "graph_class", "toolchain", "kind"])
def test_no_axis_may_be_DEFAULTED_into_existence(axis: str) -> None:
    """An artifact that cannot state an axis must REFUSE, never invent one.

    Found by red-proving pgw#1277, and it is the pgw#1270 failure verbatim:
    giving ``sm`` an ``or "cpu"`` default inside TCG left the whole pgw suite
    green, because nothing here asserted the CARDLESS case. That exact edit is
    what once turned a deterministic refusal into a published, unadoptable
    artifact — the line that looks like a default is the line that MANUFACTURES
    an identity decision.

    So every required axis is dropped in turn and the refusal must be typed.
    A key minted from a defaulted axis names a compiled graph nobody compiled.
    """
    cardless = json.loads(json.dumps(TCG_METADATA))
    cardless.pop(axis)
    with pytest.raises(tcg_identity.IdentityError):
        tcg_identity.from_artifact_metadata(cardless)


@pytest.mark.parametrize("axis", ["sm", "kind"])
def test_an_empty_axis_is_refused_as_hard_as_a_missing_one(axis: str) -> None:
    """``""`` is the shape a probe returns when it could not answer — the
    likeliest way a default sneaks in is an empty string surviving as a value.
    """
    blank = json.loads(json.dumps(TCG_METADATA))
    blank[axis] = ""
    with pytest.raises(tcg_identity.IdentityError):
        tcg_identity.from_artifact_metadata(blank)


def test_a_key_is_refused_where_a_fact_belongs() -> None:
    """A key is the OUTPUT of the computation and never an input to it.

    Hashing an identity into another identity produces a key no artifact can
    restate; the constructor refuses it by name.
    """
    forged = json.loads(json.dumps(TCG_METADATA))
    forged["graph_class"]["class_hash"] = "cg-key-v1-" + "f" * 56
    with pytest.raises(tcg_identity.IdentityError):
        tcg_identity.from_artifact_metadata(forged)


#: KNOWN-ANSWER VECTORS — the only thing in this repo that pins the key VALUE.
#:
#: Found by red-proving pgw#1277 rather than by design: mutating TCG's canonical
#: fold (``separators=(",", ":")`` -> ``(", ", ": ")``) left the entire pgw
#: suite GREEN. Every other identity assertion here is RELATIONAL — equal,
#: not-equal, has the prefix, matches the grammar — and a fold change moves all
#: keys together, so relations survive it intact. The one test that did bind the
#: value, ``test_worker_publish_projection_matches_public_tcg_key``, needs torch
#: and cannot run on a control-plane box.
#:
#: A fold change is a FLEET RE-KEY: every published artifact is orphaned. That
#: must never be reachable by an accidental edit that no gate notices, so these
#: three vectors are pinned by hand. If one fails, do not "update the expected
#: value" — establish first that a deliberate re-key was intended.
KAT_AXES = {"graph": "0f0e0d0c0b0a0908", "sm": "sm_90", "toolchain": "bb11cc22dd33ee44"}
KAT_KEY = "cg-key-v1-3460b43a25317f2d8d5ab5fce1e5cf5b7817d6447ef2343dc6f3b78b"
KAT_TOOLCHAIN = {"torch": "aa" * 8, "ptxas": "bb" * 8, "diffusers": "cc" * 8}
KAT_TOOLCHAIN_AXIS = "4863b405217cfc4b"
KAT_META = {
    "kind": "aot-inductor",
    "sm": "sm_90",
    "graph_class": {"class_hash": "0f0e0d0c0b0a0908"},
    "toolchain": {"torch": "aa" * 8, "ptxas": "bb" * 8},
}
KAT_META_KEY = "cg-key-v1-7d14d604a8fbe097eb5f730f0b9290b4ce4e2482268de6ea3d76b626"


def test_the_key_fold_is_pinned_by_value_not_only_by_relation() -> None:
    """Any change to the canonical fold moves these, and nothing else here."""
    assert tcg_identity.from_axes(KAT_AXES).value == KAT_KEY


def test_the_toolchain_axis_is_pinned_by_value() -> None:
    """Pins the fold AND the eviction together: ``diffusers`` is in the input
    and must not be in the digest (pgw#1050)."""
    assert tcg_identity.toolchain_axis_digest(KAT_TOOLCHAIN) == KAT_TOOLCHAIN_AXIS


def test_the_artifact_derivation_is_pinned_by_value() -> None:
    """The whole path an artifact travels — recorded block to published key."""
    assert tcg_identity.from_artifact_metadata(KAT_META).value == KAT_META_KEY


def test_the_worker_holds_no_second_identity_module() -> None:
    """The deletion IS the fix — a correctly-named duplicate computing wrong
    keys with nothing raising is the failure this unit removes."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("gen_worker.compiled_graph_key")
