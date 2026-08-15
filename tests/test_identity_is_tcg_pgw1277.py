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

import pytest
from torch_compiled_graphs import identity as tcg_identity
from torch_compiled_graphs import is_compiled_graph_key

from gen_worker import fleet_cells

#: Exactly the metadata shape ``aot_mint`` records for a minted graph class:
#: TCG's ``graph_class`` block, never an ``entry`` block.
TCG_METADATA = {
    "kind": "aot-inductor",
    "sm": "sm_89",
    "graph_class": {"name": "unet", "target": "unet", "class_hash": "a" * 16},
    "toolchain": {"torch": "c" * 16, "ptxas": "d" * 16},
}

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
    axes = fleet_cells._identity_axes("toy", dict(TCG_METADATA))
    expected = tcg_identity.from_artifact_metadata(TCG_METADATA)
    assert axes["graph"] == "a" * 16
    assert axes["sm"] == "sm_89"
    assert axes["toolchain"] == tcg_identity.toolchain_axis_digest(
        TCG_METADATA["toolchain"])
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


def test_a_key_is_refused_where_a_fact_belongs() -> None:
    """A key is the OUTPUT of the computation and never an input to it.

    Hashing an identity into another identity produces a key no artifact can
    restate; the constructor refuses it by name.
    """
    forged = json.loads(json.dumps(TCG_METADATA))
    forged["graph_class"]["class_hash"] = "cg-key-v1-" + "f" * 56
    with pytest.raises(tcg_identity.IdentityError):
        tcg_identity.from_artifact_metadata(forged)


def test_the_worker_holds_no_second_identity_module() -> None:
    """The deletion IS the fix — a correctly-named duplicate computing wrong
    keys with nothing raising is the failure this unit removes."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("gen_worker.compiled_graph_key")
