"""Boot-time adoption degrades; it does not die.

Subject-named, not incident-named (pgw#1362 / 4.34b); lineage in comments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.cli.daemon import BootSpec, _adoption_source


def _store_with_unreadable_document(root: Path, module: str) -> None:
    """Plant a graph-set document this build's torchcg cannot decode.

    Written through the store's OWN ref/object API rather than by hand, so the
    fixture is a store that genuinely holds an undecodable document — not a
    corrupt directory that happens to fail earlier for a different reason.
    """
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.torchcg.store import _document_ref

    root.mkdir(parents=True, exist_ok=True)
    cas = LocalCAS(root)
    ref = cas.put_bytes(b'{"v": 1, "this is a v1 document": true}')
    cas.compare_and_swap_ref(_document_ref(module), ref, expected=None)


def test_an_unreadable_graph_document_serves_eager_instead_of_killing_boot(
    tmp_path, capsys
):
    """# pgw#1525: the adopt-time StoreError cold-start regression.

    A document this build cannot decode means the box holds nothing usable —
    the same fact an empty store states. Both must reach the same outcome, or
    the format bump that produced the stale document also takes the endpoint
    down, precisely when re-minting it is what the operator is trying to do.
    """
    store = tmp_path / "graph-cas"
    _store_with_unreadable_document(store, "toy.main")

    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")
    adopted_store, document = _adoption_source(spec, "toy.main")

    assert document is None, (
        "an undecodable document must read as a MISS — the eager bridge"
    )
    assert adopted_store is not None, (
        "the store stays bound: it is still WRITABLE, so a re-mint can refill "
        "THIS store rather than requiring it be deleted first"
    )
    said = capsys.readouterr().err
    assert "graph_store_unreadable" in said      # typed, not a bare traceback
    assert "SERVING EAGER" in said               # states the outcome
    assert "gen-worker compile" in said          # names the remedy


def test_an_unreadable_document_lands_where_an_empty_store_lands(tmp_path):
    """# pgw#1525: two ways of holding nothing usable, ONE outcome.

    Asserted as equality of the DOCUMENT slot against the empty-store case
    rather than against a hand-written literal, so the two paths cannot drift
    apart without this failing.
    """
    empty = tmp_path / "empty-cas"
    empty.mkdir(parents=True)
    _, empty_doc = _adoption_source(
        BootSpec(endpoint_dir=tmp_path, graph_store=empty, sm="sm_89"), "toy.main"
    )

    stale = tmp_path / "stale-cas"
    _store_with_unreadable_document(stale, "toy.main")
    _, stale_doc = _adoption_source(
        BootSpec(endpoint_dir=tmp_path, graph_store=stale, sm="sm_89"), "toy.main"
    )

    assert stale_doc is empty_doc is None


def test_a_clean_empty_store_is_still_a_miss(tmp_path):
    """A store with no document at all: the ordinary cold start."""
    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")
    _, document = _adoption_source(spec, "toy.main")
    assert document is None


def test_adopting_without_an_sm_still_refuses(tmp_path):
    """# pgw#1523: a STATED graph store with no sm is a real user error.

    Degrading the unreadable-document case must not soften this one: the user
    asked for adoption and cannot have it, so they are told rather than quietly
    served eager.
    """
    from gen_worker.cli.daemon import BootError

    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="")
    with pytest.raises(BootError) as caught:
        _adoption_source(spec, "toy.main")
    assert "--sm is required" in str(caught.value)
