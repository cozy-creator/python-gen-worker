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
    assert "SERVING EAGER" in said               # states the outcome
    assert "gen-worker compile" in said          # names the remedy


def test_the_discard_itself_is_announced_and_names_what_went_away(
    tmp_path, caplog
):
    """WHICH of the two "nothing usable" states it was is still stated.

    The adopt lines deliberately say the same thing for a stale store and an
    empty one -- that is this issue's conclusion, not an omission. The
    difference is not lost: torchcg's discard WARNING names the graph-set, the
    bytes it dropped and the decode failure. This asserts the operator gets
    BOTH, because the adopt line alone would leave a silent deletion.
    """
    import logging

    store = tmp_path / "graph-cas"
    _store_with_unreadable_document(store, "toy.main")
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")

    with caplog.at_level(
        logging.WARNING, logger="gen_worker._vendor.torchcg.store"
    ):
        _adoption_source(spec, "toy.main")

    said = "\n".join(
        r.getMessage()
        for r in caplog.records
        if r.name == "gen_worker._vendor.torchcg.store"
    )
    assert "DISCARDED" in said
    assert "toy.main" in said                    # which graph-set
    assert "sha256:" in said                     # which bytes
    assert "unreadable by this build" in said    # why


def test_the_shape_actually_found_on_disk_is_the_version_mismatch(
    tmp_path, caplog
):
    """The document that blocked va#3's arm 2 was v=2 with a VALID field set.

    Worth its own case because the fixture above fails on the field set and
    would pass a reader that only rejected unknown fields. This one can only
    fail on the version, which is the shape a re-vendor actually produces.
    """
    import logging

    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.torchcg.store import _document_ref

    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    cas = LocalCAS(store)
    ref = cas.put_bytes(b'{"lanes":[],"stack":[],"v":2}')
    cas.compare_and_swap_ref(_document_ref("toy.main"), ref, expected=None)

    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")
    with caplog.at_level(
        logging.WARNING, logger="gen_worker._vendor.torchcg.store"
    ):
        _, document = _adoption_source(spec, "toy.main")

    assert document is None
    said = "\n".join(r.getMessage() for r in caplog.records)
    assert "document v must be 3" in said
    assert cas.read_ref(_document_ref("toy.main")) is None


def test_a_store_that_still_raises_keeps_the_typed_refusal(
    tmp_path, capsys, monkeypatch
):
    """The `StoreError` branch is NOT dead -- it guards the other backends.

    tcg#69 taught the LOCAL store to answer a clean miss, so that branch no
    longer fires for `LocalGraphStore`. The hub-backed store is a different
    implementation of the same protocol and still raises, and a `up` must
    degrade on it too rather than die. Driven through a store double so this
    is a test of the CALL SITE, which is what owns the outcome.
    """
    import gen_worker._vendor.torchcg.store as store_module
    from gen_worker._vendor.torchcg.store import StoreError

    class RaisingStore:
        def get_graphs(self, name: str) -> None:
            raise StoreError(f"graph-set document {name!r} is unreadable: nope")

    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")

    # `_adoption_source` imports the class inside the function, so the module
    # attribute is what it resolves at call time.
    monkeypatch.setattr(store_module, "LocalGraphStore", lambda _cas: RaisingStore())

    adopted_store, document = _adoption_source(spec, "toy.main")

    assert document is None
    assert adopted_store is not None
    said = capsys.readouterr().err
    assert "graph_store_unreadable" in said
    assert "SERVING EAGER" in said
    assert "gen-worker compile" in said


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
