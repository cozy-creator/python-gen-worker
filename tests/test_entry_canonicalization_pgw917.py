"""pgw#917 — an area-preserving aspect family is ONE dispatchable compiled graph.

The measured failure, on the standing chaos stack (read-only SQL, 2026-08-03):
``worker_activity_events`` holds **24 ``aot_ingress_refused`` rows, every one
``phase='compiled_graph_ambiguous'``** — zero ``no_compiled_graph_admits``, zero anything else —
summing to **4,200 refused calls** across gen-worker 0.89.0 and 0.90.0, against
a compiled graph that adopted and armed (``mode=regional compiled graphs=72 precision=
w8a8-lora64``).  The compiled graph armed, advertised, and served nothing.

The mechanism is arithmetic, not a race.  112x144 = 144x112 = 168x96 = 96x168
= 16,128: the four aspect rows of one megapixel bucket.  A block-level target
never sees ``H_lat`` and ``W_lat`` — it sees the flattened sequence
``(B, H_lat*W_lat, C)``.  The declaration keys compiled graphs on the pair; the
INGRESS CONTRACT can only observe the product.  Ambiguity is therefore
guaranteed for every area-preserving aspect family at a fixed bucket, which is
exactly how the fleet's shape rows are generated.

So the compiled graph key and the ingress contract must be the same object: rows that
reduce to one contract over one target with byte-identical code are merged to
one compiled graph with the declared names kept as aliases (36 of that compiled graph's 72
compiles bought nothing — the direct pgw#847 win), and a collision whose
members are NOT the same artifact is refused by name and by differing axis.

Real ``torch.export`` programs, real ``aot_package.input_contract``, real
``aot_serve.assert_ingress``, real ``graph_hash``.  CPU, no AOTInductor
compile, no GPU.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_flatten, aot_mint, compiled_graph_key  # noqa: E402

FAMILY = "sdxl"
DIM = 640
XATTN = 2048
TEXT_LEN = 77

#: The four area-preserving aspect rows of ONE megapixel bucket, verbatim from
#: the live refusal detail. Every product is 16,128.
AREA_PRESERVING: Tuple[Tuple[int, int], ...] = (
    (112, 144), (144, 112), (168, 96), (96, 168),
)


class _Block(nn.Module):
    """A ``BasicTransformerBlock``-shaped ingress: the spatial extents are
    already flattened away by the time this module is called."""

    def forward(self, hidden_states: Any, encoder_hidden_states: Any) -> Any:
        return hidden_states.mean() + encoder_hidden_states.mean()


class _OtherBlock(nn.Module):
    """Same ingress signature, DIFFERENT code — the conflicting-identity
    fixture. Nothing about its contract discriminates it, so only the graph
    identity can, which is the axis the refusal must name."""

    def forward(self, hidden_states: Any, encoder_hidden_states: Any) -> Any:
        return hidden_states.sum() - encoder_hidden_states.sum()


def _export(h: int, w: int, *, b: int = 2, module: nn.Module | None = None) -> Any:
    args = (torch.zeros(b, h * w, DIM), torch.zeros(b, TEXT_LEN, XATTN))
    return torch.export.export(
        (module or _Block()).eval(), args, strict=True)


def _row_name(b: int, h: int, w: int, block: int = 0) -> str:
    return (f"unet/adapter=false,block=BasicTransformerBlock#{block},cfg=true"
            f"/B={b},H_lat={h},T_txt={TEXT_LEN},W_lat={w}")


def _compiled_graph(name: str, program: Any, *, h: int, w: int, b: int = 2,
           target: str = "unet") -> Any:
    return aot_mint._MintedCompiledGraph(
        name=name,
        spec=aot_mint.ExportSpec(
            family=FAMILY, target=target,
            fork=((aot_mint.ADAPTER_FORK, False), ("cfg", True)),
            class_dims=(("B", b), ("H_lat", h), ("T_txt", TEXT_LEN),
                        ("W_lat", w)),
        ),
        module=None,
        owner=None,
        program=program,
        input_names=("hidden_states", "encoder_hidden_states"),
        flat_leaves=tuple(
            aot_flatten.Leaf(param=n, param_position=i, path=())
            for i, n in enumerate(("hidden_states", "encoder_hidden_states"))),
        files=[],
        timings={},
    )


def _the_live_declaration() -> list:
    return [
        _compiled_graph(_row_name(2, h, w), _export(h, w), h=h, w=w)
        for h, w in AREA_PRESERVING
    ]


def test_the_exact_live_sdxl_collision_canonicalizes_to_one_compiled_graph():
    """The acceptance, verbatim: (112,144)/(144,112)/(168,96)/(96,168)
    canonicalize to ONE compiled graph with four aliases."""
    minted = _the_live_declaration()
    assert len(minted) == 4

    # Precondition — this really is the live failure: every compiled graph admits
    # every other compiled graph's declared call, which is what dispatch refuses.
    contracts = [aot_mint._compiled_graph_ingress_declaration(row) for row in minted]
    for _c, calls, _m in contracts:
        for contract, _calls, _meta in contracts:
            assert all(aot_mint._admits(contract, call) for call in calls)

    kept, aliases = aot_mint.canonicalize_dispatch_classes(minted)
    assert len(kept) == 1, "four identical-contract rows are ONE compiled_graph"
    survivor = kept[0].name
    assert survivor in aliases
    assert sorted(row.name for row in aliases[survivor]) == sorted(
        row.name for row in minted if row.name != survivor)
    assert len(aliases[survivor]) == 3


def test_the_merge_is_the_pgw847_win_three_compiles_of_four_bought_nothing():
    """12 blocks x 4 area-preserving rows = 48 declared compiled graphs collapse to
    12 — the '36 compiles of the 72 that bought nothing' arithmetic."""
    minted = []
    for block in range(12):
        for h, w in AREA_PRESERVING:
            minted.append(_compiled_graph(
                _row_name(2, h, w, block=block), _export(h, w), h=h, w=w,
                target=f"unet[BasicTransformerBlock#{block}]"))
    assert len(minted) == 48
    kept, aliases = aot_mint.canonicalize_dispatch_classes(minted)
    assert len(kept) == 12
    assert sum(len(rows) for rows in aliases.values()) == 36


def test_a_same_contract_collision_with_different_code_REFUSES_by_axis():
    """The other half of the acceptance: identical ingress contract, differing
    code, refused before publish with the differing axis named."""
    minted = [
        _compiled_graph(_row_name(2, 112, 144), _export(112, 144), h=112, w=144),
        _compiled_graph(_row_name(2, 144, 112), _export(144, 112, module=_OtherBlock()),
               h=144, w=112),
    ]
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint.canonicalize_dispatch_classes(minted)
    text = str(err.value)
    assert "compiled_graph_ambiguous" in text
    assert "'graph'" in text, "the differing axis must be named"
    assert "H_lat=112" in text and "H_lat=144" in text, (
        "the colliding pair must be named")


def test_a_same_contract_collision_with_different_compat_metadata_REFUSES():
    """Compatibility metadata is an identity axis too — two rows that admit
    identically but were traced under different lanes are different
    artifacts, and merging them would ship one lane's code under both names."""
    left = _compiled_graph(_row_name(2, 112, 144), _export(112, 144), h=112, w=144)
    right = _compiled_graph(_row_name(2, 144, 112), _export(144, 112), h=144, w=112)
    right.spec.weight_lane = "w8a8"
    with pytest.raises(aot_mint.MintRefused) as err:
        aot_mint.canonicalize_dispatch_classes([left, right])
    assert "'specialization'" in str(err.value)


def test_a_merged_compiled_graph_declares_one_dispatchable_compiled_graph_and_keeps_the_names():
    """End of the acceptance chain: what the envelope records.  The aliases
    ride the surviving compiled graph, and they are NOT a ``class_hash`` fact — an
    alias declares no traffic the survivor's own contract does not already
    declare, so an otherwise identical compiled graph must not re-key."""
    from gen_worker import aot_serve

    minted = _the_live_declaration()
    kept, aliases = aot_mint.canonicalize_dispatch_classes(minted)
    survivor = kept[0]
    block: Dict[str, Any] = {
        "target": survivor.spec.target,
        "fork": [[str(n), v] for n, v in sorted(survivor.spec.fork)],
        "class_dims": [
            [str(n), int(v)] for n, v in sorted(survivor.spec.class_dims)],
        "inputs": [
            {"name": "hidden_states", "position": 0, "dtype": "float32",
             "shape": [2, 16128, DIM]},
            {"name": "encoder_hidden_states", "position": 1,
             "dtype": "float32", "shape": [2, TEXT_LEN, XATTN]},
        ],
        "symbols": {},
        "constants": [],
    }
    bare = aot_serve.compiled_graph_metadata(
        family=FAMILY, precision="w8a8", compiled_graph_key="",
        name=survivor.name, compiled_graph=dict(block))
    with_aliases = dict(block)
    with_aliases["aliases"] = [
        {"name": row.name,
         "class_dims": [[str(n), int(v)] for n, v in sorted(row.spec.class_dims)]}
        for row in sorted(aliases[survivor.name], key=lambda r: r.name)
    ]
    stamped = aot_serve.compiled_graph_metadata(
        family=FAMILY, precision="w8a8", compiled_graph_key="",
        name=survivor.name, compiled_graph=with_aliases)

    # pgw#1176: one artifact, one class — and the claim that mattered is
    # sharper per compiled graph: recording the merged declared-class names must not
    # move THIS class's identity, which is now the key itself rather than a
    # digest over a collection.
    assert stamped[compiled_graph_key.COMPILED_GRAPH_BLOCK_KEY]["class_hash"] == \
        bare[compiled_graph_key.COMPILED_GRAPH_BLOCK_KEY]["class_hash"], (
            "recording the merged declared-class names must not re-key the "
            "compiled_graph")
    assert len(stamped[compiled_graph_key.COMPILED_GRAPH_BLOCK_KEY]["aliases"]) == 3
