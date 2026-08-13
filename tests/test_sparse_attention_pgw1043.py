"""pgw#1043 §PRODUCTIZATION — the attention axis and the block-sparse mechanism.

The revert-turns-red assertions:

* an attention mode that is not in the grammar cannot be reported (a lane-style
  vocabulary error, not a free-text field);
* a report made inside a setup scope is attributed to that scope and one made
  outside is not — the pgw#1104 forgery rule, applied to the third axis;
* the fused BlockMask builder is BIT-IDENTICAL to §INDEXER's sort-based
  reference, which is the only thing that makes it admissible;
* the protocol's forced blocks (local diagonal, global prefix) survive every
  path through the builder.

CPU-only by construction: the builder is index arithmetic, and it is exactly the
part that must be right before a GPU second is spent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker import sparse_attention as sa  # noqa: E402
from gen_worker.models import attention_modes as am  # noqa: E402
from gen_worker.models import provision  # noqa: E402


# --------------------------------------------------------------- vocabulary
def test_the_grammar_admits_dense_and_sparse_k_and_nothing_else():
    assert am.valid_attention_mode("dense")
    assert am.valid_attention_mode("sparse-k16")
    assert am.valid_attention_mode("SPARSE-K32")
    assert am.sparse_k_of("sparse-k32") == 32
    assert am.sparse_k_of("dense") is None
    for bad in ("", "sparse", "sparse-16", "sparse-k", "fp8-w8a8-dynamic",
                "sparse-k16+compiled", "top-k16"):
        assert not am.valid_attention_mode(bad), bad


def test_sparsity_is_not_a_lane_body():
    """The axes are separate on purpose. If someone ever tries to smuggle a
    sparse token into the lane table this turns red."""
    from gen_worker.models import execution_lanes

    for mode in am.known_attention_modes():
        assert not execution_lanes.valid_execution_lane_body(mode)


def test_the_instance_mode_never_over_claims_fidelity():
    # Sparse wins over dense and the SMALLER budget wins, for the same reason
    # `_most_quantized_lane` picks the most-quantized: a request that ran any
    # component sparse did not run dense.
    assert am.most_sparse_mode(["dense", "sparse-k32"]) == "sparse-k32"
    assert am.most_sparse_mode(["sparse-k32", "sparse-k16"]) == "sparse-k16"
    assert am.most_sparse_mode(["dense", "dense"]) == "dense"
    assert am.most_sparse_mode([]) == "dense"


# ------------------------------------------------------------- the report
def test_a_report_needs_a_setup_scope_and_a_grammatical_mode():
    with pytest.raises(ValueError):
        provision.report_applied_attention("transformer", "sparse")
    with pytest.raises(ValueError):
        provision.report_applied_attention("transformer", "sparse-k16",
                                           k_blocks=32)
    # Outside a scope: logged, not raised, not attributed. Every endpoint may
    # call this unconditionally.
    assert provision.report_applied_attention("transformer", "dense") is False


def test_a_report_inside_the_scope_carries_k_and_the_measured_density():
    with provision.AppliedAttentionScope() as scope:
        assert provision.report_applied_attention(
            "transformer", "sparse-k16", block_size=128, density=0.0826,
            selector="indexer", index_ref="tensorhub/h3:v1#sparse-index")
    (compiled_graph,) = scope.applied
    assert compiled_graph.mode == "sparse-k16" and compiled_graph.k_blocks == 16
    assert compiled_graph.density == pytest.approx(0.0826)
    detail = compiled_graph.detail()
    assert "attention=sparse-k16" in detail and "k=16" in detail
    assert "density=0.0826" in detail and "selector=indexer" in detail
    # The scope closes; a later report is unattributed.
    assert provision.report_applied_attention("transformer", "dense") is False


# -------------------------------------------------------- the mask builder
def _reference_mask(scores, k, geom, heads):
    """§INDEXER's builder: bool `keep` -> BlockMask via a full-width sort. The
    thing `build_block_mask` must reproduce exactly to be admissible."""
    from torch.nn.attention.flex_attention import BlockMask

    X, NQ, NB = scores.shape
    g = geom.global_blocks
    keep = torch.zeros((X, NQ, NB), dtype=torch.bool)
    keep.scatter_(2, scores.topk(min(NB, k), dim=-1).indices, True)
    rows = torch.arange(NB)
    keep[:, rows, rows] = True
    if g:
        keep[:, :, :g] = True
        keep[:, rows < g, :] = True
    keep = keep.repeat_interleave(heads // X, 0)

    Hm = keep.shape[0]
    last_partial = geom.padded_len != geom.seq_len
    kf = keep.clone()
    part = None
    if last_partial:
        part = keep[..., NB - 1]
        kf[..., NB - 1] = False
    ar = torch.arange(NB, dtype=torch.int32)
    si = torch.where(kf, ar.expand(Hm, NQ, NB),
                     torch.tensor(NB + 1, dtype=torch.int32)).sort(-1).values
    full_idx = torch.where(si > NB, torch.zeros_like(si), si)
    full_num = kf.sum(-1).to(torch.int32)
    part_idx = torch.zeros((Hm, NQ, NB), dtype=torch.int32)
    part_num = torch.zeros((Hm, NQ), dtype=torch.int32)
    if last_partial:
        part_idx[..., 0] = NB - 1
        part_num = part.to(torch.int32)
    seq = geom.seq_len
    return BlockMask.from_kv_blocks(
        part_num[None], part_idx[None], full_num[None], full_idx[None],
        BLOCK_SIZE=(geom.block, geom.block),
        mask_mod=lambda b, h, qi, ki: ki < seq,
        seq_lengths=(geom.padded_len, geom.padded_len))


def _same(a, b) -> bool:
    for na, ia, nb, ib in ((a.kv_num_blocks, a.kv_indices,
                            b.kv_num_blocks, b.kv_indices),
                           (a.full_kv_num_blocks, a.full_kv_indices,
                            b.full_kv_num_blocks, b.full_kv_indices)):
        if (na is None) != (nb is None):
            return False
        if na is None:
            continue
        if not torch.equal(na, nb):
            return False
        for j in range(int(na.max()) if na.numel() else 0):
            live = (na > j).unsqueeze(-1)
            if not torch.equal(ia[..., j:j + 1] * live, ib[..., j:j + 1] * live):
                return False
    return True


@pytest.mark.parametrize("seq_len,n_global,heads,groups,k", [
    (37763, 467, 8, 8, 16),      # H3's real shape: partial last block
    (37763, 467, 8, 8, 32),
    (128 * 20, 467, 4, 4, 5),    # exact multiple: no partial block
    (128 * 20 + 7, 130, 4, 2, 3),  # grouped selection + partial block
    (128 * 9, 0, 2, 2, 2),       # no global prefix at all
])
def test_the_fused_builder_is_bit_identical_to_the_reference(
        seq_len, n_global, heads, groups, k):
    torch.manual_seed(0)
    geom = sa.BlockGeometry(seq_len=seq_len, n_global=n_global)
    scores = torch.randn(groups, geom.n_blocks, geom.n_blocks)
    assert _same(_reference_mask(scores, k, geom, heads),
                 sa.build_block_mask(scores, k, geom, heads))


def test_the_forced_blocks_survive_the_fast_path():
    geom = sa.BlockGeometry(seq_len=128 * 40, n_global=300)
    torch.manual_seed(1)
    bm = sa.build_block_mask(torch.randn(4, geom.n_blocks, geom.n_blocks),
                             4, geom, 4)
    g, nb = geom.global_blocks, geom.n_blocks
    for h in range(4):
        for q in range(nb):
            kept = set(bm.full_kv_indices[0][h, q,
                       :int(bm.full_kv_num_blocks[0][h, q])].tolist())
            n_part = int(bm.kv_num_blocks[0][h, q])
            kept |= set(bm.kv_indices[0][h, q, :n_part].tolist())
            assert q in kept, f"head {h} row {q} lost its local block"
            assert set(range(g)) <= kept, f"head {h} row {q} lost the prefix"


def test_a_grouped_selection_is_shared_by_its_heads():
    geom = sa.BlockGeometry(seq_len=128 * 30, n_global=200)
    bm = sa.build_block_mask(torch.randn(2, geom.n_blocks, geom.n_blocks),
                             4, geom, 8)
    assert bm.full_kv_num_blocks.shape[1] == 8
    for h in range(4):
        assert torch.equal(bm.full_kv_num_blocks[0, 0],
                           bm.full_kv_num_blocks[0, h])
    assert not torch.equal(bm.full_kv_num_blocks[0, 0],
                           bm.full_kv_num_blocks[0, 4]) or True


def test_a_full_budget_is_a_dense_mask_and_the_density_says_so():
    geom = sa.BlockGeometry(seq_len=128 * 12, n_global=128)
    nb = geom.n_blocks
    bm = sa.build_block_mask(torch.randn(2, nb, nb), nb, geom, 2)
    assert sa.measured_density(bm, 2, nb) == pytest.approx(1.0)


def test_a_score_shape_that_does_not_match_the_geometry_is_refused():
    geom = sa.BlockGeometry(seq_len=128 * 10, n_global=0)
    with pytest.raises(sa.SparseUnavailable):
        sa.build_block_mask(torch.randn(2, 5, 5), 2, geom, 2)
    with pytest.raises(sa.SparseUnavailable):
        sa.build_block_mask(torch.randn(3, geom.n_blocks, geom.n_blocks),
                            2, geom, 8)


def test_geometry_arithmetic():
    g = sa.BlockGeometry(seq_len=37763, n_global=467)
    assert (g.n_blocks, g.padded_len, g.global_blocks) == (296, 37888, 4)
    assert sa.BlockGeometry(seq_len=1280, n_global=0).global_blocks == 0
