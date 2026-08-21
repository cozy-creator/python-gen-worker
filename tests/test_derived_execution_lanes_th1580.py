"""th#1580 A4: the endpoint-side execution-lane set is DERIVED from the
decoders in the image, not hand-listed.

Proven here against a FAKE image package (two real decoders, one that cannot
import) plus the real gen_worker.models tree:

  1. two decoders declare exactly their two QUANT RULES and no others;
  2. a decoder module that fails to import is EXCLUDED WITH ITS REASON;
  3. the execution axis comes from the runtime lane table, never the decoder;
  4. exclusions are DERIVED from a function's declared traits (A4 corollary:
     no exclusion marker exists to test);
  5. an unratified quant-rule handle fails the BUILD (A2).

**pgw#1621 re-keyed what a decoder declares.** It named a v1 CONTRACT plus five
`DecodeDimensions` axes — elements, scales, key topologies, file layouts, bakes
— because a v1 handle named a byte FORMAT and said nothing about which of that
format's legal shapes the decoder read. A v2 QUANT RULE carries its conventions
as IDENTITY, so naming the rule is the whole declaration and `decodes=` is
deleted rather than made optional. The properties above are unchanged; only the
vocabulary they are stated in moved.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from gen_worker.discovery.execution_lanes import (
    DERIVATION,
    derive_execution_lanes,
    execution_lanes_for_function,
    manifest_block,
)
from gen_worker.models.tensor_layout_contract import implements_quant_rule

#: Two RATIFIED rules the real image does NOT declare, so the fake image's set
#: cannot be confused with the real one. Chosen for their lane bodies:
#: `bf16-w16a16` is the one body the runtime table offers BOTH execution
#: options for, and `fp8-w8a8-dynamic` is compiled-only — which is what makes
#: test (3) below a real statement about the cross rather than an identity.
FAKE_A_RULE = "plain.f16@1"
FAKE_B_RULE = "hf.fp8-blockwise@1"

_GOOD_A = '''
from gen_worker.models.tensor_layout_contract import implements_quant_rule

@implements_quant_rule(
    rule="plain.f16@1", serves=("bf16-w16a16",), composes_lora=True,
    why="fake fp16 dense decoder",
)
def decode_dense(tensors):
    return tensors
'''

_GOOD_B = '''
from gen_worker.models.tensor_layout_contract import implements_quant_rule

@implements_quant_rule(
    rule="hf.fp8-blockwise@1", serves=("fp8-w8a8-dynamic",),
    composes_lora=False,
    why="fake blockwise fp8 decoder",
)
def decode_fp8(tensors):
    return tensors
'''

# A decoder whose dependency is absent — the real failure this excludes is an
# image built without the kernel extension its decoder imports at module top.
_BROKEN = '''
import a_dependency_this_image_does_not_have  # noqa: F401

from gen_worker.models.tensor_layout_contract import implements_quant_rule

@implements_quant_rule(
    rule="bfl.nvfp4-preswizzled@1", serves=("nvfp4-w4a4-static",),
    composes_lora=False,
    why="never reached",
)
def decode_nvfp4(tensors):
    return tensors
'''


@pytest.fixture
def fake_image(tmp_path: pytest.TempPathFactory, monkeypatch):
    """A package standing in for one image's decoder set."""
    root = Path(str(tmp_path))
    pkg = root / "fake_decoders"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", "utf-8")
    (pkg / "dense_decoder.py").write_text(textwrap.dedent(_GOOD_A), "utf-8")
    (pkg / "fp8_decoder.py").write_text(textwrap.dedent(_GOOD_B), "utf-8")
    (pkg / "broken_decoder.py").write_text(textwrap.dedent(_BROKEN), "utf-8")
    monkeypatch.syspath_prepend(str(root))
    for name in [n for n in sys.modules if n.startswith("fake_decoders")]:
        del sys.modules[name]
    try:
        yield "fake_decoders"
    finally:
        for name in [n for n in sys.modules if n.startswith("fake_decoders")]:
            del sys.modules[name]


def test_two_decoders_declare_exactly_their_quant_rules(fake_image):
    derived = derive_execution_lanes(packages=(fake_image,))

    assert [c.rule for c in derived.contracts] == [FAKE_B_RULE, FAKE_A_RULE]
    # And NOTHING else: the third ratified rule the broken module would have
    # claimed is absent, and the rules no decoder mentions are absent.
    # "Supports everything" is not a reachable answer here.
    declared = {c.rule for c in derived.contracts}
    assert "bfl.nvfp4-preswizzled@1" not in declared
    assert "plain.bf16@1" not in declared
    assert "cozy.fp8-rowwise@1" not in declared
    assert derived.derivation == DERIVATION


def test_import_failure_is_excluded_with_a_reason(fake_image):
    derived = derive_execution_lanes(packages=(fake_image,))

    excluded = {m.module: m.reason for m in derived.excluded_modules}
    assert "fake_decoders.broken_decoder" in excluded
    reason = excluded["fake_decoders.broken_decoder"]
    assert "ModuleNotFoundError" in reason
    assert "a_dependency_this_image_does_not_have" in reason
    # Excluded means excluded — not "declared but flagged".
    lanes = set(derived.execution_lanes)
    assert "nvfp4-w4a4-static+compiled" not in lanes
    # ...and the exclusion is VISIBLE in what the hub receives.
    block = manifest_block(derived)
    assert block["excluded_modules"] == [
        {"module": "fake_decoders.broken_decoder", "reason": reason}
    ]


def test_execution_axis_comes_from_the_runtime_table(fake_image):
    """The decoder declares a lane BODY; eager/compiled is the platform's.

    Neither fake decoder said a word about execution, and the two bodies they
    DID name expand differently: the runtime table offers `bf16-w16a16` both
    ways and `fp8-w8a8-dynamic` compiled-only, so two declarations become
    three lanes. A decoder that could name the execution axis would be able to
    contradict that table; it cannot, because it has nowhere to write it.
    """
    derived = derive_execution_lanes(packages=(fake_image,))

    assert set(derived.execution_lanes) == {
        "fp8-w8a8-dynamic+compiled",
        "bf16-w16a16+compiled",
        "bf16-w16a16+eager",
    }


def test_the_function_lane_set_is_the_image_set_ranked(fake_image):
    """pgw#1599 sweep: `lora_bucket` and the per-function EXCLUSION it
    computed are DELETED, so a function's lane set IS the image's, ranked.

    The deleted machinery narrowed a function's lanes by `lora_bucket x
    composes_lora`. It could never fire: nothing in the tree ever set
    `lora_bucket` non-zero — the sole production caller passed a literal 0 —
    so the narrowing was unreachable and its always-empty exclusions list
    looked like a live instrument. The axis survives in two sibling places
    that die in their own changes: torchcg still HASHES it into graph
    identity (a dead axis in an identity hash is a live source of phantom
    cache misses), and tensorhub still accepts it on the wire."""
    derived = derive_execution_lanes(packages=(fake_image,))

    lanes = execution_lanes_for_function(derived)
    assert set(lanes) == set(derived.execution_lanes)
    # RANKED, deterministically — the one thing this function still does.
    assert list(lanes) == sorted(lanes, key=list(lanes).index)

    # The vocabulary is gone from the signature, not merely defaulted.
    import inspect

    assert "lora_bucket" not in inspect.signature(
        execution_lanes_for_function
    ).parameters


def test_an_unratified_quant_rule_fails_the_build():
    """A2: quant rules are RATIFIED DOCUMENTS. A decoder cannot mint one at
    the marker, and there is no longer a side axis to mint one WITH.

    pgw#1621 sharpened this: under v1 the refusal read "not registered" against
    a table transcribed in this repo. It now reads against the VENDORED
    `spec/v2/rules/` corpus, so the remedy the message names — author the
    document upstream and re-vendor — is the only one there is.
    """
    with pytest.raises(ValueError, match="not in the vendored v2 corpus"):
        @implements_quant_rule(
            rule="acme.secret-format@1",
            serves=("bf16-w16a16",),
            composes_lora=False,
        )
        def _decode(x):
            return x

    with pytest.raises(ValueError, match="not a quant-rule handle"):
        @implements_quant_rule(
            rule="nvfp4", serves=("bf16-w16a16",), composes_lora=False,
        )
        def _decode2(x):
            return x

    with pytest.raises(ValueError, match="not a known lane body"):
        @implements_quant_rule(
            rule="cozy.nvfp4-flat@1",
            serves=("nvfp4-w4a4-static+compiled",),  # execution axis, not a body
            composes_lora=False,
        )
        def _decode3(x):
            return x

    # And the `decodes=` axis is GONE from the signature rather than defaulted:
    # a declaration that still passed one would be silently ignored, which is
    # the shape this whole re-key removed.
    with pytest.raises(TypeError):
        implements_quant_rule(  # type: ignore[call-arg]
            rule="plain.bf16@1", serves=("bf16-w16a16",),
            composes_lora=False, decodes=("bf16",),
        )


def test_real_image_tree_derives_its_own_set():
    """The real gen_worker.models tree, imported the way the bake imports it."""
    from gen_worker.discovery.heavy_deps import stub_missing_heavy_deps

    with stub_missing_heavy_deps():
        derived = derive_execution_lanes()

    by_rule = {c.rule: c for c in derived.contracts}
    assert "cozy.fp8-rowwise@1" in by_rule
    assert "cozy.nvfp4-flat@1" in by_rule
    assert "hf.fp8-blockwise@1" in by_rule
    assert "plain.bf16@1" in by_rule
    # The BFL pre-swizzled packaging is a different rule and a different
    # digest, and nothing in this image reads it — te#151 measured what
    # conflating it with our LOW-nibble flat packaging costs (LPIPS 1.11).
    assert "bfl.nvfp4-preswizzled@1" not in by_rule
    # `nunchaku.v1@1` was a v1 CONTRACT and has no v2 successor: no ratified
    # rule names nunchaku's SVDQ packaging. So the svdq decoder declares an
    # UNREGISTERED DECODE PATH rather than inventing a handle, which is the
    # honest answer and is visible in the census instead of in a comment.
    decoders = {u.decoder for u in derived.decode_set.unregistered}
    assert "gen_worker.models.svdq_layout:decode_linear" in decoders
    assert "gen_worker.models.loading:load_gguf_pipeline" in decoders
    assert not any(r.startswith("nunchaku.") for r in by_rule)
    # An unregistered decoder contributes NO lane — the honest answer.
    assert "svdq-fp4-w4a4+eager" not in set(derived.execution_lanes)
    # ...and the three branch-capable lanes carry the adapter trait.
    assert by_rule["cozy.fp8-rowwise@1"].composes_lora is True
    assert by_rule["plain.bf16@1"].composes_lora is True
    assert by_rule["cozy.nvfp4-flat@1"].composes_lora is False
