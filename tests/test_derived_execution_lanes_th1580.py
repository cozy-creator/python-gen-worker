"""th#1580 A4: the endpoint-side execution-lane set is DERIVED from the
decoders in the image, not hand-listed.

Proven here against a FAKE image package (two real decoders, one that cannot
import) plus the real gen_worker.models tree:

  1. two decoders declare exactly their two contracts and no others;
  2. a decoder module that fails to import is EXCLUDED WITH ITS REASON;
  3. the execution axis comes from the runtime lane table, never the decoder;
  4. exclusions are DERIVED from a function's declared traits (A4 corollary:
     no exclusion marker exists to test);
  5. an unregistered contract handle fails the BUILD (A2).
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
from gen_worker.models.tensor_layout_contract import (
    CONTRACT_COZY_FP8_ROWWISE,
    CONTRACT_NUNCHAKU_V1,
    implements_contract,
)

_GOOD_A = '''
from gen_worker.models.tensor_layout_contract import implements_contract

@implements_contract(
    contract="nunchaku.v1@1", serves=("svdq-fp4-w4a4",), composes_lora=False,
    why="fake svdq decoder",
)
def decode_svdq(tensors):
    return tensors
'''

_GOOD_B = '''
from gen_worker.models.tensor_layout_contract import implements_contract

@implements_contract(
    contract="cozy.fp8-rowwise@1", serves=("fp8-w8a8-dynamic",),
    composes_lora=True, why="fake fp8 decoder",
)
def decode_fp8(tensors):
    return tensors
'''

# A decoder whose dependency is absent — the real failure this excludes is an
# image built without the kernel extension its decoder imports at module top.
_BROKEN = '''
import a_dependency_this_image_does_not_have  # noqa: F401

from gen_worker.models.tensor_layout_contract import implements_contract

@implements_contract(
    contract="bfl.nvfp4-preswizzled@1", serves=("nvfp4-w4a4-static",),
    composes_lora=False, why="never reached",
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
    (pkg / "svdq_decoder.py").write_text(textwrap.dedent(_GOOD_A), "utf-8")
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


def test_two_decoders_declare_exactly_their_contracts(fake_image):
    derived = derive_execution_lanes(packages=(fake_image,))

    assert [c.contract for c in derived.contracts] == [
        CONTRACT_COZY_FP8_ROWWISE,
        CONTRACT_NUNCHAKU_V1,
    ]
    # And NOTHING else: the third registered contract the broken module would
    # have claimed is absent, and the two contracts no decoder mentions are
    # absent. "Supports everything" is not a reachable answer here.
    assert "bfl.nvfp4-preswizzled@1" not in {c.contract for c in derived.contracts}
    assert "plain.bf16@1" not in {c.contract for c in derived.contracts}
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
    """The decoder declares a lane BODY; eager/compiled is the platform's."""
    derived = derive_execution_lanes(packages=(fake_image,))

    lanes = set(derived.execution_lanes)
    # svdq is eager-only and w8a8 compiled-only in the lane table, and neither
    # decoder said so — the cross produced it.
    assert lanes == {"fp8-w8a8-dynamic+compiled", "svdq-fp4-w4a4+eager"}


def test_lora_bucket_derives_the_exclusion(fake_image):
    """A4 corollary: no exclusion marker. `lora_bucket` x `composes_lora`."""
    derived = derive_execution_lanes(packages=(fake_image,))

    plain, no_exclusions = execution_lanes_for_function(derived, lora_bucket=0)
    assert set(plain) == {"fp8-w8a8-dynamic+compiled", "svdq-fp4-w4a4+eager"}
    assert no_exclusions == ()

    lora, exclusions = execution_lanes_for_function(derived, lora_bucket=4)
    assert set(lora) == {"fp8-w8a8-dynamic+compiled"}
    assert [e.execution_lane for e in exclusions] == ["svdq-fp4-w4a4+eager"]
    assert "lora_bucket=4" in exclusions[0].reason
    assert "nunchaku.v1@1" in exclusions[0].reason


def test_unregistered_contract_fails_the_build():
    """A2: contracts are code. A decoder cannot mint one at the marker."""
    with pytest.raises(ValueError, match="is not registered"):
        @implements_contract(
            contract="acme.secret-format@1",
            serves=("bf16-w16a16",),
            composes_lora=False,
        )
        def _decode(x):
            return x

    with pytest.raises(ValueError, match="not a contract handle"):
        @implements_contract(
            contract="nvfp4", serves=("bf16-w16a16",), composes_lora=False
        )
        def _decode2(x):
            return x

    with pytest.raises(ValueError, match="not a known lane body"):
        @implements_contract(
            contract=CONTRACT_NUNCHAKU_V1,
            serves=("svdq-fp4-w4a4+eager",),  # execution axis is not a body
            composes_lora=False,
        )
        def _decode3(x):
            return x


def test_real_image_tree_derives_its_own_set():
    """The real gen_worker.models tree, imported the way the bake imports it."""
    from gen_worker.discovery.heavy_deps import stub_missing_heavy_deps

    with stub_missing_heavy_deps():
        derived = derive_execution_lanes()

    by_contract = {c.contract: c for c in derived.contracts}
    assert CONTRACT_NUNCHAKU_V1 in by_contract
    assert CONTRACT_COZY_FP8_ROWWISE in by_contract
    assert "plain.bf16@1" in by_contract
    # Our own flat-nvfp4 w4a4 decoder claims nothing: the registry has no
    # entry for it yet (ours is LOW-nibble + unswizzled scales, which is NOT
    # bfl.nvfp4-preswizzled@1 — te#151 measured what conflating them costs).
    # An unclaimed decoder contributes no lane, which is the honest answer.
    assert "bfl.nvfp4-preswizzled@1" not in by_contract
    assert "nvfp4-w4a4-static+compiled" not in set(derived.execution_lanes)
    # svdq has no adapter branch; the three branch-capable lanes do.
    assert by_contract[CONTRACT_NUNCHAKU_V1].composes_lora is False
    assert by_contract[CONTRACT_COZY_FP8_ROWWISE].composes_lora is True
    assert by_contract["plain.bf16@1"].composes_lora is True
