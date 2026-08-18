"""gw#581/th#883 (redefined by pgw#1059): the ONE worker-owned compiled graph-key
brain + the local-compiled graph verdict invariants.

Outcome-level: a key is deterministic and axis-sensitive on exactly the
four ck1 axes; the local (torch-inductor-cache) store verdict compares
recorded facts with the producer's own derivations; a SELF-VERIFIED compiled graph
that fails to arm surfaces as compiled_graph_selection_bug (never a silent eager
fallback); foreign compiled graphs keep the compatibility-miss policy.

The redefinition's own invariants (membership axiom, one-derivation fence,
old/new non-collision, envelope canonicalization) live in
``tests/test_compiled_graph_key_pgw1059.py``.
"""

from __future__ import annotations

import pytest

from gen_worker._vendor.torchcg import identity as ck
from gen_worker._vendor.torchcg import is_compiled_graph_key
from gen_worker import compile_cache as cc


class _ContractCfg:
    """Duck-typed declared-compile-contract source (registry.CompileContract)."""

    def __init__(
        self, *, shapes=((768, 768),), targets=("transformer",), text_len=0,
        dynamic=(), regional=False, lora_bucket=0, guidance_scales=(),
    ):
        self.shapes = shapes
        self.targets = targets
        self.text_len = text_len
        self.dynamic = dynamic
        self.regional = regional
        self.lora_bucket = lora_bucket
        self.guidance_scales = guidance_scales


_AXES = {
    "graph": "0f0e0d0c0b0a0908", "sm": "sm_100", "toolchain": "bb11cc22dd33ee44",
}

_RT = {
    "sku": "b200", "sm": "sm_100", "torch": "2.13.0+cu130",
    "triton": "3.7.1", "cuda": "13.0", "cuda_driver": "13020",
    "image_digest": "",
}


@pytest.fixture()
def fixed_runtime(monkeypatch):
    """Pin every probe the verdicts read so outcomes are host-independent."""
    monkeypatch.setattr(cc, "runtime_key", lambda: dict(_RT))
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.36.10")
    monkeypatch.setattr(
        cc, "_lib_versions",
        lambda: {"diffusers": "0.39.0", "transformers": "5.13.1"})
    monkeypatch.delenv("WORKER_IMAGE_DIGEST", raising=False)


def test_key_deterministic_and_axis_sensitive():
    a = ck.from_axes(_AXES)
    assert a.value == ck.from_axes(dict(_AXES)).value
    assert is_compiled_graph_key(a.value)
    for axis in ("graph", "sm", "toolchain"):
        bumped = dict(_AXES, **{axis: _AXES[axis] + "x"})
        assert ck.from_axes(bumped).value != a.value, axis


def test_unknown_and_missing_axes_refuse():
    with pytest.raises(ck.IdentityError):
        ck.from_axes(dict(_AXES, cuda_driver="13020"))  # host lottery axis
    with pytest.raises(ck.IdentityError):
        ck.from_axes(dict(_AXES, sku="b200"))  # observability, never identity
    with pytest.raises(ck.IdentityError):
        ck.from_axes(dict(_AXES, torch="2.13.0"))  # version axes are gone
    with pytest.raises(ck.IdentityError):
        ck.from_axes({k: v for k, v in _AXES.items() if k != "toolchain"})


def test_the_deriver_and_the_validator_agree_on_cg_key_v1():
    """pgw#1213: `is_compiled_graph_key` admits exactly what `from_axes(...).value` mints.

    THE row that can go red on a scheme change: the deriver writes the
    scheme through `_PREFIX` and the validator reads it through the
    right-anchored digest match, so a change to one that is not made in the
    other fails here rather than at a resolve nobody watches.

    The grammar itself is the CROSS-REPO contract with tensorhub's
    `compilecache.IsCompiledGraphKey`; its rows live in
    `tests/test_key_grammar_vectors_th1897.py`, against the corpus both repos
    vendor.
    """
    key = ck.from_axes(_AXES).value
    assert key.startswith("cg-key-v1-")
    assert is_compiled_graph_key(key) is True


def test_the_grammar_refuses_shape_never_scheme():
    """th#1183, restored (th#1897): pgw#1213's first pass hard-cut `is_key` to
    `cg-key-v1` only, and the shared corpus refuses that reading — a pinned
    scheme means a newer fleet's key stops being addressable by an older hub,
    so the two can never again ship in different windows. A foreign scheme is
    admitted here and ruled on by AXES, which is where it actually misses.

    SHAPE is what is refused, and the digest is the SUFFIX — never split on
    `-`, because the scheme carries hyphens itself.
    """
    for foreign in ("ek1-", "ck1-", "cg-key-v2-", "a-", "cg.key_v1-"):
        assert is_compiled_graph_key(foreign + "a" * 56), foreign
        assert ck.from_axes(_AXES).value != foreign + "a" * 56
    assert not is_compiled_graph_key("cg-key-v1-" + "a" * 55)     # digest too short
    assert not is_compiled_graph_key("cg-key-v1-" + "a" * 57)     # digest too long
    assert not is_compiled_graph_key("cg-key-v1-" + "A" * 56)     # uppercase hex
    assert not is_compiled_graph_key("cg-key-v1" + "a" * 56)      # no separator
    assert not is_compiled_graph_key("-" + "a" * 56)              # empty scheme
    assert not is_compiled_graph_key("cg-key-v1-" + "a" * 56 + "\n")  # \Z, not $
    assert not is_compiled_graph_key("")


def test_execution_lane_canonicalization():
    """fp8-hooks and w8a16 are one lane label; buckets fold into it. The
    lane is store metadata + discovery scoping since pgw#1059 — the
    one-derivation rule stands so a compiled graph is scoped under the same spelling
    it was stamped with."""
    assert (cc.execution_lane_label("fp8-hooks")
            == cc.execution_lane_label("w8a16"))
    assert (cc.execution_lane_label("w8a8-lora128")
            == cc.execution_lane_label("w8a8", 128))
    assert cc.execution_lane_label("w8a8") != cc.execution_lane_label("")


# pgw#1181 REMOVED the six local-compiled graph-verdict rows:
# `test_local_cell_has_no_key_stamp`, `test_local_verdict_ignores_sku_and_pins_sm`,
# `test_declared_contract_fences_newer_contract`,
# `test_self_requested_drift_is_selection_bug`,
# `test_self_requested_no_target_is_selection_bug` and
# `test_foreign_cell_drift_stays_eager`.
#
# Their subject is the `torch-inductor-cache` store verdict —
# `compile_cache.local_compiled_graph_mismatch` over `artifact_metadata`, and the
# `compiled_graph_selection_bug` a self-requested compiled graph of that format raised when it then
# refused to arm. The format has had no writer since pgw#1178 deleted
# `mint_artifact`, and pgw#1181 deleted the format: there is no local kind, no
# verdict to render on one, and `enable` no longer takes a compiled graph to reject.
#
# Every property they fenced survives on the exported lane BY CONSTRUCTION
# rather than by comparison, which is the point of a content-addressed key:
# sm, the declared contract, the env seal and the lane are all axes of `cg-key-v1`
# or fold into one (pgw#1176),
# so an entry that disagrees on any of them has a different key and never
# resolves. `tests/test_compiled_graph_key_pgw1059.py` is where that is stated, with the
# staleness matrix naming each axis. What is left here is the key itself —
# determinism, axis sensitivity, the cg-key-v1 scheme, and lane canonicalization.
