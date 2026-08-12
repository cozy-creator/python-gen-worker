"""gw#581/th#883 (redefined by pgw#1059): the ONE worker-owned cell-key
brain + the local-cell verdict invariants.

Outcome-level: a key is deterministic and axis-sensitive on exactly the
four ck1 axes; the local (torch-inductor-cache) store verdict compares
recorded facts with the producer's own derivations; a SELF-VERIFIED cell
that fails to arm surfaces as cell_selection_bug (never a silent eager
fallback); foreign cells keep the compatibility-miss policy.

The redefinition's own invariants (membership axiom, one-derivation fence,
old/new non-collision, envelope canonicalization) live in
``tests/test_cell_key_pgw1059.py``.
"""

from __future__ import annotations

import pytest

from gen_worker import cell_key as ck
from gen_worker import compile_cache as cc


class _ContractCfg:
    """Duck-typed declared-compile-contract source (registry.CompileCell)."""

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
    "graph": "0f0e0d0c0b0a0908", "envelope": "aa00bb11cc22dd33",
    "sm": "sm_100", "toolchain": "bb11cc22dd33ee44",
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
    assert a.digest == ck.from_axes(dict(_AXES)).digest
    assert ck.is_key(a.digest)
    for axis in ("graph", "envelope", "sm", "toolchain"):
        bumped = dict(_AXES, **{axis: _AXES[axis] + "x"})
        assert ck.from_axes(bumped).digest != a.digest, axis


def test_unknown_and_missing_axes_refuse():
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(_AXES, cuda_driver="13020"))  # host lottery axis
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(_AXES, sku="b200"))  # observability, never identity
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(_AXES, torch="2.13.0"))  # version axes are gone
    with pytest.raises(ck.CellKeyError):
        ck.from_axes({k: v for k, v in _AXES.items() if k != "toolchain"})


def test_key_scheme_ck1_foreign_keys_are_key_shaped_but_distinct():
    """pgw#958 (§1.27(g)) + pgw#1059 amendment 1: ck1 is the only scheme
    this runtime mints — the redefinition kept the number (Paul: "stick
    with version-1 for now since we're still pre-launch") and PURGES the
    pre-redefinition corpus instead of minting ck2.

    Shape stays scheme-AGNOSTIC, byte-identical to tensorhub's
    `compilecache.IsCellKey` (th#1183): a foreign-scheme token IS
    key-shaped; it simply names no artifact this runtime computes.
    """
    key = ck.from_axes(_AXES).digest
    assert key.startswith("ek1-")
    for dead in ("ck2-", "ck3-", "ck4-", "ck5-", "ck6-"):
        token = dead + "a" * 56
        assert ck.is_key(token), "a foreign-scheme token is still key-SHAPED"
        assert token != key
    assert not ck.is_key("ck-" + "a" * 56)      # no scheme digits
    assert not ck.is_key("ek1-" + "a" * 55)     # wrong digest width
    assert not ck.is_key("ek1-" + "A" * 56)     # uppercase hex


def test_execution_lane_canonicalization():
    """fp8-hooks and w8a16 are one lane label; buckets fold into it. The
    lane is store metadata + discovery scoping since pgw#1059 — the
    one-derivation rule stands so a cell is scoped under the same spelling
    it was stamped with."""
    assert (cc.execution_lane_label("fp8-hooks")
            == cc.execution_lane_label("w8a16"))
    assert (cc.execution_lane_label("w8a8-lora128")
            == cc.execution_lane_label("w8a8", 128))
    assert cc.execution_lane_label("w8a8") != cc.execution_lane_label("")


# pgw#1181 REMOVED the six local-cell-verdict rows:
# `test_local_cell_has_no_key_stamp`, `test_local_verdict_ignores_sku_and_pins_sm`,
# `test_declared_contract_fences_newer_contract`,
# `test_self_requested_drift_is_selection_bug`,
# `test_self_requested_no_target_is_selection_bug` and
# `test_foreign_cell_drift_stays_eager`.
#
# Their subject is the `torch-inductor-cache` store verdict —
# `compile_cache.local_cell_mismatch` over `artifact_metadata`, and the
# `cell_selection_bug` a self-requested cell of that format raised when it then
# refused to arm. The format has had no writer since pgw#1178 deleted
# `mint_artifact`, and pgw#1181 deleted the format: there is no local kind, no
# verdict to render on one, and `enable` no longer takes a cell to reject.
#
# Every property they fenced survives on the exported lane BY CONSTRUCTION
# rather than by comparison, which is the point of a content-addressed key:
# sm, the declared contract, the env seal and the lane are all axes of `ck1`,
# so a cell that disagrees on any of them has a different key and never
# resolves. `tests/test_cell_key_pgw1059.py` is where that is stated, with the
# staleness matrix naming each axis. What is left here is the key itself —
# determinism, axis sensitivity, the ck1 scheme, and lane canonicalization.
