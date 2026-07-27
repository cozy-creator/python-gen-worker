"""gw#581/th#883: the ONE worker-owned cell-key brain + receipt invariants.

Outcome-level: a key is deterministic and axis-sensitive; mint metadata
stamps exactly the key the same runtime would request; a SELF-REQUESTED cell
that fails to arm surfaces as cell_selection_bug (never a silent eager
fallback); foreign/pre-key cells keep the legacy verify/eager policy.
"""

from __future__ import annotations

import pytest

from gen_worker import Compile
from gen_worker import cell_key as ck
from gen_worker import compile_cache as cc


class _ContractCfg:
    """Duck-typed declared-shape-contract source (registry.CompileCell shape)."""

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


_FACTS = cc.declared_contract_facts(_ContractCfg())
_CONTRACT = ck.contract_digest(_FACTS)

# ck5 recipe axes: version strings and image identity live in METADATA
# only (observability); content digests carry the identity.
_AXES = {
    "format": "2", "kind": "inductor", "family": "ltx-2.3", "lane": "w8a8",
    "sm": "sm_100", "contract": _CONTRACT,
    "env_seal": "aa00bb11cc22dd33", "toolchain": "bb11cc22dd33ee44",
    "code_closure": "cc22dd33ee44ff55",
}

_RT = {
    "sku": "b200", "sm": "sm_100", "torch": "2.13.0+cu130",
    "triton": "3.7.1", "cuda": "13.0", "cuda_driver": "13020",
    "image_digest": "",
}


@pytest.fixture()
def fixed_runtime(monkeypatch):
    """Pin every probe the brain reads so keys are host-independent."""
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
    for axis in ("family", "lane", "sm", "contract", "env_seal",
                 "toolchain", "code_closure"):
        bumped = dict(_AXES, **{axis: _AXES[axis] + "x"})
        assert ck.from_axes(bumped).digest != a.digest, axis


def test_empty_optional_axis_equals_absent():
    absent = {k: v for k, v in _AXES.items() if k not in ("lane", "mode")}
    empty = dict(absent, lane="", mode="")
    assert ck.from_axes(absent).digest == ck.from_axes(empty).digest


def test_unknown_and_missing_axes_refuse():
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(_AXES, cuda_driver="13020"))  # host lottery axis
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(_AXES, sku="b200"))  # demoted to metadata (pgw#691)
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(_AXES, torch="2.13.0"))  # version axes left in ck5
    with pytest.raises(ck.CellKeyError):
        ck.from_axes({k: v for k, v in _AXES.items() if k != "toolchain"})


def test_sku_is_not_identity(fixed_runtime):
    """pgw#691: the audited corpus held two byte-identical cell PAIRS split
    only by sku (a40 vs rtx-3090 on sm_86; l4 vs rtx-4090 on sm_89) — 19%
    redundant mints. No dynamo guard observes a SKU; sm + cuda + torch +
    triton pin every hardware fact the guard set carries. Two artifacts
    identical on every axis but sku share ONE identity; sm remains one."""

    def _meta(sku, sm="sm_86"):
        meta = cc.artifact_metadata(
            family="sdxl", shapes=((768, 768),), targets=("transformer",),
            shape_contract=_FACTS,
        )
        meta.update({"sku": sku, "sm": sm})
        return ck.stamp(meta)

    a40, rtx3090 = _meta("a40"), _meta("rtx-3090")
    assert a40["cell_key"] == rtx3090["cell_key"]
    # sku SURVIVES in metadata — the hub-side selection preference and the
    # publish-intent anti-forgery attestation both read it from there.
    assert a40["sku"] == "a40" and rtx3090["sku"] == "rtx-3090"
    # sm stays identity: a different arch is a different cell.
    assert _meta("a40", sm="sm_89")["cell_key"] != a40["cell_key"]


def test_key_scheme_ck5_old_keys_never_half_match():
    """Each axis-set change bumps the scheme (sku collapse -> ck3, env_seal
    -> ck4, recipe identity -> ck5): an older digest is no longer a key at
    all — a clean MISS, never a half-match."""
    key = ck.from_axes(_AXES).digest
    assert key.startswith("ck5-")
    for dead in ("ck2-", "ck3-", "ck4-"):
        assert not ck.is_key(dead + "a" * 56)
        assert "not a cell key" in ck.mismatch({}, dead + "a" * 56)


def test_compute_matches_artifact_metadata_stamp(fixed_runtime):
    """Mint-side stamp == consumer-side request, by construction: the SAME
    declared-contract facts feed compute() and the artifact's recorded
    shape_contract block."""
    want = ck.compute("ltx-2.3", "w8a8", contract=_CONTRACT).digest
    meta = cc.artifact_metadata(
        family="ltx-2.3", shapes=((768, 768),), targets=("transformer",),
        weight_lane="w8a8", shape_contract=_FACTS,
    )
    assert meta["cell_key"] == want
    assert ck.mismatch(meta, want) == ""
    assert ck.from_artifact_metadata(meta).digest == want


def test_lane_canonicalization(fixed_runtime):
    """fp8-hooks and w8a16 are one graph family; buckets fold into the lane."""
    assert (ck.compute("f", "fp8-hooks", contract=_CONTRACT).digest
            == ck.compute("f", "w8a16", contract=_CONTRACT).digest)
    assert (ck.compute("f", "w8a8-lora128", contract=_CONTRACT).digest
            == ck.compute("f", "w8a8", 128, contract=_CONTRACT).digest)
    assert (ck.compute("f", "w8a8", contract=_CONTRACT).digest
            != ck.compute("f", "", contract=_CONTRACT).digest)


def test_regional_mode_is_identity(fixed_runtime):
    """Regional per-block cells are different artifacts (ie#381)."""
    assert (ck.compute("f", regional=True, contract=_CONTRACT).digest
            != ck.compute("f", contract=_CONTRACT).digest)
    assert (ck.from_axes(dict(_AXES, mode="regional")).digest
            != ck.from_axes(_AXES).digest)
    facts = cc.declared_contract_facts(_ContractCfg(regional=True))
    meta = cc.artifact_metadata(
        family="f", shapes=((768, 768),), targets=("transformer",),
        compile_mode="regional", shape_contract=facts,
    )
    assert meta["cell_key"] == ck.compute(
        "f", regional=True, contract=ck.contract_digest(facts)).digest


def test_contract_axis_fences_newer_contract(fixed_runtime):
    """pgw#647: the declared shape contract is a key axis. Two artifacts
    identical except for their shape_contract get DIFFERENT keys, and a
    worker computing with contract A never matches a cell recorded with
    contract B — a newer contract must not consume an older cell."""
    facts_a = cc.declared_contract_facts(_ContractCfg(text_len=0))
    facts_b = cc.declared_contract_facts(_ContractCfg(text_len=512))
    assert facts_a != facts_b

    def _cell(facts):
        return cc.artifact_metadata(
            family="ltx-2.3", shapes=((768, 768),), targets=("transformer",),
            shape_contract=facts,
        )

    meta_a, meta_b = _cell(facts_a), _cell(facts_b)
    assert meta_a["cell_key"] != meta_b["cell_key"]

    want_a = ck.compute("ltx-2.3", contract=ck.contract_digest(facts_a))
    assert ck.mismatch(meta_a, want_a) == ""
    reason = ck.mismatch(meta_b, want_a)
    assert reason.startswith("contract:")  # the named-axis refusal

    # Pre-ck2 cells record no shape_contract: deliberately NO ck2 identity —
    # never a stamped key, never a self-requested match.
    legacy = cc.artifact_metadata(
        family="ltx-2.3", shapes=((768, 768),), targets=("transformer",))
    assert "cell_key" not in legacy
    with pytest.raises(ck.CellKeyError):
        ck.from_artifact_metadata(legacy)
    assert "no computable key" in ck.mismatch(legacy, want_a)


def test_trt_metadata_has_no_cell_key():
    with pytest.raises(ck.CellKeyError):
        ck.from_artifact_metadata(dict(_AXES, kind="trt-engine"))


def test_is_cache_ref_accepts_key_flavor():
    key = ck.from_axes(_AXES).digest
    assert cc.is_cache_ref(f"root/family-ltx-2.3#{key}")
    assert cc.is_cache_ref(f"root/family-ltx-2.3#{key}", "ltx-2.3")
    assert not cc.is_cache_ref(f"root/family-ltx-2.3#{key}", "sdxl")
    assert not cc.is_cache_ref(f"owner/repo#{key}")


def test_cell_lane_matcher_uses_candidate_keys():
    from gen_worker.executor import _cell_lane_matches

    key = ck.from_axes(_AXES).digest
    ref = f"root/family-ltx-2.3#{key}"
    assert _cell_lane_matches(
        ref, "ltx-2.3", want_lane="w8a8", want_bucket=0,
        candidate_keys={key})
    assert not _cell_lane_matches(
        ref, "ltx-2.3", want_lane="w8a8", want_bucket=0,
        candidate_keys={"ck1-" + "0" * 56})
    # legacy labels keep the lane-parse policy
    assert _cell_lane_matches(
        "root/family-ltx-2.3#inductor-b200-torch2.13-w8a8",
        "ltx-2.3", want_lane="w8a8", want_bucket=0)


class _Target:
    def forward(self, value):
        return value


class _Pipeline:
    def __init__(self):
        self.transformer = _Target()


def _self_cell(tmp_path, drift: str = ""):
    """Pack a cell whose axes describe exactly this (pinned) runtime."""
    pipe = _Pipeline()
    cfg = Compile(
        shapes=((768, 768),), family="sd15", targets=("transformer",),
    )
    signature, contract = cc.execution_contract(pipe, cfg)
    meta = cc.artifact_metadata(
        family="sd15", shapes=cfg.shapes, targets=cfg.targets,
        graph_signature=signature, weight_contract=contract,
        shape_contract=cc.declared_contract_facts(cfg),
    )
    if drift:
        meta["graph_signature"] = drift
    source = tmp_path / "candidate"
    for sub in ("inductor", "triton"):
        (source / sub).mkdir(parents=True, exist_ok=True)
    (source / "inductor" / "graph.py").write_text("x")
    artifact = cc.pack(source, tmp_path / "cell.tar.gz", meta)
    return pipe, cfg, artifact


def test_self_requested_drift_is_selection_bug(
    tmp_path, monkeypatch, fixed_runtime,
):
    """A cell whose axes ARE this runtime's own key must never silently
    fall back to eager on parity drift — that's the bug class (th#883)."""
    pipe, cfg, artifact = _self_cell(tmp_path, drift="different-module-graph")
    monkeypatch.setattr(cc, "apply", lambda *a, **k: False)
    with pytest.raises(cc.CellSelectionBugError) as exc:
        cc.enable(pipe, cfg, tmp_path / "cache", artifact)
    assert "refused to arm" in str(exc.value)


def test_self_requested_no_target_is_selection_bug(
    tmp_path, monkeypatch, fixed_runtime,
):
    pipe, cfg, artifact = _self_cell(tmp_path)
    monkeypatch.setattr(cc, "apply", lambda *a, **k: False)
    with pytest.raises(cc.CellSelectionBugError) as exc:
        cc.enable(pipe, cfg, tmp_path / "cache", artifact)
    assert "armed no compile target" in str(exc.value)


def test_foreign_cell_drift_stays_eager(
    tmp_path, monkeypatch, fixed_runtime,
):
    """The identical drift on a NON-self-keyed cell keeps the legacy silent
    eager policy — compatibility outcomes are not bugs."""
    pipe, cfg, artifact = _self_cell(tmp_path, drift="different-module-graph")
    monkeypatch.setattr(cc, "apply", lambda *a, **k: False)
    monkeypatch.setattr(
        cc, "gen_worker_version", lambda: "9.9.9")  # not my key anymore
    assert cc.enable(pipe, cfg, tmp_path / "cache", artifact) is False
