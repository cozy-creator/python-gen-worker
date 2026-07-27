"""pgw#700/#710/#711/#712: equivalence adoption — change-detection tiers.

Paul's th#1229 ruling: version barriers drop; adoption is gated on DETECTED
change. FAST tier = recorded code closure byte-identical; SLOW tier =
manifest + composition-fingerprint proof; safety floor = confirmed-only
candidates (#711 digests on publish), republish fencing (#712), toolchain
content digests (#710). Red-verified against real module trees and real
file digests — no mocked closure entries where a real file exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

torch = pytest.importorskip("torch")

from gen_worker import cell_key as ck
from gen_worker import compile_cache as cc
from gen_worker import equivalence as eq
from gen_worker import fleet_cells as fc
from gen_worker import guard_closure as gc
from gen_worker.registry import CompileCell

FAMILY = "toyfam"


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _cfg(**overrides: Any) -> CompileCell:
    base: Dict[str, Any] = dict(
        shapes=((64, 64),), targets=("transformer",), family=FAMILY,
        regional=False, text_len=None, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=(),
    )
    base.update(overrides)
    return CompileCell(**base)


class _Tree(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(8, 8)
        self.lin2 = torch.nn.Linear(8, 8)

    def forward(self, x: Any) -> Any:
        return self.lin2(self.lin1(x))


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Tree()


_RT = {
    "sku": "l4", "sm": "sm_89", "cuda": "13.0", "cuda_driver": "13000",
    "torch": "2.13.0+cu130", "triton": "3.7.1", "image_digest": "",
}


@pytest.fixture()
def pinned_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "runtime_key", lambda: dict(_RT))
    monkeypatch.setattr(cc, "_lib_versions", lambda: {})
    monkeypatch.delenv("WORKER_IMAGE_DIGEST", raising=False)


def _manifest() -> Dict[str, Any]:
    return {
        "v": 2,
        "graphs": [{"target": "transformer", "code": "forward", "entry": 0,
                    "guards": [{"type": "TENSOR_MATCH", "source": "L['x']",
                                "expr": "e", "verdict": gc.CANONICALIZED,
                                "axis": "ingress"}]}],
        "verdicts": {}, "leaks": [],
        gc.POSTURE_KEY: dict(gc.CANONICAL_POSTURE),
    }


def _mint(pipe: Any, cfg: Any) -> Dict[str, Any]:
    """A full self-consistent minted metadata for ``pipe`` at the CURRENT
    (monkeypatched) runtime identity."""
    signature, weight_contract = cc.execution_contract(pipe, cfg)
    meta = cc.artifact_metadata(
        family=FAMILY, shapes=cfg.shapes, targets=cfg.targets,
        guidance_scales=cfg.guidance_scales, graph_signature=signature,
        weight_contract=weight_contract,
        shape_contract=cc.declared_contract_facts(cfg),
        composition=cc.composition_fingerprint(pipe, cfg),
    )
    meta[gc.MANIFEST_KEY] = _manifest()
    return meta


def _want(monkeypatch: pytest.MonkeyPatch, gen_worker_version: str) -> ck.CellKey:
    monkeypatch.setattr(cc, "gen_worker_version", lambda: gen_worker_version)
    return ck.compute(
        FAMILY, "", 0,
        contract=ck.contract_digest(cc.declared_contract_facts(_cfg())),
    )


# ---------------------------------------------------------------------------
# #710: toolchain content digests + the recorded code closure
# ---------------------------------------------------------------------------


def test_toolchain_and_closure_are_recorded() -> None:
    toolchain = dict(cc.toolchain_digest())
    assert "torch" in toolchain and "triton" in toolchain
    assert all(len(v) == 16 for v in toolchain.values())
    closure = dict(cc.code_closure())
    assert "gen_worker/compile_cache.py" in closure
    assert all(len(v) == 16 for v in closure.values())
    assert all(not k.startswith("/") for k in closure)  # never absolute
    meta = cc.artifact_metadata(
        family=FAMILY, shapes=((64, 64),), targets=("transformer",))
    assert meta["toolchain"] == toolchain
    assert set(meta["code_closure"]) >= set(closure)


def test_closure_delta_redigests_real_files() -> None:
    closure = dict(cc.code_closure())
    assert eq.closure_delta(closure) == []  # same venv: byte-identical
    tampered = dict(closure)
    tampered["gen_worker/compile_cache.py"] = "0" * 16
    diffs = eq.closure_delta(tampered)
    assert len(diffs) == 1 and "gen_worker/compile_cache.py" in diffs[0]
    missing = {"gen_worker/does_not_exist_xyz.py": "0" * 16}
    assert "absent in this runtime" in eq.closure_delta(missing)[0]


# ---------------------------------------------------------------------------
# The verdict: flag, tiers, named refusals
# ---------------------------------------------------------------------------


def test_flag_off_refuses(pinned_runtime: None,
                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(eq.FLAG_ENV, raising=False)
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "1.0.0")
    pipe = _Pipe()
    meta = _mint(pipe, _cfg())
    verdict = eq.verdict(meta, _want(monkeypatch, "1.0.1"), pipe, _cfg())
    assert eq.FLAG_ENV in verdict
    assert eq.ADOPTION_MARK not in meta


def test_gen_worker_bridge_adopts_via_fast_tier(
    pinned_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Paul-ruling case: an SDK version bump with byte-identical code
    closure adopts WITHOUT the composition proof — removing the composition
    rows proves the fast tier carried it."""
    monkeypatch.setenv(eq.FLAG_ENV, "1")
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "1.0.0")
    pipe = _Pipe()
    meta = _mint(pipe, _cfg())
    del meta["composition"]  # slow tier would refuse without this
    want = _want(monkeypatch, "1.0.1")  # consumer bumped the SDK
    assert eq.verdict(meta, want, pipe, _cfg()) == ""
    assert meta[eq.ADOPTION_MARK] == ["gen_worker"]


def test_closure_drift_falls_to_slow_tier(
    pinned_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(eq.FLAG_ENV, "1")
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "1.0.0")
    pipe = _Pipe()
    meta = _mint(pipe, _cfg())
    meta["code_closure"] = dict(
        meta["code_closure"], **{"gen_worker/compile_cache.py": "0" * 16})
    want = _want(monkeypatch, "1.0.1")
    # Slow tier: composition matches -> still adoptable, by proof.
    assert eq.verdict(dict(meta), want, pipe, _cfg()) == ""
    # Slow tier with no composition rows -> named refusal.
    bare = dict(meta)
    del bare["composition"]
    assert "composition fingerprint" in eq.verdict(bare, want, pipe, _cfg())
    # Slow tier with a REAL module drift -> names the module.
    drifted = _Pipe()
    drifted.transformer.lin2.half()
    assert "transformer:lin2" in eq.verdict(
        dict(meta), want, drifted, _cfg())


def test_undesignated_axis_refuses_named(
    pinned_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(eq.FLAG_ENV, "1")
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "1.0.0")
    pipe = _Pipe()
    meta = _mint(pipe, _cfg())
    monkeypatch.setattr(
        cc, "runtime_key", lambda: dict(_RT, torch="2.14.0+cu130"))
    verdict = eq.verdict(meta, _want(monkeypatch, "1.0.0"), pipe, _cfg())
    assert "'torch'" in verdict and "not equivalence-designated" in verdict


def test_safety_floor_refusals_are_named(
    pinned_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(eq.FLAG_ENV, "1")
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "1.0.0")
    pipe = _Pipe()
    cfg = _cfg()
    # Mint everything FIRST (under 1.0.0); _want re-patches the version.
    no_manifest = _mint(pipe, cfg)
    leaky = _mint(pipe, cfg)
    meta = _mint(pipe, cfg)
    stack = _mint(pipe, cfg)
    tools = _mint(pipe, cfg)
    want = _want(monkeypatch, "1.0.1")

    del no_manifest[gc.MANIFEST_KEY]
    assert "guard manifest" in eq.verdict(no_manifest, want, pipe, cfg)

    leaky[gc.MANIFEST_KEY]["leaks"] = ["row"]
    assert "leaks" in eq.verdict(leaky, want, pipe, cfg)

    with torch.no_grad():
        assert "grad_enabled" in eq.verdict(meta, want, pipe, cfg)

    stack["content_keys"] = dict(stack["content_keys"], torch="f" * 16)
    verdict = eq.verdict(stack, want, pipe, cfg)
    assert "content_keys/torch" in verdict and "byte-identical" in verdict

    tools["toolchain"] = dict(tools["toolchain"], torch="f" * 16)
    assert "toolchain/torch" in eq.verdict(tools, want, pipe, cfg)


def test_exact_key_is_not_equivalence(
    pinned_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(eq.FLAG_ENV, "1")
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "1.0.0")
    pipe = _Pipe()
    meta = _mint(pipe, _cfg())
    assert eq.verdict(meta, _want(monkeypatch, "1.0.0"), pipe, _cfg()) == ""
    assert eq.ADOPTION_MARK not in meta  # exact hits carry no mark


# ---------------------------------------------------------------------------
# #712: fencing + unicity
# ---------------------------------------------------------------------------


def test_equivalence_adopted_cell_never_republishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    posts: list = []

    def _post(url: str, **kwargs: Any) -> Any:  # pragma: no cover - fenced
        posts.append(url)
        raise AssertionError("fence must fire before any hub call")

    import requests

    monkeypatch.setattr(requests, "post", _post)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="")
    meta = {"cell_key": "ck4-" + "a" * 56,
            eq.ADOPTION_MARK: ["gen_worker"]}
    with pytest.raises(fc.CellPublishRefused, match="pgw#712"):
        pub.publish(FAMILY, artifact, meta)
    assert posts == []


def test_select_enforces_unicity() -> None:
    a = {gc.MANIFEST_KEY: _manifest()}
    b = {gc.MANIFEST_KEY: _manifest()}
    index, reason = eq.select([a, b])
    assert (index, reason) == (0, "")
    divergent = {gc.MANIFEST_KEY: dict(_manifest(), leaks=["x"])}
    index, reason = eq.select([a, divergent])
    assert index == -1 and "cell_equivalence_divergence" in reason
    index, reason = eq.select([{"no": "manifest"}])
    assert index == -1 and "no candidate" in reason


# ---------------------------------------------------------------------------
# #711 (SDK half): the publish carries the confirmation digests
# ---------------------------------------------------------------------------


def test_publish_complete_carries_confirmation_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    posts: list = []
    key = "ck4-" + "c" * 56

    class _FakeResp:
        def __init__(self, code: int, body: Dict[str, Any]) -> None:
            self.status_code = code
            self.text = json.dumps(body)

        def json(self) -> Dict[str, Any]:
            return json.loads(self.text)

    def _post(url: str, headers: Any = None, json_body: Any = None,
              json: Any = None, timeout: Any = None) -> Any:
        posts.append((url, json))
        if url.endswith("/publish-intent"):
            return _FakeResp(200, {
                "capability_token": "cap", "repo": f"root/family-{FAMILY}",
                "cell_key": key})
        return _FakeResp(200, {"recorded": True})

    import requests

    monkeypatch.setattr(requests, "post", _post)

    class _FakeHub:
        def __init__(self, **kw: Any) -> None: ...

        def commit(self, **kw: Any) -> Any:
            class _R:
                checkpoint_id = "cp-1"

            return _R()

    import gen_worker.convert.hub as hub_mod

    monkeypatch.setattr(hub_mod, "HubClient", _FakeHub)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"cell-bytes")
    meta = {"cell_key": key, "sku": "l4", "gen_worker": "1.0.0",
            gc.MANIFEST_KEY: _manifest()}
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="")
    assert pub.publish(FAMILY, artifact, meta) == "cp-1"

    complete_url, body = posts[-1]
    assert complete_url.endswith("/publish-complete")
    assert body["ok"] is True
    assert body["artifact_digest"].startswith("blake3:")
    assert len(body["artifact_digest"]) == len("blake3:") + 64
    assert body["manifest_digest"] == gc.manifest_digest(_manifest())
    assert body["manifest_digest"].startswith("sha256:")
