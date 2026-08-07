"""ck5 recipe identity (Paul's exact-identity ruling) + kept trust pieces.

The key IS the recipe: family + contract + sm + lane/mode + env_seal +
toolchain CONTENT + static code-closure CONTENT. No version axes, no
relaxable axes, no cross-key candidates — a recipe change strands old
cells by design. Kept from the equivalence arc as TRUST (not versioning):
pgw#711 publish digests, pgw#712 no-republish fence, pgw#710 toolchain
digests. Plus the closure-completeness mint gate: executed ⊆ static,
fail-loud naming the module.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

torch = pytest.importorskip("torch")

from gen_worker import cell_key as ck
from gen_worker import compile_cache as cc
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


@pytest.fixture()
def pinned_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "runtime_key", lambda: {
        "sku": "l4", "sm": "sm_89", "cuda": "13.0", "cuda_driver": "13000",
        "torch": "2.13.0+cu130", "triton": "3.7.1", "image_digest": "",
    })
    monkeypatch.setattr(cc, "_lib_versions", lambda: {})


# ---------------------------------------------------------------------------
# The static closure: sound, deterministic, complete
# ---------------------------------------------------------------------------


def test_static_closure_reaches_the_composition_code() -> None:
    closure = dict(cc.static_code_closure())
    # Entrypoints and their import graph (root-imports makes this sound).
    for probe in ("gen_worker/compile_cache.py", "gen_worker/cell_key.py",
                  "gen_worker/guard_closure.py", "gen_worker/env_seal.py",
                  "gen_worker/models/loading.py", "gen_worker/__init__.py"):
        assert probe in closure, probe
    assert all(len(v) == 16 for v in closure.values())
    assert all(not p.startswith("/") for p in closure)  # never absolute
    assert cc.static_code_closure() == cc.static_code_closure()  # stable


def test_endpoint_roots_join_the_closure(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint's own module shapes the graphs: passing it as a closure
    root pulls its source (and its intra-package imports) into the recipe."""
    pkg = tmp_path / "fakeendpoint"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "helper.py").write_text("X = 1\n")
    (pkg / "main.py").write_text("from fakeendpoint import helper\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    closure = dict(cc.static_code_closure(("fakeendpoint.main",)))
    assert "fakeendpoint/main.py" in closure
    assert "fakeendpoint/helper.py" in closure  # via the static import graph
    assert "fakeendpoint/__init__.py" in closure  # parents execute too
    # A content change changes the recipe digest (strands old cells).
    before = ck.facts_digest(closure)
    (pkg / "helper.py").write_text("X = 2\n")
    cc.static_code_closure.cache_clear()
    after = ck.facts_digest(dict(cc.static_code_closure(("fakeendpoint.main",))))
    assert before != after
    cc.static_code_closure.cache_clear()


def test_completeness_diagnostic_names_dynamic_imports() -> None:
    """Diagnostics for the deferred memo issue (NOT a mint gate — Paul's
    ck6 ruling): a module loaded in the composition namespaces but
    invisible to the static walk is named. Order-independent: other suite
    tests may load executor-side models/* modules (the documented
    scope-vs-walk finding), so only the ghost's presence is asserted."""
    ghost = types.ModuleType("gen_worker.models.ghost_dynamic")
    ghost.__file__ = str(
        Path(cc.__file__).parent / "models" / "ghost_dynamic.py")
    sys.modules["gen_worker.models.ghost_dynamic"] = ghost
    try:
        gaps = cc.closure_completeness_gap()
        assert "gen_worker.models.ghost_dynamic" in gaps
        with pytest.raises(RuntimeError, match="ghost_dynamic"):
            cc.assert_closure_complete()
    finally:
        del sys.modules["gen_worker.models.ghost_dynamic"]
    assert "gen_worker.models.ghost_dynamic" not in cc.closure_completeness_gap()


# ---------------------------------------------------------------------------
# ck5: the recipe digest
# ---------------------------------------------------------------------------


def test_ck5_axes_are_the_recipe(pinned_runtime: None,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    facts = cc.declared_contract_facts(_cfg())
    key = ck.compute(FAMILY, contract=ck.contract_digest(facts))
    assert key.digest.startswith("ck6-")
    axes = key.axes_dict()
    assert set(axes) == {"format", "kind", "family", "sm", "contract",
                         "env_seal", "toolchain"}
    # Version axes are GONE from the key: a version-string bump alone can
    # never re-key a cell (content digests decide).
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "99.0.0")
    monkeypatch.setenv("WORKER_IMAGE_DIGEST", "sha256:other")
    assert ck.compute(
        FAMILY, contract=ck.contract_digest(facts)).digest == key.digest
    # Older schemes can never collide with a current key; they stay
    # key-SHAPED (pgw#990 — is_key mirrors tensorhub's scheme-agnostic
    # IsCellKey) and are ruled on by axes, not by their label.
    for dead in ("ck2-", "ck3-", "ck4-", "ck5-"):
        assert ck.is_key(dead + "a" * 56)
        assert key.digest != dead + "a" * 56
    # Version-string axes are rejected outright.
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(axes, torch="2.13.0"))


def test_recipe_change_changes_the_key(pinned_runtime: None,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    facts = cc.declared_contract_facts(_cfg())
    base = ck.compute(FAMILY, contract=ck.contract_digest(facts)).digest
    # Toolchain content change -> new identity.
    monkeypatch.setattr(
        cc, "toolchain_digest", lambda: (("torch", "f" * 16),))
    assert ck.compute(
        FAMILY, contract=ck.contract_digest(facts)).digest != base


def test_metadata_roundtrips_the_recipe_key(pinned_runtime: None) -> None:
    """Mint stamp == consumer request, from the recorded recipe blocks —
    never trusted as a stamp."""
    facts = cc.declared_contract_facts(_cfg())
    meta = cc.artifact_metadata(
        family=FAMILY, shapes=((64, 64),), targets=("transformer",),
        shape_contract=facts,
    )
    want = ck.compute(FAMILY, contract=ck.contract_digest(facts))
    assert meta["cell_key"] == want.digest
    assert ck.mismatch(meta, want) == ""
    # The version strings ride metadata for observability only.
    assert meta["torch"] and meta["gen_worker"]
    # A cell with no toolchain block has no recipe identity.
    legacy = {k: v for k, v in meta.items() if k != "toolchain"}
    with pytest.raises(ck.CellKeyError, match="recipe"):
        ck.from_artifact_metadata(legacy)
    # pgw#990: `code_closure` is still RECORDED and still drives the local
    # re-trace memo, but it is NOT identity — drifting it must leave the key
    # alone. It used to re-key every cell in the fleet whenever an unrelated
    # plumbing file moved, which is what a 147-file content hash does.
    drifted = json.loads(json.dumps(meta))
    assert drifted["code_closure"], "the closure is still recorded"
    drifted["code_closure"]["gen_worker/compile_cache.py"] = "0" * 16
    assert ck.from_artifact_metadata(drifted).digest == want.digest
    assert ck.mismatch(drifted, want) == ""
    # A TOOLCHAIN drift is identity, and mismatch names the axis.
    retooled = json.loads(json.dumps(meta))
    retooled["toolchain"]["torch"] = "0" * 16
    assert ck.from_artifact_metadata(retooled).digest != want.digest
    assert ck.mismatch(retooled, want).startswith("toolchain:")


def test_toolchain_covers_the_model_libraries() -> None:
    toolchain = dict(cc.toolchain_digest())
    for pkg in ("torch", "triton", "diffusers", "transformers"):
        assert pkg in toolchain, pkg


# ---------------------------------------------------------------------------
# Kept trust pieces: pgw#711 publish digests + pgw#712 fence
# ---------------------------------------------------------------------------


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


def test_marked_cell_never_republishes(monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path) -> None:
    def _post(url: str, **kwargs: Any) -> Any:  # pragma: no cover - fenced
        raise AssertionError("fence must fire before any hub call")

    import requests

    monkeypatch.setattr(requests, "post", _post)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="")
    meta = {"cell_key": "ck5-" + "a" * 56, fc.ADOPTION_MARK: ["foreign"]}
    with pytest.raises(fc.CellPublishRefused, match="pgw#712"):
        pub.publish(FAMILY, artifact, meta)


def test_publish_complete_carries_only_what_the_hub_decodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    posts: list = []
    key = "ck5-" + "c" * 56

    class _FakeResp:
        def __init__(self, code: int, body: Dict[str, Any]) -> None:
            self.status_code = code
            self.text = json.dumps(body)

        def json(self) -> Dict[str, Any]:
            return json.loads(self.text)

    def _post(url: str, headers: Any = None, json: Any = None,
              timeout: Any = None) -> Any:
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

        def publish_v2(self, **kw: Any) -> Any:
            class _R:
                checkpoint_id = "cp-1"
                revision_id = "pub-1"
                uploaded = 1
                deduped = 0
                total_bytes = 10

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
    # pgw#807: `artifact_digest`/`manifest_digest` are GONE. The hub's
    # publish-complete route decodes family, cell_key, checkpoint_id, ok and
    # error — nothing else — so the SDK was paying a whole-artifact blake3
    # pass to send two fields no reader had, and the delta-1 seam refuses
    # unlisted body keys outright.
    assert set(body) == {"family", "cell_key", "checkpoint_id", "ok"}
