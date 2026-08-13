"""Recipe identity (Paul's exact-identity ruling) + kept trust pieces.

The key IS the recipe (pgw#1059): graph x envelope x sm x toolchain — the
traced computation, its declared serving region, the GPU architecture, and
the compiler stack as we configure it (binaries + settings declaration).
No version axes, no relaxable axes, no cross-key candidates — a recipe
change strands old cells by design. Kept from the equivalence arc as TRUST
(not versioning): pgw#711 publish digests, pgw#712 no-republish fence,
pgw#710 toolchain digests. Plus the closure-completeness mint gate:
executed ⊆ static, fail-loud naming the module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

# pgw#1181: `guard_closure.MANIFEST_KEY` went with `closure_manifest`, the
# only writer of this block, when the `torch-inductor-cache` format was
# deleted. The BLOCK NAME stays spelled out here because
# `fleet_cells._UNBOUNDED_ENVELOPE_BLOCKS` still lists it as a literal:
# the control-plane cap is a defensive filter over whatever an envelope
# carries, and what these rows prove — that an unbounded block is dropped
# before the hub sees it, and that a 200 MB cell still publishes — is a
# property of the CAP, not of any one producer.
GUARD_MANIFEST_BLOCK = "guard_manifest"


torch = pytest.importorskip("torch")

from gen_worker import cell_key as ck
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells as fc
from gen_worker import guard_closure as gc
from gen_worker.registry import CompileCell
from harness.cell_meta import exported_cell_meta

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


# ---------------------------------------------------------------------------
# ck1: the recipe digest
# ---------------------------------------------------------------------------


def test_ek1_axes_are_the_recipe(pinned_runtime: None,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    meta = exported_cell_meta()
    key = ck.from_entry_metadata(meta)
    assert key.digest.startswith("ek1-")
    axes = key.axes_dict()
    assert set(axes) == {"graph", "sm", "toolchain"}
    # Version strings and image identity are GONE from the key: a
    # version-string bump alone can never re-key a cell (content digests
    # decide) — they are not even inputs to the derivation.
    bumped = dict(meta, gen_worker="99.0.0", torch="9.9.9",
                  image_digest="sha256:other")
    assert ck.from_entry_metadata(bumped).digest == key.digest
    # Foreign schemes can never collide with a current key; they stay
    # key-SHAPED (pgw#990 — is_key mirrors tensorhub's scheme-agnostic
    # IsCellKey) and are ruled on by axes, not by their label.
    for dead in ("ek2-", "ek3-", "ek4-", "ek5-", "ek6-"):
        assert ck.is_key(dead + "a" * 56)
        assert key.digest != dead + "a" * 56
    # pgw#1176: a ``ck`` key is NOT merely a foreign scheme — it names a
    # 36-entry all-or-nothing cell this runtime cannot arm at all, so it does
    # not even parse. That is what makes an orphaned ref fail at the
    # comparison rather than late, inside a per-entry code path.
    # fence-symbol-exempt: `ck1` is the SUPERSEDED scheme and naming it IS the
    # assertion — the sixth instance of a blanket rename eating the one line
    # whose job is to name the old vocabulary. Do not sweep this.
    assert not ck.is_key("ck1-" + "a" * 56)
    # Version-string axes are rejected outright.
    with pytest.raises(ck.CellKeyError):
        ck.from_axes(dict(axes, torch="2.13.0"))


def test_recipe_change_changes_the_key(pinned_runtime: None) -> None:
    meta = exported_cell_meta()
    base = ck.from_entry_metadata(meta).digest
    # Toolchain content change -> new identity.
    retooled = json.loads(json.dumps(meta))
    retooled["toolchain"]["torch"] = "f" * 16
    assert ck.from_entry_metadata(retooled).digest != base
    # A deliberate settings-declaration change re-keys THROUGH toolchain
    # (pgw#1059 amendment 4) — the axis it honestly belongs to.
    reconfigured = json.loads(json.dumps(meta))
    reconfigured["toolchain"]["settings_declaration"] = "e" * 16
    assert ck.from_entry_metadata(reconfigured).digest != base


def test_metadata_roundtrips_the_recipe_key(pinned_runtime: None) -> None:
    """Mint stamp == publish recompute, from the recorded recipe blocks —
    never trusted as a stamp."""
    meta = exported_cell_meta()
    want = ck.from_entry_metadata(meta)
    assert meta["cell_key"] == want.digest
    # A cell with no toolchain block has no recipe identity.
    legacy = {k: v for k, v in meta.items() if k != "toolchain"}
    with pytest.raises(ck.CellKeyError, match="recipe"):
        ck.from_entry_metadata(legacy)
    # pgw#990's memo half is GONE with its subject (pgw#1181): the local JIT
    # kind that recorded `code_closure` and carried no `cell_key` was the
    # `torch-inductor-cache` artifact, and there is no longer any kind without
    # a key. What survives is the statement below — on the exported kind,
    # which is the only kind.
    # A TOOLCHAIN drift is identity on the exported kind.
    retooled = json.loads(json.dumps(meta))
    retooled["toolchain"]["torch"] = "0" * 16
    assert ck.from_entry_metadata(retooled).digest != want.digest


def test_toolchain_covers_the_compiler_and_not_the_model_libraries() -> None:
    """pgw#1050 INVERTS this test. It used to demand ``diffusers`` and
    ``transformers`` in the axis; their whole effect on a cell arrives through
    the traced ``graph`` axis, so folding them here re-keyed the fleet on every
    model-library bump for a computation that had not moved. Membership is the
    compiler — see ``tests/test_toolchain_membership_pgw1050.py``."""
    toolchain = dict(cc.toolchain_digest())
    for pkg in ("torch", "triton"):
        assert pkg in toolchain, pkg
    for pkg in ("diffusers", "transformers", "peft"):
        assert pkg not in toolchain, pkg


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
        "posture": dict(gc.CANONICAL_POSTURE),
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
    meta = {"cell_key": "ek1-" + "a" * 56, fc.ADOPTION_MARK: ["foreign"]}
    with pytest.raises(fc.CellPublishRefused, match="pgw#712"):
        pub.publish(FAMILY, artifact, meta)


def test_publish_complete_carries_only_what_the_hub_decodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    posts: list = []
    # pgw#1046: a real exported-cell envelope — publish recomputes the key from
    # the recorded blocks and refuses a cell that cannot state one.
    meta = exported_cell_meta(family=FAMILY, sku="l4", gen_worker="1.0.0",
                              **{GUARD_MANIFEST_BLOCK: _manifest()})
    key = meta["cell_key"]

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

    import gen_worker.hubio.client as hub_mod

    monkeypatch.setattr(hub_mod, "HubClient", _FakeHub)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"cell-bytes")
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
