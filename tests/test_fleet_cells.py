"""gw#587 fleet self-mint: arming policy outcomes + the publish protocol.

The torch.compile capture itself needs a GPU + toolchain and is proven live
(the gw#587 live proof); these tests pin the POLICY around it: delivered
cell first, self-mint on miss, publish best-effort and never load-bearing
for serving, the cell_selection_bug receipt invariant untouched, and the
typed quantized refusal only at genuine mint impossibilities.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from gen_worker import compile_cache as cc
from gen_worker import fleet_cells as fc
from gen_worker.models import provision


class _Cfg:
    def __init__(self, family="fam", shapes=((64, 64),), targets=("transformer",)):
        self.family = family
        self.shapes = tuple(tuple(s) for s in shapes)
        self.targets = tuple(targets)
        self.regional = False
        self.guidance_scales = ()
        self.lora_bucket = 0


class _Pipe:
    _cozy_low_vram_mode = "off"


def _publisher(calls):
    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta):
            calls.append((family, Path(artifact), dict(meta)))
            return "cp-1"

    return _Pub(base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="sha256:img")


# ---------------------------------------------------------------------------
# arming policy
# ---------------------------------------------------------------------------


def test_delivered_cell_hit_never_mints_or_publishes(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(provision, "enable_compiled", lambda *a, **k: True)
    monkeypatch.setattr(
        cc, "mint_artifact",
        lambda *a, **k: pytest.fail("HIT must never mint"))
    assert fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(calls))
    assert calls == []


def test_cell_selection_bug_propagates_untouched(monkeypatch, tmp_path):
    """th#883 receipt invariant: a self-requested, identity-verified cell
    that refuses to arm is a BUG — self-mint must never mask it by
    re-minting over it. Revert-turns-red for the invariant."""

    def _raise(*a, **k):
        raise cc.CellSelectionBugError("self-requested cell refused to arm")

    monkeypatch.setattr(provision, "enable_compiled", _raise)
    monkeypatch.setattr(
        cc, "mint_artifact",
        lambda *a, **k: pytest.fail("selection bug must never trigger a mint"))
    with pytest.raises(cc.CellSelectionBugError):
        fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None, publisher=_publisher([]))


def test_miss_mints_adopts_and_publishes(monkeypatch, tmp_path):
    """The core gw#587 outcome: MISS -> local mint -> serve compiled ->
    publish attempt (with the mint's own metadata)."""
    calls: list = []
    published = threading.Event()

    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta):
            calls.append((family, Path(artifact), dict(meta)))
            published.set()
            return "cp-1"

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="sha256:img")
    monkeypatch.setattr(provision, "enable_compiled", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)

    def _fake_mint(pipe, cfg, family, target, capture, **kw):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"cell-bytes")
        return {"cell_key": "ck1-" + "a" * 56, "sku": "b200", "gen_worker": "0.39.0"}

    monkeypatch.setattr(cc, "mint_artifact", _fake_mint)
    monkeypatch.setattr(cc, "unwrap", lambda pipe: None)
    monkeypatch.setattr(cc, "enable", lambda *a, **k: True)

    assert fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None, publisher=pub)
    assert published.wait(5), "publish attempt must be recorded on every miss"
    (family, artifact, meta) = calls[0]
    assert family == "fam"
    assert meta["cell_key"].startswith("ck1-")


def test_publish_failure_never_affects_serving(monkeypatch, tmp_path):
    """The request that triggered the miss is served from the local mint even
    when the hub refuses the publish (untrusted tier / forged axis / quota)."""
    refused = threading.Event()

    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta):
            refused.set()
            raise fc.CellPublishRefused("cell_publish_untrusted_tier: community_tier")

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="d")
    monkeypatch.setattr(provision, "enable_compiled", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)

    def _fake_mint(pipe, cfg, family, target, capture, **kw):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"cell-bytes")
        return {"cell_key": "ck1-" + "b" * 56}

    monkeypatch.setattr(cc, "mint_artifact", _fake_mint)
    monkeypatch.setattr(cc, "unwrap", lambda pipe: None)
    monkeypatch.setattr(cc, "enable", lambda *a, **k: True)

    assert fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None, publisher=pub)
    assert refused.wait(5)


def test_mint_impossible_keeps_quantized_typed_refusal(monkeypatch, tmp_path):
    """No CUDA => plain lanes serve eager (False), quantized lanes keep the
    typed fail-closed refusal — never a silent slow eager serve."""
    monkeypatch.setattr(provision, "enable_compiled", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_cuda_ready", lambda: False)

    plain = _Pipe()
    assert fc.enable_compiled(plain, _Cfg(), tmp_path, None, publisher=None) is False

    w8a8 = _Pipe()
    setattr(w8a8, "_cozy_weight_lane", "w8a8")
    monkeypatch.setattr(
        "gen_worker.models.loading.pipeline_weight_lane", lambda p: "w8a8")
    with pytest.raises(cc.CompiledLaneUnavailableError, match="self-mint is unavailable"):
        fc.enable_compiled(w8a8, _Cfg(), tmp_path, None, publisher=None)


# ---------------------------------------------------------------------------
# publish protocol (intent -> commit -> complete)
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self):
        return json.loads(self.text)


def test_publisher_drives_intent_commit_complete(monkeypatch, tmp_path):
    posts: list = []
    key = "ck1-" + "c" * 56

    def _post(url, headers=None, json=None, timeout=None):
        posts.append((url, json))
        if url.endswith("/publish-intent"):
            return _FakeResp(200, {
                "capability_token": "cap-token",
                "repo": "_system/family-fam",
                "cell_key": key,
            })
        return _FakeResp(200, {"recorded": True})

    import requests

    monkeypatch.setattr(requests, "post", _post)

    committed: list = []

    class _FakeHub:
        def __init__(self, **kw):
            committed.append(("client", kw))

        def commit(self, **kw):
            committed.append(("commit", kw))

            class _R:
                checkpoint_id = "cp-42"

            return _R()

    import gen_worker.convert.hub as hub_mod

    monkeypatch.setattr(hub_mod, "HubClient", _FakeHub)

    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "worker-jwt",
        image_digest="sha256:img")
    meta = {"cell_key": key, "sku": "b200", "gen_worker": "0.39.0"}
    assert pub.publish("fam", artifact, meta) == "cp-42"

    intent_url, intent_body = posts[0]
    assert intent_url.endswith("/v1/worker/cells/publish-intent")
    # The claimed axes the hub will attest.
    assert intent_body["axes"] == {
        "sku": "b200", "image_digest": "sha256:img", "gen_worker": "0.39.0"}
    assert intent_body["cell_key"] == key

    kind, kw = committed[0]
    assert kind == "client" and kw["token"] == "cap-token"
    kind, kw = committed[1]
    assert kind == "commit"
    assert kw["destination_repo"] == "_system/family-fam"
    assert kw["mode"] == "replace"
    assert kw["flavor"] == key
    assert "tags" not in kw  # a cell publish never binds tags

    complete_url, complete_body = posts[-1]
    assert complete_url.endswith("/v1/worker/cells/publish-complete")
    assert complete_body["ok"] is True and complete_body["checkpoint_id"] == "cp-42"


def test_publisher_typed_refusal_is_terminal(monkeypatch, tmp_path):
    def _post(url, headers=None, json=None, timeout=None):
        return _FakeResp(403, {"error": "cell_publish_forged_axis", "message": "axis=sku"})

    import requests

    monkeypatch.setattr(requests, "post", _post)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "worker-jwt", image_digest="d")
    with pytest.raises(fc.CellPublishRefused, match="cell_publish_forged_axis"):
        pub.publish("fam", artifact, {"cell_key": "ck1-" + "d" * 56})


def test_publisher_reports_commit_failure(monkeypatch, tmp_path):
    """A failed commit still files publish-complete ok=false so the hub's
    ledger/alarms record the outcome (miss => publish attempt recorded)."""
    posts: list = []

    def _post(url, headers=None, json=None, timeout=None):
        posts.append((url, json))
        if url.endswith("/publish-intent"):
            return _FakeResp(200, {
                "capability_token": "cap", "repo": "_system/family-fam"})
        return _FakeResp(200, {"recorded": True})

    import requests

    monkeypatch.setattr(requests, "post", _post)

    class _FakeHub:
        def __init__(self, **kw):
            pass

        def commit(self, **kw):
            raise RuntimeError("upload exploded")

    import gen_worker.convert.hub as hub_mod

    monkeypatch.setattr(hub_mod, "HubClient", _FakeHub)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "worker-jwt", image_digest="d")
    with pytest.raises(RuntimeError, match="upload exploded"):
        pub.publish("fam", artifact, {"cell_key": "ck1-" + "e" * 56})
    complete_url, complete_body = posts[-1]
    assert complete_url.endswith("/publish-complete")
    assert complete_body["ok"] is False and "upload exploded" in complete_body["error"]
