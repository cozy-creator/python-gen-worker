"""gw#587 fleet self-mint: arming policy outcomes + the publish protocol.

The torch.compile capture itself needs a GPU + toolchain and is proven live
(the gw#587 live proof); these tests pin the POLICY around it: delivered
cell first, PROVE-PRODUCES-THE-MINT on miss (arm cold for capture; only
the executor's passed warmup proof packs + publishes — the gw#586-class
fix: no synthetic producer warm loop exists in the fleet path anymore),
publish best-effort and never load-bearing for serving, the
cell_selection_bug receipt invariant untouched, and the typed quantized
refusal only at genuine mint impossibilities.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from gen_worker import cell_key
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


FAKE_KEY = "ck1-" + "a" * 56


@pytest.fixture(autouse=True)
def _clear_pending():
    with fc._PENDING_LOCK:
        fc._PENDING.clear()
    yield
    with fc._PENDING_LOCK:
        fc._PENDING.clear()


def _mintable(monkeypatch, *, key=FAKE_KEY):
    """Route a MISS into the cold-capture arm with no CUDA/toolchain."""
    monkeypatch.setattr(provision, "enable_compiled", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)

    class _Key:
        digest = key

    monkeypatch.setattr(cell_key, "compute", lambda *a, **k: _Key())

    def _begin(pipe, cfg, capture):
        # the real begin_fleet_mint latches env + arms cold; the capture
        # content itself arrives during the (real, GPU) warmup — simulate
        # the layout the proof window would produce.
        (capture / "inductor" / "g").mkdir(parents=True, exist_ok=True)
        (capture / "inductor" / "g" / "kernel.py").write_text("compiled")
        (capture / "triton").mkdir(exist_ok=True)

    monkeypatch.setattr(cc, "begin_fleet_mint", _begin)


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
        cc, "begin_fleet_mint",
        lambda *a, **k: pytest.fail("HIT must never open a mint capture"))
    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(calls))
    assert outcome and outcome.self_mint is None
    assert calls == []


def test_cell_selection_bug_propagates_untouched(monkeypatch, tmp_path):
    """th#883 receipt invariant: a self-requested, identity-verified cell
    that refuses to arm is a BUG — self-mint must never mask it by
    re-minting over it. Revert-turns-red for the invariant."""

    def _raise(*a, **k):
        raise cc.CellSelectionBugError("self-requested cell refused to arm")

    monkeypatch.setattr(provision, "enable_compiled", _raise)
    monkeypatch.setattr(
        cc, "begin_fleet_mint",
        lambda *a, **k: pytest.fail("selection bug must never open a capture"))
    with pytest.raises(cc.CellSelectionBugError):
        fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None, publisher=_publisher([]))


def test_miss_arms_pending_capture_without_packing_or_publishing(
    monkeypatch, tmp_path,
):
    """gw#587 CORRECT FIX direction (b): a MISS opens a cold capture and
    returns a PENDING mint — nothing is packed, nothing is published, and
    no synthetic warm call runs (that separate producer-shaped execution
    was the gw#586-class defect the live proof caught). Publish before the
    proof reverts this test red."""
    calls: list = []
    _mintable(monkeypatch)
    monkeypatch.setattr(
        cc, "mint_artifact",
        lambda *a, **k: pytest.fail(
            "the fleet miss path must never run the producer warm loop"))

    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(calls))
    assert outcome.armed
    pending = outcome.self_mint
    assert isinstance(pending, fc.PendingSelfMint)
    assert pending.cell_key == FAKE_KEY
    assert pending.ref == f"root/family-fam#{FAKE_KEY}"
    assert not pending.target.exists(), "nothing packed before the proof"
    assert calls == [], "nothing published before the proof"


def test_finalize_packs_the_proven_capture_and_publishes_it(
    monkeypatch, tmp_path,
):
    """gw#587 CORRECT FIX direction (a): after the proof passes, finalize
    packs EXACTLY the capture the proof window populated and publishes
    those bytes; the advertised digest is the digest of that packed
    artifact."""
    calls: list = []
    published = threading.Event()

    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta):
            calls.append((family, artifact.read_bytes(), dict(meta)))
            published.set()
            return "cp-1"

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt",
               image_digest="sha256:img")
    _mintable(monkeypatch)
    pipe = _Pipe()
    outcome = fc.enable_compiled(pipe, _Cfg(), tmp_path, None, publisher=pub)
    pending = outcome.self_mint
    assert isinstance(pending, fc.PendingSelfMint)

    minted = fc.finalize_self_mint(pipe, pending)
    assert minted is not None
    assert minted.cell_key == FAKE_KEY
    assert minted.ref == f"root/family-fam#{FAKE_KEY}"
    assert minted.snapshot_digest.startswith("blake3:")
    assert len(minted.snapshot_digest) == len("blake3:") + 64
    assert published.wait(5), "a finalized mint must attempt publish"
    (family, tar_bytes, meta) = calls[0]
    assert family == "fam"
    # The published bytes ARE the packed proven capture, and the advertised
    # digest is the digest of exactly those bytes.
    from gen_worker.convert.hub import blake3_file

    copy = tmp_path / "published-copy.tar.gz"
    copy.write_bytes(tar_bytes)
    assert minted.snapshot_digest == "blake3:" + blake3_file(copy)
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        names = tar.getnames()
    assert any("kernel.py" in n for n in names), (
        "published cell must contain the capture the proof produced")
    # Finalize is memoized for same-key siblings: no double pack/publish.
    assert fc.finalize_self_mint(pipe, pending) is minted
    assert len(calls) == 1


def test_abandon_never_publishes(monkeypatch, tmp_path):
    """A capture whose proof did not certify it is abandoned: nothing
    packed, nothing published, temp dir removed."""
    calls: list = []
    _mintable(monkeypatch)
    pipe = _Pipe()
    outcome = fc.enable_compiled(
        pipe, _Cfg(), tmp_path, None, publisher=_publisher(calls))
    pending = outcome.self_mint
    fc.abandon_self_mint(pending)
    assert calls == []
    assert not pending.mint_root.exists()
    with fc._PENDING_LOCK:
        assert fc._PENDING == {}


def test_same_key_sibling_joins_the_pending_capture(monkeypatch, tmp_path):
    """Two pipes of one record computing the same key share ONE capture
    (the union family cell); a second mint_root is never created."""
    _mintable(monkeypatch)
    a, b = _Pipe(), _Pipe()
    first = fc.enable_compiled(a, _Cfg(), tmp_path, None).self_mint
    second = fc.enable_compiled(b, _Cfg(), tmp_path, None).self_mint
    assert second is first


def test_publish_failure_never_affects_serving(monkeypatch, tmp_path):
    """The request that triggered the miss is served from the proven local
    capture even when the hub refuses the publish (untrusted tier / forged
    axis / quota)."""
    refused = threading.Event()

    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta):
            refused.set()
            raise fc.CellPublishRefused("cell_publish_untrusted_tier: community_tier")

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="d")
    _mintable(monkeypatch)
    pipe = _Pipe()
    outcome = fc.enable_compiled(pipe, _Cfg(), tmp_path, None, publisher=pub)
    minted = fc.finalize_self_mint(pipe, outcome.self_mint)
    assert minted is not None, "hub refusal must never fail the finalize"
    assert refused.wait(5)


def test_mint_impossible_keeps_quantized_typed_refusal(monkeypatch, tmp_path):
    """No CUDA => plain lanes serve eager (False), quantized lanes keep the
    typed fail-closed refusal — never a silent slow eager serve."""
    monkeypatch.setattr(provision, "enable_compiled", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_cuda_ready", lambda: False)

    plain = _Pipe()
    outcome = fc.enable_compiled(plain, _Cfg(), tmp_path, None, publisher=None)
    assert outcome.armed is False and outcome.self_mint is None

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
                "repo": "root/family-fam",
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
    assert kw["destination_repo"] == "root/family-fam"
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
                "capability_token": "cap", "repo": "root/family-fam"})
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
