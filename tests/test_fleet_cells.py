"""gw#587 fleet self-mint: arming policy outcomes + the publish protocol.

The mint itself needs a GPU + toolchain and is proven live (the gw#587 live
proof, and the micro rig since pgw#978); these tests pin the POLICY around it:
delivered cell first, a DELEGATED AOT mint on a miss (the live pipe untouched,
serving eager, the obligation on the pending), publish best-effort and never
load-bearing for serving, the cell_selection_bug receipt invariant untouched,
and the typed quantized refusal only at genuine mint impossibilities.

pgw#1010 re-aimed this module. What it used to cover — arm cold into a capture
dir, pack the proven capture, publish the packed bytes — was the DYNAMO cell
path, and a dynamo cell has no consumer (`aot_cells` adopts `aot-inductor`
only, by name), so it is deleted rather than kept. A miss on a family with no
export declaration now serves JIT INTAKE and produces nothing; that half is
covered by `test_dynamo_no_publish_pgw1010.py`.

`mint_recipe` is stubbed to `RECIPE_AOT` where a delegated pending is the
subject: WHICH recipe a miss runs has its own coverage
(`test_aot_flip_pgw722.py`, `test_mint_recipe_parity_pgw984_pgw985.py`), and a
test box registers no export declarations at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from harness.cell_meta import exported_cell_meta

from gen_worker import compile_cache as cc
from gen_worker import fleet_cells as fc
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.models import provision


class _Cfg:
    def __init__(self, family="fam", shapes=((64, 64),), targets=("transformer",)):
        self.family = family
        self.shapes = tuple(tuple(s) for s in shapes)
        self.targets = tuple(targets)
        self.regional = False
        self.guidance_scales = ()
        self.lora_bucket = 0


class _Denoiser:
    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        return None


class _Pipe:
    _cozy_low_vram_mode = "off"

    def __init__(self):
        # A resolvable compile target: the gw#608 gate declines the
        # self-mint (BEFORE any process-global cache-dir mutation) for
        # slot objects that could never arm.
        self.transformer = _Denoiser()


FAKE_KEY = "cg-key-v1-" + "a" * 56


def test_finalized_mint_speaks_compiled_graph_vocabulary(tmp_path: Path) -> None:
    minted = fc.SelfMint(
        family="fam",
        compiled_graph_key=FAKE_KEY,
        ref=f"root/family-fam#{FAKE_KEY}",
        snapshot_digest="sha256:" + "0" * 64,
        artifact=tmp_path / "graph.tgz",
    )

    assert minted.compiled_graph_key == FAKE_KEY
    assert not hasattr(minted, "cell_key")


@pytest.fixture(autouse=True)
def _clear_pending():
    with fc._PENDING_LOCK:
        fc._PENDING.clear()
    yield
    with fc._PENDING_LOCK:
        fc._PENDING.clear()


def _mintable(monkeypatch, *, key=FAKE_KEY):
    """Route a MISS into the delegated AOT mint with no CUDA/toolchain."""
    monkeypatch.setattr(
        provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fc, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    # pgw#1181: the pgw#681 gate that used to be simmed here is gone.
    # `guard_closure.closure_manifest` ran at the MINT, classified every
    # compiled graph and wrote the result into the cell's metadata; it is
    # deleted with the `torch-inductor-cache` format that carried it, so a rig
    # whose compiles never touch dynamo no longer has a gate to satisfy.

    class _Arm:
        token = key

        def facts_dict(self):
            return {}

    monkeypatch.setattr(fc, "arm_identity", lambda *a, **k: _Arm())
    # A test box registers no export declarations, so the real `mint_recipe`
    # would decline every miss to JIT intake. The recipe DECISION is covered
    # elsewhere; what is under test here is what the policy does once the
    # answer is "AOT".
    monkeypatch.setattr(
        fc, "mint_recipe", lambda *a, **k: fc.RECIPE_AOT)
    monkeypatch.setattr(
        cc, "arm_jit_intake",
        lambda *a, **k: pytest.fail(
            "an AOT miss must not arm JIT intake on the live pipe"))


def _publisher(calls):
    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta, mint_duration_ms=0):
            calls.append((family, Path(artifact), dict(meta), int(mint_duration_ms)))
            return "cp-1"

    return _Pub(base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="sha256:img")


# ---------------------------------------------------------------------------
# arming policy
# ---------------------------------------------------------------------------


_ADOPT_META = exported_cell_meta(family="fam")


def _adopted(monkeypatch, pending):
    """Drive the pending to ADOPTED the way the delegated driver does: the
    child wrote one TCG artifact, the one arm seam accepted its exact key, and
    `adopt_delegated_mint` records the durable object. TCG admission and real
    binding have their own integration tests; this helper isolates fleet
    obligation/publish policy without reviving the deleted artifact format."""
    child = pending.mint_root / "compiled-graph.tar.gz"
    child.parent.mkdir(parents=True, exist_ok=True)
    child.write_bytes(b"opaque-tcg-artifact")
    key = str(_ADOPT_META["compiled_graph_key"])
    monkeypatch.setattr(fc, "_stage_durable", lambda *_a, **_k: key)
    monkeypatch.setattr(
        fc,
        "_arm_compiled_graph",
        lambda *_a, **_k: (True, dict(_ADOPT_META), ("", "")),
    )
    monkeypatch.setattr(
        fc,
        "_admit_durable",
        lambda *_a, **_k: SimpleNamespace(artifact=child),
    )
    return fc.adopt_delegated_mint(_Pipe(), pending, [child])


def test_delivered_cell_hit_never_mints_or_publishes(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(
        provision, "enable_compiled", lambda *a, **k: AdoptOutcome.hit())
    monkeypatch.setattr(
        cc, "arm_jit_intake",
        lambda *a, **k: pytest.fail("HIT must never arm anything else"))
    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(calls))
    assert outcome and outcome.self_mint is None
    assert calls == []


def test_cell_selection_bug_reports_and_self_mints(monkeypatch, tmp_path):
    """th#1031: a self-requested, identity-verified cell that refuses to
    arm is still a BUG — reported to the caller via ``ArmOutcome.
    selection_bug`` (unchanged wire visibility) — but no longer aborts
    arming. A worker stuck retrying an exact-key artifact that still cannot
    arm is worse than
    identical unusable cell forever is worse than a loud report + recovery.
    This call falls through to self-mint exactly like an ordinary miss."""
    bug = cc.CellSelectionBugError("self-requested cell refused to arm")

    def _raise(*a, **k):
        raise bug

    _mintable(monkeypatch)
    monkeypatch.setattr(provision, "enable_compiled", _raise)

    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher([]))
    assert not outcome.armed, "the pipe serves eager while the child mints"
    assert isinstance(outcome.self_mint, fc.PendingSelfMint)
    assert outcome.selection_bug is bug


def test_cell_selection_bug_still_fail_closed_when_mint_impossible(
    monkeypatch, tmp_path,
):
    """A caught selection bug does not weaken the quantized-lane refusal at
    a GENUINE mint impossibility (no CUDA here) — the typed refusal still
    raises, chained from the selection bug so the report is never dropped."""

    def _raise(*a, **k):
        raise cc.CellSelectionBugError("self-requested cell refused to arm")

    monkeypatch.setattr(provision, "enable_compiled", _raise)
    monkeypatch.setattr(fc, "_cuda_ready", lambda: False)

    class _W8A8Pipe(_Pipe):
        pass

    monkeypatch.setattr(
        "gen_worker.models.loading.pipeline_weight_lane", lambda pipe: "w8a8")
    with pytest.raises(cc.CompiledExecutionLaneUnavailableError) as exc:
        fc.enable_compiled(
            _W8A8Pipe(), _Cfg(), tmp_path, None, publisher=_publisher([]))
    assert isinstance(exc.value.__cause__, cc.CellSelectionBugError)


def test_miss_opens_a_delegated_mint_without_packing_or_publishing(
    monkeypatch, tmp_path,
):
    """A MISS hands the mint to a child and returns a PENDING — the live pipe
    is untouched, nothing is packed, nothing is published. Publishing before
    the child's cell adopts reverts this test red."""
    calls: list = []
    _mintable(monkeypatch)
    # pgw#1178 deleted `compile_cache.mint_artifact`, the producer warm loop
    # this used to fence against by name. The property is now structural —
    # there is no producer on this path to run — and the assertions below
    # (nothing packed, nothing published) are what state it.

    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(calls))
    assert not outcome.armed
    pending = outcome.self_mint
    assert isinstance(pending, fc.PendingSelfMint)
    assert pending.delegated is True
    assert pending.arm_token == FAKE_KEY
    assert pending.ref == f"root/family-fam#{FAKE_KEY}"
    assert not pending.target.exists(), "nothing packed before the child runs"
    assert calls == [], "nothing published before the cell adopts"


def test_adopt_publishes_exactly_the_bytes_that_armed(monkeypatch, tmp_path):
    """The cell that ADOPTED is the cell that ships, and the advertised digest
    is the digest of exactly those bytes (gw#587's direction (a), delegated)."""
    calls: list = []
    published = threading.Event()

    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta, mint_duration_ms=0):
            calls.append((family, artifact.read_bytes(), dict(meta),
                          int(mint_duration_ms)))
            published.set()
            return "cp-1"

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt",
               image_digest="sha256:img")
    _mintable(monkeypatch)
    outcome = fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None, publisher=pub)
    pending = outcome.self_mint
    assert isinstance(pending, fc.PendingSelfMint)

    minted = _adopted(monkeypatch, pending)
    assert minted is not None
    assert calls == [], "adoption arms; only the coverage gate publishes"
    fc.publish_self_mint(pending)
    assert minted.ref == f"root/family-fam#{minted.compiled_graph_key}"
    assert minted.snapshot_digest.startswith("sha256:")
    assert len(minted.snapshot_digest) == len("sha256:") + 64
    assert published.wait(5), "an adopted mint must attempt publish"
    (family, tar_bytes, meta, mint_duration_ms) = calls[0]
    assert family == "fam"
    # th#1355: the mint cost travels with the publish, so "what did this cell
    # cost to build" can be joined to the cell it describes.
    assert mint_duration_ms >= 0
    copy = tmp_path / "published-copy.tar.gz"
    copy.write_bytes(tar_bytes)
    assert minted.snapshot_digest == "sha256:" + hashlib.sha256(tar_bytes).hexdigest()
    # Adoption is memoized for same-key siblings: publish resolves once.
    #
    # Asserted by calling the adopt DIRECTLY on the same child path, rather
    # than re-staging bytes through `_adopted`. Two reasons, and the second is
    # why this line used to fail under load: a memo that short-circuits never
    # opens the file, so not touching it is the stronger claim — and
    # `publish_self_mint` above has already `rmtree`'d `pending.mint_root` on
    # its own thread, so `_adopted`'s mkdir-then-write raced that reaper and
    # lost with a FileNotFoundError whenever the publish path got there first.
    assert fc.adopt_delegated_mint(
        _Pipe(), pending,
        [pending.mint_root / "child-cell.tar.gz"]) is minted
    fc.publish_self_mint(pending)
    assert len(calls) == 1


def test_abandon_never_publishes(monkeypatch, tmp_path):
    """A mint whose child produced nothing adoptable is abandoned: nothing
    published, temp dir removed."""
    calls: list = []
    _mintable(monkeypatch)
    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(calls))
    pending = outcome.self_mint
    fc.abandon_self_mint(pending)
    assert calls == []
    assert not pending.mint_root.exists()
    with fc._PENDING_LOCK:
        assert fc._PENDING == {}


def test_same_key_sibling_shares_the_pending_mint_root(monkeypatch, tmp_path):
    """Two pipes of one record computing the same key share ONE mint (the
    union family cell); a second mint_root is never created."""
    _mintable(monkeypatch)
    first = fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None).self_mint
    second = fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None).self_mint
    assert second.mint_root == first.mint_root
    assert second.target == first.target


def test_publish_failure_never_affects_serving(monkeypatch, tmp_path):
    """The pipe that triggered the miss serves from its adopted cell even
    when the hub refuses the publish (untrusted tier / forged axis / quota)."""
    refused = threading.Event()

    class _Pub(fc.CellPublisher):
        def publish(self, family, artifact, meta, mint_duration_ms=0):
            refused.set()
            raise fc.CellPublishRefused(
                "compiled_graph_publish_untrusted_tier: community_tier"
            )

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt", image_digest="d")
    _mintable(monkeypatch)
    outcome = fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None, publisher=pub)
    minted = _adopted(monkeypatch, outcome.self_mint)
    assert minted is not None, "hub refusal must never fail the adoption"
    fc.publish_self_mint(outcome.self_mint)
    assert refused.wait(5)


def test_withheld_publish_never_ships_and_is_final(monkeypatch, tmp_path):
    """gw#612: an incomplete family cell (a mandatory sibling not covered by
    the mint) is never published — and once the publish is resolved
    (withheld), a later publish call is a no-op."""
    calls: list = []
    _mintable(monkeypatch)
    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(calls))
    pending = outcome.self_mint
    minted = _adopted(monkeypatch, pending)
    assert minted is not None
    fc.withhold_self_mint_publish(pending, "sibling lane unexercised")
    assert calls == []
    assert not pending.mint_root.exists(), "withheld mint dir is cleaned"
    fc.publish_self_mint(pending)  # resolution is final
    assert calls == []
    # Serving state is untouched: the finalized identity stays memoized for
    # sibling advertisement.
    assert pending._state["minted"] is minted


def test_mint_impossible_keeps_quantized_typed_refusal(monkeypatch, tmp_path):
    """No CUDA => plain lanes serve eager (False), quantized lanes keep the
    typed fail-closed refusal — never a silent slow eager serve."""
    monkeypatch.setattr(
        provision, "enable_compiled",
        lambda *a, **k: AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fc, "_cuda_ready", lambda: False)

    plain = _Pipe()
    outcome = fc.enable_compiled(plain, _Cfg(), tmp_path, None, publisher=None)
    assert outcome.armed is False and outcome.self_mint is None

    w8a8 = _Pipe()
    setattr(w8a8, "_cozy_weight_lane", "w8a8")
    monkeypatch.setattr(
        "gen_worker.models.loading.pipeline_weight_lane", lambda p: "w8a8")
    with pytest.raises(cc.CompiledExecutionLaneUnavailableError, match="self-mint is unavailable"):
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


def _granted(body, repo: str) -> dict:
    """The hub's batch answer: one grant per entry, in request order, each
    with ITS OWN token (pgw#1224 / th#1842 PR #1121)."""
    entries = (body or {}).get("entries") or []
    return {
        "object": "compiled_graph_publish_intent_batch",
        "repo": repo,
        "family": str((body or {}).get("family") or ""),
        "granted": len(entries),
        "answers": [
            {"compiled_graph_key": str(e.get("compiled_graph_key") or ""),
             "status": "granted",
             "capability_token": f"cap-token-{i}",
             "expires_at_unix": 4102444800}
            for i, e in enumerate(entries)
        ],
    }


def test_publisher_drives_intent_publish_v2_complete(monkeypatch, tmp_path):
    posts: list = []
    # pgw#1046: a real exported-cell envelope — the publish path recomputes the
    # key from its blocks, so an invented one is refused before the intent.
    meta = exported_cell_meta(sku="b200", gen_worker="0.39.0")
    key = meta["compiled_graph_key"]
    monkeypatch.setattr(cc, "runtime_key", lambda: {"sku": "b200"})
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.39.0")

    def _post(url, headers=None, json=None, timeout=None):
        posts.append((url, json))
        if url.endswith("/publish-intent"):
            return _FakeResp(200, _granted(json, "root/family-fam"))
        return _FakeResp(200, {"recorded": True})

    import requests

    monkeypatch.setattr(requests, "post", _post)

    committed: list = []

    class _FakeHub:
        def __init__(self, **kw):
            committed.append(("client", kw))

        def commit(self, **kw):  # pragma: no cover - the frozen v1 route
            raise AssertionError(
                "the cell publisher must never call the v1 (blake3) commit "
                "route: it is frozen hub-side and 410s (pgw#807 item 3)")

        def publish_v2(self, **kw):
            committed.append(("publish_v2", kw))
            on_stage = kw.get("on_stage")
            if on_stage is not None:
                on_stage("declared", {"publish_id": "pub-1", "need": 1})
                on_stage("uploading", {"publish_id": "pub-1", "objects": 1})

            class _R:
                checkpoint_id = "cp-42"
                revision_id = "pub-1"
                uploaded = 1
                deduped = 0
                total_bytes = 5

            return _R()

    import gen_worker.hubio.client as hub_mod

    monkeypatch.setattr(hub_mod, "HubClient", _FakeHub)

    events: list = []
    monkeypatch.setattr(
        fc.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: events.append((kind, phase)))

    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "worker-jwt",
        image_digest="sha256:img")
    assert pub.publish("fam", artifact, meta) == "cp-42"

    intent_url, intent_body = posts[0]
    assert intent_url.endswith("/v1/worker/compiled-graphs/publish-intent")
    # The claimed axes the hub will attest.
    assert intent_body["axes"] == {
        "sku": "b200", "image_digest": "sha256:img", "gen_worker": "0.39.0"}
    # pgw#1224: the KEY is per ENTRY, the attested axes are per BATCH. A
    # one-artifact publish is a one-entry batch — the single-entry body is gone.
    assert [e["compiled_graph_key"] for e in intent_body["entries"]] == [key]
    assert set(intent_body) == {"family", "axes", "entries"}

    kind, kw = committed[0]
    assert kind == "client" and kw["token"] == "cap-token-0"
    kind, kw = committed[1]
    assert kind == "publish_v2", "the cell publisher ships over chunked sha256"
    assert kw["destination_repo"] == "root/family-fam"
    assert kw["mode"] == "replace"
    assert "flavor" not in kw  # pgw#1159: dead hub-side, so it is not sent
    assert "tags" not in kw  # a cell publish never binds tags
    # th#1340 refuses a body that names the cell identity: it is hub-derived
    # and rides the capability token.
    for forbidden in ("compiled_graph_publish", "compiled_graph_key", "family", "owning_endpoint_id",
                      "axes", "default_flavor"):
        assert forbidden not in kw

    complete_url, complete_body = posts[-1]
    assert complete_url.endswith("/v1/worker/compiled-graphs/publish-complete")
    assert complete_body["ok"] is True and complete_body["checkpoint_id"] == "cp-42"
    # pgw#711's artifact_digest/manifest_digest are gone: the hub's
    # publish-complete route has no such fields (it decodes family, key,
    # checkpoint_id, ok, error) and the delta-1 seam refuses unlisted keys.
    assert set(complete_body) == {
        "family", "compiled_graph_key", "checkpoint_id", "ok"
    }

    # Every LEG of the publish is on the wire, not just its terminus.
    assert [p for k, p in events if k == "self_mint_publish"] == [
        "declared", "uploading", "committed"]


def test_publisher_typed_refusal_is_terminal(monkeypatch, tmp_path):
    def _post(url, headers=None, json=None, timeout=None):
        return _FakeResp(
            403,
            {
                "error": "compiled_graph_publish_forged_axis",
                "message": "axis=sku",
            },
        )

    import requests

    monkeypatch.setattr(requests, "post", _post)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "worker-jwt", image_digest="d")
    with pytest.raises(
        fc.CellPublishRefused, match="compiled_graph_publish_forged_axis"
    ):
        pub.publish("fam", artifact, exported_cell_meta())


def test_publisher_reports_commit_failure(monkeypatch, tmp_path):
    """A failed commit still files publish-complete ok=false so the hub's
    ledger/alarms record the outcome (miss => publish attempt recorded)."""
    posts: list = []

    def _post(url, headers=None, json=None, timeout=None):
        posts.append((url, json))
        if url.endswith("/publish-intent"):
            return _FakeResp(200, _granted(json, "root/family-fam"))
        return _FakeResp(200, {"recorded": True})

    import requests

    monkeypatch.setattr(requests, "post", _post)

    class _FakeHub:
        def __init__(self, **kw):
            pass

        def publish_v2(self, **kw):
            raise RuntimeError("upload exploded")

    import gen_worker.hubio.client as hub_mod

    monkeypatch.setattr(hub_mod, "HubClient", _FakeHub)
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"bytes")
    pub = fc.CellPublisher(
        base_url="http://hub", worker_jwt=lambda: "worker-jwt", image_digest="d")
    with pytest.raises(RuntimeError, match="upload exploded"):
        pub.publish("fam", artifact, exported_cell_meta())
    complete_url, complete_body = posts[-1]
    assert complete_url.endswith("/publish-complete")
    assert complete_body["ok"] is False and "upload exploded" in complete_body["error"]


def test_no_target_pipe_is_declined_before_anything_is_armed(tmp_path, monkeypatch):
    """A slot object with no resolvable compile target (the LTX upsampler
    shape) is declined by name — no intake arm, no mint, no pending.

    pgw#1010 note: this used to also assert that the process-global inductor
    cache dir was untouched, because `arm_jit_intake` re-pointed it before
    discovering there was nothing to arm (gw#608's live 8/8-miss shape).
    Nothing moves that env any more — the capture it existed for is gone — so
    what survives is the decline itself.
    """
    _mintable(monkeypatch)
    armed: list = []
    monkeypatch.setattr(cc, "arm_jit_intake", lambda *a, **k: armed.append(a))

    class _NoTargetPipe:
        _cozy_low_vram_mode = "off"

    outcome = fc.enable_compiled(
        _NoTargetPipe(), _Cfg(), tmp_path, None, publisher=None)
    assert not outcome.armed and outcome.self_mint is None
    assert outcome.eager_reason == "no_compile_target"
    assert armed == []
    with fc._PENDING_LOCK:
        assert fc._PENDING == {}


def test_a_seeded_delivered_cell_no_longer_blocks_a_later_mint(
    tmp_path, monkeypatch,
):
    """gw#608's seeded-cell gate is DELETED (pgw#1010), and this is the test
    that says so deliberately rather than by silence.

    The gate existed only because an in-process capture moved the ONE global
    inductor cache dir away from the seeded entries. No capture, no move, no
    hazard — so a sibling whose own cell is missing mints it instead of being
    eager for the rest of the pod's life because an unrelated slot got a
    delivered cell first.
    """
    _mintable(monkeypatch)
    seeded = tmp_path / "seeded-live"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(seeded))

    outcome = fc.enable_compiled(_Pipe(), _Cfg(), tmp_path, None)

    assert isinstance(outcome.self_mint, fc.PendingSelfMint)
    assert outcome.eager_reason == "mint_in_progress"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(seeded)
