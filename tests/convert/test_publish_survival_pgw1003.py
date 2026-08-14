"""pgw#1002 + pgw#1003: a publish failure must cost the UPLOAD, not the CAST.

Two defects on one exit path, which together turned a recoverable transient at
the end of a 2h16m fp8 cast into total loss:

  * the exception handler DELETEd the publish session, and `DELETE
    /publishes/:id` deletes every staged chunk hub-side — so 37 GB already on
    the wire went with the blip that interrupted it;
  * nothing recorded the ``publish_id``, so even a survivor had no way to name
    the session it should resume.

Everything here drives the real ``HubClient.publish_v2`` against the shared
fake hub, with its adversarial PUT injectors switched on (pgw#1005 C: they had
been dead code since the day they were written).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fake_hub import _FakeHub, _client
from gen_worker.hubio.client import CommitFile, HubPublishError
from gen_worker.hubio.journal import JOURNAL_NAME, PublishJournal

CS = 4096


def payload(n: int, seed: int = 1) -> bytes:
    out = bytearray(n)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def small_chunks(monkeypatch):
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch):
    """This file asserts the SHAPE of the publish's failure handling, not the
    delay — which `tests/test_chunk_upload_robustness_pgw1004.py` proves
    directly, by injection, without sleeping. Collapsing the backoff keeps a
    row that drives 15 real PUT attempts from costing a minute of wall clock."""
    monkeypatch.setattr(
        "gen_worker.models.chunk_upload.backoff_sleep_s", lambda *a, **kw: 0.0)


def write(tmp: Path, name: str, data: bytes) -> CommitFile:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return CommitFile(path=name, local_path=p, size_bytes=len(data))


def put_path(data: bytes, off: int, ln: int) -> str:
    return "/v2put/" + sha(data[off:off + ln])


# ---------------------------------------------------------------------------
# pgw#1002 B — the abort rule
# ---------------------------------------------------------------------------


def test_a_TRANSPORT_failure_leaves_the_session_and_its_staged_bytes_alone(
    fake_hub, tmp_path, small_chunks,
):
    """The defect, stated as a test: three chunks land, the fourth 5xxs
    forever, and the old handler answered by DELETEing the session — which
    reclaims the staging prefix, i.e. throws away the three that landed."""
    st = _FakeHub.state
    data = payload(CS * 4)
    f = write(tmp_path, "w.safetensors", data)
    st["fail_put_paths"] = {put_path(data, CS * 3, CS): 999}

    with pytest.raises(HubPublishError) as err:
        _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=[f],
            journal_path=tmp_path / JOURNAL_NAME,
        )

    assert "failed to upload" in str(err.value)
    assert st.get("aborts", []) == [], "a transport failure must not abort the session"
    # The three that landed are still staged, and the session still exists.
    assert len(st["v2_cas"]) == 3
    assert len(st.get("replans", [])) >= 1


def test_a_TERMINAL_repudiation_aborts_the_session_and_clears_the_journal(
    fake_hub, tmp_path, small_chunks,
):
    """The other side of the rule. A refusal the hub itself classified terminal
    IS a statement that these bytes can never be useful, so the staging prefix
    is released and the journal entry goes with it."""
    st = _FakeHub.state
    st["complete_failure"] = {
        "code": "invalid_manifest_for_kind", "retryable": False,
        "message": "missing_diffusers_single_file_safetensors",
    }
    journal = tmp_path / JOURNAL_NAME
    f = write(tmp_path, "w.safetensors", payload(CS * 2))

    with pytest.raises(HubPublishError) as err:
        _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=[f], journal_path=journal)

    assert err.value.retryable is False
    assert err.value.code == "invalid_manifest_for_kind"
    assert st.get("aborted_publishes") == ["pub-1"]
    assert PublishJournal.open(journal).entries == []


def test_a_RETRYABLE_completion_refusal_keeps_everything(
    fake_hub, tmp_path, small_chunks,
):
    """`retryable: true` from the hub is an instruction to come back. Deleting
    the staged bytes on the way out is the opposite of honouring it."""
    st = _FakeHub.state
    st["complete_failure"] = {
        "code": "verification_backlog", "retryable": True, "message": "try again",
    }
    journal = tmp_path / JOURNAL_NAME
    f = write(tmp_path, "w.safetensors", payload(CS * 2))

    with pytest.raises(HubPublishError) as err:
        _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=[f], journal_path=journal)

    assert err.value.retryable is True
    assert st.get("aborts", []) == []
    assert len(st["v2_cas"]) == 2, "staged objects must survive a retryable refusal"
    entries = PublishJournal.open(journal).entries
    assert [e.publish_id for e in entries] == ["pub-1"]


# ---------------------------------------------------------------------------
# The journal, and what it buys
# ---------------------------------------------------------------------------


def test_the_journal_is_written_BEFORE_the_first_PUT_and_cleared_on_success(
    fake_hub, tmp_path, small_chunks,
):
    """A journal written after the transfer is a journal that never survives
    the transfer failing. It records the id at declare, and only a PROMOTED
    publish removes it."""
    seen: list[list[str]] = []
    journal = tmp_path / JOURNAL_NAME
    f = write(tmp_path, "w.safetensors", payload(CS * 2))

    import gen_worker.models.chunk_upload as cu

    real = cu.upload_grants

    def _spy(*a, **kw):
        seen.append([e.publish_id for e in PublishJournal.open(journal).entries])
        return real(*a, **kw)

    cu_upload = cu.upload_grants
    try:
        cu.upload_grants = _spy  # noqa: SLF001 - module-level swap, restored below
        res = _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=[f], journal_path=journal)
    finally:
        cu.upload_grants = cu_upload

    assert seen == [["pub-1"]], "the id must be on disk before any byte moves"
    assert res.checkpoint_id
    assert PublishJournal.open(journal).entries == []
    assert json.loads(journal.read_text())["entries"] == []


def test_a_retry_RESUMES_the_journalled_session_instead_of_declaring_a_new_one(
    fake_hub, tmp_path, small_chunks,
):
    """The point of the whole exercise: the staging prefix is session-scoped,
    so re-using the id is the ONLY way to reach bytes a predecessor moved. The
    second attempt re-plans `pub-1` and never declares `pub-2`.

    (What makes the re-planned need set SHRINK is th#1654, hub-side; this
    proves the client asks the question, which is the half it owns.)
    """
    st = _FakeHub.state
    journal = tmp_path / JOURNAL_NAME
    data = payload(CS * 3, seed=7)
    f = write(tmp_path, "w.safetensors", data)
    st["fail_put_paths"] = {put_path(data, CS * 2, CS): 999}

    with pytest.raises(HubPublishError):
        _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=[f], journal_path=journal)
    assert [e.publish_id for e in PublishJournal.open(journal).entries] == ["pub-1"]
    landed = set(st["v2_cas"])
    assert len(landed) == 2

    # The blip clears; the same producer runs again over the same tree.
    st["fail_put_paths"] = {}
    stages: list[tuple[str, dict]] = []
    res = _client(fake_hub).publish_v2(
        destination_repo="acme/model", files=[f], journal_path=journal,
        on_stage=lambda s, facts: stages.append((s, facts)))

    assert res.checkpoint_id
    assert list(st["publishes"]) == ["pub-1"], "no second declare"
    assert res.revision_id == "pub-1"
    assert [s for s, _ in stages][0] == "resumed"
    # Every object the first attempt staged is still staged.
    assert landed <= set(st["v2_cas"])
    assert PublishJournal.open(journal).entries == []


def test_a_session_the_hub_no_longer_knows_falls_back_to_a_fresh_declare(
    fake_hub, tmp_path, small_chunks,
):
    """Resuming is an optimization; it must never be a way to fail. A stale
    journal entry (expired staging, wiped session) costs one round trip."""
    journal = tmp_path / JOURNAL_NAME
    f = write(tmp_path, "w.safetensors", payload(CS * 2, seed=11))
    from gen_worker.hubio.client import CommitFile as _CF  # noqa: F401
    from gen_worker.hubio.journal import JournalEntry, artifact_key
    from gen_worker.models.chunk_upload import hash_file_and_chunks

    decl = hash_file_and_chunks(Path(f.local_path), chunk_size=CS, rel_path=f.path)
    j = PublishJournal.open(journal)
    j.record(JournalEntry(
        publish_id="pub-gone", destination_repo="acme/model", mode="replace",
        artifact_key=artifact_key(["sha256:" + c.sha256 for c in decl.chunks]),
        objects=2, paths=(f.path,)))

    res = _client(fake_hub).publish_v2(
        destination_repo="acme/model", files=[f], journal_path=journal)

    assert res.revision_id == "pub-1", "a dead session must not block the publish"
    assert PublishJournal.open(journal).entries == []


def test_a_DIFFERENT_artifact_never_adopts_another_publishs_session(
    fake_hub, tmp_path, small_chunks,
):
    """The journal key is the declared object set, so a producer that re-cast
    and got different bytes declares fresh. Splicing one artifact's staging
    into another's publish is the failure mode this guards."""
    journal = tmp_path / JOURNAL_NAME
    a = write(tmp_path / "a", "w.safetensors", payload(CS * 2, seed=13))
    b = write(tmp_path / "b", "w.safetensors", payload(CS * 2, seed=17))

    r1 = _client(fake_hub).publish_v2(
        destination_repo="acme/model", files=[a], journal_path=journal)
    r2 = _client(fake_hub).publish_v2(
        destination_repo="acme/model", files=[b], journal_path=journal)
    assert r1.revision_id == "pub-1" and r2.revision_id == "pub-2"


def test_the_journal_is_never_published_as_repo_content(fake_hub, tmp_path, small_chunks):
    """It lives next to the tree, and `files_from_tree` skips it by name even
    if a caller puts one inside."""
    from gen_worker.hubio.client import files_from_tree

    tree = tmp_path / "flavor"
    tree.mkdir()
    (tree / "config.json").write_bytes(b"{}")
    (tree / JOURNAL_NAME).write_bytes(b"{}")
    assert [f.path for f in files_from_tree(tree)] == ["config.json"]


# ---------------------------------------------------------------------------
# pgw#1004 C at the publish level — a re-mint is not a failed pass
# ---------------------------------------------------------------------------


def test_expired_grants_are_RE_MINTED_without_spending_the_reupload_budget(
    fake_hub, tmp_path, small_chunks,
):
    """The hub mints CAS grants with a 2 h TTL. Crossing it mid-publish used to
    look like `_REUPLOAD_ATTEMPTS` object failures in a row; now it re-plans,
    which is exactly what the CAS path's re-mint route is."""
    st = _FakeHub.state
    st["grant_ttl_s"] = -1  # every grant is minted already dead
    journal = tmp_path / JOURNAL_NAME
    f = write(tmp_path, "w.safetensors", payload(CS * 2, seed=19))

    with pytest.raises(HubPublishError) as err:
        _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=[f], journal_path=journal)

    assert err.value.code == "grant_expiry_loop"
    # Nothing was ever sent — the client refused to start a PUT it could not
    # finish — and the failure was NOT charged to the re-upload budget.
    assert st.get("put_counts", {}) == {}
    from gen_worker.hubio.client import _EXPIRY_REPLAN_ATTEMPTS

    assert len(st["replans"]) == _EXPIRY_REPLAN_ATTEMPTS
    # And the session survives: nothing about the bytes was in question.
    assert st.get("aborts", []) == []


# ---------------------------------------------------------------------------
# The payoff: a retry re-uploads instead of re-CASTING
# ---------------------------------------------------------------------------


def test_a_retained_cast_output_is_republished_instead_of_rebuilt(
    fake_hub, tmp_path, small_chunks,
):
    """`clone.py` used to rmtree the produced tree on EVERY exit path, with
    "only the downloaded source is resumable" stated outright in the code — so
    a blip at the end of a 2h16m cast cost the cast. Now a tree a predecessor
    finished casting AND declared is recognised, and the cast does not run
    again."""
    from gen_worker.convert import clone

    st = _FakeHub.state
    workdir = tmp_path / "clone-abc"
    flavor_dir = workdir / "flavor-fp8"
    flavor_dir.mkdir(parents=True)
    data = payload(CS * 2, seed=29)
    (flavor_dir / "w.safetensors").write_bytes(data)
    (flavor_dir / "config.json").write_bytes(b'{"a":1}')

    files = [CommitFile(path=p.name, local_path=p)
             for p in sorted(flavor_dir.iterdir())]
    st["fail_puts"] = 999  # the publish dies after declaring

    with pytest.raises(HubPublishError):
        _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=files,
            journal_path=workdir / JOURNAL_NAME,
            journal_state={"spec_label": "fp8", "tree": str(flavor_dir),
                           "attrs": {"dtype": "fp8", "file_layout": "multi-file"}},
        )

    # The successor recognises its predecessor's finished output.
    attrs = clone._reusable_flavor_tree(workdir, "fp8", flavor_dir)
    assert attrs == {"dtype": "fp8", "file_layout": "multi-file"}

    # A tree that no longer matches the declaration is rebuilt, not published.
    (flavor_dir / "stray.json").write_bytes(b"{}")
    assert clone._reusable_flavor_tree(workdir, "fp8", flavor_dir) is None
    (flavor_dir / "stray.json").unlink()
    assert clone._reusable_flavor_tree(workdir, "fp8", flavor_dir) is not None
    (flavor_dir / "config.json").unlink()
    assert clone._reusable_flavor_tree(workdir, "fp8", flavor_dir) is None


def test_a_run_that_died_MID_CAST_leaves_nothing_to_reuse(tmp_path):
    """No journal entry means the predecessor never got as far as declaring,
    which happens only after the tree is complete and every file hashed. A
    partial tree from a crash is not resumable and never was."""
    from gen_worker.convert import clone

    workdir = tmp_path / "clone-def"
    flavor_dir = workdir / "flavor-bf16"
    flavor_dir.mkdir(parents=True)
    (flavor_dir / "half.safetensors").write_bytes(b"partial")
    assert clone._reusable_flavor_tree(workdir, "bf16", flavor_dir) is None
    assert clone._reusable_flavor_tree(workdir, "bf16", workdir / "nope") is None


def test_a_live_TTL_publishes_normally(fake_hub, tmp_path, small_chunks):
    st = _FakeHub.state
    st["grant_ttl_s"] = 7200.0  # the production CAS grant TTL
    f = write(tmp_path, "w.safetensors", payload(CS * 2, seed=23))
    res = _client(fake_hub).publish_v2(destination_repo="acme/model", files=[f])
    assert res.checkpoint_id and res.uploaded == 2
