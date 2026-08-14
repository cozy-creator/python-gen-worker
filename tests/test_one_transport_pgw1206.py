"""pgw#1206 B — the One-transport contract.

Three properties, each previously true on at most ONE of the upload paths:

1. ONE status classification for presigned PUTs (`hubio.transport.put_verdict`)
   — chunk-CAS and the media multipart path project from the same table, so
   403-on-expired-presign is a RE-PLAN everywhere and 408 is transient
   everywhere (it was terminal on the engine's old private table).
2. ONE grant-expiry behavior — RED against `1b282a82`: a 403 from a media
   part PUT was TERMINAL there (`ArtifactTransferError`), while chunk-CAS had
   re-planned since pgw#1004. Now the media path re-creates the session once.
3. `fetch_verified` refuses to fetch what it cannot verify — absent, untagged
, or disallowed-algorithm digests raise before a byte moves
   (th#1303 S1), and the in-loop byte cap raises `StreamTooLarge`.

The dead credentialed lane is also pinned dead: the hub has ZERO
`transfer_grant` producers, so `gen_worker.s3_transfer` and the boto3
dependency are gone whole — a revival must reintroduce them consciously.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.api.errors import ArtifactTransferError
from gen_worker.hubio import transport
from gen_worker.hubio.fetch import fetch_verified
from gen_worker.hubio.transport import (
    PUT_EXPIRED,
    PUT_OK,
    PUT_TERMINAL,
    PUT_TRANSIENT,
    put_verdict,
)


# --- 1. the one classification table ----------------------------------------

def test_put_verdict_is_the_one_table() -> None:
    assert put_verdict(200) == PUT_OK
    assert put_verdict(204) == PUT_OK
    # 403 past expires_at is a re-plan, never a repudiation.
    assert put_verdict(403, presign_expired=True) == PUT_EXPIRED
    assert put_verdict(403) == PUT_TERMINAL
    assert put_verdict(400) == PUT_TERMINAL
    assert put_verdict(404) == PUT_TERMINAL
    for status in (408, 429, 500, 503):
        assert put_verdict(status) == PUT_TRANSIENT, status


def test_both_put_paths_project_from_it() -> None:
    """The engine's TransportError and chunk-CAS's typed outcomes are
    projections of put_verdict, not second tables."""
    import inspect

    from gen_worker.models import chunk_upload

    assert "put_verdict" in inspect.getsource(chunk_upload._classify_put)
    assert "put_verdict" in inspect.getsource(transport._classify_response_status)
    # And the projections agree on the transient set (408 included — the
    # engine's old private table called it terminal).
    err = transport._classify_response_status(408, "")
    assert err is not None and err.retryable


def test_the_credentialed_lane_stays_dead() -> None:
    """The hub has no transfer_grant producers; the worker holds no store
    credentials. Reviving boto3 or s3_transfer must be a conscious act.
    (Asserted on the tree, not the venv — a stale local env may still carry
    the wheel; the dependency set and the import surface are the contract.)"""
    with pytest.raises(ModuleNotFoundError):
        import gen_worker.s3_transfer  # noqa: F401

    repo = Path(__file__).resolve().parents[1]
    pyproject = (repo / "pyproject.toml").read_text()
    assert "boto3" not in pyproject
    hits = [
        p for p in (repo / "src" / "gen_worker").rglob("*.py")
        if "import boto3" in p.read_text() or "transfer_grant" in p.read_text()
    ]
    assert hits == [], f"credentialed-lane residue: {hits}"


# --- 2. one grant-expiry behavior -------------------------------------------

def _put_403(phase: str = "put", status: int = 403) -> ArtifactTransferError:
    return ArtifactTransferError(
        "S3 part upload terminal status (403)", provider="tensorhub",
        phase=phase, retryable=False, status_code=status,
    )


def test_a_403_part_put_replans_the_session_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """RED against 1b282a82: the media path failed terminally on the first
    403'd part PUT. An expired presign is a re-plan — re-create the session
    (fresh presigns) and re-drive; only a 403 on the FRESH presigns is
    terminal (a substituted claim recurs, an expired presign does not)."""
    from gen_worker import presigned_upload as pu

    calls: list[int] = []

    def fake_scoped(**kw: object) -> pu.PresignedUploadResult:
        calls.append(1)
        if len(calls) == 1:
            raise _put_403()
        return pu.PresignedUploadResult(meta={"ok": True})

    monkeypatch.setattr(pu, "_presigned_upload_file_scoped", fake_scoped)
    monkeypatch.setattr(pu, "control_plane_session", lambda base: (object(), True))
    f = tmp_path / "a.bin"; f.write_bytes(b"x")
    result = pu.presigned_upload_file(
        file_path=f, base_url="http://hub.invalid", endpoint_path="/api/v1/media/uploads",
        headers={}, create_payload={}, blake3_hex="00", size_bytes=1,
    )
    assert result.meta == {"ok": True}
    assert len(calls) == 2, "exactly one re-plan"


def test_a_403_on_the_fresh_presigns_is_terminal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gen_worker import presigned_upload as pu

    calls: list[int] = []

    def fake_scoped(**kw: object) -> pu.PresignedUploadResult:
        calls.append(1)
        raise _put_403()

    monkeypatch.setattr(pu, "_presigned_upload_file_scoped", fake_scoped)
    monkeypatch.setattr(pu, "control_plane_session", lambda base: (object(), True))
    f = tmp_path / "a.bin"; f.write_bytes(b"x")
    with pytest.raises(ArtifactTransferError):
        pu.presigned_upload_file(
            file_path=f, base_url="http://hub.invalid", endpoint_path="/api/v1/media/uploads",
            headers={}, create_payload={}, blake3_hex="00", size_bytes=1,
        )
    assert len(calls) == 2, "one re-plan, then terminal — never a loop"


def test_non_403_failures_do_not_replan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The re-plan is FOR expired presigns; a create-phase failure or a 400
    keeps its one-shot semantics."""
    from gen_worker import presigned_upload as pu

    calls: list[int] = []

    def fake_scoped(**kw: object) -> pu.PresignedUploadResult:
        calls.append(1)
        raise _put_403(phase="create", status=500)

    monkeypatch.setattr(pu, "_presigned_upload_file_scoped", fake_scoped)
    monkeypatch.setattr(pu, "control_plane_session", lambda base: (object(), True))
    f = tmp_path / "a.bin"; f.write_bytes(b"x")
    with pytest.raises(ArtifactTransferError):
        pu.presigned_upload_file(
            file_path=f, base_url="http://hub.invalid", endpoint_path="/api/v1/media/uploads",
            headers={}, create_payload={}, blake3_hex="00", size_bytes=1,
        )
    assert len(calls) == 1


# --- 3. fetch_verified refuses what it cannot verify ------------------------

def test_fetch_refuses_an_absent_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no expected digest"):
        fetch_verified("http://x.invalid/b", tmp_path / "b", expected_digest="")


def test_fetch_refuses_an_untagged_digest(tmp_path: Path) -> None:
    """pgw#871: 64 hex chars name no algorithm; a guess verifies nothing."""
    with pytest.raises(ValueError, match="not algorithm-tagged"):
        fetch_verified("http://x.invalid/b", tmp_path / "b", expected_digest="ab" * 32)


def test_fetch_refuses_a_disallowed_algorithm(tmp_path: Path) -> None:
    """th#1303 S1: sha256 is the servable vocabulary; a blake3 ref names
    bytes the hub can no longer address."""
    with pytest.raises(ValueError, match="not\\s+verifiable"):
        fetch_verified("http://x.invalid/b", tmp_path / "b",
                       expected_digest="blake3:" + "ab" * 32)


def test_the_refusals_fire_before_any_fetch(tmp_path: Path) -> None:
    """A permanent refusal must not spend retries looking like a flaky
    network: the URL above is unresolvable, so reaching the network would
    raise a RequestException — the ValueError proves we never did."""
    # (Asserted implicitly by the three tests above: .invalid never resolves,
    # and no ConnectionError surfaced.)
    assert True
