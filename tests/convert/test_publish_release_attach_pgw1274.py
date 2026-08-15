"""pgw#1274 (HARDCUT A2b): a publish ATTACHES its artifact to a NAMED release.

The hub has taken `release` as a first-class declare field since th#1980
(tensorhub `casPublishV2DeclareRequest.Release`, `json:"release"`). Before this
lane the SDK had no parameter for it at all, so the only way a producer could
state one was to smuggle it through `metadata` — where the hub copies it into
checkpoint metadata and NOTHING reads it. Every release published that way
attached nothing.

Driven end to end through the real `publish_v2` / `publish_flavors` against the
fake hub, which resolves the release the way tensorhub does: at completion,
inside the publish transaction, refusing an uncut one.

Revert-turns-red: drop the `release` parameter and the first two rows fail on
the missing keyword; collapse the typed refusal back into a bare
`HubPublishError` and the third fails on the class.

    pytest tests/convert/test_publish_release_attach_pgw1274.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_hub import _client, _FakeHub

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import publish_flavors
from gen_worker.hubio.client import (
    CommitFile,
    HubPublishError,
    HubReleaseNotFoundError,
)

RELEASE = "2026-08-14-h3"


class _Ctx:
    """Just enough ctx for `HubClient.from_ctx`."""

    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"

    def log(self, message: str, **fields: Any) -> None:
        pass


def _file(tmp_path: Path, data: bytes = b'{"model":"x"}') -> CommitFile:
    path = tmp_path / "config.json"
    path.write_bytes(data)
    return CommitFile(path="config.json", local_path=path, size_bytes=len(data))


def _declaration() -> dict:
    return next(iter(_FakeHub.state["publishes"].values()))


def test_the_release_travels_VERBATIM_into_the_declare_request(
    fake_hub: Any, tmp_path: Path
) -> None:
    _FakeHub.state["releases"] = {RELEASE}

    result = _client(fake_hub).publish_v2(
        destination_repo="acme/model", files=[_file(tmp_path)], release=RELEASE,
    )

    # The hub's own field name, unmangled — and NOT a metadata copy, which is
    # the inert form this lane exists to replace.
    assert _declaration()["release"] == RELEASE
    assert "release" not in (_declaration().get("metadata") or {})
    assert result.checkpoint_id
    assert _FakeHub.state["attached_releases"] == {"pub-1": RELEASE}


def test_no_release_omits_the_field_rather_than_sending_an_empty_one(
    fake_hub: Any, tmp_path: Path
) -> None:
    """An empty string is not an identifier. Sending one would make every
    unattached publish indistinguishable from a publish naming `""`."""
    _client(fake_hub).publish_v2(
        destination_repo="acme/model", files=[_file(tmp_path)],
    )
    assert "release" not in _declaration()


def test_an_UNCUT_release_is_a_typed_NON_RETRYABLE_refusal(
    fake_hub: Any, tmp_path: Path
) -> None:
    """Cutting is a deliberate act, so the remedy is a control-plane call —
    never a re-upload of the bytes, which were never at fault."""
    _FakeHub.state["releases"] = set()

    with pytest.raises(HubReleaseNotFoundError) as caught:
        _client(fake_hub).publish_v2(
            destination_repo="acme/model", files=[_file(tmp_path)],
            release="never-cut",
        )

    exc = caught.value
    assert isinstance(exc, HubPublishError)  # still one publish-failure family
    assert exc.code == "release_not_found"
    assert exc.retryable is False
    assert "cut the release first" in str(exc)
    assert "do not re-upload" in str(exc)
    # The hub's own vocabulary reaches the fleet unchanged.
    from gen_worker.executor import _map_exception
    from gen_worker.pb import worker_scheduler_pb2 as pb

    status, detail = _map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert detail.startswith("release_not_found: ")


def test_publish_flavors_threads_the_release_to_every_variant(
    fake_hub: Any, tmp_path: Path
) -> None:
    """Variants of one export share a release and are told apart INSIDE it by
    contract — so the producer states the release once."""
    _FakeHub.state["releases"] = {RELEASE}
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    trees = []
    for label in ("fp8", "bf16"):
        out = tmp_path / label
        out.mkdir()
        (out / "diffusion.safetensors").write_bytes(label.encode() * 500)
        (out / "config.json").write_text('{"architectures": ["Fake"]}')
        trees.append(ProducedFlavor(path=out, flavor=label,
                                    attributes={"dtype": label}))

    publish_flavors(ctx, trees, destination_repo="acme/quant", release=RELEASE)

    declared = list(_FakeHub.state["publishes"].values())
    assert [d["release"] for d in declared] == [RELEASE, RELEASE]
    assert set(_FakeHub.state["attached_releases"].values()) == {RELEASE}
