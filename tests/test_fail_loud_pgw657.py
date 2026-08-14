"""pgw#657: the load-bearing patches say so when they stop being load-bearing.

Each row here locks a guard whose SILENT failure mode was the debt: the
huggingface_hub timeout floor reverting on a backend reshape (gw#456's
infinite-timeout hang, fleet-wide), and the gw#640 boot record living on RAM
that the very OOM it reports frees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker import net, postmortem


# ---------------------------------------------------------------------------
# net.py — the floor is PROVEN on the session huggingface_hub actually uses.
# ---------------------------------------------------------------------------


@pytest.fixture
def uninstalled_floor(monkeypatch):
    """Re-arm the install path without leaking state into other tests."""
    from huggingface_hub.utils import _http as hf_http

    factory_before = hf_http._GLOBAL_CLIENT_FACTORY
    # Back to a virgin process shape: no floor installed anywhere.
    hf_http.set_client_factory(hf_http.default_client_factory)
    monkeypatch.setattr(net, "_installed", False)
    yield hf_http
    # undo() first: a test may have patched set_client_factory itself.
    monkeypatch.undo()
    hf_http.set_client_factory(factory_before)


def test_floor_installs_and_is_verifiable(uninstalled_floor) -> None:
    net.install_hf_http_timeouts()
    client = uninstalled_floor.get_session()
    assert net._floor_timeout_hook in client.event_hooks["request"]


def test_a_backend_that_ignores_the_factory_fails_loudly(
    uninstalled_floor, monkeypatch,
) -> None:
    """The exact silent revert this issue exists for: huggingface_hub keeps
    the API but stops honouring it. Before, the worker booted happily with
    infinite HTTP timeouts."""
    monkeypatch.setattr(
        uninstalled_floor, "set_client_factory", lambda _factory: None,
    )
    with pytest.raises(net.HfHttpFloorError) as exc:
        net.install_hf_http_timeouts()
    assert "gen_worker/net.py" in str(exc.value)
    # ...and it stays un-installed, so the next call retries rather than
    # caching a lie.
    assert net._installed is False


def test_floor_hook_keeps_explicit_timeouts_and_floors_infinite_ones() -> None:
    class _Req:
        extensions: dict = {"timeout": {"connect": 3.0, "read": None}}

    req = _Req()
    net._floor_timeout_hook(req)
    assert req.extensions["timeout"]["connect"] == 3.0  # caller's number wins
    assert req.extensions["timeout"]["read"] == net.http_timeouts()[1]


# ---------------------------------------------------------------------------
# postmortem.py — a boot record on tmpfs is evidence that dies with the death.
# ---------------------------------------------------------------------------


def _mounts(tmp_path, entries):
    p = tmp_path / "mounts"
    p.write_text("".join(f"dev {point} {fstype} rw 0 0\n" for point, fstype in entries))
    return p


def test_tmpfs_carrier_is_recognised_as_volatile(tmp_path) -> None:
    mounts = _mounts(tmp_path, [("/", "ext4"), ("/tmp", "tmpfs")])
    assert postmortem.boot_record_is_volatile(Path("/tmp/x.json"), mounts)
    assert not postmortem.boot_record_is_volatile(
        Path("/var/lib/x.json"), mounts
    )


def test_longest_mount_prefix_wins(tmp_path) -> None:
    mounts = _mounts(tmp_path, [("/", "tmpfs"), ("/workspace", "ext4")])
    assert postmortem._fstype_for(
        Path("/workspace/cache/rec.json"), mounts,
    ) == "ext4"


def test_written_record_carries_its_own_durability(tmp_path, caplog) -> None:
    """A reader must be able to tell 'the pod did not die' from 'the evidence
    was on RAM'. /dev/shm is tmpfs on every Linux box, so this exercises the
    real /proc/mounts path, not a fixture."""
    import json

    durable = tmp_path / "rec.json"
    postmortem.write_boot_record(durable)
    assert json.loads(durable.read_text())["carrier_volatile"] is False

    volatile_root = Path("/dev/shm")
    if not volatile_root.is_dir():
        pytest.skip("no tmpfs mount available")
    volatile = volatile_root / "pgw657-rec.json"
    try:
        postmortem.write_boot_record(volatile)
        record = json.loads(volatile.read_text())
        assert record["carrier_volatile"] is True
        assert any("VOLATILE" in m for m in caplog.messages)
    finally:
        volatile.unlink(missing_ok=True)


def test_explicit_env_path_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("GEN_WORKER_BOOT_RECORD", "/somewhere/explicit.json")
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", "/mnt/cache")
    assert str(postmortem._default_boot_record_path()) == "/somewhere/explicit.json"


def test_cache_volume_is_preferred_over_tmp(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEN_WORKER_BOOT_RECORD", raising=False)
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path))
    assert postmortem._default_boot_record_path() == tmp_path / "gen-worker-boot-record.json"


# ---------------------------------------------------------------------------
# convert/source.py — the documented-but-unwired branch is a typed refusal.
# ---------------------------------------------------------------------------


def test_diffusers_singlefile_is_a_typed_refusal(tmp_path) -> None:
    from gen_worker.api.errors import ValidationError
    from gen_worker.convert.source import Source

    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    src = Source(tmp_path)
    assert src.file_layout == "single-file"
    with pytest.raises(ValidationError) as exc:
        list(src.iter_hf_components())
    assert "as_hf_model" in str(exc.value)
