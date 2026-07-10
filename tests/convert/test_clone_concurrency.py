"""Concurrent duplicate clones must serialize on the keyed workdir (gw#442).

Live failure (e2e J19): a crash-recovery re-queue put two clones of the same
(provider, source, destination) on one worker concurrently. Both shared
``clone-<digest>/source``; hf_hub's local-dir download unlinks + re-fetches
files the peer clone had just downloaded and was reading, so the leading
clone's convert phase hit ``FileNotFoundError: No such file or directory:
.../source/text_encoder/model-00001-of-00004.safetensors``. run_clone now
takes an exclusive flock on the workdir for its whole lifetime.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from gen_worker.convert.clone import CloneResult, run_clone
from gen_worker.convert.ingest import IngestedSource

from fake_hub import _FakeHub


class _Ctx:
    def __init__(self, server) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.request_id = "req-1"
        self.destination = {"repo": "acme/fallback"}


def _fake_source(dest_dir: Path) -> IngestedSource:
    return IngestedSource(
        provider="huggingface",
        source_ref="org/tiny",
        source_revision="sha-1",
        dir=dest_dir,
        layout="diffusers",
        model_family="",
        model_family_variant="",
        classification=None,
        attrs={"dtype": "bf16"},
        metadata={"source_provider": "huggingface"},
        repo_spec={"kind": "model", "library_name": "diffusers"},
    )


def test_concurrent_same_source_clones_serialize(fake_hub, tmp_path: Path, monkeypatch) -> None:
    """Two run_clone calls for the same source+destination never overlap in
    the ingest/convert window, and both publish."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    # Keep the test offline: plan failure is a supported path (download-skip
    # disabled, full clone runs).
    monkeypatch.setattr(
        "gen_worker.convert.clone.plan_huggingface",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    guard = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def fake_ingest(source_ref, dest_dir, **kwargs):
        with guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        # Hold the window open so an unserialized peer would overlap.
        time.sleep(0.5)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "config.json").write_text("{}")
        with guard:
            state["active"] -= 1
        return _fake_source(dest_dir)

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)

    results: dict[int, CloneResult | BaseException] = {}

    def _clone(i: int) -> None:
        try:
            results[i] = run_clone(
                _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
                destination_repo="acme/dest",
            )
        except BaseException as exc:  # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=_clone, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    for i in range(2):
        assert isinstance(results.get(i), CloneResult), f"clone {i}: {results.get(i)!r}"
        assert len(results[i].published) == 1
    assert state["max_active"] == 1, "concurrent clones shared the workdir"


def test_distinct_destinations_do_not_serialize(fake_hub, tmp_path: Path, monkeypatch) -> None:
    """The lock is per (provider, source, destination): unrelated clones
    still run in parallel."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr(
        "gen_worker.convert.clone.plan_huggingface",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    both_inside = threading.Barrier(2, timeout=10)

    def fake_ingest(source_ref, dest_dir, **kwargs):
        both_inside.wait()  # deadlocks (Barrier timeout) if serialized
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "config.json").write_text("{}")
        return _fake_source(dest_dir)

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)

    results: dict[int, CloneResult | BaseException] = {}

    def _clone(i: int) -> None:
        try:
            results[i] = run_clone(
                _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
                destination_repo=f"acme/dest-{i}",
            )
        except BaseException as exc:  # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=_clone, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for i in range(2):
        assert isinstance(results.get(i), CloneResult), f"clone {i}: {results.get(i)!r}"


def test_failed_clone_releases_lock_for_retry(fake_hub, tmp_path: Path, monkeypatch) -> None:
    """A failed clone releases the flock so the retry can take it."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr(
        "gen_worker.convert.clone.plan_huggingface",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    calls = {"n": 0}

    def fake_ingest(source_ref, dest_dir, **kwargs):
        calls["n"] += 1
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if calls["n"] == 1:
            raise RuntimeError("network died mid-download")
        (dest_dir / "config.json").write_text("{}")
        return _fake_source(dest_dir)

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)

    with pytest.raises(RuntimeError, match="network died"):
        run_clone(
            _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
            destination_repo="acme/dest",
        )
    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/tiny",
        destination_repo="acme/dest",
    )
    assert len(result.published) == 1
