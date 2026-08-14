"""Three fixed-name temp paths, classified.

A pod-wide fixed temp path can be CORRECT, so each site is classified rather
than assumed:

* ``request_context/_datasets._download_url_streamed`` — **RACY, and it is
  the real one.** ``resolve_dataset`` materializes into a pod-wide
  content-keyed cache (``/tmp/gen_worker_datasets/<owner>/<name>/<snapshot>``),
  so two requests in one container asking for one dataset reach the SAME
  ``dest``, and the temp name was derived from it. The bytes are identical —
  the destruction is in the LIFECYCLE: one writer's failure path unlinks the
  other's in-flight download, and the victim fails its own rename after paying
  for every byte. This file proves that, and proves the fix.
* ``models/disk_gc.RefIndex._save_locked`` — **process-locked.** Every mutation
  refreshes and merges under a flock on the stable cache-directory inode before
  its durable replace. There is no ownership-sensitive lock sidecar.
* ``env_seal.write_library_memo`` — **per-attempt today** (its only caller
  seeds ``<mint workdir>/seal-lib-memo.json``, and the workdir is per mint
  attempt), but the atomicity promise is the function's, so it no longer
  depends on a fact stated in another module.

Reasons live at each site, not only here.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import multiprocessing
import threading
from pathlib import Path
from typing import Any, List, Optional

import pytest

from gen_worker.env_seal import write_library_memo
from gen_worker.models.disk_gc import RefIndex
from gen_worker.request_context._datasets import _download_url_streamed

#: One bound for every rendezvous below. Each wait is on a signal the OTHER
#: thread sets unconditionally, so reaching it is a failed test, not a slow
#: one — the number only decides how long a broken run hangs before saying so.
_RENDEZVOUS_S = 60.0

_PAYLOAD = bytes(range(256)) * 8192          # 2 MiB, > the 1 MiB read chunk
_DIGEST = "sha256:" + hashlib.sha256(_PAYLOAD).hexdigest()


def _record_shared_ref(cache: str, ref: str, barrier: Any) -> None:
    index = RefIndex(Path(cache))
    barrier.wait(_RENDEZVOUS_S)
    index.record(ref, Path(cache) / ref.replace("/", "-"), 1024)


class _Rendezvous:
    """The three signals that order the two writers deterministically."""

    def __init__(self) -> None:
        self.first_partial = threading.Event()
        self.second_partial = threading.Event()
        self.first_failed = threading.Event()
        self.connections = 0
        self.lock = threading.Lock()

    def wait(self, event: threading.Event, what: str) -> None:
        assert event.wait(_RENDEZVOUS_S), f"rendezvous never reached: {what}"


@pytest.fixture()
def shard_server():
    rv = _Rendezvous()
    half = len(_PAYLOAD) // 2

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # quiet
            pass

        def do_GET(self) -> None:
            with rv.lock:
                rv.connections += 1
                nth = rv.connections
            self.send_response(200)
            self.send_header("Content-Length", str(len(_PAYLOAD)))
            self.end_headers()
            self.wfile.write(_PAYLOAD[:half])
            self.wfile.flush()
            if nth == 1:
                # Writer A: stall until B is streaming, then die mid-shard.
                rv.first_partial.set()
                rv.wait(rv.second_partial, "second writer started streaming")
                self.close_connection = True
                self.connection.close()
                return
            # Writer B: stall until A's failure path has run, then finish.
            rv.second_partial.set()
            rv.wait(rv.first_failed, "first writer's cleanup ran")
            self.wfile.write(_PAYLOAD[half:])
            self.wfile.flush()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, rv
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=_RENDEZVOUS_S)


def test_a_failing_writer_cannot_destroy_a_concurrent_one(shard_server, tmp_path):
    """Two requests, one dataset, one ``dest``. The first download dies
    mid-shard; the second must still land its verified bytes.

    Before the fix both wrote ``dest.tmp``, so the first writer's
    ``tmp.unlink()`` deleted the second's open file and the second — having
    transferred every byte and verified its own digest — died on ``replace``
    with ``FileNotFoundError``.
    """
    server, rv = shard_server
    url = f"http://127.0.0.1:{server.server_address[1]}/shard"
    dest = tmp_path / "shards" / "data-00000.bin"
    dest.parent.mkdir(parents=True)
    errors: List[Optional[BaseException]] = [None, None]

    def first() -> None:
        try:
            _download_url_streamed(url, dest, expected_digest=_DIGEST,
                                   expected_size=len(_PAYLOAD))
        except BaseException as exc:      # expected: the aborted connection
            errors[0] = exc
        finally:
            rv.first_failed.set()

    def second() -> None:
        try:
            _download_url_streamed(url, dest, expected_digest=_DIGEST,
                                   expected_size=len(_PAYLOAD))
        except BaseException as exc:
            errors[1] = exc

    a = threading.Thread(target=first)
    a.start()
    rv.wait(rv.first_partial, "first writer started streaming")
    b = threading.Thread(target=second)
    b.start()
    for t in (a, b):
        t.join(timeout=_RENDEZVOUS_S)
        assert not t.is_alive(), "a writer never finished"

    assert errors[0] is not None, "the first writer was supposed to be aborted"
    assert errors[1] is None, f"the surviving writer was destroyed: {errors[1]!r}"
    assert dest.read_bytes() == _PAYLOAD
    # The loser cleans up after itself and nothing else: only the shard is left.
    assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]


def test_two_ref_index_processes_merge_without_losing_an_owner(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_record_shared_ref, args=(str(cache), ref, barrier)
        )
        for ref in ("a/model", "b/model")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=_RENDEZVOUS_S)
        assert process.exitcode == 0

    doc = json.loads((cache / "ref-index.json").read_text("utf-8"))
    assert set(doc) == {"a/model", "b/model"}
    assert {p.name for p in cache.iterdir()} == {"ref-index.json"}


def test_ref_index_uses_directory_authority_not_root_owned_lock_metadata(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    root_writer = RefIndex(cache)
    root_writer.record("root/model", cache / "root", 1)

    # Reproduce the privilege lane's ownership boundary without requiring this
    # unit test itself to run as root: the directory remains writable, while
    # metadata left by the old root writer is read-only to the next writer.
    index_path = cache / "ref-index.json"
    index_path.chmod(0o444)
    legacy_lock = cache / ".ref-index.json.lock"
    legacy_lock.touch(mode=0o444)
    legacy_lock.chmod(0o444)

    dropped_writer = RefIndex(cache)
    dropped_writer.record("child/model", cache / "child", 2)
    assert set(RefIndex(cache).entries()) == {"root/model", "child/model"}
    assert index_path.stat().st_mode & 0o777 == 0o644


def test_ref_index_instances_refresh_before_reads_and_writes(tmp_path):
    cache = tmp_path / "cache"
    left, right = RefIndex(cache), RefIndex(cache)
    left.record("a/model", cache / "a", 1)
    assert right.path("a/model") == cache / "a"
    right.record("b/model", cache / "b", 2)
    assert set(left.entries()) == {"a/model", "b/model"}
    left.remove("a/model")
    assert right.path("a/model") is None


def test_the_library_memo_survives_a_shared_destination(tmp_path):
    """The per-attempt site: correct today by its caller's path, and now
    correct by construction. Two writers of one memo path both produce a
    parseable document."""
    memo = tmp_path / "seal-lib-memo.json"
    barrier = threading.Barrier(2, timeout=_RENDEZVOUS_S)
    counts: List[int] = []

    def write() -> None:
        barrier.wait()
        counts.append(write_library_memo(memo))

    threads = [threading.Thread(target=write) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_RENDEZVOUS_S)
        assert not t.is_alive()

    assert len(counts) == 2
    doc = json.loads(memo.read_text())
    assert isinstance(doc.get("digests"), dict)
    assert [p.name for p in Path(tmp_path).iterdir()] == [memo.name]
