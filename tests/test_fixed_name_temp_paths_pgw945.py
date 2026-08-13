"""pgw#945 (tail of pgw#938): three more fixed-name temp paths, classified.

pgw#938 replaced the SDK uploader's ``dest.tmp`` and noticed three siblings
without analysing them. Its own finding was that a pod-wide path can be
CORRECT, so each site here is classified rather than assumed:

* ``request_context/_datasets._download_url_streamed`` — **RACY, and it is
  the real one.** ``resolve_dataset`` materializes into a pod-wide
  content-keyed cache (``/tmp/gen_worker_datasets/<owner>/<name>/<snapshot>``),
  so two requests in one container asking for one dataset reach the SAME
  ``dest``, and the temp name was derived from it. The bytes are identical —
  the destruction is in the LIFECYCLE: one writer's failure path unlinks the
  other's in-flight download, and the victim fails its own rename after paying
  for every byte. This file proves that, and proves the fix.
* ``models/disk_gc.RefIndex._save_locked`` — **racy across PROCESSES.** The
  lock it holds is a thread lock; the index sits in the shared model cache dir.
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


def test_two_ref_index_writers_leave_a_whole_document(tmp_path):
    """`RefIndex`'s lock is per-process, so two processes sharing a cache dir
    are ordered by nothing. Whoever renames last wins — but every reader must
    see a COMPLETE index, never a mixture of two states."""
    cache = tmp_path / "cache"
    cache.mkdir()
    left, right = RefIndex(cache), RefIndex(cache)
    barrier = threading.Barrier(2, timeout=_RENDEZVOUS_S)

    def write(index: RefIndex, prefix: str) -> None:
        barrier.wait()
        for i in range(60):
            index.record(f"{prefix}/model-{i}", cache / f"{prefix}-{i}", 1024 * i)

    threads = [threading.Thread(target=write, args=(left, "a")),
               threading.Thread(target=write, args=(right, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_RENDEZVOUS_S)
        assert not t.is_alive()

    doc = json.loads((cache / "ref-index.json").read_text("utf-8"))
    # A torn write is not a partial dict — it is unparseable, or a document
    # holding compiled graphs whose refs came from two different writers' states.
    owners = {ref.split("/", 1)[0] for ref in doc}
    assert owners in ({"a"}, {"b"}), f"the index mixes two writers: {sorted(doc)}"
    assert not [p for p in cache.iterdir() if p.name != "ref-index.json"], \
        "a temp file was left behind"


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
