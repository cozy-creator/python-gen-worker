"""pgw#1004 / pgw#1005: the chunk-CAS data plane's retry hygiene.

The audit found this module to be the only retry loop in the tree
with none of the protections the rest of the tree has: no backoff at all, no
liveness beat, no ``expires_at`` awareness, and a semaphore that bounded
sockets while claiming to bound buffers. Every row below drives the real
``upload_grants`` against a real localhost store that behaves the way R2 does.

NO WALL-CLOCK ASSERTIONS. Backoff is proven by injecting the sleep function
and reading what the loop asked for — a test that measured elapsed time would
be a flake and would also have to actually sleep.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import http.server
import threading
from typing import List

import pytest

from gen_worker import activity as activity_mod
from gen_worker import progress as progress_mod
from gen_worker.models import chunk_upload as cu
from gen_worker.models.chunk_upload import UploadGrant, upload_grants

CS = 4096


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def b64(hexd: str) -> str:
    return base64.b64encode(bytes.fromhex(hexd)).decode()


def make(n: int, seed: int = 3) -> bytes:
    out = bytearray(n)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


def rfc3339(delta_s: float) -> str:
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(seconds=delta_s)).isoformat().replace("+00:00", "Z")


class _Store(http.server.BaseHTTPRequestHandler):
    """R2-shaped enforcing store with the adversarial injectors the audit
    asked for: N 5xx then success, a permanent 403, and a mid-PUT reset."""

    def log_message(self, *a):  # noqa: D102
        pass

    def do_PUT(self):  # noqa: N802
        srv = self.server
        with srv.lock:
            srv.attempts[self.path] = srv.attempts.get(self.path, 0) + 1
            fail = srv.fail_puts.get(self.path, 0)
            if fail:
                srv.fail_puts[self.path] = fail - 1
            reset = srv.reset_puts.get(self.path, 0)
            if reset:
                srv.reset_puts[self.path] = reset - 1
            forbid = srv.forbid.get(self.path, 0)
            if forbid:
                srv.forbid[self.path] = forbid - 1
        if reset:
            # Sever the connection with no HTTP answer at all.
            try:
                self.connection.close()
            except Exception:
                pass
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        if fail:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if forbid:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"<Code>SignatureDoesNotMatch</Code>")
            return
        claimed = self.headers.get("x-amz-checksum-sha256")
        if claimed is None or b64(sha(body)) != claimed:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<Code>BadDigest</Code>")
            return
        with srv.lock:
            srv.objects[self.path] = body
        self.send_response(200)
        self.end_headers()


class Store:
    def __init__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Store)
        self.httpd.objects, self.httpd.attempts = {}, {}
        self.httpd.fail_puts, self.httpd.reset_puts, self.httpd.forbid = {}, {}, {}
        self.httpd.lock = threading.Lock()
        self._t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._t.start()

    @property
    def base(self):
        h, p = self.httpd.server_address[:2]
        return f"http://{h}:{p}"

    def grant(self, data: bytes, *, expires_at: str = "") -> UploadGrant:
        hexd = sha(data)
        return UploadGrant(
            digest="sha256:" + hexd, size_bytes=len(data),
            put_url=f"{self.base}/staging/{hexd}",
            headers={"x-amz-checksum-sha256": b64(hexd)},
            staging_key=f"staging/{hexd}", expires_at=expires_at,
        )

    def path_of(self, data: bytes) -> str:
        return f"/staging/{sha(data)}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._t.join(timeout=5)


@pytest.fixture()
def store():
    s = Store()
    try:
        yield s
    finally:
        s.close()


class _Sleeps:
    """Records what the retry loop ASKED to sleep. Never actually sleeps."""

    def __init__(self) -> None:
        self.calls: List[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))


@pytest.fixture()
def sleeps(monkeypatch):
    rec = _Sleeps()
    real = cu._put_one

    def _patched(session, grant, body, **kw):
        kw.setdefault("sleep", rec)
        return real(session, grant, body, **kw)

    monkeypatch.setattr(cu, "_put_one", _patched)
    return rec


# ---------------------------------------------------------------------------
# A. backoff — the defect: five immediate retries into a store that just 503'd
# ---------------------------------------------------------------------------


def test_a_transient_5xx_is_retried_AFTER_a_backoff_and_then_succeeds(store, tmp_path, sleeps):
    """The injector the audit found dead (`fail_puts`), switched on: two 503s
    then success. Before pgw#1004 this passed too — with ZERO delay, which is
    a retry storm. The delay is the assertion."""
    data = make(CS)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    store.httpd.fail_puts[store.path_of(data)] = 2

    rep = upload_grants([store.grant(data)], lambda d: (f, 0, len(data)), parallel=1)

    assert rep.ok, rep.failures
    assert store.httpd.attempts[store.path_of(data)] == 3
    # One sleep per retried attempt, each a positive, bounded, jittered delay.
    assert len(sleeps.calls) == 2, sleeps.calls
    assert all(0 < s <= 20.0 for s in sleeps.calls), sleeps.calls
    # Decorrelated jitter widens the window with the attempt number. (Pure
    # function, sampled — never a measurement of how long anything took.)
    early = [cu.backoff_sleep_s(1) for _ in range(50)]
    late = [cu.backoff_sleep_s(4) for _ in range(50)]
    assert max(early) < max(late)


def test_a_mid_PUT_RESET_is_transport_retryable_not_a_refusal(store, tmp_path, sleeps):
    """The `reset_puts` injector: the connection is severed with no HTTP
    answer. A transport failure must retry (with backoff), never classify as
    the store refusing our bytes."""
    data = make(CS, seed=9)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    store.httpd.reset_puts[store.path_of(data)] = 1

    rep = upload_grants([store.grant(data)], lambda d: (f, 0, len(data)), parallel=1)

    assert rep.ok, rep.failures
    assert store.httpd.objects[store.path_of(data)] == data
    assert len(sleeps.calls) == 1


def test_a_terminal_4xx_is_raised_WITHOUT_charging_a_backoff(store, tmp_path, sleeps):
    """Classify before charging: a 403 on a live grant is a refusal of what we
    sent. One attempt, no sleep, no storm."""
    data = make(CS, seed=11)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    store.httpd.forbid[store.path_of(data)] = 9

    rep = upload_grants([store.grant(data, expires_at=rfc3339(3600))],
                        lambda d: (f, 0, len(data)), parallel=1)

    assert not rep.ok and not rep.expired
    assert "403" in rep.failures[0], rep.failures
    assert store.httpd.attempts[store.path_of(data)] == 1
    assert sleeps.calls == []


def test_give_up_is_count_based_and_reports_the_last_cause(store, tmp_path, sleeps):
    data = make(CS, seed=13)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    store.httpd.fail_puts[store.path_of(data)] = 99

    rep = upload_grants([store.grant(data)], lambda d: (f, 0, len(data)), parallel=1)

    assert not rep.ok
    assert store.httpd.attempts[store.path_of(data)] == cu._MAX_ATTEMPTS
    # The last attempt is not followed by a pointless sleep.
    assert len(sleeps.calls) == cu._MAX_ATTEMPTS - 1
    assert "503" in rep.failures[0]


# ---------------------------------------------------------------------------
# C. expires_at — on the wire since th#1303, read by nobody until pgw#1004
# ---------------------------------------------------------------------------


def test_expires_at_is_parsed_off_the_wire():
    live = UploadGrant(digest="sha256:aa", size_bytes=1, put_url="", headers={},
                       expires_at=rfc3339(3600))
    dead = UploadGrant(digest="sha256:aa", size_bytes=1, put_url="", headers={},
                       expires_at=rfc3339(-1))
    silent = UploadGrant(digest="sha256:aa", size_bytes=1, put_url="", headers={})
    assert live.expires_at_unix() > 0 and not live.expired()
    assert dead.expired()
    # A hub that names no expiry gets no expiry check invented for it.
    assert silent.expires_at_unix() == 0.0 and not silent.expired()
    # Unparseable is treated as "named nothing", never as expired.
    assert not UploadGrant(digest="sha256:aa", size_bytes=1, put_url="",
                           headers={}, expires_at="tomorrow").expired()


def test_an_expired_grant_is_NOT_SENT_and_asks_for_a_re_plan(store, tmp_path):
    """The margin exists so a 64 MiB body is never started on a presign that
    will die under it. Nothing reaches the wire and nothing is charged to the
    failure budget — the caller re-plans."""
    data = make(CS, seed=17)
    f = tmp_path / "m.bin"
    f.write_bytes(data)

    rep = upload_grants([store.grant(data, expires_at=rfc3339(5))],
                        lambda d: (f, 0, len(data)), parallel=1)

    assert rep.expired == ["sha256:" + sha(data)]
    assert rep.failures == [] and rep.needs_replan and not rep.ok
    assert store.httpd.attempts == {}, "an expired grant must not be sent"


def test_a_403_on_an_ALREADY_EXPIRED_grant_re_plans_instead_of_repudiating(
    store, tmp_path,
):
    """gw#570 / pgw#1004 C: an expired URL and a substituted claim are the SAME
    403 on the wire. `expires_at` is what tells them apart — and without it
    the safe classification (terminal) cost a whole re-plan pass."""
    data = make(CS, seed=19)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    store.httpd.forbid[store.path_of(data)] = 9
    # A grant inside the margin would never be sent; force the send by using a
    # zero margin here — this is the "expired between the check and the PUT"
    # race, which is the one the response classifier must catch.
    grant = store.grant(data, expires_at=rfc3339(-1))
    assert isinstance(
        cu._classify_put(grant, 403, "<Code>AccessDenied</Code>"), cu.GrantExpired)
    live = store.grant(data, expires_at=rfc3339(3600))
    assert isinstance(cu._classify_put(live, 403, "x"), ValueError)
    assert cu._classify_put(live, 200, "") is None
    assert isinstance(cu._classify_put(live, 500, "x"), cu._Transient)
    assert isinstance(cu._classify_put(live, 429, "x"), cu._Transient)
    assert isinstance(cu._classify_put(live, 408, "x"), cu._Transient)


# ---------------------------------------------------------------------------
# B. the liveness beat the whole data plane was missing
# ---------------------------------------------------------------------------


def test_every_uploaded_object_feeds_the_activity_counter_and_the_beat(
    store, tmp_path, monkeypatch,
):
    """`chunk_upload.py` imported neither `activity` nor `progress`, so a
    healthy multi-GB publish emitted the same silence a wedge does and the
    hub's 10-minute activity stall window could not tell them apart."""
    progress_mod.reset()
    beats: List[int] = []
    monkeypatch.setattr(activity_mod, "note_progress", lambda: beats.append(1))

    blocks = [make(CS, seed=s) for s in (31, 37, 41)]
    files = []
    for i, b in enumerate(blocks):
        p = tmp_path / f"b{i}.bin"
        p.write_bytes(b)
        files.append(p)
    index = {"sha256:" + sha(b): (files[i], 0, len(b)) for i, b in enumerate(blocks)}

    with activity_mod.running("convert_publish", "uploading"):
        rep = upload_grants([store.grant(b) for b in blocks],
                            lambda d: index[d], parallel=2)
        snaps = {s.name: s for s in progress_mod.snapshot()}

    assert rep.ok, rep.failures
    assert "upload:bytes" in snaps, snaps
    assert snaps["upload:bytes"].done == sum(len(b) for b in blocks)
    assert len(beats) == len(blocks)


# ---------------------------------------------------------------------------
# D. the semaphore bounds BUFFERS, not just sockets
# ---------------------------------------------------------------------------


def test_the_put_budget_is_taken_before_the_span_is_read(store, tmp_path, monkeypatch):
    """`_read_span` used to materialize a whole 64 MiB chunk BEFORE acquiring
    the PUT slot, so the thing that looked like it bounded memory did not."""
    held: List[bool] = []
    real_read = cu._read_span

    class _Watched:
        def __init__(self, inner):
            self._inner = inner
            self.depth = 0
            self._lock = threading.Lock()

        def acquire(self, *a, **kw):
            got = self._inner.acquire(*a, **kw)
            with self._lock:
                self.depth += 1
            return got

        def release(self):
            with self._lock:
                self.depth -= 1
            self._inner.release()

    watched = _Watched(threading.BoundedSemaphore(cu._PUT_BUDGET))
    monkeypatch.setattr(cu, "_put_slots", watched)
    monkeypatch.setattr(
        cu, "_read_span",
        lambda *a, **kw: (held.append(watched.depth > 0), real_read(*a, **kw))[1])

    data = make(CS, seed=43)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    rep = upload_grants([store.grant(data)], lambda d: (f, 0, len(data)), parallel=1)

    assert rep.ok, rep.failures
    assert held == [True], "the span must be read while holding a PUT slot"


def test_concurrency_defaults_use_the_uplink_and_bound_the_process(store, tmp_path):
    """pgw#1004 D. The numbers themselves are the assertion: 4-in-flight was
    well under what a pod NIC can do, and the whole point of raising them is
    that the ceiling stays a ceiling."""
    assert cu._DEFAULT_PARALLEL == 8
    assert cu._PUT_BUDGET == 16
    assert cu._put_slots._initial_value == cu._PUT_BUDGET
    # And the default is what upload_grants actually uses.
    data = make(64, seed=47)
    f = tmp_path / "s.bin"
    f.write_bytes(data)
    rep = upload_grants([store.grant(data)], lambda d: (f, 0, len(data)))
    assert rep.ok and rep.uploaded == 1


def test_a_realistic_chunk_size_keeps_peak_buffer_bounded(store, tmp_path):
    """No upload test has ever allocated the real 64 MiB `_read_span` buffer —
    every one of them monkeypatches CAS_CHUNK_SIZE_BYTES to 4096. One row at
    production size, asserting the process does not grow by more than a
    bounded multiple of the in-flight window."""
    import gc
    import resource

    size = cu.CAS_CHUNK_SIZE_BYTES  # the real 64 MiB
    data = make(size, seed=53)
    f = tmp_path / "big.bin"
    f.write_bytes(data)
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    rep = upload_grants([store.grant(data)], lambda d: (f, 0, size), parallel=1)

    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rep.ok, rep.failures
    assert store.httpd.objects[store.path_of(data)] == data
    # ru_maxrss is a high-water mark in KiB. One 64 MiB span in flight plus
    # requests' own framing must not balloon into hundreds of megabytes.
    grew_bytes = max(0, after - before) * 1024
    assert grew_bytes < 6 * size, f"peak RSS grew {grew_bytes} bytes for a {size}-byte span"
