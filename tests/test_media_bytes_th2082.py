"""th#2082 — the worker's ONE read of ``RunJob.media_bytes``, fenced on the
value it RECEIVED rather than on the value the hub set.

``media_bytes`` (field 9, enum ``MediaBytes``) is the client's
``Prefer: bytes=inline|url`` media-delivery preference and nothing else. It was
spelled ``output_mode``, a token this module ALSO uses for the function's output
cardinality (``EndpointSpec.output_mode``, pgw#1320) — two unrelated questions,
one word, one file. th#2082 renames the wire enum; the field number and the enum
numbers are unchanged, so the wire is byte-identical.

WHY THE FENCE IS SHAPED THIS WAY. ``executor._legacy_order`` is the only reader
tree-wide. If it stops reading the field nothing raises: every
``Prefer: bytes=inline`` request quietly degrades to URL delivery, uploading
bytes the client asked to have handed back. There is no error, no metric and no
log to notice. So the arms below run a REAL worker over a REAL gRPC socket
against a REAL upload sink, and read the decision from two independent places —
the sink (were the bytes uploaded?) and the handler's own view of the Asset it
got back. The tensorhub half asserts the producing end
(``internal/orchestrator/grpc/media_bytes_th2082_test.go``).
"""

from __future__ import annotations

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.media_bytes_th2082 import MediaIn, MediaOut
from harness.upload_sink import (
    DedupUploadSink,
    reset_upload_sink,
    serve_upload_sink,
)

_ORG = "00000000-0000-0000-0000-000000000001"
_MODULES = ("harness.media_bytes_th2082",)


def _render(request_id: str, **run_job_kwargs: object) -> MediaOut:
    """Dispatch one small-media job and return what the handler observed."""
    reset_upload_sink()
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(modules=_MODULES, file_base_url=base_url) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id=request_id, attempt=1, function_name="render",
                input_payload=msgspec.msgpack.encode(MediaIn(text="x")),
                org=_ORG, capability_token="cap-token",
                **run_job_kwargs,  # type: ignore[arg-type]
            ))
            res = conn.wait_for(is_result_for(request_id)).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            return msgspec.msgpack.decode(res.inline, type=MediaOut)
    finally:
        httpd.shutdown()


def test_inline_preference_is_honoured_by_the_worker_that_received_it() -> None:
    """RED when the consumer stops reading field 9: the bytes get uploaded and
    the client's `Prefer: bytes=inline` is silently answered with a URL."""
    out = _render("r-2082-inline", media_bytes=pb.MEDIA_BYTES_INLINE)

    assert out.inline, (
        "the worker received MEDIA_BYTES_INLINE and still returned a ref-only "
        "Asset — `Prefer: bytes=inline` degraded to URL delivery, which is the "
        "SILENT failure this fence exists for"
    )
    assert not DedupUploadSink.requests_seen, (
        "MEDIA_BYTES_INLINE must skip the tensorhub upload entirely; the sink "
        f"was hit {len(DedupUploadSink.requests_seen)} time(s)"
    )
    assert out.size_bytes == 2048


def test_url_preference_is_honoured_by_the_worker_that_received_it() -> None:
    out = _render("r-2082-url", media_bytes=pb.MEDIA_BYTES_URL)

    assert not out.inline, "MEDIA_BYTES_URL must upload and return a ref"
    assert DedupUploadSink.requests_seen, (
        "MEDIA_BYTES_URL must really upload — a ref naming bytes that never "
        "left the process is pgw#767's defect"
    )


def test_unspecified_falls_to_the_workers_own_default() -> None:
    """No `Prefer:` header = no hub opinion; the worker's default (upload) wins.
    Asserted so an accidental INLINE default cannot pass as 'unset'."""
    out = _render("r-2082-unset")

    assert not out.inline
    assert DedupUploadSink.requests_seen


def test_the_rename_did_not_move_the_wire() -> None:
    """Field NUMBER 9 and enum numbers 0/1/2 are what travel. A worker on an
    older wheel keeps parsing a new hub's dispatch precisely because of this,
    so the rename is source-level — prove it rather than assert it in prose.
    The tensorhub half pins the identical bytes from the producing side."""
    wire = pb.RunJob(
        request_id="r-2082-wire", media_bytes=pb.MEDIA_BYTES_INLINE,
    ).SerializeToString()

    # tag = (9 << 3) | 0 (varint) = 0x48; MEDIA_BYTES_INLINE = 2.
    assert b"\x48\x02" in wire, (
        f"field 9 does not carry varint 2 on the wire ({wire!r}) — the number "
        "moved, which IS a wire break even though the rename was not"
    )

    received = pb.RunJob()
    received.ParseFromString(wire)
    assert received.media_bytes == pb.MEDIA_BYTES_INLINE
    assert int(pb.MEDIA_BYTES_UNSPECIFIED) == 0
    assert int(pb.MEDIA_BYTES_URL) == 1
    assert int(pb.MEDIA_BYTES_INLINE) == 2


# --- the rename is a hard cut, no alias (pre-launch) ------------------------
#
# The scan is narrow ON PURPOSE, twice over. `output_mode` still has a LIVE,
# unrelated meaning in this repo — `EndpointSpec.output_mode`, the function's
# output cardinality (pgw#1320) — so a fence on the bare token would be red on
# arrival and would have to be neutered to pass; what cannot come back is the
# WIRE enum's spelling, which is unambiguous. And it scans CODE only, because
# prose that explains a rename must be free to name what was renamed (th#2079).

def _grep(pattern: str) -> str:
    import subprocess
    done = subprocess.run(
        ["git", "grep", "--untracked", "-nI", "-e", pattern, "--",
         "*.py", "*.proto", ":!tests/test_media_bytes_th2082.py"],
        capture_output=True, text=True, check=False,
    )
    return done.stdout.strip()


def test_retired_wire_enum_spelling_is_gone() -> None:
    for retired in ("OUTPUT_MODE_", "OutputMode", "run.output_mode"):
        hits = _grep(retired)
        assert not hits, (
            f"th#2082: {retired!r} is back. The client's `Prefer: bytes=` "
            f"preference is `media_bytes`/`MediaBytes`; `EndpointSpec."
            f"output_mode` is a DIFFERENT question and keeps its own name "
            f"until pgw#1320:\n{hits}"
        )


def test_the_scanner_can_actually_find_things() -> None:
    """An absent-string fence passes just as well when its scanner is broken.
    Prove it on the token that IS present."""
    assert _grep("MEDIA_BYTES_INLINE"), (
        "th#2082 fence scanner found no `MEDIA_BYTES_INLINE` — the scanner is "
        "broken, so the absence fence above proves nothing"
    )
