"""pgw#1013 (§4.24 absence sweep): the boot-fatal ack reader honours the frame
bound its own module already enforces.

`procsplit/frames.py` declares `MAX_FRAME_BYTES = 128 MiB` and BOTH ends of the
normal path enforce it — `read_frame` refuses an oversized declaration and
`FrameWriter.frame` refuses to emit one. `child._wait_boot_fatal_ack` is a
hand-rolled reader for one message type, and it skipped the check: a 4-byte
big-endian length off the control socket could declare up to 4 GiB, and the
reader would sit there accumulating it.

This is the shape the bounds census could not see. The bound EXISTS and is
correctly stated; what was missing was its application at one site — so an
inventory of bounds finds nothing wrong, and only an inventory of READ SITES
does.

Driven through a real `AF_UNIX` socket against the real reader — no mock, no
monkeypatched constant.
"""

from __future__ import annotations

import socket
import threading

import pytest

from gen_worker.procsplit import frames
from gen_worker.procsplit.child import _recv_exact, _wait_boot_fatal_ack


def _socketpair():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    return a, b


def _frame_header(ftype: int, length: int) -> bytes:
    return bytes([ftype]) + length.to_bytes(4, "big")


def test_oversized_declared_frame_is_refused_before_it_is_read():
    """The runaway: a peer declares ~4 GiB on the control socket.

    RED (before the fix): the reader loops in _recv_exact accumulating whatever
    the peer sends, with no ceiling — the declared length is never compared to
    anything.
    """
    ours, theirs = _socketpair()
    try:
        ours.settimeout(5.0)
        # Declare the largest length the 4-byte field can express, then send a
        # trickle. A correct reader refuses on the DECLARATION and never waits
        # for the bytes.
        theirs.sendall(_frame_header(frames.T_HELLO_REQ, 0xFFFFFFFF))
        theirs.sendall(b"\x00" * 64)

        with pytest.raises(ValueError, match="exceeds"):
            _wait_boot_fatal_ack(ours)
    finally:
        ours.close()
        theirs.close()


def test_the_bound_is_the_module_s_own_constant_not_a_new_number():
    """§4.24: one threat, one number.

    The refusal must quote frames.MAX_FRAME_BYTES. If this site grew its own
    constant, the two could drift and the sibling readers would disagree about
    what a legal frame is.
    """
    ours, theirs = _socketpair()
    try:
        ours.settimeout(5.0)
        theirs.sendall(_frame_header(frames.T_HELLO_REQ, frames.MAX_FRAME_BYTES + 1))
        with pytest.raises(ValueError) as exc:
            _wait_boot_fatal_ack(ours)
        assert str(frames.MAX_FRAME_BYTES) in str(exc.value)
    finally:
        ours.close()
        theirs.close()


def test_a_legal_frame_still_passes_through_and_the_ack_is_seen():
    """The bound must not have made the reader stricter than the protocol.

    An unrelated frame ahead of the ack is legal (the docstring says so) and
    must still be skipped, then the ack must terminate the loop.
    """
    ours, theirs = _socketpair()
    try:
        ours.settimeout(5.0)
        payload = b"x" * 1024
        theirs.sendall(_frame_header(frames.T_HELLO_REQ, len(payload)) + payload)
        theirs.sendall(_frame_header(frames.T_BOOT_FATAL_ACK, 0))

        _wait_boot_fatal_ack(ours)  # returns cleanly
    finally:
        ours.close()
        theirs.close()


def test_recv_exact_returns_bytes_and_accumulates_across_chunks():
    """The O(n^2) `buf += chunk` became a bytearray; the contract must not move.

    Callers index and compare the result, so it has to stay `bytes`.
    """
    ours, theirs = _socketpair()
    try:
        ours.settimeout(5.0)

        def dribble():
            for i in range(8):
                theirs.sendall(bytes([i]) * 4)

        t = threading.Thread(target=dribble, daemon=True)
        t.start()
        got = _recv_exact(ours, 32)
        t.join(timeout=5)

        assert isinstance(got, bytes)
        assert len(got) == 32
        assert got[:4] == b"\x00\x00\x00\x00"
    finally:
        ours.close()
        theirs.close()
