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
    """The runaway: a peer declares ~4 GiB on the control socket."""
    ours, theirs = _socketpair()
    try:
        ours.settimeout(5.0)
        theirs.sendall(_frame_header(frames.T_HELLO_REQ, 0xFFFFFFFF))
        theirs.sendall(b"\x00" * 64)

        with pytest.raises(ValueError, match="exceeds"):
            _wait_boot_fatal_ack(ours)
    finally:
        ours.close()
        theirs.close()


def test_the_bound_is_the_module_s_own_constant_not_a_new_number():
    """§4.24: one threat, one number."""
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
    """The bound must not have made the reader stricter than the protocol."""
    ours, theirs = _socketpair()
    try:
        ours.settimeout(5.0)
        payload = b"x" * 1024
        theirs.sendall(_frame_header(frames.T_HELLO_REQ, len(payload)) + payload)
        theirs.sendall(_frame_header(frames.T_BOOT_FATAL_ACK, 0))

        _wait_boot_fatal_ack(ours)
    finally:
        ours.close()
        theirs.close()


def test_recv_exact_returns_bytes_and_accumulates_across_chunks():
    """The O(n^2) `buf += chunk` became a bytearray; the contract must not move."""
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
