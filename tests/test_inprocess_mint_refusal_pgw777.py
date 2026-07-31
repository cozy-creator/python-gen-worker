"""pgw#777/DPA-8: the in-process capture is REFUSED at G>1, never arbitrated.

`capture_env` moves the process-global TORCHINDUCTOR_CACHE_DIR and clears
inductor's latch for the whole interpreter — under G in-process execution
groups that lands mid-compile or mid-serve on G-1 sibling cards. The
DELEGATED mint (pgw#784) dissolves this by putting the capture in the mint
child's own process; when delegation is refused, a multi-group worker refuses
the in-process capture rather than pretending a process-global control plane
is per-group. (The full mint-once-adopt-N story belongs to the pgw#783
process split, where each group IS its own process.)
"""

from __future__ import annotations

import pytest

from gen_worker import fleet_cells as fc
from gen_worker import topology as topology_mod
from gen_worker.topology import ExecutionTopology

from test_fleet_cells import (  # noqa: F401 — fixture comes with it
    _Cfg,
    _Pipe,
    _clear_pending,
    _mintable,
    _publisher,
)


@pytest.fixture(autouse=True)
def _in_process_shape(monkeypatch):
    monkeypatch.setenv("GEN_WORKER_MINT_IN_PROCESS", "1")
    yield
    topology_mod.install_topology(None)


def test_multi_group_worker_refuses_the_in_process_capture(monkeypatch, tmp_path):
    _mintable(monkeypatch)
    captured = []
    monkeypatch.setattr(
        fc.cc, "begin_fleet_mint",
        lambda *a, **k: captured.append(a))
    topology_mod.install_topology(ExecutionTopology(gpu_count=4, group_degree=1))

    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher([]))

    assert captured == [], (
        "a G=4 in-process worker opened a process-global inductor capture "
        "under three sibling groups (pgw#777/DPA-8)")
    assert not outcome.armed
    assert outcome.self_mint is None, "no pending mint obligation was created"
    with fc._PENDING_LOCK:
        assert not fc._PENDING


def test_single_group_worker_still_opens_the_capture(monkeypatch, tmp_path):
    """The refusal is scoped: G == 1 (every pod today) keeps the exact
    in-process capture path pgw#784's `delegatable` falls back to."""
    _mintable(monkeypatch)
    topology_mod.install_topology(ExecutionTopology(gpu_count=1, group_degree=1))

    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher([]))

    assert outcome.armed
    assert outcome.self_mint is not None
