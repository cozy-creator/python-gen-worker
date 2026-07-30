"""pgw#763: an oversized ``.to("cpu")`` is refused typed, not cgroup-killed.

te#138's shape end-to-end through the REAL patched ``torch.nn.Module.to``:
real torch modules, real cgroup files on disk (pgw#752 style), the real
probe. The only synthetic parts are the cgroup budget and the meta-device
weights (this box runs no models).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gen_worker import host_move_guard
from gen_worker.api.errors import HostRamMoveRefusedError

_GIB = 1 << 30


def _cgroup(tmp_path: Path, *, limit: int, current: int) -> tuple[Path, Path]:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(limit))
    (root / "memory.current").write_text(str(current))
    (root / "memory.stat").write_text(
        "anon {a}\nfile 0\nactive_file 0\ninactive_file 0\nshmem 0\n"
        "file_dirty 0\nfile_writeback 0\n".format(a=current))
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/\n")
    return root, proc


@pytest.fixture()
def guard(tmp_path, monkeypatch):
    """Arm the real guard against a synthetic 8GiB cgroup with 6GiB held
    (te#138's proportions: the resident pipeline nearly fills the budget)."""
    monkeypatch.delenv("GEN_WORKER_HOST_MOVE_GUARD", raising=False)
    root, proc = _cgroup(tmp_path, limit=8 * _GIB, current=6 * _GIB)
    monkeypatch.setattr(host_move_guard, "_probe_root", root)
    monkeypatch.setattr(host_move_guard, "_probe_self", proc)
    assert host_move_guard.install()
    return host_move_guard


def _big_meta_module() -> torch.nn.Module:
    # ~3.2GiB of fp32 weights that exist only as metadata: device.type="meta"
    # counts as incoming (not CPU-resident), and no RAM is ever allocated.
    with torch.device("meta"):
        return torch.nn.Linear(20480, 41984, bias=False)


def test_oversized_cpu_move_is_refused_typed(guard) -> None:
    """te#138's ``_free()``: a module bigger than the remaining budget asks
    for CPU. The kernel would SIGKILL; the guard must refuse by name."""
    module = _big_meta_module()
    with pytest.raises(HostRamMoveRefusedError) as exc:
        module.to("cpu")
    msg = str(exc.value)
    assert "host-RAM move refused" in msg
    assert exc.value.incoming_bytes >= 3 * _GIB
    # ~2GiB available (8 limit - 6 held), floored — the numbers are named.
    assert exc.value.available_bytes <= 3 * _GIB

    # .cpu() is the same door.
    with pytest.raises(HostRamMoveRefusedError):
        _big_meta_module().cpu()


def test_small_and_non_cpu_moves_pass_through(guard) -> None:
    """The guard must not tax or break ordinary moves: under-threshold
    modules skip the probe entirely, and non-CPU targets are untouched."""
    small = torch.nn.Linear(64, 64)  # CPU-resident already, tiny
    out = small.to("cpu")
    assert out is small
    # dtype-only .to() on a big meta module: not a CPU landing, no refusal.
    _big_meta_module().to(dtype=torch.bfloat16)


def test_kill_switch_disables(guard, monkeypatch) -> None:
    monkeypatch.setenv("GEN_WORKER_HOST_MOVE_GUARD", "0")
    module = _big_meta_module()
    # Guard steps aside; torch itself then refuses to materialize meta
    # tensors via .to() — any non-HostRamMoveRefusedError shape is fine.
    try:
        module.to("cpu")
    except HostRamMoveRefusedError:  # pragma: no cover
        pytest.fail("kill switch did not disable the guard")
    except Exception:
        pass
