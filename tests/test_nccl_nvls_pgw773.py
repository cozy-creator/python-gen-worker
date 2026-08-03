"""pgw#773 residual (live): NVLS multicast must be off before NCCL builds a comm.

Measured on a 4xH100-80GB-HBM3 SXM pod (NCCL 2.29.7, NV18): the group forms and
CP installs, then the FIRST all-to-all of EVERY arm dies with
``ncclUnhandledCudaError`` / *"Failed to bind NVLink SHARP (NVLS) Multicast
memory ... CUDA error 401"*. Sequence parallelism did not work at all on a stock
4-GPU pod. Nothing on a CPU rig can see this — gloo has no NVLS.

Layer exercised: `parallel.group.init_rank` (the one function every rank of
every group runs before any communicator exists).

Contract UPDATED by pgw#929 §1.17 (AMBIGUOUS #3): the write is UNCONDITIONAL.
pgw#773's respect-if-set behaviour was superseded, not regressed — see
`test_nvls_is_off_whatever_the_image_says` for the argument and for the
condition the override's removal has to keep meeting.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="needs torch.distributed with the gloo backend",
)

from gen_worker.parallel.group import (  # noqa: E402
    _NVLS_ENV,
    RankSpec,
    _refuse_nvls_multicast,
    init_rank,
)


def test_nvls_is_off_whatever_the_image_says(monkeypatch, caplog) -> None:
    """pgw#929 §1.17 AMBIGUOUS #3 supersedes pgw#773's respect-if-set default.

    pgw#773 wrote ``0`` only when the variable was UNSET and called that "a
    default, not an override". It never argued for the override: the issue was
    a live CUDA-401 bug hunt, and respect-if-set was the incidental shape of
    the one-line fix, not a considered escape hatch with a named user. pgw#929
    re-adjudicated the same variable and ruled it a LIBRARY-ADAPTER HANDOFF
    rather than operator configuration, because the failure it guards is TOTAL
    (every all-to-all of every arm dies) rather than gradual — so an env
    inherited from an image can silently take sequence parallelism to zero.

    This test therefore pins the ruling's own acceptance criterion: start at
    ``1``, prove the adapter overwrites it to ``0`` before NCCL can observe it.
    """
    monkeypatch.delenv(_NVLS_ENV, raising=False)
    _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "0"

    monkeypatch.setenv(_NVLS_ENV, "1")
    with caplog.at_level("WARNING", logger="gen_worker.parallel.group"):
        _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "0", "an image default may not re-enable NVLS"

    # The losing half of the ruling still has to hold: removing an escape hatch
    # is only acceptable while the removal REPORTS itself. Whoever set the
    # variable learns that it was dropped, what it was, and where the real
    # route runs — never a silent inversion of what they asked for.
    assert caplog.records, "a dropped operator override must not be silent"
    msg = caplog.records[-1].getMessage()
    assert _NVLS_ENV in msg and "'1'" in msg
    assert "new issue" in msg, "the warning must name the route, not just the refusal"


@pytest.mark.parametrize("preset", ["1", "0", "", "true", "TRUE", "yes", "2", " 1 "])
def test_no_image_value_can_revive_nvls(monkeypatch, preset: str) -> None:
    """The ratchet, stated behaviourally rather than over source text.

    The override is back the moment ANY incoming value survives, so the pin is
    that every one of them lands on ``0`` — including the truthy spellings a
    hand-written Dockerfile actually uses. A branch reintroduced anywhere on
    the read side shows up here as a value that is not ``0``.
    """
    monkeypatch.setenv(_NVLS_ENV, preset)
    _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "0", f"{preset!r} survived the adapter"


def test_forming_a_rank_sets_it_before_any_communicator_exists(monkeypatch) -> None:
    # A rank group of one over gloo: `init_rank` is the same call every NCCL
    # rank makes, and it must have decided NVLS before the PG is constructed.
    monkeypatch.delenv(_NVLS_ENV, raising=False)
    import torch.distributed as dist

    spec = RankSpec(0, 1, 0, "127.0.0.1", 0, "gloo", group_name="nvls-test")
    pg = init_rank(spec, store=dist.HashStore())
    try:
        assert os.environ[_NVLS_ENV] == "0"
    finally:
        dist.destroy_process_group(pg)
