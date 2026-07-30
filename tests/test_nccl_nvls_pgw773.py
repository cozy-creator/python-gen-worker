"""pgw#773 residual (live): NVLS multicast must be off before NCCL builds a comm.

Measured on a 4xH100-80GB-HBM3 SXM pod (NCCL 2.29.7, NV18): the group forms and
CP installs, then the FIRST all-to-all of EVERY arm dies with
``ncclUnhandledCudaError`` / *"Failed to bind NVLink SHARP (NVLS) Multicast
memory ... CUDA error 401"*. Sequence parallelism did not work at all on a stock
4-GPU pod. Nothing on a CPU rig can see this — gloo has no NVLS.

Layer exercised: `parallel.group.init_rank` (the one function every rank of
every group runs before any communicator exists).
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


def test_nvls_is_off_unless_the_image_says_otherwise(monkeypatch) -> None:
    monkeypatch.delenv(_NVLS_ENV, raising=False)
    _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "0"

    # An image that CAN bind multicast keeps its own answer: this decides a
    # default, it is not an override.
    monkeypatch.setenv(_NVLS_ENV, "1")
    _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "1"


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
