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
    monkeypatch.delenv(_NVLS_ENV, raising=False)
    _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "0"

    monkeypatch.setenv(_NVLS_ENV, "1")
    with caplog.at_level("WARNING", logger="gen_worker.parallel.group"):
        _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "0", "an image default may not re-enable NVLS"

    assert caplog.records, "a dropped operator override must not be silent"
    msg = caplog.records[-1].getMessage()
    assert _NVLS_ENV in msg and "'1'" in msg
    assert "new issue" in msg, "the warning must name the route, not just the refusal"


@pytest.mark.parametrize("preset", ["1", "0", "", "true", "TRUE", "yes", "2", " 1 "])
def test_no_image_value_can_revive_nvls(monkeypatch, preset: str) -> None:
    """The ratchet, stated behaviourally rather than over source text."""
    monkeypatch.setenv(_NVLS_ENV, preset)
    _refuse_nvls_multicast()
    assert os.environ[_NVLS_ENV] == "0", f"{preset!r} survived the adapter"


def test_forming_a_rank_sets_it_before_any_communicator_exists(monkeypatch) -> None:
    monkeypatch.delenv(_NVLS_ENV, raising=False)
    import torch.distributed as dist

    spec = RankSpec(0, 1, 0, "127.0.0.1", 0, "gloo", group_name="nvls-test")
    pg = init_rank(spec, store=dist.HashStore())
    try:
        assert os.environ[_NVLS_ENV] == "0"
    finally:
        dist.destroy_process_group(pg)
