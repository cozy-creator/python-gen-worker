"""gw#574: compile arm() must not apply channels_last to rank-5 (causal/
video) VAEs — qwen's AutoencoderKLQwenImage crashed the ie#501 cell
producer with "required rank 4 tensor to use channels_last format"."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from gen_worker.compile_cache import _vae_supports_channels_last  # noqa: E402


class _Vae2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)


class _Vae3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 8, 3)


def test_rank4_vae_supports_channels_last() -> None:
    assert _vae_supports_channels_last(_Vae2D())


def test_rank5_vae_refuses_channels_last() -> None:
    assert not _vae_supports_channels_last(_Vae3D())


def test_gate_is_load_bearing_rank5_to_channels_last_raises() -> None:
    # The exact production crash the gate prevents.
    with pytest.raises(RuntimeError, match="rank 4"):
        _Vae3D().to(memory_format=torch.channels_last)


def test_non_module_is_refused_not_crashed() -> None:
    assert not _vae_supports_channels_last(object())
