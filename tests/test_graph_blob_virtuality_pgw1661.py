"""The graph-blob weight guard asks about STORAGE, never about TYPE (pgw#1661).

pgw#1198's ruling, re-proved at the site that re-broke it: `_assert_weights_free`
is the last thing between a hollow derive and a graph blob, and it read
`isinstance(..., FakeTensor)`. A wrapper subclass over fake data is a FakeTensor
to nobody, so minimax-h3's ~300 virtual denoiser weights priced out at ~23 GB of
"REAL" tensors on an 8 GiB card that reported `allocated=0.0GiB`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from gen_worker.release.derive import (  # noqa: E402
    _REAL_TENSOR_BYTES_CEILING,
    _assert_weights_free,
    DeriveError,
)
from test_wrapper_subclass_virtuality_pgw1198 import (  # noqa: E402
    Quantized,
    real_quantized,
)

#: bf16 over this shape is 128 MiB — twice the guard's ceiling, so a tensor of
#: this shape is heavy whichever side of the predicate it lands on.
HEAVY = (8192, 8192)


class _Program:
    """`torch.export.ExportedProgram`'s two weight-bearing holders, and nothing
    else — the guard reads `state_dict` and `constants` and asks no more."""

    def __init__(self, state_dict: Optional[Dict[str, Any]] = None,
                 constants: Optional[Dict[str, Any]] = None) -> None:
        self.state_dict = dict(state_dict or {})
        self.constants = dict(constants or {})


def _fake(shape: Any, dtype: Any) -> Any:
    with torch._subclasses.fake_tensor.FakeTensorMode():
        return torch.empty(tuple(shape), dtype=dtype)


def _quantized_over_fake(shape: Any) -> Any:
    """What a `setup()`-time quantizer leaves on a hollow denoiser."""
    return Quantized(
        _fake(shape, torch.float8_e4m3fn),
        _fake((shape[0], 1), torch.float32),
        torch.bfloat16,
    )


def test_a_wrapper_subclass_over_fake_data_is_not_a_weight() -> None:
    """h3's sink death, in one tensor."""
    from torch._subclasses.fake_tensor import FakeTensor

    virtual = _quantized_over_fake(HEAVY)
    assert not isinstance(virtual, FakeTensor), (
        "the premise: the object the old predicate could not see")
    assert virtual.numel() * virtual.element_size() > _REAL_TENSOR_BYTES_CEILING, (
        "and it prices heavy off its OUTER metadata, which is the whole hazard")

    _assert_weights_free(torch, _Program({"transformer_blocks.0.attn.to_q.weight": virtual}))


def test_a_plain_fake_tensor_is_not_a_weight() -> None:
    _assert_weights_free(torch, _Program({"w": _fake(HEAVY, torch.bfloat16)}))


def test_a_meta_tensor_is_not_a_weight() -> None:
    _assert_weights_free(torch, _Program({"w": torch.empty(HEAVY, dtype=torch.bfloat16,
                                                           device="meta")}))


def test_a_hollow_denoiser_of_three_hundred_such_weights_stores() -> None:
    """The measured shape of the refusal: 300 tensors, ~23 GB of nothing."""
    program = _Program({
        f"transformer_blocks.{i}.attn.to_q.weight": _quantized_over_fake(HEAVY)
        for i in range(300)
    })

    _assert_weights_free(torch, program)


def test_the_guard_still_FIRES_on_a_real_weight() -> None:
    """The fence must stay able to go red — that is the point of pgw#1198's fence."""
    program = _Program({"transformer_blocks.0.attn.to_q.weight":
                        torch.empty(HEAVY, dtype=torch.bfloat16)})

    with pytest.raises(DeriveError) as caught:
        _assert_weights_free(torch, program)
    assert "1 REAL tensor(s)" in str(caught.value)
    assert "transformer_blocks.0.attn.to_q.weight" in str(caught.value)


def test_the_guard_still_FIRES_on_a_wrapper_subclass_over_REAL_data() -> None:
    """Storage, not type, in the other direction too: the same subclass over
    real bytes IS weights, and a type-blind predicate would have to keep saying
    so."""
    program = _Program(constants={"quantized": real_quantized(HEAVY)})

    with pytest.raises(DeriveError) as caught:
        _assert_weights_free(torch, program)
    assert "constants.quantized" in str(caught.value)


def test_a_real_weight_is_still_found_BESIDE_three_hundred_virtual_ones() -> None:
    """The mixed case is the one a coarse fix would lose: exempting the holder
    because most of it is virtual would silently pass a checkpoint."""
    state = {
        f"transformer_blocks.{i}.attn.to_q.weight": _quantized_over_fake(HEAVY)
        for i in range(300)
    }
    state["transformer_blocks.7.ff.net.2.weight"] = torch.empty(
        HEAVY, dtype=torch.bfloat16)

    with pytest.raises(DeriveError) as caught:
        _assert_weights_free(torch, _Program(state))
    assert "1 REAL tensor(s)" in str(caught.value)
    assert "transformer_blocks.7.ff.net.2.weight" in str(caught.value)


def test_a_small_real_tensor_is_a_config_derived_buffer_and_passes() -> None:
    _assert_weights_free(torch, _Program({"rope.freqs": torch.empty(
        (16, 16), dtype=torch.float32)}))
