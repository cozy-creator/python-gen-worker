"""The pin fence reads a component's NARROWEST dtype, not what it is mostly.

pgw#1668 replaced a majority-by-tensor-count measure with a strict one, and in
doing so it disarmed this fence for exactly the case the fence exists for. The
old measure answered `bf16` for a component that was 20 tensors cast and 3 not,
so an fp32 pin refused it. The new one answers `mixed` — and `dtype_bits`
of `mixed` is 0, which makes `is_narrowing` False against everything. **A pin
that was fully violated was refused; a pin that was HALF violated published
silently.** The fence got strictly less able to fire the more of the component
survived, which is backwards.

The rollup is still what the checkpoint REPORTS — `mixed` is the honest label.
It is simply not a comparand: a pin is violated by any tensor below it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from gen_worker.convert.dtype_pins import (
    ComponentDtypePinViolation,
    component_dtypes_on_disk,
    component_narrowest_dtype,
    verify_produced_tree,
)

_W = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8}


def _safetensors(tensors: dict[str, tuple[str, int]]) -> bytes:
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, count) in tensors.items():
        end = offset + count * _W[dtype]
        header[name] = {"dtype": dtype, "shape": [count],
                        "data_offsets": [offset, end]}
        offset = end
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + bytes(offset)


def _tree(root: Path, vae: dict[str, tuple[str, int]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipeline",
        # AutoencoderKLWan carries an fp32 load pin in families.facts.
        "vae": ["diffusers", "AutoencoderKLWan"],
    }))
    (root / "vae").mkdir(exist_ok=True)
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(
        _safetensors(vae))
    return root


_ALL_F32 = {f"a{i}.weight": ("F32", 4096) for i in range(20)}
_HALF_CAST = {**{f"a{i}.weight": ("BF16", 4096) for i in range(20)},
              **{f"b{i}.weight": ("F32", 4096) for i in range(3)}}
_ALL_CAST = {f"a{i}.weight": ("BF16", 4096) for i in range(20)}


def test_a_HALF_violated_pin_is_refused_the_same_as_a_fully_violated_one(
    tmp_path: Path,
) -> None:
    """THE REGRESSION. Both trees narrowed an fp32-pinned VAE below its pin;
    only one of them used to be caught."""

    source = _tree(tmp_path / "source", _ALL_F32)

    for name, vae in (("half", _HALF_CAST), ("all", _ALL_CAST)):
        tree = _tree(tmp_path / name, vae)
        with pytest.raises(ComponentDtypePinViolation) as excinfo:
            verify_produced_tree(tree, source_dir=source)
        assert "'vae'" in str(excinfo.value)
        assert "pinned to fp32" in str(excinfo.value)

    # And the label the checkpoint carries is unchanged and still honest: the
    # half-cast tree is genuinely mixed, and saying so is the point of pgw#1668.
    assert component_dtypes_on_disk(tmp_path / "half") == {"vae": "mixed"}
    assert component_narrowest_dtype(tmp_path / "half" / "vae") == "bf16"


def test_a_component_that_honours_its_pin_still_publishes(tmp_path: Path) -> None:
    """The fence must not fire on the tree the cast is SUPPOSED to produce —
    a pinned component the cast stepped over, which is the normal outcome."""

    source = _tree(tmp_path / "source", _ALL_F32)
    tree = _tree(tmp_path / "produced", _ALL_F32)
    assert verify_produced_tree(tree, source_dir=source) == {"vae": "fp32"}
    assert component_narrowest_dtype(tree / "vae") == "fp32"


def test_a_source_that_was_ALREADY_below_the_pin_is_not_this_jobs_narrowing(
    tmp_path: Path,
) -> None:
    """Mirroring a tree that ships narrow is not narrowing it, and the source
    is read by the same narrowest rule — otherwise a mixed SOURCE would stop
    excusing a mixed produced tree and every such mirror would refuse."""

    source = _tree(tmp_path / "source", _HALF_CAST)
    tree = _tree(tmp_path / "produced", _HALF_CAST)
    assert verify_produced_tree(tree, source_dir=source) == {"vae": "mixed"}
