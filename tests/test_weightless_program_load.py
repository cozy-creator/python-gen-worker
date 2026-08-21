from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from gen_worker.serving import weightless_program


@pytest.fixture(autouse=True)
def _restore_loader():
    import logging

    from torch.export.pt2_archive import _package

    noisy = logging.getLogger("torch._export.serde.serialize")
    was_disabled = noisy.disabled
    noisy.disabled = True

    weightless_program.uninstall()
    before = _package._load_state_dict
    assert not getattr(before, "_pgw1468", False), (
        "the pgw#1468 patch survived uninstall(); every stock-path assertion "
        "in this module would be measuring our own loader")
    try:
        yield
    finally:
        _package._load_state_dict = before
        noisy.disabled = was_disabled


def _weightless_archive(tmp_path: Path, shapes: list[tuple[int, int]]) -> Path:
    from torch._subclasses.fake_tensor import FakeTensorMode

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for index, (out_features, in_features) in enumerate(shapes):
                setattr(
                    self,
                    f"layer{index}",
                    torch.nn.Linear(in_features, out_features, bias=False,
                                    dtype=torch.bfloat16),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            total = x
            for index in range(len(shapes)):
                total = total + getattr(self, f"layer{index}").weight.sum()
            return total

    with FakeTensorMode(allow_non_fake_inputs=True):
        model = Model()
        example = torch.randn(2, dtype=torch.bfloat16)
        exported = torch.export.export(model, (example,))

    path = tmp_path / "graph.pt2"
    torch.export.save(exported, str(path))
    return path


def test_the_derive_writes_a_zero_byte_payload_for_EVERY_parameter(
    tmp_path: Path,
) -> None:
    """The premise."""
    archive = _weightless_archive(tmp_path, [(320, 36), (1280, 320), (1280, 2560)])

    with zipfile.ZipFile(archive) as bundle:
        payloads = {
            name.rsplit("/", 1)[-1]: bundle.getinfo(name).file_size
            for name in bundle.namelist()
            if "/weights/" in name and not name.endswith(".json")
        }
        config_name = next(
            name for name in bundle.namelist()
            if name.endswith("model_weights_config.json")
        )
        config = json.loads(bundle.read(config_name))["config"]

    assert payloads == {weightless_program.SHARED_PAYLOAD: 0}, (
        "the derive is expected to write exactly one, EMPTY, weight payload; "
        f"got {payloads}"
    )
    assert {entry["path_name"] for entry in config.values()} == {
        weightless_program.SHARED_PAYLOAD
    }
    assert {entry["tensor_meta"]["storage_offset"]["as_int"]
            for entry in config.values()} == {0}
    assert len(config) == 3


def test_a_STOCK_load_of_a_real_weightless_blob_FAILS(tmp_path: Path) -> None:
    """The red arm: this is the failure `mint_child` hit on every derive blob."""
    archive = _weightless_archive(tmp_path, [(320, 36), (1280, 320), (1280, 2560)])

    with pytest.raises(RuntimeError) as failure:
        torch.export.load(str(archive))
    assert "deserializ" in str(failure.value).lower()


def test_the_rebuilt_load_SUCCEEDS_and_gives_each_parameter_its_own_storage(
    tmp_path: Path,
) -> None:
    """The green arm, and `distinct storages` is the half that matters."""
    shapes = [(320, 36), (1280, 320), (1280, 2560)]
    archive = _weightless_archive(tmp_path, shapes)

    weightless_program.install()
    exported = torch.export.load(str(archive))

    state = exported.state_dict
    assert len(state) == len(shapes)
    assert sorted(tuple(v.shape) for v in state.values()) == sorted(shapes)
    assert {str(v.device) for v in state.values()} == {"cpu"}
    assert {v.dtype for v in state.values()} == {torch.bfloat16}
    assert len({v.data_ptr() for v in state.values()}) == len(shapes)
    assert all(isinstance(v, torch.nn.Parameter) for v in state.values())


def test_a_UNIFORM_shape_blob_hides_the_defect_which_is_why_it_survived(
    tmp_path: Path,
) -> None:
    archive = _weightless_archive(tmp_path, [(1280, 320)] * 3)

    from torch.export.pt2_archive import _package

    assert not getattr(_package._load_state_dict, "_pgw1468", False)

    exported = torch.export.load(str(archive))
    pointers = {v.data_ptr() for v in exported.state_dict.values()}
    assert len(exported.state_dict) == 3
    assert len(pointers) == 1
