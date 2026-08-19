"""pgw#1468: the derive's weights-free blob must LOAD.

These drive real `torch.export.export` / `save` / `load` against a real pt2
archive — no mocks, no hand-written fixture archive. The archive under test is
produced the same way the derive produces one: export a module whose parameters
are FakeTensors, then save it.

CPU throughout. The defect is device-independent (measured on both cpu and
cuda), so nothing here needs a GPU.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from gen_worker.serving import weightless_program


@pytest.fixture(autouse=True)
def _restore_loader():
    """Undo `install()` around every test in this module.

    `install()` patches `torch.export`'s loader process-wide, which is right in
    production and wrong in a suite: without this, whichever test installs first
    silently converts every later "stock path" assertion into a patched-path
    assertion, and the red arm stops measuring anything. Snapshot-and-restore
    keeps each test's premise its own.
    """
    import logging

    from torch.export.pt2_archive import _package

    # torch 2.13.0 `_export/serde/serialize.py::deserialize_torch_artifact` logs
    #   log.warning("... type %s after initial failure: %s", type(artifact),
    #               exc_info=e)
    # — two `%s`, ONE positional arg (`exc_info` is a keyword). Formatting the
    # record raises `TypeError: not enough arguments for format string`, and
    # under pytest's log capture that surfaces as a test failure. It fires on
    # every successful load here, because these archives take the
    # `weights_only=False` fallback. Upstream's bug, not ours, and not
    # something this suite should assert on.
    noisy = logging.getLogger("torch._export.serde.serialize")
    was_disabled = noisy.disabled
    noisy.disabled = True

    # pgw#1485: UNINSTALL FIRST, then snapshot. Snapshotting alone restores
    # whatever was in place at setup — which, on an xdist worker where any
    # earlier test called `install()` (mint_child does, on import-and-run
    # paths), is the PATCHED loader. The stock arms then assert against our own
    # patch and the red arm stops being red. Two of them flipped to failing on
    # a run whose only change was test DISTRIBUTION.
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
    """A real pt2 archive with real metadata and NO weight bytes.

    Built by exporting under `FakeTensorMode`, which is what the derive's hollow
    session does — so the archive's degenerate shape is produced, never asserted.
    """
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
    """The premise. If this ever changes, the rest of this module is moot."""
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
    # Every parameter dedups onto that one empty file at offset 0 — which is
    # precisely why torch cannot size a storage that fits more than the first.
    assert {entry["path_name"] for entry in config.values()} == {
        weightless_program.SHARED_PAYLOAD
    }
    assert {entry["tensor_meta"]["storage_offset"]["as_int"]
            for entry in config.values()} == {0}
    assert len(config) == 3


def test_a_STOCK_load_of_a_real_weightless_blob_FAILS(tmp_path: Path) -> None:
    """The red arm: this is the failure `mint_child` hit on every derive blob.

    Non-uniform shapes are the discriminator — see the uniform-shape test below
    for why a same-shape check reports success and proves nothing.
    """
    archive = _weightless_archive(tmp_path, [(320, 36), (1280, 320), (1280, 2560)])

    with pytest.raises(RuntimeError) as failure:
        torch.export.load(str(archive))
    # torch wraps the cause; the storage-bounds violation is the thing that
    # makes this a payload problem rather than a schema one.
    assert "deserializ" in str(failure.value).lower()


def test_the_rebuilt_load_SUCCEEDS_and_gives_each_parameter_its_own_storage(
    tmp_path: Path,
) -> None:
    """The green arm, and `distinct storages` is the half that matters.

    Shapes alone would pass even on the broken path for a uniform model, with
    every parameter aliasing one buffer. Distinct `data_ptr`s are what say the
    tensors were rebuilt rather than re-aliased.
    """
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
    # `is_param: true` in the config has to come back as a real Parameter or the
    # export verifier rejects the program at the very last step.
    assert all(isinstance(v, torch.nn.Parameter) for v in state.values())


def test_a_UNIFORM_shape_blob_hides_the_defect_which_is_why_it_survived(
    tmp_path: Path,
) -> None:
    """Why the pgw#1465 evidence read green while no real model could load.

    Same-shaped parameters cannot exceed a storage sized from the first, so the
    stock path returns successfully — and returns every parameter aliased onto
    ONE buffer. This test exists to keep that trap documented in executable
    form: the round-trip a hand-written check would write is exactly the one
    that cannot fail.
    """
    archive = _weightless_archive(tmp_path, [(1280, 320)] * 3)

    # The autouse fixture guarantees the STOCK loader is in place here, so this
    # measures torch's own behaviour rather than ours.
    from torch.export.pt2_archive import _package

    assert not getattr(_package._load_state_dict, "_pgw1468", False)

    exported = torch.export.load(str(archive))
    pointers = {v.data_ptr() for v in exported.state_dict.values()}
    assert len(exported.state_dict) == 3
    # One storage for three parameters: "loads" without being intact.
    assert len(pointers) == 1
