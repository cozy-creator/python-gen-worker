"""pgw#1648: tensorfs owns bytes; pgw hands it destination data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from gen_worker.serving.streaming.fill_client import Destination, HostFillClient


class _Reader:
    def __init__(self) -> None:
        self.call: tuple[Any, ...] = ()

    def fill_host_address(self, *args: Any, **kwargs: Any) -> object:
        self.call = (*args, kwargs)
        return object()


def test_destination_map_crosses_as_plain_data_only() -> None:
    destination = Destination(
        name="layer.weight",
        pointer=1234,
        capacity=4096,
        source_offset=8192,
        shape=(16, 32),
        element_bytes=2,
        layout="torch.contiguous@1",
    )
    annotations = get_type_hints(Destination)
    assert set(annotations) == {
        "name",
        "pointer",
        "capacity",
        "source_offset",
        "shape",
        "element_bytes",
        "layout",
    }
    assert all("torch" not in str(annotation).lower() for annotation in annotations.values())
    shape = annotations["shape"]
    assert get_origin(shape) in (tuple, None) or tuple in get_args(shape)

    reader = _Reader()
    result = HostFillClient().fill(reader, destination)

    assert result is not None
    assert reader.call == (
        "layer.weight",
        1234,
        4096,
        {"layout": "torch.contiguous@1"},
    )
    assert all(isinstance(value, (str, int, tuple, dict)) for value in reader.call)


def test_the_replaced_pgw_byte_plane_is_deleted() -> None:
    root = Path(__file__).parents[1] / "src/gen_worker/serving/streaming"
    assert not (root / "staging.py").exists()
    sources = "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py")))
    for dead in (
        "BridgeWeightStore",
        "StagingPool",
        "cudaMemcpyAsync",
        "class _Placement",
        "def _walk(",
    ):
        assert dead not in sources


def test_the_fill_seam_module_has_no_torch_import_or_type() -> None:
    import gen_worker.serving.streaming.fill_client as fill_client

    source = Path(fill_client.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.Tensor" not in source
