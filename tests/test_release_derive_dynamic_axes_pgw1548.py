from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401
import torch  # noqa: E402

from gen_worker.release.derive import dynamic_dim_policy  # noqa: E402
from gen_worker.serving.lane_spec import (  # noqa: E402
    DYNAMIC,
    STATIC,
    LaneDeclarationError,
    parse_shapes,
)

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    'version = 1\n'
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def config_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import sys

    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    tree: Path = tiny_tree.save_config_only(tmp_path_factory.mktemp("fan-tree"))
    return tree


def _lane(tree: Path, out: Path, module: str) -> dict:
    from gen_worker.cli import main

    lockfile = out.parent / f"{module}-uv.lock"
    lockfile.write_text(LOCK)
    code = main(
        [
            "release", "derive",
            "--dir", str(FIXTURES),
            "--module", module,
            "--checkpoint", str(tree),
            "--lockfile", str(lockfile),
            "--out", str(out),
        ]
    )
    assert code == 0, f"derive {module} failed"
    document = json.loads(out.read_bytes())
    (lane,) = document["graphs"]["lanes"]
    return dict(lane)


@pytest.fixture(scope="module")
def lanes(config_only_tree: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    room = tmp_path_factory.mktemp("fan-derives")
    return {
        arm: _lane(config_only_tree, room / f"{arm}.json", module)
        for arm, module in (
            ("static", "static_axes_endpoint"),
            ("aspect", "dynamic_axes_endpoint"),
        )
    }


def test_the_static_declaration_bakes_the_whole_fan(lanes: dict) -> None:
    """3 aspects x 2 CFG modes = six concrete specializations, declared."""

    graphs = lanes["static"]["graphs"]
    assert len(graphs) == 6
    for record in graphs:
        assert record["ingress"]["symbols"] == {}
        for row in record["ingress"]["inputs"]:
            assert all(isinstance(dim, int) for dim in row["shape"])


def test_the_declared_dynamic_aspect_collapses_the_fan(lanes: dict) -> None:
    """The SAME program, the SAME payload axes, ONE word different in the header: 6 keys become 2."""

    graphs = lanes["aspect"]["graphs"]
    assert len(graphs) == 2

    for record in graphs:
        sample = next(
            row for row in record["ingress"]["inputs"] if row["name"] == "sample"
        )
        batch, channels, height, width = sample["shape"]
        assert channels == 4
        assert batch in (1, 2), "CFG/batch stays CONCRETE — permanently static"
        assert isinstance(height, str) and isinstance(width, str)
        symbols = record["ingress"]["symbols"]
        assert symbols[height] == [48, 80] and symbols[width] == [48, 80]
        assert height != width, "H and W are two degrees of freedom"

    assert lanes["static"]["contract"] == lanes["aspect"]["contract"]


def test_the_dynamic_records_still_dispatch_every_observed_shape(
    lanes: dict,
) -> None:
    """A collapsed record must ADMIT what the fan it replaced admitted."""

    from gen_worker._vendor.torchcg.adopt import _matches
    from gen_worker._vendor.torchcg.document import GraphRecord
    from gen_worker._vendor.torchcg.ingress import CallIngress

    raw = lanes["aspect"]["graphs"][0]
    record = GraphRecord(
        graph=raw["graph"],
        target=raw["target"],
        ingress=CallIngress.decode(raw["ingress"]),
    )
    rows = {row.name: row for row in record.ingress.inputs}
    text = rows["encoder_hidden_states"]

    def call(batch: int, height: int, width: int) -> bool:
        return _matches(
            record,
            (),
            {
                "sample": torch.zeros(
                    (batch, 4, height, width),
                    dtype=getattr(torch, rows["sample"].dtype),
                ),
                "timestep": torch.zeros(
                    (), dtype=getattr(torch, rows["timestep"].dtype)
                ),
                "encoder_hidden_states": torch.zeros(
                    (batch, int(text.shape[1]), int(text.shape[2])),
                    dtype=getattr(torch, text.dtype),
                ),
            },
        )

    mine = int(rows["sample"].shape[0])
    for height, width in ((64, 64), (80, 48), (48, 80)):
        assert call(mine, height, width), f"{mine}x{height}x{width}"
    assert not call(4, 64, 64)
    assert not call(mine, 96, 96)


def test_the_collapsed_record_is_keyed_at_the_LANES_dtype(lanes: dict) -> None:

    for record in lanes["aspect"]["graphs"]:
        for row in record["ingress"]["inputs"]:
            if row["name"] in ("sample", "encoder_hidden_states"):
                assert row["dtype"] == "bfloat16"


def test_an_unknown_axis_name_REFUSES_rather_than_defaulting() -> None:
    """The refusal moved to the DECLARATION, where the author can read it."""

    with pytest.raises(LaneDeclarationError, match="not a shape axis"):
        parse_shapes("X", {"aspect": STATIC, "spatial": DYNAMIC},
                     marks_compile=True)


def test_batch_CANNOT_be_declared_dynamic_in_either_direction() -> None:

    with pytest.raises(LaneDeclarationError, match="PERMANENTLY STATIC"):
        parse_shapes("X", {"aspect": STATIC, "batch": DYNAMIC},
                     marks_compile=True)
    with pytest.raises(LaneDeclarationError, match="PERMANENTLY STATIC"):
        parse_shapes("X", {"aspect": STATIC, "batch": STATIC},
                     marks_compile=True)


def test_declared_shape_axes_always_trace_symbolically() -> None:
    """pgw#1603: STATIC no longer means an absent policy — the trace is
    symbolic either way, and STATIC means the buckets are STAMPED from the
    parent (``static_bind_declared``). Only an axis-free class stays on the
    per-bucket trace path."""

    from gen_worker.release.derive import static_bind_declared

    assert dynamic_dim_policy({}) is None
    assert static_bind_declared({}) is False
    for declared in ({"aspect": STATIC}, {"aspect": DYNAMIC}):
        policy = dynamic_dim_policy(declared)
        assert policy is not None
        assert policy("unet", "sample", 0) is False, "batch is NEVER offered"
        assert policy("unet", "sample", 1) is False
        assert policy("unet", "sample", 2) is True
        assert policy("unet", "sample", 3) is True
    assert static_bind_declared({"aspect": STATIC}) is True
    assert static_bind_declared({"aspect": DYNAMIC}) is False
