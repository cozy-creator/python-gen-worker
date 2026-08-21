"""pgw#1548: `--dynamic-axes` collapses the shape fan, one axis at a time.

Through the REAL `gen-worker release derive` codepath, over a fixture endpoint
whose payload enumeration reproduces sd15's actual structure — three aspect
buckets x two CFG modes. sd15 ships 14 specializations (2 x 7) and sdxl 18
(2 x 9); nothing declares those counts, they fall out of the enumeration
driving the marked UNet at a different shape each pass.

The static arm is the control and it stays exactly as it was: the default is
`off`, and a lock derived without the flag is the lock the fleet has today.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401
import torch  # noqa: E402

from gen_worker.release.derive import DeriveError, dynamic_dim_policy  # noqa: E402

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


def _lane(tree: Path, out: Path, axes: str) -> dict:
    from gen_worker.cli import main

    lockfile = out.parent / f"{axes}-uv.lock"
    lockfile.write_text(LOCK)
    code = main(
        [
            "release", "derive",
            "--dir", str(FIXTURES),
            "--module", "dynamic_axes_endpoint",
            "--checkpoint", str(tree),
            "--lockfile", str(lockfile),
            "--dynamic-axes", axes,
            "--out", str(out),
        ]
    )
    assert code == 0, f"derive --dynamic-axes {axes} failed"
    document = json.loads(out.read_bytes())
    (lane,) = document["graphs"]["lanes"]
    return dict(lane)


@pytest.fixture(scope="module")
def lanes(config_only_tree: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    room = tmp_path_factory.mktemp("fan-derives")
    return {
        axes: _lane(config_only_tree, room / f"{axes}.json", axes)
        for axes in ("off", "batch", "aspect", "all")
    }


def test_the_static_fan_is_the_default_and_is_unchanged(lanes: dict) -> None:
    """The control: 3 aspects x 2 CFG modes = six concrete specializations."""

    graphs = lanes["off"]["graphs"]
    assert len(graphs) == 6
    for record in graphs:
        assert record["ingress"]["symbols"] == {}
        for row in record["ingress"]["inputs"]:
            assert all(isinstance(dim, int) for dim in row["shape"])


def test_every_axis_dynamic_collapses_the_aspect_fan(lanes: dict) -> None:
    """Six specializations become TWO: the aspects collapse, CFG does not.

    AMENDED BY tcg#78. This asserted ONE graph, with the CFG axis symbolic
    over [1, 2] — and that graph could not be minted. Torch specializes the
    sizes 0 and 1 rather than reason about them symbolically, so it guards
    every dynamic dim `>= 2`; an axis observed at 1 and 2 is contradicted by
    the graph's own guards the moment it is exported, and the artifact that
    came out of compiling it answered a batch-1 call with a batch-2 tensor of
    garbage and raised nothing. The derive now refuses that axis by name.

    Refusing it costs the ASPECT collapse nothing, which is the point: the
    aspect axis is the one this program is for (sdxl 18 -> 2).
    """

    graphs = lanes["all"]["graphs"]
    assert len(graphs) == 2
    for record in graphs:
        sample = next(
            row for row in record["ingress"]["inputs"] if row["name"] == "sample"
        )
        batch, channels, height, width = sample["shape"]
        assert channels == 4
        assert batch in (1, 2), "the CFG axis is concrete in each record"
        symbols = record["ingress"]["symbols"]
        assert symbols[height] == [48, 80] and symbols[width] == [48, 80]
        assert height != width, "H and W are two degrees of freedom"
    assert {
        next(
            row for row in record["ingress"]["inputs"] if row["name"] == "sample"
        )["shape"][0]
        for record in graphs
    } == {1, 2}


def test_ONE_axis_at_a_time_is_the_acceptance_gates_shape(lanes: dict) -> None:
    """Paul's gate adopts per axis, so the derive must be able to do that.

    `aspect` collapses the three buckets and leaves the two CFG batches
    concrete (6 -> 2). WHICH axis moved is visible in the record.

    AMENDED BY tcg#78: `batch` now collapses NOTHING and says so, because that
    axis is the one torch's `>= 2` guard contradicts. The fan comes back whole
    (6 -> 6) — which is also what this lane measured on the real sd15 endpoint
    from the other direction: the batch axis removed ZERO specializations
    (14 -> 14) even when it appeared to work. The gate's SHAPE is intact; the
    axis that pays is `aspect`.
    """

    assert len(lanes["batch"]["graphs"]) == 6
    assert len(lanes["aspect"]["graphs"]) == 2

    for record in lanes["batch"]["graphs"]:
        assert record["ingress"]["symbols"] == {}
        shape = record["ingress"]["inputs"][0]["shape"]
        assert all(isinstance(dim, int) for dim in shape), "the axis was refused"

    for record in lanes["aspect"]["graphs"]:
        shape = record["ingress"]["inputs"][0]["shape"]
        assert shape[0] in (1, 2), "batch stays concrete"
        assert isinstance(shape[2], str) and isinstance(shape[3], str)


def test_the_dynamic_records_still_dispatch_every_observed_shape(
    lanes: dict,
) -> None:
    """A collapsed record must ADMIT what the fan it replaced admitted.

    Through the serving dispatcher's own predicate, not a re-implementation.
    """

    from gen_worker._vendor.torchcg.adopt import _matches
    from gen_worker._vendor.torchcg.document import GraphRecord
    from gen_worker._vendor.torchcg.ingress import CallIngress

    raw = next(
        candidate
        for candidate in lanes["all"]["graphs"]
        if next(
            row for row in candidate["ingress"]["inputs"] if row["name"] == "sample"
        )["shape"][0] == 2
    )
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

    for height, width in ((64, 64), (80, 48), (48, 80)):
        assert call(2, height, width), f"2x{height}x{width}"
    # And refuses what it was never exported for — a dynamic record is a
    # range, so an unobserved shape still falls to eager. Batch is concrete
    # here (tcg#78), so the other CFG mode is the OTHER record's business.
    assert not call(1, 64, 64)
    assert not call(4, 64, 64)
    assert not call(2, 96, 96)


def test_the_collapsed_record_is_keyed_at_the_LANES_dtype(lanes: dict) -> None:
    """pgw#1567 holds through the collapse: bf16 lane, bf16 ingress."""

    for record in lanes["all"]["graphs"]:
        for row in record["ingress"]["inputs"]:
            if row["name"] in ("sample", "encoder_hidden_states"):
                assert row["dtype"] == "bfloat16"


def test_an_unknown_axis_name_REFUSES_rather_than_defaulting() -> None:
    with pytest.raises(DeriveError, match="axis name"):
        dynamic_dim_policy("spatial")


def test_off_is_the_absence_of_a_policy_and_not_an_empty_one() -> None:
    """`off` must be `None`, so torchcg takes the untouched static path."""

    assert dynamic_dim_policy("off") is None
    assert dynamic_dim_policy("all")("unet", "sample", 0) is True
    assert dynamic_dim_policy("batch")("unet", "sample", 2) is False
    assert dynamic_dim_policy("aspect")("unet", "sample", 0) is False
    assert dynamic_dim_policy("aspect")("unet", "sample", 3) is True
    # Axis 1 is a channel or a sequence length: never offered.
    assert dynamic_dim_policy("all")("unet", "sample", 1) is False
