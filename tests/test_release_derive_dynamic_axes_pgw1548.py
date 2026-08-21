"""pgw#1548 + pgw#1599: the DECLARED aspect axis collapses the shape fan.

Through the REAL `gen-worker release derive` codepath, over two fixture
endpoints whose payload enumeration reproduces sd15's actual structure —
three aspect buckets x two CFG modes. sd15 ships 14 specializations (2 x 7)
and sdxl 18 (2 x 9); nothing declares those counts, they fall out of the
enumeration driving the marked UNet at a different shape each pass.

**pgw#1599 changed WHERE the choice is written, not what it does.** The
global `--dynamic-axes` CLI flag is DELETED: it was a whole-run switch, so it
could only ever be right for every model in the run at once, and it left no
record on the model of what was chosen or why. The choice is now
`shapes={"aspect": STATIC | DYNAMIC}` on the model class, written by the
author who measured what a symbolic aspect dim costs THIS model.

So the two arms here are two FIXTURES, identical but for that one word, and
the comparison is the controlled measurement pgw#1599 acceptance (d) asks
for: both spellings are expressible and NEITHER is presumed.

Two flag values that used to be tested are GONE and cannot come back:
`batch` and `all`. CFG/batch is a PERMANENTLY STATIC fork (Paul, 2026-08-20),
and this is now enforced ONE STEP EARLIER than it was — a difference worth
stating, because master reached the same answer from the other end.

tcg#78 (vendored just before this landed) made the derive REFUSE a
contradicted dynamic axis by name: torch specializes the sizes 0 and 1 rather
than reason about them symbolically, so it guards every dynamic dim `>= 2`,
and an axis observed at 1 AND 2 is contradicted by the graph's own guards the
moment it is exported. The artifact that came out of compiling one answered a
batch-1 call with a batch-2 tensor of garbage and raised nothing. Under that
fix, asking for `--dynamic-axes batch` cost a full derive and gave back the
whole 6-graph fan with a refusal logged.

pgw#1599 makes it unreachable instead: `batch` is not a declarable axis, so
the refusal happens at CLASS DEFINITION — before any author code runs, before
any trace — and the two costs the old path paid (a wasted derive, and an
author who could write the wrong thing at all) are both gone. The measured
grounds are unchanged and are why the ruling stands even if tcg#78 is ever
fixed: batch-dynamic removed ZERO specializations on the real sd15 endpoint
(14 -> 14).
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
    """The SAME program, the SAME payload axes, ONE word different in the
    header: 6 keys become 2. The CFG x2 survives in both, by ruling."""

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

    # The declaration is the ONLY difference between the two derives, and it
    # is visible in the document rather than inferred from the graph count.
    assert lanes["static"]["contract"] == lanes["aspect"]["contract"]


def test_the_dynamic_records_still_dispatch_every_observed_shape(
    lanes: dict,
) -> None:
    """A collapsed record must ADMIT what the fan it replaced admitted.

    Through the serving dispatcher's own predicate, not a re-implementation.
    """

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

    # This record is ONE of the two CFG buckets, so it admits its own batch
    # across the whole aspect range and refuses the other bucket's.
    mine = int(rows["sample"].shape[0])
    for height, width in ((64, 64), (80, 48), (48, 80)):
        assert call(mine, height, width), f"{mine}x{height}x{width}"
    # And refuses what it was never exported for — a dynamic record is a
    # range, so an unobserved shape still falls to eager.
    assert not call(4, 64, 64)
    assert not call(mine, 96, 96)


def test_the_collapsed_record_is_keyed_at_the_LANES_dtype(lanes: dict) -> None:
    """pgw#1567 holds through the collapse: bf16 lane, bf16 ingress."""

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
    """Paul, 2026-08-20: *"CFG stays a fork axes permanently."* The old flag
    offered `batch` and `all`; the declaration offers neither."""

    with pytest.raises(LaneDeclarationError, match="PERMANENTLY STATIC"):
        parse_shapes("X", {"aspect": STATIC, "batch": DYNAMIC},
                     marks_compile=True)
    # ...and not even redundantly as STATIC: it is not a declarable axis.
    with pytest.raises(LaneDeclarationError, match="PERMANENTLY STATIC"):
        parse_shapes("X", {"aspect": STATIC, "batch": STATIC},
                     marks_compile=True)


def test_static_is_the_absence_of_a_policy_and_not_an_empty_one() -> None:
    """STATIC must yield `None`, so torchcg takes the untouched static path."""

    assert dynamic_dim_policy({"aspect": STATIC}) is None
    assert dynamic_dim_policy({}) is None
    policy = dynamic_dim_policy({"aspect": DYNAMIC})
    assert policy("unet", "sample", 0) is False, "batch is NEVER offered"
    assert policy("unet", "sample", 2) is True
    assert policy("unet", "sample", 3) is True
    # Axis 1 is a channel or a sequence length: never offered.
    assert policy("unet", "sample", 1) is False
