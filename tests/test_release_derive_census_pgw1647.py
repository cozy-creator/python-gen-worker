"""The CONSTRUCTION CENSUS is RELEASE-BUILD DATA (pgw#1647, hub arm th#2281).

Moment one of three. The census is computed inside the endpoint image, on the
config-only tree, per declared lane — the pgw#1370 derive seam, which is already
the place where the tree and the IMAGE meet. That is the whole reason it binds
here and not at the tree alone: ties, computed buffers and quantizer swaps are
code x config facts decided by THIS image's transformers and diffusers, so the
tensorfs stamp (the BYTES' identity) cannot carry them and never could.

REFUSE ON TRACEBACK (Paul's derive ruling). A tree that declares a pipeline and
cannot be censused FAILS the build, named — never a soft row-marking, never a
green release with the reason in a log. The point of moving this question to
publish time is that a release which cannot say what module it builds must not
reach a card, and four of them did.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

from gen_worker.release.derive import (  # noqa: E402
    NO_PIPELINE_INDEX,
    DeriveError,
    derive_release,
)
from gen_worker.serving.streaming import census  # noqa: E402

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
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("census-config-only"))


@pytest.fixture(scope="module")
def lockfile(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("census-lock") / "uv.lock"
    path.write_text(LOCK)
    return path


def _endpoint(name: str) -> ModuleType:
    import importlib

    sys.path.insert(0, str(FIXTURES))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(FIXTURES))


def test_the_release_document_carries_the_census_in_th2281_s_ENVELOPE_pgw1647(
    config_only_tree: Path, lockfile: Path
) -> None:
    """Green — and the envelope is the one the hub actually binds.

    th#2281's `releasecensus.Decode` reads `v`, `pipeline_class` and
    `components` (a non-empty OBJECT) and treats everything inside a component
    as opaque. Anything else here would be refused
    `release_construction_census_unknown_version` at the door, which is a
    cross-repo break that no pgw-side test would otherwise catch.
    """
    result = derive_release(
        _endpoint("tiny_endpoint"),
        checkpoint_dir=config_only_tree,
        lockfile=lockfile,
    )
    document = json.loads(result.document)
    (tree_row,) = document["construction_censuses"]
    assert tree_row["owners"] == ["$primary"]
    row = tree_row["census"]
    assert row["v"] == census.CENSUS_VERSION
    assert row["kind"] == census.CENSUS_KIND
    assert row["pipeline_class"] == "StableDiffusionPipeline"
    assert isinstance(row["components"], dict) and row["components"]
    assert "digest" not in row, (
        "the census carries its own digest — a sha256 over the document it "
        "sits in is a second carrier of one fact (th#2287's law), and the hub "
        "computes it from the same bytes"
    )

    parsed = census.Census.from_document(row)
    assert {c.component for c in parsed.components} == {
        "unet", "vae", "text_encoder",
    }, sorted(c.component for c in parsed.components)
    assert parsed.tensor_count > 0
    for component in parsed.components:
        assert component.eval_mode is True, component.component
    assert "construction_census" not in document, (
        "the deleted primary-only field survived the per-tree hardcut"
    )


def test_the_census_is_LANE_INVARIANT_and_says_so_pgw1647(
    config_only_tree: Path, lockfile: Path
) -> None:
    """ONE census per config tree, not one per lane — and the reason is in the row.

    A lane's only effect on construction is the dtype it casts wide floats to,
    and that fact already has a precise owner (the lane contract, and
    `engine._assert_lane_dtype`, which names the tensor AND the dtype it holds).
    Copying it in would make the document lane-dependent to carry a second copy
    of an answer the code already gives exactly. So a lane-governed wide float
    records `"lane"`, everything else records its dtype, and the census stops
    moving when the lane does.
    """
    result = derive_release(
        _endpoint("tiny_endpoint"),
        checkpoint_dir=config_only_tree,
        lockfile=lockfile,
    )
    document = json.loads(result.document)
    parsed = census.Census.from_document(
        document["construction_censuses"][0]["census"])
    unet = parsed.by_component()["unet"]

    dtypes = {row.dtype for row in unet.tensors}
    assert census.LANE_DTYPE in dtypes, dtypes
    assert not (dtypes & {"float32", "float16", "bfloat16", "float64"}), (
        f"a lane-governed wide float recorded a concrete dtype: {dtypes} — the "
        f"census is lane-dependent again"
    )

    # The invariance is a MEASUREMENT, not a claim: the same tree censused at a
    # different lane dtype is byte-identical.
    import torch

    from gen_worker.serving.streaming import skeleton

    at_f32 = skeleton.build_modules(
        config_only_tree, compute_dtype=torch.float32).census()
    at_bf16 = skeleton.build_modules(
        config_only_tree, compute_dtype=torch.bfloat16).census()
    assert at_f32.canonical() == at_bf16.canonical()
    assert at_f32.digest == parsed.digest


def test_a_tree_that_is_not_a_pipeline_says_so_rather_than_guessing_pgw1647(
    tmp_path: Path, lockfile: Path
) -> None:
    """No `model_index.json` means the streaming loader never binds this tree.

    Recorded as a FACT, not as a silent absence and not as a refusal: whether a
    streaming-served release may ship without a census is the hub door's call
    (th#2281), and this side states what it found.
    """
    from gen_worker.release import derive as derive_mod

    row = derive_mod._construction_census(tmp_path, (("lane@1", None),))
    assert row == {"absent": NO_PIPELINE_INDEX}


def test_a_tree_that_declares_a_component_it_lacks_FAILS_the_build_pgw1647(
    config_only_tree: Path, lockfile: Path, tmp_path: Path
) -> None:
    """REFUSE ON TRACEBACK, and the refusal names the census.

    This is th#2265's own shape from the loader's side — an index declaring a
    component with no directory — reaching the publish door at $0 instead of an
    11-minute 105 GB materialization on a rented H200.
    """
    broken = tmp_path / "broken"
    shutil.copytree(config_only_tree, broken)
    shutil.rmtree(broken / "vae")

    with pytest.raises(DeriveError) as caught:
        derive_release(
            _endpoint("tiny_endpoint"),
            checkpoint_dir=broken,
            lockfile=lockfile,
        )
    message = str(caught.value)
    assert "CONSTRUCTION CENSUS" in message, message
    assert "vae" in message, message
    assert "pgw#1626" in message and "1644" in message, message


def test_a_census_the_image_cannot_compute_FAILS_the_build_pgw1647(
    config_only_tree: Path, lockfile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any traceback, not only the ones we predicted.

    A soft row-marking here would put the truth in a build log and ship the
    release, which is the exact failure mode the derive ruling exists to
    forbid.
    """
    from gen_worker.serving.streaming import census as census_mod

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise MemoryError("the image ran out of address space building a skeleton")

    monkeypatch.setattr(census_mod, "for_tree", _boom)

    with pytest.raises(DeriveError) as caught:
        derive_release(
            _endpoint("tiny_endpoint"),
            checkpoint_dir=config_only_tree,
            lockfile=lockfile,
        )
    assert "MemoryError" in str(caught.value)
    assert "CONSTRUCTION CENSUS" in str(caught.value)


def test_the_release_census_verifies_the_module_the_serve_path_builds_pgw1647(
    config_only_tree: Path, lockfile: Path
) -> None:
    """The claim the whole design rests on: it REPLAYS.

    Moment one produced this census. If moment three cannot replay it against
    the module the serving prepare seam builds from the same tree in the same
    image, then the release document is describing something nobody serves.
    """
    from gen_worker.serving.streaming import skeleton

    result = derive_release(
        _endpoint("tiny_endpoint"),
        checkpoint_dir=config_only_tree,
        lockfile=lockfile,
    )
    published = census.Census.from_document(
        json.loads(result.document)["construction_censuses"][0]["census"])

    import torch

    served = skeleton.build_modules(
        config_only_tree, compute_dtype=torch.float32).census()
    census.verify(published, served, where="tiny_endpoint")


def test_two_lanes_that_build_DIFFERENT_modules_fail_the_build_pgw1647(
    config_only_tree: Path, lockfile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariance is CHECKED, not assumed.

    One census for N lanes is only honest if every lane really does build the
    same module. So the derive builds one per lane and refuses if two disagree —
    a lane declares a dtype and a layout, not a different model, and publishing
    one of two disagreeing censuses would make the release document describe a
    module half of its lanes do not build.
    """
    from gen_worker.release import derive as derive_mod
    from gen_worker.serving.streaming import census as census_mod

    real = census_mod.for_tree
    seen: list[object] = []

    def _drifting(tree: object, *, compute_dtype: object = None) -> object:
        taken = real(tree, compute_dtype=compute_dtype)
        seen.append(compute_dtype)
        if len(seen) == 1:
            return taken
        head = taken.components[0]
        moved = census_mod.ComponentCensus(
            component=head.component, module_class="SomethingElse",
            tensors=head.tensors, ties=head.ties, quant_rule=head.quant_rule,
            quant_modules=head.quant_modules, eval_mode=head.eval_mode,
        )
        return census_mod.Census(
            (moved, *taken.components[1:]), pipeline_class=taken.pipeline_class)

    monkeypatch.setattr(census_mod, "for_tree", _drifting)

    with pytest.raises(DeriveError, match="build DIFFERENT"):
        derive_mod._construction_census(
            config_only_tree, (("a@1+plain.f32@1", None), ("b@1+plain.bf16@1", None)),
        )
