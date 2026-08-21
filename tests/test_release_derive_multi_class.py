"""pgw#1650 — a release derives EVERY compile-marking model class.

Paul, 2026-08-21: *"Of course both qwen image and qwen image edit can exist in
the same endpoint. Why wouldn't they be able to? Just compile each component
and swap them in and out of the pipeline."*

The REAL derive runs here — the shipped CLI, the shipped trace session, the
shipped document writer — over a two-class endpoint on ONE lane with a separate
checkpoint tree per class, which is the qwen-image shape exactly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
LANE = "sd15.diffusers@1+plain.f32@1"

LOCK = (
    'version = 1\n'
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def primary_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import sys

    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("primary-config-only"))


@pytest.fixture(scope="module")
def edit_tree(primary_tree: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A DIFFERENT checkpoint: same family, a wider unet.

    The width is what makes this tree falsifiable — a derive that quietly used
    the primary tree for both classes produces the same graph twice, and the
    disjointness assertion below fails.
    """

    tree = tmp_path_factory.mktemp("edit-root") / "config-only"
    shutil.copytree(primary_tree, tree)
    config_path = tree / "unet" / "config.json"
    config = json.loads(config_path.read_text())
    config["block_out_channels"] = [8, 24]
    config_path.write_text(json.dumps(config))
    return tree


def _derive(module: str, tree: Path, out: Path, *extra: str) -> int:
    from gen_worker.cli import main

    lockfile = out.parent / "uv.lock"
    lockfile.write_text(LOCK)
    return main([
        "release", "derive",
        "--dir", str(FIXTURES),
        "--module", module,
        "--checkpoint", str(tree),
        "--lockfile", str(lockfile),
        "--out", str(out),
        *extra,
    ])


def test_a_release_derives_every_compile_marking_class(
    primary_tree: Path, edit_tree: Path, tmp_path: Path
) -> None:
    out = tmp_path / "release.json"
    assert _derive(
        "two_class_endpoint", primary_tree, out,
        "--checkpoint", f"EditModel={edit_tree}",
    ) == 0
    document = json.loads(out.read_bytes())

    # BOTH classes are the release, and the endpoint name says so.
    assert document["endpoint"].endswith(":EditModel+PrimaryModel")
    classes = {row["class"]: row for row in document["classes"]}
    assert sorted(classes) == ["EditModel", "PrimaryModel"]

    # Each class states its own graph set, under its own lane row.
    per_class: dict[str, set[str]] = {}
    for name, row in classes.items():
        (lane,) = row["graphs"]["lanes"]
        assert lane["contract"] == LANE, name
        assert lane["graphs"], f"{name} derived no graph"
        assert {record["target"] for record in lane["graphs"]} == {"unet"}, name
        assert row["lane_contracts"][LANE]["stamp"] == LANE
        per_class[name] = {record["graph"] for record in lane["graphs"]}

    # A separate tree per class is REAL: same lane, same target, disjoint
    # graphs, because the two unets are not the same module.
    assert per_class["EditModel"].isdisjoint(per_class["PrimaryModel"])

    # BOTH entrypoints reach the document. Master derived one class and DROPPED
    # every entrypoint that class did not own, which is what took `edit` with
    # it.
    assert sorted(document["entrypoints"]) == ["edit", "generate"]
    assert document["entrypoints"]["edit"]["model_slots"] == {"model": "EditModel"}
    assert document["entrypoints"]["generate"]["traced_passes"] >= 1
    assert document["entrypoints"]["edit"]["traced_passes"] >= 1

    # The release-wide view is the UNION over the classes, on ONE lane row —
    # the hub keys `release_compiled_graph_documents.lanes` by the stamp alone.
    (merged,) = document["graphs"]["lanes"]
    assert merged["contract"] == LANE
    assert {record["graph"] for record in merged["graphs"]} == (
        per_class["EditModel"] | per_class["PrimaryModel"]
    )
    assert merged["unobserved_targets"] == []

    # `lane_contracts` keys BY STAMP and its entry stamps itself the same, or
    # the hub refuses `release_compiled_graphs_invalid_lane`. The `demand` row
    # it carries is the largest worst case, and it NAMES the class it is.
    (contract_key,) = document["lane_contracts"]
    assert contract_key == LANE
    entry = document["lane_contracts"][LANE]
    assert entry["stamp"] == LANE
    assert entry["worst_case_class"] == "EditModel"
    assert entry["demand"]["worst_case_request_bytes"] == max(
        row["lane_contracts"][LANE]["demand"]["worst_case_request_bytes"]
        for row in classes.values()
    )

    # One model type across both classes, so the release still states one.
    assert document["model_type"] == "SDXL"
    assert document["checkpoint_defaults_schema"] is not None


def test_two_classes_and_no_compile_mark_still_refuses(
    primary_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal that SURVIVES: no mark anywhere, so no subject is readable."""

    out = tmp_path / "release.json"
    assert _derive("two_unmarked_endpoint", primary_tree, out) == 1
    stderr = capsys.readouterr().err
    assert "has more than one model class" in stderr
    assert "NONE of them marks" in stderr
    assert not out.exists()


def test_a_checkpoint_tree_nothing_loads_from_is_a_typo(
    primary_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A key that names neither a class nor a slot used to be silently ignored."""

    out = tmp_path / "release.json"
    assert _derive(
        "two_class_endpoint", primary_tree, out,
        "--checkpoint", f"EditModal={primary_tree}",
    ) == 1
    stderr = capsys.readouterr().err
    assert "EditModal" in stderr and "EditModel" in stderr
