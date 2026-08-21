from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    "version = 1\n"
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def primary_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("primary-config-only"))


@pytest.fixture(scope="module")
def aide_tree(primary_tree: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The auxiliary model's OWN tree: a bare UNet config, not a pipeline."""

    tree = tmp_path_factory.mktemp("aide-config-only")
    for name in ("config.json",):
        (tree / name).write_text((primary_tree / "unet" / name).read_text())
    return tree


def _derive(
    tmp_path: Path, *checkpoints: str
) -> tuple[int, Path]:
    from gen_worker.cli import main

    out = tmp_path / "release.json"
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(LOCK)
    argv = [
        "release", "derive",
        "--dir", str(FIXTURES),
        "--module", "two_slot_endpoint",
        "--lockfile", str(lockfile),
        "--out", str(out),
    ]
    for checkpoint in checkpoints:
        argv += ["--checkpoint", checkpoint]
    return main(argv), out


def test_two_slots_with_two_checkpoints_derive(
    primary_tree: Path, aide_tree: Path, tmp_path: Path
) -> None:
    """The whole issue: each slot loads from ITS OWN tree."""

    code, out = _derive(tmp_path, str(primary_tree), f"aide={aide_tree}")
    assert code == 0

    (lane,) = json.loads(out.read_bytes())["graphs"]["lanes"]
    assert lane["unobserved_targets"] == []
    assert {record["target"] for record in lane["graphs"]} == {"unet"}


def test_the_aide_pointed_at_the_PRIMARY_tree_refuses_naming_slot_and_tree(
    primary_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:

    code, _out = _derive(tmp_path, str(primary_tree))
    assert code == 1

    message = capsys.readouterr().err
    assert "slot 'aide'" in message
    assert str(primary_tree) in message
    assert "the PRIMARY checkpoint" in message
    # pgw#1650: the remedy names the CLASS, which is what owns a checkpoint —
    # two entrypoints can hold the same slot NAME over different classes (both
    # qwen arms take `model:`). The slot name still keys a tree.
    assert "--checkpoint-ref AideModel=<ref>" in message


def test_an_eager_only_slot_STILL_hydrates(
    primary_tree: Path, aide_tree: Path, tmp_path: Path
) -> None:
    """Skipping eager_only slots was the tempting shortcut and is wrong."""

    from gen_worker.models import __name__ as _models  # noqa: F401

    code, out = _derive(tmp_path, str(primary_tree), f"aide={aide_tree}")
    assert code == 0
    document = json.loads(out.read_bytes())
    assert len(document["graphs"]["lanes"]) == 1


def test_a_slot_named_twice_refuses(
    primary_tree: Path, aide_tree: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _out = _derive(
        tmp_path, str(primary_tree), f"aide={aide_tree}", f"aide={primary_tree}"
    )
    assert code == 2
    # pgw#1650: the key namespace is classes AND slots, so the message names
    # the KEY rather than asserting which of the two it was.
    assert "given twice for 'aide'" in capsys.readouterr().err


def test_only_secondary_slots_named_refuses(
    aide_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bare form is the primary's; omitting it is not a default."""

    code, _out = _derive(tmp_path, f"aide={aide_tree}")
    assert code == 2
    assert "the tree every other model loads from is the bare form" in (
        capsys.readouterr().err
    )


def test_the_slot_spelling_never_eats_a_ref_or_a_path() -> None:
    """`slot=` splits only on a plain identifier before the FIRST `=`."""

    from gen_worker.cli.lock import _split_slot

    assert _split_slot("owner/name@rev") == ("", "owner/name@rev")
    assert _split_slot("/tmp/a=b") == ("", "/tmp/a=b")
    assert _split_slot("rife=owner/name@rev") == ("rife", "owner/name@rev")
    assert _split_slot("rife=/tmp/tree") == ("rife", "/tmp/tree")


def test_an_auxiliary_checkpoint_moves_the_REUSE_KEY(tmp_path: Path) -> None:
    """Otherwise swapping rife under a stable primary reuses a stale trace."""

    from gen_worker.cli import endpoint_lock as el

    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(LOCK)
    common = dict(
        root=tmp_path,
        module_name="two_slot_endpoint",
        checkpoint_ref="owner/primary@1",
        trace_device="cuda",
        lockfile=lockfile,
    )
    bare = el.inputs_digest(**common)  # type: ignore[arg-type]
    assert bare == el.inputs_digest(**common, extra=())  # type: ignore[arg-type]
    first = el.inputs_digest(**common, extra=("aide=owner/rife@1",))  # type: ignore[arg-type]
    second = el.inputs_digest(**common, extra=("aide=owner/rife@2",))  # type: ignore[arg-type]
    assert bare != first
    assert first != second
