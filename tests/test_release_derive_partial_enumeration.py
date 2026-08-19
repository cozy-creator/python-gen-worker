"""pgw#1449: one unenumerable entrypoint must not cost the whole module.

Verbatim, on minimax-h3, at master before this:

    derive error: @entrypoint generate_long: required payload field 'slots' of
    type list[minimax_h3.long_video.LongVideoSlot] cannot be auto-synthesized
    for the trace.

`generate_long` is one of three, and its unsynthesizable field stopped
`generate` and `reference_to_video` from being derived at all -- so
`gen-worker lock` died there and wrote NO lock, and even the enumerable
entrypoint's specializations never reached the document.

The derive is a pre-warming completeness aid, never a correctness gate (its
own words), so an entrypoint it cannot enumerate is SKIPPED-AND-STATED --
exactly as an enumerated combination the author refuses with
``ValidationError`` already is.
"""

from __future__ import annotations

import enum
import json
import sys
from pathlib import Path

import msgspec
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

#: The two h3-shaped signatures the enumerator cannot reach.
REFUSED = ("generate_long", "reference_to_video")


class Empty(enum.Enum):
    """An author defect, not an enumerability limit."""


class EmptyAxis(msgspec.Struct):
    mode: Empty


class NotAStruct:
    pass


class Unsynthesizable(msgspec.Struct):
    slots: list[int]


@pytest.fixture(scope="module")
def config_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("mixed-config-only"))


def _derive(tree: Path, out: Path) -> int:
    from gen_worker.cli import main

    lockfile = out.parent / "uv.lock"
    lockfile.write_text(LOCK)
    return main(
        [
            "release",
            "derive",
            "--dir",
            str(FIXTURES),
            "--module",
            "mixed_enumerability_endpoint",
            "--checkpoint",
            str(tree),
            "--lockfile",
            str(lockfile),
            "--out",
            str(out),
        ]
    )


def test_the_enumerable_entrypoint_derives_while_two_others_cannot(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """2 of 3 is strictly more useful than none -- and the module SUCCEEDS."""

    out = tmp_path / "release.json"
    assert _derive(config_only_tree, out) == 0

    document = json.loads(out.read_bytes())
    entrypoints = document["entrypoints"]
    assert sorted(entrypoints) == ["generate", "generate_long", "reference_to_video"]

    # The one that CAN be enumerated is fully derived: 2 Size values, and the
    # lane carries their graph specializations.
    assert entrypoints["generate"]["traced_passes"] == 2
    assert "unenumerable" not in entrypoints["generate"]
    (lane,) = document["graphs"]["lanes"]
    assert lane["contract"] == "tiny.diffusers-fp32@1"
    assert len(lane["graphs"]) == 2
    assert lane["unobserved_targets"] == []


def test_each_refused_entrypoint_is_STATED_in_the_document_not_dropped(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """Typed, per entrypoint, naming the field and its type.

    A silent drop would leave the hub publishing an envelope schema for an
    entrypoint with no traced coverage and no way to know that.
    """

    out = tmp_path / "release.json"
    assert _derive(config_only_tree, out) == 0
    entrypoints = json.loads(out.read_bytes())["entrypoints"]

    for name in REFUSED:
        row = entrypoints[name]
        assert row["traced_passes"] == 0
        assert row["unenumerable"]["reason"] == "payload_field_not_synthesizable"
        assert row["unenumerable"]["type"].startswith("list[")
        # Still published as API surface: the envelope schema and model slots
        # come from the SIGNATURE and never needed the enumeration.
        assert row["envelope_schema"]
        assert row["model_slots"] == {"model": "MixedModel"}

    assert entrypoints["generate_long"]["unenumerable"]["field"] == "slots"
    assert entrypoints["reference_to_video"]["unenumerable"]["field"] == "references"


def test_the_result_names_both_refusals_so_a_caller_need_not_parse_prose(
    config_only_tree: Path, tmp_path: Path
) -> None:
    """`gen-worker lock` prints these and puts the names in its JSON."""

    from gen_worker.release.derive import derive_release

    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(LOCK)
    sys.path.insert(0, str(FIXTURES))
    try:
        import importlib

        module = importlib.import_module("mixed_enumerability_endpoint")
    finally:
        sys.path.remove(str(FIXTURES))

    result = derive_release(
        module, checkpoint_dir=config_only_tree, lockfile=lockfile
    )

    assert [name for name, _reason in result.unenumerable_entrypoints] == list(REFUSED)
    for _name, reason in result.unenumerable_entrypoints:
        assert "cannot be auto-synthesized" in reason
    # The prose warning survives too -- it carries the author's remedy.
    assert any("NOT enumerated" in warning for warning in result.warnings)


def test_the_refusal_stays_NARROW_and_a_real_defect_still_fails_the_module() -> None:
    """The isolation must not become a blanket swallow.

    ``PayloadEnumerationRefused`` has exactly one raise site. An empty enum,
    a non-msgspec payload and an out-of-order signature are author defects
    and still take the module down -- catching ``DeriveError`` instead of the
    narrow subclass would have quietly turned all of them into "eager".
    """

    from gen_worker.release.derive import (
        DeriveError,
        PayloadEnumerationRefused,
        _auto_payloads,
    )

    assert issubclass(PayloadEnumerationRefused, DeriveError)

    with pytest.raises(DeriveError) as defect:
        _auto_payloads("@entrypoint x", EmptyAxis)
    assert not isinstance(defect.value, PayloadEnumerationRefused)

    with pytest.raises(DeriveError) as second:
        _auto_payloads("@entrypoint x", NotAStruct)  # type: ignore[arg-type]
    assert not isinstance(second.value, PayloadEnumerationRefused)

    # And the narrow one still fires where it should.
    with pytest.raises(PayloadEnumerationRefused) as refusal:
        _auto_payloads("@entrypoint x", Unsynthesizable)
    assert refusal.value.field == "slots"
