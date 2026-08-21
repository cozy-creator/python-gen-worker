"""pgw#1252's `file_layouts` axis — what survived pgw#1621, and what did not.

**READ THIS BEFORE ASSUMING THIS FILE STILL GUARDS WHAT ITS NAME SAYS.**

pgw#1252 gave the decode-set a `file_layouts` DECODE AXIS: each decoder declared
which on-disk shapes it could read, `require_decodable` intersected the
observed shape against it, and a single-file svdq snapshot was refused BY
LAYOUT before the CUDA gate. pgw#1621 deleted the five decode dimensions
wholesale, because four of them (elements, scales, key topologies, bakes) became
part of a v2 quant rule's IDENTITY. `file_layouts` is the one that did not:
`src/gen_worker/discovery/decode_set.py` says so in its own words — *"the
file-layout question has no successor in this image at all"*.

So the two claims this file used to make now split:

**One home — STILL TRUE, still tested.** The tokens come from
`models/file_layout.py`, the same module `convert/publish.py` validates
through. A transcription would be the fourth spelling of one axis. That is
asserted below, on the module OBJECT.

**A declaration nothing enforces — GONE, and the loss is pinned below.** No
decoder declares a file layout any more, so there is nothing to enforce and
nothing to be wrong about. `test_a_single_file_svdq_snapshot_no_longer_refuses_
by_layout` is the record: it asserts the CURRENT behaviour, which is the
pgw#1252 defect exactly — a bare svdq file reaches the hardware gate and reports
"needs Blackwell fp4 tensor cores" for an artifact no GPU would have helped.
**That test is written to go RED the day a successor lands**, so the hole is
visible in the suite instead of only in this docstring.

What is NOT lost: `observed_file_layout` still classifies, `validate_file_layout`
still refuses the dead pre-th#1937 spellings at the publish boundary, and svdq
artifact DETECTION (component vs. bare file) is unchanged. Those are tested
here because they are what the successor would have to be built on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import save_file  # noqa: E402

from gen_worker.discovery.decode_set import runtime_decode_set  # noqa: E402
from gen_worker.models import file_layout as fl  # noqa: E402
from gen_worker.models.loading import load_from_pretrained  # noqa: E402
from gen_worker.models.svdq import detect_svdq_artifact  # noqa: E402

SVDQ_META = {
    "model_class": "NunchakuFluxTransformer2dModel",
    "quantization_config": json.dumps({
        "method": "svdquant",
        "weight": {"dtype": "fp4_e2m1_all"},
        "rank": 32,
    }),
}


class _Pipe:
    """A pipeline class shaped like the ones the loader dispatches on. Never
    constructed: every arm under test refuses before any load."""

    @staticmethod
    def from_pretrained(*a: Any, **kw: Any) -> Any:  # pragma: no cover
        raise AssertionError("nothing here should reach the real loader")


def _svdq_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": torch.zeros(4, dtype=torch.uint8)}, str(path),
              metadata=SVDQ_META)


# ── one home, imported ───────────────────────────────────────────────────────

def test_the_axis_is_the_imported_vocabulary_not_a_transcription() -> None:
    """The publish side validates through the same module OBJECT — not a
    same-looking copy. This half of pgw#1252 is untouched by the v2 cut."""
    from gen_worker.convert import publish

    # `getattr`, not attribute syntax: `publish` IMPORTS this name to use it
    # and does not re-export it, so strict mypy is right to refuse
    # `publish.validate_file_layout`. Putting it in `publish.__all__` to
    # satisfy the checker would make the module claim a public export it does
    # not have — the identity below is the claim, and it is unchanged.
    assert getattr(publish, "validate_file_layout") is fl.validate_file_layout
    assert fl.KNOWN_FILE_LAYOUTS == {fl.MULTI_FILE, fl.SINGLE_FILE}


def test_a_dead_spelling_cannot_be_published() -> None:
    """No aliases: the pre-th#1937 spellings fail where they are written,
    which is what stops a fourth one appearing.

    This used to be asserted at the `@implements_contract` marker, because a
    decoder declared the axis there. It has no decoder-side declaration site
    any more, so it is asserted at the one boundary that still validates the
    token — the publish path — which is where the vocabulary is actually
    consumed.
    """
    for dead in ("singlefile", "diffusers", "single_file", "multifile"):
        with pytest.raises(ValueError):
            fl.validate_file_layout(dead)

    # ...and the live spellings survive round-tripping, so this is a refusal
    # of the DEAD tokens rather than of everything.
    for live in (fl.MULTI_FILE, fl.SINGLE_FILE, fl.NOT_APPLICABLE):
        assert fl.validate_file_layout(live) == live


def test_no_decoder_declares_a_file_layout_any_more() -> None:
    """The axis is GONE from the decode-set entry, not merely empty.

    An empty tuple would be a declaration nobody wrote; an absent FIELD cannot
    be silently ignored. This is the assertion that fails if someone
    reintroduces the axis on the decoder side without reconnecting the
    intersection — which would be pgw#1252's original "a declaration nothing
    enforces" defect rebuilt.
    """
    entries = runtime_decode_set().entries
    assert entries, "no decoder declared anything — every assertion below is vacuous"
    for entry in entries:
        assert not hasattr(entry, "decodes"), entry.decoder
        assert not hasattr(entry, "file_layouts"), entry.decoder


# ── the observation still classifies ─────────────────────────────────────────

def test_observed_layout_matches_the_publish_side_shapes(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline"
    (pipeline / "transformer").mkdir(parents=True)
    (pipeline / "model_index.json").write_text("{}", encoding="utf-8")
    assert fl.observed_file_layout(pipeline) == fl.MULTI_FILE

    loose = tmp_path / "loose"
    loose.mkdir()
    save_file({"x": torch.zeros(2)}, str(loose / "model.safetensors"))
    assert fl.observed_file_layout(loose) == fl.SINGLE_FILE
    assert fl.observed_file_layout(loose / "model.safetensors") == fl.SINGLE_FILE

    # An unclassifiable shape states NOTHING rather than guessing.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert fl.observed_file_layout(bare) == fl.NOT_APPLICABLE


# ── the refusal that is GONE ─────────────────────────────────────────────────

def test_a_single_file_svdq_snapshot_no_longer_refuses_by_layout(
    tmp_path: Path,
) -> None:
    """⚠️ THIS TEST PINS A HOLE, NOT A GUARANTEE. It is expected to go RED.

    pgw#1252's whole point: this snapshot is a bare nunchaku-FORMAT file with
    no component subdirectory, and `load_svdq_native_pipeline` refuses an
    artifact that is only that file (`not art.component`). Before pgw#1252 the
    refusal happened AFTER the hardware gate, so the operator was told
    `svdq artifacts require a CUDA GPU` for an artifact no GPU would have
    helped. pgw#1252 moved the refusal in front of the gate by intersecting the
    OBSERVED layout against the decoder's declared `file_layouts`.

    pgw#1621 deleted that axis with no successor, so the refusal moved back.
    What is asserted here is that it really did: the observation is still
    `single-file`, the artifact is still detected with no component, and the
    exception that arrives is NOT about the layout.

    **When a successor lands** — a layout intersection rebuilt on the v2
    vocabulary, or the engine simply checking `art.component` before the
    hardware gate — this assertion goes red and the reader is sent here. That
    is the intent: the assertion below is the RECORD of a regression, and a
    regression nobody can see is the thing this file was written about.
    """
    snapshot = tmp_path / "bare-svdq"
    _svdq_file(snapshot / "svdq-fp4_r32-flux.safetensors")

    # The two facts the deleted gate was built out of are both still true.
    art = detect_svdq_artifact(snapshot)
    assert art is not None and art.component == ""
    assert fl.observed_file_layout(snapshot) == fl.SINGLE_FILE

    with pytest.raises(Exception) as excinfo:
        load_from_pretrained(_Pipe, snapshot)
    message = str(excinfo.value)
    # The refusal that used to be here is gone: nothing names the LAYOUT.
    assert "single-file" not in message and "multi-file" not in message, (
        "a layout-shaped refusal is back — pgw#1252's guard has a successor "
        "again. Delete this test and restore the real one above it."
    )


def test_a_multi_file_svdq_tree_is_detected_with_its_component(
    tmp_path: Path,
) -> None:
    """The shape `convert/svdq.py` actually builds. Its component is read, so
    a successor gate has the fact it needs to distinguish the two arms — which
    is why this survives the deletion of the gate itself."""
    snapshot = tmp_path / "flavor"
    _svdq_file(snapshot / "transformer" / "svdq-fp4_r32-flux.safetensors")
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}), encoding="utf-8")
    assert fl.observed_file_layout(snapshot) == fl.MULTI_FILE

    art = detect_svdq_artifact(snapshot)
    assert art is not None and art.component == "transformer"


def test_the_svdq_decode_path_declares_itself_UNREGISTERED() -> None:
    """The v2 answer to "which bytes can this image read" for svdq.

    `nunchaku.v1@1` was a v1 CONTRACT and has no ratified v2 quant rule, so
    the decoder records an UNREGISTERED DECODE PATH with a reason rather than
    inventing a handle. That record is what a refusal can read; a source
    comment is not. It is also why the file-layout intersection has nothing to
    hang off any more — there is no declared rule for svdq to carry an axis.
    """
    unregistered = {u.decoder: u.reason
                    for u in runtime_decode_set().unregistered}
    assert "gen_worker.models.svdq_layout:decode_linear" in unregistered
    assert unregistered["gen_worker.models.svdq_layout:decode_linear"].strip()
