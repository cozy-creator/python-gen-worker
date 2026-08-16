"""pgw#1252: the decode-set's `file_layouts` axis — IMPORTED, and enforced.

Two claims, each tested rather than asserted in prose.

**One home.** The tokens come from `models/file_layout.py`, the same module
`convert/` publishes through. The `models -> convert` direction the issue
proposed is a HARD CYCLE (`convert/__init__` reaches `api.slot` and back into
`models`), so the vocabulary moved rather than being copied — a transcription
here would have been the fourth spelling of one axis.

**A declaration nothing enforces is a declaration nothing can be wrong about.**
The svdq decoders declare `multi-file` ONLY, and that is measured, not
inherited from the checkpoint's shape: the nunchaku-FORMAT file is one flat
namespace, but the engine refuses an artifact that is only that file
(`load_svdq_native_pipeline`, `not art.component`). Before this axis that fact
lived only in a `raise` reached AFTER the CUDA gate, so a GPU-less host
reported "svdq artifacts require a CUDA GPU" for an artifact no GPU would have
helped. (pgw#1298 deleted the second engine; the axis is unchanged.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import save_file  # noqa: E402

from gen_worker.discovery.decode_set import (  # noqa: E402
    REFUSAL_FILE_LAYOUT_UNSUPPORTED,
    FileLayoutUnsupportedError,
    accepted_file_layouts,
    runtime_decode_set,
)
from gen_worker.models import file_layout as fl  # noqa: E402
from gen_worker.models.loading import load_from_pretrained  # noqa: E402
from gen_worker.models.svdq import detect_svdq_artifact  # noqa: E402
from gen_worker.models.tensor_layout_contract import (  # noqa: E402
    CONTRACT_NUNCHAKU_V1,
    KNOWN_ELEMENTS,
)

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
        raise AssertionError("the layout gate should have refused first")


def _svdq_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": torch.zeros(4, dtype=torch.uint8)}, str(path),
              metadata=SVDQ_META)


# ── one home, imported ───────────────────────────────────────────────────────

def test_the_axis_is_the_imported_vocabulary_not_a_transcription() -> None:
    """Every declared token is a member of the ONE ruled set, and the publish
    side validates through the same module object — not a same-looking copy."""
    from gen_worker.convert import publish

    assert getattr(publish, "validate_file_layout") is fl.validate_file_layout

    declared = {
        token
        for entry in runtime_decode_set().entries
        for token in entry.decodes.file_layouts
    }
    assert declared, "no decoder declares a file layout"
    assert declared <= fl.KNOWN_FILE_LAYOUTS
    # and the axis is not silently borrowing another axis's tokens
    assert not (declared & set(KNOWN_ELEMENTS))


def test_a_dead_spelling_cannot_be_declared() -> None:
    """No aliases: the pre-th#1937 spellings fail the BUILD where they are
    written, which is what stops a fourth one appearing."""
    from gen_worker.models.tensor_layout_contract import (
        CONTRACT_PLAIN_BF16,
        DecodeDimensions,
        implements_contract,
    )

    for dead in ("singlefile", "diffusers", "single_file"):
        with pytest.raises(ValueError, match="is not registered"):
            @implements_contract(
                contract=CONTRACT_PLAIN_BF16, serves=("bf16-w16a16",),
                composes_lora=False,
                decodes=DecodeDimensions(
                    elements=("bf16",), scales=("none",), key_topologies=(),
                    file_layouts=(dead,), bakes=()),
            )
            def _dead(x: Any) -> Any:
                return x

    with pytest.raises(TypeError):
        DecodeDimensions(  # type: ignore[call-arg]
            elements=("bf16",), scales=("none",), key_topologies=(), bakes=())


def test_every_decoder_declares_the_axis() -> None:
    missing = [e.decoder for e in runtime_decode_set().entries
               if not e.decodes.file_layouts]
    assert missing == [], f"decoders with no file_layouts declaration: {missing}"


# ── the observation agrees with what publish stamps ──────────────────────────

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


# ── the refusal, through the production dispatch ─────────────────────────────

def test_single_file_svdq_snapshot_refuses_by_layout(tmp_path: Path) -> None:
    """RED before pgw#1252: this reached the svdq lane and died on the CUDA
    gate — `svdq artifacts require a CUDA GPU`, which is not the reason."""
    snapshot = tmp_path / "bare-svdq"
    _svdq_file(snapshot / "svdq-fp4_r32-flux.safetensors")
    # The artifact really is detected as svdq, so the arm under test is reached.
    art = detect_svdq_artifact(snapshot)
    assert art is not None and art.component == ""

    with pytest.raises(FileLayoutUnsupportedError) as excinfo:
        load_from_pretrained(_Pipe, snapshot)
    err = excinfo.value
    assert err.code == REFUSAL_FILE_LAYOUT_UNSUPPORTED
    assert err.observed == fl.SINGLE_FILE
    assert err.accepted == (fl.MULTI_FILE,)
    assert CONTRACT_NUNCHAKU_V1 in str(err)


def test_multi_file_svdq_tree_passes_the_layout_gate(tmp_path: Path) -> None:
    """The shape `convert/svdq.py` actually builds is admitted — the guard has
    to let the real artifact through or it is just a refusal."""
    snapshot = tmp_path / "flavor"
    _svdq_file(snapshot / "transformer" / "svdq-fp4_r32-flux.safetensors")
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}), encoding="utf-8")
    assert fl.observed_file_layout(snapshot) == fl.MULTI_FILE

    art = detect_svdq_artifact(snapshot)
    assert art is not None and art.component == "transformer"
    # It gets past the layout gate and fails later, on the engine — which is
    # the honest cause on a host with no CUDA.
    with pytest.raises(Exception) as excinfo:
        load_from_pretrained(_Pipe, snapshot)
    assert not isinstance(excinfo.value, FileLayoutUnsupportedError)


def test_an_unconstrained_contract_is_not_evaluated() -> None:
    """`accepted_file_layouts` on a contract nothing declares returns (), and
    an empty accepted set is the UNDECLARED rung — not a refusal of every
    artifact."""
    ds = runtime_decode_set()
    assert accepted_file_layouts("plain.bf16@1", ds) == (
        fl.MULTI_FILE, fl.SINGLE_FILE)
    assert accepted_file_layouts("nope.nothing@9", ds) == ()
