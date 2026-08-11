"""pgw#1117 / th#1777: refuse an artifact that cannot fit its declared
envelope BEFORE staging it, instead of OOMing after.

The red case is ie#642, to the byte. `tensorhub/hidream-o1-image` declared
``vram_gb = 22`` with ``strict_vram`` and its bare ``prod`` head had been
repointed by the th#1754 wipe at the fp32 archive clone —
35,231,236,996 B = 32.81 GiB. The worker printed BOTH numbers
("staged 0.67 GiB of 32.81 GiB"), staged anyway, and OOMed inside ``setup()``
192 MiB short of a 23.52 GiB RTX 4090.

Every snapshot here is built from safetensors HEADERS only: a header declares
its tensors' ``data_offsets``, and the estimator never reads past them, so a
32.81 GiB artifact is a few hundred bytes on disk. $0, no GPU, no torch.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import pytest

from gen_worker.models import provision
from gen_worker.models.envelope import (
    ArtifactEnvelopeExceeded,
    envelope_refusal,
    estimate_resident_bytes,
)

GIB = 1 << 30

# ie#642's own numbers.
FP32_CLONE_BYTES = 35_231_236_996      # 32.81 GiB, the archive head
BF16_SERVING_BYTES = 17_621_456_640    # 16.41 GiB, the checkpoint that serves
DECLARED_VRAM_GB = 22.0
CLONE_DIGEST = "sha256:8a8676a6" + "0" * 56

_ELEM = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "I64": 8, "U8": 1}


def write_safetensors(path: Path, tensors: Sequence[Tuple[str, str, int]]) -> Path:
    """One safetensors file whose HEADER declares ``tensors`` (name, dtype,
    byte length). The tensor data is never written — nothing in the estimator
    reads it, which is what makes a 32 GiB artifact a $0 fixture."""
    header: Dict[str, object] = {}
    offset = 0
    for name, dtype, nbytes in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": [nbytes // _ELEM[dtype]],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    return path


def snapshot(root: Path, *, total_bytes: int, dtype: str = "F32") -> Path:
    """A two-component diffusers-shaped tree weighing ``total_bytes``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "HiDreamImagePipeline"}), encoding="utf-8")
    denoiser = int(total_bytes * 0.9)
    denoiser -= denoiser % _ELEM[dtype]
    rest = total_bytes - denoiser
    rest -= rest % _ELEM[dtype]
    write_safetensors(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        [("transformer.weight", dtype, denoiser)])
    write_safetensors(
        root / "text_encoder" / "model.safetensors",
        [("encoder.weight", dtype, rest)])
    return root


def refusal_for(root: Path, **over: object):
    kwargs: Dict[str, object] = dict(
        declared_vram_gb=DECLARED_VRAM_GB,
        strict_vram=True,
        artifact_digest=CLONE_DIGEST,
        slot="pipeline",
        ref="tensorhub/hidream-o1-image:prod",
    )
    kwargs.update(over)
    return envelope_refusal([root], **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# RED: the ie#642 artifact is refused before anything is staged
# --------------------------------------------------------------------------


def test_fp32_clone_against_a_22gb_declaration_is_refused(tmp_path: Path) -> None:
    root = snapshot(tmp_path / "fp32-clone", total_bytes=FP32_CLONE_BYTES)

    exc = refusal_for(root)

    assert isinstance(exc, ArtifactEnvelopeExceeded)
    text = str(exc)
    # BOTH numbers, in the message, by the issue's requirement.
    assert "32.8" in text, text
    assert "22.00" in text, text
    assert CLONE_DIGEST in text, text
    # And both of them again as machine-readable axes for the hub.
    assert exc.axes()["declared_vram_gb"] == "22.00"
    assert exc.axes()["estimated_vram_gb"].startswith("32.8")
    assert exc.axes()["artifact_digest"] == CLONE_DIGEST
    assert exc.axes()["on_disk_dtype"] == "fp32"
    assert exc.reason == "artifact_envelope_exceeded"


def test_the_refusal_is_not_a_hardware_verdict() -> None:
    """Blame routing: this is a RELEASE/BINDING fault. Typing it into the
    HardwareUnmetError family would make the hub treat it as a machine fact
    (sticky, never re-probed) and would feed the buy-floor learner a demand
    for a card big enough to serve an archive clone."""
    from gen_worker.capability import HardwareUnmetError

    assert not issubclass(ArtifactEnvelopeExceeded, HardwareUnmetError)


def test_refusal_happens_before_the_loader_is_ever_called(tmp_path: Path) -> None:
    """The seam, exercised through the REAL `provision.load_slot`: a refused
    slot never reaches `from_pretrained`, so nothing is staged onto the card
    and no GPU second is billed."""
    root = snapshot(tmp_path / "fp32-clone", total_bytes=FP32_CLONE_BYTES)
    calls: list[str] = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> "FakePipeline":
            calls.append(path)
            return cls()

    with pytest.raises(ArtifactEnvelopeExceeded):
        provision.load_slot(
            FakePipeline,
            str(root),
            slot="pipeline",
            ref="tensorhub/hidream-o1-image:prod",
            declared_vram_gb=DECLARED_VRAM_GB,
            strict_vram=True,
            artifact_digest=CLONE_DIGEST,
        )

    assert calls == [], "the loader ran: the refusal is not a PRE-stage one"


# --------------------------------------------------------------------------
# GREEN: everything that must still be attempted
# --------------------------------------------------------------------------


def test_the_bf16_checkpoint_that_actually_serves_is_admitted(tmp_path: Path) -> None:
    """16.41 GiB under a 22 GB declaration — the checkpoint ie#642 repointed
    `prod` back to. It served a billed 2048x2048 render on the same card."""
    root = snapshot(tmp_path / "bf16", total_bytes=BF16_SERVING_BYTES, dtype="BF16")
    assert refusal_for(root) is None


@pytest.mark.parametrize("gib", [22.0, 23.0, 24.0])
def test_a_marginal_artifact_still_tries(tmp_path: Path, gib: float) -> None:
    """The estimate has error bars, so only a CLEAR breach refuses. Anything
    inside the estimator's error band gets its chance at the card exactly as
    it did before this precondition existed."""
    root = snapshot(tmp_path / f"marginal-{gib}", total_bytes=int(gib * GIB))
    assert refusal_for(root) is None


def test_the_band_is_bounded_and_the_clear_breach_is_still_refused(
    tmp_path: Path,
) -> None:
    """A band that swallowed the incident would be a band that fixes nothing:
    just past it, the refusal is back."""
    root = snapshot(tmp_path / "just-over", total_bytes=int(25.0 * GIB))
    assert refusal_for(root) is not None


def test_without_strict_vram_the_offload_rung_still_carries_it(tmp_path: Path) -> None:
    """"Degrade, don't OOM": without `strict_vram` the fit ladder may place an
    oversized artifact on the offload rung, slowly but alive. This
    precondition only fires where the author already closed that escape."""
    root = snapshot(tmp_path / "fp32-clone", total_bytes=FP32_CLONE_BYTES)
    assert refusal_for(root, strict_vram=False) is None


def test_an_undeclared_envelope_is_not_a_zero_envelope(tmp_path: Path) -> None:
    root = snapshot(tmp_path / "fp32-clone", total_bytes=FP32_CLONE_BYTES)
    assert refusal_for(root, declared_vram_gb=0.0) is None


# --------------------------------------------------------------------------
# The estimate is DTYPE-AWARE, and honest about what it cannot weigh
# --------------------------------------------------------------------------


def test_a_binding_dtype_cast_is_counted(tmp_path: Path) -> None:
    """The same fp32 bytes bound with `dtype="bf16"` load at half the width,
    so they FIT — the estimate weighs the artifact as it will load, not as it
    sits on disk. A disk-bytes check would have refused this wrongly."""
    root = snapshot(tmp_path / "fp32-cast", total_bytes=FP32_CLONE_BYTES)

    est = estimate_resident_bytes([root], cast_dtype="bf16")
    assert est.on_disk_dtype == "fp32"
    assert est.load_dtype == "bf16"
    assert est.resident_bytes == pytest.approx(est.disk_bytes / 2, rel=1e-6)

    assert refusal_for(root, cast_dtype="bf16") is None
    # ...and the uncast load of the same tree is still refused.
    assert refusal_for(root) is not None


def test_integer_tensors_are_not_recast(tmp_path: Path) -> None:
    """A cast re-widths FLOAT weights; int64 position ids and uint8 buffers
    keep their storage whatever the compute dtype is."""
    root = tmp_path / "mixed"
    write_safetensors(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        [("w", "F32", 4 * 1024 * 1024), ("ids", "I64", 8 * 1024)],
    )
    est = estimate_resident_bytes([root], cast_dtype="bf16")
    assert est.resident_bytes == 2 * 1024 * 1024 + 8 * 1024


def test_variant_twins_are_counted_once(tmp_path: Path) -> None:
    """`diffusion_pytorch_model.safetensors` plus its `.fp16.` twin is ONE
    tensor set in memory. Counting both would inflate the estimate, which is
    the one direction a refusal-grade number must never err in."""
    root = tmp_path / "twins"
    write_safetensors(
        root / "unet" / "diffusion_pytorch_model.safetensors",
        [("w", "F32", 8 * 1024 * 1024)])
    write_safetensors(
        root / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        [("w", "F16", 4 * 1024 * 1024)])

    assert estimate_resident_bytes([root]).disk_bytes == 8 * 1024 * 1024
    assert estimate_resident_bytes([root], variant="fp16").disk_bytes == 4 * 1024 * 1024


def test_sharded_variant_twins_are_counted_once(tmp_path: Path) -> None:
    """The twin marker sits INSIDE a shard name
    (`diffusion_pytorch_model.fp16-00001-of-00002.safetensors`), not at the
    end of it. A suffix-only match would miss every sharded twin and double
    the estimate for the biggest models — the exact class of artifact this
    precondition is aimed at."""
    root = tmp_path / "sharded-twins"
    for i in (1, 2):
        write_safetensors(
            root / "transformer" / f"diffusion_pytorch_model-0000{i}-of-00002.safetensors",
            [(f"w{i}", "F32", 8 * 1024 * 1024)])
        write_safetensors(
            root / "transformer" / f"diffusion_pytorch_model.fp16-0000{i}-of-00002.safetensors",
            [(f"w{i}", "F16", 4 * 1024 * 1024)])

    assert estimate_resident_bytes([root]).disk_bytes == 16 * 1024 * 1024
    assert estimate_resident_bytes([root], variant="fp16").disk_bytes == 8 * 1024 * 1024


def test_an_unweighable_dtype_abstains_instead_of_guessing(tmp_path: Path) -> None:
    root = tmp_path / "exotic"
    write_safetensors(root / "w.safetensors", [("w", "F32", 4096)])
    # Rewrite the header with a token the table does not carry.
    raw = (root / "w.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    header["w"]["dtype"] = "E3M2"
    blob = json.dumps(header).encode("utf-8")
    (root / "w.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)

    est = estimate_resident_bytes([root])
    assert not est.measurable
    assert "E3M2" in est.method
    assert refusal_for(root) is None


def test_a_weightless_or_unreadable_tree_abstains(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not estimate_resident_bytes([empty]).measurable
    assert refusal_for(empty) is None


def test_a_runtime_storage_cast_abstains(tmp_path: Path) -> None:
    """`storage_dtype=fp8` restructures a subset of modules AFTER the load and
    which subset is a property of the pipeline class, not of the tree. Not
    plainly computable -> no verdict."""
    root = snapshot(tmp_path / "fp32-clone", total_bytes=FP32_CLONE_BYTES)
    assert refusal_for(root, storage_dtype="fp8") is None


def test_a_specialized_weight_layout_abstains(tmp_path: Path) -> None:
    """svdq / w8a8 / w4a4 / gguf / on-disk-quantized trees have a header story
    that differs from their in-memory story."""
    root = snapshot(tmp_path / "fp32-clone", total_bytes=FP32_CLONE_BYTES)
    assert refusal_for(root, specialized_layout="w4a4") is None


def test_component_override_trees_are_weighed_too(tmp_path: Path) -> None:
    """th#980/pgw#617 component overrides are staged alongside the base tree,
    so they count toward residency."""
    base = snapshot(tmp_path / "base", total_bytes=int(12.0 * GIB), dtype="BF16")
    override = tmp_path / "override"
    write_safetensors(
        override / "diffusion_pytorch_model.safetensors",
        [("w", "F32", int(20.0 * GIB))])

    assert envelope_refusal(
        [base], declared_vram_gb=DECLARED_VRAM_GB, strict_vram=True) is None
    assert envelope_refusal(
        [base, override], declared_vram_gb=DECLARED_VRAM_GB, strict_vram=True
    ) is not None


# --------------------------------------------------------------------------
# Blame routing at the executor seam
# --------------------------------------------------------------------------


class _Rec:
    def __init__(self) -> None:
        self.specs = [SimpleNamespace(name="generate")]
        self.failed = None


class _Executor:
    def __init__(self) -> None:
        self.unavailable: Dict[str, object] = {}

    def _on_state_change(self) -> None:
        pass


def _classify(exc: BaseException) -> Tuple[str, Dict[str, str], str]:
    from gen_worker.executor import Executor

    ex, rec = _Executor(), _Rec()
    Executor._mark_setup_failed(ex, rec, exc)  # type: ignore[arg-type]
    reason, detail, axes = ex.unavailable["generate"]  # type: ignore[misc]
    return reason, axes, detail


def test_the_executor_reports_the_typed_reason_not_setup_failed() -> None:
    """`setup_failed` is what the hub reads as "the pod could not boot", and
    it is what fed the streak that darkened the endpoint. The refusal must
    arrive under its own token, carrying both numbers as axes."""
    exc = ArtifactEnvelopeExceeded(
        "artifact does not fit", estimated_bytes=FP32_CLONE_BYTES,
        declared_vram_gb=DECLARED_VRAM_GB, artifact_digest=CLONE_DIGEST,
        ref="tensorhub/hidream-o1-image:prod", on_disk_dtype="fp32")

    reason, axes, detail = _classify(exc)

    assert reason == "artifact_envelope_exceeded"
    assert axes["declared_vram_gb"] == "22.00"
    assert axes["estimated_vram_gb"].startswith("32.8")
    assert axes["artifact_digest"] == CLONE_DIGEST
    assert "artifact does not fit" in detail

    # Control: an ordinary setup exception is unchanged.
    assert _classify(RuntimeError("boom"))[0] == "setup_failed"


def test_the_reason_token_is_in_the_wire_contract() -> None:
    proto = (Path(__file__).resolve().parents[1]
             / "proto" / "worker_scheduler.proto").read_text(encoding="utf-8")
    assert "artifact_envelope_exceeded" in proto


