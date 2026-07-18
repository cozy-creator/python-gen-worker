"""th#901 regression: a publish_as_is source (single dense weight-set
strategies — transformers / diffusers_component / peft / sentence_transformers /
native_lora) must not silently republish its own on-disk dtype when the
caller explicitly requested a DIFFERENT one.

Live bug (HiDream-O1-Image, ~35GB fp32 UiT backbone, classifies
strategy="transformers"): `outputs=[{"dtype": "bf16", "file_layout":
"diffusers"}]` against an fp32 source resolved onto the identical
pre-existing fp32 checkpoint — no bf16 flavor was ever produced, and no
error surfaced. Root cause: `run_clone`'s publish_as_is branch special-cased
`spec.dtype == "bf16"` to skip its own mismatch check, then fell through to
publishing `source.dir` unchanged under `flavor_label = source_dtype`.

These tests exercise the REAL run_clone orchestration (real fake-tensorhub
HTTP server, real workdir/lock lifecycle, real commit wire body) — only
`build_flavor_tree`'s tensor-casting internals (torch/safetensors) are
stubbed, matching this repo's "no GPU, no weight downloads" unit-test
convention for decision logic. `tests/convert/test_integration.py` covers a
real end-to-end cast against a real tiny HF model separately.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gen_worker.convert.clone import run_clone
from gen_worker.convert.ingest import IngestedSource

from fake_hub import _FakeHub


class _Ctx:
    def __init__(self, server) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.request_id = "req-1"
        self.destination = {"repo": "acme/fallback"}


def _transformers_source(dest_dir: Path, *, dtype: str = "fp32") -> IngestedSource:
    """A single-root-weight-set 'transformers backbone' source, HiDream-O1's
    UiT shape: config.json + root safetensors, no model_index.json."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "config.json").write_text('{"architectures": ["FakeBackbone"]}')
    (dest_dir / "model.safetensors").write_bytes(b"\x00" * 64)
    return IngestedSource(
        provider="huggingface",
        source_ref="org/hidream-like",
        source_revision="sha-1",
        dir=dest_dir,
        layout="singlefile",
        model_family="hidream",
        model_family_variant="",
        classification=SimpleNamespace(strategy="transformers"),
        attrs={"dtype": dtype, "file_layout": "singlefile"},
        metadata={"source_provider": "huggingface"},
        repo_spec={"kind": "model", "library_name": "transformers"},
    )


def _install_fake_ingest(monkeypatch, source: IngestedSource):
    def fake_ingest(source_ref, dest_dir, **kwargs):
        return source

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)


def _stub_build_flavor_tree(monkeypatch, calls: list):
    """Replace the real (torch-backed) cast with a cheap stand-in that still
    exercises run_clone's real control flow and real commit wire body —
    proves the DECISION to cast fires, without paying for tensor math."""

    def fake_build_flavor_tree(source, spec, out_dir, *, quantize_components=None):
        calls.append({
            "dtype": spec.dtype, "file_layout": spec.file_layout,
            "file_type": spec.file_type,
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.safetensors").write_bytes(b"\x01" * 32)  # "cast" bytes
        attrs = {"dtype": spec.dtype, "file_layout": spec.file_layout,
                 "file_type": spec.file_type}
        return out_dir, attrs

    monkeypatch.setattr("gen_worker.convert.clone.build_flavor_tree", fake_build_flavor_tree)


def test_mismatched_dtype_runs_real_conversion_not_silent_passthrough(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """th#901 core regression: bf16 requested against an fp32 'transformers'
    source must invoke the cast path and publish flavor=bf16 — never
    silently republish the fp32 source under the requested label."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    source = _transformers_source(tmp_path / "source", dtype="fp32")
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/hidream-like",
        destination_repo="acme/dest",
        outputs=[{"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert len(result.published) == 1
    assert result.published[0]["flavor"] == "bf16"
    # the cast really ran (not a passthrough of the fp32 tree)
    assert len(calls) == 1
    assert calls[0]["dtype"] == "bf16"
    # organizational layout is never repackaged inline for these strategies —
    # the cast targets the source's OWN on-disk layout (singlefile), not the
    # caller's requested "diffusers" layout. The commit is honest about it.
    assert calls[0]["file_layout"] == "singlefile"
    req = _FakeHub.state["commit_request"]
    assert req["flavor"] == "bf16"
    assert req["dtype"] == "bf16"
    assert req["file_layout"] == "singlefile"


def test_matching_dtype_still_passes_through_without_casting(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """No regression: a request that already matches the source's own dtype
    is genuinely zero-work and must NOT call build_flavor_tree."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    source = _transformers_source(tmp_path / "source", dtype="fp32")
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/hidream-like",
        destination_repo="acme/dest",
        outputs=[{"dtype": "fp32", "file_layout": "diffusers", "file_type": "safetensors"}],
    )

    assert not result.failed_flavors
    assert len(result.published) == 1
    assert result.published[0]["flavor"] == "fp32"
    assert calls == []  # no cast attempted — already had it


def test_mismatched_dtype_on_gguf_source_fails_loud_not_silent(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """Non-cast-eligible publish_as_is strategies (gguf: binary quant
    container, not a dense safetensors tree) still refuse a mismatched
    dtype loudly instead of silently republishing under the wrong label —
    the th#901 fix must not weaken this existing protection."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    dest = tmp_path / "source"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.q4_k_m.gguf").write_bytes(b"\x00" * 32)
    source = IngestedSource(
        provider="huggingface", source_ref="org/gguf-src", source_revision="sha-1",
        dir=dest, layout="singlefile", model_family="", model_family_variant="",
        classification=SimpleNamespace(strategy="gguf"),
        attrs={"dtype": "q4_k_m", "file_layout": "singlefile"},
        metadata={}, repo_spec={},
    )
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="publish as-is"):
        run_clone(
            _Ctx(fake_hub), provider="huggingface", source_ref="org/gguf-src",
            destination_repo="acme/dest",
            outputs=[{"dtype": "q8_0", "file_layout": "singlefile", "file_type": "gguf"}],
        )

    assert calls == []  # never silently cast a binary quant container


def test_second_spec_on_publish_as_is_source_still_refused(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """i>0 stays refused even for a cast-eligible strategy: only the first
    (primary) output spec runs inline conversion — extra flavors are still
    a separate job, unchanged by th#901."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    source = _transformers_source(tmp_path / "source", dtype="fp32")
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/hidream-like",
        destination_repo="acme/dest",
        outputs=[
            {"dtype": "fp32", "file_layout": "diffusers", "file_type": "safetensors"},
            {"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"},
        ],
    )

    assert len(result.published) == 1
    assert result.published[0]["flavor"] == "fp32"
    assert len(result.failed_flavors) == 1
    assert result.failed_flavors[0]["dtype"] == "bf16"
    assert calls == []
