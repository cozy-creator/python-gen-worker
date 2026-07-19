"""gw#592: LTX-2.3 output routing.

Lightricks/LTX-2.3 classifies strategy="aio_singlefile" (classify_repo has no
dedicated LTX-2 bucket — it's a bare root safetensors repo like any other).
Before this fix, run_clone's non-publish_as_is branch (build_flavor_tree) hit
the layout-repackage guard whenever the caller requested file_layout=
"diffusers" (the historical default), and family="unknown" (pre gw#592) or
family="ltx2" (post) is not in `_REPACKAGE_NORMALIZED_FAMILIES` — died
"clone produced no publishable flavor". The te#70 trainer resolves the
native singlefile/repackage snapshot directly (no diffusers pipeline exists
for LTX-2), so the fix routes ltx2's aio_singlefile source through the
publish_as_is path instead of building a repackager nobody needs.

Same "real run_clone orchestration, stubbed tensor math" convention as
tests/convert/test_publish_as_is_dtype.py (th#901).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def _ltx2_source(dest_dir: Path, *, dtype: str = "bf16") -> IngestedSource:
    """A monolith LTX-2.3 singlefile source: strategy=aio_singlefile,
    model_family=ltx2 (as detected by layout.detect_huggingface_source_layout
    post gw#592), no model_index.json / no diffusers pipeline."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "ltx-2.3-13b-dev.safetensors").write_bytes(b"\x00" * 64)
    return IngestedSource(
        provider="huggingface",
        source_ref="Lightricks/LTX-2.3",
        source_revision="sha-1",
        dir=dest_dir,
        layout="singlefile",
        model_family="ltx2",
        model_family_variant="ltx2",
        classification=SimpleNamespace(strategy="aio_singlefile"),
        attrs={"dtype": dtype, "file_layout": "singlefile"},
        metadata={"source_provider": "huggingface"},
        repo_spec={"kind": "model", "library_name": ""},
    )


def _install_fake_ingest(monkeypatch, source: IngestedSource):
    def fake_ingest(source_ref, dest_dir, **kwargs):
        return source

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)


def _stub_build_flavor_tree(monkeypatch, calls: list):
    def fake_build_flavor_tree(source, spec, out_dir, *, quantize_components=None):
        calls.append({
            "dtype": spec.dtype, "file_layout": spec.file_layout,
            "file_type": spec.file_type,
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.safetensors").write_bytes(b"\x01" * 32)
        attrs = {"dtype": spec.dtype, "file_layout": spec.file_layout,
                 "file_type": spec.file_type}
        return out_dir, attrs

    monkeypatch.setattr("gen_worker.convert.clone.build_flavor_tree", fake_build_flavor_tree)


def test_ltx2_diffusers_layout_request_publishes_native_singlefile_not_repackage(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """The historical bug: requesting layout='diffusers' against an ltx2
    source must NOT reach build_flavor_tree's repackage guard (which has no
    rule for ltx2 and would raise 'clone produced no publishable flavor').
    It must publish the native singlefile snapshot instead."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    source = _ltx2_source(tmp_path / "source", dtype="bf16")
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="Lightricks/LTX-2.3",
        destination_repo="acme/ltx-2.3-dev",
        outputs=[{"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert len(result.published) == 1
    assert result.published[0]["flavor"] == "bf16"
    req = _FakeHub.state["commit_request"]
    assert req["dtype"] == "bf16"
    # publish_as_is is organizationally honest: the source's OWN native
    # layout ships, never the caller's requested (nonexistent) diffusers one.
    assert req["file_layout"] == "singlefile"


def test_ltx2_mismatched_dtype_casts_inline_not_silent_passthrough(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """th#901 precedent applies to ltx2 too: an explicitly mismatched dtype
    is real castable work, not a silent republish under the wrong label."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    source = _ltx2_source(tmp_path / "source", dtype="fp32")
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="Lightricks/LTX-2.3",
        destination_repo="acme/ltx-2.3-dev",
        outputs=[{"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert len(result.published) == 1
    assert result.published[0]["flavor"] == "bf16"
    assert len(calls) == 1
    assert calls[0]["dtype"] == "bf16"
    assert calls[0]["file_layout"] == "singlefile"


def test_non_ltx2_aio_singlefile_is_unaffected(
    fake_hub, tmp_path: Path, monkeypatch,
) -> None:
    """Guardrail: the routing must be gated on family, not just strategy —
    an ordinary (non-ltx2) aio_singlefile source (e.g. a bare SD1.5 ckpt)
    must still go through build_flavor_tree, since THOSE families have a
    real singlefile->diffusers repackager other tenants rely on."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    dest = tmp_path / "source"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "v1-5-pruned.safetensors").write_bytes(b"\x00" * 64)
    source = IngestedSource(
        provider="huggingface", source_ref="org/sd15-ckpt", source_revision="sha-1",
        dir=dest, layout="singlefile", model_family="sd15_sd2", model_family_variant="sd15",
        classification=SimpleNamespace(strategy="aio_singlefile"),
        attrs={"dtype": "fp32", "file_layout": "singlefile"},
        metadata={}, repo_spec={},
    )
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/sd15-ckpt",
        destination_repo="acme/dest",
        outputs=[{"dtype": "fp32", "file_layout": "singlefile", "file_type": "safetensors"}],
    )

    # build_flavor_tree ran at all (proves this took the non-publish_as_is
    # branch) — a real fp32-source/fp32-request call would internally no-op
    # via build_flavor_tree's own passthrough branch, but that internal
    # short-circuit isn't observable through the stub; what matters here is
    # WHICH branch of run_clone dispatched to it.
    assert not result.failed_flavors, result.failed_flavors
    assert len(calls) == 1
    assert calls[0]["file_layout"] == "singlefile"
