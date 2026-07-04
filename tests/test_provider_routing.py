"""Provider routing + ensure_local dispatch (#366).

Exercises the REAL routing path of ``gen_worker.models.download``:

  * build_provider_index_from_manifest over a real endpoint.lock-shaped dict,
  * lookup_provider_for_ref against the process-global index,
  * ensure_local dispatching to the right provider branch (hf/civitai leaves
    stubbed only at the network edge — every routing decision is real),
  * retry-after-failure: a failed ensure_local attempt does not poison the ref,
  * the safetensors-only gate against real tmp files.

Named regressions kept as explicit cases:
  * tag-stripping (live 2026-05-16 failure: ``:latest`` stamped HF ref),
  * HF #flavor suffix stripped before producing the huggingface_hub repo_id.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import pytest

import gen_worker.models.download as dl_mod
from gen_worker.models.download import (
    build_provider_index_from_manifest,
    ensure_local,
    lookup_provider_for_ref,
    set_provider_index,
)
from gen_worker.models.unsafe_format import UnsafeFileFormat, assert_safe_weight_format


def _make_weight_files(root: Path, files: List[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)
    return root


@pytest.fixture(autouse=True)
def _clean_index():
    set_provider_index({})
    yield
    set_provider_index({})


# --------------------------------------------------------------------------- #
# build_provider_index_from_manifest — real endpoint.lock shapes
# --------------------------------------------------------------------------- #


def _manifest(*binding_blocks: dict) -> dict:
    return {
        "functions": [
            {"name": f"fn{i}", "bindings": b} for i, b in enumerate(binding_blocks)
        ]
    }


@pytest.mark.parametrize(
    "block,expect",
    [
        (
            {"pipeline": {"kind": "fixed", "provider": "hf",
                          "ref": "bfl/FLUX.2-klein-4B", "flavor": "bf16", "tag": "prod"}},
            {"bfl/FLUX.2-klein-4B#bf16": "hf", "bfl/FLUX.2-klein-4B": "hf"},
        ),
        (
            {"pipeline": {"kind": "fixed", "provider": "tensorhub",
                          "ref": "acme/flux", "flavor": "fp8", "tag": "prod"}},
            {"acme/flux:prod#fp8": "tensorhub", "acme/flux#fp8": "tensorhub"},
        ),
        (
            {"pipeline": {"kind": "dispatch", "field": "variant", "table": {
                "bf16": {"provider": "hf", "ref": "owner/flux", "flavor": "bf16", "tag": "prod"},
                "fp8": {"provider": "tensorhub", "ref": "owner/flux", "flavor": "fp8", "tag": "prod"},
            }}},
            {"owner/flux#bf16": "hf", "owner/flux:prod#fp8": "tensorhub"},
        ),
    ],
)
def test_provider_index_extracted_from_manifest(block: dict, expect: dict) -> None:
    index = build_provider_index_from_manifest(_manifest(block))
    for ref, provider in expect.items():
        assert index.get(ref) == provider, (ref, index)

    assert build_provider_index_from_manifest(None) == {}
    assert build_provider_index_from_manifest({"functions": []}) == {}
    bad = build_provider_index_from_manifest({
        "functions": [
            {"bindings": {"pipeline": {"kind": "fixed", "provider": "hf"}}},  # no ref
            {"bindings": "not-a-dict"},
            {"bindings": {"pipeline": {"kind": "fixed", "provider": "hf", "ref": "ok/ref"}}},
        ]
    })
    assert bad == {"ok/ref": "hf"}


# --------------------------------------------------------------------------- #
# lookup_provider_for_ref — global index + tag-stripping
# --------------------------------------------------------------------------- #


def test_lookup_default_and_index() -> None:
    assert lookup_provider_for_ref("foo/bar") == "tensorhub"
    assert lookup_provider_for_ref("foo/bar", default="hf") == "hf"
    set_provider_index({"acme/flux#bf16": "hf"})
    assert lookup_provider_for_ref("acme/flux#bf16") == "hf"
    assert lookup_provider_for_ref("not/in-index") == "tensorhub"
    set_provider_index(None)
    assert lookup_provider_for_ref("acme/flux#bf16") == "tensorhub"  # cleared


@pytest.mark.parametrize(
    "wire_ref",
    [
        "bfl/FLUX.2-klein-4B:latest#bf16",  # canonicalizer stamped :latest
        "bfl/FLUX.2-klein-4B:prod#bf16",    # non-default tag
        "bfl/FLUX.2-klein-4B#bf16",         # bare form (no regression)
    ],
)
def test_lookup_tag_strip(wire_ref: str) -> None:
    """Live 2026-05-16 failure: a runtime payload stamps a ``:tag`` onto an HF
    ref but the index only carries the bare HF form, so the lookup must strip
    the tag (tag is meaningless for HF)."""
    assert lookup_provider_for_ref(wire_ref) == "tensorhub"  # default before install
    set_provider_index({"bfl/FLUX.2-klein-4B#bf16": "hf"})
    assert lookup_provider_for_ref(wire_ref) == "hf"


# --------------------------------------------------------------------------- #
# ensure_local dispatch — real branch selection
# --------------------------------------------------------------------------- #


def test_hf_indexed_ref_routes_to_hf_branch(tmp_path: Path, monkeypatch) -> None:
    snap = _make_weight_files(tmp_path / "snap", ["model.safetensors"])
    calls: list = []

    def _fake_hf(ref, **kw):
        calls.append(ref)
        return snap

    monkeypatch.setattr(dl_mod, "download_hf", _fake_hf)
    set_provider_index({"bfl/FLUX.2-klein-4B#bf16": "hf"})
    out = asyncio.run(ensure_local("bfl/FLUX.2-klein-4B#bf16", cache_dir=tmp_path))
    assert len(calls) == 1
    assert calls[0].repo_id == "bfl/FLUX.2-klein-4B"  # #flavor stripped
    assert out == snap


def test_civitai_ref_routes_to_civitai_branch(tmp_path: Path, monkeypatch) -> None:
    got: dict = {}

    def _fake_civitai(version_id, out_dir, **kw):
        got["version_id"] = version_id
        got["out_dir"] = Path(out_dir)
        return Path(out_dir) / "model.safetensors"

    monkeypatch.setattr(dl_mod, "download_civitai", _fake_civitai)
    out = asyncio.run(ensure_local("987654", provider="civitai", cache_dir=tmp_path))
    assert got["version_id"] == 987654
    assert got["out_dir"] == tmp_path / "civitai" / "987654"
    assert out.name == "model.safetensors"


@pytest.mark.parametrize(
    "ref,index",
    [
        ("acme/cozy-only#fp8", {"acme/cozy-only#fp8": "tensorhub"}),  # indexed tensorhub
        ("acme/unindexed", {"other/ref": "hf"}),                      # defaults to tensorhub
        ("acme/no-index", {}),                                        # no index at all
    ],
)
def test_tensorhub_refs_require_a_snapshot(tmp_path: Path, monkeypatch, ref, index) -> None:
    """Workers never resolve tensorhub refs themselves — the orchestrator
    pre-resolves and ships a Snapshot. Without one, the tensorhub branch
    raises RETRYABLE (a hub residency bug, not client input; #373) and the
    HF branch is NOT touched."""
    from gen_worker.api.errors import RetryableError

    calls: list = []
    monkeypatch.setattr(dl_mod, "download_hf", lambda *a, **k: calls.append(a) or tmp_path)
    set_provider_index(index)
    with pytest.raises(RetryableError, match="orchestrator-resolved snapshot"):
        asyncio.run(ensure_local(ref, cache_dir=tmp_path))
    assert calls == []


def test_ensure_local_failure_then_retry_succeeds(tmp_path: Path, monkeypatch) -> None:
    """Retry-after-failure: a failed attempt must not poison the ref — the
    next ensure_local call re-dispatches and can succeed."""
    snap = _make_weight_files(tmp_path / "snap", ["model.safetensors"])
    attempts = {"n": 0}

    def _flaky(ref, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient network failure")
        return snap

    monkeypatch.setattr(dl_mod, "download_hf", _flaky)
    with pytest.raises(RuntimeError, match="transient"):
        asyncio.run(ensure_local("owner/repo", provider="hf", cache_dir=tmp_path))
    out = asyncio.run(ensure_local("owner/repo", provider="hf", cache_dir=tmp_path))
    assert out == snap
    assert attempts["n"] == 2


# --------------------------------------------------------------------------- #
# Safetensors-only gate — real files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "files,raises",
    [
        (["model.safetensors", "config.json"], False),
        (["model.flashpack"], False),
        (["pytorch_model.bin", "model.safetensors"], False),  # safetensors sibling ok
        (["config.json", "tokenizer.json"], False),           # no weights -> no-op
        (["pytorch_model.bin"], True),
        (["model.pt"], True),
        (["model.ckpt"], True),
    ],
)
def test_safetensors_gate(tmp_path: Path, files: List[str], raises: bool) -> None:
    snap = _make_weight_files(tmp_path / "snap", files)
    if raises:
        with pytest.raises(UnsafeFileFormat, match="refusing to load"):
            assert_safe_weight_format(snap, ref="x/y")
    else:
        assert_safe_weight_format(snap, ref="x/y")
