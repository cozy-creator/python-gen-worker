"""Provider routing + ensure_local dispatch (#366).

Exercises the REAL routing path of ``gen_worker.models.download``:

  * build_provider_index_from_manifest over a real endpoint.lock-shaped dict,
  * lookup_provider_for_ref against the process-global index,
  * ensure_local dispatching to the right provider branch (hf/civitai leaves
    stubbed only at the network edge — every routing decision is real),
  * retry-after-failure: a failed ensure_local attempt does not poison the ref.

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
            # a manifest `flavor` field is DEAD (§1.32(d)) — the key
            # is the repo, and an entry that still carries one is not a
            # second address.
            {"pipeline": {"kind": "fixed", "provider": "hf",
                          "ref": "bfl/FLUX.2-klein-4B"}},
            {"bfl/FLUX.2-klein-4B": "hf"},
        ),
        (
            # th#1987: the release rides the REF; there is no side-channel
            # key left to fold, and the normal form keys the index.
            {"pipeline": {"kind": "fixed", "provider": "tensorhub",
                          "ref": "acme/flux@canary"}},
            {"acme/flux@canary": "tensorhub"},
        ),
        (
            # retired dispatch-kind entries are ignored, not parsed.
            {"pipeline": {"kind": "dispatch", "field": "variant", "table": {
                "bf16": {"provider": "hf", "ref": "owner/flux"},
            }}},
            {"owner/flux": None},
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
    set_provider_index({"acme/flux@canary": "hf"})
    assert lookup_provider_for_ref("acme/flux@canary") == "hf"
    assert lookup_provider_for_ref("not/in-index") == "tensorhub"
    set_provider_index(None)
    assert lookup_provider_for_ref("acme/flux@canary") == "tensorhub"  # cleared


def test_lookup_exact_tag_beats_repo_fallback() -> None:
    """pgw#1148: the index key is (repo, tag), not (repo, flavor) — the
    flavor was the sub-selector §1.32(d) deleted. The exact normal-form key
    still disambiguates, and a hub-minted DIGEST pick routes via the
    repo-identity fallback."""
    set_provider_index({"owner/flux@latest": "hf", "owner/flux@canary": "tensorhub"})
    assert lookup_provider_for_ref("owner/flux@canary") == "tensorhub"
    assert lookup_provider_for_ref("owner/flux@latest") == "hf"
    # normalization: the DEFAULT tag (':prod', th#1276) folds to the bare key,
    # which is not indexed here -> the repo-identity fallback.
    assert lookup_provider_for_ref("owner/flux@prod") == "hf"
    # a digest pick is not an indexed key -> repo-identity fallback
    assert lookup_provider_for_ref("owner/flux@sha256:" + "ab" * 32) == "hf"
    set_provider_index(None)


@pytest.mark.parametrize(
    "wire_ref",
    [
        "bfl/FLUX.2-klein-4B@latest",  # a release-addressed spelling
        "bfl/FLUX.2-klein-4B@prod",    # the release that used to be the default
        "bfl/FLUX.2-klein-4B",         # bare form (no regression)
    ],
)
def test_lookup_release_strip(wire_ref: str) -> None:
    """Live 2026-05-16 failure: a runtime payload stamps a release onto an HF
    ref but the index only carries the bare HF form, so the lookup must strip
    it (a tensorhub release is meaningless for HF).

    th#2031 re-spelled the fixtures: the `#bf16` tail these carried is now a
    typed refusal, and the release strip is what the rows were ever about."""
    assert lookup_provider_for_ref(wire_ref) == "tensorhub"  # default before install
    set_provider_index({"bfl/FLUX.2-klein-4B": "hf"})
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
    set_provider_index({"bfl/FLUX.2-klein-4B": "hf"})
    out = asyncio.run(ensure_local("bfl/FLUX.2-klein-4B", cache_dir=tmp_path))
    assert len(calls) == 1
    assert calls[0].repo_id == "bfl/FLUX.2-klein-4B"
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
        ("acme/cozy-only", {"acme/cozy-only": "tensorhub"}),          # indexed tensorhub
        ("acme/unindexed", {"other/ref": "hf"}),                      # defaults to tensorhub
        ("acme/no-index", {}),                                        # no index at all
    ],
)
def test_tensorhub_refs_require_a_snapshot(tmp_path: Path, monkeypatch, ref, index) -> None:
    """Workers never resolve tensorhub refs themselves — the orchestrator
    pre-resolves and ships a Snapshot. Without one, the tensorhub branch
    raises the typed terminal MissingSnapshotError and the
    HF branch is NOT touched."""
    from gen_worker.models.errors import MissingSnapshotError

    calls: list = []
    monkeypatch.setattr(dl_mod, "download_hf", lambda *a, **k: calls.append(a) or tmp_path)
    set_provider_index(index)
    with pytest.raises(MissingSnapshotError, match="orchestrator-resolved snapshot"):
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
