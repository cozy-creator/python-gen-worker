"""Provider routing + ensure_local dispatch (#366)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gen_worker.models.download import (
    build_provider_index_from_manifest,
    ensure_local,
    lookup_provider_for_ref,
    set_provider_index,
)


@pytest.fixture(autouse=True)
def _clean_index():
    set_provider_index({})
    yield
    set_provider_index({})


def _manifest(*binding_blocks: dict) -> dict:
    return {
        "entrypoints": [
            {"name": f"fn{i}", "bindings": b} for i, b in enumerate(binding_blocks)
        ]
    }


@pytest.mark.parametrize(
    "block,expect",
    [
        (
            {"pipeline": {"kind": "fixed", "provider": "hf",
                          "ref": "bfl/FLUX.2-klein-4B"}},
            {"bfl/FLUX.2-klein-4B": "hf"},
        ),
        (
            {"pipeline": {"kind": "fixed", "provider": "tensorhub",
                          "ref": "acme/flux@canary"}},
            {"acme/flux@canary": "tensorhub"},
        ),
        (
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
    assert build_provider_index_from_manifest({"entrypoints": []}) == {}
    bad = build_provider_index_from_manifest({
        "entrypoints": [
            {"bindings": {"pipeline": {"kind": "fixed", "provider": "hf"}}},
            {"bindings": "not-a-dict"},
            {"bindings": {"pipeline": {"kind": "fixed", "provider": "hf", "ref": "ok/ref"}}},
        ]
    })
    assert bad == {"ok/ref": "hf"}


def test_lookup_default_and_index() -> None:
    assert lookup_provider_for_ref("foo/bar") == "tensorhub"
    assert lookup_provider_for_ref("foo/bar", default="hf") == "hf"
    set_provider_index({"acme/flux@canary": "hf"})
    assert lookup_provider_for_ref("acme/flux@canary") == "hf"
    assert lookup_provider_for_ref("not/in-index") == "tensorhub"
    set_provider_index(None)
    assert lookup_provider_for_ref("acme/flux@canary") == "tensorhub"


def test_lookup_exact_tag_beats_repo_fallback() -> None:
    set_provider_index({"owner/flux@latest": "hf", "owner/flux@canary": "tensorhub"})
    assert lookup_provider_for_ref("owner/flux@canary") == "tensorhub"
    assert lookup_provider_for_ref("owner/flux@latest") == "hf"
    assert lookup_provider_for_ref("owner/flux@prod") == "hf"
    assert lookup_provider_for_ref("owner/flux@sha256:" + "ab" * 32) == "hf"
    set_provider_index(None)


@pytest.mark.parametrize(
    "wire_ref",
    [
        "bfl/FLUX.2-klein-4B@latest",
        "bfl/FLUX.2-klein-4B@prod",
        "bfl/FLUX.2-klein-4B",
    ],
)
def test_lookup_release_strip(wire_ref: str) -> None:
    assert lookup_provider_for_ref(wire_ref) == "tensorhub"
    set_provider_index({"bfl/FLUX.2-klein-4B": "hf"})
    assert lookup_provider_for_ref(wire_ref) == "hf"


def test_an_hf_indexed_ref_is_REFUSED_not_routed(tmp_path: Path) -> None:
    from gen_worker.models.errors import NonCasWeightSourceRefused

    set_provider_index({"bfl/FLUX.2-klein-4B": "hf"})
    with pytest.raises(NonCasWeightSourceRefused) as caught:
        asyncio.run(ensure_local("bfl/FLUX.2-klein-4B", cache_dir=tmp_path))
    assert caught.value.provider == "hf"
    assert not list(tmp_path.iterdir()), (
        "a refused ref must leave no cache directory behind")


def test_a_civitai_ref_is_REFUSED_not_routed(tmp_path: Path) -> None:
    from gen_worker.models.errors import NonCasWeightSourceRefused

    with pytest.raises(NonCasWeightSourceRefused) as caught:
        asyncio.run(ensure_local("987654", provider="civitai", cache_dir=tmp_path))
    assert caught.value.provider == "civitai"
    assert not (tmp_path / "civitai").exists(), (
        "the deleted branch used to mkdir a civitai staging dir; nothing may")


def test_a_modelscope_ref_is_REFUSED_not_routed(tmp_path: Path) -> None:
    from gen_worker.models.errors import NonCasWeightSourceRefused

    with pytest.raises(NonCasWeightSourceRefused) as caught:
        asyncio.run(ensure_local("org/model", provider="modelscope", cache_dir=tmp_path))
    assert caught.value.provider == "modelscope"


@pytest.mark.parametrize(
    "ref,index",
    [
        ("acme/cozy-only", {"acme/cozy-only": "tensorhub"}),
        ("acme/unindexed", {"other/ref": "hf"}),
        ("acme/no-index", {}),
    ],
)
def test_tensorhub_refs_require_a_snapshot(tmp_path: Path, ref, index) -> None:
    """Workers never resolve tensorhub refs themselves — the orchestrator pre-resolves and ships a Snapshot."""
    from gen_worker.models.errors import (
        MissingSnapshotError,
        NonCasWeightSourceRefused,
    )

    set_provider_index(index)
    with pytest.raises(MissingSnapshotError, match="orchestrator-resolved snapshot") as caught:
        asyncio.run(ensure_local(ref, cache_dir=tmp_path))
    assert not isinstance(caught.value, NonCasWeightSourceRefused)


def test_a_refused_ref_is_not_poisoned_and_refuses_again_identically(
    tmp_path: Path,
) -> None:
    """Retry-after-refusal."""
    from gen_worker.models.errors import NonCasWeightSourceRefused

    first: list = []
    for _ in range(2):
        with pytest.raises(NonCasWeightSourceRefused) as caught:
            asyncio.run(ensure_local("owner/repo", provider="hf", cache_dir=tmp_path))
        first.append(str(caught.value))
    assert first[0] == first[1]
