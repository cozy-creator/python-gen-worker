"""Provider routing + ensure_local dispatch (#366).

Exercises the REAL routing path of ``gen_worker.models.download``:

  * build_provider_index_from_manifest over a real endpoint.lock-shaped dict,
  * lookup_provider_for_ref against the process-global index,
  * ensure_local dispatching on that provider — which since pgw#1524 means
    dispatching to the right VERDICT: the CAS branch, the orchestrator-owes-a-
    resolve refusal, or the unservable-source refusal,
  * retry-after-refusal: the verdict is stable across calls.

Named regressions kept as explicit cases:
  * tag-stripping (live 2026-05-16 failure: ``:latest`` stamped HF ref),
  * HF #flavor suffix stripped before producing the huggingface_hub repo_id.
"""

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


# --------------------------------------------------------------------------- #
# build_provider_index_from_manifest — real endpoint.lock shapes
# --------------------------------------------------------------------------- #


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
    assert build_provider_index_from_manifest({"entrypoints": []}) == {}
    bad = build_provider_index_from_manifest({
        "entrypoints": [
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


def test_an_hf_indexed_ref_is_REFUSED_not_routed(tmp_path: Path) -> None:
    """pgw#1524: the index still ROUTES — it decides which verdict the ref
    gets — but "hf" no longer names a branch that fetches. It names the
    refusal, and the refusal carries the ingest route."""
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
        ("acme/cozy-only", {"acme/cozy-only": "tensorhub"}),          # indexed tensorhub
        ("acme/unindexed", {"other/ref": "hf"}),                      # defaults to tensorhub
        ("acme/no-index", {}),                                        # no index at all
    ],
)
def test_tensorhub_refs_require_a_snapshot(tmp_path: Path, ref, index) -> None:
    """Workers never resolve tensorhub refs themselves — the orchestrator
    pre-resolves and ships a Snapshot. Without one, the tensorhub branch raises
    the typed terminal MissingSnapshotError, which pgw#1524 keeps DISTINCT from
    the unservable-source refusal: this one the orchestrator can fix by
    re-minting, that one it never can."""
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
    """Retry-after-refusal. The old rig proved a transient fetch failure did
    not poison the ref; there is no fetch to be transient any more, so the
    property that matters is that the verdict is STABLE — a refusal that
    became something else on the second call would send a retrying caller
    somewhere new."""
    from gen_worker.models.errors import NonCasWeightSourceRefused

    first: list = []
    for _ in range(2):
        with pytest.raises(NonCasWeightSourceRefused) as caught:
            asyncio.run(ensure_local("owner/repo", provider="hf", cache_dir=tmp_path))
        first.append(str(caught.value))
    assert first[0] == first[1]
