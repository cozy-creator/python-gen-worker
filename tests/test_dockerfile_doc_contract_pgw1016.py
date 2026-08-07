"""The Dockerfile the docs teach must be one the hub will accept (pgw#1016).

`docs/dockerfile.md` presents the canonical org Dockerfile — the file an
endpoint author is told to copy. It taught a BuildKit cache mount, and
tensorhub's publish validator refuses one outright
(`builder/validation.go`: `ErrBuildKitCache`), so an org that followed the
documentation could not publish: `400 invalid_tarball`. The doc had no
first-party consumer — every `inference-endpoints` family lets the hub
synthesize its Dockerfile — so nothing ever exercised it.

This is the guard that keeps the two sides from drifting apart again from
THIS repo, which is where the doc lives. It mirrors the hub's pattern rather
than importing it (separate repo, different language); the mirrored source is
named above so a change there has a place to land.

The hub matches the RAW BYTES of the whole Dockerfile, comments included, so
these checks deliberately do not strip comments either: a file that merely
mentions the directive while explaining the ban is refused exactly like one
that uses it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# tensorhub/internal/builder/validation.go: reBuildKitCacheMount
RE_BUILDKIT_CACHE_MOUNT = re.compile(r"(?i)--mount\s*=\s*type\s*=\s*cache\b")

DOC = REPO / "docs" / "dockerfile.md"
FENCED_DOCKERFILE = re.compile(r"```dockerfile\n(.*?)```", re.DOTALL)


def _example_dockerfiles() -> list[Path]:
    return sorted((REPO / "examples").rglob("Dockerfile"))


def test_doc_dockerfile_blocks_pass_the_hub_validator() -> None:
    blocks = FENCED_DOCKERFILE.findall(DOC.read_text())
    assert blocks, "docs/dockerfile.md has no dockerfile blocks to check"
    for block in blocks:
        assert not RE_BUILDKIT_CACHE_MOUNT.search(block), (
            "a documented Dockerfile block carries a BuildKit cache mount; the "
            "hub refuses it at publish (400 invalid_tarball). Shared cache ids "
            "are a cross-tenant poisoning channel — install uncached, or "
            "namespace the id per org and teach the validator to enforce it"
        )


def test_the_ban_is_explained_without_spelling_the_directive() -> None:
    """The prose must survive its own copy-paste.

    The validator reads comments too, so a `why` paragraph that spells the
    directive becomes a refusal the moment an author pastes it into a comment.
    """
    assert not RE_BUILDKIT_CACHE_MOUNT.search(DOC.read_text()), (
        "docs/dockerfile.md spells the banned directive in prose; pasting that "
        "sentence into a Dockerfile comment is refused exactly like using it"
    )


@pytest.mark.parametrize(
    "path", _example_dockerfiles(), ids=lambda p: p.parent.name
)
def test_example_dockerfiles_pass_the_hub_validator(path: Path) -> None:
    assert not RE_BUILDKIT_CACHE_MOUNT.search(path.read_text()), (
        f"{path.relative_to(REPO)} carries a BuildKit cache mount and cannot "
        f"be published"
    )


def test_the_canonical_example_carries_the_aot_host_toolchain() -> None:
    """pgw#1017: the AUTHOR installs it, and the docs must say so.

    A custom Dockerfile is author-owned content — the platform verifies the
    toolchain at build time (`aot_preconditions.CHECK_CXX_TOOLCHAIN`) rather
    than injecting a layer into someone else's file. That only works if the
    canonical file an author copies has the layer.
    """
    path = REPO / "examples" / "micro-diffusion" / "Dockerfile"
    content = path.read_text()
    assert "ca-certificates curl g++" in content, (
        "micro-diffusion declares an AOT export, so its image must carry a C++ "
        "compiler; the build refuses without one (aot precondition "
        "cxx_toolchain)"
    )
    assert "--no-install-recommends" in content
    instructions = "\n".join(
        line for line in content.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "build-essential" not in instructions, (
        "build-essential drags ~250 MB of make/dpkg-dev the AOTI wrapper "
        "compile never invokes; g++ with recommends off is +80 MB"
    )
    assert "ca-certificates curl g++" in DOC.read_text(), (
        "the doc and the example must teach the same toolchain layer"
    )
