"""The Dockerfile the docs teach must be one the hub will accept.

`docs/dockerfile.md` presents the canonical org Dockerfile — the file an
endpoint author is told to copy. tensorhub's publish validator refuses a
BuildKit cache mount outright (`builder/validation.go`: `ErrBuildKitCache`), so
a doc that teaches one leaves an org unable to publish (`400 invalid_tarball`),
and the doc has no first-party consumer to catch it — every
`inference-endpoints` family lets the hub synthesize its Dockerfile.

This is the guard that keeps the two sides from drifting apart again from
THIS repo, which is where the doc lives. It mirrors the hub's pattern rather
than importing it (separate repo, different language); the mirrored source is
named above so a change there has a place to land.

The hub matches the RAW BYTES of the whole Dockerfile, comments included, so
these checks deliberately do not strip comments either: a file that merely
mentions the directive while explaining the ban is refused exactly like one
that uses it.

The ban itself is the ``buildkit_cache_mount`` row of
``gen_worker.build_guarantees``, checked over every example and every documented
block alongside the other steps a hand-written Dockerfile owes. What stays HERE
is specific to this instance — the doc's prose surviving its own copy-paste, and
the size argument for ``g++`` over ``build-essential`` — read from the registry
rather than kept as a second spelling of the pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

from gen_worker.build_guarantees import RE_BUILDKIT_CACHE_MOUNT

REPO = Path(__file__).resolve().parents[1]

DOC = REPO / "docs" / "dockerfile.md"
FENCED_DOCKERFILE = re.compile(r"```dockerfile\n(.*?)```", re.DOTALL)


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


def test_the_doc_teaches_the_aot_host_toolchain_layer() -> None:
    """pgw#1017: the AUTHOR installs it, and the docs must say so.

    A custom Dockerfile is author-owned content — the platform verifies the
    toolchain at build time rather than injecting a layer into someone else's
    file. That only works if the file an author is TAUGHT to copy has the layer.

    This was `test_the_canonical_example_carries_the_aot_host_toolchain` and it
    read `examples/micro-diffusion/Dockerfile`. 56d89b7f (pgw#1373) deleted
    `examples/` — every one of them declared against the v1 SDK — so there is no
    canonical example in this repo to read, and the row's other half (the doc
    and the example teach the SAME layer) has nothing left to agree with. The
    doc is now the only place the layer is taught here, so the doc is what is
    asserted. The SHAPE argument below is why: `build_guarantees.cxx_toolchain`
    asks whether a layer exists; this asks whether it is the +80 MB one rather
    than the +250 MB one, which is a measured size argument and not a platform
    refusal.
    """
    doc = DOC.read_text()
    assert "ca-certificates curl g++" in doc, (
        "an endpoint declaring an AOT export needs a C++ compiler in its "
        "image; the build refuses without one (aot precondition cxx_toolchain)"
    )
    assert "--no-install-recommends" in doc
    # The doc NAMES build-essential in order to warn against it. That is the
    # opposite of the deleted example's assertion (where the word appearing in
    # an instruction was the defect) and it is the right claim for prose: the
    # teaching must be explicit, because an author reaching for a C++ compiler
    # reaches for build-essential by habit.
    assert "build-essential" in doc, (
        "the doc must say WHY not build-essential, not merely omit it: it "
        "drags ~250 MB of make/dpkg-dev the AOTI wrapper compile never "
        "invokes, against +80 MB for g++ with recommends off"
    )
