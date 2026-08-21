"""Property tests over the ref grammar, the CAS-ref parser, and the chunked-manifest entry decode."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest
from gen_worker._vendor.tensorfs import CASRef
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from gen_worker.models.hub_client import HubResolveError, parse_chunk_list, resolved_entry_digest
from gen_worker.models.refs import (
    MAX_FRAGMENT_LEN,
    REF_GRAMMAR_SEPARATORS,
    ParsedModelRef,
    format_model_ref,
    normalize_model_ref,
    parse_model_ref,
)

REF_VECTORS = pathlib.Path(__file__).parent / "testdata" / "ref_grammar_vectors.json"
HEX64 = "a" * 64


def _ref_seeds() -> list[str]:
    seeds = [
        "", " ", "/", "//", "a/", "/b", "owner/repo", "owner/repo:", "owner/repo@prod",
        "owner/repo@latest", "owner/repo#fp8", "owner/repo#FP8", "owner/repo#",
        "owner/repo:prod",
        "owner/repo:latest",
        "owner/repo#a#b", "owner/repo#fp8?attr=1",
        "owner/repo#" + "a" * (MAX_FRAGMENT_LEN + 1),
        f"owner/repo@sha256:{HEX64}", f"owner/repo@blake3:{'b' * 64}",
        f"owner/repo@SHA256:{HEX64.upper()}", "owner/repo@sha256:", "owner/repo@deadbeef",
        "owner/repo@v1",
        f"owner/repo:tag@sha256:{HEX64}#fp8",
        f"owner/rİpo@SHA256:{HEX64}",
        f"owner/repo@blake3:x@sha256:{HEX64}",
        "owner/repo/extra", "owner/repo:a:b", "0/0:/", "0/::", "  owner/repo  ",
    ]
    doc = json.loads(REF_VECTORS.read_text())
    for v in doc["vectors"]:
        seeds.append(v["ref"])
        for key in ("canonical", "address"):
            if v.get(key):
                seeds.append(v[key])
    return seeds


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.one_of(st.sampled_from(_ref_seeds()), st.text(max_size=60)))
@example("owner/repo")
@example("owner/repo:latest#fp8")
@example(f"owner/repo@sha256:{HEX64}")
@example(f"owner/rİpo@SHA256:{HEX64}")
@example("owner/repo/extra")
@example("0/::")
@example("owner/repo@prod?quant=plain.bf16@1")
@example("owner/repo@prod?")
def test_ref_normal_form_is_a_fixed_point(raw: str) -> None:
    """``parse(format(parse(s))) == address(parse(s))``, formatting idempotent."""
    try:
        parsed = parse_model_ref(raw)
    except ValueError:
        return
    normal = format_model_ref(parsed)
    again = parse_model_ref(str(normal))
    assert again == _address_of(parsed), (
        f"normal form is not a fixed point: {raw!r} -> {normal!r}")
    assert str(format_model_ref(again)) == str(normal), "formatting is not idempotent"
    assert str(normalize_model_ref(raw)) == str(normal)
    assert again.tensorhub is not None and again.tensorhub.lane_spec == ""


def _address_of(parsed: ParsedModelRef) -> ParsedModelRef:
    th = parsed.tensorhub
    assert th is not None
    return dataclasses.replace(
        parsed, tensorhub=dataclasses.replace(th, lane_spec=""))


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.one_of(st.sampled_from(_ref_seeds()), st.text(max_size=60)))
def test_ref_acceptance_implies_well_formed_components(raw: str) -> None:
    """Acceptance must mean every component was LOOKED AT, not merely kept."""
    try:
        parsed = parse_model_ref(raw)
    except ValueError:
        return
    th = parsed.tensorhub
    assert th is not None and parsed.provider == "tensorhub"
    assert th.owner and th.repo, f"{raw!r} accepted with an empty owner/repo"
    for name, part in (("owner", th.owner), ("repo", th.repo), ("release", th.release)):
        assert not any(c in part for c in REF_GRAMMAR_SEPARATORS), (
            f"{raw!r} accepted with a separator inside {name}={part!r}")
    if th.fragment is not None:
        assert th.owner == "root" and th.repo.startswith("family-"), (
            f"{raw!r} accepted a fragment on a non-compiled graph repo")
        assert th.fragment == th.fragment.lower()
        assert 1 <= len(th.fragment) <= MAX_FRAGMENT_LEN
    if th.digest is not None:
        assert ":" in th.digest, f"{raw!r} accepted an untagged digest {th.digest!r}"
        algo, _, _ = th.digest.partition(":")
        assert algo in ("sha256", "blake3"), f"{raw!r} accepted digest algorithm {algo!r}"


def test_ref_grammar_lower_index_pgw872() -> None:
    parsed = parse_model_ref(f"owner/r\u0130po@SHA256:{HEX64}").tensorhub
    assert parsed is not None
    assert parsed.repo == "r\u0130po", (
        f"repo={parsed.repo!r}: the split landed on an index taken from a lowercased copy"
    )
    assert parsed.digest == f"sha256:{HEX64}", f"digest={parsed.digest!r}"
    for spelling in ("@sha256:", "@SHA256:", "@ShA256:", "@sHa256:"):
        th = parse_model_ref(f"owner/repo{spelling}{HEX64}").tensorhub
        assert th is not None and th.repo == "repo" and th.digest == f"sha256:{HEX64}"


@settings(max_examples=400, deadline=None)
@given(st.one_of(
    st.text(max_size=80),
    st.sampled_from([
        "", " ", ":", "sha256:", ":" + HEX64, HEX64, HEX64.upper(),
        f"sha256:{HEX64}", f"blake3:{HEX64}", f"SHA256:{HEX64.upper()}",
        f"  sha256:{HEX64}  ", f"blake3:sha256:{HEX64}", f"sha256:{'a' * 63}",
        f"sha256:{'a' * 65}", f"md5:{'a' * 32}", f"sha256:{'g' * 64}",
        f"owner/repo@sha256:{HEX64}", "sha256:../../etc/passwd",
    ]),
))
@example(HEX64)
@example(f"sha256:{HEX64}")
def test_parse_cas_ref_acceptance_is_fully_determined(ref: str) -> None:
    """Acceptance implies a complete, self-consistent (algo, hex) pair, and the parse is idempotent under its own normal form."""
    try:
        parsed = CASRef.parse(ref)
    except ValueError:
        return
    hexpart = parsed.digest
    assert len(hexpart) == 64
    assert all(c in "0123456789abcdef" for c in hexpart)
    assert hexpart == hexpart.lower()
    assert CASRef.parse(str(parsed)) == parsed, "the normal form does not re-parse to itself"
    assert CASRef.parse(str(parsed).upper()) == parsed, "case changes the identity of a ref"


def test_bare_hex_is_refused_pgw871() -> None:
    with pytest.raises(ValueError, match="algorithm-tagged"):
        CASRef.parse(HEX64)
    with pytest.raises(ValueError):
        resolved_entry_digest({"digest": HEX64})
    assert CASRef.parse(f"sha256:{HEX64}") == CASRef(HEX64)
    with pytest.raises(ValueError, match="unsupported algorithm"):
        CASRef.parse(f"blake3:{HEX64}")


_CHUNK = st.fixed_dictionaries({}, optional={
    "digest": st.one_of(st.text(max_size=70), st.just(HEX64), st.just(f"sha256:{HEX64}"), st.none()),
    "sha256": st.one_of(st.text(max_size=70), st.just(HEX64), st.none()),
    "url": st.one_of(st.text(max_size=20), st.just("https://x/1"), st.none()),
    "len": st.one_of(st.integers(min_value=-2, max_value=1 << 40), st.none()),
    "length": st.one_of(st.integers(min_value=-2, max_value=1 << 40), st.none()),
})


@settings(max_examples=400, deadline=None)
@given(
    st.lists(st.one_of(_CHUNK, st.integers(), st.text(max_size=4), st.none()), max_size=4),
    st.one_of(st.none(), st.lists(st.text(max_size=20), max_size=5), st.integers()),
)
@example([{"digest": HEX64, "url": "https://x/1", "len": 5}], None)
@example([{"digest": HEX64, "len": 5}], ["https://x/1"])
@example([{"digest": HEX64, "len": 5}], [])
@example([{"digest": HEX64, "len": 5}], ["a", "b"])
@example([{"digest": HEX64, "url": "u", "len": 0}], None)
def test_chunk_list_is_all_or_typed_refusal(raw: Any, urls: Any) -> None:
    """A malformed chunk list is a HARD failure, never a silent short list."""
    try:
        out = parse_chunk_list("t", "p", raw, urls)
    except HubResolveError:
        return
    except (ValueError, TypeError, AttributeError) as exc:
        pytest.fail(
            f"parse_chunk_list raised an UNTYPED {type(exc).__name__}: {exc} — "
            "callers catch HubResolveError, so this crashes a resolve instead of refusing"
        )
    if not raw:
        assert out == ()
        return
    assert len(out) == len(raw), (
        "a chunk list must parse to exactly as many chunks as were declared; a short "
        "list silently truncates the file"
    )
    for chunk in out:
        assert len(chunk.sha256) == 64 and chunk.sha256 == chunk.sha256.lower()
        assert chunk.url, "a chunk with no URL cannot be fetched"
        assert chunk.length > 0, "a non-positive chunk length makes the offset meaningless"


@settings(max_examples=200, deadline=None)
@given(st.dictionaries(st.sampled_from(["digest", "path", "url"]),
                       st.one_of(st.text(max_size=70), st.none(), st.integers()), max_size=3))
@example({"digest": HEX64})
@example({"digest": f"sha256:{HEX64}"})
@example({})
def test_resolved_entry_digest_never_infers(entry: dict[str, Any]) -> None:
    """An integrity check with no digest is a REFUSAL, never a skip — and an untagged digest is never given an algorithm."""
    try:
        digest = resolved_entry_digest(entry)
    except ValueError:
        return
    assert ":" in digest, f"{entry!r} yielded an untagged digest {digest!r}"
    assert digest == digest.strip().lower()
