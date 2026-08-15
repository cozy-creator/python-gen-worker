"""Property tests over the ref grammar, the CAS-ref parser, and the
chunked-manifest entry decode.

Three decode boundaries, one theme: each turns cross-service bytes into a value
the worker then acts on, and each fails SILENTLY rather than loudly.

* **ref grammar** (``gen_worker.models.refs``) is the Python half of a contract
  whose vectors are vendored byte-identically from tensorhub
  (``ref_grammar_vectors.json``). The vectors assert the cases somebody thought
  of; the properties here assert the ones nobody did — chiefly that the NORMAL
  FORM IS A FIXED POINT, because hub-minted refs and worker wire refs are
  compared byte-wise, so a ref that normalizes two ways is a cache miss that
  presents as a missing model.
* **CAS refs** (``hashrepo.CASRef``): a bare hex
  string must not default to ``blake3:``. ``len(digest) == 64`` cannot tell
  blake3 from sha256 — both are 32 bytes — so a length check is not a
  discriminator, it only looks like one.
* **chunked entries** (``gen_worker.models.hub_client.parse_chunk_list``): a
  malformed chunk list must be a hard failure, never a silent empty list. An
  empty list is indistinguishable from "stored whole", and reading a chunked
  file as whole is how a 40 GiB shard becomes a 0-byte one.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from hashrepo import CASRef
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from gen_worker.models.hub_client import HubResolveError, parse_chunk_list, resolved_entry_digest
from gen_worker.models.refs import (
    MAX_FRAGMENT_LEN,
    REF_GRAMMAR_SEPARATORS,
    format_model_ref,
    normalize_model_ref,
    parse_model_ref,
)

REF_VECTORS = pathlib.Path(__file__).parent / "testdata" / "ref_grammar_vectors.json"
HEX64 = "a" * 64


# ---------------------------------------------------------------------------
# ref grammar
# ---------------------------------------------------------------------------

def _ref_seeds() -> list[str]:
    seeds = [
        "", " ", "/", "//", "a/", "/b", "owner/repo", "owner/repo:", "owner/repo@prod",
        "owner/repo@latest", "owner/repo#fp8", "owner/repo#FP8", "owner/repo#",
        "owner/repo:prod",    # refused: the retired tag production
        "owner/repo:latest",  # refused: the retired tag production
        "owner/repo#a#b", "owner/repo#fp8?attr=1",
        # An over-long fragment. pgw#1213 widened the cap to MAX_FRAGMENT_LEN
        # so a `cg-key-v1` key (66 chars) fits, so the seed tracks the cap
        # rather than the old literal 64 — its job is to be one too long.
        "owner/repo#" + "a" * (MAX_FRAGMENT_LEN + 1),
        f"owner/repo@sha256:{HEX64}", f"owner/repo@blake3:{'b' * 64}",
        f"owner/repo@SHA256:{HEX64.upper()}", "owner/repo@sha256:", "owner/repo@deadbeef",
        "owner/repo@v1",
        f"owner/repo:tag@sha256:{HEX64}#fp8",  # refused: the retired tag production
        # th#1388's shape: a codepoint whose .lower() is not length-preserving,
        # which shifts an index computed on the lowercased copy.
        f"owner/rİpo@SHA256:{HEX64}",
        f"owner/repo@blake3:x@sha256:{HEX64}",
        "owner/repo/extra", "owner/repo:a:b", "0/0:/", "0/::", "  owner/repo  ",
    ]
    doc = json.loads(REF_VECTORS.read_text())
    for v in doc["vectors"]:
        seeds.append(v["ref"])
        if v.get("canonical"):
            seeds.append(v["canonical"])
    return seeds


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.one_of(st.sampled_from(_ref_seeds()), st.text(max_size=60)))
@example("owner/repo")
@example("owner/repo:latest#fp8")  # refused: the retired tag production
@example(f"owner/repo@sha256:{HEX64}")
@example(f"owner/rİpo@SHA256:{HEX64}")   # pgw#872 index/slice mismatch (fixed)
@example("owner/repo/extra")                   # th#1387 unbounded path segments
@example("0/::")                               # th#1387 round-trip break
def test_ref_normal_form_is_a_fixed_point(raw: str) -> None:
    """``parse(format(parse(s))) == parse(s)`` and formatting is idempotent.

    This is the property the whole grammar exists to provide.
    """
    try:
        parsed = parse_model_ref(raw)
    except ValueError:
        return  # a typed refusal is a correct outcome
    normal = format_model_ref(parsed)
    again = parse_model_ref(str(normal))
    assert again == parsed, f"normal form is not a fixed point: {raw!r} -> {normal!r}"
    assert str(format_model_ref(again)) == str(normal), "formatting is not idempotent"
    assert str(normalize_model_ref(raw)) == str(normal)


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
    # th#1387's hole is CLOSED (th#1987 re-key): no accepted component may
    # carry a grammar separator, which is what makes the normal form injective.
    for name, part in (("owner", th.owner), ("repo", th.repo), ("release", th.release)):
        assert not any(c in part for c in REF_GRAMMAR_SEPARATORS), (
            f"{raw!r} accepted with a separator inside {name}={part!r}")
    if th.flavor is not None:
        assert th.flavor == th.flavor.lower()
        assert 1 <= len(th.flavor) <= MAX_FRAGMENT_LEN
    if th.digest is not None:
        # A digest that reached a parsed ref is used to ADDRESS CAS objects, so
        # "present" is not enough — it must name its algorithm. An inferred
        # algorithm addresses the wrong namespace silently.
        assert ":" in th.digest, f"{raw!r} accepted an untagged digest {th.digest!r}"
        algo, _, _ = th.digest.partition(":")
        assert algo in ("sha256", "blake3"), f"{raw!r} accepted digest algorithm {algo!r}"


def test_ref_grammar_lower_index_pgw872() -> None:
    """pgw#872 FIXED (tensorhub twin: th#1388) — revert-turns-red guard.

    ``parse_model_ref`` used to compute ``low = s.lower()``, find ``@sha256:``
    with ``low.index(...)``, and slice ``s`` with that index. ``str.lower()`` is
    not length-preserving (``len("\u0130") == 1`` but ``len("\u0130".lower()) == 2``),
    so the split landed in the wrong place: the repo kept a stray ``@`` and the
    digest silently lost a hex character while still LOOKING like a digest — a
    wrong CAS address with no error anywhere. The Go twin PANICKED on the
    mirror-image input (``slice bounds out of range``), because ToLower GROWS
    invalid UTF-8.

    The fix on both sides is the same: index and slice the SAME string. Here
    ``_ascii_lower`` is a 1:1 character map, so an index taken on it is a valid
    index into the original.
    """
    parsed = parse_model_ref(f"owner/r\u0130po@SHA256:{HEX64}").tensorhub
    assert parsed is not None
    assert parsed.repo == "r\u0130po", (
        f"repo={parsed.repo!r}: the split landed on an index taken from a lowercased copy"
    )
    assert parsed.digest == f"sha256:{HEX64}", f"digest={parsed.digest!r}"
    # The fold itself: every case spelling of the marker finds the same split.
    for spelling in ("@sha256:", "@SHA256:", "@ShA256:", "@sHa256:"):
        th = parse_model_ref(f"owner/repo{spelling}{HEX64}").tensorhub
        assert th is not None and th.repo == "repo" and th.digest == f"sha256:{HEX64}"


# ---------------------------------------------------------------------------
# CAS refs
# ---------------------------------------------------------------------------

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
@example(HEX64)             # the th#1357 museum piece
@example(f"sha256:{HEX64}")
def test_parse_cas_ref_acceptance_is_fully_determined(ref: str) -> None:
    """Acceptance implies a complete, self-consistent (algo, hex) pair, and the
    parse is idempotent under its own normal form."""
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
    """pgw#871 FIXED — the two CAS readers in this repo agree with each other
    AND with the hub. Revert-turns-red guard.

    th#1357 DELETED the bare-hex read-path default from the hub:
    ``storage.ParseCASRef`` refuses an untagged ref outright ("bare hex is
    refused; write \"sha256:<hex>\""). ``parse_cas_ref`` used to infer
    ``blake3`` from bare hex while its docstring claimed to match the hub — so
    the same 64-character string the hub REFUSED, this side resolved, into the
    WRONG namespace: a 64-char hex cannot distinguish blake3 from sha256,
    because both digests are 32 bytes. The length check is not a discriminator,
    it only looks like one.
    """
    with pytest.raises(ValueError, match="algorithm-tagged"):
        CASRef.parse(HEX64)
    # The other CAS-digest reader in this repo, which already refused it.
    with pytest.raises(ValueError):
        resolved_entry_digest({"digest": HEX64})
    assert CASRef.parse(f"sha256:{HEX64}") == CASRef(HEX64)
    with pytest.raises(ValueError, match="unsupported algorithm"):
        CASRef.parse(f"blake3:{HEX64}")


# ---------------------------------------------------------------------------
# chunked manifest entries
# ---------------------------------------------------------------------------

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
@example([{"digest": HEX64, "len": 5}], [])                    # empty url list
@example([{"digest": HEX64, "len": 5}], ["a", "b"])            # misaligned url list
@example([{"digest": HEX64, "url": "u", "len": 0}], None)      # the `or 0` launder
def test_chunk_list_is_all_or_typed_refusal(raw: Any, urls: Any) -> None:
    """A malformed chunk list is a HARD failure, never a silent short list.

    Index alignment is the only thing binding a URL to its digest, so a list
    that parses to FEWER chunks than were declared is the defect: it reads as
    "fetch fewer chunks" and the file is silently truncated.
    """
    try:
        out = parse_chunk_list("t", "p", raw, urls)
    except HubResolveError:
        return  # typed refusal — the correct outcome for anything malformed
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
@example({"digest": HEX64})                 # untagged — must refuse
@example({"digest": f"sha256:{HEX64}"})
@example({})
def test_resolved_entry_digest_never_infers(entry: dict[str, Any]) -> None:
    """An integrity check with no digest is a REFUSAL, never a skip — and an
    untagged digest is never given an algorithm."""
    try:
        digest = resolved_entry_digest(entry)
    except ValueError:
        return
    assert ":" in digest, f"{entry!r} yielded an untagged digest {digest!r}"
    assert digest == digest.strip().lower()
