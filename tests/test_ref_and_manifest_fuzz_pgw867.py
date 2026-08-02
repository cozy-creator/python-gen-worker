"""pgw#867 / th#1382 — property tests over the ref grammar, the CAS-ref parser,
and the chunked-manifest entry decode.

Three decode boundaries, one theme: each turns cross-service bytes into a value
the worker then acts on, and each has a history of being wrong SILENTLY rather
than loudly.

* **ref grammar** (``gen_worker.models.refs``) is the Python half of a contract
  whose vectors are vendored byte-identically from tensorhub
  (``ref_grammar_vectors.json``, th#1276). The vectors assert the cases somebody
  thought of; the properties here assert the ones nobody did — chiefly that the
  NORMAL FORM IS A FIXED POINT, because hub-minted refs and worker wire refs are
  compared byte-wise, so a ref that normalizes two ways is a cache miss that
  presents as a missing model.
* **CAS refs** (``gen_worker.models.chunk_cas.parse_cas_ref``) is where a bare
  hex string silently defaulted to ``blake3:`` for years. ``len(digest) == 64``
  cannot tell blake3 from sha256 — both are 32 bytes — so a length check is not
  a discriminator, it only looks like one.
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
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from gen_worker.models.chunk_cas import parse_cas_ref
from gen_worker.models.hub_client import HubResolveError, parse_chunk_list, resolved_entry_digest
from gen_worker.models.refs import (
    DEFAULT_REF_TAG,
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
        "", " ", "/", "//", "a/", "/b", "owner/repo", "owner/repo:", "owner/repo:prod",
        "owner/repo:latest", "owner/repo#fp8", "owner/repo#FP8", "owner/repo#",
        "owner/repo#a#b", "owner/repo#fp8?attr=1", "owner/repo#" + "a" * 65,
        f"owner/repo@sha256:{HEX64}", f"owner/repo@blake3:{'b' * 64}",
        f"owner/repo@SHA256:{HEX64.upper()}", "owner/repo@sha256:", "owner/repo@deadbeef",
        "owner/repo@v1", f"owner/repo:tag@sha256:{HEX64}#fp8",
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


def _known_grammar_hole_th1387(parsed: Any) -> bool:
    """th#1387 / pgw#872 — the grammar applies NO charset check to owner, repo or
    tag, so a component can carry a separator the grammar excludes and the normal
    form stops being injective. Verified identical in tensorhub's
    ``ParseCanonicalRef``, so this is a SHARED hole, not a divergence — which is
    the point worth keeping: a defect both sides implement the same way is
    invisible to the shared fixture and to the conformance test, and only the
    single-language invariants can see it."""
    th = parsed.tensorhub
    return (
        any(c in th.owner for c in "/@:#")
        or any(c in th.repo for c in "/@:#")
        or any(c in th.tag for c in "/@:#")
    )


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.one_of(st.sampled_from(_ref_seeds()), st.text(max_size=60)))
@example("owner/repo")
@example("owner/repo:latest#fp8")
@example(f"owner/repo@sha256:{HEX64}")
@example(f"owner/rİpo@SHA256:{HEX64}")   # th#1388 index/slice mismatch
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
    if _known_grammar_hole_th1387(parsed):
        return
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
    assert th.tag, f"{raw!r} accepted with an empty tag; the default is {DEFAULT_REF_TAG!r}"
    if th.flavor is not None:
        assert th.flavor == th.flavor.lower()
        assert 1 <= len(th.flavor) <= 64
    if th.digest is not None:
        # A digest that reached a parsed ref is used to ADDRESS CAS objects, so
        # "present" is not enough — it must name its algorithm. An inferred
        # algorithm addresses the wrong namespace silently (th#1357).
        assert ":" in th.digest, f"{raw!r} accepted an untagged digest {th.digest!r}"
        algo, _, _ = th.digest.partition(":")
        assert algo in ("sha256", "blake3"), f"{raw!r} accepted digest algorithm {algo!r}"


def test_ref_grammar_lower_index_ledger_pgw872() -> None:
    """LEDGER for pgw#872 (tensorhub twin: th#1388).

    ``parse_model_ref`` computes ``low = s.lower()``, finds ``@sha256:`` with
    ``low.index(...)``, and slices ``s`` with that index. ``str.lower()`` is not
    length-preserving (``len("İ") == 1`` but ``len("İ".lower()) == 2``),
    so the split lands in the wrong place: the repo keeps a stray ``@`` and the
    digest silently loses a hex character while still LOOKING like a digest — a
    wrong CAS address with no error anywhere.

    The Go twin panics instead (``slice bounds out of range``) because Go slices
    bytes; same cause, louder symptom. Recorded, not fixed here.
    """
    parsed = parse_model_ref(f"owner/rİpo@SHA256:{HEX64}").tensorhub
    assert parsed is not None
    if parsed.repo == "rİpo" and parsed.digest == f"sha256:{HEX64}":
        pytest.fail(
            "pgw#872 appears fixed — remove this ledger and the th#1387/#1388 "
            "skips from the property tests above"
        )
    assert parsed.repo.endswith("@"), "pgw#872 ledger drifted: the stray '@' is gone"
    _, _, hexpart = (parsed.digest or "").partition(":")
    assert len(hexpart) == 63, f"pgw#872 ledger drifted: digest width is {len(hexpart)}"


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
        algo, hexpart = parse_cas_ref(ref)
    except ValueError:
        return
    assert algo in ("sha256", "blake3")
    assert len(hexpart) == 64
    assert all(c in "0123456789abcdef" for c in hexpart)
    assert hexpart == hexpart.lower()
    algo2, hex2 = parse_cas_ref(f"{algo}:{hexpart}")
    assert (algo2, hex2) == (algo, hexpart), "the normal form does not re-parse to itself"
    algo3, hex3 = parse_cas_ref(f"{algo}:{hexpart}".upper())
    assert (algo3, hex3) == (algo, hexpart), "case changes the identity of a ref"


def test_bare_hex_still_infers_blake3_ledger_pgw871() -> None:
    """LEDGER for pgw#871 — the th#1357 defect, still live on this side.

    th#1357 DELETED the bare-hex read-path default from the hub:
    ``storage.ParseCASRef`` now refuses an untagged ref outright ("bare hex is
    refused; write \\"sha256:<hex>\\""). This function still infers ``blake3``,
    and its docstring still claims it is "matching the hub's read-path rule" —
    a claim that is no longer true.

    That is the asymmetry the bug history warns about: a digest the hub refuses,
    this side resolves — into the WRONG namespace, since a 64-char hex cannot
    distinguish blake3 from sha256. Reachable from the snapshot download/verify
    path (``models/cozy_snapshot.py``), which decodes hub-delivered manifest
    entries. Filed, not fixed in passing.
    """
    algo, hexpart = parse_cas_ref(HEX64)
    if algo != "blake3":
        pytest.fail(
            f"pgw#871 appears fixed (bare hex now parses as {algo!r}) — delete this ledger"
        )
    assert hexpart == HEX64
    # And the asymmetry itself: the same string is a REFUSAL on the hub side.
    with pytest.raises(ValueError):
        resolved_entry_digest({"digest": HEX64})


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
