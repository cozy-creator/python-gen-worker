"""THE ref grammar module (gw#492): parse + format + fold, nothing else mints.

Normal form — the ONE canonical string for a model ref (grammar th#597 C5,
re-keyed by th#1987; shared vectors ``tests/testdata/ref_grammar_vectors.json``):

    tensorhub:  owner/repo[@<release>|@sha256:<hex>|@blake3:<hex>][?<lane-spec>][#<cell-fragment>]
    hf:         owner/repo[@revision]

THE `:tag` PRODUCTION IS DEAD (th#1987, HARDCUT A9). A tag was a movable
pointer with a supplied default; a release is cut once, deliberately, and is
immutable. There is NO default and NOTHING is elided: a bare ``owner/repo``
names a repo and no artifact inside it, and a hub resolve for one is the
terminal ``release_not_found`` (``catalog.ResolveCheckpointSelector``). A ref
carrying ``:`` raises :class:`RetiredTagRef` naming the remedy — the
client-side twin of the hub's own refusal, so a stale pin is told what to
write instead of resolving to nothing on a rented pod.

``format(parse(s))`` is the normalization projection; ``parse(format(v)) == v``
for every value. A digest WINS over a release when both are set (the digest is
exact; the release is the set it was picked from) and they share one ``@`` slot
because a second one destroys injectivity (th#1387). Every ref string the
worker mints (wire, residency keys, cache keys, telemetry) MUST come from
:func:`wire_ref` (bindings), :func:`fold_ref` (string + release overlay), or
:func:`format_model_ref` / ``.canonical()`` (parsed values).

THE `?<lane-spec>` TAIL IS A RESOLUTION INPUT, NEVER AN ADDRESS (th#2006).
It is the compact tensor-layout-contract pattern the hub's config record
already speaks (``endpointconfig.ParseCompact`` → ``contractspec.ParsePattern``)
and it names WHICH variant of a release the caller means when the release holds
more than one. This module cuts it and carries it BESIDE the address, in
:attr:`TensorhubRef.lane_spec`; :meth:`TensorhubRef.canonical` DROPS it, because
canonical() is the wire/residency-key minter and a spec on a wire ref is an
unread copy of a decision the hub already made. Go's ``CanonicalRef.String``
keeps it and ``CanonicalRef.Address`` drops it — ``canonical()`` is the twin of
``Address().String()``, not of ``String()``. No validation happens here: this
module owns the grammar's SHAPE, the hub owns what a pattern may say.

THE FLAVOR IS DEAD ENTIRELY (pgw#1290 / th#2031, completing §1.32(d)). The
`#` tail has exactly ONE meaning left — the COMPILE CELL fragment of a
platform cell repo, ``root/family-<f>#<key>`` — and :func:`parse_model_ref`
REFUSES it on every other repo with :class:`RefFragmentRemoved`. The refusal
moved into the parser because a fragment that parses and is then dropped
resolves to the release default and reads as success; the earlier
:func:`refuse_ref_fragment` chokepoint covered the weight paths that
remembered to call it and nothing else. Selection within a release is the
`?<lane-spec>` tail plus tensor-layout contract compatibility (§1.33,
``Slot(layouts=…)``); one exact checkpoint is ``owner/repo@sha256:…``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType, Optional

from gen_worker.refgrammar import MAX_FRAGMENT_LEN as _MAX_FRAGMENT_LEN

# th#597 C5: `#` fragment charset [a-z0-9][a-z0-9._-]*, bounded by
# MAX_FRAGMENT_LEN (matches tensorhub's isValidFragmentToken). Cell fragments
# only — see the module docstring.
#
# th#1897/pgw#1213: the bound is 96, MIRRORING tensorhub's
# internal/refgrammar.MaxFragmentLen byte-for-byte. It lives in
# gen_worker.refgrammar for the same reason Go puts it in a leaf package —
# the ref parser and the compiled-graph key grammar both need it and neither
# may import the other. The boundary is pinned by vectors at 96 and 97 in BOTH
# vendored corpora, so a one-sided change to this number fails a gate rather
# than surfacing 45 minutes into a mint.
MAX_FRAGMENT_LEN = _MAX_FRAGMENT_LEN
_TENSORHUB_FRAGMENT_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]{0,%d}" % (MAX_FRAGMENT_LEN - 1)
)

# The grammar's OWN delimiters. th#1387: a component carrying one destroys
# injectivity — String() re-parses to a different value, and the hub compares
# its minted refs against worker wire refs BYTE-WISE, so a ref that normalizes
# two ways is a cache miss presenting as a missing model. Structural, not a
# naming charset: it says nothing about which letters a repo may use.
REF_GRAMMAR_SEPARATORS = "/@:#"


class RefFragmentRemoved(ValueError):
    """A ref carried a `#` fragment somewhere it means nothing. §1.32(d) /
    th#1803 / th#2031: THE FLAVOR SYSTEM IS DEAD — deleted, not aliased.

    The client-side twin of the hub's ``ref_fragment_removed`` /
    ``binding_variant_selector_removed`` 400: the SDK refuses at the boundary
    rather than minting a ref the hub will reject, so the caller is told what
    to write instead of reading a server error about a selector they thought
    was supported.
    """


class RetiredTagRef(ValueError):
    """A ref carried a `:tag`. th#1987 DELETED the tag production.

    The client-side twin of tensorhub's `ParseCanonicalRef` refusal. Refusing
    here rather than letting the colon land inside the repo name matters: a
    parsed ``repo:prod`` addresses a repo nobody has, so the defect would
    surface as an empty resolve on a rented pod instead of at the line that
    wrote the pin.
    """


def _fragment_removed_message(ref: str) -> str:
    return (
        f"model ref {ref!r} carries a `#` fragment, which names a COMPILE CELL "
        "key on root/family-<f> and nothing else (th#2031 — deleted, not "
        "aliased). Narrow the variant with the lane-spec tail "
        "'owner/repo@<release>?<contract pattern>', declare what the code "
        "accepts with Slot(layouts=...), or address one exact checkpoint with "
        "'owner/repo@sha256:<hex>'."
    )


def _retired_tag_message(ref: str) -> str:
    return (
        f"model ref {ref!r} carries a ':' tag; tags were deleted (th#1987) — "
        "write 'owner/repo@<release>' (cut a release, then attach artifacts "
        "to it). There is no default and no floating pointer to inherit."
    )


def refuse_ref_fragment(ref: str, *, where: str = "") -> None:
    """Refuse a WEIGHT ref that carries a `#` fragment, naming the site.

    :func:`parse_model_ref` already refuses the same strings; this exists so
    the message can name the binding or resolution the author wrote, and so a
    path that never parses still refuses.
    """
    s = (ref or "").strip()
    if "#" not in s:
        return
    msg = _fragment_removed_message(s)
    raise RefFragmentRemoved(f"{where}: {msg}" if where else msg)


# pgw#872 / th#1388: a LENGTH-PRESERVING case fold. An index computed on
# ``s.lower()`` is not a valid index into ``s`` — the map is 1:1 here, so it is.
# Only ASCII case matters: every marker this module searches for is ASCII.
_ASCII_LOWER = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)


def _ascii_lower(s: str) -> str:
    return s.translate(_ASCII_LOWER)

# A ref string in NORMAL FORM (minted by this module). Annotate wire/residency
# key surfaces with WireRef so mixing raw and normalized strings fails mypy.
WireRef = NewType("WireRef", str)


@dataclass(frozen=True)
class TensorhubRef:
    owner: str
    repo: str
    #: The author-chosen release identifier the non-digest ``@`` tail named.
    #: There is NO default: "" means the ref addresses the repo and nothing
    #: inside it, exactly as tensorhub's ``CanonicalRef.Release`` does.
    release: str = ""
    digest: Optional[str] = None  # snapshot digest, including algorithm prefix (e.g. "blake3:<hex>")
    #: The `#` tail. A COMPILE CELL fragment (``root/family-<f>#<key>``)
    #: and nothing else — never a weight selector (§1.32(d), th#2031). The
    #: parser refuses it on any other repo; the compile cache is the one
    #: reader (``compile_cache.parse_cell_ref``).
    fragment: Optional[str] = None
    #: The `?` tail (th#2006): the compact contract pattern naming WHICH
    #: variant of the release is meant. A resolution input carried BESIDE the
    #: address — never in :meth:`canonical`, never on the wire, never in a
    #: residency key. "" means the ref states no preference.
    lane_spec: str = ""

    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"

    def canonical(self) -> "WireRef":
        """Normal form: ``owner/repo[@release|@digest][#cell-fragment]``.

        Nothing is elided (th#1987) and the digest takes the single ``@`` slot
        when both are set. The `?<lane-spec>` tail is DROPPED (th#2006): this
        is the ADDRESS, and the spec is a resolution input. Tensorhub is the
        default provider so no prefix is emitted; consumers track provider
        separately."""
        out = self.repo_id()
        if self.digest:
            out = f"{out}@{self.digest}"
        elif self.release:
            out = f"{out}@{self.release}"
        if self.fragment:
            out = f"{out}#{self.fragment}"
        return WireRef(out)


@dataclass(frozen=True)
class HuggingFaceRef:
    repo_id: str
    revision: Optional[str] = None

    def canonical(self) -> "WireRef":
        """Normal form ``owner/repo[@revision]``. Provider is tracked
        separately.

        pgw#1148: the `#flavor` tail and the CACHE-KEY FOLD that gave two
        flavors of one HF repo two distinct residency entries are DELETED.
        HF has no flavor axis of its own — it never did; the tail was the
        orchestrator's per-flavor routing convention, and §1.32(d) deleted
        the flavor as an address. File selection is binding metadata
        (``allow_patterns``), carried beside the ref and never inside it.
        """
        base = self.repo_id
        if self.revision:
            base = f"{base}@{self.revision}"
        return WireRef(base)


@dataclass(frozen=True)
class CivitaiRef:
    model_id: str

    def canonical(self) -> "WireRef":
        return WireRef(self.model_id)


@dataclass(frozen=True)
class ModelScopeRef:
    repo_id: str
    revision: Optional[str] = None

    def canonical(self) -> "WireRef":
        if self.revision:
            return WireRef(f"{self.repo_id}@{self.revision}")
        return WireRef(self.repo_id)


@dataclass(frozen=True)
class ParsedModelRef:
    """Decoded model ref.

    ``provider`` is the canonical provider tag — ``"tensorhub"`` (default),
    ``"hf"`` (huggingface), or ``"civitai"`` — matching the binding class's
    ``PROVIDER`` constant. Exactly one of the typed payload fields is
    populated to match the provider tag.
    """

    provider: str  # "tensorhub" | "hf" | "civitai" | "modelscope"
    tensorhub: Optional[TensorhubRef] = None
    hf: Optional[HuggingFaceRef] = None
    civitai: Optional[CivitaiRef] = None
    modelscope: Optional[ModelScopeRef] = None


def _parse_tensorhub_ref(raw: str, s: str) -> TensorhubRef:
    """The tensorhub production, mirroring ``release.ParseCanonicalRef``
    decision for decision. The order is load-bearing: fragment, lane spec,
    digest, release, then the retired-tag refusal, then owner/repo."""
    digest = None
    fragment = None
    lane_spec = ""

    if "#" in s:
        s, fragment_part = s.split("#", 1)
        fragment_part = fragment_part.strip()
        if "?" in fragment_part:
            fragment_part = fragment_part.split("?", 1)[0].strip()
        fragment_part = fragment_part.lower()
        if not fragment_part:
            raise ValueError("tensorhub ref fragment is empty")
        # th#597 C5: ONE fragment token per ref, charset
        # [a-z0-9][a-z0-9._-]{0,95} — `#a#b` is invalid (cells encode
        # conjunction inside one token). Shared grammar vectors:
        # tests/testdata/ref_grammar_vectors.json (byte-identical copy in
        # tensorhub internal/orchestrator/release/testdata/, whose Go
        # ParseCanonicalRef still parses the tail for the same reason).
        if not _TENSORHUB_FRAGMENT_RE.fullmatch(fragment_part):
            raise ValueError(
                f"tensorhub ref fragment {fragment_part!r} is not a valid token"
            )
        fragment = fragment_part
        s = s.strip()

    # th#2006: the LANE SPEC tail, `?<compact contract pattern>`. Cut AFTER the
    # fragment split, exactly as the Go twin does, so the older "a trailing
    # ?query on the FRAGMENT is stripped" rule (lockfile attribution refs) is
    # untouched — only a `?` on the address half is a spec. Not validated here:
    # this module owns the grammar's shape, `contractspec` owns the pattern.
    if "?" in s:
        s, spec_part = s.split("?", 1)
        s = s.strip()
        lane_spec = spec_part.strip()
        if not lane_spec:
            raise ValueError(
                "tensorhub ref lane spec is empty; omit the '?' entirely to "
                "mean any variant")

    # th#1387: take the EARLIEST digest marker in the string, not the first
    # algorithm in a fixed list — scanning by algorithm made
    # "@blake3:x@sha256:<hex>" split on the LATER marker, silently absorbing
    # "@blake3" into the repo name.
    #
    # pgw#872: the index MUST come from the same string it slices.
    # ``str.lower()`` is not length-preserving (``len("İ") == 1`` but
    # ``len("İ".lower()) == 2``), so an index taken on a lowercased copy
    # splits the original in the wrong place and silently truncates the
    # digest. ``_ascii_lower`` is a 1:1 character map, so it cannot move
    # an index.
    low = _ascii_lower(s)
    best_idx, best_algo = -1, ""
    for algo in ("sha256", "blake3"):
        idx = low.find(f"@{algo}:")
        if idx >= 0 and (best_idx < 0 or idx < best_idx):
            best_idx, best_algo = idx, algo
    if best_idx >= 0:
        hex_part = s[best_idx + len(f"@{best_algo}:"):].strip()
        if not hex_part:
            raise ValueError(f"tensorhub ref {best_algo} digest is empty")
        # th#1387: the hex tail runs to end-of-string, so a second marker
        # after it was being absorbed INTO the digest — a well-formed-looking
        # ref addressing a CAS object that does not exist.
        if any(c in hex_part for c in REF_GRAMMAR_SEPARATORS):
            raise ValueError(
                f"tensorhub ref {best_algo} digest {hex_part!r} contains a "
                "grammar separator")
        digest = f"{best_algo}:{hex_part.lower()}"
        s = s[:best_idx].strip()

    release = ""
    if "@" in s:
        # th#1987: a non-digest `@` tail is the RELEASE. th#1387's rule
        # survives the re-key: exactly one tail, carrying no separator.
        s, release_part = s.split("@", 1)
        s = s.strip()
        release = release_part.strip()
        if not release:
            raise ValueError("tensorhub ref release is empty")
        if any(c in release for c in REF_GRAMMAR_SEPARATORS):
            raise ValueError(
                f"tensorhub ref release {release!r} contains a grammar separator")
        # The catalog's `^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$` CHECK is enforced
        # where a release is CUT (migration 0044), not where a ref is parsed —
        # tensorhub's own parser applies no charset here, and duplicating it
        # would refuse refs the hub reads fine.

    # th#1987: the tag production is GONE. Refuse rather than let the colon
    # land inside the repo name, where it would address a repo nobody has.
    if ":" in s:
        raise RetiredTagRef(_retired_tag_message(raw.strip()))

    # th#1387: an unbounded split let "owner/repo/extra" parse with
    # repo="repo/extra" — a path separator inside the component that builds
    # roots/{org}/{repo}.json, whose inversion is then ambiguous.
    if "/" not in s:
        raise ValueError(
            "tensorhub ref must be 'owner/repo' (optionally with @<release>, "
            "?<lane-spec>, @sha256:<hex>, or @blake3:<hex>)")
    owner, repo = s.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        raise ValueError("tensorhub ref must be 'owner/repo'")
    for name, val in (("owner", owner), ("repo", repo)):
        if any(c in val for c in REF_GRAMMAR_SEPARATORS):
            raise ValueError(
                f"tensorhub ref {name} {val!r} contains a grammar separator")

    # th#2031 / pgw#1290: the fragment's ONE surviving meaning is the compile
    # cell key of a platform cell repo. Anywhere else it was the dead `#flavor`
    # selector, which parsed and was then DROPPED — resolving to whatever the
    # release's default variant is, with no error to see. Mirrors the Go twin's
    # placement: after owner/repo, because that is when the question is
    # answerable.
    if fragment and not (owner == "root" and repo.startswith("family-")
                         and len(repo) > len("family-")):
        raise RefFragmentRemoved(_fragment_removed_message(raw.strip()))

    return TensorhubRef(
        owner=owner, repo=repo, release=release, digest=digest,
        fragment=fragment, lane_spec=lane_spec)


def parse_model_ref(raw: str, *, provider: str = "tensorhub") -> ParsedModelRef:
    """Decode a model ref string into a typed payload.

    The wire-format contract carries provider as a separate field; this
    function consumes the bare ref string plus an explicit ``provider``
    keyword argument (default ``"tensorhub"``). No string prefixes are
    accepted — callers must split prefix/payload upstream and pass them
    in explicitly.

    Accepts either spelling of the huggingface provider — the short
    internal form ``"hf"`` and the pgw#511 wire form ``"huggingface"``
    (``ModelRef.source``, since pgw#523 deleted the ``.provider`` alias
    that used to narrow it before callers got here) — and always returns
    ``ParsedModelRef(provider="hf", ...)`` so every existing ``== "hf"``
    comparison downstream keeps working unchanged.
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty model ref")

    if provider in ("hf", "huggingface"):
        repo = s
        # pgw#1148: an HF ref has no `#` tail. The HF Hub never had a flavor
        # notion — the tail was the orchestrator's per-flavor routing
        # convention, deleted with the flavor as an address (§1.32(d)).
        # Refuse it typed rather than silently stripping it, so a caller who
        # still writes one is told, not quietly given a different repo.
        if "#" in repo:
            raise RefFragmentRemoved(_fragment_removed_message(raw))
        revision = None
        if "@" in repo:
            repo, revision = repo.split("@", 1)
            repo = repo.strip()
            revision = revision.strip() or None
        if "/" not in repo:
            raise ValueError("hf ref must be 'owner/repo'")
        return ParsedModelRef(
            provider="hf",
            hf=HuggingFaceRef(repo_id=repo, revision=revision),
        )

    if provider == "civitai":
        return ParsedModelRef(provider="civitai", civitai=CivitaiRef(model_id=s))

    if provider == "modelscope":
        # ModelScope repos are 'owner/repo' with an optional '@revision'. Like
        # HF there is no flavor; file selection (allow_patterns) is binding
        # metadata carried separately, not encoded in the ref string.
        repo = s
        revision = None
        if "@" in repo:
            repo, revision = repo.split("@", 1)
            repo = repo.strip()
            revision = revision.strip() or None
        if "/" not in repo:
            raise ValueError("modelscope ref must be 'owner/repo'")
        return ParsedModelRef(
            provider="modelscope",
            modelscope=ModelScopeRef(repo_id=repo, revision=revision),
        )

    if provider == "tensorhub":
        return ParsedModelRef(
            provider="tensorhub", tensorhub=_parse_tensorhub_ref(raw, s))

    raise ValueError(f"unsupported model ref provider: {provider!r}")


def format_model_ref(parsed: ParsedModelRef) -> WireRef:
    """THE formatter: normal-form string for a parsed ref (any provider)."""
    payload = parsed.tensorhub or parsed.hf or parsed.civitai or parsed.modelscope
    if payload is None:
        raise ValueError(f"parsed ref has no payload (provider={parsed.provider!r})")
    return payload.canonical()


def normalize_model_ref(raw: str, *, provider: str = "tensorhub") -> WireRef:
    """Project ``raw`` onto the normal form: ``format(parse(raw))``.

    Raises ``ValueError`` on grammar violations, exactly like
    :func:`parse_model_ref`.
    """
    return format_model_ref(parse_model_ref(raw, provider=provider))


def fold_ref(
    ref: str,
    *,
    release: str = "",
    provider: str = "tensorhub",
) -> WireRef:
    """Fold a side-channel ``release`` field into a ref string and return the
    normal form.

    An explicit non-empty release wins over one already embedded in ``ref``; an
    empty release preserves whatever the ref carries. Non-tensorhub providers
    have no release axis. pgw#1148 deleted the ``flavor=`` overlay with the
    flavor itself — there is no second selector to fold.

    The result is an ADDRESS, so a `?<lane-spec>` on the input is dropped
    (th#2006). Keeping it across the fold is what minted tensorhub's
    ``owner/repo@prod@sha256:…?quant=…`` — a double-``@`` ref, the shape
    th#1387 established destroys injectivity.
    """
    parsed = parse_model_ref(ref, provider=provider)
    release = (release or "").strip()
    if parsed.tensorhub is not None:
        th = parsed.tensorhub
        if release:
            th = TensorhubRef(
                owner=th.owner,
                repo=th.repo,
                release=release,
                digest=th.digest,
                fragment=th.fragment,
                lane_spec=th.lane_spec,
            )
        return th.canonical()
    return format_model_ref(parsed)
