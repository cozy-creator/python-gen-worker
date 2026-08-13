"""THE ref grammar module (gw#492): parse + format + fold, nothing else mints.

Normal form — the ONE canonical string for a model ref (grammar th#597 C5,
shared vectors ``tests/testdata/ref_grammar_vectors.json``):

    tensorhub:  owner/repo[:tag][@sha256:<hex>|@blake3:<hex>][#<cell-fragment>]
    hf:         owner/repo[@revision]

The tag is ELIDED when it equals ``prod`` (the grammar default) and stamped
verbatim otherwise — including ``latest``. th#1276: ``prod`` is the STABLE
SERVING pointer and is the grammar default; ``latest`` is the MOVING PUBLISH
pointer that the finalize path auto-binds, so it must always be written
explicitly. ``format(parse(s))`` is the normalization projection;
``parse(format(v)) == v`` for every value. Every ref string the worker mints
(wire, residency keys, cache keys, telemetry) MUST come from :func:`wire_ref`
(bindings), :func:`fold_ref` (string + tag overlay), or
:func:`format_model_ref` / ``.canonical()`` (parsed values). A grep-guard test
(``tests/test_ref_normal_form.py``) rejects new ad-hoc formatter/parser sites.

THE FLAVOR IS DEAD AS A WEIGHT ADDRESS (§1.32(d), th#1803, pgw#1148). The
`#` tail no longer selects a checkpoint: selection within a tag group is
tensor-layout contract compatibility (§1.33, ``Slot(layouts=…)``), and a
specific checkpoint is addressed by ``owner/repo@sha256:…``. A weight ref
carrying `#` is refused by :func:`refuse_flavor_selector` — the client-side
twin of the hub's ``flavor_selection_removed`` 400.

The `#` tail SURVIVES IN THE GRAMMAR for one reason only: COMPILE CELL refs
are `#`-shaped (``root/family-<f>:cells#ck1-…``), exactly as tensorhub's own
``release.ParseCanonicalRef`` still parses them. That retirement is a
separate item on both sides; ``TensorhubRef.flavor`` is that cell fragment
and has no other reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType, Optional

from gen_worker.refgrammar import MAX_FRAGMENT_LEN as _MAX_FRAGMENT_LEN

# th#597 C5: `#` fragment charset [a-z0-9][a-z0-9._-]*, bounded by
# MAX_FRAGMENT_LEN (matches tensorhub's validation.IsValidFlavorToken). Cell
# fragments only — see the module docstring.
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


class FlavorSelectorRemoved(ValueError):
    """A weight ref carried a `#` selector. §1.32(d)/th#1803: THE FLAVOR
    SYSTEM IS DEAD — deleted, not aliased.

    The client-side twin of the hub's ``flavor_selection_removed`` /
    ``binding_flavor_removed`` 400: the SDK refuses at the boundary rather
    than minting a ref the hub will reject, so the caller is told what to
    write instead of reading a server error about a selector they thought
    was supported.
    """


def _flavor_removed_message(ref: str) -> str:
    return (
        f"model ref {ref!r} carries a `#` flavor selector, which is REMOVED "
        "(§1.32(d), th#1803 / pgw#1148 — deleted, not aliased). Selection "
        "within a tag group is tensor-layout contract compatibility: declare "
        "what the code accepts with Slot(layouts=...) and bind "
        "'owner/repo[:tag]', or address one exact checkpoint with "
        "'owner/repo@sha256:<hex>'."
    )


def refuse_flavor_selector(ref: str, *, where: str = "") -> None:
    """Refuse a WEIGHT ref that carries a `#` selector (§1.32(d)).

    THE weight-path chokepoint. Cell refs do not pass through here — the
    compile cache parses its own `#`-shaped keys, which is the only reason
    the tail survives in the grammar at all.
    """
    s = (ref or "").strip()
    if "#" not in s:
        return
    msg = _flavor_removed_message(s)
    raise FlavorSelectorRemoved(f"{where}: {msg}" if where else msg)

# th#1276: the ref grammar's default tag — a bare ``owner/repo`` means
# ``owner/repo:prod``. ``prod`` is the STABLE SERVING pointer; ``latest`` is
# the MOVING PUBLISH pointer that finalize auto-binds and must always be
# written explicitly. This is the Python twin of tensorhub's
# ``release.DefaultRefTag``. Use it at every GRAMMAR-COUPLED site (parse
# default, normal-form elision) so those sites stay greppable and distinct
# from code that genuinely means the ``latest`` publish tag.
DEFAULT_REF_TAG = "prod"

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
    tag: str = DEFAULT_REF_TAG
    digest: Optional[str] = None  # snapshot digest, including algorithm prefix (e.g. "blake3:<hex>")
    #: The `#` tail. A COMPILE CELL fragment (``root/family-<f>:cells#ck1-…``)
    #: and nothing else — never a weight selector (§1.32(d)). Weight paths
    #: call :func:`refuse_flavor_selector`; the compile cache is the one
    #: reader (``compile_cache.parse_cell_ref``).
    flavor: Optional[str] = None

    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"

    def canonical(self) -> "WireRef":
        """Normal form: ``owner/repo[:tag][@digest][#cell-fragment]``; the tag is
        elided when it is ``prod`` (the grammar default) and stamped verbatim
        otherwise, including ``latest``. Tensorhub is the default provider so
        no prefix is emitted; consumers track provider separately."""
        out = self.repo_id()
        if self.tag and self.tag != DEFAULT_REF_TAG:
            out = f"{out}:{self.tag}"
        if self.digest:
            out = f"{out}@{self.digest}"
        if self.flavor:
            out = f"{out}#{self.flavor}"
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
            raise FlavorSelectorRemoved(_flavor_removed_message(raw))
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
        digest = None
        flavor = None
        # th#1276: the grammar default is the STABLE SERVING pointer "prod";
        # the moving publish pointer "latest" must be written explicitly.
        tag = DEFAULT_REF_TAG

        if "#" in s:
            s, flavor_part = s.split("#", 1)
            flavor_part = flavor_part.strip()
            if "?" in flavor_part:
                flavor_part = flavor_part.split("?", 1)[0].strip()
            flavor_part = flavor_part.lower()
            if not flavor_part:
                raise ValueError("tensorhub ref fragment is empty")
            # th#597 C5: ONE fragment token per ref, charset
            # [a-z0-9][a-z0-9._-]{0,63} — `#a#b` is invalid (cells encode
            # conjunction inside one token). Shared grammar vectors:
            # tests/testdata/ref_grammar_vectors.json (byte-identical copy in
            # tensorhub internal/orchestrator/release/testdata/, whose Go
            # ParseCanonicalRef still parses the tail for the same reason).
            if not _TENSORHUB_FRAGMENT_RE.fullmatch(flavor_part):
                raise ValueError(
                    f"tensorhub ref fragment {flavor_part!r} is not a valid token"
                )
            flavor = flavor_part

        # pgw#872: the index MUST come from the same string it slices.
        # ``str.lower()`` is not length-preserving (``len("İ") == 1`` but
        # ``len("İ".lower()) == 2``), so an index taken on a lowercased copy
        # splits the original in the wrong place and silently truncates the
        # digest. ``_ascii_lower`` is a 1:1 character map, so it cannot move
        # an index.
        low = _ascii_lower(s)
        if "@sha256:" in low:
            idx = low.index("@sha256:")
            repo_id = s[:idx].strip()
            dig = s[idx + len("@sha256:"):].strip()
            if not dig:
                raise ValueError("tensorhub ref sha256 digest is empty")
            digest = f"sha256:{dig}"
            s = repo_id
        elif "@blake3:" in low:
            idx = low.index("@blake3:")
            repo_id = s[:idx].strip()
            dig = s[idx + len("@blake3:"):].strip()
            if not dig:
                raise ValueError("tensorhub ref blake3 digest is empty")
            digest = f"blake3:{dig}"
            s = repo_id
        elif "@" in s:
            raise ValueError("tensorhub ref digest must use @sha256:<hex> or @blake3:<hex>")
        if ":" in s:
            repo_id, tag_part = s.rsplit(":", 1)
            repo_id = repo_id.strip()
            tag = tag_part.strip() or DEFAULT_REF_TAG
        else:
            repo_id = s
        if "/" not in repo_id:
            raise ValueError("tensorhub ref must be 'owner/repo' (optionally with :tag, #flavor, @sha256:<hex>, or @blake3:<hex>)")
        owner, repo = repo_id.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        if not owner or not repo:
            raise ValueError("tensorhub ref must be 'owner/repo'")
        return ParsedModelRef(provider="tensorhub", tensorhub=TensorhubRef(owner=owner, repo=repo, tag=tag, digest=digest, flavor=flavor))

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
    tag: str = "",
    provider: str = "tensorhub",
) -> WireRef:
    """Fold a side-channel ``tag`` field into a ref string and return the
    normal form — the grammar-correct Python twin of tensorhub's
    ``release.ModelRefWithTag``.

    An explicit non-empty tag wins over one already embedded in ``ref``; an
    empty tag preserves whatever the ref carries. Non-tensorhub providers
    have no tag axis. pgw#1148 deleted the ``flavor=`` overlay with the
    flavor itself — there is no second selector to fold.
    """
    parsed = parse_model_ref(ref, provider=provider)
    tag = (tag or "").strip()
    if parsed.tensorhub is not None:
        th = parsed.tensorhub
        if tag:
            th = TensorhubRef(
                owner=th.owner,
                repo=th.repo,
                tag=tag,
                digest=th.digest,
                flavor=th.flavor,
            )
        return th.canonical()
    return format_model_ref(parsed)

