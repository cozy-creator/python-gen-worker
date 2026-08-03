"""THE ref grammar module (gw#492): parse + format + fold, nothing else mints.

Normal form — the ONE canonical string for a model ref (grammar th#597 C5,
shared vectors ``tests/testdata/ref_grammar_vectors.json``):

    tensorhub:  owner/repo[:tag][@sha256:<hex>|@blake3:<hex>][#flavor]
    hf:         owner/repo[@revision][#flavor]

The tag is ELIDED when it equals ``prod`` (the grammar default) and stamped
verbatim otherwise — including ``latest``. th#1276: ``prod`` is the STABLE
SERVING pointer and is the grammar default; ``latest`` is the MOVING PUBLISH
pointer that the finalize path auto-binds, so it must always be written
explicitly. ``format(parse(s))`` is the normalization projection;
``parse(format(v)) == v`` for every value. Every ref string the worker mints
(wire, residency keys, cache keys, telemetry) MUST come from :func:`wire_ref`
(bindings), :func:`fold_ref` (string + tag/flavor overlay), or
:func:`format_model_ref` / ``.canonical()`` (parsed values). A grep-guard test
(``tests/test_ref_normal_form.py``) rejects new ad-hoc formatter/parser sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType, Optional

# th#597 C5: flavor charset [a-z0-9][a-z0-9._-]{0,63} (matches tensorhub's
# validation.IsValidFlavorToken).
_TENSORHUB_FLAVOR_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

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
    flavor: Optional[str] = None

    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"

    def canonical(self) -> "WireRef":
        """Normal form: ``owner/repo[:tag][@digest][#flavor]``; the tag is
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
    flavor: Optional[str] = None

    def canonical(self) -> "WireRef":
        """Normal form ``owner/repo[@revision][#flavor]``. Provider is
        tracked separately.

        Flavor is a binding-metadata side channel that the worker uses to
        pick a weight-precision subset out of an HF repo. The orchestrator
        carries the flavor folded into the ref as ``owner/repo#flavor`` in
        its routing maps (RequiredRepoRefs, etc.). Keep the same form in
        canonical() so disk_models / vram_models advertisements use the
        same identity the orchestrator's cache-locality scorer expects.
        Two flavors of the same repo therefore get two distinct cache
        entries — matching how the orchestrator routes per-flavor refs.
        """
        base = self.repo_id
        if self.revision:
            base = f"{base}@{self.revision}"
        if self.flavor:
            base = f"{base}#{self.flavor}"
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
        # Issue #17: runtime wire format carries flavor in the ref string
        # (e.g. "owner/repo#bf16") to identify which variant of an HF
        # binding the orchestrator is referring to. The HF Hub itself has
        # no notion of flavor — strip the `#flavor` tail before parsing
        # so `huggingface_hub.snapshot_download` sees a valid repo_id.
        flavor: Optional[str] = None
        if "#" in repo:
            repo, flavor_part = repo.split("#", 1)
            repo = repo.strip()
            flavor_part = flavor_part.strip()
            flavor = flavor_part or None
        revision = None
        if "@" in repo:
            repo, revision = repo.split("@", 1)
            repo = repo.strip()
            revision = revision.strip() or None
        if "/" not in repo:
            raise ValueError("hf ref must be 'owner/repo'")
        return ParsedModelRef(
            provider="hf",
            hf=HuggingFaceRef(repo_id=repo, revision=revision, flavor=flavor),
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
                raise ValueError("tensorhub ref flavor is empty")
            # th#597 C5: ONE flavor token per ref, charset
            # [a-z0-9][a-z0-9._-]{0,63} — `#a#b` is invalid (cells encode
            # conjunction inside one token). Shared grammar vectors:
            # tests/testdata/ref_grammar_vectors.json (byte-identical copy in
            # tensorhub internal/orchestrator/release/testdata/).
            if not _TENSORHUB_FLAVOR_RE.fullmatch(flavor_part):
                raise ValueError(
                    f"tensorhub ref flavor {flavor_part!r} is not a valid flavor token"
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
    flavor: str = "",
    provider: str = "tensorhub",
) -> WireRef:
    """Fold side-channel ``tag``/``flavor`` fields into a ref string and
    return the normal form — the grammar-correct Python twin of tensorhub's
    ``release.ModelRefWithTagFlavor``.

    An explicit non-empty field wins over a tag/flavor already embedded in
    ``ref``; empty fields preserve whatever the ref carries. Non-tensorhub
    providers have no tag axis; flavor folds as ``#flavor`` (the orchestrator
    routing convention for HF refs).
    """
    parsed = parse_model_ref(ref, provider=provider)
    tag = (tag or "").strip()
    flavor = (flavor or "").strip().lower()
    if parsed.tensorhub is not None:
        th = parsed.tensorhub
        if tag or flavor:
            th = TensorhubRef(
                owner=th.owner,
                repo=th.repo,
                tag=tag or th.tag,
                digest=th.digest,
                flavor=flavor or th.flavor,
            )
        return th.canonical()
    if parsed.hf is not None and flavor:
        return HuggingFaceRef(
            repo_id=parsed.hf.repo_id,
            revision=parsed.hf.revision,
            flavor=flavor,
        ).canonical()
    return format_model_ref(parsed)


def flavor_token(v: str) -> str:
    """Wire-boundary flavor-token hygiene (gw#488): the internal dtype-axis
    colon forms (``gguf:q4_k_m``, ``int8:awq``) publish as ``-`` forms to fit
    the flavor charset ``[a-z0-9][a-z0-9._-]{0,63}``. The ONE implementation —
    do not inline ``.replace(":", "-")`` at call sites."""
    return str(v or "").replace(":", "-")

