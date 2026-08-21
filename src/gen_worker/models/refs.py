"""THE ref grammar module: parse + format + fold; nothing else mints. Normal form (shared vectors tests/testdata/ref_grammar_vectors.json): tensorhub `owner/repo[@<release>|@sha256:<hex>|@blake3:<hex>][?<lane-spec>][#<compiled-graph-fragment>]`; hf `owner/repo[@revision]`. The `:tag` production is DEAD — no default, nothing elided; a bare owner/repo names a repo and no artifact. `format(parse(s))` is the normalization projection and `parse(format(v)) == v`; a digest WINS over a release when both are set, and they share one `@` slot (a second destroys injectivity). The `?<lane-spec>` tail is a RESOLUTION INPUT, never an address — canonical() drops it (the twin of Go's Address().String(), not String()). The `#` fragment has exactly one meaning — the compile fragment of a platform compiled-graph repo — and parse REFUSES it on every other repo. Every ref string the worker mints MUST come from wire_ref, fold_ref, or format_model_ref/.canonical()."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType, Optional

from gen_worker.refgrammar import MAX_FRAGMENT_LEN as _MAX_FRAGMENT_LEN

MAX_FRAGMENT_LEN = _MAX_FRAGMENT_LEN
_TENSORHUB_FRAGMENT_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]{0,%d}" % (MAX_FRAGMENT_LEN - 1)
)

REF_GRAMMAR_SEPARATORS = "/@:#"


class RefFragmentRemoved(ValueError):
    """A ref carried a `#` fragment somewhere it means nothing."""


class RetiredTagRef(ValueError):
    """A ref carried a `:tag`."""


def _fragment_removed_message(ref: str) -> str:
    return (
        f"model ref {ref!r} carries a `#` fragment, which names a COMPILE COMPILED GRAPH "
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
    """Refuse a WEIGHT ref that carries a `#` fragment, naming the site."""
    s = (ref or "").strip()
    if "#" not in s:
        return
    msg = _fragment_removed_message(s)
    raise RefFragmentRemoved(f"{where}: {msg}" if where else msg)


_ASCII_LOWER = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)


def _ascii_lower(s: str) -> str:
    return s.translate(_ASCII_LOWER)

WireRef = NewType("WireRef", str)


@dataclass(frozen=True)
class TensorhubRef:
    owner: str
    repo: str
    release: str = ""
    digest: Optional[str] = None
    fragment: Optional[str] = None
    lane_spec: str = ""

    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"

    def canonical(self) -> "WireRef":
        """Normal form: ``owner/repo[@release|@digest][#compiled graph-fragment]``."""
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
        """Normal form ``owner/repo[@revision]``."""
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
    """Decoded model ref."""

    provider: str
    tensorhub: Optional[TensorhubRef] = None
    hf: Optional[HuggingFaceRef] = None
    civitai: Optional[CivitaiRef] = None
    modelscope: Optional[ModelScopeRef] = None


def _parse_tensorhub_ref(raw: str, s: str) -> TensorhubRef:
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
        if not _TENSORHUB_FRAGMENT_RE.fullmatch(fragment_part):
            raise ValueError(
                f"tensorhub ref fragment {fragment_part!r} is not a valid token"
            )
        fragment = fragment_part
        s = s.strip()

    if "?" in s:
        s, spec_part = s.split("?", 1)
        s = s.strip()
        lane_spec = spec_part.strip()
        if not lane_spec:
            raise ValueError(
                "tensorhub ref lane spec is empty; omit the '?' entirely to "
                "mean any variant")

    # Take the EARLIEST digest marker in the string, never the first algorithm in a fixed list (scanning by algorithm split "@blake3:x@sha256:<hex>" on the LATER marker). And the index MUST come from the same string it slices: str.lower() is not length-preserving (len("İ".lower()) == 2), so an index taken on a lowercased copy splits the original in the wrong place and silently truncates the digest — _ascii_lower is a 1:1 character map and cannot move an index.
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
        if any(c in hex_part for c in REF_GRAMMAR_SEPARATORS):
            raise ValueError(
                f"tensorhub ref {best_algo} digest {hex_part!r} contains a "
                "grammar separator")
        digest = f"{best_algo}:{hex_part.lower()}"
        s = s[:best_idx].strip()

    release = ""
    if "@" in s:
        s, release_part = s.split("@", 1)
        s = s.strip()
        release = release_part.strip()
        if not release:
            raise ValueError("tensorhub ref release is empty")
        if any(c in release for c in REF_GRAMMAR_SEPARATORS):
            raise ValueError(
                f"tensorhub ref release {release!r} contains a grammar separator")

    if ":" in s:
        raise RetiredTagRef(_retired_tag_message(raw.strip()))

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

    if fragment and not (owner == "root" and repo.startswith("family-")
                         and len(repo) > len("family-")):
        raise RefFragmentRemoved(_fragment_removed_message(raw.strip()))

    return TensorhubRef(
        owner=owner, repo=repo, release=release, digest=digest,
        fragment=fragment, lane_spec=lane_spec)


def parse_model_ref(raw: str, *, provider: str = "tensorhub") -> ParsedModelRef:
    """Decode a model ref string into a typed payload."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty model ref")

    if provider in ("hf", "huggingface"):
        repo = s
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
    """Project ``raw`` onto the normal form: ``format(parse(raw))``."""
    return format_model_ref(parse_model_ref(raw, provider=provider))


def fold_ref(
    ref: str,
    *,
    release: str = "",
    provider: str = "tensorhub",
) -> WireRef:
    """Fold a side-channel ``release`` field into a ref string and return the normal form."""
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
