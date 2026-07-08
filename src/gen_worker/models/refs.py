from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# th#597 C5: flavor charset [a-z0-9][a-z0-9._-]{0,63} (matches tensorhub's
# validation.IsValidFlavorToken).
_TENSORHUB_FLAVOR_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


@dataclass(frozen=True)
class TensorhubRef:
    owner: str
    repo: str
    tag: str = "latest"
    digest: Optional[str] = None  # snapshot digest, including algorithm prefix (e.g. "blake3:<hex>")
    flavor: Optional[str] = None

    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"

    def canonical(self) -> str:
        """Canonical bare-ref form. Tensorhub is the default provider so
        no prefix is emitted. Consumers track provider separately."""
        flavor = f"#{self.flavor}" if self.flavor else ""
        if self.digest:
            return f"{self.repo_id()}@{self.digest}{flavor}"
        return f"{self.repo_id()}:{self.tag}{flavor}"


@dataclass(frozen=True)
class HuggingFaceRef:
    repo_id: str
    revision: Optional[str] = None
    flavor: Optional[str] = None

    def canonical(self) -> str:
        """Canonical bare-ref form. Provider is tracked separately.

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
            return f"{base}#{self.flavor}"
        return base


@dataclass(frozen=True)
class CivitaiRef:
    model_id: str

    def canonical(self) -> str:
        return self.model_id


@dataclass(frozen=True)
class ModelScopeRef:
    repo_id: str
    revision: Optional[str] = None

    def canonical(self) -> str:
        if self.revision:
            return f"{self.repo_id}@{self.revision}"
        return self.repo_id


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
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty model ref")

    if provider == "hf":
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
        tag = "latest"

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

        low = s.lower()
        if "@sha256:" in low:
            idx = low.index("@sha256:")
            repo_id = s[:idx].strip()
            dig = s[idx + len("@sha256:"):].strip()
            if not dig:
                raise ValueError("tensorhub ref sha256 digest is empty")
            digest = f"sha256:{dig}"
            s = repo_id
            low = s.lower()
        elif "@blake3:" in low:
            idx = low.index("@blake3:")
            repo_id = s[:idx].strip()
            dig = s[idx + len("@blake3:"):].strip()
            if not dig:
                raise ValueError("tensorhub ref blake3 digest is empty")
            digest = f"blake3:{dig}"
            s = repo_id
            low = s.lower()
        elif "@" in s:
            raise ValueError("tensorhub ref digest must use @sha256:<hex> or @blake3:<hex>")
        if ":" in s:
            repo_id, tag_part = s.rsplit(":", 1)
            repo_id = repo_id.strip()
            tag = tag_part.strip() or "latest"
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


def canonical_id(ref: str, provider: str = "tensorhub", *, tag: str = "prod", flavor: str = "") -> str:
    """Canonical in-process identity for a (provider, ref, tag, flavor) tuple.

    Used as the key in cross-process maps like ``resolved_repos_by_id``.
    Tensorhub (default provider) refs render as bare. Non-tensorhub
    providers render with a single-colon prefix so two providers' refs
    can coexist in the same map without collision.
    """
    base = ref.strip()
    if not base:
        return ""
    out = base
    if tag and tag != "prod":
        out = f"{out}:{tag}"
    if flavor:
        out = f"{out}#{flavor}"
    if provider and provider != "tensorhub":
        out = f"{provider}:{out}"
    return out
