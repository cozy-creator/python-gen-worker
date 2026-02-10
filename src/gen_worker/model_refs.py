from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CozyRef:
    owner: str
    repo: str
    tag: str = "latest"
    digest: Optional[str] = None  # snapshot digest, including algorithm prefix (e.g. "blake3:<hex>")

    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"

    def canonical(self) -> str:
        if self.digest:
            return f"cozy:{self.repo_id()}@{self.digest}"
        return f"cozy:{self.repo_id()}:{self.tag}"


@dataclass(frozen=True)
class HuggingFaceRef:
    repo_id: str
    revision: Optional[str] = None

    def canonical(self) -> str:
        if self.revision:
            return f"hf:{self.repo_id}@{self.revision}"
        return f"hf:{self.repo_id}"


@dataclass(frozen=True)
class ParsedModelRef:
    scheme: str  # "cozy" | "hf"
    cozy: Optional[CozyRef] = None
    hf: Optional[HuggingFaceRef] = None


def _strip_scheme(raw: str) -> tuple[Optional[str], str]:
    s = (raw or "").strip()
    if ":" not in s:
        return None, s
    scheme, rest = s.split(":", 1)
    scheme = scheme.strip().lower()
    if scheme in ("cozy", "hf"):
        return scheme, rest.strip()
    return None, s


def parse_model_ref(raw: str) -> ParsedModelRef:
    """
    Parse a model ref string.

    Phase 1 supported schemes:
      - Cozy Hub (default): "owner/repo", "owner/repo:tag", "owner/repo@sha256:<hex>",
        optionally prefixed with "cozy:".
      - Hugging Face: "hf:owner/repo" or "hf:owner/repo@revision".

    Returns:
        ParsedModelRef with scheme + typed payload.
    """
    scheme, rest = _strip_scheme(raw)
    if scheme is None:
        scheme = "cozy"

    if scheme == "hf":
        repo = rest.strip()
        # Be tolerant of accidentally double-prefixed refs like:
        #   hf:hf:org/repo[@rev]
        # This can happen when a canonical "hf:..." string is wrapped again by a caller.
        if repo.lower().startswith("hf:"):
            repo = repo.split(":", 1)[1].strip()
        if not repo:
            raise ValueError("empty hf model ref")
        revision = None
        if "@" in repo:
            repo, revision = repo.split("@", 1)
            repo = repo.strip()
            revision = revision.strip() or None
        if "/" not in repo:
            raise ValueError("hf ref must be 'owner/repo'")
        return ParsedModelRef(scheme="hf", hf=HuggingFaceRef(repo_id=repo, revision=revision))

    if scheme == "cozy":
        s = rest.strip()
        if not s:
            raise ValueError("empty cozy model ref")
        digest = None
        tag = "latest"
        low = s.lower()
        if "@sha256:" in low:
            idx = low.index("@sha256:")
            repo_id = s[:idx].strip()
            dig = s[idx + len("@sha256:"):].strip()
            if not dig:
                raise ValueError("cozy ref sha256 digest is empty")
            digest = f"sha256:{dig}"
            s = repo_id
            low = s.lower()
        elif "@blake3:" in low:
            idx = low.index("@blake3:")
            repo_id = s[:idx].strip()
            dig = s[idx + len("@blake3:"):].strip()
            if not dig:
                raise ValueError("cozy ref blake3 digest is empty")
            digest = f"blake3:{dig}"
            s = repo_id
            low = s.lower()
        elif "@" in s:
            # Reserve @ for future; require explicit algorithm prefix.
            raise ValueError("cozy ref digest must use @sha256:<hex> or @blake3:<hex>")
        if ":" in s:
            repo_id, tag_part = s.rsplit(":", 1)
            repo_id = repo_id.strip()
            tag = tag_part.strip() or "latest"
        else:
            repo_id = s
        if "/" not in repo_id:
            raise ValueError("cozy ref must be 'owner/repo' (optionally with :tag or @sha256:<hex> or @blake3:<hex>)")
        owner, repo = repo_id.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        if not owner or not repo:
            raise ValueError("cozy ref must be 'owner/repo'")
        return ParsedModelRef(scheme="cozy", cozy=CozyRef(owner=owner, repo=repo, tag=tag, digest=digest))

    raise ValueError(f"unsupported model ref scheme: {scheme!r}")
