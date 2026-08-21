"""Scratch-repo naming vocabulary — the SDK twin of the hub's internal/scratchrepo. A publishing run never writes the destination the submitter named: the hub rewrites it on the wire to the per-run scratch repo <org>/_job-<job-id>. The leading underscore is deliberately OUTSIDE the public repo-name grammar, so a user-supplied ref can never name or squat a scratch repo — every SDK validator that sees a hub-authored destination must admit this ONE prefix, and nothing else that starts with "_"."""

PREFIX = "_job-"


def is_scratch_name(name: str) -> bool:
    """Whether a repo NAME (the half after the ``/``) is scratch-reserved."""
    return str(name or "").strip().lower().startswith(PREFIX)


def derives_its_release(ref: str) -> bool:
    """Whether a publish into ``ref`` gets its release DERIVED by the hub."""
    text = str(ref or "").strip().lower()
    for sep in (":", "@", "#"):
        text = text.split(sep, 1)[0]
    return is_scratch_name(text.rsplit("/", 1)[-1])


__all__ = ["PREFIX", "is_scratch_name", "derives_its_release"]
