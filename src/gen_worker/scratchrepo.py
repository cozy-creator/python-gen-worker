"""Scratch-repo naming vocabulary — the SDK twin of the hub's
``internal/scratchrepo`` (th#733, generalized to every repo-writing producer by
th#1901/th#2068).

A publishing run never writes the destination the submitter named: the hub
rewrites it on the wire to the per-run scratch repo ``<org>/_job-<job-id>``.
The leading underscore is deliberately OUTSIDE the public repo-name grammar, so
a user-supplied ref can never name or squat a scratch repo. Every SDK validator
that sees a hub-authored destination must admit this ONE prefix — and nothing
else that starts with ``_``.
"""

#: The reserved repo-name prefix. Mirrors ``scratchrepo.Prefix``.
PREFIX = "_job-"


def is_scratch_name(name: str) -> bool:
    """Whether a repo NAME (the half after the ``/``) is scratch-reserved."""
    return str(name or "").strip().lower().startswith(PREFIX)


__all__ = ["PREFIX", "is_scratch_name"]
