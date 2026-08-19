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


def derives_its_release(ref: str) -> bool:
    """Whether a publish into ``ref`` gets its release DERIVED by the hub.

    th#2202: a scratch repo cuts its own release on every publish, because
    th#1987's "name the release" is an act of DELIBERATION and nobody authors
    ``_job-<request-id>`` — the hub names it, hard-privates it, TTLs it and
    reaps it. So an empty ``release`` is LEGAL here and only here, and the SDK
    must not refuse it: this precondition used to fire client-side, before any
    HTTP, and killed a training run at its first checkpoint on a rented A100.

    Accepts either ``owner/name`` or a bare name; selectors are ignored.
    """
    text = str(ref or "").strip().lower()
    for sep in (":", "@", "#"):
        text = text.split(sep, 1)[0]
    return is_scratch_name(text.rsplit("/", 1)[-1])


__all__ = ["PREFIX", "is_scratch_name", "derives_its_release"]
