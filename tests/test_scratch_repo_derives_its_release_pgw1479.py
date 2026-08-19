"""pgw#1479 / th#2202 — a SCRATCH destination DERIVES its release, so the SDK
must not refuse an empty one.

The bug, measured by the te#304 lane at step 200 of a real training run on a
$1.39/hr A100, after the pod, the image and 15.8 GB of base weights were paid
for::

    RuntimeError: save_checkpoint('checkpoints/lora_000000200.safetensors')
    cannot publish into tensorhub/_job-b102f721-...: the request named no
    `destination.release`, and th#1987 made it mandatory

That refusal fired CLIENT-SIDE, before any HTTP, and it was unsatisfiable:
``ctx._repo_job_release()`` reads ``execution_hints["destination_release"]``,
which the executor derives ONLY from ``payload.destination.release`` — a
reserved struct an endpoint whose typed input declares the scalar
``destination_repo`` with ``forbid_unknown_fields=True`` can never be given.
So the destination the hub itself hands EVERY publishing run (``th#1901``
rewrites every producer's destination to ``<org>/_job-<request-id>``) was the
one destination this SDK could not publish into.

th#2202 ruled it: the release stops being a payload fact for a repo nobody
authored. The hub cuts one per checkpoint. This SDK precondition is the mirror
of the hub's rule and may not be stricter than it.

RED-VERIFY (each independently):
  - drop ``not derives`` from HubClient.publish_v2's guard ->
    ``test_publish_v2_admits_an_empty_release_into_a_scratch_repo`` fails;
  - drop the scratch arm of ``destination_release`` ->
    ``test_destination_release_derives_for_a_scratch_destination`` fails;
  - make ``RequestContext._checkpoint_release`` raise on an empty release
    unconditionally -> ``test_checkpoint_release_derives_for_the_runs_own_
    scratch_repo`` fails with the verbatim step-200 message;
  - keep either guard unconditional -> the corresponding "still refuses"
    control keeps passing, so none of these is an always-green assertion.
"""

from __future__ import annotations

import pytest

from gen_worker import scratchrepo
from gen_worker.convert.publish import destination_release
from gen_worker.request_context import RequestContext
from gen_worker.hubio.client import (
    COMPILED_GRAPH_NO_RELEASE,
    CommitFile,
    HubClient,
    HubPublishError,
    HubReleaseRequiredError,
)


class _Ctx:
    """The reserved-struct half of a RequestContext, and nothing else."""

    def __init__(self, destination: dict | None = None) -> None:
        self.destination = destination or {}


# --------------------------------------------------------------------------
# The naming predicate — the one fact both halves key on.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ref,expected", [
    ("tensorhub/_job-b102f721-4d62-41c4-9df2-136f9a72df83", True),
    ("_job-b102f721", True),
    ("TENSORHUB/_JOB-B102F721", True),
    ("tensorhub/_job-b102f721@r1", True),
    ("tensorhub/te304-renewal-lora", False),
    ("tensorhub/flux2-klein-4b", False),
    # A repo merely CONTAINING the prefix is not one. The reserved grammar is
    # anchored, or a user repo could squat the derivation.
    ("tensorhub/not_job-b102f721", False),
    ("", False),
])
def test_derives_its_release_reads_the_reserved_grammar(ref: str, expected: bool) -> None:
    assert scratchrepo.derives_its_release(ref) is expected


# --------------------------------------------------------------------------
# HubClient.publish_v2 — the guard that fired before any HTTP.
# --------------------------------------------------------------------------

def _publish(destination_repo: str, release: str) -> None:
    """Drive publish_v2 far enough to clear (or trip) the release guard.

    The guard is the FIRST thing after the empty-files check, and the next step
    reaches the local CAS — so a by-reference file gives us a deterministic,
    network-free stop AFTER the guard. Reaching that second refusal is the
    proof the first one did not fire.
    """
    client = HubClient(base_url="http://hub.invalid", token="t")
    return client.publish_v2(
        destination_repo=destination_repo,
        files=[CommitFile(path="checkpoints/lora_000000200.safetensors", local_path=None)],
        release=release,
    )


def test_publish_v2_admits_an_empty_release_into_a_scratch_repo() -> None:
    with pytest.raises(HubPublishError) as caught:
        _publish("tensorhub/_job-b102f721-4d62-41c4-9df2-136f9a72df83", "")
    assert not isinstance(caught.value, HubReleaseRequiredError), (
        "the SDK refused a scratch publish with no release — this is the "
        "client-side raise that killed a training run at step 200 on an A100"
    )
    assert "by-reference" in str(caught.value)


def test_publish_v2_still_refuses_an_empty_release_into_a_repo_a_person_owns() -> None:
    """The control. th#1987 is untouched for a repo with an author."""
    with pytest.raises(HubReleaseRequiredError):
        _publish("tensorhub/te304-renewal-lora", "")


def test_publish_v2_still_admits_the_compiled_graph_exemption() -> None:
    with pytest.raises(HubPublishError) as caught:
        _publish("tensorhub/te304-renewal-lora", COMPILED_GRAPH_NO_RELEASE)
    assert not isinstance(caught.value, HubReleaseRequiredError)


# --------------------------------------------------------------------------
# convert.publish.destination_release — the FINAL publish of a training run.
# --------------------------------------------------------------------------

def test_destination_release_derives_for_a_scratch_destination() -> None:
    got = destination_release(_Ctx(), "", "tensorhub/_job-b102f721-4d62-41c4")
    assert got == "", "a scratch destination derives; the SDK states nothing"


def test_destination_release_still_refuses_an_authored_destination() -> None:
    with pytest.raises(ValueError, match="release is required"):
        destination_release(_Ctx(), "", "tensorhub/te304-renewal-lora")


def test_an_explicit_release_always_wins() -> None:
    assert destination_release(_Ctx(), "v1", "tensorhub/_job-b102f721") == "v1"


def test_the_reserved_struct_still_carries_a_release_when_the_caller_names_one() -> None:
    """The reserved object is NOT retired: a caller who can state a release
    still does, and it still reaches the publish."""
    ctx = _Ctx({"ref": "tensorhub/te304-renewal-lora", "release": "v1"})
    assert destination_release(ctx, "", "tensorhub/_job-b102f721") == "v1"


# --------------------------------------------------------------------------
# RequestContext._checkpoint_release — THE line that raised at step 200.
#
# It is a named method rather than an inline `if` for exactly this reason: the
# decision is now assertable at $0, off the production context, with no pod, no
# hub and no file. th#2199's lesson one layer down.
# --------------------------------------------------------------------------

def _ctx(release: str = "") -> RequestContext:
    hints = {"kind": "training", "destination_repo": "tensorhub/_job-b102f721"}
    if release:
        hints["destination_release"] = release
    return RequestContext("b102f721-4d62-41c4", job_id="j", execution_hints=hints, publishes=True)


def test_checkpoint_release_derives_for_the_runs_own_scratch_repo() -> None:
    assert _ctx()._checkpoint_release("_job-b102f721-4d62-41c4") == ""


def test_checkpoint_release_still_refuses_a_repo_with_an_author() -> None:
    with pytest.raises(RuntimeError, match="release_required"):
        _ctx()._checkpoint_release("te304-renewal-lora")


def test_a_named_release_is_carried_through_untouched() -> None:
    assert _ctx("v1")._checkpoint_release("te304-renewal-lora") == "v1"
    assert _ctx("v1")._checkpoint_release("_job-b102f721-4d62-41c4") == "v1"
