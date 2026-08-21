from __future__ import annotations

import pytest

from gen_worker import scratchrepo
from gen_worker.convert.publish import destination_release
from gen_worker.request_context import RequestContext
from gen_worker.hubio.client import (
    COMPILED_GRAPH_NO_RELEASE,
    CommitFile,
    CommitResult,
    HubClient,
    HubPublishError,
    HubReleaseRequiredError,
)


class _Ctx:

    def __init__(self, destination: dict | None = None) -> None:
        self.destination = destination or {}


@pytest.mark.parametrize("ref,expected", [
    ("tensorhub/_job-b102f721-4d62-41c4-9df2-136f9a72df83", True),
    ("_job-b102f721", True),
    ("TENSORHUB/_JOB-B102F721", True),
    ("tensorhub/_job-b102f721@r1", True),
    ("tensorhub/te304-renewal-lora", False),
    ("tensorhub/flux2-klein-4b", False),
    ("tensorhub/not_job-b102f721", False),
    ("", False),
])
def test_derives_its_release_reads_the_reserved_grammar(ref: str, expected: bool) -> None:
    assert scratchrepo.derives_its_release(ref) is expected


def _publish(destination_repo: str, release: str) -> CommitResult:
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
    """The control."""
    with pytest.raises(HubReleaseRequiredError):
        _publish("tensorhub/te304-renewal-lora", "")


def test_publish_v2_still_admits_the_compiled_graph_exemption() -> None:
    with pytest.raises(HubPublishError) as caught:
        _publish("tensorhub/te304-renewal-lora", COMPILED_GRAPH_NO_RELEASE)
    assert not isinstance(caught.value, HubReleaseRequiredError)


def test_destination_release_derives_for_a_scratch_destination() -> None:
    got = destination_release(_Ctx(), "", "tensorhub/_job-b102f721-4d62-41c4")
    assert got == "", "a scratch destination derives; the SDK states nothing"


def test_destination_release_still_refuses_an_authored_destination() -> None:
    with pytest.raises(ValueError, match="release is required"):
        destination_release(_Ctx(), "", "tensorhub/te304-renewal-lora")


def test_an_explicit_release_always_wins() -> None:
    assert destination_release(_Ctx(), "v1", "tensorhub/_job-b102f721") == "v1"


def test_the_reserved_struct_still_carries_a_release_when_the_caller_names_one() -> None:
    ctx = _Ctx({"ref": "tensorhub/te304-renewal-lora", "release": "v1"})
    assert destination_release(ctx, "", "tensorhub/_job-b102f721") == "v1"


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
