"""th#1084: INPUT-caused refusals map INVALID (never FATAL) so the hub fails
only the request and never feeds release-health blame."""

import pytest

from gen_worker.api.errors import ValidationError
from gen_worker.convert.classifier import RepoRefusal, SourceIncludeError, apply_source_include
from gen_worker.convert.ingest import _raise_input_refusal
from gen_worker.executor import _map_exception
from gen_worker.pb import worker_scheduler_pb2 as pb


def test_repo_refusal_maps_invalid():
    status, msg = _map_exception(RepoRefusal("missing_safetensors", files_seen=["a.bin"]))
    assert status == pb.JOB_STATUS_INVALID
    assert "missing_safetensors" in msg


def test_hf_access_errors_map_invalid():
    import requests
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    response = requests.Response()
    response.status_code = 404
    for exc_cls in (RepositoryNotFoundError, GatedRepoError):
        with pytest.raises(ValidationError) as ei:
            _raise_input_refusal(exc_cls(
                "404 Client Error for url https://huggingface.co/api/models/x",
                response=response))
        status, msg = _map_exception(ei.value)
        assert status == pb.JOB_STATUS_INVALID
        assert msg.startswith(exc_cls.__name__)


def test_source_include_typo_maps_invalid():
    with pytest.raises(SourceIncludeError) as ei:
        apply_source_include(["model.safetensors"], ["nonexistent-*.safetensors"])
    status, _ = _map_exception(ei.value)
    assert status == pb.JOB_STATUS_INVALID
