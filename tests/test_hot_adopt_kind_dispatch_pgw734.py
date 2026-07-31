"""Hot adoption dispatches on artifact KIND (pgw#734).

Boot arming already routes by kind (`provision.enable_compiled`). Hot adoption —
a cell delivered to a RUNNING worker — did not: every non-TRT ref went to the
dynamo stager, so an exported `.pt2` was unpacked as an inductor cache tree and
died as `artifact_invalid`. These tests pin that each kind reaches its own
backend, and that the dynamo path is untouched.

The tests drive the real `handle_model_op` adoption path with the existing
executor harness; only the backend calls are observed.
"""

from __future__ import annotations

from pathlib import Path


from gen_worker import aot_serve
from gen_worker import compile_cache as cc
from gen_worker.pb import worker_scheduler_pb2 as pb

from test_executor_adopt import (  # noqa: E402  (shared harness)
    FAMILY,
    _adopt,
    _artifact,
    _events,
    _spec,
    _wire_executor,
)

AOT_REF = f"root/family-{FAMILY}#" + aot_serve.flavor_label(
    "l4", "2.13.0+cu130", "w8a8")


def _seen(monkeypatch, module, name):
    calls: list = []
    original = getattr(module, name)

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, _spy)
    return calls


def test_exported_ref_is_admitted_and_routed_to_the_exported_backend(
    tmp_path, monkeypatch,
):
    """The two halves of the bug: an exported ref used to be refused as
    `bad_ref` before arming, and anything that got past went to the dynamo
    stager."""
    _artifact(tmp_path)
    ex, sent = _wire_executor(_spec(), tmp_path)

    staged = _seen(monkeypatch, cc, "stage_artifact")
    loaded: list = []

    def _fake_load(pipeline, cfg, artifact, cache_dir=None):
        loaded.append(Path(artifact))
        raise cc.AdoptError("constants_unresolved", "no resident weights here")

    monkeypatch.setattr(aot_serve, "load_and_wrap", _fake_load)
    _adopt(ex, ref=AOT_REF)

    # Routed to the exported backend...
    assert loaded, "an exported ref never reached aot_serve.load_and_wrap"
    # ...and never to the dynamo stager, which would unpack a .pt2 as a tar
    # cache tree.
    assert not staged, "an exported artifact was handed to the dynamo stager"
    # Fail-closed with the backend's own classified reason, not `bad_ref`.
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and "constants_unresolved" in failed[0].error
    assert "bad_ref" not in failed[0].error


def test_unknown_kind_is_still_refused(tmp_path):
    """Admitting a third kind must not admit a fourth by accident."""
    _artifact(tmp_path)
    ex, sent = _wire_executor(_spec(), tmp_path)
    _adopt(ex, ref=f"root/family-{FAMILY}#not-a-real-kind")
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and "bad_ref" in failed[0].error
