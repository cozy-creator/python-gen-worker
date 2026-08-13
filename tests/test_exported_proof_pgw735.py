"""The exported lane proves adoption its OWN way (pgw#735).

`executor.py`'s adoption gates required an FX cache hit. That is the right proof
for a dynamo compiled graph — a hit means the delivered graph was reused rather than
silently re-traced — but an EXPORTED artifact performs no FX lookup at all, so
the gate could never pass and every `.pt2` adoption scored as a failure.

The fix must not be a synthesized hit counter: that would be a lie inside the one
path whose whole job is to detect lies about serving compiled. These tests pin
the honest predicate and the fail-closed cases.
"""

from __future__ import annotations

import pytest

from gen_worker import aot_serve, compiled_graph_key


class _Pipe:
    """A pipeline carrying only what the proof reads."""


def _dispatch(calls: int, failed: bool) -> aot_serve.CompiledGraphDispatch:
    """One armed compiled graph, or a de-armed one — the registry state `is_armed`
    and `execution_count` actually read."""
    runner = aot_serve.ArtifactRunner(
        package=None, contract=aot_serve.ArtifactContract(inputs=(), symbols={}),
        constants=(), module_name="unet", compiled_graph="unet/main")
    runner.calls = calls
    dispatch = aot_serve.CompiledGraphDispatch(declared=("unet/main",))
    dispatch.add("unet/main", runner)
    if failed:
        dispatch.remove("unet/main", "revoked")
    return dispatch


def _arm(pipe, *, calls: int, failed: bool = False) -> None:
    """The marker shape `arm_compiled_graph` ACTUALLY publishes.

    pgw#1176: this used to build a bare pipeline-level ``state``, a shape no
    production path has ever written — `arm_compiled_graph` writes ``targets`` and
    `wrap_module` writes the bare ``state`` on the MODULE. `_marker_states`
    carried a fallback for it whose own docstring said "the legacy
    single-``state`` shape tests use", i.e. a production branch kept alive by
    a fixture constructing something production cannot construct. The fallback
    is deleted; this builds the real thing.
    """
    setattr(pipe, aot_serve._MARKER_ATTR, {
        "meta": {"sku": "sm_89", "torch": "2.13", "precision": "bf16"},
        "targets": {"unet": {
            "module": None, "attr": "forward",
            "state": {
                "successful_calls": calls, "failed": failed,
                # pgw#1176: `is_armed` asks the REGISTRY what is armed rather
                # than reading a boolean, so a fixture that wants to model an
                # armed pipeline has to carry one. That is the point: there is
                # no longer a flag that can claim more than the pod serves.
                "runner": _dispatch(calls, failed),
            },
        }},
        "compiled_graphs": {"unet/main": {"key": "ek1-" + "0" * 56}},
    })


def test_proven_since_requires_new_successful_calls_and_no_revocation():
    pipe = _Pipe()
    assert not aot_serve.proven_since(pipe, 0)   # unarmed is never proven
    _arm(pipe, calls=0)
    assert not aot_serve.proven_since(pipe, 0)
    _arm(pipe, calls=3)
    assert aot_serve.proven_since(pipe, 0)
    assert aot_serve.proven_since(pipe, 2)
    # The delta matters, not the absolute count: a previous boot's calls
    # cannot prove THIS warmup exercised the artifact.
    _arm(pipe, calls=7)
    assert not aot_serve.proven_since(pipe, 7)
    assert not aot_serve.proven_since(pipe, 9)
    # An artifact that ran and then revoked (a B1/B2 refusal) has proven
    # nothing — fail closed, exactly like a dynamo compiled graph with zero hits.
    _arm(pipe, calls=5, failed=True)
    assert aot_serve.execution_count(pipe) == 5
    assert not aot_serve.proven_since(pipe, 0)


def test_exported_kind_compiled_graph_key_refusals_are_named():
    """pgw#1059: only exported (`aot-inductor`) compiled graphs are keyed, and every
    refusal names the missing fact instead of failing opaquely."""
    for kind in ("torch-inductor-cache", "an-unknown-kind", ""):
        with pytest.raises(compiled_graph_key.CompiledGraphKeyError) as unknown:
            compiled_graph_key.from_compiled_graph_metadata({"kind": kind})
        assert "no compiled-graph-key identity" in str(unknown.value)

    with pytest.raises(compiled_graph_key.CompiledGraphKeyError) as no_sm:
        compiled_graph_key.from_compiled_graph_metadata({"kind": "aot-inductor"})
    assert "sm" in str(no_sm.value)

    with pytest.raises(compiled_graph_key.CompiledGraphKeyError) as no_compiled_graph:
        compiled_graph_key.from_compiled_graph_metadata({"kind": "aot-inductor", "sm": "sm_89"})
    assert "compiled_graph" in str(no_compiled_graph.value)
