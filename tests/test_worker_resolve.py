"""Worker model-id resolution for the new binding model.

`Worker._resolve_model_id_for_injection` consults `resolved_models`
(orchestrator-stamped overrides) first, then falls back to the binding
default — fixed Repo or Dispatch-table lookup on the discriminator.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import msgspec
import pytest

from gen_worker import Repo, dispatch
from gen_worker.worker import InjectionSpec, Worker, _resolved_repo_id


class _Pipe:
    pass


class _DispatchPayload(msgspec.Struct):
    variant: Literal["nf4", "int8"]
    prompt: str = ""


class _BasicPayload(msgspec.Struct):
    prompt: str = ""


def _bare_worker() -> Worker:
    """Construct a Worker without going through __init__ so we can poke at
    just the resolution methods. The methods we exercise don't touch any
    network state.
    """
    w = Worker.__new__(Worker)
    w._release_allowed_model_ids = None
    return w


def test_fixed_binding_uses_default_when_no_override() -> None:
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=Repo("acme/flux").flavor("bf16"),
    )
    payload = _BasicPayload(prompt="hi")
    model_id, key = w._resolve_model_id_for_injection(
        "generate-bf16", inj, payload=payload, resolved_models={},
    )
    assert model_id == _resolved_repo_id("acme/flux", flavor="bf16", tag="prod")
    assert "acme/flux" in model_id
    assert "bf16" in model_id


def test_fixed_binding_with_override_uses_resolved_models() -> None:
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=Repo("acme/flux").flavor("bf16").allow_override(_Pipe),
    )
    payload = _BasicPayload(prompt="hi")
    model_id, _ = w._resolve_model_id_for_injection(
        "generate-bf16-overridable",
        inj,
        payload=payload,
        resolved_models={
            "pipeline": {"ref": "acme/custom-flux", "tag": "prod", "flavor": "bf16"},
        },
    )
    assert "acme/custom-flux" in model_id


def test_fixed_binding_override_rejected_when_allow_override_false() -> None:
    """Defense-in-depth: if orchestrator stamps resolved_models for a binding
    without allow_override, the worker errors loudly (orchestrator drift).
    """
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=Repo("acme/flux").flavor("bf16"),  # no allow_override
    )
    payload = _BasicPayload()
    with pytest.raises(ValueError, match="no allow_override"):
        w._resolve_model_id_for_injection(
            "fn",
            inj,
            payload=payload,
            resolved_models={
                "pipeline": {"ref": "acme/other", "tag": "prod", "flavor": ""},
            },
        )


def test_dispatch_binding_resolves_via_discriminator() -> None:
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=dispatch(
            field="variant",
            table={
                "nf4": Repo("acme/flux").flavor("nf4"),
                "int8": Repo("acme/flux").flavor("int8"),
            },
        ),
    )
    payload = _DispatchPayload(variant="nf4")
    model_id, key = w._resolve_model_id_for_injection(
        "generate-bnb", inj, payload=payload, resolved_models={},
    )
    assert key == "nf4"
    assert "nf4" in model_id

    payload2 = _DispatchPayload(variant="int8")
    model_id2, key2 = w._resolve_model_id_for_injection(
        "generate-bnb", inj, payload=payload2, resolved_models={},
    )
    assert key2 == "int8"
    assert "int8" in model_id2


def test_dispatch_binding_override_bypasses_table() -> None:
    w = _bare_worker()
    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=dispatch(
            field="variant",
            table={"nf4": Repo("acme/flux").flavor("nf4")},
        ).allow_override(_Pipe),
    )
    payload = _DispatchPayload(variant="nf4")
    model_id, _ = w._resolve_model_id_for_injection(
        "fn",
        inj,
        payload=payload,
        resolved_models={
            "pipeline": {"ref": "acme/custom", "tag": "prod", "flavor": "bf16"},
        },
    )
    assert "acme/custom" in model_id
    assert "bf16" in model_id


def test_dispatch_binding_rejects_unknown_key() -> None:
    w = _bare_worker()

    class _Payload(msgspec.Struct):
        variant: str = ""

    inj = InjectionSpec(
        param_name="pipeline",
        param_type=_Pipe,
        binding=dispatch(
            field="variant",
            table={"nf4": Repo("acme/flux").flavor("nf4")},
        ),
    )
    payload = _Payload(variant="not-a-key")
    with pytest.raises(ValueError, match="not in table"):
        w._resolve_model_id_for_injection(
            "fn", inj, payload=payload, resolved_models={},
        )


def test_resolved_models_for_request_dict_shape() -> None:
    """Worker._resolved_models_for_request accepts both protobuf-map and
    plain-dict shapes (the latter is what test fixtures send).

    Issue #18: the extracted dict now carries `provider` as well; defaults
    to "tensorhub" when the entry omits it (pre-#358 wire-format compat).
    """
    w = _bare_worker()
    fake_request = MagicMock()
    fake_request.resolved_models = {
        "pipeline": {"ref": "acme/r", "tag": "prod", "flavor": "bf16"},
    }
    out = w._resolved_models_for_request(fake_request)
    assert out == {
        "pipeline": {"ref": "acme/r", "tag": "prod", "flavor": "bf16", "provider": "tensorhub"},
    }


def test_resolved_models_for_request_empty() -> None:
    w = _bare_worker()
    fake_request = MagicMock()
    fake_request.resolved_models = {}
    assert w._resolved_models_for_request(fake_request) == {}
