"""Binding immutability + wire round-trip — collapsed integration suite (floor 13).

  * chainable modifiers are IMMUTABLE + commutative (original untouched),
  * provider property per subclass + invalid-ref rejection at construction,
  * _binding_to_wire round-trips ref/provider/flavor(dtype) to the exact shape
    the orchestrator's release resolver consumes — including a mixed-provider
    Dispatch binding,
  * allow_override accepts class/string FQN, dedups, rejects zero-arg.

Also folds the floor-21 typed-payload-error validation path (structured
size_bytes/max_bytes fields, subclassing ValidationError).
"""

from __future__ import annotations

import pytest

from gen_worker import CivitaiRepo, Dispatch, HFRepo, Repo, dispatch
from gen_worker.worker import _binding_to_wire, _wire_ref


# --------------------------------------------------------------------------- #
# Construction + provider + immutability                                       #
# --------------------------------------------------------------------------- #


def test_construction_provider_and_invalid_ref_rejection() -> None:
    # provider property per subclass.
    assert Repo("acme/flux").provider == "tensorhub"
    assert HFRepo("Qwen/Qwen2.5-1.5B-Instruct").provider == "hf"
    assert CivitaiRepo("123456").provider == "civitai"
    # invalid refs rejected at construction.
    for ctor, arg in [(Repo, ""), (Repo, "   "), (HFRepo, "no-slash")]:
        with pytest.raises(ValueError):
            ctor(arg)


def test_modifiers_are_immutable_and_commutative() -> None:
    base = Repo("  acme/flux  ")
    assert base.ref == "acme/flux"  # ref is stripped
    flavored = base.flavor("nf4")
    tagged = flavored.tag("canary")

    # Original untouched at every step; each modifier returns a NEW instance.
    assert base._flavor == "" and base._tag == "prod"
    assert flavored._flavor == "nf4" and flavored._tag == "prod"
    assert flavored is not base and tagged is not flavored

    # Chain order is commutative.
    a = Repo("acme/flux").flavor("nf4").tag("canary")
    b = Repo("acme/flux").tag("canary").flavor("nf4")
    assert (a._flavor, a._tag) == (b._flavor, b._tag) == ("nf4", "canary")

    # Empty modifier args rejected.
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            Repo("acme/flux").flavor(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# allow_override                                                               #
# --------------------------------------------------------------------------- #


class _PipeA:
    pass


class _PipeB:
    pass


def test_allow_override_accepts_class_string_dedups_and_rejects_zero_arg() -> None:
    b = Repo("acme/flux").allow_override(_PipeA, _PipeB)
    assert b._allow_override is True
    assert any("_PipeA" in c for c in b._pipeline_classes)
    assert any("_PipeB" in c for c in b._pipeline_classes)

    fqn = f"{__name__}._PipeA"
    deduped = Repo("acme/flux").allow_override(_PipeA, fqn)  # same class twice
    assert deduped._pipeline_classes.count(fqn) == 1

    with pytest.raises(ValueError):
        Repo("acme/flux").allow_override()


# --------------------------------------------------------------------------- #
# Wire serialization — exact shape the orchestrator resolver consumes           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "binding,expect,bare_ref",
    [
        (Repo("acme/myrepo").flavor("nf4"),
         {"kind": "fixed", "ref": "acme/myrepo", "flavor": "nf4", "tag": "prod"}, "acme/myrepo"),
        (HFRepo("Qwen/Qwen2.5-1.5B-Instruct").dtype("bf16"),
         {"kind": "fixed", "ref": "Qwen/Qwen2.5-1.5B-Instruct", "provider": "hf", "dtype": "bf16"},
         "Qwen/Qwen2.5-1.5B-Instruct"),
        (CivitaiRepo("123456"),
         {"kind": "fixed", "ref": "123456", "provider": "civitai"}, "123456"),
    ],
)
def test_binding_to_wire_fixed(binding: Repo, expect: dict, bare_ref: str) -> None:
    out = _binding_to_wire("pipeline", str, binding)["binding"]
    for key, val in expect.items():
        assert out[key] == val, (key, out)
    # Tensorhub binding must NOT carry an explicit provider field; wire ref bare.
    if binding.provider == "tensorhub":
        assert "provider" not in out
    assert _wire_ref(binding) == bare_ref


def test_dispatch_wire_mixes_providers_per_entry() -> None:
    d = dispatch(field="variant", table={
        "bf16": HFRepo("owner/flux").dtype("bf16"),
        "fp8": Repo("owner/flux").flavor("fp8"),
    })
    assert isinstance(d, Dispatch)
    out = _binding_to_wire("pipeline", str, d)["binding"]
    assert out["kind"] == "dispatch" and out["field"] == "variant"
    assert out["table"]["bf16"]["provider"] == "hf"
    assert out["table"]["bf16"]["dtype"] == "bf16"
    # tensorhub entry omits provider.
    assert "provider" not in out["table"]["fp8"]
    assert out["table"]["fp8"]["flavor"] == "fp8"


# --------------------------------------------------------------------------- #
# Floor 21: typed payload-size error validation path                           #
# --------------------------------------------------------------------------- #


def test_typed_payload_size_errors_expose_structured_fields() -> None:
    from gen_worker import InputTooLargeError, OutputTooLargeError, ValidationError

    out = OutputTooLargeError(size_bytes=200, max_bytes=100)
    assert out.size_bytes == 200 and out.max_bytes == 100

    inp = InputTooLargeError(size_bytes=200, max_bytes=100, source="input file")
    assert inp.size_bytes == 200 and inp.max_bytes == 100 and inp.source == "input file"
    assert InputTooLargeError(size_bytes=10, max_bytes=5).source == "input"  # default

    assert issubclass(OutputTooLargeError, ValidationError)
    assert issubclass(InputTooLargeError, ValidationError)
