"""Binding immutability + wire round-trip — collapsed integration suite (floor 13).

  * chainable modifiers are IMMUTABLE + commutative (original untouched),
  * provider property per subclass + invalid-ref rejection at construction,
  * allow_override accepts class/string FQN, dedups, rejects zero-arg.

Also folds the floor-21 typed-payload-error validation path (structured
size_bytes/max_bytes fields, subclassing ValidationError).
"""

from __future__ import annotations

import pytest

from gen_worker import CivitaiRepo, HFRepo, Repo


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
