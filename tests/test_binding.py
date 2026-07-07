"""Binding constructors: single positional ref + kw metadata, immutability,
wire-ref encoding."""

from __future__ import annotations

import pytest

from gen_worker import HF, Civitai, Hub, ModelScope
from gen_worker.api.binding import wire_ref


def test_construction_provider_and_invalid_ref_rejection() -> None:
    assert Hub("owner/repo").provider == "tensorhub"
    assert HF("owner/repo").provider == "hf"
    assert Civitai("123456").provider == "civitai"
    assert ModelScope("owner/repo").provider == "modelscope"

    with pytest.raises(ValueError):
        Hub("")
    with pytest.raises(ValueError):
        HF("norepo")  # must be owner/repo
    with pytest.raises(ValueError):
        Civitai("")
    with pytest.raises(ValueError):
        ModelScope("norepo")


def test_kw_metadata_normalized_and_frozen() -> None:
    b = HF(" owner/repo ", revision=" main ", dtype=" bf16 ", subfolder=" te ",
           files=(" a/*.safetensors ", ""))
    assert b.ref == "owner/repo"
    assert b.revision == "main"
    assert b.dtype == "bf16"
    assert b.subfolder == "te"
    assert b.files == ("a/*.safetensors",)
    with pytest.raises(Exception):
        b.dtype = "fp16"  # frozen

    hub = Hub("o/r", tag="", flavor=" nf4 ")
    assert hub.tag == "prod"
    assert hub.flavor == "nf4"


def test_bindings_are_hashable_and_equal_by_value() -> None:
    assert HF("o/r", dtype="bf16") == HF("o/r", dtype="bf16")
    assert len({HF("o/r"), HF("o/r"), HF("o/r", dtype="bf16")}) == 2


def test_quantize_metadata_validated_and_normalized() -> None:
    from gen_worker.api.binding import QUANTIZE_METHODS

    assert HF("o/r", quantize=" NF4 ").quantize == "nf4"
    assert Hub("o/r", quantize="int8-torchao").quantize == "int8-torchao"
    assert HF("o/r").quantize == ""
    for m in QUANTIZE_METHODS:
        assert HF("o/r", quantize=m).quantize == m
    with pytest.raises(ValueError, match="unknown quantize method"):
        HF("o/r", quantize="q4_k_m")
    with pytest.raises(ValueError):
        Hub("o/r", quantize="8bit")


def test_quantize_never_enters_wire_ref_but_surfaces_in_manifest() -> None:
    from gen_worker.cli.listing import describe_binding

    assert wire_ref(HF("o/r", quantize="nf4")) == "o/r"
    assert wire_ref(Hub("o/r", quantize="int8")) == "o/r"
    assert describe_binding(HF("o/r", quantize="nf4"))["quantize"] == "nf4"
    assert describe_binding(Hub("o/r", flavor="bf16", quantize="int8"))["quantize"] == "int8"
    assert "quantize" not in describe_binding(HF("o/r"))


def test_wire_ref_encoding() -> None:
    assert wire_ref(Hub("o/r")) == "o/r"
    assert wire_ref(Hub("o/r", tag="canary")) == "o/r:canary"
    assert wire_ref(Hub("o/r", flavor="nf4")) == "o/r#nf4"
    assert wire_ref(Hub("o/r", tag="canary", flavor="nf4")) == "o/r:canary#nf4"
    assert wire_ref(HF("o/r")) == "o/r"
    assert wire_ref(HF("o/r", revision="abc")) == "o/r@abc"
    # load-time metadata never enters the ref
    assert wire_ref(HF("o/r", dtype="bf16", subfolder="te")) == "o/r"
    assert wire_ref(Civitai("123", version="456")) == "123"


def test_typed_payload_size_errors_expose_structured_fields() -> None:
    from gen_worker import ValidationError
    from gen_worker.api.errors import InputTooLargeError, OutputTooLargeError

    out_err = OutputTooLargeError(size_bytes=10, max_bytes=5)
    assert isinstance(out_err, ValidationError)
    assert (out_err.size_bytes, out_err.max_bytes) == (10, 5)
    in_err = InputTooLargeError(size_bytes=7, max_bytes=3, source="payload")
    assert isinstance(in_err, ValidationError)
    assert (in_err.size_bytes, in_err.max_bytes, in_err.source) == (7, 3, "payload")
