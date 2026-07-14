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
    assert hub.tag == "latest"  # the grammar default (gw#492)
    assert hub.flavor == "nf4"


def test_storage_dtype_validated_and_normalized() -> None:
    assert HF("o/r", storage_dtype=" FP8 ").storage_dtype == "fp8"
    assert Hub("o/r", storage_dtype="fp8").storage_dtype == "fp8"
    assert HF("o/r").storage_dtype == ""
    with pytest.raises(ValueError, match="unknown storage_dtype"):
        HF("o/r", storage_dtype="nf4")  # runtime quant is not a binding kwarg
    with pytest.raises(ValueError):
        Hub("o/r", storage_dtype="int8")


def test_storage_dtype_never_enters_wire_ref_but_surfaces_in_manifest() -> None:
    from gen_worker.cli.listing import describe_binding

    assert wire_ref(HF("o/r", storage_dtype="fp8")) == "o/r"
    assert wire_ref(Hub("o/r", storage_dtype="fp8")) == "o/r"
    assert describe_binding(HF("o/r", storage_dtype="fp8"))["storage_dtype"] == "fp8"
    assert describe_binding(Hub("o/r", flavor="fp8", storage_dtype="fp8"))["storage_dtype"] == "fp8"
    assert "storage_dtype" not in describe_binding(HF("o/r"))


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


def test_components_allowed_on_tensorhub_and_huggingface_only() -> None:
    hub = Hub("o/r", components=(" vae ", "", "unet"))
    assert hub.components == ("vae", "unet")

    hf = HF("o/r", components=("vae",))
    assert hf.components == ("vae",)

    # Civitai()/ModelScope() don't even expose a components= kwarg; direct
    # ModelRef construction is the only way to hit the validation.
    from gen_worker.api.binding import ModelRef

    with pytest.raises(ValueError, match="components="):
        ModelRef(source="civitai", path="123", components=("vae",))
    with pytest.raises(ValueError, match="components="):
        ModelRef(source="modelscope", path="o/r", components=("vae",))


def test_typed_payload_size_errors_expose_structured_fields() -> None:
    from gen_worker import ValidationError
    from gen_worker.api.errors import OutputTooLargeError

    out_err = OutputTooLargeError(size_bytes=10, max_bytes=5)
    assert isinstance(out_err, ValidationError)
    assert (out_err.size_bytes, out_err.max_bytes) == (10, 5)
