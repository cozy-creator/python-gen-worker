"""Direct (ref, flavor) ModelRef shape — issue #8."""

import pytest

from gen_worker import ModelRef, ModelRefSource as Src
from gen_worker.discovery.discover import _synthesize_model_key


def test_direct_ref_flavor_construction():
    r = ModelRef(Src.FIXED, ref="owner/repo", flavor="nf4")
    assert r.ref == "owner/repo"
    assert r.flavor == "nf4"
    assert r.tag == "prod"  # default
    assert r.key == ""


def test_tag_overrideable():
    r = ModelRef(Src.FIXED, ref="owner/repo", tag="canary", flavor="fp8")
    assert r.tag == "canary"


def test_legacy_key_form_still_works():
    r = ModelRef(Src.FIXED, "local-key")
    assert r.key == "local-key"
    assert r.ref is None


def test_payload_key_required():
    r = ModelRef(Src.PAYLOAD, "variant")
    assert r.key == "variant"
    with pytest.raises(ValueError, match="requires `key="):
        ModelRef(Src.PAYLOAD)


def test_payload_rejects_ref():
    with pytest.raises(ValueError, match="only valid for Src.FIXED"):
        ModelRef(Src.PAYLOAD, "variant", ref="owner/repo")


def test_fixed_rejects_both_key_and_ref():
    with pytest.raises(ValueError, match="EITHER `key=` .* OR `ref=`"):
        ModelRef(Src.FIXED, "k", ref="owner/repo")


def test_fixed_rejects_neither_key_nor_ref():
    with pytest.raises(ValueError, match="requires either `ref=`"):
        ModelRef(Src.FIXED)


def test_fixed_rejects_flavor_without_ref():
    with pytest.raises(ValueError, match="`flavor=` requires `ref=`"):
        ModelRef(Src.FIXED, "k", flavor="nf4")


def test_synthesize_key_deterministic():
    a = _synthesize_model_key("black-forest-labs/flux.2-klein-4b-turbo", "nf4")
    b = _synthesize_model_key("black-forest-labs/flux.2-klein-4b-turbo", "nf4")
    assert a == b
    assert a == "black-forest-labs__flux-2-klein-4b-turbo__nf4"


def test_synthesize_key_no_flavor():
    k = _synthesize_model_key("owner/repo", "")
    assert k == "owner__repo"


def test_discovery_synthesizes_models_entry(tmp_path, monkeypatch):
    """End-to-end: function declares direct (ref, flavor); discovery emits
    a synthesized [models] entry without endpoint.toml needing one."""
    src_pkg = tmp_path / "demo_direct"
    src_pkg.mkdir()
    (src_pkg / "__init__.py").write_text("")
    (src_pkg / "main.py").write_text(
        "from typing import Annotated\n"
        "import msgspec\n"
        "from gen_worker import RequestContext, inference_function, ModelRef, ModelRefSource as Src\n"
        "\n"
        "class Inp(msgspec.Struct):\n"
        "    prompt: str = ''\n"
        "\n"
        "class Out(msgspec.Struct):\n"
        "    text: str = ''\n"
        "\n"
        "@inference_function\n"
        "def generate(\n"
        "    ctx: RequestContext,\n"
        "    model: Annotated[object, ModelRef(Src.FIXED, ref='owner/repo', flavor='nf4')],\n"
        "    payload: Inp,\n"
        ") -> Out:\n"
        "    return Out(text=payload.prompt)\n"
    )
    (tmp_path / "endpoint.toml").write_text(
        'schema_version = 1\nname = "demo"\nmain = "demo_direct.main"\n\n'
        "[resources]\nram_gb = 2\ncpu_cores = 1\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    from gen_worker.discovery.discover import discover_manifest

    manifest = discover_manifest(tmp_path)
    assert manifest["functions"][0]["required_models"] == ["owner__repo__nf4"]
    assert manifest["models"]["owner__repo__nf4"]["ref"] == "owner/repo"
    assert manifest["models"]["owner__repo__nf4"]["flavor"] == "nf4"
