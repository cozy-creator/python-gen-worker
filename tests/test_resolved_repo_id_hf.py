"""Issue #20 fix 2: `_resolved_repo_id` drops `#flavor` for HF / civitai
refs. Flavor is tensorhub-only — leaking it through HF model_ids confused
``PipelineLoader.load`` (which stripped `#bf16` and fell back to a default
dtype, silently loading `#fp8` and `#nf4` refs at bf16).
"""

from __future__ import annotations

from gen_worker.worker import _resolved_repo_id


def test_tensorhub_ref_keeps_flavor_in_model_id() -> None:
    """Tensorhub flow unchanged: flavor is part of the canonical id."""
    model_id = _resolved_repo_id(
        "acme/y", flavor="bf16", tag="prod", provider="tensorhub"
    )
    assert model_id == "acme/y#bf16"


def test_tensorhub_ref_with_custom_tag_and_flavor() -> None:
    model_id = _resolved_repo_id(
        "acme/y", flavor="bf16", tag="canary", provider="tensorhub"
    )
    assert model_id == "acme/y:canary#bf16"


def test_hf_ref_drops_flavor_from_model_id() -> None:
    """HF model_ids never carry #flavor — HF has dtype, not flavor."""
    model_id = _resolved_repo_id(
        "owner/repo", flavor="bf16", tag="prod", provider="hf"
    )
    assert model_id == "hf::owner/repo"
    assert "#" not in model_id


def test_hf_ref_bare_when_no_flavor() -> None:
    model_id = _resolved_repo_id("owner/repo", flavor="", provider="hf")
    assert model_id == "hf::owner/repo"


def test_civitai_ref_drops_flavor_from_model_id() -> None:
    """Civitai also has no flavor concept on the ref side."""
    model_id = _resolved_repo_id("123456", flavor="fp16", provider="civitai")
    assert model_id == "civitai::123456"


def test_default_provider_is_tensorhub() -> None:
    """Back-compat: callers that don't pass provider still get tensorhub
    shape (with flavor)."""
    assert _resolved_repo_id("acme/y", flavor="nf4") == "acme/y#nf4"


def test_empty_provider_treated_as_tensorhub() -> None:
    """Empty-string provider should land on the tensorhub branch (keeps flavor)."""
    assert _resolved_repo_id("acme/y", flavor="nf4", provider="") == "acme/y#nf4"
