"""Per-COMPONENT load dtype — the wan fp32-VAE case.

A component's dtype is part of its resident identity and is decided AT LOAD:
upcasting a bf16-loaded VAE afterwards recovers no precision, it only hides the
truncation. These tapes use the REAL diffusers classes the fact is about
(``WanPipeline`` / ``AutoencoderKLWan``) and a real tiny ``AutoencoderKLWan``
built and saved locally (~11k params, no download, no GPU), so what is asserted
is the dtype of actual tensors on the real load paths:

  * the derived component tree carries the dtype opinion, sourced from the
    component CLASS via the one facts table — never declared per endpoint;
  * a snapshot's own ``model_index.json`` classes are authoritative at load
    time (a fine-tune may substitute a class) and the pipeline class's
    ``__init__`` annotations are the build-time source;
  * ``load_from_pretrained`` widens exactly that part and leaves the rest at
    the composition's compute dtype;
  * a uniform composition (sdxl) is untouched — no dict dtype, no behaviour
    change;
  * a SUBSTITUTED component keeps the fact (``load_component_override``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from diffusers import (
    AutoencoderKLWan,
    StableDiffusionXLPipeline,
    WanImageToVideoPipeline,
    WanPipeline,
)

from gen_worker.api.tree import (
    component_classes,
    component_dtypes,
    components_manifest,
)
from gen_worker.families.facts import (
    COMPONENT_DTYPES,
    ComponentDtype,
    component_dtype_for_class,
)
from gen_worker.models.loading import (
    component_load_dtypes,
    load_component_override,
    load_from_pretrained,
    model_index_component_classes,
)


# ---------------------------------------------------------------------------
# tiny REAL AutoencoderKLWan (built + saved locally; nothing is downloaded)
# ---------------------------------------------------------------------------


def _tiny_wan_vae() -> AutoencoderKLWan:
    return AutoencoderKLWan(
        base_dim=4, z_dim=4, dim_mult=[1, 1], num_res_blocks=1,
        attn_scales=[], temperal_downsample=[False], dropout=0.0,
    )


def _wan_tree(root: Path, *, vae_class: str = "AutoencoderKLWan") -> Path:
    """A diffusers-layout snapshot naming a Wan VAE in model_index.json, with a
    real (tiny) VAE on disk under ``vae/`` and a stand-in transformer dir."""
    root.mkdir(parents=True, exist_ok=True)
    _tiny_wan_vae().save_pretrained(root / "vae")
    (root / "transformer").mkdir(exist_ok=True)
    (root / "transformer" / "weights.txt").write_text("tiny-transformer")
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "WanPipeline",
        "vae": ["diffusers", vae_class],
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }))
    return root


class _DtypeMapPipeline:
    """Stand-in root pipeline implementing diffusers' documented per-component
    ``torch_dtype`` semantics (``{"default": ..., "<part>": ...}``) and loading
    the REAL tiny VAE off the tree at the dtype the map selects.

    Only the container is a stand-in: a full real WanPipeline needs a
    tokenizer, which is a download, and downloading weights on this box is
    forbidden. Everything asserted below is a real tensor dtype produced by
    the real ``AutoencoderKLWan.from_pretrained``.
    """

    last_kwargs: Dict[str, Any] = {}

    def __init__(self, vae: Any, dtype_arg: Any) -> None:
        self.vae = vae
        self.dtype_arg = dtype_arg

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "_DtypeMapPipeline":
        cls.last_kwargs = dict(kwargs)
        dtype = kwargs.get("torch_dtype")
        if isinstance(dtype, dict):
            part = dtype.get("vae", dtype.get("default", torch.float32))
        else:
            part = dtype if dtype is not None else torch.float32
        vae = AutoencoderKLWan.from_pretrained(
            str(Path(path) / "vae"), torch_dtype=part)
        return cls(vae, dtype)

    def to(self, *_a: Any, **_kw: Any) -> "_DtypeMapPipeline":
        return self


def _vae_dtype(vae: Any) -> torch.dtype:
    return next(vae.parameters()).dtype


# ---------------------------------------------------------------------------
# the facts table and the derived tree
# ---------------------------------------------------------------------------


def test_facts_table_keys_on_the_component_class() -> None:
    fact = component_dtype_for_class("AutoencoderKLWan")
    assert fact is not None and fact.dtype == "fp32"
    assert fact.reason, "a fact carries its reason as DATA (the loader logs it)"
    assert component_dtype_for_class("AutoencoderKL") is None
    assert component_dtype_for_class("") is None


def test_every_fact_row_is_a_legal_load_dtype_with_a_reason() -> None:
    for name, fact in COMPONENT_DTYPES.items():
        assert name and name[0].isupper(), f"{name!r} must be a class name"
        assert isinstance(fact, ComponentDtype)
        # __post_init__ enforces both; this locks the table itself.
        ComponentDtype(fact.dtype, fact.reason)


def test_illegal_fact_rows_are_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        ComponentDtype("fp8", "storage lanes are the fit ladder's business")
    with pytest.raises(ValueError):
        ComponentDtype("fp32", "   ")


def test_real_wan_pipeline_classes_derive_the_vae_fact_with_no_declaration() -> None:
    # The fact rides the CLASS, so every wan pipeline class inherits it and no
    # endpoint declares anything.
    assert component_classes(WanPipeline)["vae"] == "AutoencoderKLWan"
    for cls in (WanPipeline, WanImageToVideoPipeline):
        facts = component_dtypes(cls)
        assert set(facts) == {"vae"}, cls.__name__
        assert facts["vae"].dtype == "fp32"


def test_a_uniform_composition_derives_nothing() -> None:
    assert component_dtypes(StableDiffusionXLPipeline) == {}


def test_manifest_publishes_the_dtype_with_the_component_tree() -> None:
    rows = components_manifest(WanPipeline)
    assert rows is not None
    by_name = {r["name"]: r for r in rows}
    assert by_name["vae"]["kind"] == "weights"
    assert by_name["vae"]["dtype"] == "fp32"
    # Parts with no opinion publish no dtype key at all.
    assert "dtype" not in by_name["transformer"]
    assert all("dtype" not in r for r in components_manifest(
        StableDiffusionXLPipeline) or [])


def test_model_index_classes_are_authoritative_at_load_time(tmp_path: Path) -> None:
    tree = _wan_tree(tmp_path / "wan")
    assert model_index_component_classes(tree)["vae"] == "AutoencoderKLWan"
    # A fine-tune substituting a plain VAE class loses the fact — the bytes on
    # disk decide, not the pipeline class's annotation.
    swapped = _wan_tree(tmp_path / "swapped", vae_class="AutoencoderKL")
    assert component_load_dtypes(WanPipeline, swapped) == {}
    assert set(component_load_dtypes(WanPipeline, tree)) == {"vae"}
    # No model_index.json at all: annotations still carry the build-time fact.
    assert set(component_load_dtypes(WanPipeline, tmp_path / "missing")) == {"vae"}


# ---------------------------------------------------------------------------
# the materialize half
# ---------------------------------------------------------------------------


def test_load_widens_only_the_fragile_component(tmp_path: Path) -> None:
    tree = _wan_tree(tmp_path / "wan")
    pipe = load_from_pretrained(_DtypeMapPipeline, tree, dtype="bf16")
    passed = _DtypeMapPipeline.last_kwargs["torch_dtype"]
    assert isinstance(passed, dict), "a fragile part needs a per-component map"
    assert passed["vae"] is torch.float32
    assert passed["default"] is torch.bfloat16, \
        "the rest of the composition keeps the binding's compute dtype"
    assert _vae_dtype(pipe.vae) is torch.float32, \
        "the VAE's real tensors are fp32 AT LOAD, not upcast afterwards"


def test_uniform_composition_load_is_byte_identical(tmp_path: Path) -> None:
    # Same tree, VAE class swapped: no fact, so nothing about the load changes.
    tree = _wan_tree(tmp_path / "plain", vae_class="AutoencoderKL")
    load_from_pretrained(_DtypeMapPipeline, tree, dtype="bf16")
    assert _DtypeMapPipeline.last_kwargs["torch_dtype"] is torch.bfloat16


def test_default_falls_back_to_the_snapshot_dtype_not_the_fact(
    tmp_path: Path,
) -> None:
    # No declared binding dtype: the composition default is still derived the
    # usual way (bf16), and only the fragile part is widened.
    tree = _wan_tree(tmp_path / "wan")
    load_from_pretrained(_DtypeMapPipeline, tree)
    passed = _DtypeMapPipeline.last_kwargs["torch_dtype"]
    assert passed["default"] is torch.bfloat16
    assert passed["vae"] is torch.float32


def test_a_fact_agreeing_with_the_default_stays_out_of_the_kwargs(
    tmp_path: Path,
) -> None:
    # An fp32-computing composition needs no map: the widening would be a no-op
    # and a redundant dict would change the kwargs for nothing.
    tree = _wan_tree(tmp_path / "wan")
    load_from_pretrained(_DtypeMapPipeline, tree, dtype="fp32")
    assert _DtypeMapPipeline.last_kwargs["torch_dtype"] is torch.float32


def test_substituted_component_keeps_the_fact(tmp_path: Path) -> None:
    # The base composition computes bf16, but the
    # substituted Wan VAE must still be resident fp32.
    base = _wan_tree(tmp_path / "base")
    override = tmp_path / "override"
    override.mkdir()
    _tiny_wan_vae().save_pretrained(override / "vae")
    module = load_component_override(base, "vae", override, dtype="bf16")
    assert isinstance(module, AutoencoderKLWan)
    assert _vae_dtype(module) is torch.float32
