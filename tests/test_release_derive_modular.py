"""pgw#1450: a MODULAR pipeline's components must reach the derive.

``gen-worker release derive`` over a modular-pipeline endpoint, through the
actual CLI codepath. Before tcg#65 this refused --

    minimax-h3 serve recipe: no DiT resolves on this pipeline -- arming nothing
    derive error: lane 'minimax.h3-dit-diffusers@1': load() marked nothing via
    ctx.compile()

-- because ``ModularPipeline.from_pretrained`` builds component SPECS and
leaves every attribute ``None``, deferring the build to whoever holds the
object. At serve that holder is the streaming engine; a derive has none.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    "version = 1\n"
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def modular_config_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import modular_tiny_tree as modular_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return modular_tree.save_config_only(tmp_path_factory.mktemp("modular-config-only"))


def _derive(tree: Path, out: Path) -> int:
    from gen_worker.cli import main

    lockfile = out.parent / "uv.lock"
    lockfile.write_text(LOCK)
    return main(
        [
            "release",
            "derive",
            "--dir",
            str(FIXTURES),
            "--module",
            "modular_tiny_endpoint",
            "--checkpoint",
            str(tree),
            "--lockfile",
            str(lockfile),
            "--out",
            str(out),
        ]
    )


def test_a_modular_endpoint_derives_its_marked_denoiser(
    modular_config_tree: Path, tmp_path: Path
) -> None:
    """The whole issue in one assertion: the lane carries graphs.

    ``load() marked nothing via ctx.compile()`` is a DeriveError, so a red run
    does not reach the document at all -- the exit code is the first signal.
    """

    out = tmp_path / "release.json"
    assert _derive(modular_config_tree, out) == 0

    document = json.loads(out.read_bytes())
    assert document["kind"] == "gen-worker.release-metadata@1"
    assert document["endpoint"].endswith(":ModularModel")

    (lane,) = document["graphs"]["lanes"]
    assert lane["contract"] == "sd15.diffusers@1+plain.f32@1"
    assert lane["unobserved_targets"] == []
    # One specialization per enumerated payload arm: Size.SMALL/LARGE change
    # the denoiser's latent side, which is what the enumerator is for.
    assert len(lane["graphs"]) == 2
    assert {record["target"] for record in lane["graphs"]} == {"unet"}
    sides = sorted(record["ingress"]["inputs"][0]["shape"][-1] for record in lane["graphs"])
    assert sides == [4, 8]


def test_the_pipeline_the_author_marked_carries_every_declared_component(
    modular_config_tree: Path,
) -> None:
    """The mechanism, at the seam, without the CLI in the way.

    pgw#1450 measured the opposite on the real H3 tree: ``pipe.components``
    naming eleven and every attribute ``None``.
    """

    sys.path.insert(0, str(FIXTURES))
    try:
        import modular_tiny_tree as modular_tree
    finally:
        sys.path.remove(str(FIXTURES))

    import torch
    from gen_worker._vendor.torchcg.hollow import hollow_session

    with hollow_session("cuda"):
        pipeline = modular_tree.TinyStreamingPipeline.from_pretrained(
            str(modular_config_tree), torch_dtype=torch.float32
        )

    components = pipeline.components
    assert sorted(components) == sorted(modular_tree.COMPONENTS)
    assert [name for name, value in components.items() if value is None] == []
    modules = sorted(
        name for name, value in components.items() if isinstance(value, torch.nn.Module)
    )
    assert modules == ["text_encoder", "unet", "vae"]
