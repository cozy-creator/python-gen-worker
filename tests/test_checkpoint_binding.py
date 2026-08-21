"""One hub ``ModelBinding`` becoming one worker ``DeployBinding``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from gen_worker.models import Flux2Klein
from gen_worker.models.cozy_snapshot import snapshot_dir_key
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.serving import load_endpoint
from gen_worker.serving.context import LoadContext
from gen_worker.serving.residency import ResidencyManager
from gen_worker.serving.serve_loop import ServeLoop, manifest_sizer
from gen_worker.worker import (
    _DISPATCH,
    CheckpointUnresolved,
    HubBindingResolver,
    _picks_of,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"

GB = 1024**3
DREAM = "org/dreamshaper@2"
KLEIN = "bfl/flux.2-klein-4b@turbo"
DIGEST = "sha256:c0ffee"

TURBO_CHECKPOINT_ROW: Dict[str, Any] = {
    "steps": {"default": 4},
    "cfg": False,
    "step_distilled": True,
}

TURBO_ADAPTER = {
    "ref": "cozy/lightning-4step@1", "path": "/adapters/lightning.safetensors",
    "name": "lightning-4step", "distillation": True,
}


def _dispatch(
    tmp_path: Path,
    *,
    ref: str,
    model: str,
    row: Dict[str, Any] | None,
    digest: str = DIGEST,
) -> HubBindingResolver:
    root = tmp_path / "snapshots"
    tree = root / snapshot_dir_key(digest.split(":", 1)[-1])
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "config.json").write_text(json.dumps({"seed": 1}))
    run = pb.RunJob(
        models=[
            pb.ModelBinding(
                slot="model",
                ref=ref,
                model=model,
                inference_defaults="" if row is None else json.dumps(row),
                bind_contract_digest="contract",
                bind_contract_url="https://hub.invalid/bind",
            )
        ]
    )
    _DISPATCH.set(_picks_of(run))
    resolver = HubBindingResolver(snapshots_root=root)
    resolver._bind_contracts["contract"] = SimpleNamespace(
        digest="contract",
        identity=SimpleNamespace(release_id="release"),
        census=None,
    )
    return resolver


def test_the_classification_column_is_on_the_wire_at_the_hubs_number() -> None:
    """These numbers are TENSORHUB's, and tensorhub is the only SENDER."""
    fields = pb.ModelBinding.DESCRIPTOR.fields_by_name
    assert fields["model"].number == 9
    assert fields["bind_contract_digest"].number == 11
    assert fields["bind_contract_url"].number == 12
    assert "manifest_digest" not in fields
    assert pb.LoraOverlay.DESCRIPTOR.fields_by_name["model"].number == 4


def test_resolve_carries_the_hubs_classification_onto_the_binding(
    tmp_path: Path,
) -> None:
    """RED before the fix: `resolve` parsed the row and dropped the name."""
    resolver = _dispatch(
        tmp_path, ref=KLEIN, model="flux2-klein", row=TURBO_CHECKPOINT_ROW
    )
    binding = resolver.resolve(Flux2Klein, KLEIN)
    assert binding.model == "flux2-klein"
    assert binding.defaults == TURBO_CHECKPOINT_ROW


def test_an_unclassified_checkpoint_still_binds_with_no_name(
    tmp_path: Path,
) -> None:
    resolver = _dispatch(tmp_path, ref=KLEIN, model="", row=None)
    binding = resolver.resolve(Flux2Klein, KLEIN)
    assert binding.model is None
    with pytest.warns(UserWarning, match="unclassified"):
        assert LoadContext(binding=binding, model_type=Flux2Klein).defaults() \
            == Flux2Klein.Defaults()


def test_a_row_without_its_name_is_refused_not_silently_downgraded(
    tmp_path: Path,
) -> None:
    """The fence."""
    resolver = _dispatch(tmp_path, ref=KLEIN, model="", row=TURBO_CHECKPOINT_ROW)
    with pytest.raises(CheckpointUnresolved, match="no `model` classification"):
        resolver.resolve(Flux2Klein, KLEIN)


def test_the_turbo_row_reaches_ctx_defaults_and_flips_the_measurement(
    tmp_path: Path,
) -> None:
    """hub row -> resolver -> DeployBinding -> decode -> ctx.defaults()."""
    resolver = _dispatch(
        tmp_path, ref=KLEIN, model="flux2-klein", row=TURBO_CHECKPOINT_ROW
    )
    ctx: LoadContext[Any] = LoadContext(
        binding=resolver.resolve(Flux2Klein, KLEIN), model_type=Flux2Klein
    )
    served = ctx.defaults()
    assert (served.steps.default, served.cfg) == (4, False)
    assert served.step_distilled is True

    fallbacks = Flux2Klein.Defaults()
    assert (fallbacks.steps.default, fallbacks.cfg) == (28, True)


def test_the_row_reaches_what_an_entrypoint_actually_receives(
    tmp_path: Path,
) -> None:
    """The whole stack: real RunJob -> real HubBindingResolver -> ServeLoop -> the fixture endpoint's `@entrypoint`."""
    resolver = _dispatch(
        tmp_path,
        ref=DREAM,
        model="sdxl",
        row={"step_distilled": True, "cfg": False},
    )
    loop = ServeLoop(
        load_endpoint(FIXTURE_DIR),
        residency=ResidencyManager(
            64 * GB, manifest_sizer({DREAM: 3 * GB}, headroom_bytes=1 * GB)
        ),
        resolver=resolver,
        lane_contract="sdxl.diffusers@1+plain.bf16@1",
        output_dir=tmp_path / "outputs",
    )
    outcome = loop.invoke(
        "generate",
        {"model": DREAM, "adapters": {"turbo": TURBO_ADAPTER},
         "input": {"prompt": "a lighthouse", "guidance_scale": 7.5}},
        request_id="req-1415",
    )
    assert any("already step-distilled" in w for w in outcome.warnings)
    assert outcome.result.loras == []
    assert any("without classifier-free guidance" in w for w in outcome.warnings)
