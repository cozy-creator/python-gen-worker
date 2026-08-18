"""pgw#1415 — the hub's per-checkpoint defaults row reaches the entrypoint.

The defect was pure plumbing with a wire cause. tensorhub's th#2140 landed
`ModelBinding.model = 9`; this repo's vendored copy still called 9
`manifest_digest` (th#1941, a field NOTHING has ever sent), so the hub stamped
a model NAME into a slot the worker read as a fetch POINTER, `DeployBinding`
was built without `model=`, and `decode_model_defaults` warned "unclassified"
and served type fallbacks for every classified checkpoint in the fleet. On
FLUX.2 Klein Turbo that is 28 steps with CFG on for a 4-step guidance-free
checkpoint: a different image at ~7x the cost.

Everything here drives the REAL chain — a real `pb.RunJob`, the real
`HubBindingResolver`, the real `ServeLoop` over the real fixture endpoint —
because the defect lived exactly in the gap between the parts, and every
piece in isolation was already correct.
"""

from __future__ import annotations

import json
from pathlib import Path
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

#: The Turbo checkpoint's real row shape: a step-distilled, guidance-free
#: recipe. Against Klein's platform values (steps 28, cfg True) this is the
#: measured delta the umbrella filed.
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
    """One real dispatch in flight: a `pb.RunJob` projected the way the wire
    head projects it, plus this pod's materialized snapshot tree."""
    root = tmp_path / "snapshots"
    tree = root / snapshot_dir_key(digest.split(":", 1)[-1])
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "config.json").write_text(json.dumps({"seed": 1}))
    run = pb.RunJob(
        models=[
            pb.ModelBinding(
                slot="model",
                ref=ref,
                manifest_digest=digest,
                model=model,
                inference_defaults="" if row is None else json.dumps(row),
            )
        ]
    )
    _DISPATCH.set(_picks_of(run))
    return HubBindingResolver(snapshots_root=root)


# ── the wire vocabulary ──────────────────────────────────────────────────────


def test_the_classification_column_is_on_the_wire_at_the_hubs_number() -> None:
    """These numbers are TENSORHUB's, and tensorhub is the only SENDER.

    `mb.Model` is stamped on every dispatch (`scheduler_dispatch.go`), while
    `manifest_digest` has never had a sender in any repo (th#2116 still owes
    it). When the two protos disagreed on tag 9, the field with a live writer
    won — which is why `manifest_digest` is 10 here and must be 10 there.
    """
    fields = pb.ModelBinding.DESCRIPTOR.fields_by_name
    assert fields["model"].number == 9
    assert fields["manifest_digest"].number == 10
    # The adapter half of the two-column pair (`sdxl.lora` rows).
    assert pb.LoraOverlay.DESCRIPTOR.fields_by_name["model"].number == 4


# ── resolver: the fact it holds is the fact it passes on ─────────────────────


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
    """The arm that was always CORRECT: no name and no row is a genuinely
    unclassified checkpoint, and it binds to platform fallbacks under
    pgw#1377's named warning rather than to a guess."""
    resolver = _dispatch(tmp_path, ref=KLEIN, model="", row=None)
    binding = resolver.resolve(Flux2Klein, KLEIN)
    assert binding.model is None
    with pytest.warns(UserWarning, match="unclassified"):
        assert LoadContext(binding=binding, model_type=Flux2Klein).defaults() \
            == Flux2Klein.Defaults()


def test_a_row_without_its_name_is_refused_not_silently_downgraded(
    tmp_path: Path,
) -> None:
    """The fence (acceptance 4). The hub's `model` column is NOT NULL beside
    the defaults JSONB (th#2140 migration 0104), so a row arriving WITHOUT a
    name is a pair that broke in transit — which is this defect's exact
    signature. It refuses loudly instead of serving fallbacks while holding
    the checkpoint's own tuned recipe."""
    resolver = _dispatch(tmp_path, ref=KLEIN, model="", row=TURBO_CHECKPOINT_ROW)
    with pytest.raises(CheckpointUnresolved, match="no `model` classification"):
        resolver.resolve(Flux2Klein, KLEIN)


# ── the measurement, through the real path ───────────────────────────────────


def test_the_turbo_row_reaches_ctx_defaults_and_flips_the_measurement(
    tmp_path: Path,
) -> None:
    """hub row -> resolver -> DeployBinding -> decode -> ctx.defaults().

    The umbrella's numbers, reproduced on the shipped chain rather than on a
    unit shim: `model=None` served Klein's platform 28/True; the hub's own
    `flux2-klein` row serves 4/False.
    """
    resolver = _dispatch(
        tmp_path, ref=KLEIN, model="flux2-klein", row=TURBO_CHECKPOINT_ROW
    )
    ctx: LoadContext[Any] = LoadContext(
        binding=resolver.resolve(Flux2Klein, KLEIN), model_type=Flux2Klein
    )
    served = ctx.defaults()
    assert (served.steps.default, served.cfg) == (4, False)
    assert served.step_distilled is True

    # What production served before the fix — kept as the CONTRAST, since a
    # number is only evidence beside the number it replaced.
    fallbacks = Flux2Klein.Defaults()
    assert (fallbacks.steps.default, fallbacks.cfg) == (28, True)


def test_the_row_reaches_what_an_entrypoint_actually_receives(
    tmp_path: Path,
) -> None:
    """The whole stack: real RunJob -> real HubBindingResolver -> ServeLoop ->
    the fixture endpoint's `@entrypoint`.

    `step_distilled` is the sharpest observable there is: its ONLY purpose is
    refusing to stack a step-distillation on an already-distilled checkpoint,
    so the warning can only fire if the hub's row survived the whole way into
    author code. Before the fix the row was dropped, the fallback said
    `step_distilled=False`, and the endpoint stacked the adapter.
    """
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
        lane_contract="sdxl.diffusers-bf16@1",
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
    # cfg=False came from the same row: no unconditional branch, so the
    # request's guidance is refused out loud rather than quietly applied.
    assert any("without classifier-free guidance" in w for w in outcome.warnings)
