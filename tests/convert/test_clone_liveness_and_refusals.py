"""What a clone SAYS while it works, and what it says when it cannot.

pgw#1397 / jobs#294 — three facts, all three measured on real pods first:

1.  the declared POSITION advances. Hub job liveness is position advance
    inside a phase budget (th#2050) and the position is an int64 the hub
    accepts only on a STRICT increase, so the old 0..1 fraction was the
    constant 0 and a 6.9 GB fetch was indistinguishable from a wedged one.
2.  a repo whose weights are ALL pickle refuses ``pickle_only``, not the
    generic ``missing_safetensors`` — different facts, different advice.
3.  a request no flavor of which could be produced is refused from the
    source's METADATA, before the download is paid for.

Revert-turns-red: restore ``fn(p, stage=stage)`` in ``run_clone._progress``
and test 1 sees exactly one accepted position, ``clone.plan 0`` — verbatim the
fault the battery hit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gen_worker.api.errors import ValidationError
from gen_worker.convert.classifier import RepoRefusal, classify_repo
from gen_worker.convert.clone import (
    normalize_outputs,
    refuse_unproducible_layout,
    run_clone,
)
from gen_worker.convert.ingest import CivitaiSourcePlan, IngestedSource

from fake_hub import _FakeHub

# Juggernaut XL v1759168 — the exact source the battery killed.
_JUGGERNAUT_BYTES = 6_939_220_248


class _Ctx:
    """Records what reached ``ctx.progress``, nothing else."""

    def __init__(self, server: Any) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "tensorhub"
        self.request_id = "req-1397"
        self.destination = {"repo": "tensorhub/fallback"}
        self.ticks: list[tuple[Any, Any, Any, Any]] = []

    def progress(self, progress: Any = None, stage: Any = None, *,
                 step: Any = None, total: Any = None, position: Any = None,
                 phase: Any = None) -> None:
        self.ticks.append((progress, stage or phase, position, total))


def _source(dest_dir: Path) -> IngestedSource:
    """A civitai checkpoint as `ingest_civitai` really returns one: single
    file, NO classification (only the gguf-only shape ever gets one)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "model.safetensors").write_bytes(b"\x00" * (300 * 1024 * 1024))
    return IngestedSource(
        provider="civitai", source_ref="1759168", source_revision="sha256:x",
        dir=dest_dir, layout="single-file", model_family="unknown",
        model_family_variant="unknown", classification=None,
        attrs={"dtype": "fp32", "file_layout": "single-file"},
        metadata={"source_provider": "civitai"},
        repo_spec={"kind": "model", "library_name": ""},
    )


def _hub_accepts(ticks: list[tuple[Any, Any, Any, Any]]) -> list[tuple[str, int]]:
    """The hub's own rule, verbatim: ``step = int(position)``
    (``ParseRequestProgressPayload``), and ``AdvanceJobProgress`` updates the
    row — and with it ``progress_at``, the stall clock — only on a STRICT
    increase."""
    accepted: list[tuple[str, int]] = []
    last: int | None = None
    for _fraction, stage, position, _total in ticks:
        step = int(position) if position is not None else 0
        if last is None or step > last:
            last = step
            accepted.append((str(stage), step))
    return accepted


def test_the_declared_position_advances_through_fetch_and_upload(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    src = _source(tmp_path / "source")

    def _fake_ingest(_version_id: int, _dest: Path, **kwargs: Any) -> IngestedSource:
        report = kwargs.get("progress")
        if report is not None:
            for tenth in range(11):
                report(int(_JUGGERNAUT_BYTES * tenth / 10), _JUGGERNAUT_BYTES)
        return src

    monkeypatch.setattr("gen_worker.convert.clone.ingest_civitai", _fake_ingest)
    monkeypatch.setattr("gen_worker.convert.clone.plan_civitai", lambda *a, **k: None)

    ctx = _Ctx(fake_hub)
    run_clone(
        ctx, provider="civitai", civitai_model_version_id=1759168,
        destination_repo="tensorhub/juggernaut-xl", destination_release="r1",
        target_layout="single-file",
        outputs=[{"dtype": "fp32", "file_layout": "single-file",
                  "file_type": "safetensors"}],
    )

    accepted = _hub_accepts(ctx.ticks)
    # Nothing is dropped: every tick this clone emits moves the clock.
    assert len(accepted) == len(ctx.ticks)
    phases = [stage for stage, _ in accepted]
    assert phases[0] == "clone.plan"
    assert phases.count("clone.download") == 10
    # The UPLOAD advances it too — the other leg that outlasts the budget on a
    # real-sized model.
    assert any(stage.startswith("clone.publish.") for stage, _ in accepted)
    assert [step for _, step in accepted] == sorted({step for _, step in accepted})
    # MiB moved, so the numbers an operator reads are the model's own size.
    download_end = max(step for stage, step in accepted if stage == "clone.download")
    assert download_end == pytest.approx(_JUGGERNAUT_BYTES // (1024 * 1024), rel=0.01)


def test_a_pickle_only_repo_refuses_by_its_own_name() -> None:
    """`nitrosocke/mo-di-diffusion`: 5 weight files, every one a pickle."""
    with pytest.raises(RepoRefusal) as refusal:
        classify_repo(files=[
            "model_index.json", "README.md", "moDi-v1-pruned.ckpt",
            "unet/config.json", "unet/diffusion_pytorch_model.bin",
            "vae/config.json", "vae/diffusion_pytorch_model.bin",
            "text_encoder/config.json", "text_encoder/pytorch_model.bin",
            "safety_checker/pytorch_model.bin",
            "scheduler/scheduler_config.json",
        ], config_json=None)
    assert refusal.value.reason == "pickle_only"
    assert refusal.value.files_seen and all(
        f.endswith((".bin", ".ckpt")) for f in refusal.value.files_seen)

    # And the OTHER fact keeps its own token: no weights at all is not the
    # same thing as weights we will never load.
    with pytest.raises(RepoRefusal) as empty:
        classify_repo(files=["model_index.json", "unet/config.json"],
                      config_json=None)
    assert empty.value.reason == "missing_safetensors"


def test_an_unproducible_layout_is_refused_before_the_download() -> None:
    """The 6.9 GB refusal, decided from one metadata call."""
    plan = CivitaiSourcePlan(
        version_id=1759168,
        payload={"baseModel": "SDXL 1.0", "modelId": 133005},
        files=[{"name": "juggernautXL.safetensors",
                "size_bytes": _JUGGERNAUT_BYTES}],
        revision="sha256:x",
    )
    with pytest.raises(ValidationError) as refused:
        refuse_unproducible_layout(plan, normalize_outputs([], layout_hint="multi-file"))
    assert "single-file->multi-file" in str(refused.value)
    assert "before the download" in str(refused.value)

    # What CivitAI sources ARE runs, and is not second-guessed here.
    refuse_unproducible_layout(plan, normalize_outputs([], layout_hint="single-file"))
