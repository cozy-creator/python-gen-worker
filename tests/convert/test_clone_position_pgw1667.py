"""What a clone SAYS while it is planning, and what a wedged one still says.

pgw#1667, measured on pod `mb5zw02csftwjx`: job `01a02910` mirroring
`sensenova/SenseNova-U1.5-8B-MoT-Preview` (13 shards, 50.19 GB) died
`job_progress_stalled` / `zero_progress` at 604 GPU-s — *"declared position
clone.plan 0 stopped advancing 10m4s ago"* — without downloading a weight byte.
`plan_huggingface` took no `progress=` at all, so the phase's position was 0 by
construction while it walked the repo tree and read safetensors headers, and the
hub's 10-minute budget is the only thing that could happen next.

The watchdog is right and is not touched here: a position that does not move is
the only evidence of a wedged job that a heartbeat cannot fake. The producer owes
a position, so every phase now declares one from real work units.

TWO REGRESSIONS, one test file. `cd46c957` (the v1 SDK hardcut) deleted
pgw#1397's position machinery out of `run_clone` and its test in the same commit
— so `clone.download` and `clone.publish` had silently gone back to declaring
nothing either, and the next large mirror would have died in the download for
exactly the same reason. Both halves are asserted below.

Revert-turns-red: drop `progress=` from `plan_huggingface` and the first test
sees ONE position for the whole plan; restore `fn(p, stage=stage)` in
`run_clone`'s emitter and the second sees `clone.plan 0` and nothing else, which
is the failed job's message verbatim.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from gen_worker.convert.clone import run_clone
from gen_worker.convert.ingest import (
    IngestedSource,
    plan_civitai,
    plan_huggingface,
)

from fake_hub import _FakeHub

#: SenseNova-U1.5-8B-MoT-Preview, as the failed job saw it.
SHARDS = 13
SOURCE_BYTES = 50_190_000_000


# ---------------------------------------------------------------------------
# The hub's own rules, replayed over what reached `ctx.progress`.
# ---------------------------------------------------------------------------

Tick = tuple[float, Any, Any, Any, Any]


def _accepted(ticks: list[Tick]) -> list[tuple[float, str, int]]:
    """`ParseRequestProgressPayload` takes `step = int(position)`, and
    `AdvanceJobProgress` updates the row — and with it `progress_at`, the stall
    clock — only on a STRICT increase. Every other tick is dropped."""
    out: list[tuple[float, str, int]] = []
    last: Optional[int] = None
    for at, _fraction, stage, position, _total in ticks:
        step = int(position) if position is not None else 0
        if last is None or step > last:
            last = step
            out.append((at, str(stage), step))
    return out


def _longest_silence(ticks: list[Tick], until: float) -> float:
    """The longest stretch with no accepted advance — what the `zero_progress`
    budget is compared against."""
    marks = [at for at, _stage, _step in _accepted(ticks)]
    if not marks:
        return until - (ticks[0][0] if ticks else until)
    return max(
        max((b - a for a, b in zip(marks, marks[1:])), default=0.0),
        until - marks[-1],
    )


class _Ctx:
    """Records what reached `ctx.progress`, with the arrival time."""

    def __init__(self, server: Any = None) -> None:
        if server is not None:
            self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "tensorhub"
        self.request_id = "req-1667"
        self.destination = {"repo": "tensorhub/fallback"}
        self.ticks: list[Tick] = []

    def progress(self, progress: Any = None, stage: Any = None, *,
                 step: Any = None, total: Any = None, position: Any = None,
                 phase: Any = None) -> None:
        self.ticks.append(
            (time.monotonic(), progress, stage or phase, position, total))


# ---------------------------------------------------------------------------
# A 13-shard transformers source, served entirely from local bytes.
# ---------------------------------------------------------------------------

def _safetensors_bytes(params: int) -> bytes:
    header = {"blocks.0.weight": {
        "dtype": "BF16", "shape": [params], "data_offsets": [0, params * 2]}}
    blob = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob + b"\x00" * (params * 2)


def _sensenova_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "remote"
    remote.mkdir(parents=True)
    (remote / "config.json").write_text(
        json.dumps({"architectures": ["SenseNovaForCausalLM"],
                    "model_type": "sensenova"}), encoding="utf-8")
    (remote / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "README.md",
                 "generation_config.json"):
        (remote / name).write_text("{}", encoding="utf-8")
    for i in range(1, SHARDS + 1):
        (remote / f"model-{i:05d}-of-{SHARDS:05d}.safetensors").write_bytes(
            _safetensors_bytes(4))
    return remote


def _fake_hf(remote: Path) -> Any:
    files = sorted(p for p in remote.rglob("*") if p.is_file())

    class _Api:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def repo_info(self, repo_id: str, revision: str | None = None) -> Any:
            return SimpleNamespace(sha="a" * 40)

        def list_repo_tree(self, repo_id: str, revision: str | None = None,
                           recursive: bool = False) -> Any:
            for p in files:
                rel = p.relative_to(remote).as_posix()
                size = (SOURCE_BYTES // SHARDS
                        if rel.endswith(".safetensors") else p.stat().st_size)
                yield SimpleNamespace(
                    path=rel, size=size,
                    lfs=SimpleNamespace(sha256=f"{abs(hash(rel)):064x}"[:64]),
                    blob_id="")

        def parse_safetensors_file_metadata(
            self, repo_id: str, filename: str, *, revision: str | None = None,
            repo_type: str | None = None, token: str | None = None,
        ) -> Any:
            raw = (remote / filename).read_bytes()
            (n,) = struct.unpack("<Q", raw[:8])
            header = json.loads(raw[8:8 + n])
            counts: dict[str, int] = {}
            for value in header.values():
                params = 1
                for dim in value.get("shape") or []:
                    params *= int(dim)
                counts[str(value["dtype"])] = (
                    counts.get(str(value["dtype"]), 0) + params)
            return SimpleNamespace(parameter_count=counts, metadata={})

    return SimpleNamespace(
        HfApi=_Api,
        hf_hub_download=lambda repo_id, filename, revision=None, token=None:
            str(remote / filename),
        get_safetensors_metadata=lambda *a, **k: SimpleNamespace(
            files_metadata={}),
        ModelCard=SimpleNamespace(
            load=lambda *a, **k: SimpleNamespace(
                data=SimpleNamespace(tags=["text-generation"]))),
        snapshot_download=lambda *a, **k: "",
    )


# ---------------------------------------------------------------------------
# 1. The plan phase declares a position, per real work unit.
# ---------------------------------------------------------------------------

def test_the_plan_advances_its_position_once_per_remote_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured defect. Pre-fix `plan_huggingface` takes no `progress=`, so
    the whole walk of a 13-shard repo declares nothing."""
    fake = _fake_hf(_sensenova_remote(tmp_path))
    monkeypatch.setattr("gen_worker.convert.ingest.hf", lambda: fake)

    seen: list[int] = []
    plan = plan_huggingface(
        "sensenova/SenseNova-U1.5-8B-MoT-Preview",
        progress=lambda done, _total: seen.append(done),
    )

    assert len(plan.paths) == SHARDS + 6
    # One per enumerated file, plus the identity call, the config fetch, the
    # metadata read, the model card and the header reads. The point is not the
    # exact number — it is that it is not ONE.
    assert len(seen) >= len(plan.paths)
    assert seen == sorted(seen) and seen[0] == 1 and len(set(seen)) == len(seen)


def test_plan_civitai_reports_on_the_same_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other clone source strategy — the sweep, not a special case. There
    are exactly two `plan_*` strategies in this repo and both now report."""
    payload = {"id": 1759168, "baseModel": "SDXL 1.0", "files": [
        {"name": "juggernautXL.safetensors", "type": "Model",
         "sizeKB": 6_776_582, "hashes": {"SHA256": "ab" * 32},
         "primary": True}]}
    monkeypatch.setattr("gen_worker.convert.ingest.fetch_civitai_model_version",
                        lambda _v, api_key="": payload)

    seen: list[int] = []
    plan = plan_civitai(
        1759168, progress=lambda done, _total: seen.append(done))
    assert plan.version_id == 1759168
    assert seen == [1, 2]


# ---------------------------------------------------------------------------
# 2. Every phase of a real clone declares one, and the hub accepts every tick.
# ---------------------------------------------------------------------------

def test_a_clone_declares_an_advancing_position_from_plan_to_upload(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1397's proof, restored: `cd46c957` deleted the machinery and the
    test together, and every phase went back to declaring `0`."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    fake = _fake_hf(_sensenova_remote(tmp_path))
    monkeypatch.setattr("gen_worker.convert.ingest.hf", lambda: fake)

    local = tmp_path / "source"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}", encoding="utf-8")
    (local / "model.safetensors").write_bytes(b"\x00" * (300 * 1024 * 1024))
    source = IngestedSource(
        provider="huggingface",
        source_ref="sensenova/SenseNova-U1.5-8B-MoT-Preview",
        source_revision="a" * 40, dir=local, layout="single-file",
        model_family="unknown", model_family_variant="unknown",
        classification=None,
        attrs={"dtype": "bf16", "file_layout": "single-file"},
        metadata={"source_provider": "huggingface"},
        repo_spec={"kind": "model", "library_name": ""},
    )

    def _fake_ingest(_ref: str, _dest: Path, **kwargs: Any) -> IngestedSource:
        report = kwargs.get("progress")
        if report is not None:
            for tenth in range(11):
                report(int(SOURCE_BYTES * tenth / 10), SOURCE_BYTES)
        return source

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface",
                        _fake_ingest)
    monkeypatch.setattr("gen_worker.convert.clone._preflight_disk",
                        lambda *a, **k: None)

    ctx = _Ctx(fake_hub)
    run_clone(
        ctx, provider="huggingface",
        source_ref="sensenova/SenseNova-U1.5-8B-MoT-Preview",
        destination_repo="tensorhub/sensenova-u1", destination_release="r1",
        target_layout="single-file",
        outputs=[{"dtype": "bf16", "file_layout": "single-file",
                  "file_type": "safetensors"}],
    )

    accepted = _accepted(ctx.ticks)
    # Nothing the clone says is dropped by the hub's strict-increase predicate.
    assert len(accepted) == len(ctx.ticks)
    phases = [stage for _at, stage, _step in accepted]
    # THE FILED DEFECT: the plan is not one row at 0 any more.
    assert phases.count("clone.plan") > 1
    assert phases.count("clone.download") == 10
    # THE SILENT REGRESSION beside it: the upload advances the same position.
    assert any(p.startswith("clone.publish.") for p in phases)
    steps = [step for _at, _stage, step in accepted]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
    # MiB moved, so an operator reads the model's own size.
    download_end = max(step for _at, stage, step in accepted
                       if stage == "clone.download")
    assert download_end == pytest.approx(SOURCE_BYTES // (1024 * 1024), rel=0.01)


# ---------------------------------------------------------------------------
# 3. The watchdog stays armed: a plan that genuinely hangs still reads as one.
# ---------------------------------------------------------------------------

_BUDGET_S = 0.4
_HANG_S = 1.5


def test_a_genuinely_hung_plan_still_reads_as_zero_progress(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix must not buy liveness with a counter that ticks on its own.

    Same `run_clone`, same emitter: a plan wedged on an unreachable Hugging Face
    declares nothing — because nothing happened — and the hub's rule kills it,
    exactly as it killed the real job. The healthy source in the test above,
    replayed against the SAME budget, is silent for a small fraction of it.
    """
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))

    def _wedged(*_a: Any, **kwargs: Any) -> Any:
        # It is HANDED a `progress=` and never calls it, which is what an
        # unreachable remote looks like from in here.
        assert kwargs.get("progress") is not None
        time.sleep(_HANG_S)
        raise TimeoutError("huggingface unreachable")

    monkeypatch.setattr("gen_worker.convert.clone.plan_huggingface", _wedged)
    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("the plan phase is the subject")))

    ctx = _Ctx(fake_hub)
    with pytest.raises(RuntimeError):
        run_clone(
            ctx, provider="huggingface",
            source_ref="sensenova/SenseNova-U1.5-8B-MoT-Preview",
            destination_repo="tensorhub/sensenova-u1",
            destination_release="r1", target_layout="single-file",
            outputs=[{"dtype": "bf16", "file_layout": "single-file",
                      "file_type": "safetensors"}],
        )

    plan_ticks = [t for t in ctx.ticks if str(t[2]) == "clone.plan"]
    assert len(plan_ticks) == 1, "a wedged plan must not manufacture ticks"
    assert _longest_silence(plan_ticks, time.monotonic()) > _BUDGET_S
