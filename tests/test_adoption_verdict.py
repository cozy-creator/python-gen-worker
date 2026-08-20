"""pgw#1564: a boot that arms ZERO claimed graphs says so, WITH the reasons.

The field shape this encodes (2026-08-20 09:41, sd15, `gen-worker up -d`):
adoption claimed 14 graphs and holed all 14 — each Hole carrying
`cannot decompress` (pgw#1561's bare-ZIP blobs) — and NOT ONE LINE reached
the resident log, because the adopt summary sat at INFO (`up` surfaces
WARNING+ only) and hole REASONS were logged nowhere at any level. The zero
was then investigated for hours as a brand-new silent-adoption defect. A
zero that cannot go red is the disease; these tests are the cure's red arm.

Same fixture ecosystem as ``test_serving_adopt_first``: the real endpoint
fixture, a real ``LocalGraphStore``, real publish-time discovery — and for
the red arm deliberately NO loader stub, because the reason under test is
produced by the REAL loader refusing real unloadable bytes (torchcg's
``materialize`` raises on a ZIP long before any GPU work).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tcg_artifacts
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker._vendor.torchcg.requirements import RequirementsManifest
from gen_worker._vendor.torchcg.store import LocalGraphStore
from gen_worker.cli import daemon as daemon_mod
from gen_worker.serving import DeployBinding, EndpointHost, load_endpoint

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
LANE = "sdxl.diffusers-bf16@1"
SM = "sm_89"
STACK: tuple[tuple[str, str], ...] = (("torch", torch.__version__),)
ENV = EnvIdentity(stack=STACK, sm=SM)


import json


def make_binding(tmp_path: Path) -> DeployBinding:
    """The adopt-first suite's binding, built directly (its `binding` is a
    pytest fixture and fixtures do not import)."""
    from test_serving_adopt_first import OVERRIDES

    root = tmp_path / "checkpoint"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"seed": 7, "scheduler": {"prediction_type": "epsilon"}})
    )
    return DeployBinding(
        checkpoint_ref="ckpt:tiny@1", checkpoint_dir=root,
        model="sdxl", defaults=dict(OVERRIDES),
    )


def test_zero_armed_is_a_warning_with_its_hole_reasons(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from test_serving_adopt_first import (  # the suite's own real fixtures
        fresh_host,
        publish_document,
    )

    binding = make_binding(tmp_path)
    host = fresh_host(binding, tmp_path)
    host.setup()  # the eager bridge boot the publish-time derive drives
    document = publish_document(host)

    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    for record in document.lanes[0].graphs:
        # What every pre-pgw#1561 publisher banked: the bare AOTI package.
        bare = tcg_artifacts.aoti_package(
            tmp_path / f"{record.graph[-8:]}.pt2",
            graph_specialization=record.graph,
        )
        store.publish_artifact(
            record.graph, ENV, bare,
            RequirementsManifest(
                include_set=(("torch", torch.__version__),), sm_compiled=SM),
        )

    booted = fresh_host(binding, tmp_path)
    with caplog.at_level(logging.INFO):
        booted.setup(
            store=store, document=document, sm=SM,
            # NO loader stub: the reason under test is the REAL loader's.
            artifacts_dir=tmp_path / "adopted",
            stack=STACK,
        )

    claimed = len(document.lanes[0].graphs)
    assert len(booted.holes) == claimed and not booted.adoption.adopted

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    zero_lines = [r for r in warnings if "ZERO of" in r.getMessage()]
    assert zero_lines, "a declared-compiled boot that armed nothing must WARN"
    assert f"ZERO of {claimed} claimed graph(s) armed" in zero_lines[0].getMessage()
    reason_lines = [r for r in warnings if "hole " in r.getMessage()]
    assert reason_lines, "the WHY must ride the warning, not sit unread on the session"
    assert any("ArtifactFormatSkew" in r.getMessage() for r in reason_lines), (
        "the reason class the real loader raised must be named"
    )

    # And the reasons are handle-visible facts, not log-only prose.
    for hole in booted.holes:
        assert "ArtifactFormatSkew" in hole.reason


def test_an_armed_boot_stays_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Polarity: the WARNING fires only on the zero — an armed boot logs one
    INFO summary, or the alarm becomes noise operators learn to ignore."""
    from test_serving_adopt_first import (
        counting_loader,
        fresh_host,
        manifest,
        publish_document,
    )

    binding = make_binding(tmp_path)
    host = fresh_host(binding, tmp_path)
    host.setup()
    document = publish_document(host)
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    for record in document.lanes[0].graphs:
        blob = tmp_path / f"{record.graph[-8:]}.bin"
        blob.write_bytes(b"compiled")
        store.publish_artifact(record.graph, ENV, blob, manifest())

    booted = fresh_host(binding, tmp_path)
    with caplog.at_level(logging.INFO):
        booted.setup(
            store=store, document=document, sm=SM,
            loader=counting_loader([]),
            artifacts_dir=tmp_path / "adopted",
            stack=STACK,
        )

    assert booted.adoption.adopted, "fixture drift: nothing armed"
    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "ZERO of" in r.getMessage()
    ]


def test_the_handle_carries_hole_reasons_and_the_verdict(tmp_path: Path) -> None:
    """The handle is the ONE thing an out-of-process caller can read after
    boot; names without reasons is what made the field zero reasonless."""
    booted = daemon_mod.Booted(
        host=SimpleNamespace(adoption=SimpleNamespace()),
        loaded=SimpleNamespace(module_name="m", entrypoints={}),
        counter=None,
        checkpoint_dir=tmp_path,
        adopted=(),
        holes=("cg-graph-v1-" + "a" * 56,),
        hole_reasons=(("cg-graph-v1-" + "a" * 56, "ArtifactFormatSkew: bare package"),),
    )
    resident = SimpleNamespace(
        booted=booted,
        spec=SimpleNamespace(endpoint_dir=tmp_path, checkpoint_refs=(),
                             output_dir=tmp_path, sm=SM, lane=LANE),
        handle=SimpleNamespace(socket_path=tmp_path / "sock"),
        _booted_at="", _served=0,
        _primary_fields=lambda: {},
    )
    from typing import Any, cast

    document = daemon_mod.ResidentEndpoint._document(cast(Any, resident), "ready")
    assert document["hole_reasons"] == [
        {"graph": "cg-graph-v1-" + "a" * 56,
         "reason": "ArtifactFormatSkew: bare package"}
    ]
    assert document["adoption"] == {"engaged": True, "armed": 0, "claimed": 1}


def test_the_mint_mints_what_it_was_armed_on_not_a_second_read(
    tmp_path: Path,
) -> None:
    """The 13 ms `completed 0/14` (L4 pod, 2026-08-20): ``arm`` counted holes
    off the LIVE ``host.holes`` property, ``run`` re-read it, the second read
    answered empty, and an empty run settles as COMPLETE — a mint that did
    nothing, declared done, with the status still claiming N holes. The
    work-list is read ONCE at arm; what was armed is what gets minted."""
    from gen_worker.serving.self_mint import SelfMint

    class Vanishing:
        """A host whose hole list is live — and empties after the first read."""

        def __init__(self) -> None:
            self._reads = 0
            self.adoption = SimpleNamespace(
                env=SimpleNamespace(value="lane-a", sm=SM),
                arm=lambda record, artifact: None,
            )

        @property
        def holes(self):  # noqa: ANN201 — a fixture property
            self._reads += 1
            if self._reads > 1:
                return ()
            return tuple(
                SimpleNamespace(record=SimpleNamespace(graph=f"g{i}", target="unet"))
                for i in range(3)
            )

    compiled: list[str] = []

    def compiler(blob: Path, record: SimpleNamespace, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"artifact")
        compiled.append(record.graph)
        return destination

    def program_source(graph: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"program")
        return destination

    box = SelfMint(
        store=None, artifacts_dir=tmp_path / "artifacts",
        compiler=compiler, program_source=program_source, vcpus=2,
    )
    armed = box.arm(Vanishing())
    assert armed.holes == 3
    final = box.join(30.0)
    assert sorted(compiled) == ["g0", "g1", "g2"], (
        "the mint must mint the ARMED list, not a second read of a live "
        "property that already emptied"
    )
    assert final.landed == 3 and not final.running
