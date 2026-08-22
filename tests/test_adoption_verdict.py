from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tcg_artifacts
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker.graphs.env import ArtifactEnv as EnvIdentity
from gen_worker.graphs.requirements import RequirementsManifest
from gen_worker.graphs.store import LocalGraphStore
from gen_worker.cli import daemon as daemon_mod
from gen_worker.serving import DeployBinding, EndpointHost, load_endpoint

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
LANE = "sdxl.diffusers@1+plain.bf16@1"
SM = "sm_89"
STACK: tuple[tuple[str, str], ...] = (("torch", torch.__version__),)
ENV = EnvIdentity(stack=STACK, sm=SM)


import json


def make_binding(tmp_path: Path) -> DeployBinding:
    """The adopt-first suite's binding, built directly (its `binding` is a pytest fixture and fixtures do not import)."""
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
    from test_serving_adopt_first import (
        fresh_host,
        publish_document,
    )

    binding = make_binding(tmp_path)
    host = fresh_host(binding, tmp_path)
    host.setup()
    document = publish_document(host)

    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    for record in document.lanes[0].graphs:
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

    for hole in booted.holes:
        assert "ArtifactFormatSkew" in hole.reason


def test_an_armed_boot_stays_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Polarity: the WARNING fires only on the zero — an armed boot logs one INFO summary, or the alarm becomes noise operators learn to ignore."""
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
    """The handle is the ONE thing an out-of-process caller can read after boot; names without reasons is what made the field zero reasonless."""
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
    from gen_worker.serving.self_mint import SelfMint

    from gen_worker.serving.mint_store import graph_store

    graphs = [
        "cg-graph-v1-" + f"{index:056x}".replace("x", "0")
        for index in range(3)
    ]

    class Vanishing:
        """A host whose hole list is live — and empties after the first read."""

        def __init__(self) -> None:
            self._reads = 0
            self.adoption = SimpleNamespace(
                env=ENV, arm=lambda record, artifact: None,
            )

        @property
        def holes(self):  # noqa: ANN201 — a fixture property
            self._reads += 1
            if self._reads > 1:
                return ()
            return tuple(
                SimpleNamespace(record=SimpleNamespace(graph=graph, target="unet"))
                for graph in graphs
            )

    compiled: list[str] = []

    def compiler(blob: Path, record: SimpleNamespace, destination: Path) -> Path:
        compiled.append(record.graph)
        return tcg_artifacts.unpacked(
            destination, graph_specialization=record.graph, sm=SM)

    def program_source(graph: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"program")
        return destination

    box = SelfMint(
        store=graph_store(tmp_path / "cas", None, tmp_path / "no-baked"),
        artifacts_dir=tmp_path / "artifacts",
        compiler=compiler, program_source=program_source, vcpus=2,
    )
    armed = box.arm(Vanishing())
    assert armed.holes == 3
    final = box.join(60.0)
    assert sorted(compiled) == sorted(graphs), (
        "the mint must mint the ARMED list, not a second read of a live "
        "property that already emptied"
    )
    assert final.landed == 3 and not final.running


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> "list[tuple]":
    from gen_worker import activity as activity_mod

    seen: "list[tuple]" = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, phase="", **kw: seen.append((kind, phase, detail, kw)))
    return seen


def test_the_armed_zero_verdict_is_a_durable_row_not_a_log_line(
    tmp_path: Path, wire: "list[tuple]", caplog: pytest.LogCaptureFixture
) -> None:
    import tcg_artifacts
    from gen_worker.graphs.requirements import RequirementsManifest
    from test_serving_adopt_first import fresh_host, publish_document

    binding = make_binding(tmp_path)
    host = fresh_host(binding, tmp_path)
    host.setup()
    document = publish_document(host)
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    for record in document.lanes[0].graphs:
        bare = tcg_artifacts.aoti_package(
            tmp_path / f"{record.graph[-8:]}.pt2",
            graph_specialization=record.graph)
        store.publish_artifact(
            record.graph, ENV, bare,
            RequirementsManifest(
                include_set=(("torch", torch.__version__),), sm_compiled=SM))

    booted = fresh_host(binding, tmp_path)
    with caplog.at_level(logging.WARNING):
        booted.setup(store=store, document=document, sm=SM,
                     artifacts_dir=tmp_path / "adopted", stack=STACK)

    rows = [row for row in wire if row[1] == "armed_zero"]
    assert len(rows) == 1, wire
    _kind, _phase, detail, counts = rows[0]
    assert "armed ZERO" in detail and "ArtifactFormatSkew" in detail
    assert counts["step"] == 0 and counts["total_steps"] == len(
        document.lanes[0].graphs)


def test_the_mint_terminal_verdict_rides_the_wire_with_its_identity(
    tmp_path: Path, wire: "list[tuple]",
) -> None:
    from gen_worker.serving.mint_store import graph_store
    from gen_worker.serving.self_mint import SelfMint

    compiled: "list[str]" = []
    graphs = ["cg-graph-v1-" + f"{2 + index:056d}" for index in range(2)]

    def compiler(blob: Path, record: SimpleNamespace, destination: Path) -> Path:
        compiled.append(record.graph)
        return tcg_artifacts.unpacked(
            destination, graph_specialization=record.graph, sm=SM)

    def program_source(graph: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"program")
        return destination

    host = SimpleNamespace(
        holes=tuple(
            SimpleNamespace(record=SimpleNamespace(graph=graph, target="unet"))
            for graph in graphs),
        adoption=SimpleNamespace(
            env=ENV, arm=lambda record, artifact: None),
    )
    box = SelfMint(
        store=graph_store(tmp_path / "cas", None, tmp_path / "no-baked"),
        artifacts_dir=tmp_path / "artifacts",
        compiler=compiler, program_source=program_source, vcpus=2)
    box.arm(host)
    final = box.join(30.0)
    assert final.landed == 2

    terminal = [row for row in wire if row[1].startswith("terminal_")]
    assert len(terminal) == 1, wire
    _kind, phase, detail, counts = terminal[0]
    assert phase == "terminal_complete"
    assert "gen-worker " in detail and "contract " in detail, (
        "the executed code identity must ride the verdict — 'pin inferred "
        "from the build chain' is what this row exists to end")
    assert "DIVERGENT" not in detail
    assert counts["step"] == 2 and counts["total_steps"] == 2


def test_a_divergent_work_list_is_named_on_the_verdict(
    tmp_path: Path, wire: "list[tuple]",
) -> None:
    """The 13/23 ms no-op class, as ONE named fact: the run processed fewer holes than were armed."""
    from gen_worker.serving.mint import MintOutcome
    from gen_worker.serving.self_mint import SelfMint

    box = SelfMint(
        store=None, artifacts_dir=tmp_path,
        compiler=lambda blob, record, destination: Path(destination),
        program_source=lambda graph, destination: Path(destination),
    )
    box._settle(MintOutcome(holes=0, width=0, elapsed_s=0.001), "", 14)

    terminal = [row for row in wire if row[1].startswith("terminal_")]
    assert len(terminal) == 1
    detail = terminal[0][2]
    assert "DIVERGENT WORK-LIST: armed 14 hole(s), the run processed 0" in detail
