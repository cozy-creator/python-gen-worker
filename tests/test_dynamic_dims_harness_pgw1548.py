"""pgw#1548: the benchmark harness reads a REAL lock, and says so when it can't.

Caught by running the reader against a real `endpoint.lock` before the GPU
window rather than during it: `endpoint_lock.read_lock` answers a plain **dict**
(not an object with attributes) and the derive document sits under
`["derive"]["document"]` as a JSON **string**. The harness had `block.document`,
which would have raised on the pod — minutes into a paid window, after the
compile it was supposed to plan.

The lesson is the test: every structural assumption the harness makes about
another module's return value is asserted at $0, because the alternative is
asserting it at $0.49/hr.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "benchmarks" / "dynamic_dims_pgw1548.py"
_spec = importlib.util.spec_from_file_location("dynamic_dims_pgw1548", _MODULE)
assert _spec and _spec.loader
harness = importlib.util.module_from_spec(_spec)
# Registered BEFORE exec: dataclass field resolution reads the module out of
# sys.modules, and an unregistered module fails deep inside `dataclasses`.
sys.modules["dynamic_dims_pgw1548"] = harness
_spec.loader.exec_module(harness)


def _bench(tmp_path: Path) -> Any:
    args = argparse.Namespace(
        endpoint=str(tmp_path), out=str(tmp_path / "out"), substrate="local"
    )
    bench = harness.Bench.__new__(harness.Bench)
    bench.args = args
    bench.endpoint = Path(args.endpoint)
    bench.out = Path(args.out)
    return bench


def _lock(room: Path, records: list[dict], *, derive: bool = True) -> Path:
    """A lock in the REAL shape: TOML, with derive.document a JSON STRING.

    Both facts matter and both were assumed wrong at some point: the file is
    TOML (not JSON), and the document inside it is a JSON string (not a
    table).
    """

    room.mkdir(parents=True, exist_ok=True)
    lines = ["[[entrypoints]]", 'name = "generate"', ""]
    if derive:
        document = {"graphs": {"lanes": [{"contract": "x@1", "graphs": records}]}}
        encoded = json.dumps(json.dumps(document))
        lines += [
            "[derive]",
            "v = 1",
            f'document = {encoded}',
            f'document_digest = "{"d" * 64}"',
            'endpoint = "x:Y"',
            'interface_v = 4',
            'inputs_digest = "%s"' % ("e" * 64),
            'trace_device = "cuda"',
        ]
    (room / "endpoint.lock").write_text("\n".join(lines) + "\n")
    return room


def _record(graph: str, shape: list) -> dict:
    return {
        "graph": graph,
        "target": "unet",
        "ingress": {"inputs": [{"name": "sample", "shape": shape}]},
    }


def test_the_reader_gets_every_specialization_out_of_a_real_shaped_lock(
    tmp_path: Path,
) -> None:
    room = _lock(tmp_path / "arm", [
        _record("cg-graph-v1-" + "a" * 52, [2, 4, 64, 64]),
        _record("cg-graph-v1-" + "b" * 52, [1, 4, 64, 64]),
    ])
    records = _bench(tmp_path).specializations(room)
    assert [r["graph"][-16:] for r in records] == ["a" * 16, "b" * 16]


def test_a_discovery_only_lock_REFUSES_instead_of_reading_zero(
    tmp_path: Path,
) -> None:
    """`--discovery-only` writes no [derive]; benchmarking it is meaningless.

    Silently reading zero specializations would compile nothing, serve eager
    and report a 0% delta — the same vacuous green the preflight exists to
    stop, arriving from the lock side instead of the tree side.
    """

    room = _lock(tmp_path / "arm", [], derive=False)
    with pytest.raises(SystemExit, match="no .derive. document"):
        _bench(tmp_path).specializations(room)


def test_a_lock_declaring_zero_graphs_REFUSES(tmp_path: Path) -> None:
    room = _lock(tmp_path / "arm", [])
    with pytest.raises(SystemExit, match="zero specializations"):
        _bench(tmp_path).specializations(room)


def test_the_default_selectors_are_PREFIXES_the_compiler_can_match(
    tmp_path: Path,
) -> None:
    """`--first` matches a facet by equality or the graph id by PREFIX.

    `compile.Spec.short` is `graph[:16]` — scheme included — and `_selects`
    does `spec.graph.startswith(term)`. A suffix matches neither, so a harness
    passing one gets "names no specialization this endpoint has" for every
    arm, on the pod, inside the paid window. This test fails if the harness
    ever goes back to a suffix.
    """

    graph = "cg-graph-v1-" + "a" * 52
    room = _lock(tmp_path / "arm", [_record(graph, [2, 4, 64, 64])])
    records = _bench(tmp_path).specializations(room)
    selector = records[0]["graph"][:16]
    assert selector == "cg-graph-v1-aaaa"
    assert graph.startswith(selector), "the compiler matches by prefix"
    assert len(selector) >= 8, "eight characters minimum, so a word cannot collide"
    assert not graph.startswith(records[0]["graph"][-16:]), (
        "a suffix does NOT match — this is the bug the prefix fixes"
    )


def test_the_substrate_note_is_carried_into_every_rendered_table() -> None:
    table = harness.Table()
    for round_index in range(2):
        table.add(harness.Sample("static", "1:1", "on", 1.0, round=round_index))
        table.add(harness.Sample("aspect", "1:1", "on", 1.0, round=round_index))
    rendered = table.render("raw-pod")
    assert "not the deploy path" in rendered.splitlines()[0]
    assert "| 1:1 | on |" in rendered


def _armed(tmp_path: Path, arm: str, text: str) -> Any:
    """A bench whose named arm has a daemon log carrying `text`."""

    log = tmp_path / f"{arm}-daemon.log"
    log.write_text(text)
    bench = _bench(tmp_path)
    bench._daemon_log = {arm: log}
    return bench


def test_a_DISPLACED_arm_REFUSES_to_contribute_a_row(tmp_path: Path) -> None:
    """pgw#1591: two displaced arms compare to ~0% and read as 'free'.

    This is the whole reason it is an abort and not a warning — the numbers a
    displaced arm produces are perfectly plausible and entirely meaningless.
    """

    bench = _armed(tmp_path, "static", (
        "wrapper.tcg run_impl\n"
        "dispatch: DISPLACED on UNet2DConditionModel — the compiled dispatcher "
        "is no longer this module's forward, so all 21 call(s) ran eager\n"
    ))
    with pytest.raises(SystemExit, match="DISPLACED"):
        bench.assert_compiled("static")


def test_an_arm_that_MATCHED_NOTHING_refuses(tmp_path: Path) -> None:
    """tcg#76's trace means the guards refused every armed record."""

    bench = _armed(tmp_path, "aspect",
                   "torchcg dispatch: NO armed graph matched a call on UNet\n")
    with pytest.raises(SystemExit, match="matched NOTHING"):
        bench.assert_compiled("aspect")


def test_an_arm_with_NO_compiled_wrapper_at_all_refuses(tmp_path: Path) -> None:
    """Silence is not success: no wrapper ran, so nothing was shown."""

    bench = _armed(tmp_path, "static", "ready — functions: generate\n")
    with pytest.raises(SystemExit, match="not been shown to serve compiled"):
        bench.assert_compiled("static")


def test_an_absent_daemon_log_refuses_rather_than_assuming(tmp_path: Path) -> None:
    bench = _bench(tmp_path)
    bench._daemon_log = {"static": tmp_path / "nope.log"}
    with pytest.raises(SystemExit, match="absent"):
        bench.assert_compiled("static")


def test_a_genuinely_compiled_arm_PASSES(tmp_path: Path) -> None:
    """The green arm: a wrapper ran, nothing displaced, nothing unmatched."""

    bench = _armed(tmp_path, "static", (
        "gen-worker up: ready\n"
        "[W] cw54nq...wrapper.tcg.cpp:26541 Warning: run_impl\n"
    ))
    bench.assert_compiled("static")
