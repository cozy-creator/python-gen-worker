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


def _sample(compiled: int, eager: int, displaced: tuple = ()) -> Any:
    return harness.Sample(
        arm="static", aspect="1:1", cfg="on", seconds=1.0,
        compiled_calls=compiled, eager_calls=eager, displaced=displaced,
    )


def test_an_arm_serving_ZERO_compiled_calls_REFUSES(tmp_path: Path) -> None:
    """pgw#1591's lesson, keyed on the COUNT rather than on a log string."""

    bench = _bench(tmp_path)
    bench._daemon_log = {}
    with pytest.raises(SystemExit, match="ZERO compiled calls"):
        bench.assert_compiled("static", _sample(compiled=0, eager=21))


def test_MIXED_execution_REFUSES(tmp_path: Path) -> None:
    """Part compiled, part eager is not this axis's cost."""

    bench = _bench(tmp_path)
    bench._daemon_log = {}
    with pytest.raises(SystemExit, match="MIXED execution"):
        bench.assert_compiled("static", _sample(compiled=10, eager=11))


def test_a_DISPLACED_module_that_STILL_SERVED_COMPILED_is_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """pgw#1591's fix: displacement and dispatch are SEPARATE facts.

    The old check aborted on the word DISPLACED alone. After #1591 a displaced
    module can still have served every call compiled, and refusing that would
    throw away a valid arm for a flag that no longer implies what it used to.
    """

    bench = _bench(tmp_path)
    bench._daemon_log = {}
    bench.assert_compiled("static", _sample(compiled=21, eager=0, displaced=("UNet",)))
    assert "still served compiled" in capsys.readouterr().out


def test_a_clean_compiled_arm_PASSES(tmp_path: Path) -> None:
    bench = _bench(tmp_path)
    bench._daemon_log = {}
    bench.assert_compiled("static", _sample(compiled=21, eager=0))


def test_an_envelope_without_dispatch_facts_REFUSES(tmp_path: Path) -> None:
    """No counter means the premise is unmeasured — the pgw#1591 near-miss."""

    bench = _bench(tmp_path)
    with pytest.raises(SystemExit, match="no `dispatch` facts"):
        bench._dispatch_facts("static", '{"ok": true, "result": {}}')


def test_the_facts_are_read_off_the_real_envelope_shape(tmp_path: Path) -> None:
    """The shape `gen-worker run --json` actually prints."""

    envelope = {
        "ok": True,
        "dispatch": {
            "module_calls": 21, "compiled_graph_calls": 21, "eager_calls": 0,
            "armed_modules": 1, "armed_graphs": 14, "displaced_modules": [],
        },
    }
    facts = _bench(tmp_path)._dispatch_facts("static", json.dumps(envelope))
    assert facts["compiled_graph_calls"] == 21
