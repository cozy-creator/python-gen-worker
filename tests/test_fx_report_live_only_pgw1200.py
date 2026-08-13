"""pgw#1200: the FX-cache failure report stops claiming a discrimination it
cannot make.

`fx_cache_failure_report` classified a failed store-served warmup proof into
three named classes, and the classification is what the caller reads:

* ``fresh_keys>0``  => the boot computed DIFFERENT keys from the compiled graph (B1),
* ``fresh_keys=0`` with same-key re-saves => the keys MATCHED and the miss is
  in torch's candidate-load path (B2),
* ``compiled_graph_keys=0``  => the artifact was unreadable.

All three are differences measured against a COMPILED GRAPH side that the function read
out of a `torch-inductor-cache` tarball's `inductor/fxgraph/` tree. pgw#1178
deleted that format's last writer and pgw#1181 deleted the format, so the tar
walk can only ever yield nothing — and with an empty compiled graph side the arithmetic
does not degrade gracefully, it INVERTS:

* `fresh = live_keys - seeded_names` becomes EVERY live key, so `fresh_keys>0`
  fires on every boot that has any FX compiled graph at all. The report names B1 —
  "the boot computed different keys" — always, whatever happened.
* `samekey_resaves` is computed only for keys the compiled graph seeded, so it is
  structurally 0 and B2 is unreportable.
* `compiled_graph_keys=0`, the "unreadable artifact" class, is now the normal case.

A diagnostic that always names one class is worse than no diagnostic: it is
read as evidence. §4.35's amendment decides the disposition — the compiled graph half
cannot be switched ON (there is no format to read) and finishing it would mean
resurrecting the format, so it is DELETED, and what remains is the live-cache
census the dynamo lane can actually observe.

The call site was already scoped correctly (`if proves_inductor and ...`,
pgw#722 finding 2: "FX forensics describe the dynamo lane only"), which is why
this went unnoticed — the branch is reachable, its yield is not.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, List

import pytest

from gen_worker import compile_cache as cc


class _FakeFxCompiledGraph:
    """What torch pickles into an fxgraph cache compiled graph, at the two attributes
    the report reads."""

    def __init__(self, guards: Any = None, extern: str = "") -> None:
        self.guards_expr = guards
        self.extern_libs_key = extern


def _write_live_compiled_graph(root: Path, key: str, name: str = "compiled_graph.bin") -> Path:
    d = root / "fxgraph" / key[1:3] / key
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(pickle.dumps(_FakeFxCompiledGraph(guards="L['x'].size()[0] == 2")))
    return p


def test_the_report_still_runs_and_names_the_live_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half that survives: on a dynamo boot whose warmup proof failed, the
    report says how many FX keys this boot actually has."""
    live = tmp_path / "live"
    _write_live_compiled_graph(live, "fkey00000001")
    _write_live_compiled_graph(live, "fkey00000002")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(live))

    report = cc.fx_cache_failure_report()
    assert "live_keys=2" in report
    assert "extern_current=" in report


def test_a_missing_cache_dir_is_named_not_silently_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguishing "no cache directory" from "a directory with no compiled graphs"
    is the whole reason this returns a report instead of a count."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "nope"))

    report = cc.fx_cache_failure_report()
    assert "live_dir_missing=" in report
    assert "live_keys=0" in report


def test_the_report_never_raises_whatever_the_cache_dir_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its one hard contract: it runs on a failure path, so it may never add a
    second failure to the one being diagnosed."""
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    assert cc.fx_cache_failure_report()  # non-empty, no raise


def test_the_compiled_graph_side_vocabulary_is_GONE_not_merely_empty() -> None:
    """The point of pgw#1200.

    Every one of these tokens is a difference measured against a compiled graph side that
    no format can supply. Reporting them from an empty compiled graph side does not read
    as "no evidence", it reads as a verdict — `fresh_keys>0` names B1 on every
    boot with any FX compiled graph, and `samekey_resaves=0` names "not B2" when B2 is
    simply unmeasurable. Absent evidence must not be dressed as a finding
    (§1.22), so the tokens go with the side that fed them.
    """
    import inspect

    report = cc.fx_cache_failure_report()
    for token in ("compiled_graph_keys", "fresh_keys", "samekey_resave", "compiled_graph_guards",
                  "compiled_graph_extern", "compiled_graph_unpickle", "compiled_graph_read",
                  "live_compiled_graph_compiled_graph_unpickle", "divergence"):
        assert token not in report, (
            f"{token!r} is a compiled-graph-side fact and the compiled_graph side is deleted: "
            f"{report}")
    # And the argument that fed them is gone, so no caller can reintroduce the
    # vocabulary by passing one.
    assert "artifact" not in inspect.signature(cc.fx_cache_failure_report).parameters


def test_the_forensics_helpers_go_with_their_only_caller() -> None:
    """`fx_key_forensics` compared the compiled graph's recorded FxGraphHashDetails lines
    with the boot's. With no compiled graph side there is nothing to compare, and
    `fx_cache_failure_report` was its only production caller — pgw#1178 kept
    all three alive by anchoring them here."""
    for gone in ("fx_key_forensics", "_fx_compiled_graph_lines", "_fx_components"):
        assert not hasattr(cc, gone), gone


def test_the_executor_still_asks_for_the_report_on_the_dynamo_lane() -> None:
    """The caller is not deleted — only the argument it could not supply.

    `proves_inductor` is what scopes this to the dynamo lane (pgw#722 finding
    2), and a dynamo boot that failed its warmup proof still wants the live
    cache state. A report nobody calls would be the defect one layer over.
    """
    import inspect

    from gen_worker import executor as executor_mod

    source = inspect.getsource(executor_mod)
    calls = source.count("fx_cache_failure_report(")
    assert calls >= 2, f"the executor's two call sites are gone ({calls})"
    assert "fx_cache_failure_report(\n" in source or \
        "fx_cache_failure_report()" in source


def test_the_live_census_reads_real_torch_compiled_graph_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dotfiles and empty key dirs are not compiled graphs — torch writes lock and
    temp siblings, and counting them would inflate the one number left."""
    live = tmp_path / "live"
    compiled_graph = _write_live_compiled_graph(live, "fkey00000001")
    (compiled_graph.parent / ".lock").write_bytes(b"")
    (live / "fxgraph" / "em" / "fkeyempty0000").mkdir(parents=True)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(live))

    report = cc.fx_cache_failure_report()
    assert "live_keys=1" in report, report


def test_an_unreadable_compiled_graph_never_breaks_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = tmp_path / "live"
    d = live / "fxgraph" / "ke" / "fkey00000001"
    d.mkdir(parents=True)
    (d / "compiled_graph.bin").write_bytes(b"not-a-pickle")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(live))

    report = cc.fx_cache_failure_report()
    assert "live_keys=1" in report
    assert report  # and no raise


def _unused_keep_imports() -> List[Any]:
    return [os, Path]
