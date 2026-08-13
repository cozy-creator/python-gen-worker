"""Repo-root pytest plugin: the skip census.

**The defect this exists to remove.** In a green run a skipped row and a passing
row are the same character of nothing — no line a reader looks at, no way to
tell "3117 passed" apart from "3117 passed, and the one row that measures the
thing you changed did not run". pgw#858 is that failure in production: its
container row skips on `ubuntu-latest` (the image cannot exec the runner's
uv-managed interpreter, handled by a by-name skip on exit 127), so the row that
proves the uid boundary has only ever executed on developer boxes — which is
how the pgw#956 flake reached `dev` behind a green CI.

So: record what the session did NOT run, keyed stably, and let
`scripts/lint_skip_census.py` gate on it against `scripts/skip_census.txt`. A
skip whose key is not classified there fails CI; a key classified `MUST-RUN` and
observed anyway fails CI. Neither can be satisfied by silence.

The key is ``<test file relative to the repo>|<normalized reason>`` — NOT a line
number, which every edit above the site would churn. Digits, paths and hex runs
are collapsed, so a reason built from an f-string ("box has only 5.1GiB of guard
headroom") is one stable key across boxes.

Emitting is opt-in (``--skip-census-out=PATH``); the hooks are inert otherwise,
and nothing here touches collection, ordering or fixtures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

_REPO = Path(__file__).resolve().parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-census-out",
        action="store",
        default=None,
        metavar="PATH",
        help="pgw#966: write this session's skip census (JSON) to PATH",
    )


# Order matters: paths before bare numbers, so a path's digits do not survive as
# a separate token.
_PATHISH = re.compile(r"(?:/[\w.+@-]+){2,}/?")
_HEXISH = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
# `#764` is an issue reference and belongs in the key; every other number is a
# measurement of the box and must not be.
_NUM = re.compile(r"#\d+|\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")


def normalize_reason(reason: str) -> str:
    """A reason string reduced to what is stable across boxes and edits."""
    text = reason.strip()
    for prefix in ("Skipped: ", "Skipped ", "[XFAIL] ", "reason: "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = _PATHISH.sub("<path>", text)
    text = _HEXISH.sub("<hex>", text)
    text = _NUM.sub(lambda m: m.group(0) if m.group(0)[0] == "#" else "#", text)
    text = _WS.sub(" ", text).strip().lower()
    return text[:120]


def _longrepr_parts(longrepr: Any) -> Tuple[str | None, str]:
    """``(file, reason)`` out of the shapes pytest uses for a skip."""
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[0]), str(longrepr[2])
    return None, "" if longrepr is None else str(longrepr)


class SkipCensus:
    """Collects skip/xfail rows on the controller and writes them once."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.rows: Dict[str, Dict[str, Any]] = {}

    # -- collection ------------------------------------------------------
    def _relpath(self, nodeid: str, fspath: str | None) -> str:
        if fspath:
            resolved = Path(fspath)
            try:
                return resolved.resolve().relative_to(_REPO).as_posix()
            except ValueError:
                return resolved.as_posix()
        return nodeid.split("::", 1)[0]

    def add(self, kind: str, nodeid: str, longrepr: Any) -> None:
        fspath, raw = _longrepr_parts(longrepr)
        rel = self._relpath(nodeid, fspath)
        key = f"{rel}|{normalize_reason(raw)}"
        row = self.rows.setdefault(
            key,
            {"key": key, "file": rel, "kind": kind, "count": 0,
             "reason": raw.strip()[:400], "examples": []},
        )
        row["count"] += 1
        if len(row["examples"]) < 3:
            row["examples"].append(nodeid)

    # -- hooks -----------------------------------------------------------
    @pytest.hookimpl(trylast=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # xdist forwards worker reports to the controller, so this sees every
        # row of an `-n 4` run without the workers writing anything.
        if not report.skipped:
            return
        # An xfail's reason lives on `wasxfail`; its `longrepr` is the marker's
        # own source text, which is neither stable nor readable as a key.
        xfail = getattr(report, "wasxfail", None)
        if xfail is not None:
            self.add("xfail", report.nodeid, (report.location[0], 0, str(xfail)))
        else:
            self.add("skip", report.nodeid, report.longrepr)

    @pytest.hookimpl(trylast=True)
    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        # `pytest.skip(..., allow_module_level=True)` and a module-level
        # `importorskip` never produce a test report at all — they are the whole
        # FILE going missing, which is the most invisible skip of the lot.
        if report.skipped:
            self.add("module", report.nodeid, report.longrepr)

    def write(self, args: List[str]) -> None:
        rows: List[Dict[str, Any]] = sorted(self.rows.values(), key=lambda r: r["key"])
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(
            json.dumps({"args": args, "rows": rows}, indent=2) + "\n", encoding="utf-8")


def pytest_configure(config: pytest.Config) -> None:
    out = config.getoption("--skip-census-out")
    if not out or hasattr(config, "workerinput"):
        return  # workers report through the controller; only it collects.
    config.pluginmanager.register(SkipCensus(Path(str(out))), "pgw966-skip-census")


def pytest_sessionfinish(session: pytest.Session) -> None:
    plugin = session.config.pluginmanager.get_plugin("pgw966-skip-census")
    if isinstance(plugin, SkipCensus):
        plugin.write([str(a) for a in session.config.args])
