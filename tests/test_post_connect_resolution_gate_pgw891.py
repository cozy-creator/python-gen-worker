"""pgw#891: the post-connect resolution gate must be able to go RED.

A gate nobody has watched fail is not a gate — it is a green light of unknown
provenance. `scripts/lint_post_connect_resolution.py` is green on arrival by
construction (it is baselined against today's tree), so the only evidence that
it works is driving it against trees that MUST fail.

Both failure directions are covered, because they are different defects:
a NEW call site is the surface growing (the thing the gate exists to stop), and
a STALE allowlist entry is an exemption outliving its site — which would
silently re-permit that site if it ever came back.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint_post_connect_resolution.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgw891_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path: Path, source: str, allowlist: str) -> ModuleType:
    src = tmp_path / "src" / "gen_worker"
    src.mkdir(parents=True)
    (src / "consumer.py").write_text(source, encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    allow = tmp_path / "scripts" / "allow.txt"
    allow.write_text(allowlist, encoding="utf-8")

    mod = _load()
    mod.REPO = tmp_path
    mod.SRC_ROOT = src
    mod.ALLOWLIST = allow
    return mod


CALLS_DISCOVER = "from x import aot_cells\n\ndef f():\n    return aot_cells.discover(1)\n"


def test_a_new_call_site_fails(tmp_path: Path) -> None:
    mod = _tree(tmp_path, CALLS_DISCOVER, "# nothing accepted\n")
    assert mod.main() == 1


def test_the_same_site_passes_once_classified(tmp_path: Path) -> None:
    mod = _tree(
        tmp_path, CALLS_DISCOVER,
        "src/gen_worker/consumer.py::discover  CONNECTED  gated on pgw#891\n",
    )
    assert mod.main() == 0


def test_a_stale_allowlist_entry_fails(tmp_path: Path) -> None:
    """The site is gone; the exemption must not outlive it."""
    mod = _tree(
        tmp_path, "def f():\n    return 1\n",
        "src/gen_worker/consumer.py::discover  CONNECTED  gated on pgw#891\n",
    )
    assert mod.main() == 1


def test_an_unknown_classification_is_refused(tmp_path: Path) -> None:
    mod = _tree(
        tmp_path, CALLS_DISCOVER,
        "src/gen_worker/consumer.py::discover  PROBABLY_FINE  hand-wave\n",
    )
    with pytest.raises(SystemExit):
        mod.main()


def test_a_line_with_no_reason_is_refused(tmp_path: Path) -> None:
    """The reason column is the whole point — an unreasoned exemption is how an
    allowlist decays into prose that nobody can re-derive."""
    mod = _tree(
        tmp_path, CALLS_DISCOVER,
        "src/gen_worker/consumer.py::discover  CONNECTED\n",
    )
    with pytest.raises(SystemExit):
        mod.main()


def test_the_real_tree_is_green_and_the_census_is_not_empty() -> None:
    """Baseline: the shipped allowlist matches the shipped tree. The
    non-emptiness assert matters — a census that silently found nothing would
    also be 'green', and would prove nothing at all."""
    mod = _load()
    found = mod.census()
    assert found, "the watched surface cannot be empty — the walk would be broken"
    assert mod.main() == 0
    assert set(found) == set(mod.allowed())
