"""Import-graph guard (#367): ``import gen_worker`` must never pull in the
conversion ETL (cozy_convert), torch, or any leftover clone/conversion module.

Runs in a subprocess so this test observes a cold import, not whatever the
rest of the suite already loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys

_PROBE = """
import json, sys
import gen_worker
mods = sorted(sys.modules)
print(json.dumps(mods))
"""

_FORBIDDEN_EXACT = {"torch", "cozy_convert", "safetensors", "transformers", "diffusers"}
_FORBIDDEN_SUBSTR = ("gen_worker.conversion", "gen_worker.clone", "cozy_convert.")


def test_import_gen_worker_is_torch_free_and_conversion_free() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True,
    )
    mods = set(json.loads(out.stdout.strip().splitlines()[-1]))
    hits = sorted(
        m for m in mods
        if m in _FORBIDDEN_EXACT or any(s in m for s in _FORBIDDEN_SUBSTR)
    )
    assert not hits, f"import gen_worker pulled in forbidden modules: {hits}"


def test_no_conversion_packages_exist() -> None:
    import importlib.util

    for name in ("gen_worker.conversion", "gen_worker.clone"):
        assert importlib.util.find_spec(name) is None, f"{name} must not exist"
