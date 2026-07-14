"""README hello-world: <=20 lines, and it actually runs through the CLI."""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import gen_worker.cli as cli

ROOT = Path(__file__).resolve().parents[1]


def _readme_python_blocks() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_hello_world_is_at_most_20_lines() -> None:
    block = _readme_python_blocks()[0]
    lines = [ln for ln in block.strip().splitlines() if ln.strip()]
    assert 0 < len(lines) <= 20, f"hello-world is {len(lines)} lines"


def test_hello_world_executes_through_gen_worker_run(capsys) -> None:
    block = _readme_python_blocks()[0]
    name = "_readme_hello"
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    exec(compile(block, "<README hello-world>", "exec"), mod.__dict__)
    sys.modules[name] = mod
    try:
        rc = cli.main([
            "run", "--module", name, "--payload", json.dumps({"prompt": "hello"}),
        ])
        assert rc == 0
        out = capsys.readouterr().out.strip().splitlines()[-1]
        assert json.loads(out)["value"] == {"text": "got: hello"}
    finally:
        sys.modules.pop(name, None)
