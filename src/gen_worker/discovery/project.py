"""Endpoint project config: ``[tool.gen_worker]`` in pyproject.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    name: str
    main: str
    discovery_heavy_deps: tuple[str, ...] = ()


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    """Load the endpoint project config from ``pyproject.toml``."""
    p = Path(path).expanduser().resolve() if path else Path.cwd().resolve()
    pyproject = p if p.name == "pyproject.toml" else p / "pyproject.toml"
    root = pyproject.parent
    if not pyproject.exists():
        raise FileNotFoundError(
            f"pyproject.toml not found at {pyproject}; run from the endpoint "
            "root or pass --config."
        )
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    tool = data.get("tool") if isinstance(data, dict) else None
    gw = tool.get("gen_worker") if isinstance(tool, dict) else None
    main = str((gw or {}).get("main") or "").strip() if isinstance(gw, dict) else ""
    if not main:
        raise ValueError(
            f"{pyproject}: missing [tool.gen_worker] main. Add:\n"
            '    [tool.gen_worker]\n    main = "your_package.main"'
        )
    raw_heavy = (gw or {}).get("discovery_heavy_deps") if isinstance(gw, dict) else None
    if raw_heavy is None:
        heavy: tuple[str, ...] = ()
    elif isinstance(raw_heavy, list) and all(isinstance(v, str) for v in raw_heavy):
        heavy = tuple(v.strip() for v in raw_heavy if v.strip())
    else:
        raise ValueError(
            f"{pyproject}: [tool.gen_worker] discovery_heavy_deps must be a "
            "list of import-root strings"
        )
    project = data.get("project") if isinstance(data, dict) else None
    name = str((project or {}).get("name") or "").strip() if isinstance(project, dict) else ""
    return ProjectConfig(root=root, name=name, main=main, discovery_heavy_deps=heavy)
