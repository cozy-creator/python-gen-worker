"""gw#468: the typed Settings loader is the ONLY ambient env-read surface.

Every `os.environ` / `os.getenv` in src/gen_worker outside config/ must be a
known internal-plumbing site (parent→child env passing, per-call test-tunable
knobs, subprocess env assembly) listed in ALLOWED below. A new worker knob
belongs on `Settings` (config/settings.py + loader `_ENV_TO_FIELD`), not on
this list. See docs/environment.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "gen_worker"

# relpath -> substrings; every env-touching line in that file must contain
# one of them. One line each on why the site earned raw-env status:
ALLOWED: dict[str, tuple[str, ...]] = {
    # setdefault before torch import — must precede any Settings consumer.
    "entrypoint.py": ("PYTORCH_CUDA_ALLOC_CONF",),
    # http timeout floors are read per-call BY DESIGN (test-tunable, gw#456).
    "net.py": ("os.environ.get(name",),
    # producer→child plumbing: inductor/triton cache dirs + cold-compile
    # opt-in are exported for spawned compile workers.
    "compile_cache.py": ("os.environ[env]", "os.environ[ENV_ALLOW_COLD]"),
    # subprocess env assembly (full environ passthrough to the runtime server).
    "runtimes/server.py": ("dict(os.environ)",),
    # live veto lane — removal owned by PR #139; drop this entry with it.
    "models/memory.py": ("GEN_WORKER_FORBID_CPU_OFFLOAD",),
    # per-call test-tunable retry knob (mirrors net.py's pattern).
    "models/download.py": ("COZY_CIVITAI_DOWNLOAD_ATTEMPTS",),
    # parent→child: `--device` flag exported for endpoint code + ladder
    # opt-out read per-invocation on the local path.
    "cli/run.py": ("GEN_WORKER_LOCAL_DEVICE", "GEN_WORKER_NO_PRECISION_LADDER"),
    "cli/serve.py": ("GEN_WORKER_LOCAL_DEVICE",),
    # cozy-local app plumbing + login-name fallback.
    "cli/local_context.py": ("GEN_WORKER_LOCAL_OUTPUT_DIR", '"USER"'),
    # convert-job scratch/workdir knobs, set by the invoking job harness.
    "convert/clone.py": ("COZY_CONVERT_",),
    "convert/ingest.py": ("_DOWNLOAD_ATTEMPTS_ENV",),
}


def _env_touch_lines(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in ("environ", "getenv", "putenv")
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            if any(a.name in ("environ", "getenv", "putenv") for a in node.names):
                lines.append(node.lineno)
    return lines


def test_no_raw_env_reads_outside_settings_loader():
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("config/"):
            continue  # the sanctioned surface itself
        source = path.read_text(encoding="utf-8")
        if "os." not in source and "import os" not in source:
            continue
        src_lines = source.splitlines()
        allowed = ALLOWED.get(rel, ())
        for lineno in _env_touch_lines(ast.parse(source, filename=rel)):
            line = src_lines[lineno - 1].strip()
            if any(marker in line for marker in allowed):
                continue
            violations.append(f"{rel}:{lineno}: {line}")
    assert not violations, (
        "raw os.environ/os.getenv outside config/loader.py — move the knob "
        "onto Settings (config/settings.py + _ENV_TO_FIELD) or, for genuine "
        "internal plumbing, extend ALLOWED with a one-line justification:\n"
        + "\n".join(violations)
    )
