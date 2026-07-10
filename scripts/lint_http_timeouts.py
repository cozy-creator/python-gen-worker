#!/usr/bin/env python3
"""gw#467: no HTTP call in src/gen_worker may be able to wait forever.

Fails when any file (except the sanctioned gen_worker/net.py, where the
COZY_HTTP_*_TIMEOUT_S floor lives) does one of:

  1. import huggingface_hub network entry points — HfApi, snapshot_download,
     hf_hub_download, ... must be reached via gen_worker.net.hf(), which
     installs the timeout floor first. Non-network submodules
     (huggingface_hub.errors / .constants) stay importable.
  2. construct httpx.Client/AsyncClient or call an httpx request method
     without an explicit timeout= keyword.
  3. call a requests method (requests.get/post/... or *.request) without an
     explicit timeout= keyword.

Background: huggingface_hub's default client carries timeout=None (there is
no hf_hub 2.x to fix it — latest is 1.x; we own the floor). A single bare
call site reintroduces the gw#456 forever-hang.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "gen_worker"
SANCTIONED = SRC_ROOT / "net.py"

# huggingface_hub submodules with no network entry points.
HF_ALLOWED_SUBMODULES = {"errors", "constants"}

HTTPX_NEEDS_TIMEOUT = {
    "Client", "AsyncClient", "request", "stream",
    "get", "post", "put", "patch", "delete", "head", "options",
}
REQUESTS_NEEDS_TIMEOUT = {
    "request", "get", "post", "put", "patch", "delete", "head", "options",
}


def _hf_module_banned(name: str) -> bool:
    if name == "huggingface_hub":
        return True
    if name.startswith("huggingface_hub."):
        return name.split(".")[1] not in HF_ALLOWED_SUBMODULES
    return False


def _has_timeout_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "timeout" for kw in call.keywords) or any(
        kw.arg is None for kw in call.keywords  # **kwargs: assume forwarded
    )


def check_file(path: Path) -> list[tuple[int, str]]:
    problems: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _hf_module_banned(alias.name):
                    problems.append((node.lineno,
                                     f"import {alias.name} — use gen_worker.net.hf() (gw#467)"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and _hf_module_banned(mod):
                problems.append((node.lineno,
                                 f"from {mod} import ... — use gen_worker.net.hf() (gw#467)"))
        elif isinstance(node, ast.Call):
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)):
                continue
            base, attr = fn.value.id, fn.attr
            needs = (base == "httpx" and attr in HTTPX_NEEDS_TIMEOUT) or \
                    (base == "requests" and attr in REQUESTS_NEEDS_TIMEOUT)
            if needs and not _has_timeout_kwarg(node):
                problems.append((node.lineno,
                                 f"{base}.{attr}(...) without explicit timeout= (gw#467)"))
    return problems


def main() -> int:
    bad = 0
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path == SANCTIONED:
            continue
        for lineno, msg in check_file(path):
            print(f"{path.relative_to(SRC_ROOT.parents[1])}:{lineno}: {msg}")
            bad += 1
    if bad:
        print(f"\nlint_http_timeouts: {bad} violation(s). Every HTTP call needs an explicit "
              "timeout; huggingface_hub goes through gen_worker.net.hf().", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
