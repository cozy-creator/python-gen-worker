#!/usr/bin/env python3
"""pgw#1049: exactly ONE authority writes torch settings.

Paul's directive (2026-08-09): "everything has to go through _us_ …
`us -> pytorch settings`, not `[us, random other things] -> pytorch
settings`." The settings authority (`gen_worker.settings_authority` and the
modules it names in `AUTHORITY_MODULES`) is the only sanctioned writer of
torch/dynamo/inductor process settings. Any OTHER write site in
`src/gen_worker` must appear in `scripts/settings_writers_allowlist.txt`
with a classification, or this script fails — the same enforcement shape as
`lint_config_reads.py` (§1.18) and th#1678's wirecontract census:
an unclassified site is red, and a stale allowlist row is red.

What counts as a WRITE:

* an attribute assignment whose dotted path roots in a torch settings
  surface: ``torch.backends.*``, ``torch._dynamo.config.*``,
  ``torch._inductor.config.*``, ``torch._functorch.config.*``,
  ``torch.compiler.config.*`` — including through import aliases
  (``import torch._inductor.config as inductor_config``);
* a call to a global torch setter: ``set_float32_matmul_precision``,
  ``set_grad_enabled``, ``use_deterministic_algorithms``,
  ``set_default_device``;
* a ``.patch(...)`` call on a torch config module (scoped, restores on
  exit — classified SCOPED so every one is visible, never silent);
* an ``os.environ`` write (``[...] =``, ``.setdefault``, ``.update``,
  ``.pop``) whose key literal lands in a torch-consulted namespace
  (`WATCHED_ENV`).

Classifications:

    SCOPED     a context-managed ``config.patch`` that restores on exit;
               the serving window it opens is covered by dynamo's
               GlobalStateGuard + the pgw#680 guard-miss doctrine
    PLUMBING   a cache/path redirect (TORCHINDUCTOR_CACHE_DIR, ...) —
               points where bytes land, never what bytes are generated;
               inductor compiled graphs are content-addressed
    SCRUB      an erase (``os.environ.pop`` in a scrub loop) — removal of
               ambient input, the opposite of a second writer

There is deliberately NO general-purpose exemption class: a write that is
none of the above moves into the authority, it does not get a new label.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO / "src" / "gen_worker"
ALLOWLIST = REPO / "scripts" / "settings_writers_allowlist.txt"

CLASSIFICATIONS = {"SCOPED", "PLUMBING", "SCRUB"}

#: Modules whose writes ARE the authority (kept in sync with
#: gen_worker.settings_authority.AUTHORITY_MODULES by test_settings_fence).
AUTHORITY_FILES = {
    "settings_authority.py",
    "env_seal.py",
    "host_isa.py",
    "guard_closure.py",
}

#: Dotted prefixes that name a torch settings surface.
SETTINGS_ROOTS: Tuple[str, ...] = (
    "torch.backends",
    "torch._dynamo.config",
    "torch._inductor.config",
    "torch._functorch.config",
    "torch.compiler.config",
)

#: Global torch setter calls (write-equivalent).
SETTER_CALLS: Tuple[str, ...] = (
    "torch.set_float32_matmul_precision",
    "torch.set_grad_enabled",
    "torch.use_deterministic_algorithms",
    "torch.set_default_device",
)

#: Env namespaces torch (or its toolchain) consults for BEHAVIOR.
WATCHED_ENV: Tuple[str, ...] = (
    "TORCH", "PYTORCH", "TRITON", "CUBLAS", "CUDNN", "CUDA_", "NCCL",
    "NVIDIA", "OMP_", "MKL_", "PYTHONHASHSEED",
)


class _Aliases(ast.NodeVisitor):
    """Import-alias map: local name -> dotted module path."""

    def __init__(self) -> None:
        self.names: Dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names[alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and not node.level:
            for alias in node.names:
                self.names[alias.asname or alias.name] = (
                    f"{node.module}.{alias.name}")


def _dotted(node: ast.AST, aliases: Dict[str, str]) -> str:
    """Resolve an attribute chain to its dotted path, expanding aliases."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(aliases.get(node.id, node.id))
        return ".".join(reversed(parts))
    return ""


def _settings_root(path: str) -> Optional[str]:
    for root in SETTINGS_ROOTS:
        if path == root or path.startswith(root + "."):
            return root
    return None


class _Writes(ast.NodeVisitor):
    """Every torch-settings write in one module."""

    def __init__(self, aliases: Dict[str, str],
                 consts: Optional[Dict[str, str]] = None) -> None:
        self.aliases = aliases
        self.consts = consts or {}
        self.hits: List[Tuple[int, str]] = []  # (line, site key)

    def _env_key(self, node: ast.AST) -> str:
        """A static env-var name for a key expression: a string literal, or a
        module-level string constant (`_NVLS_ENV`); "" when computed."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.consts.get(node.id, "")
        return ""

    def _check_target(self, target: ast.AST, lineno: int) -> None:
        if isinstance(target, ast.Attribute):
            path = _dotted(target, self.aliases)
            root = _settings_root(path)
            if root:
                self.hits.append((lineno, path))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._check_target(elt, lineno)
        elif isinstance(target, ast.Subscript):
            # os.environ["X"] = ... — and settings-module item writes.
            base = _dotted(target.value, self.aliases)
            if base.endswith("os.environ") or base == "os.environ":
                name = self._env_key(target.slice)
                if not name:
                    # A computed key could be ANY namespace — classify it.
                    self.hits.append((lineno, "os.environ[<dynamic>]"))
                elif name.startswith(WATCHED_ENV):
                    self.hits.append((lineno, f"os.environ[{name}]"))
            elif _settings_root(base):
                self.hits.append((lineno, base + "[...]"))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        path = _dotted(node.func, self.aliases)
        if path in SETTER_CALLS:
            self.hits.append((node.lineno, path))
        elif path.endswith(".patch") and _settings_root(path[: -len(".patch")]):
            self.hits.append((node.lineno, path))
        elif path in ("os.environ.setdefault", "os.environ.update",
                      "os.environ.pop"):
            name = self._env_key(node.args[0]) if node.args else ""
            if path.endswith("update") or name.startswith(WATCHED_ENV):
                self.hits.append(
                    (node.lineno, f"{path}({name or '<dynamic>'})"))
        self.generic_visit(node)


def scan(root: Path = SRC_ROOT) -> Dict[Tuple[str, str], int]:
    """Every write site outside the authority, keyed (path, site) — never by
    line number (a line is a fact other people change; pgw#931's lesson)."""
    sites: Dict[Tuple[str, str], int] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name in AUTHORITY_FILES and path.parent == root:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = _Aliases()
        aliases.visit(tree)
        consts: Dict[str, str] = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                consts[node.targets[0].id] = node.value.value
        writes = _Writes(aliases.names, consts)
        writes.visit(tree)
        try:
            rel = str(path.relative_to(REPO))
        except ValueError:
            rel = str(path)
        for lineno, site in writes.hits:
            sites.setdefault((rel, site), lineno)
    return sites


def load_allowlist(
    path: Path = ALLOWLIST,
) -> Tuple[Dict[Tuple[str, str], str], List[str]]:
    """Parse ``<path>::<site>  <CLASSIFICATION>  <reason>`` lines."""
    allowed: Dict[Tuple[str, str], str] = {}
    errors: List[str] = []
    if not path.is_file():
        return allowed, [f"{path} is missing"]
    for num, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            errors.append(f"{path.name}:{num}: need '<path>::<site> "
                          f"<CLASSIFICATION> <reason>', got {line!r}")
            continue
        key, classification = parts[0], parts[1]
        if "::" not in key:
            errors.append(f"{path.name}:{num}: site key {key!r} lacks '::'")
            continue
        if classification not in CLASSIFICATIONS:
            errors.append(
                f"{path.name}:{num}: unknown classification "
                f"{classification!r} (want one of {sorted(CLASSIFICATIONS)})")
            continue
        file_part, site = key.split("::", 1)
        allowed[(file_part, site)] = classification
    return allowed, errors


def check(
    sites: Dict[Tuple[str, str], int],
    allowed: Dict[Tuple[str, str], str],
) -> List[str]:
    problems: List[str] = []
    for (rel, site), lineno in sorted(sites.items()):
        if (rel, site) not in allowed:
            problems.append(
                f"{rel}:{lineno}: UNCLASSIFIED torch-settings write: {site} — "
                "a second writer is a defect (pgw#1049). Move the write into "
                "the settings authority, or classify it in "
                "scripts/settings_writers_allowlist.txt")
    live = set(sites)
    for key in sorted(set(allowed) - live):
        problems.append(
            f"stale allowlist row {key[0]}::{key[1]} matches no write site — "
            "delete it (a row matching nothing is a boundary that lies)")
    return problems


def main() -> int:
    sites = scan()
    allowed, errors = load_allowlist()
    problems = errors + check(sites, allowed)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"settings-writer fence: {len(sites)} classified site(s), "
          f"authority = {sorted(AUTHORITY_FILES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
