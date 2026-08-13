#!/usr/bin/env python3
"""The compute child cannot answer a question about itself with a credential
it does not hold — so every site that reads one is classified.

The compute child (the only execution model) holds **no worker credential by
construction**: the parent strips ``WORKER_JWT`` from its environment and no
frame carries it. A gate that reads that credential to answer "who am I?" or
"is there a hub?" is therefore not wrong on some pods — it is wrong on **every
real serving pod, always**, and it looks like a decision while it does it.

Every read of a worker credential inside
``src/gen_worker`` must appear in ``scripts/credential_identity_allowlist.txt``
under one of the classifications below, or this script fails — the same
enforcement shape as ``lint_settings_writers.py`` and
``lint_config_reads.py`` (§1.18): an unclassified site is red, and a stale row
is red.

Classifications:

    PARENT      the site runs ONLY in the control parent, the mint CLI or a
                single-process worker — a process that genuinely holds the
                credential
    BEARER      the value is used as an HTTP ``Authorization`` bearer and
                nothing else; under the split the parent supplies the real one
                and ignores this, so an empty string here costs nothing
    READINESS   credential-presence as a "can we reach the hub" signal, and it
                MUST carry the seam fallback (``broker.active()``), or the
                child reads "no credential" as "no hub to ask"
    WIRING      the provider is passed through to somebody else and never read
                here
    RELAYED     it derives a value the parent ALSO hands the child as a plain
                env value, so an empty credential here changes nothing
    DIAGNOSTIC  a log/annotation that degrades to silence in the child, never
                a decision

**There is deliberately NO ``IDENTITY`` classification.** A site that needs to
know which endpoint or org this pod serves calls ``gen_worker.worker_identity.
viewer()``, which asks this process's own credential when it has one and the
control PARENT when it does not, and refuses TYPED when neither can answer. A
third instance of this bug has no label to write down.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO / "src" / "gen_worker"
ALLOWLIST = REPO / "scripts" / "credential_identity_allowlist.txt"

CLASSIFICATIONS = {
    "PARENT", "BEARER", "READINESS", "WIRING", "RELAYED", "DIAGNOSTIC",
}

#: The ONE module allowed to turn a credential into an identity. Exempt because
#: it IS the resolver every other site is required to use.
RESOLVER_FILES = {"worker_identity.py"}

#: The module that OWNS the credential itself — the single source whose whole
#: job is holding and handing out the token.
CREDENTIAL_FILES = {"worker_credential.py"}

#: Attribute names that name a worker credential. Deliberately by NAME rather
#: than by resolved receiver: the receiver is `self`, `cfg`, `remote`, a
#: provider closure or a dataclass field at different sites, and a fence that
#: only catches the spellings we thought of is not a fence.
CREDENTIAL_ATTRS: Tuple[str, ...] = (
    "current_worker_jwt",
    "worker_jwt_provider",
    "bootstrap_worker_jwt",
    "worker_jwt",
    "_worker_jwt",
)

#: The process-wide credential source.
CREDENTIAL_CALLS: Tuple[str, ...] = ("worker_credential.current",)

#: The env the credential arrives in at pod launch. Reading it is how the child
#: used to get one; the parent now strips it (`_CHILD_FORBIDDEN_ENVS`).
CREDENTIAL_ENV = "WORKER_JWT"


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


class _Reads(ast.NodeVisitor):
    """Every worker-credential read in one module.

    Sites are keyed ``<enclosing function>::<name>``, not by file alone: the
    executor already reads the same provider for wiring and for a readiness
    gate, and a fence that merges them would wave through the next identity
    read in a file that has any legitimate one. Function scope is the unit a
    reviewer actually reasons about, and it survives edits that move lines.
    """

    def __init__(self) -> None:
        self.hits: List[Tuple[int, str]] = []
        self._scope: List[str] = []

    def _where(self) -> str:
        return ".".join(self._scope) or "<module>"

    def _push(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push(node, node.name)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in CREDENTIAL_ATTRS:
            self.hits.append((node.lineno, f"{self._where()}::{node.attr}"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        path = _dotted(node.func)
        if path in CREDENTIAL_CALLS:
            self.hits.append((node.lineno, f"{self._where()}::{path}"))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value == CREDENTIAL_ENV:
            self.hits.append(
                (node.lineno, f"{self._where()}::env:{CREDENTIAL_ENV}"))
        self.generic_visit(node)


def scan(root: Path = SRC_ROOT) -> Dict[Tuple[str, str], int]:
    """Credential-read sites outside the resolver, keyed ``(path, site)``.

    Never keyed by line number: a line is a fact other people change.
    """
    sites: Dict[Tuple[str, str], int] = {}
    for path in sorted(root.rglob("*.py")):
        if path.parent == root and path.name in (
            RESOLVER_FILES | CREDENTIAL_FILES
        ):
            continue
        reads = _Reads()
        reads.visit(ast.parse(path.read_text(encoding="utf-8")))
        try:
            rel = str(path.relative_to(REPO))
        except ValueError:
            rel = str(path)
        for lineno, site in reads.hits:
            sites.setdefault((rel, site), lineno)
    return sites


def load_allowlist(
    path: Path = ALLOWLIST,
) -> Tuple[Dict[Tuple[str, str], str], List[str]]:
    """Parse ``<path>::<name>  <CLASSIFICATION>  <reason>`` lines."""
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
            errors.append(
                f"{path.name}:{num}: need '<path>::<name> <CLASSIFICATION> "
                f"<reason>', got {line!r}")
            continue
        key, classification = parts[0], parts[1]
        if "::" not in key:
            errors.append(f"{path.name}:{num}: site key {key!r} lacks '::'")
            continue
        if classification not in CLASSIFICATIONS:
            errors.append(
                f"{path.name}:{num}: unknown classification "
                f"{classification!r} (want one of {sorted(CLASSIFICATIONS)}). "
                "There is no IDENTITY class: identity comes from "
                "gen_worker.worker_identity.viewer()")
            continue
        file_part, name = key.split("::", 1)
        allowed[(file_part, name)] = classification
    return allowed, errors


def check(
    sites: Dict[Tuple[str, str], int],
    allowed: Dict[Tuple[str, str], str],
) -> List[str]:
    problems: List[str] = []
    for (rel, name), lineno in sorted(sites.items()):
        if (rel, name) not in allowed:
            problems.append(
                f"{rel}:{lineno}: UNCLASSIFIED worker-credential read: {name} — "
                "the compute child holds none by construction (pgw#763 delta 1), "
                "so a gate that reads one refuses on every real serving pod "
                "(pgw#1108, pgw#1122). If you need this pod's IDENTITY call "
                "gen_worker.worker_identity.viewer(); otherwise classify the "
                "site in scripts/credential_identity_allowlist.txt")
    for key in sorted(set(allowed) - set(sites)):
        problems.append(
            f"stale allowlist row {key[0]}::{key[1]} matches no read site — "
            "delete it (a row matching nothing is a boundary that lies)")
    return problems


def main() -> int:
    sites = scan()
    allowed, errors = load_allowlist()
    problems = errors + check(sites, allowed)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"credential-identity fence: {len(sites)} classified read(s); "
          f"the resolver is {sorted(RESOLVER_FILES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
