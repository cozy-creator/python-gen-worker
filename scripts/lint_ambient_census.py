#!/usr/bin/env python3
"""pgw#1049: the ambient-input census — structural half (no torch needed).

`scripts/ambient_inputs_census.txt` classifies every env input torch/triton
consult. This lint verifies the claims that are checkable from the tree
alone; `tests/test_ambient_census_pgw1049.py` verifies completeness against
the INSTALLED torch (an unclassified input is red — the th#1678 shape).

Checked here:

1. row syntax + known classes;
2. every IMPOSED row names a key in `settings_authority.DECLARED_ENV`
   (AST-extracted — the census may not claim an imposition that does not
   exist), and every DECLARED_ENV key has an IMPOSED row (an imposition the
   census does not name is an unclassified input);
3. every NEUTRALIZED row is COVERED by `env_seal.SCRUB_PREFIXES` — the
   pattern's fixed stem must start with a scrub prefix, so "erased before
   torch imports" is a fact, not a hope;
4. every watched-namespace env literal in `src/gen_worker` (reads and
   writes) is classified by some row.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO / "src" / "gen_worker"
CENSUS = REPO / "scripts" / "ambient_inputs_census.txt"

CLASSES = {"IMPOSED", "NEUTRALIZED", "PLUMBING", "IRRELEVANT"}

#: Env namespaces whose literals in OUR tree must be classified.
WATCHED = ("TORCH", "PYTORCH", "TRITON", "CUBLAS", "CUDNN", "CUDA_", "NCCL",
           "NVIDIA", "OMP_", "MKL_", "KMP_", "ATEN_", "AOTI", "AOT_",
           "INDUCTOR_", "CUTLASS_", "CUTEDSL_", "PYTHONHASHSEED")


def load_census(
    path: Path = CENSUS,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """[(pattern, class, reason)] in file order; first match wins."""
    rows: List[Tuple[str, str, str]] = []
    errors: List[str] = []
    if not path.is_file():
        return rows, [f"{path} is missing"]
    for num, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            errors.append(f"{path.name}:{num}: need '<PATTERN> <CLASS> "
                          f"<reason>', got {line!r}")
            continue
        pattern, klass, reason = parts
        if klass not in CLASSES:
            errors.append(f"{path.name}:{num}: unknown class {klass!r} "
                          f"(want one of {sorted(CLASSES)})")
            continue
        rows.append((pattern, klass, reason))
    return rows, errors


def classify(name: str, rows: List[Tuple[str, str, str]]) -> Optional[str]:
    for pattern, klass, _ in rows:
        if fnmatch.fnmatchcase(name, pattern):
            return klass
    return None


def _module_str_tuple(path: Path, symbol: str) -> Tuple[str, ...]:
    """AST-extract a module-level tuple/dict of string constants."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == symbol):
            continue
        if isinstance(value, (ast.Tuple, ast.List)):
            return tuple(e.value for e in value.elts
                         if isinstance(e, ast.Constant)
                         and isinstance(e.value, str))
        if isinstance(value, ast.Dict):
            return tuple(k.value for k in value.keys
                         if isinstance(k, ast.Constant)
                         and isinstance(k.value, str))
    return ()


def scrub_prefixes() -> Tuple[str, ...]:
    return _module_str_tuple(SRC_ROOT / "env_seal.py", "SCRUB_PREFIXES")


def declared_env_keys() -> Tuple[str, ...]:
    return _module_str_tuple(
        SRC_ROOT / "settings_authority.py", "DECLARED_ENV")


def _pattern_stem(pattern: str) -> str:
    """The fixed prefix of an fnmatch pattern (up to the first wildcard)."""
    return re.split(r"[*?\[]", pattern, 1)[0]


_ENV_LITERAL = re.compile(
    r'["\']([A-Z][A-Z0-9_]{2,})["\']')


def src_watched_literals() -> Dict[str, str]:
    """Watched-namespace env-var literals in our tree: {name: first path}.

    Regex over string literals, not an AST env-read walk — the census cares
    about every place a watched NAME appears (child env dicts, scrub lists,
    docs-in-code aside), and a false positive costs one census row while a
    false negative hides an input."""
    out: Dict[str, str] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in _ENV_LITERAL.finditer(text):
            name = m.group(1)
            if name.startswith(WATCHED):
                out.setdefault(name, str(path.relative_to(REPO)))
    return out


#: Literal env-read shapes in torch/triton source. Dynamic reads (inductor's
#: env_name_default machinery constructs names at config install) are covered
#: by the NAMESPACE rows — the constructed names all live in TORCHINDUCTOR_*/
#: TORCHDYNAMO_* etc., which are scrubbed wholesale.
_TORCH_READS = (
    re.compile(r'os\.environ\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)'),
    re.compile(r'os\.environ\[\s*["\']([A-Za-z_][A-Za-z0-9_]*)'),
    re.compile(r'os\.getenv\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)'),
    re.compile(r'\bgetenv\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)'),
)


def scan_installed_tree(
    packages: Tuple[str, ...] = ("torch", "triton", "functorch"),
) -> Dict[str, str]:
    """Every literal env name the INSTALLED packages read: {name: rel path}.
    Enumerated from source on disk — no import, so no side effects."""
    import importlib.util

    out: Dict[str, str] = {}
    for pkg in packages:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for root in spec.submodule_search_locations:
            for path in sorted(Path(root).rglob("*.py")):
                try:
                    text = path.read_text(errors="replace")
                except OSError:
                    continue
                for pat in _TORCH_READS:
                    for m in pat.finditer(text):
                        out.setdefault(m.group(1), f"{pkg}/{path.name}")
    return out


def check(census_path: Path = CENSUS) -> List[str]:
    rows, problems = load_census(census_path)
    if problems:
        return problems
    prefixes = scrub_prefixes()
    declared = declared_env_keys()
    if not prefixes:
        problems.append("could not extract SCRUB_PREFIXES from env_seal.py")
    if not declared:
        problems.append(
            "could not extract DECLARED_ENV from settings_authority.py")

    imposed_rows = [p for p, k, _ in rows if k == "IMPOSED"]
    for pattern in imposed_rows:
        if pattern not in declared:
            problems.append(
                f"census row {pattern!r} claims IMPOSED but "
                "settings_authority.DECLARED_ENV has no such key — the "
                "census may not claim an imposition that does not exist")
    for key in declared:
        if key not in imposed_rows:
            problems.append(
                f"DECLARED_ENV[{key!r}] has no IMPOSED census row — an "
                "imposition the census does not name is an unclassified "
                "input")

    for pattern, klass, _ in rows:
        if klass != "NEUTRALIZED":
            continue
        stem = _pattern_stem(pattern)
        if not any(stem.startswith(p) for p in prefixes):
            problems.append(
                f"census row {pattern!r} claims NEUTRALIZED but no "
                f"env_seal.SCRUB_PREFIXES compiled_graph covers stem {stem!r} — "
                "'erased before torch imports' must be a fact")

    for name, where in sorted(src_watched_literals().items()):
        if classify(name, rows) is None:
            problems.append(
                f"{where}: watched env name {name!r} appears in the tree "
                "but has no census row — classify it in "
                "scripts/ambient_inputs_census.txt")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    rows, _ = load_census()
    print(f"ambient-input census: {len(rows)} row(s), structural claims hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
