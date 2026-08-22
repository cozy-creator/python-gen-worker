#!/usr/bin/env python3
"""THE ONE WRITER of a vendored package's rev — and it re-locks.

A re-vendor moves the same rev in three places, as ONE ACT:
``src/gen_worker/_vendor/VENDORED.toml`` (snapshot + per-file sha256 table),
``pyproject.toml`` ``[tool.uv.sources]`` (the pin the derive resolves), and
``uv.lock`` (what ``uv sync --locked`` installs). Editing them by hand forgets
the third and breaks ``uv sync --locked`` fleet-wide;
``scripts/lint_lock_pin_agreement.py`` refuses a tree where they disagree.

Usage:
    python3 scripts/vendor_snapshot.py torchcg <rev>
    python3 scripts/vendor_snapshot.py torchcg <rev> --pins-only

``--pins-only`` skips the file copy (use when the snapshot is already extracted
by hand, required for packages carrying non-mechanical rewrites).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src" / "gen_worker" / "_vendor" / "VENDORED.toml"
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"

#: Import rewrites applied to a vendored tree so it reaches its VENDORED peers
#: rather than whatever happens to be installed.
#:
#: The indentation group is load-bearing and was missing until pgw#1656: the
#: pattern was anchored at `^`, so a LAZY import — `from tensorfs import ...`
#: inside a function, which torchcg uses deliberately to keep its cold resolve
#: path torch-free — was left pointing at the top-level package. That fails
#: closed only by luck: it raises where the installed tensorfs is older, and
#: silently uses the WRONG COPY where it is merely different, which is the
#: outcome vendoring exists to prevent.
MECHANICAL_REWRITES: dict[str, list[tuple[str, str]]] = {
    "torchcg": [(r"(?m)^([ \t]*)from tensorfs import ", r"\1from ..tensorfs import ")],
}

#: After rewriting, no spelling of a NON-vendored sibling may survive. A rewrite
#: rule that silently matches nothing is how the bug above lived: the script
#: printed a cheerful `rewrite: store.py` for the one file it did catch.
FORBIDDEN_AFTER_REWRITE: dict[str, list[tuple[str, str]]] = {
    "torchcg": [
        (r"(?m)^[ \t]*from tensorfs import ", "an unrewritten `from tensorfs import`"),
        (r"(?m)^[ \t]*import tensorfs\b", "a bare `import tensorfs`"),
    ],
}

SKIP = {"__pycache__", ".git"}


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload,no-any-return]


def _files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part in SKIP for part in p.relative_to(root).parts)
    )


def _digests(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in _files(root)
    }


def _section(text: str, header: str) -> tuple[int, int]:
    start = text.index(f"\n[{header}]\n") + 1
    nxt = text.find("\n[", start + 1)
    return start, len(text) if nxt < 0 else nxt + 1


def _replace_rev(text: str, header: str, rev: str) -> str:
    start, end = _section(text, header)
    body, count = re.subn(r'(?m)^rev = "[0-9a-f]+"$', f'rev = "{rev}"', text[start:end], count=1)
    if count != 1:
        raise SystemExit(f"{header}: expected exactly one `rev = ` line, found {count}")
    return text[:start] + body + text[end:]


def _replace_files_table(text: str, header: str, digests: dict[str, str]) -> str:
    start, end = _section(text, header)
    rows = "".join(f'"{path}" = "{sha}"\n' for path, sha in sorted(digests.items()))
    return text[:start] + f"[{header}]\n" + rows + text[end:]


def _extract(repo: str, rev: str, subdir: str, target: Path, package: str) -> None:
    if package not in MECHANICAL_REWRITES:
        raise SystemExit(
            f"{package}: VENDORED.toml declares non-mechanical rewrites for this "
            f"package, so this script will not overwrite the tree. Extract by "
            f"hand (`git archive {rev} {subdir} | tar -x`), re-apply the rewrites, "
            f"then rerun with --pins-only."
        )
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "upstream"
        _run(["git", "clone", "--quiet", repo, str(clone)])
        _run(["git", "-C", str(clone), "checkout", "--quiet", rev])
        staged = Path(tmp) / "staged"
        staged.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(clone), "archive", rev, subdir],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(staged)], input=archive, check=True)
        source = staged / subdir
        if not source.is_dir():
            raise SystemExit(f"{package}: `{subdir}` is not a directory at {rev}")

        for pattern, replacement in MECHANICAL_REWRITES[package]:
            for path in _files(source):
                if path.suffix != ".py":
                    continue
                text = path.read_text()
                rewritten = re.sub(pattern, replacement, text)
                if rewritten != text:
                    print(f"  rewrite: {path.relative_to(source)}")
                    path.write_text(rewritten)

        for pattern, what in FORBIDDEN_AFTER_REWRITE.get(package, []):
            offenders = [
                f"{path.relative_to(source)}:{text[:m.start()].count(chr(10)) + 1}"
                for path in _files(source)
                if path.suffix == ".py"
                for text in (path.read_text(),)
                for m in re.finditer(pattern, text)
            ]
            if offenders:
                raise SystemExit(
                    f"{package}: {what} survives the vendor rewrite at "
                    f"{', '.join(offenders)}. The vendored tree would reach an "
                    f"INSTALLED sibling instead of the vendored one; fix the "
                    f"rewrite rule rather than the symptom."
                )

        shutil.rmtree(target)
        shutil.copytree(source, target)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("rev")
    parser.add_argument(
        "--pins-only",
        action="store_true",
        help="do not touch the vendored files; re-pin, re-digest and re-lock only",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="skip `uv lock` (for offline dry runs ONLY — it is the whole point)",
    )
    args = parser.parse_args(argv[1:])

    manifest_text = MANIFEST.read_text()
    manifest = tomllib.loads(manifest_text)
    if args.package not in manifest.get("packages", {}):
        raise SystemExit(
            f"{args.package}: not in VENDORED.toml "
            f"({', '.join(sorted(manifest['packages']))})"
        )
    spec = manifest["packages"][args.package]
    repo, subdir = str(spec["repo"]), str(spec["subdir"])
    target = MANIFEST.parent / args.package

    print(f"resolving {args.rev} in {repo}")
    resolved = subprocess.run(
        ["git", "ls-remote", repo, args.rev],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\t")[0].strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        if not re.fullmatch(r"[0-9a-f]{7,40}", args.rev):
            raise SystemExit(f"{args.rev}: neither a ref on {repo} nor a sha")
        resolved = args.rev
    print(f"  -> {resolved}")

    if not args.pins_only:
        _extract(repo, resolved, subdir, target, args.package)
        if len(resolved) != 40:
            with tempfile.TemporaryDirectory() as tmp:
                clone = Path(tmp) / "u"
                _run(["git", "clone", "--quiet", repo, str(clone)])
                resolved = subprocess.run(
                    ["git", "-C", str(clone), "rev-parse", f"{args.rev}^{{commit}}"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

    if len(resolved) != 40:
        raise SystemExit(
            f"{resolved}: --pins-only needs the FULL 40-hex rev (a prefix is not "
            f"a pin; uv writes 40 hex into uv.lock and the two must be comparable)"
        )

    header = f"packages.{args.package}"
    manifest_text = _replace_rev(manifest_text, header, resolved)
    manifest_text = _replace_files_table(
        manifest_text, f"{header}.files", _digests(target)
    )
    MANIFEST.write_text(manifest_text)
    print(f"VENDORED.toml: {args.package} rev + {len(_digests(target))} digests")

    pyproject_text = PYPROJECT.read_text()
    pin = re.compile(
        rf'(?m)^({re.escape(args.package)} = \{{ git = "[^"]+", rev = ")[0-9a-f]+(")'
    )
    if pin.search(pyproject_text):
        PYPROJECT.write_text(pin.sub(rf"\g<1>{resolved}\g<2>", pyproject_text))
        print(f"pyproject.toml: [tool.uv.sources].{args.package} -> {resolved}")
    else:
        print(f"pyproject.toml: no [tool.uv.sources] entry for {args.package} (fine)")

    if args.no_lock:
        print("uv lock: SKIPPED (--no-lock) — uv.lock is now stale ON PURPOSE")
    else:
        _run(["uv", "lock"], cwd=ROOT)

    _run(
        ["git", "add", "--", str(MANIFEST), str(PYPROJECT), str(LOCK), str(target)],
        cwd=ROOT,
    )

    print("\nverifying the three spellings agree:")
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "lint_lock_pin_agreement.py")],
        cwd=ROOT,
    )
    if check.returncode != 0:
        return check.returncode
    print("  ok — pyproject.toml, uv.lock and VENDORED.toml all name", resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
