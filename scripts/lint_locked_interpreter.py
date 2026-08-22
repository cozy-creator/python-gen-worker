#!/usr/bin/env python3
"""pgw#1657 — the tree must PIN an interpreter, and uv.lock must carry a hash
for every wheel a Linux environment on that interpreter can select.

`uv sync --locked` validates a downloaded wheel against the hashes uv.lock
records **for the version**, not for the file. The PyTorch cu130 index
publishes `#sha256=` on only some rows, so uv.lock faithfully stores rows with
no hash — and downloading one of those compares its bytes against a different
row's hash and reports `Hash mismatch`, blaming the index for what is really an
interpreter mismatch. That is what happens on 3.13 with torch 2.13.0+cu130,
half a gigabyte into the download, in CI's first step.

Both halves are checked here, stdlib-only and ahead of uv itself:
  * `.python-version` exists (without it uv picks whatever interpreter the
    machine happens to have — 3.13 on a box with 3.13 installed, 3.12 on a
    GitHub runner, which is why this passed CI and failed everywhere else);
  * every wheel row selectable by that interpreter on Linux carries a hash.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

_PIN = re.compile(r"^(?:cpython-)?(\d+)\.(\d+)")


def _pinned_tags(root: Path) -> tuple[list[str], str | None]:
    """Return (accepted python tags, failure) for the tree's interpreter pin."""
    path = root / ".python-version"
    if not path.is_file():
        return [], (
            ".python-version is MISSING, so nothing pins the interpreter.\n"
            "      `uv sync` then picks whatever compatible interpreter the machine\n"
            "      offers — 3.12 on a GitHub runner, 3.13 on a box that has 3.13 —\n"
            "      and uv.lock does not record a hash for every interpreter's wheels\n"
            "      (pgw#1657). Restore it: `echo 3.12 > .python-version`."
        )
    raw = path.read_text().strip()
    match = _PIN.match(raw)
    if not match:
        return [], (
            f".python-version says {raw!r}, which names no X.Y version.\n"
            "      Write a bare `3.12` — uv reads this file to choose the interpreter."
        )
    major, minor = match.group(1), match.group(2)
    return [f"cp{major}{minor}", f"py{major}{minor}", f"py{major}", "py2.py3"], raw


def _wheel_tags(url: str) -> tuple[list[str], list[str]] | None:
    """Return (python tags, platform tags) of a wheel URL, or None if not one."""
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    if not name.endswith(".whl"):
        return None
    fields = name[: -len(".whl")].split("-")
    if len(fields) < 5:
        return None
    return fields[-3].split("."), fields[-1].split(".")


def _selectable_on_linux(url: str, accepted: list[str]) -> bool:
    tags = _wheel_tags(url)
    if tags is None:
        return False
    python_tags, platform_tags = tags
    if not any(tag in accepted for tag in python_tags):
        return False
    return any(
        tag == "any" or "linux" in tag  # manylinux_*, musllinux_*, linux_*
        for tag in platform_tags
    )


def check(root: Path) -> list[str]:
    """Return one human-readable failure per hole (empty == green)."""
    accepted, pin = _pinned_tags(root)
    if not accepted:
        return [pin or ".python-version is unreadable."]

    lock = tomllib.loads((root / "uv.lock").read_text())
    failures: list[str] = []
    for package in lock.get("package", []):
        name = package.get("name", "?")
        version = package.get("version", "?")
        index = package.get("source", {}).get("registry", "?")
        for wheel in package.get("wheels", []):
            url = str(wheel.get("url", ""))
            if wheel.get("hash") or not _selectable_on_linux(url, accepted):
                continue
            failures.append(
                f"{name}=={version}: uv.lock records NO hash for a wheel this tree's\n"
                f"      interpreter selects on Linux.\n"
                f"      .python-version  {pin}\n"
                f"      wheel            {unquote(url.rsplit('/', 1)[-1])}\n"
                f"      index            {index}\n"
                f"      `uv sync --locked` will download it and check it against a\n"
                f"      DIFFERENT row's hash, then report `Hash mismatch` and blame\n"
                f"      the index (pgw#1657). Either pin an interpreter the index\n"
                f"      hashes, or move {name} to an index that hashes every row."
            )
    return failures


_SELFTEST_LOCK = """
version = 1
[[package]]
name = "torch"
version = "2.13.0+cu130"
source = {{ registry = "https://download.pytorch.org/whl/cu130" }}
wheels = [
    {{ url = "https://r2.invalid/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"{hashed} }},
    {{ url = "https://r2.invalid/whl/cu130/torch-2.13.0%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl" }},
    {{ url = "https://r2.invalid/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-win_amd64.whl" }},
]
"""

_HASH = ', hash = "sha256:' + "0" * 64 + '"'


def _selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "uv.lock").write_text(_SELFTEST_LOCK.format(hashed=_HASH))

        (root / ".python-version").write_text("3.12\n")
        green = check(root)
        if green:
            print(f"SELFTEST FAILED: a hashed cp312 linux row went red: {green}")
            return 1

        (root / "uv.lock").write_text(_SELFTEST_LOCK.format(hashed=""))
        red = check(root)
        if not red:
            print("SELFTEST FAILED: a hashless cp312 linux row went GREEN")
            return 1
        message = "\n".join(red)
        for owed in ("torch==2.13.0+cu130", "cp312-cp312-manylinux", ".python-version"):
            if owed not in message:
                print(f"SELFTEST FAILED: the verdict never names {owed!r}:\n{message}")
                return 1

        # Scoped, not blanket: the hashless cp313 and win_amd64 rows are not
        # selectable here, so restoring only the cp312 hash must go green.
        (root / "uv.lock").write_text(_SELFTEST_LOCK.format(hashed=_HASH))
        if check(root):
            print("SELFTEST FAILED: the check is blanket, not scoped to the pin")
            return 1

        # ... and the same lock IS red for the interpreter uv would have picked.
        (root / ".python-version").write_text("3.13\n")
        if not check(root):
            print("SELFTEST FAILED: pinning 3.13 over hashless cp313 rows went green")
            return 1

        (root / ".python-version").unlink()
        unpinned = check(root)
        if not unpinned or ".python-version" not in unpinned[0]:
            print(f"SELFTEST FAILED: an unpinned tree went green: {unpinned}")
            return 1

    print("selftest ok: hashless-under-the-pin red, off-pin rows ignored, unpinned red")
    return 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()

    root = Path(argv[1]) if len(argv) > 1 else ROOT
    failures = check(root)
    if not failures:
        return 0

    print(
        "pgw#1657: this tree cannot `uv sync --locked` on a cold cache.\n"
        "\n"
        "  It is the FIRST step of every job in every workflow, and it fails\n"
        "  with `Hash mismatch`, naming the index instead of the cause.\n",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
