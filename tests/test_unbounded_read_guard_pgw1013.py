"""pgw#1013 / th#1662 — the ratchet under the absence sweep.

The §4.24 bounds census enumerated bounds that EXIST and adjudicated each. It is
blind by construction to a read site that needs a bound and has none, and it
proved that by walking past two real ones in files it had already swept.

`scripts/lint_unbounded_reads.py` is what stops the class returning. This file
tests the guard itself, because a guard nobody tests is a guard that silently
stops guarding — and this one is the only thing standing between the repo and a
defect class that a full census could not see.

The synthetic cases below are deliberately NOT drawn from the real tree: the
guard must be provably correct on shapes that do not exist yet, or it only
documents the past.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "scripts" / "lint_unbounded_reads.py"

sys.path.insert(0, str(REPO / "scripts"))
from lint_unbounded_reads import scan_file  # noqa: E402


def _scan(src: str) -> list[str]:
    """Run the guard's analyser over a synthetic module."""
    return scan_file(REPO / "src" / "gen_worker" / "_synthetic.py", src)


# --------------------------------------------------------------------------
# The guard holds on the real tree
# --------------------------------------------------------------------------

def test_guard_passes_on_the_real_tree():
    """Every external length feeding a read in src/ is bounded or justified.

    This is the assertion the sweep exists to make true, run against the
    shipping code rather than a fixture.
    """
    proc = subprocess.run(
        [sys.executable, str(GUARD)], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, (
        "the unbounded-read guard found a regression:\n"
        f"{proc.stdout}\n{proc.stderr}")


# --------------------------------------------------------------------------
# It catches the shape the census could not see
# --------------------------------------------------------------------------

def test_catches_struct_unpack_length_sizing_a_read():
    """The exact shape of both original specimens."""
    findings = _scan(
        "import struct, json\n"
        "def parse(f):\n"
        "    (n,) = struct.unpack('<Q', f.read(8))\n"
        "    return json.loads(f.read(n))\n"
    )
    assert len(findings) == 1, findings
    assert "`n`" in findings[0]
    assert "parse()" in findings[0]


def test_catches_int_from_bytes_length_sizing_a_read():
    """The other prefix idiom in this repo."""
    findings = _scan(
        "import json\n"
        "def parse(f):\n"
        "    header_len = int.from_bytes(f.read(8), 'little')\n"
        "    return json.loads(f.read(header_len))\n"
    )
    assert len(findings) == 1, findings
    assert "`header_len`" in findings[0]


def test_catches_subscripted_unpack():
    findings = _scan(
        "import struct\n"
        "def parse(f):\n"
        "    n = struct.unpack('<Q', f.read(8))[0]\n"
        "    return f.read(n)\n"
    )
    assert len(findings) == 1, findings


# --------------------------------------------------------------------------
# It accepts BOTH answers §4.24 allows — a bound, or a stated reason
# --------------------------------------------------------------------------

def test_sanctioned_validator_satisfies_the_guard():
    findings = _scan(
        "import struct, json\n"
        "from .models.safetensors_header import header_len_ok\n"
        "def parse(f):\n"
        "    (n,) = struct.unpack('<Q', f.read(8))\n"
        "    if not header_len_ok(n):\n"
        "        raise ValueError('bad')\n"
        "    return json.loads(f.read(n))\n"
    )
    assert findings == []


def test_justification_comment_satisfies_the_guard():
    """§4.24 asks for a bound OR a stated reason none is needed.

    A guard that accepted only the bound would push honest exemptions into
    silence — or worse, into a bound added just to quiet the linter, which is
    the 'defence in depth' anti-pattern the ruling rejects.
    """
    findings = _scan(
        "import struct, json\n"
        "def parse(f):\n"
        "    (n,) = struct.unpack('<Q', f.read(8))\n"
        "    # bound-justified: n was written by this process moments ago.\n"
        "    return json.loads(f.read(n))\n"
    )
    assert findings == []


def test_justification_may_sit_a_few_lines_above():
    findings = _scan(
        "import struct, json\n"
        "def parse(f):\n"
        "    (n,) = struct.unpack('<Q', f.read(8))\n"
        "    # bound-justified: a real reason usually needs more than one line,\n"
        "    # so the guard reads the comment block above the read too.\n"
        "    #\n"
        "    payload = f.read(n)\n"
        "    return json.loads(payload)\n"
    )
    assert findings == []


def test_a_distant_justification_does_NOT_carry():
    """The escape hatch must not be inheritable.

    If a justification written for one read silently excused every later read in
    the same function, the hatch would widen itself over time — which is how an
    exemption mechanism becomes a blanket.
    """
    body = (
        "import struct, json\n"
        "def parse(f):\n"
        "    (n,) = struct.unpack('<Q', f.read(8))\n"
        "    # bound-justified: this excuse belongs to the read directly below.\n"
        "    first = f.read(n)\n"
        + "    x = 1\n" * 20
        + "    second = f.read(n)\n"
        "    return first, second\n"
    )
    findings = _scan(body)
    assert len(findings) == 1, findings
    assert "`n`" in findings[0]


# --------------------------------------------------------------------------
# It stays quiet where it should
# --------------------------------------------------------------------------

@pytest.mark.parametrize("src", [
    # A literal length is not external input.
    "def parse(f):\n    return f.read(8)\n",
    # A length that never sizes anything is not this guard's business.
    "import struct\n"
    "def parse(f):\n"
    "    (n,) = struct.unpack('<Q', f.read(8))\n"
    "    return n > 0\n",
    # A name that was never a length prefix.
    "def parse(f, n):\n    return f.read(n)\n",
])
def test_no_false_positive(src: str):
    assert _scan(src) == []
