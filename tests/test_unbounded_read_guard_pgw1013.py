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


# --------------------------------------------------------------------------
# Rule 2 — the streaming-copy loop ("check AFTER the loop")
#
# The dominant residual class the wave-1 sweep tabled: four downloaders wrote
# a whole remote body to disk and compared sizes only once it had ended. The
# rule is calibrated against the four siblings that already checked in-loop —
# it must pass every one of them and fail every one of the four that were
# fixed, which the "real tree" and "against the pre-fix sources" tests below
# assert with the shipping code rather than with fixtures.
# --------------------------------------------------------------------------

def test_catches_the_check_after_the_loop_shape():
    """The exact shape of `_download_url_streamed` and `_civitai_stream_one`:
    a counter that exists, and is compared one line too late."""
    findings = _scan(
        "def fetch(resp, dest, expected):\n"
        "    total = 0\n"
        "    with open(dest, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1 << 20):\n"
        "            f.write(chunk)\n"
        "            total += len(chunk)\n"
        "    if total != expected:\n"
        "        raise ValueError('too big')\n"
    )
    assert len(findings) == 1, findings
    assert "`total` counts the bytes" in findings[0]


def test_catches_a_stream_loop_with_no_count_at_all():
    """`_download_blob_by_digest` — the worst of the four. Nothing to compare,
    because nothing was counted."""
    findings = _scan(
        "def fetch(resp, dest):\n"
        "    with open(dest, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1 << 20):\n"
        "            if chunk:\n"
        "                f.write(chunk)\n"
    )
    assert len(findings) == 1, findings
    assert "no running byte count" in findings[0]


def test_the_in_loop_check_satisfies_rule_2():
    """The siblings' shape, which is the authority the fix copied."""
    findings = _scan(
        "def fetch(resp, dest, cap):\n"
        "    total = 0\n"
        "    with open(dest, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1 << 20):\n"
        "            total += len(chunk)\n"
        "            if total > cap:\n"
        "                raise ValueError('too big')\n"
        "            f.write(chunk)\n"
    )
    assert findings == []


def test_a_len_reading_of_a_drained_buffer_counts_as_the_bound():
    """The CLI's NDJSON readers drain the buffer as lines complete, so an
    accumulator would overcount across drains and `len(buf)` is the only honest
    measurement. Rule 2 accepts it — a rule that demanded one spelling would
    have pushed those two sites into a wrong fix."""
    findings = _scan(
        "def read_line(conn, cap):\n"
        "    buf = bytearray()\n"
        "    while b'\\n' not in buf:\n"
        "        chunk = conn.recv(65536)\n"
        "        if not chunk:\n"
        "            break\n"
        "        if len(buf) + len(chunk) > cap:\n"
        "            return None\n"
        "        buf.extend(chunk)\n"
        "    return buf\n"
    )
    assert findings == []


def test_a_progress_log_is_not_mistaken_for_a_bound():
    """A progress logger may compare ``downloaded - last_log``. A rule that
    accepted a counter anywhere inside a comparison would read that as the
    bound and pass an unbounded loop."""
    findings = _scan(
        "def fetch(resp, dest, log_every):\n"
        "    total = 0\n"
        "    last = 0\n"
        "    with open(dest, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1 << 20):\n"
        "            f.write(chunk)\n"
        "            total += len(chunk)\n"
        "            if total - last >= log_every:\n"
        "                last = total\n"
    )
    assert len(findings) == 1, findings


def test_an_emptiness_test_is_not_a_bound():
    findings = _scan(
        "def fetch(resp, dest):\n"
        "    total = 0\n"
        "    with open(dest, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1 << 20):\n"
        "            total += len(chunk)\n"
        "            if total > 0:\n"
        "                f.write(chunk)\n"
    )
    assert len(findings) == 1, findings


def test_rule_2_takes_the_justification_comment_too():
    findings = _scan(
        "def fetch(resp, dest):\n"
        "    with open(dest, 'wb') as f:\n"
        "        # bound-justified: the body is this process's own output,\n"
        "        # read back over loopback from a server it started.\n"
        "        for chunk in resp.iter_content(chunk_size=1 << 20):\n"
        "            f.write(chunk)\n"
    )
    assert findings == []


@pytest.mark.parametrize("src", [
    # A budget loop: the loop's own test is the bound (for example,
    # `procsplit.child._recv_exact`).
    "def take(src, fd, remaining):\n"
    "    buf = bytearray()\n"
    "    while len(buf) < remaining:\n"
    "        b = src.recv(4096)\n"
    "        if not b:\n"
    "            break\n"
    "        buf.extend(b)\n"
    "    return buf\n",
    # A handle THIS function opened is local IO, not an external stream —
    # every upload path in the repo reads its own artifact back in blocks.
    "def upload(path, stream):\n"
    "    with open(path, 'rb') as fin:\n"
    "        while True:\n"
    "            chunk = fin.read(8 << 20)\n"
    "            if not chunk:\n"
    "                break\n"
    "            stream.write(chunk)\n",
    # The loop derives facts and drops the bytes; nothing accumulates.
    "def plan(resp, out):\n"
    "    import hashlib\n"
    "    h = hashlib.sha256()\n"
    "    for chunk in resp.iter_content(chunk_size=1 << 20):\n"
    "        h.update(chunk)\n"
    "        out.append(len(chunk))\n",
])
def test_rule_2_no_false_positive(src: str):
    assert _scan(src) == []


# The four loops as they stood at d7881b40, transcribed verbatim from the
# shipping files (imports and surrounding logic dropped; the loop and the check
# around it are byte-for-byte). This is the CI-runnable half of the proof — the
# git-backed test below reads the real files and can only run on a full clone.
PRE_FIX_SHAPES = {
    "request_context._download_blob_by_digest": (
        "def fetch(resp, dest):\n"
        "    with open(dest, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1024 * 1024):\n"
        "            if chunk:\n"
        "                f.write(chunk)\n"
    ),
    "aot_cells whole-file branch": (
        "def fetch(dl, tmp, want_ref):\n"
        "    with open(tmp, 'wb') as f:\n"
        "        for chunk in dl.iter_content(1 << 20):\n"
        "            f.write(chunk)\n"
        "    verify_file_digest(tmp, want_ref)\n"
    ),
    "models/download._civitai_stream_one": (
        "def fetch(resp, tmp, h, on_bytes, expected_size, dst):\n"
        "    written = 0\n"
        "    with open(tmp, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):\n"
        "            if not chunk:\n"
        "                continue\n"
        "            f.write(chunk)\n"
        "            h.update(chunk)\n"
        "            written += len(chunk)\n"
        "            on_bytes(len(chunk))\n"
        "    if expected_size and abs(written - expected_size) > 1024:\n"
        "        raise ValueError('size mismatch')\n"
    ),
    "_datasets._download_url_streamed": (
        "def fetch(resp, tmp, hasher, expected_size):\n"
        "    total = 0\n"
        "    with open(tmp, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1 << 20):\n"
        "            if not chunk:\n"
        "                continue\n"
        "            f.write(chunk)\n"
        "            total += len(chunk)\n"
        "            hasher.update(chunk)\n"
        "    if expected_size is not None and total != int(expected_size):\n"
        "        raise RuntimeError('shard size mismatch')\n"
    ),
}


@pytest.mark.parametrize("site", sorted(PRE_FIX_SHAPES))
def test_rule_2_fires_on_each_filed_shape(site: str):
    """The ratchet, on every shape this issue closed.

    A rule that only passes on the fixed tree proves nothing about whether it
    would have caught the defect. Each of these is the loop the defect actually
    had, including the post-loop comparison that made it LOOK checked.
    """
    findings = _scan(PRE_FIX_SHAPES[site])
    assert len(findings) == 1, f"{site}: {findings}"
    assert "streams external bytes" in findings[0]


def test_rule_2_fires_on_all_four_filed_sites_as_they_were():
    """The same proof against the REAL files, which is what keeps the
    transcriptions above honest — a fixture that drifted from the code it
    quotes would assert about a shape that never existed.

    Needs the pre-fix commit, so it cannot run on CI's shallow checkout; the
    parametrized test above is where this property is measured there. That
    disposition is recorded in `scripts/skip_census.txt`.
    """
    # Pinned rather than derived: `merge-base HEAD origin/master` moves to the
    # fix itself the moment this lands, and the test would then assert the
    # guard fires on code that no longer has the defect. This sha is the commit
    # the fix branched from and is immutable. CI checks out shallow, so a miss
    # is a skip, not a failure — the local run is where this earns its keep.
    PRE_FIX = "d7881b40"
    sites = [
        "src/gen_worker/request_context/__init__.py",
        "src/gen_worker/aot_cells.py",
        "src/gen_worker/models/download.py",
        "src/gen_worker/request_context/_datasets.py",
    ]
    for site in sites:
        proc = subprocess.run(
            ["git", "show", f"{PRE_FIX}:{site}"],
            capture_output=True, text=True, cwd=REPO)
        if proc.returncode != 0 or not proc.stdout:
            pytest.skip(f"{PRE_FIX} not in this checkout (shallow clone)")
        findings = scan_file(REPO / site, proc.stdout)
        loop_findings = [f for f in findings if "streams external bytes" in f]
        assert loop_findings, f"the guard would have walked past {site}"
