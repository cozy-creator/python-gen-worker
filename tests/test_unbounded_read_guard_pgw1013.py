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
    return scan_file(REPO / "src" / "gen_worker" / "_synthetic.py", src)


def test_guard_passes_on_the_real_tree():
    """Every external length feeding a read in src/ is bounded or justified."""
    proc = subprocess.run(
        [sys.executable, str(GUARD)], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, (
        "the unbounded-read guard found a regression:\n"
        f"{proc.stdout}\n{proc.stderr}")


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
    """§4.24 asks for a bound OR a stated reason none is needed."""
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
    """The escape hatch must not be inheritable."""
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


@pytest.mark.parametrize("src", [
    "def parse(f):\n    return f.read(8)\n",
    "import struct\n"
    "def parse(f):\n"
    "    (n,) = struct.unpack('<Q', f.read(8))\n"
    "    return n > 0\n",
    "def parse(f, n):\n    return f.read(n)\n",
])
def test_no_false_positive(src: str):
    assert _scan(src) == []


def test_catches_the_check_after_the_loop_shape():
    """The exact shape of `_download_url_streamed` and `_civitai_stream_one`: a counter that exists, and is compared one line too late."""
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
    """`_download_blob_by_digest` — the worst of the four."""
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
    """The CLI's NDJSON readers drain the buffer as lines complete, so an accumulator would overcount across drains and `len(buf)` is the only honest measurement."""
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
    """A progress logger may compare ``downloaded - last_log``."""
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
    "def take(src, fd, remaining):\n"
    "    buf = bytearray()\n"
    "    while len(buf) < remaining:\n"
    "        b = src.recv(4096)\n"
    "        if not b:\n"
    "            break\n"
    "        buf.extend(b)\n"
    "    return buf\n",
    "def upload(path, stream):\n"
    "    with open(path, 'rb') as fin:\n"
    "        while True:\n"
    "            chunk = fin.read(8 << 20)\n"
    "            if not chunk:\n"
    "                break\n"
    "            stream.write(chunk)\n",
    "def plan(resp, out):\n"
    "    import hashlib\n"
    "    h = hashlib.sha256()\n"
    "    for chunk in resp.iter_content(chunk_size=1 << 20):\n"
    "        h.update(chunk)\n"
    "        out.append(len(chunk))\n",
])
def test_rule_2_no_false_positive(src: str):
    assert _scan(src) == []


PRE_FIX_SHAPES = {
    "request_context._download_blob_by_digest": (
        "def fetch(resp, dest):\n"
        "    with open(dest, 'wb') as f:\n"
        "        for chunk in resp.iter_content(chunk_size=1024 * 1024):\n"
        "            if chunk:\n"
        "                f.write(chunk)\n"
    ),
    "aot_compiled_graphs whole-file branch": (
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
    """The ratchet, on every shape this issue closed."""
    findings = _scan(PRE_FIX_SHAPES[site])
    assert len(findings) == 1, f"{site}: {findings}"
    assert "streams external bytes" in findings[0]


def test_rule_2_fires_on_all_four_filed_sites_as_they_were():
    """The same proof against the REAL files, which is what keeps the transcriptions above honest — a fixture that drifted from the code it quotes would assert about a shape that never existed."""
    PRE_FIX = "d7881b40"
    sites = [
        "src/gen_worker/request_context/__init__.py",
        "src/gen_worker/aot_compiled_graphs.py",
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
