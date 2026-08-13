"""th#1897 / pgw#1213 — the compiled-graph key grammar is a CROSS-REPO
contract, and this is the offline half of the fence.

Why the file exists, in one paragraph: pgw#1176 re-keyed the fleet — scheme,
store schema, resolve/publish wire, pack format and the proto — and the
tensorhub half of the validator was never written. Both repos were internally
consistent. Both CIs were green. The disagreement was observable only on a GPU
pod, 45 minutes into a compile, at the publish gate. A docstring here already
named the missing owner; a docstring is not a gate.

So the vectors are the contract. ``tests/testdata/compiled_graph_key_vectors.json``
is vendored byte-identically in tensorhub
(``internal/orchestrator/compilecache/testdata/``); each repo runs its own
implementation against its own copy, each pins that copy to the digest BOTH
repos commit, and ``scripts/grammar-vector-drift.sh`` (verbatim in both) proves
the two files are equal. A one-sided hand-edit fails HERE, in the tree where it
happened, with no network and no credentials.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

from gen_worker import cell_key
from gen_worker.compile_cache import parse_cell_ref
from gen_worker.models.refs import parse_model_ref
from gen_worker.refgrammar import MAX_FRAGMENT_LEN

TESTDATA = pathlib.Path(__file__).parent / "testdata"
KEY_VECTORS = TESTDATA / "compiled_graph_key_vectors.json"
KEY_DIGEST_FILE = TESTDATA / "KEY_GRAMMAR_DIGEST"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

HEX56 = "7692c3ad3540bb803c020b3aee66cd8887123234ea0c6e7143c0add7"


def _flavor_of(ref: str) -> str:
    """The parsed `#fragment`, refusing the None branch loudly — a ref that
    did not parse as a tensorhub ref is the bug these rows are about."""
    th = parse_model_ref(ref).tensorhub
    assert th is not None, ref
    return th.flavor or ""


def _vectors() -> list[dict]:
    return json.loads(KEY_VECTORS.read_text())["vectors"]


@pytest.mark.parametrize(
    "vector", _vectors(), ids=lambda v: (v["key"][:40] or "<empty>")
)
def test_is_key_answers_every_shared_vector(vector: dict) -> None:
    """``is_key`` and ``compilecache.IsCompiledGraphKey`` answer identically.

    Each row carries its own reason in ``note``; the ones that decide the
    design are the hyphenated scheme (``cg-key-v1``), the future scheme
    (``cg-key-v2``, admitted — th#1183 refuses shape, never scheme), and the
    scheme containing a hex run, which is the regression vector for ever
    splitting a key on ``-``.
    """
    assert cell_key.is_key(vector["key"]) is vector["valid"], vector["note"]


def test_the_corpus_matches_the_digest_both_repos_commit() -> None:
    """A one-sided edit of the corpus is red here, offline.

    tensorhub's ``TestCompiledGraphKeyVectorDigest_TH1897`` is the same
    assertion over the same bytes; the digest file is the shared value.
    """
    actual = hashlib.sha256(KEY_VECTORS.read_bytes()).hexdigest()
    recorded = ""
    for line in KEY_DIGEST_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            recorded = line.split()[0]
            break
    assert recorded == actual, (
        f"the shared key-grammar corpus changed without its digest\n"
        f"  recorded: {recorded}\n  actual:   {actual}\n"
        f"Editing the corpus is a CROSS-REPO cut: land this repo's half first "
        f"(tensorhub vendors the file byte-for-byte), then record {actual} in "
        f"KEY_GRAMMAR_DIGEST in BOTH repos."
    )


def test_no_blessed_scheme_list_anywhere_in_the_grammar() -> None:
    """th#1183 as an executable rule rather than a comment (tensorhub states
    the twin as ``TestSchemeAgnosticism``): the grammar refuses SHAPE, never
    scheme.

    This is the row that refuses the "hard-cut to `cg-key-v1` only" reading —
    which pgw#1213's first pass actually shipped, and which this branch backs
    out. Pinning the scheme means a newer fleet's key stops being addressable
    by an older hub, so hub and fleet can never again ship in different
    windows.
    """
    minted = cell_key.KEY_SCHEME
    for scheme in (minted, "cg-key-v2", "ck1", "ek1", "a", "cg.key_v1"):
        assert cell_key.is_key(f"{scheme}-{HEX56}"), scheme


def test_the_key_the_deriver_mints_is_a_key() -> None:
    """The agreement between the two halves of THIS repo, which is what goes
    red if a scheme change is made in one and not the other."""
    key = cell_key.from_axes(
        {"graph": "0f0e0d0c0b0a0908", "sm": "sm_100",
         "toolchain": "bb11cc22dd33ee44"}
    ).digest
    assert key.startswith(cell_key.KEY_SCHEME + "-")
    assert cell_key.is_key(key) is True


# ---------------------------------------------------------------------------
# The boundary, and the break it was found by
# ---------------------------------------------------------------------------


def test_the_real_key_length_fits() -> None:
    """66 — `cg-key-v1-` + 56 hex. THE number the old 64-byte cap refused,
    which is the whole reason the bound moved."""
    key = f"{cell_key.KEY_SCHEME}-{HEX56}"
    assert len(key) == 66
    assert cell_key.is_key(key)
    assert _flavor_of(f"root/family-sdxl#{key}") == key


@pytest.mark.parametrize(
    "length,valid", [(MAX_FRAGMENT_LEN, True), (MAX_FRAGMENT_LEN + 1, False)]
)
def test_the_fragment_bound_is_96_exactly(length: int, valid: bool) -> None:
    """96 accepted, 97 refused — mirroring tensorhub's
    ``refgrammar.MaxFragmentLen``. Both corpora carry this pair, so the number
    cannot move on one side alone."""
    assert MAX_FRAGMENT_LEN == 96
    key = "c" * (length - 1 - len(HEX56)) + "-" + HEX56
    assert len(key) == length
    assert cell_key.is_key(key) is valid
    ref = f"root/family-sdxl#{key}"
    if valid:
        assert _flavor_of(ref) == key
    else:
        with pytest.raises(ValueError):
            parse_model_ref(ref)


def test_a_pod_can_name_the_family_of_what_it_just_armed() -> None:
    """THE break, end to end and pod-side only — no hub appears in it.

    A compiled-graph key travels as the ``#fragment`` of
    ``root/family-<f>#<key>``. Under the 64-byte cap ``parse_model_ref``
    raised, ``parse_cell_ref`` swallowed that into ``("", "")``, and
    ``aot_serve.is_aot_ref`` therefore returned False for an artifact the
    process had armed itself.
    """
    key = f"{cell_key.KEY_SCHEME}-{HEX56}"
    assert parse_cell_ref(f"root/family-sdxl#{key}") == ("sdxl", key)


# ---------------------------------------------------------------------------
# Layer 2 of the fence
# ---------------------------------------------------------------------------


def test_the_drift_script_is_committed_and_runnable_offline() -> None:
    """``scripts/grammar-vector-drift.sh`` is committed VERBATIM in both repos
    and compares this tree's corpora against the peer's. Driven here through
    ``GRAMMAR_PEER_DIR`` against this same checkout: a script that cannot pass
    when the two sides ARE equal is not a gate anybody can act on.
    """
    script = REPO_ROOT / "scripts" / "grammar-vector-drift.sh"
    assert script.exists()
    proc = subprocess.run(
        ["bash", str(script)],
        env={"PATH": "/usr/bin:/bin", "GRAMMAR_PEER_DIR": str(TESTDATA)},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "2 corpora agree" in proc.stdout, proc.stdout


def test_the_drift_script_fails_when_a_corpus_differs(
    tmp_path: pathlib.Path,
) -> None:
    """And it goes red when they do not — the half that makes the row above
    mean something."""
    peer = tmp_path / "peer"
    peer.mkdir()
    (peer / "ref_grammar_vectors.json").write_bytes(
        (TESTDATA / "ref_grammar_vectors.json").read_bytes()
    )
    (peer / "compiled_graph_key_vectors.json").write_text("{}\n")
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "grammar-vector-drift.sh")],
        env={"PATH": "/usr/bin:/bin", "GRAMMAR_PEER_DIR": str(peer)},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1, proc.stdout
    assert "compiled_graph_key_vectors.json differs" in proc.stderr


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
