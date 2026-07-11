"""gw#492: gen_worker.models.refs is the ONLY ref formatter/parser surface.

Model-ref strings have ONE normal form (see refs.py module docstring). Every
mint goes through wire_ref / fold_ref / format_model_ref / .canonical(), and
every decode through parse_model_ref. This guard (the env-surface pattern,
tests/test_env_surface.py) rejects new ad-hoc sites: tag/flavor grammar
tokens spliced or split by hand outside the grammar module.

If this test fails on your new code, call the refs.py function instead of
re-implementing it. Genuinely new grammar surface belongs IN refs.py.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "gen_worker"

# Ad-hoc grammar spellings forbidden outside the grammar module. Each pattern
# is a symptom of a second parser/formatter growing back:
#   - splitting tag/flavor/digest tokens off a ref string by hand
#   - stamping or testing the default-tag literal
#   - inlining the flavor-token colon hygiene
_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hand-rolled flavor split", re.compile(r"""\.r?split\(["']#["']""")),
    ("hand-rolled flavor partition", re.compile(r"""\.r?partition\(["']#["']""")),
    ("hand-rolled tag split", re.compile(r"""\.rsplit\(["']:["']""")),
    ("default-tag literal", re.compile(r"""["']:(?:latest|prod)["']""")),
    ("inline flavor-token hygiene", re.compile(r"""\.replace\(["']:["'],\s*["']-["']\)""")),
)

# The grammar module itself, plus sites whose hits are NOT model-ref grammar.
# relpath -> substrings; a hit line must contain one of them to be allowed.
ALLOWED: dict[str, tuple[str, ...]] = {
    # THE grammar module: parser + formatter live here by definition.
    "models/refs.py": ("split", "partition", "replace", ":latest", ":prod"),
}


def _hits() -> list[str]:
    out: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        allowed = ALLOWED.get(rel, ())
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for label, pat in _FORBIDDEN:
                if pat.search(line) and not any(tok in line for tok in allowed):
                    out.append(f"{rel}:{lineno}: {label}: {line.strip()}")
    return out


def test_no_ad_hoc_ref_grammar_sites() -> None:
    hits = _hits()
    assert not hits, (
        "ad-hoc model-ref grammar outside gen_worker/models/refs.py "
        "(use parse_model_ref / fold_ref / wire_ref / flavor_token):\n"
        + "\n".join(hits)
    )
