"""th#1887: the behaviour-gate scanner must see the discriminated-locals shape.

`scripts/lint_config_reads.py` enforces BEHAVIOUR_GATES bidirectionally, so an
undeclared gate fails CI. But its scanner was purely syntactic: a read counted
only if it landed inside an `if` test or a predicate function's `return`. Three
real gates lived in a shape it could not see at all — read into a local, then
discriminated against a set of known strings — and sat misfiled as STANDALONE.

A registry that cannot see a whole gate SHAPE is a guard that cannot fire, which
is worse than the gates it misses: it reports OK. These tests pin the shape, not
the three instances, so the pass cannot be quietly narrowed back.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "lint_config_reads", REPO / "scripts" / "lint_config_reads.py")
assert _spec and _spec.loader
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)


def _scan(source: str) -> set[str]:
    """Env names the behaviour scanner finds in one module of source."""
    tree = ast.parse(source)
    consts = lint.EnvVisitor()
    consts.load_consts(tree)
    visitor = lint.BehaviourVisitor()
    visitor._scan_conditions(tree, consts.consts)
    visitor._scan_discriminated_locals(
        tree, consts.consts, lint._literal_collections(tree))
    return {name for _, name in visitor.hits}


def test_read_into_local_then_compared_is_a_gate() -> None:
    assert "X_MODE" in _scan(
        'import os\n'
        'def pick():\n'
        '    mode = os.environ.get("X_MODE", "auto").strip().lower()\n'
        '    if mode == "x264":\n'
        '        return 1\n'
        '    return 2\n')


def test_membership_against_a_module_level_tuple_is_a_gate() -> None:
    """svdq's shape: `raw not in SVDQ_ENGINES`, the alternatives named elsewhere."""
    assert "X_ENGINE" in _scan(
        'import os\n'
        'ENGINES = ("a", "b")\n'
        '_ENV = "X_ENGINE"\n'
        'def override() -> str:\n'
        '    raw = str(os.environ.get(_ENV, "") or "").strip().lower()\n'
        '    if raw and raw not in ENGINES:\n'
        '        raise ValueError(raw)\n'
        '    return raw\n')


def test_match_subject_is_a_gate() -> None:
    assert "X_KIND" in _scan(
        'import os\n'
        'def pick():\n'
        '    kind = os.environ.get("X_KIND", "")\n'
        '    match kind:\n'
        '        case "fast":\n'
        '            return 1\n'
        '    return 0\n')


def test_a_path_built_from_env_is_not_a_gate() -> None:
    """The read must be the WHOLE right-hand side once scalar wrappers peel.

    Without the root-only rule this pass accuses TORCHINDUCTOR_CACHE_DIR and
    PYTHONHASHSEED — a cache directory and a hash seed — of being behaviour
    switches, which is the false-positive flood that would get it turned off.
    """
    # The discriminator is deliberately a BARE NAME against a string constant,
    # so the ONLY thing that can keep this out of the results is the root-only
    # peel rule. Written the obvious way (`if str(base) == "/tmp"`) the test
    # would pass even with that rule deleted — a false-clean.
    assert "X_DIR" not in _scan(
        'import os\n'
        'from pathlib import Path\n'
        'def where():\n'
        '    base = Path(os.environ.get("X_DIR", "") or "")\n'
        '    if base == "/tmp":\n'
        '        return None\n'
        '    return base\n')


def test_env_inside_a_dict_literal_is_not_a_gate() -> None:
    assert "X_SEED" not in _scan(
        'import os\n'
        'def seal():\n'
        '    base = {"seed": os.environ.get("X_SEED", "")}\n'
        '    if base == "sentinel":\n'
        '        return None\n'
        '    return base\n')


def test_truthiness_alone_is_not_a_gate() -> None:
    """`if not x: x = default` is the dominant shape of genuine value config."""
    assert "X_URL" not in _scan(
        'import os\n'
        'def url():\n'
        '    u = os.environ.get("X_URL", "")\n'
        '    if not u:\n'
        '        u = "https://default"\n'
        '    return u\n')


def test_the_three_th1887_gates_are_still_seen_in_the_real_tree() -> None:
    """Instance-level backstop for the shape tests above."""
    found = set(lint.scan_behaviour())
    for site in (
        ("src/gen_worker/video_encode.py", "GEN_WORKER_VIDEO_ENCODER"),
        ("src/gen_worker/models/svdq.py", "GEN_WORKER_SVDQ_ENGINE"),
        ("src/gen_worker/models/native_kernels.py", "GEN_WORKER_NATIVE_KERNELS"),
    ):
        assert site in found, f"{site} went invisible again"
