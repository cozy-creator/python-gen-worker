"""pgw#1606 acceptance (d) — the ONE reader for hardware/numerics branches in
an endpoint's `load()`.

Two halves, deliberately:

* **Inline fixtures** — always run, no sibling repo, no torch. They prove the
  reader finds what it must AND that it does not fire on the shapes that look
  similar and are fine (a comment saying a model does not branch; a helper
  named `load` that is not the ruled signature).
* **The real fleet** — runs only when `serverless-endpoints` is checked out
  beside this repo. It asserts the EXACT set of offenders the audit named, so
  the count is a ledger: it may go down as se#816 migrates endpoints, and a
  reader that quietly stops seeing them goes red rather than green.

The second half is why the first half exists. A cross-repo scan cannot run in
CI (there are no sibling checkouts there), and the workspace has been bitten
before by a scan that skipped a missing sibling and still printed green — so
the reader's own red arm is forced here, on fixtures, where it always runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gen_worker.serving import lane_fence as F


def _findings(src: str) -> tuple[F.Finding, ...]:
    return F.scan_source(src)


# --------------------------------------------------------------------------
# The reader fires on what it must
# --------------------------------------------------------------------------


def test_it_finds_a_capability_read():
    found = _findings("""
class M:
    def load(self, ctx):
        if torch.cuda.get_device_capability() >= (8, 9):
            self.fp8 = True
""")
    assert [f.kind for f in found] == [F.KIND_HARDWARE]
    assert found[0].name == "get_device_capability"


def test_it_finds_the_bf16_support_branch_joycaption_actually_has():
    found = _findings("""
class M:
    def load(self, ctx):
        dtype = None
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.pipe = P.from_pretrained(str(ctx.checkpoint_dir), torch_dtype=dtype)
""")
    kinds = {f.kind for f in found}
    assert F.KIND_HARDWARE in kinds and F.KIND_DTYPE in kinds
    assert {f.name for f in found} == {"is_bf16_supported", "torch_dtype"}


def test_it_finds_an_in_place_quantize_the_shape_minimax_h3_has():
    found = _findings("""
class M:
    def load(self, ctx):
        self.pipe = ctx.load(P)
        quantize_(self.pipe.transformer, cfg, filter_fn=fn)
""")
    assert [f.kind for f in found] == [F.KIND_NUMERICS]
    assert found[0].name == "quantize_"


def test_it_finds_the_fp8_storage_call_ltx_makes():
    found = _findings("""
class M:
    def load(self, ctx):
        pipe = ctx.load(P)
        apply_fp8_storage(pipe, components=("text_encoder",))
""")
    assert [f.name for f in found] == ["apply_fp8_storage"]


def test_it_finds_the_dequant_helper_anima_calls():
    found = _findings("""
class M:
    def load(self, ctx):
        sd = sanitize_w8a8_state_dict(sd, dtype)
""")
    assert [f.name for f in found] == ["sanitize_w8a8_state_dict"]


# --------------------------------------------------------------------------
# ...and does NOT fire on the shapes that merely look like it
# --------------------------------------------------------------------------


def test_a_clean_load_is_clean():
    assert _findings("""
class M:
    def load(self, ctx):
        self.pipe = ctx.load_pipeline(P)
        self.defaults = ctx.defaults()
        self.pipe.unet = ctx.compile(self.pipe.unet)
""") == ()


def test_a_comment_explaining_that_it_does_NOT_branch_is_not_a_branch():
    """Parsed, never grepped. A substring check would refuse exactly the
    classes that documented themselves best — which is the same reason
    `load_marks_compile` parses."""
    assert _findings('''
class M:
    def load(self, ctx):
        """This model does NOT call torch.cuda.get_device_capability() or
        quantize_(); the lane ladder decides. See pgw#1606."""
        # apply_fp8_storage(pipe) would be wrong here.
        self.pipe = ctx.load_pipeline(P)
''') == ()


def test_a_feature_branch_is_not_a_dtype_branch():
    """sdxl's FreeU branch is a FEATURE decision the author owns. The fence
    must not sweep it in — an over-broad fence gets disabled, and then it
    protects nothing."""
    assert _findings("""
class M:
    def load(self, ctx):
        self.pipe = ctx.load_pipeline(P)
        self.defaults = ctx.defaults()
        if self.defaults.enable_freeu:
            self.pipe.enable_freeu(b1=1.0, b2=1.0, s1=0.9, s2=0.2)
        else:
            self.pipe.unet = ctx.compile(self.pipe.unet)
""") == ()


def test_a_helper_named_load_with_a_different_signature_is_not_the_ruled_one():
    assert _findings("""
def load(path, dtype):
    return P.from_pretrained(path, torch_dtype=dtype)
""") == ()


def test_the_ruled_signature_is_matched_on_shape_not_on_the_ctx_name():
    found = _findings("""
class M:
    def load(self, context):
        torch.cuda.get_device_capability()
""")
    assert len(found) == 1, "load(self, <anything>) is the ruled two-arg shape"


# --------------------------------------------------------------------------
# The reader's own red arm, forced
# --------------------------------------------------------------------------


def test_the_refusal_names_the_rows_and_points_at_the_replacement():
    found = _findings("""
class M:
    def load(self, ctx):
        if torch.cuda.get_device_capability() >= (8, 9):
            quantize_(self.pipe.transformer, cfg)
""")
    message = F.refusal("MinimaxH3Model", found)
    assert "MinimaxH3Model.load() branches on hardware or numerics" in message
    assert "hardware_read:get_device_capability@" in message
    assert "numerics_call:quantize_@" in message
    assert "ctx.load_pipeline" in message


def test_a_non_function_node_is_answered_not_crashed_on():
    assert F.load_branches_on_hardware(ast.parse("x = 1")) == ()
    assert F.load_branches_on_hardware(None) == ()


# --------------------------------------------------------------------------
# The real fleet — a LEDGER, and it may only go down
# --------------------------------------------------------------------------


_FLEET = Path.home() / "cozy" / "serverless-endpoints"

#: MEASURED 2026-08-20 against the fleet, per endpoint: (branches inside
#: `load()`, branches anywhere in the endpoint package). se#816 drives both
#: columns to zero; this table is the countdown and it may only shrink.
#:
#: 🔴 READ THE SECOND COLUMN. It is the reason acceptance (d)'s wording —
#: "zero dtype/quant branches in endpoint `load()` bodies" — is very nearly
#: VACUOUS: 21 of the 23 branches live ONE FUNCTION outside `load()`, and four
#: of these five endpoints already pass the `load()`-scoped scan today with
#: every branch intact. A fence written to the letter of (d) would have gone
#: green on the day it landed and protected nothing.
FLEET_LEDGER: dict[str, tuple[int, int]] = {
    "anima": (0, 5),
    "joycaption": (2, 3),
    "ltx-video-2.3": (0, 4),
    "minimax-h3": (0, 8),
    "wan-2.2": (0, 3),
}


def _fleet_counts() -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for pkg in sorted(p for p in _FLEET.iterdir() if p.is_dir()):
        srcs = [f for f in pkg.glob("src/*/*.py") if ".venv" not in f.parts]
        if not srcs:
            continue
        in_load = anywhere = 0
        for f in srcs:
            text = f.read_text(encoding="utf-8", errors="replace")
            in_load += len(F.scan_source(text))
            anywhere += len(F.scan_module(text))
        if in_load or anywhere:
            counts[pkg.name] = (in_load, anywhere)
    return counts


@pytest.mark.skipif(not _FLEET.is_dir(),
                    reason="serverless-endpoints is not checked out beside "
                           "this repo (CI has no sibling checkouts) — the "
                           "reader's own red arm is forced on fixtures above, "
                           "which always run")
def test_the_fleet_ledger_matches_the_audit_and_may_only_shrink():
    counts = _fleet_counts()

    assert counts, (
        "the scan found NOTHING across the whole fleet, which contradicts the "
        "audit. Read the COUNT, not the verdict: an empty result here means "
        "the glob matched no files or the reader broke, not that the fleet "
        "migrated"
    )
    new = set(counts) - set(FLEET_LEDGER)
    assert not new, (
        f"NEW endpoints branch on hardware or numerics: "
        f"{ {k: counts[k] for k in sorted(new)} }. Lane selection is PLATFORM "
        f"machinery (pgw#1606) — declare the lanes and call ctx.load_pipeline"
    )
    grew = {
        name: (FLEET_LEDGER[name], counts[name])
        for name in counts
        if counts[name][0] > FLEET_LEDGER[name][0]
        or counts[name][1] > FLEET_LEDGER[name][1]
    }
    assert not grew, f"the ledger may only shrink; these grew: {grew}"


@pytest.mark.skipif(not _FLEET.is_dir(), reason="no sibling checkout")
def test_the_load_scoped_fence_is_measurably_weaker_than_the_module_one():
    """The finding itself, as an assertion — so that if someone later narrows
    the fence back to `load()` to make it pass, this goes red and says why.

    This is not pedantry. The workspace's own recurring defect is a guard that
    is green because of its scope rather than because of the tree, and it is
    invisible from either side once it lands."""
    counts = _fleet_counts()
    in_load = sum(c[0] for c in counts.values())
    anywhere = sum(c[1] for c in counts.values())
    assert anywhere > in_load, (
        "if these ever match, the module-wide scan stopped seeing helpers and "
        "the fence has quietly become the load()-scoped one again"
    )
    clean_by_scope = [n for n, c in counts.items() if c[0] == 0 and c[1] > 0]
    assert clean_by_scope, (
        "at least one endpoint must be 'clean in load(), dirty in the package' "
        "for this test to mean anything; if none are, the fleet migrated and "
        "this test and FLEET_LEDGER should both be deleted")
