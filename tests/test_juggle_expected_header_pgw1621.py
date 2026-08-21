"""pgw#1621: the juggle gate against the COMPUTED `quant(topology)` header.

Every expectation here is diffed against a REAL Go-produced layout, banked in
`tests/testdata/layout_v2/` (see its README for the exact `compute_layout`
command). Nothing in this file computes a layout, and nothing computes an
admission verdict: `quant(topology)` is the Go engine's and the verdict is the
hub bind gate's, so the worker's job — the only job this file exercises — is to
ENFORCE what it was handed.

The gate is shown RED three ways (wrong shape, wrong dtype, missing tensor) and
GREEN once, per fixture. A gate only ever proven green is a gate nobody has
proven is connected.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gen_worker._vendor.tensorfs.layout2 import ExpectedHeader  # noqa: E402
from gen_worker.models.checkpoint_juggle import (  # noqa: E402
    LANE_VERDICT_DERIVABLE,
    LANE_VERDICT_INCOMPATIBLE,
    LANE_VERDICT_SATISFIES,
    SlotSource,
    admission_refusal,
)

DATA = Path(__file__).resolve().parent / "testdata" / "layout_v2"

LTX2_UPSAMPLER = "ltx2-upsampler.diffusers@1+plain.bf16@1"
FLUX2_KLEIN = "flux2-klein.diffusers@1+cozy.nvfp4-flat@1"


def load(stamp: str) -> ExpectedHeader:
    plain = DATA / f"{stamp}.json"
    if plain.exists():
        return ExpectedHeader.from_document(plain.read_bytes())
    return ExpectedHeader.from_document(gzip.decompress((DATA / f"{stamp}.json.gz").read_bytes()))


def manifest_of(expected: ExpectedHeader) -> Dict[str, SlotSource]:
    """The header a CONFORMING checkpoint would present, as a manifest.

    Byte ranges are synthetic — the gate is header-only by construction, so it
    never opens a file — but the KEYS, SHAPES and DTYPE SPELLINGS are the
    computed layout's own, read out of the Go document rather than restated.
    Every dtype takes the FIRST (most canonical) spelling, so the
    reference-tolerant alternatives stay untested until a case asks for one.
    """
    multi = len(expected.components) > 1
    out: Dict[str, SlotSource] = {}
    for name, tensors in expected.components.items():
        for key, entry in tensors.items():
            out[f"{name}.{key}" if multi else key] = SlotSource(
                path=Path("/dev/null"),
                offset=0,
                length=0,
                dtype_code=0,
                dtype_bits=0,
                shape=entry.shape,
                spelling=entry.dtypes[0],
            )
    return out


@pytest.fixture(params=[LTX2_UPSAMPLER, FLUX2_KLEIN])
def expected(request: pytest.FixtureRequest) -> ExpectedHeader:
    return load(request.param)


# -- the fixtures are what they claim to be ---------------------------------


def test_the_banked_fixtures_are_the_pairs_they_are_named_for() -> None:
    """A fixture whose stamp drifted from its filename would silently prove
    something else. The name is the claim; this is the check."""
    for stamp in (LTX2_UPSAMPLER, FLUX2_KLEIN):
        header = load(stamp)
        assert header.stamp == stamp
        assert header.topology_digest and header.quant.digest
        assert header.quant.handle == stamp.split("+", 1)[1]


def test_the_klein_fixture_carries_the_two_cases_it_was_chosen_for() -> None:
    """If a re-vendor ever flattened these, the arm's two hard cases would go
    untested while every assertion below still passed."""
    header = load(FLUX2_KLEIN)
    assert len(header.components) == 3
    tensors = header.component("vae")
    tolerant = [k for k, v in tensors.items() if len(v.dtypes) > 1]
    assert tolerant == ["bn.num_batches_tracked"]
    assert tensors["bn.num_batches_tracked"].dtypes == ("BF16", "I64")
    optional = [k for k, v in header.component("transformer").items() if v.optional]
    assert len(optional) == 200


# -- GREEN ------------------------------------------------------------------


def test_the_real_computed_header_admits_a_conforming_checkpoint(
    expected: ExpectedHeader,
) -> None:
    assert (
        admission_refusal(
            expected, manifest_of(expected), verdict=LANE_VERDICT_SATISFIES
        )
        is None
    )


def test_a_named_component_admits_on_its_own_keys() -> None:
    """A multi-component layout is also checkable one component at a time —
    the manifest is then keyed WITHOUT the component prefix, exactly as
    `read_manifest` of that one directory keys it."""
    header = load(FLUX2_KLEIN)
    manifest = {
        key: SlotSource(
            path=Path("/dev/null"), offset=0, length=0, dtype_code=0,
            dtype_bits=0, shape=entry.shape, spelling=entry.dtypes[0],
        )
        for key, entry in header.component("vae").items()
    }
    assert admission_refusal(
        header, manifest, verdict=LANE_VERDICT_SATISFIES, component="vae"
    ) is None


def test_an_optional_tensor_may_be_absent() -> None:
    """`input_scale` / `pre_quant_scale` are what a calibration MAY not have
    produced. Dropping all 200 must stay green."""
    header = load(FLUX2_KLEIN)
    manifest = manifest_of(header)
    dropped = [
        key for key in list(manifest)
        if key.startswith("transformer.")
        and header.component("transformer")[key.removeprefix("transformer.")].optional
    ]
    assert len(dropped) == 200
    for key in dropped:
        del manifest[key]
    assert admission_refusal(
        header, manifest, verdict=LANE_VERDICT_SATISFIES
    ) is None


def test_a_reference_tolerant_key_accepts_every_spelling_the_rule_lists() -> None:
    """`bn.num_batches_tracked` really is I64 in the reference packaging the
    topology was extracted from. Accepting only `dtypes[0]` would refuse the
    checkpoint the topology came from."""
    header = load(FLUX2_KLEIN)
    key = "vae.bn.num_batches_tracked"
    for spelling in header.component("vae")["bn.num_batches_tracked"].dtypes:
        manifest = manifest_of(header)
        row = manifest[key]
        manifest[key] = SlotSource(
            path=row.path, offset=0, length=0, dtype_code=0, dtype_bits=0,
            shape=row.shape, spelling=spelling,
        )
        assert admission_refusal(
            header, manifest, verdict=LANE_VERDICT_SATISFIES
        ) is None, spelling


def test_an_extra_tensor_the_header_does_not_name_is_not_a_refusal(
    expected: ExpectedHeader,
) -> None:
    """Bytes this lane does not read. What the checkpoint IS was decided by the
    hub's stamping; this gate asks only whether the lane's needs are met."""
    manifest = manifest_of(expected)
    manifest["some.unrelated.buffer"] = SlotSource(
        path=Path("/dev/null"), offset=0, length=0, dtype_code=0,
        dtype_bits=0, shape=(7,), spelling="F32",
    )
    assert admission_refusal(
        expected, manifest, verdict=LANE_VERDICT_SATISFIES
    ) is None


# -- RED: the three byte-level mismatches -----------------------------------


def test_a_wrong_shape_is_refused_by_name(expected: ExpectedHeader) -> None:
    manifest = manifest_of(expected)
    key = sorted(manifest)[0]
    row = manifest[key]
    manifest[key] = SlotSource(
        path=row.path, offset=0, length=0, dtype_code=0, dtype_bits=0,
        shape=row.shape + (13,), spelling=row.spelling,
    )
    refusal = admission_refusal(
        expected, manifest, verdict=LANE_VERDICT_SATISFIES
    )
    assert refusal is not None
    assert key in refusal
    assert str(list(row.shape + (13,))) in refusal   # what the checkpoint says
    assert str(list(row.shape)) in refusal           # what the lane expects
    assert expected.stamp in refusal


def test_a_wrong_dtype_is_refused_by_name(expected: ExpectedHeader) -> None:
    manifest = manifest_of(expected)
    key = sorted(manifest)[0]
    row = manifest[key]
    assert row.spelling != "I8"
    manifest[key] = SlotSource(
        path=row.path, offset=0, length=0, dtype_code=0, dtype_bits=0,
        shape=row.shape, spelling="I8",
    )
    refusal = admission_refusal(
        expected, manifest, verdict=LANE_VERDICT_SATISFIES
    )
    assert refusal is not None
    assert key in refusal and "I8" in refusal and row.spelling in refusal


def test_a_missing_required_tensor_is_refused_by_name(
    expected: ExpectedHeader,
) -> None:
    manifest = manifest_of(expected)
    key = next(
        k for k in sorted(manifest)
        if not _entry_for(expected, k).optional
    )
    del manifest[key]
    refusal = admission_refusal(
        expected, manifest, verdict=LANE_VERDICT_SATISFIES
    )
    assert refusal is not None
    assert key in refusal and "absent" in refusal


def test_a_dtype_spelling_the_manifest_never_recorded_refuses(
    expected: ExpectedHeader,
) -> None:
    """An empty spelling must not read as 'nothing to check'."""
    manifest = manifest_of(expected)
    key = sorted(manifest)[0]
    row = manifest[key]
    manifest[key] = SlotSource(
        path=row.path, offset=0, length=0, dtype_code=0, dtype_bits=0,
        shape=row.shape, spelling="",
    )
    refusal = admission_refusal(
        expected, manifest, verdict=LANE_VERDICT_SATISFIES
    )
    assert refusal is not None and key in refusal


# -- RED: the binding-carried verdict ---------------------------------------


def test_a_missing_verdict_refuses_rather_than_admitting(
    expected: ExpectedHeader,
) -> None:
    """The worker ENFORCES a verdict; it never reaches one. A clean diff with
    no verdict behind it is exactly the second admit-author the v2 cut deleted,
    and the direction that fails silently is the permissive one."""
    refusal = admission_refusal(expected, manifest_of(expected))
    assert refusal is not None
    assert "verdict" in refusal and expected.stamp in refusal


@pytest.mark.parametrize(
    "verdict", [LANE_VERDICT_DERIVABLE, LANE_VERDICT_INCOMPATIBLE, "satisfied"]
)
def test_only_satisfies_admits(expected: ExpectedHeader, verdict: str) -> None:
    """Including a near-miss spelling: a verdict this worker does not recognise
    is refused, never treated as absent and never guessed at."""
    refusal = admission_refusal(
        expected, manifest_of(expected), verdict=verdict
    )
    assert refusal is not None and verdict in refusal


def test_the_verdict_gate_runs_before_the_byte_diff(
    expected: ExpectedHeader,
) -> None:
    """A refused verdict on a checkpoint that would ALSO fail the diff must
    name the verdict — the hub's answer is the reason, and re-deciding it from
    the bytes is the thing this arm must not do."""
    manifest = manifest_of(expected)
    manifest.clear()
    refusal = admission_refusal(
        expected, manifest, verdict=LANE_VERDICT_INCOMPATIBLE
    )
    assert refusal is not None and LANE_VERDICT_INCOMPATIBLE in refusal


# -- the ArenaLayout arm keeps its own signature ----------------------------


def test_the_arena_arm_refuses_the_v2_keywords() -> None:
    """One gate, two templates — and the keywords do not bleed. An arena layout
    is this process's own allocation; there is no carried decision for it."""

    class _Layout:
        regions = ()

    assert admission_refusal(_Layout(), {}) is None
    with pytest.raises(TypeError, match="ExpectedHeader arm"):
        admission_refusal(_Layout(), {}, verdict=LANE_VERDICT_SATISFIES)
    with pytest.raises(TypeError, match="ExpectedHeader arm"):
        admission_refusal(_Layout(), {}, component="vae")


def _entry_for(expected: ExpectedHeader, key: str):
    if len(expected.components) == 1:
        return next(iter(expected.components.values()))[key]
    component, _, rest = key.partition(".")
    return expected.component(component)[rest]


def test_the_fixture_readme_names_the_regenerating_command() -> None:
    """The fixtures cannot be regenerated from this repo — the evaluator is in
    Go, deliberately — so the command that produces them is banked beside
    them."""
    readme = (DATA / "README").read_text(encoding="utf-8")
    assert "go run ./scripts/compute_layout" in readme
    for stamp in (LTX2_UPSAMPLER, FLUX2_KLEIN):
        assert stamp in readme


def test_the_fixtures_stay_small() -> None:
    """A 300 KB sdxl header proves nothing these do not, and every reader of
    this directory pays for it."""
    for path in DATA.iterdir():
        if path.name != "README":
            assert path.stat().st_size < 32 * 1024, path.name


def test_the_computed_header_is_read_not_recomputed() -> None:
    """The vendored reader parses and accesses, and that is the whole of it.
    A `matches()` growing here is the second evaluator v2 exists to delete."""
    from gen_worker._vendor.tensorfs import layout2

    for banned in ("matches", "verdict", "evaluate", "eligible"):
        assert not hasattr(layout2.ExpectedHeader, banned)
        assert not hasattr(layout2, banned)


def test_json_fixture_round_trips_through_from_document() -> None:
    """`from_document` takes bytes, str or a parsed mapping — one identity."""
    raw = (DATA / f"{LTX2_UPSAMPLER}.json").read_bytes()
    a = ExpectedHeader.from_document(raw)
    b = ExpectedHeader.from_document(json.loads(raw))
    assert a == b
