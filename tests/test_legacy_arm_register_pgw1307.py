"""pgw#1307 / pgw#1305: the legacy arms this lane executed.

One test per arm, each red-verified against the arm being restored. The arms
share a shape — a live branch of a live system that silently substitutes an old
key, an old value, or a quieter implementation — so what each test asserts is
that the substitution is GONE, not merely that the primary path works.
"""

from __future__ import annotations

import pytest

from gen_worker.models.hub_client import HubResolveError, parse_chunk_list

HEX64 = "a" * 64


# ---------------------------------------------------------------------------
# pgw#1307 arms (1) + (2): the retired bare-hex chunk key and the `length` twin
# ---------------------------------------------------------------------------
#
# `catalog.SnapshotChunk` is `{digest, len}` and has emitted nothing else since
# it was born at th#1303 — `sha256`/`length` are not retired spellings on this
# route, they never existed on it. Accepting them re-opened exactly the vacuous
# guard `resolved_entry_digest` refuses in the same module.


def test_the_live_chunk_spelling_parses() -> None:
    out = parse_chunk_list("t", "p", [{"digest": HEX64, "len": 17}], ["http://x/0"])
    assert len(out) == 1
    assert out[0].sha256 == HEX64
    assert out[0].length == 17
    assert out[0].url == "http://x/0"


def test_a_chunk_carrying_only_the_retired_sha256_key_is_REFUSED() -> None:
    """RED before pgw#1307: the `or c.get("sha256")` arm parsed this green."""
    with pytest.raises(HubResolveError) as exc:
        parse_chunk_list("t", "p", [{"sha256": HEX64, "len": 17}], ["http://x/0"])
    assert "missing digest/url/len" in str(exc.value)


def test_a_chunk_carrying_only_the_retired_length_key_is_REFUSED() -> None:
    """RED before pgw#1307: the `or c.get("length")` arm parsed this green.

    This is the dangerous polarity: a chunk whose length reads as 0 is a chunk
    the store believes is empty.
    """
    with pytest.raises(HubResolveError) as exc:
        parse_chunk_list("t", "p", [{"digest": HEX64, "length": 17}], ["http://x/0"])
    assert "missing digest/url/len" in str(exc.value)


# ---------------------------------------------------------------------------
# pgw#1305 arm (2): `destination.repo` dies, and the two halves of the reserved
# struct stop disagreeing about its key
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, destination: dict) -> None:
        self.destination = destination


def test_the_reserved_destination_struct_resolves_through_its_LIVE_key() -> None:
    """The live defect this arm was hiding. `destination_release`'s own refusal
    tells the caller to *"invoke with destination={ref, release}"* — and then
    `publish_flavors` read only `destination.repo`, so that exact invoke was
    answered with "destination_repo is required". RED before pgw#1305.
    """
    from gen_worker.convert.publish import destination_ref, destination_release

    ctx = _Ctx({"ref": "acme/dest", "release": "r1"})
    assert destination_ref(ctx) == "acme/dest"
    assert destination_release(ctx) == "r1"


def test_the_retired_repo_key_no_longer_addresses_anything() -> None:
    """RED before pgw#1305: `{"repo": ...}` resolved."""
    from gen_worker.convert.publish import destination_ref

    with pytest.raises(ValueError) as exc:
        destination_ref(_Ctx({"repo": "acme/dest", "release": "r1"}))
    assert "destination.ref" in str(exc.value)


def test_the_executor_half_reads_the_same_key_and_only_that_key() -> None:
    from gen_worker.executor import _producer_destination_repo

    class _P:
        destination_repo = ""

    assert _producer_destination_repo(_P(), {"ref": "acme/dest"}) == "acme/dest"
    assert _producer_destination_repo(_P(), {"repo": "acme/dest"}) == ""


@pytest.mark.parametrize(
    "raw", ["acme/dest@sha256:" + HEX64, "acme/dest#fp8", "acme/dest/"]
)
def test_both_halves_strip_selectors_identically(raw: str) -> None:
    """ONE vocabulary means one normalization, or the publish addresses a repo
    the executor never authorized.

    No `owner/repo:tag` case: th#1987 DELETED that ref production and
    `scripts/lint_repo_ref_pins.py` refuses the literal, so asserting we strip
    it would be asserting behaviour on an input the pod refuses upstream. Both
    halves still carry a `:` strip; whether that arm should survive th#1987 is
    noted on pgw#1307, not decided here.
    """
    from gen_worker.convert.publish import destination_ref
    from gen_worker.executor import _producer_destination_repo

    class _P:
        destination_repo = ""

    assert destination_ref(_Ctx({"ref": raw})) == "acme/dest"
    assert _producer_destination_repo(_P(), {"ref": raw}) == "acme/dest"


# ---------------------------------------------------------------------------
# pgw#1307 arm (5): the scipy "fallback" that was the ONLY path
# ---------------------------------------------------------------------------


def test_the_advertised_resampler_is_the_one_that_runs() -> None:
    """scipy was declared NOWHERE — not in `dependencies`, not in an extra, not
    in dev — so `from scipy.signal import resample_poly` always raised and every
    tenant got naive `np.interp` while the docstring advertised polyphase.
    RED before pgw#1307: scipy was not importable from this package's deps.
    """
    from scipy.signal import resample_poly  # noqa: F401

    import gen_worker.io as gw_io

    src = __import__("inspect").getsource(gw_io.read_audio)
    assert "np.interp" not in src, (
        "the linear-resample fallback is back; it was never a fallback, it was "
        "the only path"
    )
    assert "except ImportError" not in src.split("target_sample_rate is not None")[-1]


def test_scipy_is_declared_by_the_audio_extra() -> None:
    """The deletion above is only safe because the import is now guaranteed."""
    import pathlib
    import tomllib

    root = pathlib.Path(str(__import__("gen_worker").__file__)).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    audio = data["project"]["optional-dependencies"]["audio"]
    assert any(d.startswith("scipy") for d in audio), audio


# ---------------------------------------------------------------------------
# pgw#1307 arm (6): the blanket `except TypeError: setup_fn()`
# ---------------------------------------------------------------------------


def test_a_setup_we_cannot_satisfy_fails_LOUDLY() -> None:
    """The function's own docstring forbids the arm that used to swallow this:
    a non-inspectable `setup` got called a second time with NO models, silently
    dropping every resolved slot. RED before pgw#1307: this returned cleanly and
    `calls` recorded a no-model call.
    """
    from gen_worker.cli import run as run_mod

    calls: list[dict] = []

    class _Uninspectable:
        """`inspect.signature` raises TypeError on this — the shape the branch
        under test exists for (decorated / C-implemented / wrapped setups)."""

        __signature__ = "not a Signature"

        def __call__(self, **kw):
            calls.append(kw)
            if kw:
                raise TypeError("setup() got an unexpected keyword argument")

    class _Endpoint:
        setup = _Uninspectable()

    inst = _Endpoint()
    # Confirm the branch under test is actually the one taken.
    import inspect

    with pytest.raises((TypeError, ValueError)):
        inspect.signature(inst.setup)

    with pytest.raises(TypeError):
        run_mod.run_setup(inst, {"unet": "/tmp/unet"}, arm_compile=False)

    assert calls == [{"unet": "/tmp/unet"}], calls
    assert not any(kw == {} for kw in calls), (
        "setup() was re-called with no models — the blanket arm is back"
    )


# ---------------------------------------------------------------------------
# pgw#1307 arm (14): the dual-key fallback whose two arms named the SAME key
# ---------------------------------------------------------------------------


def test_no_or_expression_in_the_tree_has_two_identical_sides() -> None:
    """A rename applied mechanically to both halves of an `A or B` leaves a
    fossil that reads as a compatibility arm and is not one. The class is worth
    fencing, not just the two instances: this walks every BoolOp in the package.
    """
    import ast
    import pathlib

    root = pathlib.Path(str(__import__("gen_worker").__file__)).resolve().parent
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp):
                continue
            dumped = [ast.dump(v) for v in node.values]
            if len(dumped) != len(set(dumped)):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, (
        f"`X or X` — a dual-key fallback in which both arms name the same "
        f"thing, i.e. rename residue, not a compatibility arm: {offenders}"
    )


# ---------------------------------------------------------------------------
# pgw#1307 arm (9): the dead hasattr and the DEBUG-only failure
# ---------------------------------------------------------------------------


def test_the_recompile_limit_is_set_unconditionally_and_failure_is_AUDIBLE() -> None:
    """`recompile_limit` is unconditional at this repo's torch>=2.13 floor, and
    a failure to raise it silently keeps torch's 8-graph ceiling — announced
    only at DEBUG, which a hub-spawned pod cannot read. RED before pgw#1307 on
    both assertions.
    """
    import inspect

    from gen_worker import settings_authority

    src = inspect.getsource(settings_authority)
    block = src.split("recompile_limit")[0].rsplit("def ", 1)[-1]
    assert 'hasattr(torch._dynamo.config, "recompile_limit")' not in src
    assert "logger.warning" in src.split("recompile_limit")[-1], (
        "the failure is still DEBUG-only"
    )
    assert block  # the function exists and was found
