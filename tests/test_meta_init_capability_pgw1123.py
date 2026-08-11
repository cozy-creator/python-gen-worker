"""pgw#1123: an image that cannot meta-instantiate must SAY SO, distinguishably.

The measured defect (pods ``ykwoaiqub6ktt3`` / ``3o09rf9ehnc4ym``, gen-worker
0.104.0, ``examples/micro-diffusion`` as the repo ships it):

    boot_adopt[structure_unsupported]: family=micro-diffusion function=generate
      key=- — 3 of 3 boot-trace child(ren) produced no class hashes:
      structure-only build of component '' (unknown class) is not possible:
      `accelerate` is not importable (No module named 'accelerate')

Two independent failures in one line, and this file holds one row for each:

1. ``structure_only`` imported ``accelerate``, which nothing declared and which
   the fleet's own probe family deliberately does not ship. No key, no resolve,
   self-mint forever. Fixed by OWNING the context manager
   (:mod:`gen_worker.models.meta_init`), not by adding a dependency that
   hard-requires torch to a base wheel that carries none.
2. It refused under ``structure_unsupported`` — the SAME token a family that is
   genuinely stranded reports, which is a correct and permanent state for the
   quantized artifact lanes. So "this image is broken for everything" and "this
   tree has no config-only structure" were one word, and the first looked
   exactly like a pod that chose to self-mint. Belt and braces after the fix:
   a stripped image or a shading family gets its own token,
   ``structure_capability_missing``, with the capability named.

Everything here drives the REAL seams — the real ``structure_only.build_component``
over micro-diffusion's real generated tree, the real refusal classifier the
trace child calls, the real ``boot_adopt`` vocabulary and the real activity
sink. Nothing stubs the function whose brokenness is the subject: the capability
is removed from the ENVIRONMENT, which is what a stripped image does.

No pod, no GPU, no mint, no compile — meta instantiation only.
"""

from __future__ import annotations

import builtins
import contextlib
import sys
from pathlib import Path
from typing import Any, Iterator, List

import pytest

torch = pytest.importorskip("torch")

from gen_worker import activity, boot_adopt  # noqa: E402
from gen_worker.models import meta_init, structure_only as so  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MICRO_SRC = REPO / "examples" / "micro-diffusion" / "src"


@contextlib.contextmanager
def _unimportable(name: str) -> Iterator[None]:
    """Make ``name`` unimportable, exactly as an image that lacks it is.

    Not a monkeypatch of the code under test: the code under test is what
    happens WHEN the package is absent, and this is the only honest way to
    produce that on a box where the package is installed.
    """
    real_import = builtins.__import__
    saved = {k: v for k, v in sys.modules.items()
             if k == name or k.startswith(name + ".")}

    def blocked(mod: str, *args: Any, **kwargs: Any) -> Any:
        if mod == name or mod.startswith(name + "."):
            raise ImportError(f"No module named {name!r}")
        return real_import(mod, *args, **kwargs)

    for key in saved:
        del sys.modules[key]
    builtins.__import__ = blocked
    try:
        yield
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def micro_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if str(MICRO_SRC) not in sys.path:
        sys.path.insert(0, str(MICRO_SRC))
    from micro_diffusion.weights import SEED, materialize

    return materialize(tmp_path_factory.mktemp("micro-tree"), seed=SEED)


# ---------------------------------------------------------------------------
# 1. The defect itself: the derivation must not need a package the probe
#    family does not ship.
# ---------------------------------------------------------------------------


def test_a_structure_only_build_needs_no_accelerate(micro_tree: Path) -> None:
    """RED on 0.105.0 with the verbatim pod message.

    The real builder, the real tree, `accelerate` absent from the process
    exactly as it is absent from the image the two pods ran.
    """
    with _unimportable("accelerate"):
        module, facts = so.build_component(micro_tree, "transformer",
                                           device="cpu")

    from gen_worker import meta_instantiation as mi

    assert facts.cls_name == "MicroDenoiser"
    assert facts.parameters > 0
    assert facts.virtual_param_bytes > 0
    assert all(mi.is_virtual(p) for _n, p in module.named_parameters())


def test_the_seam_keeps_the_invariant_the_key_depends_on() -> None:
    """Parameters on meta, buffers REAL — the property ``aot_package``'s
    literal constants and the folding fence both rest on. Upstream's version of
    this context manager reads ``ACCELERATE_INIT_INCLUDE_BUFFERS`` from the
    ambient environment and would move the buffers too; owning it is what makes
    that unreachable."""
    from torch import nn

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.register_buffer("table", torch.arange(4.0))

    with meta_init.init_empty_weights():
        tiny = _Tiny()

    assert tiny.lin.weight.device.type == "meta"
    assert tiny.get_buffer("table").device.type != "meta"
    # And the patch is undone: a module built afterwards is ordinary.
    assert _Tiny().lin.weight.device.type != "meta"


def test_the_capability_is_proven_not_assumed() -> None:
    """``require_meta_init`` builds a probe and checks the result, so a torch
    (or a shading family) that silently stopped moving parameters is caught
    here rather than by tracing real weights."""
    meta_init.require_meta_init()  # the box has torch: this must not raise

    with _unimportable("torch"):
        with pytest.raises(meta_init.MetaInitUnavailable) as caught:
            meta_init.require_meta_init()
    assert "torch" in str(caught.value)
    assert caught.value.capability == meta_init.CAPABILITY


# ---------------------------------------------------------------------------
# 2. The louder half: the two refusals are not the same word.
# ---------------------------------------------------------------------------


def test_a_broken_image_and_a_stranded_family_get_DIFFERENT_tokens(
    micro_tree: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The heart of pgw#1123. Both exceptions below are raised by the REAL
    ``structure_only`` seams; only their tokens are asserted."""
    # (a) a genuinely stranded tree — a class with no ConfigMixin surface.
    stranded = so.StructureOnlyUnsupported(
        component="transformer", cls_name="MicroDenoiser",
        lacks="it exposes no `from_config`")

    # (b) a process that cannot meta-instantiate, produced by the real builder.
    with _unimportable("torch"):
        with pytest.raises(so.StructureCapabilityMissing) as caught:
            so._init_empty_weights("transformer")
    broken = caught.value

    assert so.refusal_token(stranded) == "structure_unsupported"
    assert so.refusal_token(broken) == "structure_capability_missing"
    assert so.refusal_token(stranded) != so.refusal_token(broken)

    # It is still a StructureOnlyUnsupported, so every existing never-fatal
    # `except` keeps degrading rather than killing a boot.
    assert isinstance(broken, so.StructureOnlyUnsupported)
    # And it names the capability instead of blaming the family.
    assert meta_init.CAPABILITY in str(broken)
    assert "IMAGE defect" in str(broken)
    assert broken.capability == meta_init.CAPABILITY


def test_the_new_token_is_in_the_boot_adopt_vocabulary() -> None:
    """An event nobody can enumerate, count or alert on is the next silent
    one — which is the whole of pgw#1116's fence."""
    assert "structure_capability_missing" in boot_adopt.DERIVE_REASONS
    assert "structure_capability_missing" in boot_adopt.REASONS
    assert so.TOKEN_CAPABILITY_MISSING in boot_adopt.REASONS
    assert so.TOKEN_UNSUPPORTED in boot_adopt.REASONS


def test_the_trace_child_reports_the_capability_token_loudly() -> None:
    """The real classification site: ``boot_trace_child.run``'s ``except``
    clause routes through ``structure_only.refusal_token`` and logs the
    capability refusal at ERROR. Read out of the source so a future edit that
    collapses the two back into one token fails here."""
    src = (REPO / "src" / "gen_worker" / "boot_trace_child.py").read_text()
    assert "structure_only.refusal_token(exc)" in src
    assert "TOKEN_CAPABILITY_MISSING" in src
    assert 'logger.error("boot key cannot be derived in this image' in src


def test_the_refusal_reaches_the_hub_as_its_own_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End of the wire: the token a trace child reports is the ``phase`` of the
    typed ``boot_adopt`` event, through the real emitter and the real sink the
    hub's ``worker_activity_events`` row is built from."""
    seen: List[Any] = []
    monkeypatch.setattr(activity, "_sink", seen.append, raising=False)

    outcome = boot_adopt.refused(
        so.TOKEN_CAPABILITY_MISSING,
        f"3 of 3 boot-trace child(ren) produced no class hashes: "
        f"{meta_init.CAPABILITY} is unavailable",
        family="micro-diffusion", function="generate")

    rows = [u for u in seen if u.kind == activity.KIND_BOOT_ADOPT]
    assert len(rows) == 1
    assert rows[0].phase == "structure_capability_missing"
    assert rows[0].phase != "structure_unsupported"
    assert meta_init.CAPABILITY in rows[0].detail
    assert outcome.reason == "structure_capability_missing"
