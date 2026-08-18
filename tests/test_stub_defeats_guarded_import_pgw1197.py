"""A discovery stub that satisfies a guarded optional import makes torch
conclude triton is INSTALLED, and torch then dies touching it — e.g.
`PeftAdapterMixin` → `peft_utils` → `torch_utils` → `torch._dynamo` → build
dead, on a CPU conversion build.

THE CHAIN, exactly, on torch 2.13.0 with triton genuinely absent::

    torch/utils/_triton.py      has_triton_package():
                                    try: import triton      <- our stub SUCCEEDS
                                    return True             <- so this is True
                                    except ImportError: return False
    torch/_dynamo/utils.py:2874 if has_triton_package():
                           2875     import triton
                           2877     common_constant_types.add(triton.language.dtype)
                                                              ^^^^^^^^^^^^^^^^
                                    not inside any `try` — HeavyDepStubError, fatal

**The root cause is the IMPORT, not the exception type.** Raising an
`ImportError` subclass so upstream `try/except` keeps working is wrong twice
over, and both are pinned below: the fatal touch is not inside a `try` at all,
and the exception cannot become an `ImportError` without breaking
`from torch import nn`, which is the entire purpose of the stub.

Fooling `find_spec` availability probes is already forbidden;
`try: import X / except ImportError` is the *other* probe idiom.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from gen_worker.discovery import heavy_deps

# Blind the ordinary import machinery to a root, so the test environment is a
# faithful stand-in for one where the wheel is simply not installed. Needed
# because the stub finder is appended LAST on `sys.meta_path`, so a really
# installed package would shadow the stub and the scenario would not arise.
_BLIND = textwrap.dedent(
    """
    import importlib.machinery as _m, sys
    _orig = _m.PathFinder.find_spec.__func__
    def _blind(cls, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {roots!r}:
            return None
        return _orig(cls, fullname, path, target)
    _m.PathFinder.find_spec = classmethod(_blind)
    for _n in [n for n in sys.modules if n.split(".", 1)[0] in {roots!r}]:
        del sys.modules[_n]
    """
)


def _run(roots: set[str], body: str) -> str:
    """Run `body` in a fresh interpreter with `roots` made unimportable."""
    script = _BLIND.format(roots=roots) + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"child failed ({proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-3000:]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# 1. The defect, end to end, through torch's real code
# ---------------------------------------------------------------------------


def test_a_guarded_optional_import_still_answers_ABSENT() -> None:
    """RED on master: `has_triton_package()` answered True with triton absent.

    This is the whole defect in one assertion. A stub that satisfies an
    availability probe has not protected the caller — it has lied to it.
    """
    pytest.importorskip("torch")
    out = _run({"triton"}, """
        from gen_worker.discovery import heavy_deps
        with heavy_deps.stub_missing_heavy_deps():
            import torch.utils._triton as t
            print("HAS_TRITON", t.has_triton_package())
    """)
    assert "HAS_TRITON False" in out, (
        "the stub told torch that an absent triton was installed")


def test_a_module_reaching_torch_dynamo_still_imports() -> None:
    """The measured casualty. RED on master:
    `HeavyDepStubError: 'triton.language' was touched during discovery`."""
    pytest.importorskip("torch")
    out = _run({"triton"}, """
        from gen_worker.discovery import heavy_deps
        with heavy_deps.stub_missing_heavy_deps():
            import torch._dynamo.utils
            print("DYNAMO_OK")
    """)
    assert "DYNAMO_OK" in out


# ---------------------------------------------------------------------------
# 2. Why the exception type was NOT the fix — both arms, pinned
# ---------------------------------------------------------------------------


def test_the_fatal_touch_is_not_inside_a_try_so_ImportError_would_not_help() -> None:
    """The issue's primary fix direction would have changed the exception's
    NAME and left the build just as dead: `common_constant_types.add(
    triton.language.dtype)` has no guard around it at all."""
    torch = pytest.importorskip("torch")
    from pathlib import Path

    src = (Path(torch.__file__).parent / "_dynamo" / "utils.py").read_text()
    marker = "common_constant_types.add(triton.language.dtype)"
    assert marker in src, "torch moved the line this issue is about"
    # Walk back to the enclosing block: the guard is `if has_triton_package():`,
    # never a `try`.
    head = src[: src.index(marker)]
    tail_lines = [ln for ln in head.splitlines()[-6:] if ln.strip()]
    assert any("has_triton_package()" in ln for ln in tail_lines)
    assert not any(ln.strip().startswith("try") for ln in tail_lines), (
        "if torch ever guards this with try/except, re-open the question of "
        "whether the stub's exception type matters")


def test_the_stub_error_CANNOT_be_both_AttributeError_and_ImportError() -> None:
    """CPython refuses the dual base outright, so "just make it an ImportError
    subclass too" is not an available move."""
    with pytest.raises(TypeError, match="lay-out conflict"):
        type("Both", (AttributeError, ImportError), {})


def test_dropping_the_AttributeError_base_breaks_from_torch_import_nn() -> None:
    """…and the other arm: `from torch import nn` depends on the import
    machinery catching AttributeError from `torch.__path__`. An ImportError
    there kills the import the stub exists to make free."""
    out = _run({"torch"}, """
        from gen_worker.discovery import heavy_deps
        class _E(ImportError): pass
        def _ga(self, attr): raise _E(attr)
        heavy_deps._HeavyDepStub.__getattr__ = _ga
        with heavy_deps.stub_missing_heavy_deps():
            try:
                from torch import nn
                print("NN ok")
            except BaseException as e:
                print("NN broken", type(e).__name__, e)
    """)
    assert "NN broken" in out, (
        "if this passes, the AttributeError base is no longer load-bearing "
        "and the exception-type fix becomes available again")


def test_the_stub_still_does_its_job_for_a_genuinely_absent_torch() -> None:
    """The property the fix must not cost: a module-top `import torch` stays
    free, and defaulted probes degrade rather than explode."""
    out = _run({"torch"}, """
        from gen_worker.discovery import heavy_deps
        with heavy_deps.stub_missing_heavy_deps():
            from torch import nn          # noqa: F401
            import torch
            print("NN ok")
            print("VERSION", getattr(torch, "__version__", "<default>"))
            print("HASATTR", hasattr(torch, "zzz"))
    """)
    assert "NN ok" in out
    assert "VERSION <default>" in out
    assert "HASATTR False" in out


# ---------------------------------------------------------------------------
# 3. The class, not the instance: no probed root may be stubbed, ever
# ---------------------------------------------------------------------------


def test_no_probed_accelerator_is_in_the_default_allowlist() -> None:
    overlap = set(heavy_deps.DEFAULT_HEAVY_ROOTS) & set(heavy_deps.NEVER_STUB)
    assert not overlap, (
        f"{sorted(overlap)} are probed by `try: import X` in third-party code; "
        "stubbing them turns 'absent' into 'present but landmined'")
    assert "triton" in heavy_deps.NEVER_STUB


def test_the_extension_point_cannot_re_arm_the_landmine(capsys) -> None:
    """A project listing a probed root in `[tool.gen_worker]
    discovery_heavy_deps` gets it dropped, and told so — silently honouring it
    would reintroduce this defect with no trace in the build log."""
    with heavy_deps.stub_missing_heavy_deps(extra=["triton"]) as missing:
        assert "triton" not in missing
    assert "refusing to stub 'triton'" in capsys.readouterr().err


def test_every_never_stub_row_carries_its_reason() -> None:
    for root, reason in heavy_deps.NEVER_STUB.items():
        assert len(reason) > 30, f"{root} needs a reason a reader can act on"


# ---------------------------------------------------------------------------
# 4. The release-safety invariant: a 0.113.1 must not re-key the fleet TWICE
# ---------------------------------------------------------------------------


def test_gen_workers_own_source_is_not_a_toolchain_axis_input() -> None:
    """Why a patch release costs ONE re-mint (pgw#1050's) and not a second.

    `toolchain_digest()` enumerates the compile stack — the settings
    declaration, the loaded-native-lib manifest, dist-info RECORD hashes for
    torch/triton/nvidia-*, and triton's bundled CUDA tool binaries. **This
    wheel's own version is deliberately not among them**, so a pure
    import-machinery fix cannot move the compiled graph key.

    Measured for pgw#1197 on torch 2.13.0+cpu with triton installed (so every
    component is populated, not a comparison of empties): `toolchain_digest()`,
    `declaration_digest()` and `loaded_libs_digest()` are **byte-identical**
    before and after the fix — `declaration c757897a48482e33`,
    `loaded_libs 81e526e82b04c440`. This row is the standing form of that
    check: it fails if anyone adds a gen_worker-sourced input, which is what
    would silently turn the next patch release into a fleet-wide re-mint.
    """
    from gen_worker import compile_cache as cc

    keys = set(dict(cc.toolchain_digest()))
    assert {"settings_declaration", "loaded_libs"} <= keys
    offenders = [
        k for k in keys
        if "gen_worker" in k.lower() or "gen-worker" in k.lower()
    ]
    assert not offenders, (
        f"{offenders} put this wheel's own source into the toolchain axis — "
        "every patch release would then re-key every compiled graph on the fleet")
