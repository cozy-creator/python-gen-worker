"""pgw#754: host-CPU ISA portability of AOTI cells.

The failure mode: a worker arming a DISCOVERED aot-inductor cell dies
SIGILL (exit 132) inside
``aoti_load_package`` — the cell's host wrapper ``.so`` was compiled
``-march=native`` on an AVX-512 mint host (the shipped kernel.cpp embeds the
compile command: ``-march=native ... -mavx512f -mavx512vnni``) and carries
EVEX-encoded AVX-512F scalar instructions the serving host cannot decode.

Red-verified here through the REAL paths:

* the boot clamp (``env_seal.establish`` -> ``host_isa.impose``) pins
  ``cpp.march``/``cpp.simdlen`` to the portable target, seal-visibly;
* TCG's worker compile path (the clamp is process-global inductor config, so
  it binds every graph class identically) emits NO
  instruction above the target (objdump assertion on the produced ``.so``)
  and the package still loads via ``aoti_load_package`` — portable by
  construction;
* TCG stamps and enforces the artifact's host requirement before loading; its
  library suite owns those closed-schema admission cases, while this worker
  suite proves the process-wide compiler clamp that supplies that stamp.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_compile_child, aot_mint, env_seal, host_isa
from torch_compiled_graphs import build_call_ingress
import torch_compiled_graphs.host_isa as tcg_host_isa

pytestmark = pytest.mark.skipif(
    host_isa.machine() != "x86_64", reason="x86-64 ISA-level semantics")


# ---------------------------------------------------------------------------
# Detection + clamp
# ---------------------------------------------------------------------------


def test_host_level_is_a_defined_level() -> None:
    level = host_isa.host_level()
    assert level in {name for name, _ in host_isa._LEVELS}
    # Any box this suite runs on is at least v2 (2009+ silicon).
    assert host_isa._RANK[level] >= host_isa._RANK["x86-64-v2"]


def test_mint_target_never_exceeds_baseline_or_host() -> None:
    march = host_isa.mint_march()
    assert march is not None
    assert host_isa._RANK[march] <= host_isa._RANK[host_isa.BASELINE]
    assert host_isa._RANK[march] <= host_isa._RANK[host_isa.host_level()]


def test_impose_is_seal_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config.cpp, "march", None)
    monkeypatch.setattr(inductor_config.cpp, "simdlen", None)
    facts = host_isa.impose()
    march = host_isa.mint_march()
    assert facts["cpp_march"] == march == inductor_config.cpp.march
    assert int(facts["cpp_simdlen"]) == inductor_config.cpp.simdlen
    # The clamp reaches the sealed inductor-config digest surface...
    portable = inductor_config.save_config_portable()
    assert portable["cpp.march"] == march


def test_establish_wires_the_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config.cpp, "march", None)
    monkeypatch.setattr(inductor_config.cpp, "simdlen", None)
    seal = env_seal.establish()
    # The clamp is a DECLARED inductor fact in the seal.
    assert seal["inductor"]["cpp.march"] == host_isa.mint_march()


# ---------------------------------------------------------------------------
# Red-verify 1: the clamped mint path emits nothing above the target
# ---------------------------------------------------------------------------


class _Glue(torch.nn.Module):
    """Parameter-free module: the packaged host code is pure glue, like the
    real wrapper (all fleet compute lives in device kernels)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 1.0


#: Instructions/operands that require anything above x86-64-v2: AVX+ vector
#: registers, opmasks, and the EVEX scalar forms observed in the live
#: artifact (vcvtusi2sd / vcvttsd2usi / vrndscalesd are AVX-512F-only even
#: on xmm operands — the exact SIGILL class).
_ABOVE_V2 = re.compile(
    r"%[yz]mm|%k[0-7]\b|vcvtusi2|vcvttsd2usi|vcvttss2usi|vrndscale|"
    r"vpternlog|vgather|vscatter")


@pytest.mark.skipif(
    shutil.which("objdump") is None, reason="objdump required")
def test_clamped_compile_package_is_portable_and_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch._inductor.config as inductor_config

    # Clamp BELOW this host's level (any host running this suite is >= v2):
    # every instruction the compile emits above the clamp would be exactly
    # the live SIGILL class, just one level down.
    v2 = tcg_host_isa._Requirement(
        "x86_64",
        "x86-64-v2",
        tcg_host_isa._required_flags("x86-64-v2"),
        "x86-64-v2",
        "128",
    )
    monkeypatch.setattr(tcg_host_isa, "_host_requirement", lambda: v2)
    monkeypatch.setattr(inductor_config.cpp, "march", "x86-64-v2")
    monkeypatch.setattr(inductor_config.cpp, "simdlen", 128)

    inputs = (torch.randn(4, 8),)
    program = torch.export.export(_Glue(), inputs)
    ingress = build_call_ingress(program, ("x",), inputs, {})
    export_spec = aot_mint.ExportSpec(family="host-isa", target="model")
    traced = aot_mint.TracedClass(
        name="model/default",
        block=aot_mint.keying_block(program, ingress, export_spec),
        nodes=1,
        program=program,
    )
    engine, runtime = aot_compile_child._tcg_runtime(tmp_path / "cas")
    compiled = engine.compile(
        aot_mint.tcg_graph_class_spec(traced, export_spec),
        runtime,
        tmp_path / "materialized",
    )
    package = compiled.compiled_graph.package

    with zipfile.ZipFile(package) as zf:
        so_names = [n for n in zf.namelist() if n.endswith(".so")]
        assert so_names, "package ships no host .so"
        for name in so_names:
            zf.extract(name, tmp_path / "x")

    for name in so_names:
        dis = subprocess.run(
            ["objdump", "-d", str(tmp_path / "x" / name)],
            check=True, capture_output=True, text=True).stdout
        hits = sorted({
            m.group(0) for m in _ABOVE_V2.finditer(dis)})
        assert not hits, (
            f"{name} contains above-baseline instructions {hits} — "
            "unloadable on a host without the mint host's CPU features")

    # The live crash site: loading must succeed for a portable package.
    loaded = torch._inductor.aoti_load_package(str(package))
    assert loaded is not None


# ---------------------------------------------------------------------------
# The clamp must be PROCESS-wide, not thread-local (0.82.0 release gate)
# ---------------------------------------------------------------------------
#
# torch's config precedence puts ``user_override`` above ``default``, and
# torch's own docstring says user overrides are THREAD-LOCAL — the layer is a
# ``ContextVar``. A plain ``inductor_config.cpp.march = x`` therefore clamps
# only the assigning thread. Boot imposes on the boot thread, so every host
# compile that runs anywhere else was built ``-march=native``: unclamped,
# unportable, the pgw#754 SIGILL class. Two such threads exist and both are
# on the production mint/serve path — ``hot_swap``'s process-global
# background shape-warm/heal worker, and pgw#811's K-way ``run_impl``
# splitter pool. Both must observe the same process-wide clamp.


def _march_seen_by_a_fresh_thread() -> object:
    import threading

    import torch._inductor.config as inductor_config

    box = {}

    def _read() -> None:
        box["march"] = inductor_config.cpp.march
        box["simdlen"] = inductor_config.cpp.simdlen

    t = threading.Thread(target=_read)
    t.start()
    t.join()
    return box["march"], box["simdlen"]


def test_clamp_is_visible_to_a_thread_that_never_imposed() -> None:
    """RED before the fix: a fresh thread read ``(None, None)`` and torch
    built ``-march=native`` for it."""
    march = host_isa.mint_march()
    assert march is not None
    host_isa.impose()
    assert _march_seen_by_a_fresh_thread() == (
        march, host_isa.mint_simdlen(march))


def test_a_background_compile_thread_builds_a_clamped_argv() -> None:
    """The end the defect was actually felt at: the argv torch would build
    off the boot thread passes pgw#811's own assertion.

    This drives torch's real ``_get_cpu_arch_cflags``/``get_cpp_torch_options``
    on a foreign thread rather than re-deriving what it would emit.
    """
    from torch._inductor import cpp_builder

    host_isa.impose()

    box: dict = {}

    def _build() -> None:
        try:
            compiler = cpp_builder.get_cpp_compiler()
            box["cflags"] = cpp_builder._get_cpu_arch_cflags(compiler)
        except Exception as exc:  # pragma: no cover - torch internals moved
            box["exc"] = exc

    t = threading.Thread(target=_build)
    t.start()
    t.join()

    assert "exc" not in box, box.get("exc")
    flags = box["cflags"]
    assert "march=native" not in flags, (
        f"a foreign compile thread still builds -march=native: {flags}")
    assert all(flag != "march=native" for flag in flags)


def test_impose_refuses_if_it_cannot_reach_a_process_wide_target() -> None:
    """A torch internals change must fail LOUDLY at boot, not silently go
    back to per-thread clamping."""

    class _NoEntries:
        _config: dict = {}

    with pytest.raises(host_isa.HostIsaError) as exc_info:
        host_isa._impose_default(_NoEntries(), "cpp.march", "x86-64-v3")
    assert "process-wide" in str(exc_info.value)
