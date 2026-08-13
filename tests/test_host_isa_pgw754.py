"""pgw#754: host-CPU ISA portability of AOTI compiled graphs.

Live incident (2026-07-28, machine hv220gc4f7vu): a worker arming a
DISCOVERED aot-inductor compiled graph died SIGILL (exit 132) inside
``aoti_load_package`` — the compiled graph's host wrapper ``.so`` was compiled
``-march=native`` on an AVX-512 mint host (the shipped kernel.cpp embeds the
compile command: ``-march=native ... -mavx512f -mavx512vnni``) and carries
EVEX-encoded AVX-512F scalar instructions the serving host cannot decode.

Red-verified here through the REAL paths:

* the boot clamp (``env_seal.establish`` -> ``host_isa.impose``) pins
  ``cpp.march``/``cpp.simdlen`` to the portable target, seal-visibly;
* the mint's compile path (``aot_mint.compile_entry_files`` +
  ``package_compiled_graph`` — the clamp is process-global inductor config, so it
  binds every entry of a multi-graph compiled graph identically) emits NO
  instruction above the target (objdump assertion on the produced ``.so``)
  and the package still loads via ``aoti_load_package`` — portable by
  construction;
* an artifact whose requirement this host genuinely lacks (this box has no
  AVX-512) is refused BY NAME (``adopt_failed:host_isa_unsupported``)
  before any dlopen — including legacy compiled graphs via the ``.pt2``'s own
  ``AOTI_CPU_ISA`` stamp — and never SIGILLs the worker.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_mint, aot_serve, env_seal, host_isa

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
    # The mint host must be able to execute its own artifact.
    assert host_isa.unsupported_reason(march, "x86_64") == ""


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
    # ...and the named seal facts.
    assert env_seal.effective_config()["cpp_march"] == march


def test_establish_wires_the_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config.cpp, "march", None)
    monkeypatch.setattr(inductor_config.cpp, "simdlen", None)
    monkeypatch.setattr(env_seal, "_BOOT_READBACK", None)
    seal = env_seal.establish()
    # pgw#1049: the clamp is a DECLARED inductor fact in the seal.
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
    monkeypatch.setattr(inductor_config.cpp, "march", "x86-64-v2")
    monkeypatch.setattr(inductor_config.cpp, "simdlen", 128)

    program = torch.export.export(_Glue(), (torch.randn(4, 8),))
    files = aot_mint.compile_entry_files(program, "model")
    package = aot_mint.package_compiled_graph({"model": files}, tmp_path / "model.pt2")

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
# Red-verify 2: refusal by name, never a SIGILL
# ---------------------------------------------------------------------------


def _artifact(tmp_path: Path, pt2_bytes: bytes) -> tuple[Path, dict]:
    meta = aot_serve.entry_metadata(
        family="sdxl", precision="bf16", compiled_graph_key="",
        name="unet/main", entry={
            "target": "unet",
            "inputs": [{
                "name": "x", "position": 0, "dtype": "float32",
                "shape": [1, 4], "optional": False,
            }],
            "symbols": {}, "constants": [],
        },
    )
    content = tmp_path / "content"
    content.mkdir(exist_ok=True)
    (content / aot_serve.PACKAGE_NAME).write_bytes(pt2_bytes)
    return content, meta


def _requires_avx512() -> None:
    if "avx512f" in host_isa.host_flags():
        pytest.skip("host has AVX-512; the genuine-lack refusal needs a "
                    "host without it")


def test_stamped_requirement_refused_by_name(tmp_path: Path) -> None:
    _requires_avx512()
    content, meta = _artifact(tmp_path, b"not-a-real-pt2")
    meta["host_isa"] = {
        "machine": "x86_64", "march": "", "simdlen": 0,
        "level": "x86-64-v4",
    }
    artifact = aot_serve.pack(content, tmp_path / "compiled_graph.tar.gz", meta)
    with pytest.raises(aot_serve.AdoptError) as exc_info:
        aot_serve.stage_artifact(artifact, "sdxl")
    assert exc_info.value.reason == "host_isa_unsupported"
    assert "avx512f" in str(exc_info.value)


def test_discovery_filter_drops_stamped_requirement(tmp_path: Path) -> None:
    """The metadata-only check reaches ``aot_serve.verify`` — discovery
    (aot_compiled_graphs._candidates) skips unexecutable compiled graphs BEFORE downloading."""
    _requires_avx512()
    _, meta = _artifact(tmp_path, b"")
    meta["host_isa"] = {
        "machine": "x86_64", "march": "", "simdlen": 0,
        "level": "x86-64-v4",
    }
    reason = aot_serve.verify(meta)
    assert "avx512f" in reason


def test_an_unstamped_compiled_graph_is_refused_from_metadata_alone(
    tmp_path: Path,
) -> None:
    """pgw#950: a compiled graph carrying no ``host_isa`` stamp used to be waved past
    the metadata gate and sniffed at stage time from the ``.pt2``'s own
    ``AOTI_CPU_ISA``. That sniff is gone. An unstamped compiled graph states no
    requirement, so it is refused where the metadata is read — BEFORE any
    download — and the miss policy re-mints one that does stamp.
    """
    pt2 = tmp_path / "unstamped.pt2"
    with zipfile.ZipFile(pt2, "w") as zf:
        zf.writestr(
            "model/data/aotinductor/model/abc.wrapper_metadata.json",
            json.dumps({
                "AOTI_CPU_ISA": "AVX512 AVX512_VNNI",
                "AOTI_MACHINE": "x86_64",
            }))
    content, meta = _artifact(tmp_path, pt2.read_bytes())
    meta.pop("host_isa")
    # Discovery rules on it without fetching a byte.
    reason = aot_serve.host_isa_reason(meta)
    assert aot_serve.NO_HOST_ISA_STAMP in reason
    assert aot_serve.NO_HOST_ISA_STAMP in aot_serve.verify(meta)
    # And staging names the same refusal, on the same class.
    artifact = aot_serve.pack(content, tmp_path / "unstamped.tar.gz", meta)
    with pytest.raises(aot_serve.AdoptError) as exc_info:
        aot_serve.stage_artifact(artifact, "sdxl")
    assert exc_info.value.reason == "host_isa_unsupported"
    assert aot_serve.NO_HOST_ISA_STAMP in str(exc_info.value)


def test_satisfiable_stamp_stages(tmp_path: Path) -> None:
    content, meta = _artifact(tmp_path, b"pt2-bytes-never-read")
    # artifact_metadata stamped this host's own mint target — executable
    # here by construction.
    staged = aot_serve.stage_artifact(
        aot_serve.pack(content, tmp_path / "ok.tar.gz", meta), "sdxl")
    try:
        assert staged.metadata["host_isa"]["machine"] == "x86_64"
    finally:
        staged.close()


def test_cross_machine_artifact_refused() -> None:
    assert "aarch64" in host_isa.unsupported_reason("", "aarch64")


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
# splitter pool. Since pgw#811's ``assert_command_is_clamped`` landed
# (0.81.0) they do not merely emit native code, they FAIL: on 0.81.0 every
# background warm compile raises HostIsaError, and per pgw#680's doctrine two
# failed heals mark the signature permanently ``volatile``.


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
    # And the argv assertion pgw#811 added agrees.
    host_isa.assert_command_is_clamped(
        ["g++", "x.cpp"] + [f"-{f}" for f in flags])


def test_impose_refuses_if_it_cannot_reach_a_process_wide_target() -> None:
    """A torch internals change must fail LOUDLY at boot, not silently go
    back to per-thread clamping."""

    class _NoEntries:
        _config: dict = {}

    with pytest.raises(host_isa.HostIsaError) as exc_info:
        host_isa._impose_default(_NoEntries(), "cpp.march", "x86-64-v3")
    assert "process-wide" in str(exc_info.value)
