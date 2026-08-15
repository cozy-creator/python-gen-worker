"""pgw#896 / pgw#897 / pgw#898 — one home per host fact.

The point of the collapse is not fewer lines. It is that **a second answer
becomes unrepresentable**: two call sites that could previously disagree about
the same physical fact now cannot, because there is only one place the fact is
read from.

Each test below is the disagreement, stated as a law:

* the CUDA predicate, the ``mem_get_info`` reading and the cgroup CPU quota
  each have exactly ONE code site (a fence over real source, comments and
  docstrings excluded, so prose about the rule cannot satisfy it);
* the FOUR CPU readers that used to answer a 2.5-core quota three different
  ways now derive from one fractional observation, and the integer is never
  above it;
* the compile pool and the host-move guard credit the SAME reclaimable page
  cache against the same cgroup tree;
* ``torch.cuda.is_available()`` being False no longer erases the difference
  between a host with no card and a card that will not answer;
* ``WorkerResources`` has exactly one builder (pgw#898).
"""

from __future__ import annotations

import ast
import io
import os
import tokenize
from pathlib import Path
from typing import List, Tuple

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import cpu_budget, cuda_probe, hostfacts, lifecycle, postmortem
from gen_worker.models.memory import probe_host_ram

_GIB = 1024 ** 3
_SRC = Path(hostfacts.__file__).parent


def _code_hits(needle: str) -> List[Tuple[str, int]]:
    """Every occurrence of ``needle`` in ``src/gen_worker`` that is real code.

    Comments and string literals are dropped, so a docstring naming the rule —
    including this module's own — can never satisfy the fence.
    """
    hits: List[Tuple[str, int]] = []
    for path in sorted(_SRC.rglob("*.py")):
        if "/pb/" in str(path):
            continue  # generated
        text = path.read_text()
        if needle not in text:
            continue
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError):  # pragma: no cover
            continue
        skip = {tokenize.COMMENT, tokenize.STRING,
                getattr(tokenize, "FSTRING_MIDDLE", -1)}
        for tok in tokens:
            if tok.type in skip:
                continue
            if needle in tok.line and needle in tok.string:
                hits.append((str(path.relative_to(_SRC)), tok.start[0]))
                break
        else:
            # `needle` spans several tokens: rebuild the code-only text.
            stripped = "".join(
                t.string for t in tokens if t.type not in skip
            )
            if needle in stripped:
                hits.append((str(path.relative_to(_SRC)), 0))
    return hits


# ---------------------------------------------------------------------------
# The fences: one code site per fact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "torch.cuda.is_available",
        "torch.cuda.mem_get_info",
    ],
)
def test_the_device_probes_have_exactly_one_code_site(needle: str) -> None:
    """74 weak CUDA predicates across 42 modules, and 10 ``mem_get_info``
    call sites across 8, all now read one home.

    The weak predicate is the load-bearing one: it answers False for a host
    with no card AND for a card that will not answer, so every site that read
    it alone reported a wedged H100 as a cardless box — and zeros are what the
    fleet places on. One site means changing that answer is one edit.
    """
    hits = _code_hits(needle)
    assert [f for f, _ in hits] == ["hostfacts.py"], (
        f"{needle} is answered in {len(hits)} places: {hits}. It has one home, "
        f"gen_worker/hostfacts.py — route the new site through it."
    )


def _literal_hits(needle: str, *, exact: bool = False) -> List[str]:
    """Modules with ``needle`` inside a real string LITERAL (a path is named
    by one), docstrings excluded — so prose about the rule cannot satisfy it."""
    out: List[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if "/pb/" in str(path):
            continue
        text = path.read_text()
        if needle not in text:
            continue
        tree = ast.parse(text)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and (node.value == needle if exact
                         else needle in node.value)):
                out.append(str(path.relative_to(_SRC)))
                break
    return out


def test_the_cgroup_cpu_quota_has_exactly_one_reader() -> None:
    """``cpu_budget`` read a FIXED ``/sys/fs/cgroup/cpu.max`` (which reports
    ``max`` in a nested cgroup that really is capped) and had a v1 fallback;
    ``postmortem`` walked the node chain and had none. Two readers of one file,
    each missing what the other had."""
    for needle in ("cpu.max", "cpu.cfs_quota_us"):
        assert _literal_hits(needle, exact=True) == ["hostfacts.py"], (
            f"{needle!r} is read in {_literal_hits(needle, exact=True)}; "
            f"the quota has "
            f"one reader, hostfacts.cpu_quota()"
        )


# ---------------------------------------------------------------------------
# CPU: three reductions, three roundings -> one observation
# ---------------------------------------------------------------------------


def _pin_quota(monkeypatch: pytest.MonkeyPatch, cores: float) -> None:
    """Make every quota reader in the tree see ``cores``.

    Written to patch each module that HAS its own reader, so it stays honest
    against the pre-collapse tree (where there were three) as well as this one
    (where there is one). When this needs more than the hostfacts entry, the
    collapse has regressed.
    """
    monkeypatch.setattr(hostfacts, "cpu_quota", lambda **_: cores)
    for module, name in (
        (postmortem, "cpu_quota_cores"),
        (cpu_budget, "cgroup_cpu_quota"),
        (pool, "cpu_quota_cores"),
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda *_a, **_k: cores)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(32)))


def test_a_fractional_cpu_quota_is_never_rounded_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cpu.max = 250000 100000`` is 2.5 cores.

    ``postmortem.effective_cpu_count`` and ``aot_compile_pool.cpu_facts`` both
    did ``max(1, int(quota + 0.5))`` -> **3**, while ``cpu_budget.cpu_allowance``
    kept **2.5**. So the fleet planned against 3 cores, torch sized its intra-op
    pool above the quota, and the kernel throttled the boot window. Rounding a
    CPU quota UP is always wrong.
    """
    _pin_quota(monkeypatch, 2.5)

    fractional = cpu_budget.cpu_allowance()
    assert fractional == pytest.approx(2.5)

    for name, value in (
        ("postmortem.effective_cpu_count", postmortem.effective_cpu_count()),
        ("aot_compile_pool.cpu_facts().vcpus", pool.cpu_facts().vcpus),
        ("hostfacts.cpu_allowance().whole_cores",
         hostfacts.cpu_allowance().whole_cores),
    ):
        assert value == 2, f"{name} returned {value}; floor(2.5) is 2, not 3"
        assert value <= fractional, (
            f"{name} returned {value}, ABOVE the {fractional}-core quota — "
            f"the thread pool it sizes will be CFS-throttled"
        )


def test_the_hub_and_torch_read_the_same_quota_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hub's ``vcpus`` and torch's thread allowance need not be equal —
    one is an integer, the other fractional — but they must be the same
    OBSERVATION. They were not: two readers with different kernel coverage."""
    _pin_quota(monkeypatch, 6.25)
    allowance = hostfacts.cpu_allowance()
    assert cpu_budget.cpu_allowance() == pytest.approx(allowance.cores)
    assert postmortem.effective_cpu_count() == allowance.whole_cores
    assert pool.cpu_facts().vcpus == allowance.whole_cores
    assert postmortem.cpu_quota_cores() == pytest.approx(allowance.quota_cores)


def test_the_quota_reader_covers_both_cgroup_generations(tmp_path: Path) -> None:
    """One reader with BOTH properties its two predecessors had one each of:
    the deepest node on the chain, and the v1 fallback."""
    v2 = tmp_path / "v2"
    (v2 / "kubepods" / "podxyz").mkdir(parents=True)
    (v2 / "cpu.max").write_text("max 100000\n")          # the root is uncapped
    (v2 / "kubepods" / "podxyz" / "cpu.max").write_text("450000 100000\n")
    proc = tmp_path / "self_v2"
    proc.write_text("0::/kubepods/podxyz\n")
    assert hostfacts.cpu_quota(root=v2, proc_self_cgroup=proc) == pytest.approx(4.5)

    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("250000\n")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    empty = tmp_path / "self_v1"
    empty.write_text("")
    assert hostfacts.cpu_quota(root=v1, proc_self_cgroup=empty) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# RAM: one definition of "reclaimable"
# ---------------------------------------------------------------------------


def _cgroup_tree(tmp_path: Path, **stat: int) -> Tuple[Path, Path, Path]:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(200 * _GIB))
    (root / "memory.current").write_text(str(150 * _GIB))
    (root / "memory.stat").write_text(
        "".join(f"{k} {v}\n" for k, v in stat.items()))
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 262144000 kB\nMemAvailable: 209715200 kB\n")
    return root, proc, meminfo


def test_the_pool_and_the_load_guard_reclaim_the_same_pages(
    tmp_path: Path,
) -> None:
    """A pod that just downloaded its weights holds the snapshot on the ACTIVE
    file LRU.

    ``models/memory`` counts both file LRUs and documents why (pgw#752: a
    251GB wan-2.2 pod reported 71.5GiB available while ~180GiB of it was clean
    snapshot cache). ``aot_compile_pool`` had its own parser counting
    ``inactive_file + slab_reclaimable`` only — so on the same box the compile
    pool believed 120 GiB less was available than the loader did, and neither
    number was labelled as a different question.
    """
    root, proc, meminfo = _cgroup_tree(
        tmp_path,
        active_file=120 * _GIB,
        inactive_file=8 * _GIB,
        slab_reclaimable=1 * _GIB,
        shmem=0,
        file_dirty=0,
        file_writeback=0,
    )
    ram = probe_host_ram(root=root, proc_self_cgroup=proc, meminfo=meminfo,
                         siblings=1)
    facts = pool.memory_facts(
        meminfo=meminfo, cgroup_root=root, proc_self_cgroup=proc)

    assert ram.reclaimable_file_gb == 128.0, ram
    assert facts.cgroup_reclaimable_bytes == int(ram.reclaimable_file_gb * _GIB), (
        f"the pool credits {facts.cgroup_reclaimable_bytes / _GIB:.1f} GiB and "
        f"the loader {ram.reclaimable_file_gb:.1f} GiB of the SAME page cache"
    )
    assert facts.available_bytes == int(ram.available_gb * _GIB), (
        f"{facts} vs {ram} — two answers to 'how much host RAM is there'"
    )


def test_proc_meminfo_has_one_parser(tmp_path: Path) -> None:
    """``postmortem`` and the compile pool each parsed the file themselves;
    ``probe_host_ram`` asked psutil for the same two numbers."""
    path = tmp_path / "meminfo"
    path.write_text("MemTotal:       65536000 kB\nMemAvailable:   32768000 kB\n"
                    "Buffers:            1024 kB\n")
    assert hostfacts.meminfo_kb(path) == {
        "MemTotal": 65536000, "MemAvailable": 32768000, "Buffers": 1024}
    assert _literal_hits("/proc/meminfo", exact=True) == ["hostfacts.py"], (
        f"/proc/meminfo is opened in "
        f"{_literal_hits('/proc/meminfo', exact=True)}"
    )


# ---------------------------------------------------------------------------
# CUDA: "no answer" is not "no card"
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_cuda_state() -> None:
    hostfacts.reset_cuda_state()


@pytest.mark.parametrize(
    "reason, expected",
    [
        (cuda_probe.NO_DEVICE_REASON, hostfacts.DEVICE_ABSENT),
        ("torch unavailable: No module named 'torch'", hostfacts.DEVICE_ABSENT),
        ("RuntimeError: CUDA initialization: driver too old (found version "
         "12080)", hostfacts.DEVICE_UNREADABLE),
        ("RuntimeError: CUDA-capable device(s) is/are busy or unavailable",
         hostfacts.DEVICE_UNREADABLE),
    ],
)
def test_a_card_that_will_not_answer_is_not_a_cardless_host(
    monkeypatch: pytest.MonkeyPatch, reason: str, expected: str,
) -> None:
    """``torch.cuda.is_available()`` is False for BOTH, and that is the whole
    defect: a wedged H100 measured silent zeros and the fleet placed on them.

    The discriminator is the probe's own class, never "nvidia-smi did not
    answer" — a broken diagnostic must not buy an absent card's exemption
    (pgw#1120). Only a host with a driver TO fail can report ``driver_too_old``
    or ``cuda_error``.
    """
    monkeypatch.setattr(
        cuda_probe, "probe_cuda",
        lambda *_a, **_k: cuda_probe.CudaProbeResult(ok=False, reason=reason))
    state = hostfacts.cuda_state()
    assert state.state == expected, state
    assert state.absent is (expected == hostfacts.DEVICE_ABSENT)
    assert state.unreadable is (expected == hostfacts.DEVICE_UNREADABLE)


def test_the_cpu_rung_warning_names_which_of_the_two_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator-facing consequence, and the one production consumer of the
    three-state.

    Every function on a pod with no usable card is served on the CPU rung
    behind a warning that read *"no CUDA device detected on this pod"* — said
    identically for a cardless box and for an H100 that would not answer. The
    operator then goes looking for a provisioning mistake on a machine that
    has the card in it.
    """
    from gen_worker.models import serve_fit
    from gen_worker.models.hub_policy import TensorhubWorkerCapabilities

    class _Res:
        gpu = True
        libraries: Tuple[str, ...] = ()

    caps = TensorhubWorkerCapabilities(
        cuda_version="", gpu_sm=0, torch_version="2.13.0", installed_libs=[])

    monkeypatch.setattr(
        cuda_probe, "probe_cuda",
        lambda *_a, **_k: cuda_probe.CudaProbeResult(
            ok=False, reason=cuda_probe.NO_DEVICE_REASON))
    hostfacts.reset_cuda_state()
    cardless = serve_fit.plan_serve(_Res(), caps, 0.0)
    assert "no CUDA device detected on this pod" in cardless.warning

    monkeypatch.setattr(
        cuda_probe, "probe_cuda",
        lambda *_a, **_k: cuda_probe.CudaProbeResult(
            ok=False,
            reason="RuntimeError: CUDA-capable device(s) is/are busy or "
                   "unavailable"))
    hostfacts.reset_cuda_state()
    wedged = serve_fit.plan_serve(_Res(), caps, 0.0)
    assert "will not answer" in wedged.warning and "cuda_error" in wedged.warning
    assert wedged.warning != cardless.warning, (
        "a wedged card and a cardless box still say the same thing"
    )


def test_the_accelerator_vocabulary_is_cuda_or_none() -> None:
    """`'cpu'` is an oxymoron on this axis and is not a state of it."""
    assert {hostfacts.DEVICE_PRESENT, hostfacts.DEVICE_ABSENT} == {"cuda", "none"}
    assert hostfacts.DEVICE_UNREADABLE not in {"cuda", "none", "cpu"}


def test_an_unreadable_card_is_not_reported_as_zero_bytes_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` is the reading that did not happen; ``0`` is a saturated card.
    Collapsing them is how an unreadable card advertised an empty one."""
    monkeypatch.setattr(hostfacts, "cuda_ready", lambda: False)
    assert hostfacts.free_vram_bytes(0) is None
    assert hostfacts.total_vram_bytes(0) is None
    assert hostfacts.headroom_bytes(0) is None


# ---------------------------------------------------------------------------
# pgw#898 — one builder for WorkerResources
# ---------------------------------------------------------------------------


def test_worker_resources_has_exactly_one_builder() -> None:
    """``lifecycle.build_resources()`` measured the host in the process that
    has already imported tenant code, and ``_apply_identity_and_resources``
    replaced or cleared the result before the hub ever saw it — 53 lines whose
    output the wire discarded by construction.

    A second builder is not merely waste: a field taught to only one of them is
    dead on the wire, and the pod that ships the protobuf default gets idle
    reaped as ``cold_idle_never_dispatched`` (pgw#846).
    """
    assert not hasattr(lifecycle.Lifecycle, "build_resources"), (
        "the discarded second builder is back"
    )
    builders = _code_hits("pb.WorkerResources(")
    assert [f for f, _ in builders] == ["procsplit/parent.py"], (
        f"WorkerResources is built in {builders}; the parent — the process "
        f"that has imported no tenant code — is the only one allowed to."
    )


def test_the_measured_host_is_one_immutable_struct() -> None:
    """``probe_hardware`` returned an untyped ``Dict[str, Any]`` whose keys
    two builders and one gate spelled by hand; ``gate_functions`` read a
    ``cuda_version`` key nothing ever wrote."""
    facts = hostfacts.HostFacts(gpu_count=2, vram_total_bytes=17, gpu_sm="90")
    with pytest.raises(AttributeError):
        facts.gpu_count = 3  # type: ignore[misc]
    assert facts.as_dict()["gpu_count"] == 2
    assert "cuda_version" in hostfacts.HostFacts.__struct_fields__
    assert isinstance(lifecycle.probe_hardware(), hostfacts.HostFacts)
