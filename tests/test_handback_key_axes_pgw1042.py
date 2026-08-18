"""pgw#1042 — the delegated handback seam.

A full 36/36 sdxl mint publishing nothing, behind two defects this file pins:

1. The child's returned compiled graph carried a key the parent could not relate to its
   own. The obligation identity is not a key at all (`arm1-…`,
   `fleet_compiled_graphs.ArmIdentity`) — but every pre-trace FACT the two sides share
   must be byte-identical across the process boundary, and one was not:
   `torch._inductor.aot_compile` mutates the global `aot_inductor.metadata`
   config entry as a side effect, so a child that has compiled seals a
   different `env_seal` than its own boot (and than every other process on
   the pod). The seal digest now excludes that entry, and
   `adopt_delegated_mint` refuses BY FACT NAME when any shared fact diverges.

2. The artifact failed `update_constant_buffer_func_(... ) API call failed at
   model_container_runner.cpp:289` on the parent runtime: the per-entry copying
   bind allocates the target's FULL constant set once per entry, so a 36-entry
   sdxl compiled graph demanded ~N x 2.6 GB
   at arm and the failing cudaMalloc surfaced as that anonymous C++ error.
   Entries now bind BY REFERENCE against one marker-owned pool per target,
   and any residual AOTI failure is a typed `injection_failed` refusal
   carrying live device-memory context.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest

from gen_worker import aot_serve, env_seal, fleet_compiled_graphs, graph_facts
from gen_worker.compile_cache import AdoptError
from gen_worker._vendor.torchcg import (
    CallIngress,
    CallInput,
    CompiledGraphRunner,
    ConstantBindingError,
)
from gen_worker._vendor.torchcg import runner as tcg_runner_mod
from gen_worker._vendor.torchcg.storage import StoredCompiledGraph


def _seal_dict() -> Dict[str, Any]:
    return {
        "seal_v": env_seal.SEAL_VERSION,
        "posture": {"grad": "off"},
        "config": {"float32_matmul_precision": "high"},
        "inductor": "aaaa000011112222",
        "loaded_libs": "bbbb333344445555",
    }


_DECLARED_ENVELOPE = {
    "shapes": [[64, 64]], "text_lens": [7], "guidance": [1.0]}


def _arm_key(seal: Dict[str, Any], toolchain: Dict[str, Any]) -> fleet_compiled_graphs.ArmIdentity:
    return fleet_compiled_graphs.ArmIdentity(facts=tuple(sorted({
        "family": "micro-diffusion",
        aot_serve.COMPILED_GRAPH_FORMAT_KEY: str(aot_serve.COMPILED_GRAPH_FORMAT),
        "lane": "w8a8-lora64",
        "sm": "sm_89",
        "envelope": graph_facts.envelope_digest(_DECLARED_ENVELOPE),
        "env_seal": env_seal.seal_digest(seal),
        "toolchain": graph_facts.facts_digest(toolchain),
    }.items())))


def _envelope(seal: Dict[str, Any], toolchain: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "compiled_graph_key": "cg-key-v1-" + "e" * 56,
        "kind": "aot-inductor",
        aot_serve.COMPILED_GRAPH_FORMAT_KEY: str(aot_serve.COMPILED_GRAPH_FORMAT),
        "family": "micro-diffusion",
        "weight_lane": "w8a8",
        "lora_bucket": 64,
        "sm": "sm_89",
        graph_facts.EXPORT_ENVELOPE_KEY: dict(_DECLARED_ENVELOPE),
        env_seal.SEAL_KEY: dict(seal),
        "toolchain": dict(toolchain),
    }


TOOLCHAIN = {"libtorch.so": "cafe0123"}


def test_shared_axes_agree_is_empty() -> None:
    seal = _seal_dict()
    assert fleet_compiled_graphs.arm_axis_divergence(
        _arm_key(seal, TOOLCHAIN), _envelope(seal, TOOLCHAIN)) == ""


def test_the_env_seal_is_no_longer_comparable_at_this_seam() -> None:
    """INVERTED by pgw#1340 (th#2098), and stated rather than deleted.

    This used to assert that a child whose recorded seal differs from the
    parent's is refused ``env_seal: …`` — the measured pgw#1042 class, where
    `torch._inductor.aot_compile` mutated global config as a side effect.
    THE SEAL IS NO LONGER ON THE COMPILED GRAPH AT ALL: since pgw#1270 TCG mints every
    artifact and `validate_metadata` refuses metadata outside its closed
    vocabulary, which has no `env_seal` field. So this comparison did not
    catch a diverging seal — it read `{}` off every real compiled graph, digested that,
    and refused every handback ever made.

    The pgw#1042 root fix survives where it belongs: the seal digest still
    excludes `aot_inductor.metadata` (pinned below), so the side effect this
    axis was watching for cannot move a seal in the first place.
    """
    parent_seal = _seal_dict()
    child_seal = dict(parent_seal, inductor="ffff999988887777")
    assert "env_seal" not in fleet_compiled_graphs.ARM_ENVIRONMENT_FACTS
    assert "env_seal" in fleet_compiled_graphs.ARM_OBLIGATION_FACTS
    assert fleet_compiled_graphs.arm_axis_divergence(
        _arm_key(parent_seal, TOOLCHAIN), _envelope(child_seal, TOOLCHAIN)) == ""


@pytest.mark.parametrize("field,value,axis", [
    ("sm", "sm_80", "sm"),
    ("toolchain", {"libtorch.so": "0000ffff"}, "toolchain"),
    (aot_serve.COMPILED_GRAPH_FORMAT_KEY, "99",
     aot_serve.COMPILED_GRAPH_FORMAT_KEY),
])
def test_every_shared_axis_is_guarded(field: str, value: Any, axis: str) -> None:
    seal = _seal_dict()
    meta = _envelope(seal, TOOLCHAIN)
    meta[field] = value
    got = fleet_compiled_graphs.arm_axis_divergence(_arm_key(seal, TOOLCHAIN), meta)
    assert got.startswith(f"{axis}: "), got


def test_the_envelope_is_deliberately_NOT_compared_at_this_seam() -> None:
    """INVERTED by pgw#1176, and stated rather than merely dropped (pgw#939:
    absence is a verdict, never a skipped check).

    This used to be a sixth `test_every_shared_axis_is_guarded` case asserting
    a diverging `EXPORT_ENVELOPE_KEY` is refused `envelope: ...`. `envelope`
    left `ARM_ENVIRONMENT_FACTS` with the key: a per-entry artifact records no
    declared envelope — that is a manifest fact about the whole declaration,
    not about one graph class — so comparing it here would test a value the
    child can no longer state, and refuse every handback by construction.

    Left as a silent parametrize deletion this would read as an oversight; as
    a row it is the ruling, and it goes red if anyone puts the axis back
    without revisiting the reason."""
    seal = _seal_dict()
    assert "envelope" not in fleet_compiled_graphs.ARM_ENVIRONMENT_FACTS

    meta = _envelope(seal, TOOLCHAIN)
    meta[graph_facts.EXPORT_ENVELOPE_KEY] = {
        "shapes": [[128, 128]], "text_lens": [7], "guidance": [1.0]}
    assert fleet_compiled_graphs.arm_axis_divergence(
        _arm_key(seal, TOOLCHAIN), meta) == ""


@pytest.mark.parametrize("axis", ["family", "lane"])
def test_the_obligation_facts_are_deliberately_NOT_compared_either(
    axis: str,
) -> None:
    """pgw#1340 finished the sweep pgw#1176 started.

    ``family`` and ``lane`` left for the identical reason ``envelope`` did —
    a compiled graph has no field for them — and leaving them behind cost th#2098:
    ~$1.00 of L4 per burst, every burst, for two wheels. They still split the
    obligation on the parent's side, where the arm token carries them.
    """
    seal = _seal_dict()
    assert axis not in fleet_compiled_graphs.ARM_ENVIRONMENT_FACTS
    assert axis in fleet_compiled_graphs.ARM_OBLIGATION_FACTS
    meta = _envelope(seal, TOOLCHAIN)
    meta["family"] = "other-family"
    meta["lora_bucket"] = 0
    assert fleet_compiled_graphs.arm_axis_divergence(
        _arm_key(seal, TOOLCHAIN), meta) == ""


def test_adopt_refuses_typed_before_any_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diverging child compiled graph must fail `key_axis_divergence` at the seam —
    never reach the arm, whose C++ error was the pod's whole diagnostic.

    pgw#1340 re-aimed this at ``sm``: the seal it used to diverge is no longer
    a comparable axis (a compiled graph cannot state it), so diverging it now proves
    nothing. ``sm`` is a genuine cross-runtime divergence and the refusal is
    unchanged — which is the point of re-aiming rather than deleting.
    """
    seal = _seal_dict()
    meta = dict(_envelope(seal, TOOLCHAIN), sm="sm_80")

    from gen_worker import artifact_meta
    from gen_worker.models import provision

    # The adopt reads through `read_metadata` now — an envelope it
    # CANNOT read is its own refusal (`compiled_graph_envelope_unreadable`) rather than
    # a `None` that flows on into a gate it silently disables. This test is
    # about the DIVERGENCE verdict, so it supplies a readable envelope.
    monkeypatch.setattr(
        artifact_meta, "read_metadata", lambda _p: dict(meta))

    def _no_arm(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("arm_aot must not run on a diverged compiled graph")

    monkeypatch.setattr(provision, "arm_aot", _no_arm)

    mint_root = tmp_path / "mint-root"
    mint_root.mkdir()
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"not-a-real-compiled graph")
    arm = _arm_key(seal, TOOLCHAIN)
    pending = fleet_compiled_graphs.PendingSelfMint(
        family="micro-diffusion", arm_token=arm.token,
        ref=f"x#{arm.token}", cfg=object(), target=tmp_path / "adopted.tar.gz",
        mint_root=mint_root, publisher=None, cache_dir=tmp_path / "cache",
        arm_key=arm)

    assert fleet_compiled_graphs.adopt_delegated_mint(object(), pending, [artifact]) is None
    reason, detail = fleet_compiled_graphs.adopt_refusal(pending)
    assert reason == "key_axis_divergence"
    assert "sm: " in detail
    assert meta["compiled_graph_key"] in detail


def test_inductor_digest_ignores_aot_compile_metadata() -> None:
    """torch's aot_compile writes machine facts into global config as a side
    effect; the seal must not move with it (the measured pgw#1042 axis)."""
    torch = pytest.importorskip("torch")
    import torch._inductor.config as inductor_config

    before = env_seal.inductor_config_digest()
    prior = dict(getattr(inductor_config.aot_inductor, "metadata", {}) or {})
    try:
        inductor_config.aot_inductor.metadata = {
            "AOTI_DEVICE_KEY": "cuda", "AOTI_CPU_ISA": "AVX2",
            "AOTI_COMPUTE_CAPABILITY": "89"}
        assert env_seal.inductor_config_digest() == before
    finally:
        inductor_config.aot_inductor.metadata = prior
    assert torch is not None


class _FakeTensor:
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return 2

    def detach(self) -> "_FakeTensor":
        return self

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self._n)


class _RefusingPackage:
    """A package whose C++ update fails the way the pod's did."""

    def get_constant_fqns(self) -> list[str]:
        return ["lin.weight"]

    def load_constants(self, *_a: Any, **_k: Any) -> None:
        raise RuntimeError(
            "update_constant_buffer_func_( container_handle_, "
            "(AOTInductorConstantMapHandle)&const_map, use_inactive, "
            "check_full_update) API call failed at "
            "model_container_runner.cpp, line 289")


class _RecordingPackage:
    def __init__(self, fqns: tuple[str, ...]) -> None:
        self.fqns = fqns
        self.loaded: Dict[str, Any] = {}

    def get_constant_fqns(self) -> list[str]:
        return list(self.fqns)

    def load_constants(
        self, values: Dict[str, Any], **_kwargs: Any,
    ) -> None:
        self.loaded = dict(values)


def _tcg_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: Any,
    constants: list[Dict[str, Any]],
    *,
    literals: Dict[str, Any] | None = None,
) -> CompiledGraphRunner:
    """Create TCG's real bind authority over one admitted metadata block."""
    monkeypatch.setattr(tcg_runner_mod, "_load_package", lambda *_a: package)
    if literals is not None:
        monkeypatch.setattr(
            tcg_runner_mod, "_load_literals", lambda *_a: dict(literals))
    graph = cast(StoredCompiledGraph, SimpleNamespace(
        key="cg-key-v1-" + "9" * 56,
        metadata={
            "graph_class": {
                "name": "transformer/cfg=true",
                "constants": constants,
            },
        },
        package=tmp_path / "model.pt2",
        literals=(
            tmp_path / "constants.safetensors"
            if literals is not None else None
        ),
    ))
    return CompiledGraphRunner._from_verified_graph(graph)


def _constant(fqn: str, source: str = "state_dict") -> Dict[str, Any]:
    return {"fqn": fqn, "source": source, "dtype": "float32", "shape": [4]}


def test_cpp_bind_failure_is_typed_injection_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pod's anonymous `API call failed` becomes a NAMED refusal with the
    entry and live device-memory context — pgw#999's rule, one frame deeper."""
    runner = _tcg_runner(
        tmp_path,
        monkeypatch,
        _RefusingPackage(),
        [_constant("lin.weight")],
    )
    # TCG refuses an empty parameter tuple; the arm under test never reaches a
    # forward, so one declared input is all this metadata needs to be valid.
    ingress = CallIngress(
        parameters=("x",), flat_arity=1,
        inputs=(CallInput("x", 0, "x", 0, (), "x", "float32", (4,)),),
    )
    metadata = {
        "graph_class": {
            "name": "transformer/cfg=true",
            "target": "transformer",
            "class_hash": "a" * 16,
            "constants": [_constant("lin.weight")],
            "graph": {
                "pytree": {"ingress": ingress.as_dict()},
                "constant_fqns": ["lin.weight"],
            },
        },
    }

    class _Target:
        device = "cpu"

        def forward(self) -> None:
            return None

        def state_dict(self) -> Dict[str, Any]:
            return {"lin.weight": _FakeTensor()}

        def named_buffers(self) -> tuple[()]:
            return ()

    class _Pipeline:
        transformer = _Target()

    class _Cfg:
        family = "micro-diffusion"

    class _Engine:
        def resolve(self, _key: str, _destination: Path) -> Any:
            return SimpleNamespace(metadata=metadata)

        def runner(self, _key: str, _destination: Path) -> CompiledGraphRunner:
            return runner

    monkeypatch.setattr(aot_serve, "open_worker_engine", lambda _root=None: _Engine())
    with pytest.raises(AdoptError) as exc:
        aot_serve.arm_compiled_graph(
            _Pipeline(), _Cfg(), "cg-key-v1-" + "9" * 56, tmp_path,
        )
    assert exc.value.reason == "injection_failed"
    assert "transformer/cfg=true" in str(exc.value)
    assert not runner.bound


def test_tcg_entries_share_one_resident_REFERENCE_per_fqn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N entries share ONE pool slot per FQN — the N x constants VRAM demand
    that OOM'd the 36-entry sdxl arm must not be reconstructible.

    INVERTED by pgw#1177. This row asserted `pool["w"] is not resident`
    ("owned, not aliased", pgw#812 D3). That clone was the ONLY copy in the
    system — one full duplicate of the target's weights held for the life of
    the arm, ~5.1 GiB on sdxl's single `unet` target — in direct contradiction
    of §4.33 step 4, *"the compiled entries bind constants BY REFERENCE
    against the resident weights; there is no second copy of the model"*. The
    dedup this row actually guards is unchanged; only the ownership claim
    inverted.
    """
    torch = pytest.importorskip("torch")
    resident = torch.arange(4, dtype=torch.float32)
    literal = torch.arange(4, dtype=torch.float32) + 10
    packages = [_RecordingPackage(("w", "tbl")) for _ in range(3)]
    for package in packages:
        runner = _tcg_runner(
            tmp_path,
            monkeypatch,
            package,
            [_constant("w"), _constant("tbl", "literal")],
            literals={"tbl": literal},
        )
        runner.bind({"w": resident}, device="cpu")

    assert all(package.loaded["w"] is resident for package in packages)
    assert all(package.loaded["tbl"] is literal for package in packages)


def test_a_NON_contiguous_resident_is_the_one_thing_still_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The surviving guard from the inversion above. Dropping the blanket
    clone kept exactly one thing it also did: an AOTI container binds a RAW
    POINTER, so a non-contiguous resident cannot be bound by reference and is
    copied individually — priced per tensor rather than by cloning the whole
    pool for its sake. Without this row the inversion would read as "nothing
    is ever copied", which would be a segfault precondition."""
    torch = pytest.importorskip("torch")
    resident = torch.arange(8, dtype=torch.float32)[::2]
    assert not resident.is_contiguous()

    package = _RecordingPackage(("w",))
    runner = _tcg_runner(
        tmp_path, monkeypatch, package, [_constant("w")],
    )
    runner.bind({"w": resident}, device="cpu")
    bound = package.loaded["w"]
    assert bound is not resident
    assert bound.is_contiguous()
    assert torch.equal(bound, resident)
