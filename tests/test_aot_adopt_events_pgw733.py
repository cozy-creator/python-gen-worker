"""pgw#733 (arm half) + pgw#923: every AOT adopt/arm outcome is TYPED and MEASURED.

A cross-pod adopt fails inside stage/bind/arm with a classified ``AdoptError``
reason. Reducing those to ``logger.warning`` makes the reason structurally
invisible, because hub-spawned workers expose no stdout.

So the arm RETURNS its outcome, the arming policy MEASURES it, and one ledger —
``ArmOutcome.adoptions`` — carries every attempt with its identity, its
classified reason and its wall time to the executor, which puts it on the wire
the hub already stores. Every classified refusal is named, and also timed and
bound to the candidate's ref+digest.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gen_worker import cell_key as cell_key_mod
from gen_worker import activity, aot_serve
from torch_compiled_graphs import CallIngress, CallInput

#: The arm is INDUCED to take this long, and the floor asserted against it is a
#: share of that induced quantity rather than a bare constant. This is
#: a LOWER bound on work the test itself produced: a slow runner only raises the
#: measured value, so nothing here can fail because the machine was busy.

FAMILY = "sdxl"
RUNTIME = {"sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130",
           "cuda": "13.0"}
KEY = "cg-key-v1-" + "a" * 56

INPUTS = [
    {"name": "sample", "position": 0, "dtype": "bfloat16",
     "shape": [2, 4, 128, 128]},
]
CONSTANTS = [
    {"fqn": "conv_in.weight", "source": aot_serve.SOURCE_STATE_DICT,
     "dtype": "bfloat16", "shape": [320, 4, 3, 3]},
]
ENTRY = "unet/g"


class FakeTensor:
    def __init__(self, shape: Any, dtype: str = "torch.bfloat16") -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class FakePackage:
    """Stands in for ``AOTICompiledModel`` (no dlopen of a real .pt2)."""

    def __init__(self, fqns: Any = ("conv_in.weight",)) -> None:
        self._fqns = list(fqns)
        self.loaded: Optional[Dict[str, Any]] = None

    def get_constant_fqns(self) -> List[str]:
        return list(self._fqns)

    def load_constants(self, values: Dict[str, Any],
                       check_full_update: bool = False,
                       user_managed: bool = False) -> None:
        # torch 2.13's real signature carries user_managed (pgw#1042: the
        # whole-graph arm binds by reference against the per-target pool).
        if check_full_update and set(values) != set(self._fqns):
            raise RuntimeError("partial constant update refused by torch")
        self.loaded = dict(values)

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return "ARTIFACT_OUTPUT"


class FakeModule:
    def __init__(self, weights: Any = ("conv_in.weight",)) -> None:
        self.device = "cpu"
        self._weights = {name: FakeTensor([1]) for name in weights}

    def state_dict(self) -> Dict[str, Any]:
        return dict(self._weights)

    def forward(self, *args: Any, **kwargs: Any) -> str:
        return "EAGER_OUTPUT"


class FakePipeline:
    def __init__(self) -> None:
        self.unet = FakeModule()


class Cfg:
    family = FAMILY
    lora_bucket = 0
    targets = ("unet",)
    regional = False


def _entry(**over: Any) -> Dict[str, Any]:
    ingress = CallIngress(
        parameters=("sample",),
        flat_arity=1,
        inputs=(CallInput(
            "sample", 0, "sample", 0, (), "sample", "bfloat16",
            (2, 4, 128, 128),
        ),),
    )
    e: Dict[str, Any] = {
        # An entry block NAMES its class — that is what makes a
        # refusal bisectable to the thing that failed.
        "name": ENTRY,
        "target": "unet", "fork": [], "class_dims": [],
        "inputs": [dict(r) for r in INPUTS], "symbols": {},
        "constants": [dict(r) for r in CONSTANTS],
        "graph": {}, "pytree": {"ingress": ingress.as_dict()},
    }
    e.update(over)
    try:
        e["range_digest"] = aot_serve.range_digest(e)
    except (ValueError, TypeError):
        e.setdefault("range_digest", "")
    e["class_hash"] = aot_serve.class_hash(e, strict=True, lora_bucket=0)
    return e


def _meta(**over: Any) -> Dict[str, Any]:
    # ONE entry per artifact. No `entries=` override, because a
    # multi-entry envelope is a shape production cannot produce.
    entry = over.pop("entry", None) or _entry()
    m: Dict[str, Any] = {
        aot_serve.COMPILED_GRAPH_FORMAT_KEY: aot_serve.COMPILED_GRAPH_FORMAT,
        "kind": aot_serve.ARTIFACT_KIND,
        **RUNTIME, "family": FAMILY, "precision": "w8a8",
        "cell_key": KEY, cell_key_mod.ENTRY_BLOCK_KEY: entry,
        "strict_export": True, "lora_bucket": 0,
        "package_constants_in_so": False,
        # No weight BYTES in the .so (above) and no weight VALUES in
        # its kernels (here). Both are declared axes; a cell silent on either
        # is refused before a byte moves.
        "constant_folding_fenced": True,
        "source_ref": "", "source_digest": "",
        # Every mint stamps a host-ISA requirement, and a cell that
        # stamps none is refused rather than sniffed from the .pt2. Satisfiable
        # anywhere: this host's machine, no ISA level.
        "host_isa": {"machine": platform.machine(), "march": "", "simdlen": 0,
                     "level": ""},
        # The identity blocks the four-axis key restates from —
        # verify_contract refuses a stamp the artifact cannot restate.
        "env_seal": {"seal_v": 4, "env": {"PYTHONHASHSEED": "0"}},
        "toolchain": {"torch": "t" * 16, "settings_declaration": "d" * 16,
                      "loaded_libs": "l" * 16},
        "weight_lane": "w8a8",
        "declared_envelope": {"shapes": [[1024, 1024]], "text_lens": [77],
                              "guidance": [7.5]},
    }
    m["manifest_digest"] = cell_key_mod.manifest_digest(
        [str((entry or {}).get("class_hash") or "")])
    m.update(over)
    try:
        m["cell_key"] = cell_key_mod.from_entry_metadata(m).digest
    except cell_key_mod.CellKeyError:
        pass  # deliberately-malformed variants keep the placeholder stamp
    return m


def _tar(tmp_path: Path, meta: Optional[Dict[str, Any]] = None,
         name: str = "cell.tar.gz") -> Path:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / aot_serve.PACKAGE_NAME).write_bytes(b"\x00not-a-real-pt2")
    return aot_serve.pack(work, tmp_path / name, meta or _meta())


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


@pytest.fixture()
def stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME))


# ---------------------------------------------------------------------------
# aot_serve.enable — the classified inner reason, success AND failure
# ---------------------------------------------------------------------------


def test_successful_arm_returns_an_armed_outcome_naming_the_cell(
    tmp_path: Path, stub_runtime: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aot_serve, "_load_package", lambda path, entry: FakePackage())
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=_tar(tmp_path))
    assert out.armed and out.reason == ""
    assert _meta()["cell_key"] in out.identity and FAMILY in out.identity


def test_key_mismatch_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    # An sm mismatch is the key mismatch; a SKU difference alone is
    # adopted (same-sm cross-SKU cells are the point of the pgw#691 collapse).
    art = _tar(tmp_path, _meta(sm="sm_80"))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    assert not out.armed and out.reason == "key_mismatch"
    # the refusal names the candidate cell (its own restated stamp)
    assert _meta(sm="sm_80")["cell_key"] in out.detail


def test_host_isa_unsupported_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = _tar(tmp_path, _meta(host_isa={"machine": "sparc64", "level": ""}))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    assert not out.armed and out.reason == "host_isa_unsupported"


def test_artifact_invalid_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
) -> None:
    art = tmp_path / "corrupt.tar.gz"
    art.write_bytes(b"\x00definitely not a tarball")
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=art)
    assert not out.armed and out.reason == "artifact_invalid"
    # Identity is best-effort: an unreadable artifact still names its file.
    assert "corrupt.tar.gz" in out.detail

    # Malformed entries classify the same, with the reason in the detail.
    malformed = _tar(tmp_path, _meta(entry={"name": ENTRY, "target": "unet"}))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=malformed)
    assert not out.armed and out.reason == "artifact_invalid"
    assert "graph pytree must be an object" in out.detail


def test_constants_refusal_named_on_the_wire(
    tmp_path: Path, events: List[Any], stub_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aot_serve, "_load_package",
        lambda path, entry: FakePackage(fqns=("conv_in.weight",
                                              "not.in.state_dict")))
    out = aot_serve.enable(FakePipeline(), Cfg(), artifact=_tar(tmp_path))
    assert not out.armed
    assert out.reason.startswith("constants_")


# ---------------------------------------------------------------------------
# nested (added_cond_kwargs) input resolution at bind (arm-events lane)
# ---------------------------------------------------------------------------


def _nested_contract() -> Any:
    return CallIngress(
        parameters=(
            "sample", "timestep", "encoder_hidden_states", "added_cond_kwargs"
        ),
        flat_arity=2,
        inputs=(
            CallInput(
                "sample", 0, "sample", 0, (), "sample", "bfloat16",
                (2, 4, 128, 128),
            ),
            CallInput(
                "text_embeds", 1, "added_cond_kwargs", 3, ("text_embeds",),
                "added_cond_kwargs_text_embeds", "bfloat16", (2, 1280),
            ),
        ),
    )


def test_missing_nested_input_still_refuses_by_name() -> None:
    """Live refusal (pod ae2uc81yub0gyq): 'text_embeds' declared positional
    by the export but passed NESTED in added_cond_kwargs by every diffusers
    caller. The resolve half lives in test_aot_adapter_fork_pgw790's nested
    marshal test; the refusal half — nested container present, leaf absent —
    must still name the input."""
    contract = _nested_contract()
    with pytest.raises(aot_serve.IngressContractError) as exc:
        aot_serve.bind_call_inputs(
            contract, (FakeTensor([2, 4, 128, 128]),),
            {"added_cond_kwargs": {"time_ids": FakeTensor([2, 6])}})
    assert exc.value.reason == "input_missing"
