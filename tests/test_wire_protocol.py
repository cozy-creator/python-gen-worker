"""Hub wire protocol: the vocabulary, the major, and the fields a worker ships.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import grpc
import msgspec
import pytest
from harness.hub_double import hub_double, is_ready

from gen_worker import RequestContext, Resources, endpoint
from gen_worker import measured_posture as mp
from gen_worker.config import load_settings
from gen_worker.discovery.discover import _extract_entries
from gen_worker.families import GenerationDefaults
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit.parent import ParentControl
from gen_worker.topology import ExecutionTopology
from gen_worker.transport import _CONNECT_METHOD

# ============================================================================
# th#1597 — DESIGN-RULINGS §1.27(b),(g): the wire-protocol MAJOR is the
#   proto package, so it is in the gRPC service path.
# ============================================================================

_TIMEOUT = 15.0


_PRE_V1_SERVICE = "cozy.scheduler.WorkerScheduler"


def test_connect_path_carries_the_major() -> None:
    """§1.27(b): the major lives in the package, therefore in the path."""
    assert pb.DESCRIPTOR.package == "cozy.scheduler.v1"
    assert _CONNECT_METHOD == "/cozy.scheduler.v1.WorkerScheduler/Connect"
    # §1.27(g): the first package is v1, and nothing pre-launch claims a
    # history it does not have.
    assert pb.PROTOCOL_VERSION_CURRENT == 1


def test_v1_worker_handshakes_with_a_v1_hub() -> None:
    """th#1597: The positive control the refusal test below is only meaningful against: on the matching major th..."""
    with hub_double(worker_id="th1597-right-major") as (scheduler, harness):
        conn = scheduler.wait_connection(0)
        assert conn.hello is not None
        assert conn.hello.protocol_version == pb.PROTOCOL_VERSION_CURRENT
        conn.wait_for(is_ready)
        assert harness.alive


def _serve_only_the_pre_v1_path() -> tuple[grpc.Server, int]:
    """th#1597: A hub that serves the OLD unversioned path and nothing else."""

    def _unreached(request_iterator, context):  # pragma: no cover - never dialed
        raise AssertionError("a v1 client must not reach the pre-v1 handler")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                _PRE_V1_SERVICE,
                {
                    "Connect": grpc.stream_stream_rpc_method_handler(
                        _unreached,
                        request_deserializer=pb.WorkerMessage.FromString,
                        response_serializer=pb.SchedulerMessage.SerializeToString,
                    )
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def test_a_hub_that_does_not_serve_this_major_is_fatal_not_a_reconnect_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from gen_worker.config import load_settings
    from gen_worker.worker import Worker

    server, port = _serve_only_the_pre_v1_path()
    try:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="th1597-wrong-major",
            worker_jwt="",
        )
        worker = Worker(
            settings, ["harness.toy_endpoints"], backoff_base_s=0.05, backoff_cap_s=0.2
        )
        exit_code: Optional[int] = None

        def _run() -> None:
            nonlocal exit_code
            exit_code = worker.run()

        with caplog.at_level(logging.ERROR):
            thread = threading.Thread(target=_run, name="th1597-worker", daemon=True)
            thread.start()
            thread.join(timeout=_TIMEOUT)

        assert not thread.is_alive(), (
            "the worker is still running against a hub that does not serve its "
            "wire-protocol major — it is reconnect-looping, so no pod ever dies "
            "pre-Hello and th#874 can never mark the release boot_crashing"
        )
        assert exit_code == 1, f"expected a fatal exit, got {exit_code!r}"

        # It exited for the RIGHT reason, and the reason is typed. Asserting
        # only the exit code would pass for any fatal at all — including the
        # unreachable-hub and auth families, which are the ones this code path
        # must NOT be confused with. Deliberately NOT asserted: that zero
        # reconnect delays were recorded. A channel that is not ready yet
        # raises before any status arrives, and retrying THAT is correct; only
        # a delivered UNIMPLEMENTED is unretryable.
        fatal = "\n".join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR
        )
        assert "does not serve this wire-protocol major" in fatal, fatal
        assert _CONNECT_METHOD in fatal, fatal
    finally:
        server.stop(grace=0)


# ============================================================================
# pgw#876 — §4 — the recurrence guard for "two builders for the same
#   fact, only one of which is ever on the wire".
# ============================================================================

_INTENTIONALLY_UNSET = frozenset({"git_commit"})


_MEASUREMENT = {
    "hardware": {
        "gpu_count": 4,
        "vram_total_bytes": 85899345920,
        "vram_free_bytes": 42949672960,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_sm": "90",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "installed_libs": ["diffusers==0.36.0"],
        # The HOST driver. 580.159.04 is a real RunPod draw
        # and the tuple-vs-float trap (as floats 580.159 < the 580.65 floor).
        "driver_version": "580.159.04",
    },
    "canary": {
        "memcpy_gbps": 1.5,
        "d2h_gbps": 2.5,
        "pinned_alloc_ok": True,
        "cpu_single_mbps": 3.5,
        "cpu_multi_mbps": 4.5,
        "vcpus": 32,
        "ram_total_gb": 251.0,
        "duration_ms": 1234,
        "interconnect": "nvlink",
        "peer_gbps": 5.5,
        "peer_access": True,
        "topo_link": "NV18",
    },
    "gen_worker_version": "0.90.6",
}


def _parent_control(**settings_kw: object) -> ParentControl:
    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1",
        worker_id="w-pgw876",
        **settings_kw,
    )
    return ParentControl(
        settings,
        socket_path="/tmp/gen-worker-pgw876.sock",
        topology=ExecutionTopology.single(),
    )


def _default_valued_fields(msg: pb.WorkerResources) -> set:
    """pgw#876: Field names still carrying the protobuf default — i.e."""
    on_the_wire = {field.name for field, _ in msg.ListFields()}
    return {f.name for f in pb.WorkerResources.DESCRIPTOR.fields} - on_the_wire


def test_the_parent_builder_assigns_every_wire_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgw#876: THE `worker_mode=""` GUARD, at the value level."""
    pc = _parent_control(
        worker_image_digest="sha256:deadbeef",
        runpod_pod_id="pod-pgw876",
    )
    pc._measurement = dict(_MEASUREMENT)

    res = pc._parent_resources()
    assert res is not None

    unset = _default_valued_fields(res) - _INTENTIONALLY_UNSET
    assert not unset, (
        f"the ON-THE-WIRE WorkerResources builder never assigns {sorted(unset)}. "
        "An empty value here is not a default the hub can read as a choice — it "
        "is the signature of a field the ONE builder was never taught. "
        "th#1359 Part 2 did exactly that "
        "with `worker_mode` and every forge pod bought afterwards was "
        "idle-reaped as a serving pod."
    )


def test_the_retired_wire_words_are_gone_from_the_contract() -> None:
    """§4.28 / th#1751 W4 + pgw#1092 — the vocabulary cut, at the DESCRIPTOR and in the contract text."""
    assert "worker_mode" not in {
        f.name for f in pb.WorkerResources.DESCRIPTOR.fields}
    assert "requested_cell_axes" not in {
        f.name for f in pb.CompileTarget.DESCRIPTOR.fields}
    # ...and no lane may reclaim the numbers or the names (§1.27(f): a
    # within-major retirement reserves BOTH). The python descriptor does not
    # expose reserved ranges, so the vendored contract itself is the assertion.
    contract = (
        Path(__file__).resolve().parents[1] / "proto" / "worker_scheduler.proto"
    ).read_text()
    assert "reserved 12;" in contract and 'reserved "worker_mode";' in contract
    assert "reserved 11;" in contract
    assert 'reserved "requested_cell_axes";' in contract


def _append_unknown_string(wire: bytearray, field_no: int, value: bytes) -> None:
    """Append `field_no` as a length-delimited (wire type 2) string."""
    from google.protobuf.internal import encoder as _encoder

    _encoder._VarintEncoder()(wire.extend, (field_no << 3) | 2, False)
    _encoder._VarintEncoder()(wire.extend, len(value), False)
    wire.extend(value)


def test_a_wheel_that_still_sends_the_retired_fields_is_not_refused() -> None:
    """pgw#876: THE FLEET-SAFETY CLAIM, proved rather than asserted in a PR body."""
    old = pb.WorkerResources(gpu_count=4, gpu_sm="90")
    wire = bytearray(old.SerializeToString())
    _append_unknown_string(wire, 12, b"forge")

    fresh = pb.WorkerResources()
    assert fresh.ParseFromString(bytes(wire)) == len(wire)
    assert fresh.gpu_count == 4 and fresh.gpu_sm == "90"
    assert not hasattr(fresh, "worker_mode")

    # Same for the CompileTarget half: field 11 was a map, whose entries are
    # also length-delimited, so a single entry is the honest shape to feed it.
    target = pb.CompileTarget(family="sdxl", requested_cell_key="ck1-abc")
    twire = bytearray(target.SerializeToString())
    _append_unknown_string(twire, 11, b"\n\x03sku\x12\x04L40S")

    got = pb.CompileTarget()
    assert got.ParseFromString(bytes(twire)) == len(twire)
    assert got.family == "sdxl" and got.requested_cell_key == "ck1-abc"
    assert not hasattr(got, "requested_cell_axes")


# ============================================================================
# pgw#1314 — `cuda_version` reaches the hub from a LIVE worker.
# ============================================================================

_MEASURED: dict[str, Any] = {
    "hardware": {
        "gpu_count": 1,
        "vram_total_bytes": 85899345920,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_sm": "90",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "driver_version": "580.159.04",
        "installed_libs": ["torchao"],
    },
    "gen_worker_version": "0.118.0",
}


def _parent() -> ParentControl:
    return ParentControl(
        load_settings(orchestrator_public_addr="127.0.0.1:1",
                      worker_id="w-pgw1314"),
        socket_path="/tmp/gen-worker-pgw1314.sock",
        topology=ExecutionTopology.single(),
    )


def _wire_fields(msg: pb.WorkerResources) -> set[str]:
    """pgw#1314: What actually goes on the wire — proto3 serializes a singular scalar only when it differs from ..."""
    return {field.name for field, _ in msg.ListFields()}


def test_a_live_workers_resources_carry_the_measured_cuda_version() -> None:
    pc = _parent()
    pc._measurement = dict(_MEASURED)
    res = pc._parent_resources()
    assert res is not None
    assert res.cuda_version == "13.0"
    assert "cuda_version" in _wire_fields(res)


def test_it_rides_the_HELLO_the_hub_receives_not_just_the_builder() -> None:
    """pgw#1314: The assertion that survives a refactor of the Hello path: the field is on the message the trans..."""
    pc = _parent()
    pc._measurement = dict(_MEASURED)
    hello = pb.Hello(worker_id="stale-child-claim")
    pc._apply_identity_and_resources(hello)
    assert hello.resources.cuda_version == "13.0"
    round_tripped = pb.Hello()
    round_tripped.ParseFromString(hello.SerializeToString())
    assert round_tripped.resources.cuda_version == "13.0"


def test_an_unreadable_cuda_runtime_still_HELLOS_and_the_field_is_ABSENT(
) -> None:
    """pgw#1314: The always-runs posture, on this axis: a host whose CUDA runtime cannot be read is not a `Hardw..."""
    hardware: dict[str, Any] = {**_MEASURED["hardware"], "cuda_version": ""}
    measurement: dict[str, Any] = {**_MEASURED, "hardware": hardware}
    pc = _parent()
    pc._measurement = measurement

    res = pc._parent_resources()
    assert res is not None, "an unreadable CUDA runtime is not a dead parent"
    assert "cuda_version" not in _wire_fields(res)
    # ...and the rest of the measurement is untouched: the axis is undeclared,
    # not the machine unmeasured.
    assert res.gpu_sm == "90" and res.driver_version == "580.159.04"

    hello = pb.Hello()
    pc._apply_identity_and_resources(hello)
    assert hello.HasField("resources")
    assert "cuda_version" not in _wire_fields(hello.resources)


def test_an_unmeasured_HOST_still_ships_no_resources_at_all() -> None:
    """pgw#898's rule, unweakened: absent measurement means NO `resources`, loudly — never a partially-filled st..."""
    pc = _parent()
    pc._measurement = None
    assert pc._parent_resources() is None
    hello = pb.Hello()
    pc._apply_identity_and_resources(hello)
    assert not hello.HasField("resources")


def test_the_fact_is_spelled_the_same_on_both_carriers() -> None:
    """pgw#1314: ONE vocabulary."""
    live = {f.name for f in pb.WorkerResources.DESCRIPTOR.fields}
    corpse = {f.name for f in pb.HardwareUnsuitable.DESCRIPTOR.fields}
    assert "cuda_version" in live
    assert "torch_cuda_version" in corpse
    # The SM axis settles the one real spelling trap: `HostFacts.gpu_sm` and
    # `WorkerResources.gpu_sm` are BARE ("90"), which is also the spelling
    # `min_sm` uses. Anything dotted is a normalization at its own boundary.
    pc = _parent()
    pc._measurement = dict(_MEASURED)
    res = pc._parent_resources()
    assert res is not None and res.gpu_sm == "90"


@pytest.mark.parametrize("reserved", ["worker_mode"])
def test_the_new_field_did_not_reclaim_a_retired_number(reserved: str) -> None:
    names = {f.name: f.number for f in pb.WorkerResources.DESCRIPTOR.fields}
    assert reserved not in names
    assert names["cuda_version"] == 14


# ============================================================================
# pgw#1239 — Bind the public Hub-worker boundary corpus to python-gen-worker
#   sources.
# ============================================================================

_ROOT = Path(__file__).parents[1]


_DEFAULT_CORPUS = Path(__file__).parent / "testdata" / "hub_worker_boundary_contracts.json"


_DEFAULT_DIGEST = Path(__file__).parent / "testdata" / "HUB_WORKER_BOUNDARY_CONTRACTS_DIGEST"


_CORPUS = Path(os.environ.get("HUB_WORKER_BOUNDARY_CONTRACT_FILE", _DEFAULT_CORPUS))


_SOURCE_PATHS = {
    "loader": _ROOT / "src" / "gen_worker" / "config" / "loader.py",
    "settings": _ROOT / "src" / "gen_worker" / "config" / "settings.py",
    "c2pa": _ROOT / "src" / "gen_worker" / "content_credentials.py",
    "topology": _ROOT / "src" / "gen_worker" / "topology.py",
    "procsplit": _ROOT / "src" / "gen_worker" / "procsplit" / "__init__.py",
    "discovery": _ROOT / "src" / "gen_worker" / "discovery" / "discover.py",
    # pgw#1331: `runtime_key()` moved to `compile_facts`, the READ half of the
    # compile cache — `compile_cache` re-exports it. The raw launch-value read
    # is checked where it lives, not where it is re-exported from.
    "compile_facts": _ROOT / "src" / "gen_worker" / "compile_facts.py",
    "model_store": _ROOT / "src" / "gen_worker" / "models" / "store.py",
    "provision": _ROOT / "src" / "gen_worker" / "models" / "provision.py",
}


_SOURCE_ENV = {name: f"HUB_WORKER_BOUNDARY_{name.upper()}_SOURCE" for name in _SOURCE_PATHS}


def _document() -> dict[str, Any]:
    document = json.loads(_CORPUS.read_text(encoding="utf-8"))
    assert document["schema"] == "hub-worker-boundary-contracts-v1"
    return document


def _sources() -> dict[str, str]:
    return {
        name: Path(os.environ.get(_SOURCE_ENV[name], path)).read_text(encoding="utf-8")
        for name, path in _SOURCE_PATHS.items()
    }


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    raise AssertionError(f"source assignment {name} is missing")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    return ast.literal_eval(_assignment(tree, name))


def _strings_in_assignment(tree: ast.Module, name: str) -> set[str]:
    return {
        node.value
        for node in ast.walk(_assignment(tree, name))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _settings_fields(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    raise AssertionError("Settings class is missing")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"source function {name} is missing")


def _method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"source method {class_name}.{name} is missing")


def _prints_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "print"
        and any(isinstance(arg, ast.Name) and arg.id == name for arg in item.args)
        for item in ast.walk(node)
    )


def _calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == name
        for item in ast.walk(node)
    )


def _raw_environ_gets(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "environ"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
        ):
            continue
        key = node.args[0]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            values.add(key.value)
    return values


def _attribute_names(tree: ast.AST) -> set[str]:
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _assert_contracts(document: dict[str, Any], sources: dict[str, str]) -> None:
    contracts = document["contracts"]
    trees = {name: ast.parse(source) for name, source in sources.items()}

    loader_map = _literal_assignment(trees["loader"], "_ENV_TO_FIELD")
    settings_fields = _settings_fields(trees["settings"])

    active = {row["env"]: row["field"] for row in contracts["active_launch_settings"]}
    assert len(active) == 6, "active_launch_settings must contain exactly six rows"
    for env_name, field_name in active.items():
        assert loader_map.get(env_name) == field_name, env_name
        assert field_name in settings_fields, field_name
    assert "WORKER_IMAGE_DIGEST" in _raw_environ_gets(trees["compile_facts"]), (
        "compile_facts must consume the raw WORKER_IMAGE_DIGEST launch value"
    )

    external_secret_fields = {
        "CIVITAI_API_KEY": "civitai_api_key",
        "HF_TOKEN": "hf_token",
    }
    external_secret_names = contracts["external_secret_env_names"]
    assert external_secret_names == sorted(external_secret_names)
    assert external_secret_names == sorted(external_secret_fields)
    provision_fields = _attribute_names(trees["provision"])
    for env_name, field_name in external_secret_fields.items():
        assert loader_map.get(env_name) == field_name, env_name
        assert field_name in settings_fields, field_name
        assert field_name in provision_fields, (
            f"models.provision no longer consumes Settings.{field_name}"
        )

    c2pa = contracts["c2pa"]
    supplied = {row["env"]: row["field"] for row in c2pa["supplied"]}
    assert len(supplied) == 3, "C2PA supplied must contain exactly three rows"
    for env_name, field_name in supplied.items():
        assert loader_map.get(env_name) == field_name, env_name
        assert field_name in settings_fields, field_name

    forbidden = set(c2pa["forbidden"])
    loader_forbidden = set(_literal_assignment(trees["loader"], "REFUSED_KEY_MATERIAL"))
    runtime_forbidden = set(_literal_assignment(trees["c2pa"], "_REFUSED_KEY_ENVS"))
    assert forbidden == loader_forbidden, "loader C2PA forbidden set drifted"
    assert forbidden == runtime_forbidden, "runtime C2PA forbidden set drifted"
    assert not {name.lower().removeprefix("gen_worker_") for name in forbidden} & settings_fields, (
        "forbidden C2PA key material became a Settings field"
    )

    fill = contracts["managed_fill_source"]
    assert loader_map.get(fill["env"]) == fill["field"]
    assert fill["field"] in settings_fields
    store_init = _method(trees["model_store"], "ModelStore", "__init__")
    assert _calls_name(store_init, "tensorhub_fill_source_dir"), (
        "ModelStore no longer resolves the managed fill source"
    )
    assert fill["env"] in _raw_environ_gets(store_init), (
        "ModelStore no longer names the managed fill env in its boot diagnosis"
    )

    topology_env = contracts["execution_topology"]["env"]
    assert _literal_assignment(trees["topology"], "ENV_VAR") == topology_env
    assert _literal_assignment(trees["procsplit"], "ENV_TOPOLOGY") == topology_env
    assert topology_env in _strings_in_assignment(trees["loader"], "_OWNED_NON_SETTINGS")

    marker = contracts["build_input_failure"]["marker"]
    assert _literal_assignment(trees["discovery"], "BUILD_INPUT_FAILURE_MARKER") == marker
    assert _prints_name(
        _function(trees["discovery"], "_fail_build_input"),
        "BUILD_INPUT_FAILURE_MARKER",
    ), "discovery no longer emits its build-input failure marker"


def test_hub_worker_boundary_contracts_match_pgw1239() -> None:
    _assert_contracts(_document(), _sources())


def test_hub_worker_boundary_digest_matches_pgw1239() -> None:
    active = [
        line.strip().split()[0]
        for line in _DEFAULT_DIGEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(active) == 1
    assert active[0] == hashlib.sha256(_DEFAULT_CORPUS.read_bytes()).hexdigest()


def test_drift_script_carries_every_private_consumer_path_pgw1239() -> None:
    script = (_ROOT / "scripts" / "hub-worker-boundary-drift.sh").read_text(encoding="utf-8")
    assert 'hub_rel="internal/wirecontract/testdata"' in script
    assert 'trainer_rel="image_lora_finetuner/tests/testdata"' in script
    assert 'if [ "$side" = "pgw" ] && [ -z "$peer_dir" ]; then' in script


def _run_contract_test(
    *, corpus: Path | None = None, source_override: tuple[str, Path] | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if corpus is not None:
        env["HUB_WORKER_BOUNDARY_CONTRACT_FILE"] = os.fspath(corpus)
    if source_override is not None:
        name, path = source_override
        env[_SOURCE_ENV[name]] = os.fspath(path)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            os.fspath(Path(__file__)),
            "-k",
            "test_hub_worker_boundary_contracts_match_pgw1239",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "contract_class",
    [
        "active_launch_settings",
        "external_secret_env_names",
        "c2pa",
        "managed_fill_source",
        "execution_topology",
        "build_input_failure",
    ],
)
def test_each_contract_class_has_semantic_red_pgw1239(tmp_path: Path, contract_class: str) -> None:
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    contracts = document["contracts"]
    if contract_class == "active_launch_settings":
        contracts[contract_class][0]["field"] = "broken_active_launch_field"
    elif contract_class == "external_secret_env_names":
        contracts[contract_class][0] = "BROKEN_EXTERNAL_SECRET_ENV"
    elif contract_class == "c2pa":
        contracts[contract_class]["supplied"][0]["field"] = "broken_c2pa_field"
    elif contract_class == "managed_fill_source":
        contracts[contract_class]["field"] = "broken_fill_field"
    elif contract_class == "execution_topology":
        contracts[contract_class]["env"] = "BROKEN_EXECUTION_TOPOLOGY"
    else:
        contracts[contract_class]["marker"] = "BROKEN_BUILD_INPUT_FAILURE"

    corpus = tmp_path / _DEFAULT_CORPUS.name
    corpus.write_text(json.dumps(document), encoding="utf-8")
    got = _run_contract_test(corpus=corpus)
    assert got.returncode == 1, got.stdout + got.stderr


@pytest.mark.parametrize(
    ("source_name", "old", "new"),
    [
        ("loader", '"WORKER_ID": "worker_id"', '"WORKER_ID": "broken_worker_id"'),
        ("settings", "worker_id: str =", "broken_worker_id: str ="),
        ("loader", '"HF_TOKEN": "hf_token"', '"HF_TOKEN": "broken_hf_token"'),
        (
            "provision",
            "current_or(_STANDALONE).civitai_api_key",
            "current_or(_STANDALONE).hf_home",
        ),
        (
            "loader",
            '"GEN_WORKER_C2PA_KEY_PEM": (',
            '"GEN_WORKER_C2PA_KEY_PEM_BROKEN": (',
        ),
        (
            "c2pa",
            '"GEN_WORKER_C2PA_KEY_PEM",',
            '"GEN_WORKER_C2PA_KEY_PEM_BROKEN",',
        ),
        (
            "model_store",
            "fill_source_dir or tensorhub_fill_source_dir()",
            "fill_source_dir or tensorhub_cas_dir()",
        ),
        (
            "topology",
            'ENV_VAR = "WORKER_EXECUTION_TOPOLOGY"',
            'ENV_VAR = "BROKEN_EXECUTION_TOPOLOGY"',
        ),
        (
            "procsplit",
            'ENV_TOPOLOGY = "WORKER_EXECUTION_TOPOLOGY"',
            'ENV_TOPOLOGY = "BROKEN_EXECUTION_TOPOLOGY"',
        ),
        (
            "loader",
            '    "WORKER_EXECUTION_TOPOLOGY",',
            '    "BROKEN_EXECUTION_TOPOLOGY",',
        ),
        (
            "discovery",
            'BUILD_INPUT_FAILURE_MARKER = "TENSORHUB_BUILD_INPUT_FAILURE:discovery"',
            'BUILD_INPUT_FAILURE_MARKER = "BROKEN_BUILD_INPUT_FAILURE"',
        ),
        (
            "compile_facts",
            'os.environ.get("WORKER_IMAGE_DIGEST", "")',
            'os.environ.get("BROKEN_WORKER_IMAGE_DIGEST", "")',
        ),
    ],
)
def test_each_source_binding_has_red_pgw1239(
    tmp_path: Path, source_name: str, old: str, new: str
) -> None:
    source = _SOURCE_PATHS[source_name].read_text(encoding="utf-8")
    assert source.count(old) == 1, (source_name, old, source.count(old))
    candidate = tmp_path / f"{source_name}.py"
    candidate.write_text(source.replace(old, new, 1), encoding="utf-8")
    got = _run_contract_test(source_override=(source_name, candidate))
    assert got.returncode == 1, got.stdout + got.stderr


def test_digest_gate_can_go_red_pgw1239(tmp_path: Path) -> None:
    corpus = tmp_path / _DEFAULT_CORPUS.name
    digest = tmp_path / _DEFAULT_DIGEST.name
    corpus.write_bytes(_DEFAULT_CORPUS.read_bytes() + b"\n")
    digest.write_bytes(_DEFAULT_DIGEST.read_bytes())
    got = subprocess.run(
        [
            sys.executable,
            os.fspath(_ROOT / "scripts" / "check_hub_worker_boundary_contracts_digest.py"),
            "--corpus",
            os.fspath(corpus),
            "--digest",
            os.fspath(digest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "changed without its digest" in got.stdout


def test_redigested_peer_mutation_reaches_byte_comparison_pgw1239(
    tmp_path: Path,
) -> None:
    peer = tmp_path / "peer"
    peer.mkdir()
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    document["contracts"]["active_launch_settings"][0]["field"] = "peer_mutation"
    peer_corpus = peer / _DEFAULT_CORPUS.name
    peer_corpus.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    peer_digest = hashlib.sha256(peer_corpus.read_bytes()).hexdigest()
    (peer / _DEFAULT_DIGEST.name).write_text(
        f"{peer_digest}  {_DEFAULT_CORPUS.name}\n", encoding="utf-8"
    )

    got = subprocess.run(
        [os.fspath(_ROOT / "scripts" / "hub-worker-boundary-drift.sh")],
        env={**os.environ, "HUB_WORKER_BOUNDARY_PEER_DIR": os.fspath(peer)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "differs from python-gen-worker" in got.stderr
    assert "peer corpus does not match its digest" not in got.stderr


# ============================================================================
# pgw#1225 — Layer 1 of the posture wire fence, THIS side (th#1871 P1 /
#   pgw#1225).
# ============================================================================

VECTORS = pathlib.Path(__file__).parent / "testdata" / "posture_wire_vectors.json"


def _doc() -> Mapping[str, object]:
    return cast(Mapping[str, object], json.loads(VECTORS.read_text()))


def _vectors() -> List[Mapping[str, object]]:
    return cast(List[Mapping[str, object]], _doc()["vectors"])


def _rules() -> List[Mapping[str, object]]:
    return cast(List[Mapping[str, object]], _doc()["identity_rules"])


def _str(src: Mapping[str, object], key: str) -> str:
    return str(src.get(key, "") or "")


def _int(src: Mapping[str, object], key: str) -> int:
    return int(cast(int, src.get(key, 0) or 0))


def _float(src: Mapping[str, object], key: str) -> float:
    return float(cast(float, src.get(key, 0.0) or 0.0))


def _wire_json(posture: mp.MeasuredPosture) -> Dict[str, object]:
    """pgw#1225: The posture as the HUB will hold it — the JSON shape of Go's ``measurement.Posture``, key for k..."""
    out: Dict[str, object] = {
        "execution_lane": posture.execution_lane,
        "attention_backend": posture.attention_backend,
    }
    if posture.attention_backend_wanted:
        out["attention_backend_wanted"] = posture.attention_backend_wanted
    if posture.compile_state:
        out["compile_state"] = posture.compile_state
    if posture.compile_state_wanted:
        out["compile_state_wanted"] = posture.compile_state_wanted
    if posture.residency_mode:
        out["residency_mode"] = posture.residency_mode
    if posture.applied:
        applied: List[Dict[str, object]] = []
        for technique in posture.applied:
            entry: Dict[str, object] = {"name": technique.name}
            if technique.component:
                entry["component"] = technique.component
            if technique.wanted:
                entry["wanted"] = technique.wanted
            if technique.reason:
                entry["reason"] = technique.reason
            if technique.est_slowdown:
                entry["est_slowdown"] = technique.est_slowdown
            applied.append(entry)
        out["applied"] = applied
    if posture.components:
        components: List[Dict[str, object]] = []
        for component in posture.components:
            item: Dict[str, object] = {"component": component.component}
            if component.applied_quant:
                item["applied_quant"] = component.applied_quant
            if component.bound_quant:
                item["bound_quant"] = component.bound_quant
            if component.placement:
                item["placement"] = component.placement
            if component.size_bytes:
                item["bytes"] = component.size_bytes
            components.append(item)
        out["components"] = components
    if posture.shortfall is not None:
        shortfall: Dict[str, object] = {"resource": posture.shortfall.resource}
        if posture.shortfall.component:
            shortfall["component"] = posture.shortfall.component
        shortfall["needed_bytes"] = posture.shortfall.needed_bytes
        shortfall["available_bytes"] = posture.shortfall.available_bytes
        out["shortfall"] = shortfall
    return out


def _posture_from_proto(msg: object) -> mp.MeasuredPosture:
    """pgw#1225: Wire -> record, for the round-trip assertion below."""
    proto = cast(pb.MeasuredPosture, msg)
    shortfall: Optional[mp.ResourceShortfall] = None
    if proto.HasField("shortfall"):
        shortfall = mp.ResourceShortfall(
            resource=proto.shortfall.resource,
            component=proto.shortfall.component,
            needed_bytes=proto.shortfall.needed_bytes,
            available_bytes=proto.shortfall.available_bytes)
    return mp.MeasuredPosture(
        execution_lane=proto.execution_lane,
        attention_backend=proto.attention_backend,
        attention_backend_wanted=proto.attention_backend_wanted,
        compile_state=proto.compile_state,
        compile_state_wanted=proto.compile_state_wanted,
        residency_mode=proto.residency_mode,
        applied=tuple(
            mp.AppliedTechnique(
                name=t.name, component=t.component, wanted=t.wanted,
                reason=t.reason, est_slowdown=t.est_slowdown)
            for t in proto.applied),
        components=tuple(
            mp.ComponentPosture(
                component=c.component, applied_quant=c.applied_quant,
                bound_quant=c.bound_quant, placement=c.placement,
                size_bytes=c.bytes)
            for c in proto.components),
        shortfall=shortfall,
    )


def _posture_from_wire(wire: Mapping[str, object]) -> mp.MeasuredPosture:
    """pgw#1225: Rebuild the SDK type from the ledger's wire object."""
    applied_raw = cast(Sequence[Mapping[str, object]], wire.get("applied", []) or [])
    components_raw = cast(
        Sequence[Mapping[str, object]], wire.get("components", []) or [])
    shortfall_raw = cast(
        Optional[Mapping[str, object]], wire.get("shortfall") or None)
    shortfall: Optional[mp.ResourceShortfall] = None
    if shortfall_raw is not None:
        shortfall = mp.ResourceShortfall(
            resource=_str(shortfall_raw, "resource"),
            component=_str(shortfall_raw, "component"),
            needed_bytes=_int(shortfall_raw, "needed_bytes"),
            available_bytes=_int(shortfall_raw, "available_bytes"),
        )
    return mp.MeasuredPosture(
        execution_lane=_str(wire, "execution_lane"),
        attention_backend=_str(wire, "attention_backend"),
        attention_backend_wanted=_str(wire, "attention_backend_wanted"),
        compile_state=_str(wire, "compile_state"),
        compile_state_wanted=_str(wire, "compile_state_wanted"),
        residency_mode=_str(wire, "residency_mode"),
        applied=tuple(
            mp.AppliedTechnique(
                name=_str(t, "name"), component=_str(t, "component"),
                wanted=_str(t, "wanted"), reason=_str(t, "reason"),
                est_slowdown=_float(t, "est_slowdown"))
            for t in applied_raw),
        components=tuple(
            mp.ComponentPosture(
                component=_str(c, "component"),
                applied_quant=_str(c, "applied_quant"),
                bound_quant=_str(c, "bound_quant"),
                placement=_str(c, "placement"),
                size_bytes=_int(c, "bytes"))
            for c in components_raw),
        shortfall=shortfall,
    )


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: str(v["name"]))
def test_every_vector_is_what_this_sdk_serializes(
    vector: Mapping[str, object],
) -> None:
    """The producer half: the SDK's wire shape IS the ledger's."""
    wire = cast(Mapping[str, object], vector["wire"])
    posture = _posture_from_wire(wire)
    assert _wire_json(posture) == wire, (
        f"{vector['name']}: this SDK serializes a posture differently than the "
        f"ledger tensorhub digests. That re-keys the measurement relation "
        f"silently — every affected cell forks. ({vector['why']})")


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: str(v["name"]))
def test_every_vector_survives_the_proto_round_trip(
    vector: Mapping[str, object],
) -> None:
    """pgw#1225: And it is the same posture after the WIRE, not only in Python."""
    posture = _posture_from_wire(cast(Mapping[str, object], vector["wire"]))
    back = _posture_from_proto(posture.to_proto())
    assert _wire_json(back) == _wire_json(posture), (
        f"{vector['name']}: the posture did not survive the proto round trip — "
        f"a field the message cannot carry is a field the hub never sees")


def test_degraded_verdict_matches_the_ledger() -> None:
    """pgw#1225: The worker's own reading of "was this degraded" must match the hub's."""
    for vector in _vectors():
        posture = _posture_from_wire(cast(Mapping[str, object], vector["wire"]))
        expected = bool(vector["degraded"])
        assert posture.degraded == expected, (
            f"{vector['name']}: SDK degraded={posture.degraded}, hub says "
            f"{expected} — the pod's own logs would contradict the measurement")


def test_identity_rules_name_real_vectors_and_state_a_structural_claim() -> None:
    """pgw#1225: What the WORKER can check about the rules, which is not the digests."""
    names = {str(v["name"]) for v in _vectors()}
    assert names, "the vector ledger is empty"
    rules = _rules()
    assert rules, "a fence with no rules passes for the same reason an empty file does"
    for rule in rules:
        for side in ("a", "b"):
            assert str(rule[side]) in names, (
                f"identity rule {rule['rule']!r} names unknown vector "
                f"{rule[side]!r}")


def test_the_ie707_pair_differs_only_in_what_was_asked_for() -> None:
    """pgw#1225: The structural claim under the wanted-is-identity rule, checkable here."""
    by_name: Dict[str, Mapping[str, object]] = {
        str(v["name"]): cast(Mapping[str, object], v["wire"]) for v in _vectors()}
    flash = dict(by_name["isolated_wanted_flash"])
    sdpa = dict(by_name["isolated_wanted_sdpa"])
    differing: Tuple[str, ...] = tuple(
        sorted(k for k in set(flash) | set(sdpa)
               if flash.get(k) != sdpa.get(k)))
    assert differing == ("attention_backend_wanted",), (
        f"the isolated ie#707 pair differs in {differing}, not in the wanted "
        f"side alone — the hub's wanted-is-identity rule would then pass for a "
        f"reason that is not the rule")


# ============================================================================
# pgw#748/th#1285 — the author ENVELOPE has an SDK carrier.
# ============================================================================

class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 30


class In(msgspec.Struct):
    prompt: str = "x"


class Out(msgspec.Struct):
    ok: bool = True


@endpoint(resources=Resources(
    gpu=True, max_gpu_count=2, parallel=("sequence",)))
class SPEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


@endpoint(resources=Resources(gpu=True))
class PlainEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


def _resources(cls: type) -> Dict[str, Any]:
    (fn,) = _extract_entries(cls, "testmod")
    return fn["resources"]


def test_envelope_reaches_manifest_under_builder_keys() -> None:
    res = _resources(SPEndpoint)
    assert res["max_gpu_count"] == 2
    # In-memory the projection may hold a tuple; on the wire (JSON) it is a
    # list — the builder's normalizeParallelMechanisms accepts exactly that.
    assert list(res["parallel"]) == ["sequence"]
    assert msgspec.json.decode(msgspec.json.encode(res))["parallel"] == ["sequence"]


def test_undeclared_envelope_is_absent_not_defaulted() -> None:
    res = _resources(PlainEndpoint)
    assert "max_gpu_count" not in res
    assert "parallel" not in res


def test_max_gpu_count_implies_gpu() -> None:
    assert Resources(max_gpu_count=2).gpu is True


def test_parallel_tokens_normalized() -> None:
    r = Resources(gpu=True, max_gpu_count=4, parallel=(" Sequence ",))
    assert r.parallel == ("sequence",)


def test_unknown_mechanism_refused_at_declaration() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        Resources(gpu=True, max_gpu_count=2, parallel=("tensor",))


def test_parallel_without_headroom_refused() -> None:
    # No ceiling at all.
    with pytest.raises(ValueError, match="headroom"):
        Resources(gpu=True, parallel=("sequence",))
    # Ceiling equal to the floor — same contradiction the builder refuses.
    with pytest.raises(ValueError, match="headroom"):
        Resources(gpu=True, gpu_count=2, max_gpu_count=2, parallel=("sequence",))


def test_ceiling_below_floor_refused() -> None:
    with pytest.raises(ValueError, match="below gpu_count"):
        Resources(gpu=True, gpu_count=4, max_gpu_count=2)


@endpoint(resources=Resources(
    gpu=True, max_gpu_count=4,
    max_gpus_per_execution_group=2, parallel=("sequence",)))
class TwoByTwoEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


def test_group_width_reaches_manifest_under_the_builder_key() -> None:
    res = _resources(TwoByTwoEndpoint)
    # Both axes present and independent: 4 GPUs in the pod, 2 per request.
    assert res["max_gpu_count"] == 4
    assert res["max_gpus_per_execution_group"] == 2
    assert msgspec.json.decode(
        msgspec.json.encode(res))["max_gpus_per_execution_group"] == 2


def test_undeclared_group_width_is_absent_not_defaulted() -> None:
    # The whole existing fleet. Declaring the width axis must not start
    # emitting the degree axis with a defaulted legal value.
    assert "max_gpus_per_execution_group" not in _resources(SPEndpoint)
    assert "max_gpus_per_execution_group" not in _resources(PlainEndpoint)


def test_group_width_that_does_not_shard_is_refused() -> None:
    # 1 is the value a "default to today's behaviour" implementation would
    # have quietly accepted. It is outside the legal domain [2, ceiling], so
    # absence can never be confused with a declaration.
    with pytest.raises(ValueError, match="does not shard"):
        Resources(gpu=True, max_gpu_count=4,
                  max_gpus_per_execution_group=1, parallel=("sequence",))


def test_group_width_above_the_pod_ceiling_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds max_gpu_count"):
        Resources(gpu=True, max_gpu_count=4,
                  max_gpus_per_execution_group=8, parallel=("sequence",))
    with pytest.raises(ValueError, match="exceeds max_gpu_count"):
        Resources(gpu=True, max_gpus_per_execution_group=2,
                  parallel=("sequence",))


def test_group_width_without_a_mechanism_is_refused_as_inert() -> None:
    with pytest.raises(ValueError, match="inert"):
        Resources(gpu=True, max_gpu_count=4, max_gpus_per_execution_group=2)


def test_group_width_positive_control() -> None:
    # A validator that refused every group width would pass the table above
    # trivially. The legal shapes must construct.
    for width in (2, 3, 4):
        r = Resources(gpu=True, max_gpu_count=4,
                      max_gpus_per_execution_group=width,
                      parallel=("sequence",))
        assert r.max_gpus_per_execution_group == width
        assert r.gpu is True
