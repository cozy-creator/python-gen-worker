"""gw#555: self-minted local compile cells — store key discipline, mismatch
re-mint, save/load round-trip, and the structural no-publish trust boundary.

The torch.compile capture itself needs a GPU + toolchain and is proven live
on a pod (tracker gw#555); these tests exercise everything around it through
the real pack/verify/seed machinery."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from gen_worker import compile_cache as cc
from gen_worker import local_cells as lc


@pytest.fixture(autouse=True)
def _restore_capture_env(monkeypatch):
    """capture_env mutates these in-process; register them with monkeypatch
    so each test leaves the process env as it found it."""
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.setenv(key, os.environ.get(key, ""))


class _Cfg:
    """Stand-in for api.decorators.Compile (duck-typed in local_cells)."""

    def __init__(
        self, family="fam", shapes=((64, 64),), targets=("transformer",),
        regional=False, guidance_scales=(),
    ):
        self.family = family
        self.shapes = tuple(tuple(s) for s in shapes)
        self.targets = tuple(targets)
        self.regional = regional
        self.guidance_scales = tuple(guidance_scales)


class _Pipe:
    _cozy_low_vram_mode = "off"


def _make_cell(path: Path, **overrides) -> Path:
    """A real packed artifact whose metadata matches THIS runtime (so
    verify() passes), with optional key overrides to force mismatches."""
    meta = cc.artifact_metadata(
        family="fam", shapes=[(64, 64)], targets=["transformer"],
        low_vram_mode="off",
    )
    meta.update(overrides)
    capture = path.parent / f".capture-{path.name}"
    (capture / "inductor").mkdir(parents=True, exist_ok=True)
    (capture / "inductor" / "kernel.bin").write_bytes(b"fake-kernel")
    (capture / "triton").mkdir(parents=True, exist_ok=True)
    (capture / "triton" / "launcher.json").write_bytes(b"{}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return cc.pack(capture, path, meta)


# ---------------------------------------------------------------------------
# store paths + key
# ---------------------------------------------------------------------------


def test_store_root_env_and_default(monkeypatch):
    monkeypatch.delenv(lc.ENV_STORE_DIR, raising=False)
    assert lc.store_root() == Path.home() / ".cache" / "cozy" / "compile-cells"
    monkeypatch.setenv(lc.ENV_STORE_DIR, "/tmp/cells-here")
    assert lc.store_root() == Path("/tmp/cells-here")


def test_cell_path_keys_on_family_and_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv(lc.ENV_STORE_DIR, str(tmp_path))
    p = lc.cell_path("ltx-2.3")
    key = cc.runtime_key()
    assert p == tmp_path / "ltx-2.3" / f"{cc.flavor_label(key['sku'], key['torch'])}.tar.gz"
    # weight lane is part of the key: different lane, different cell file
    assert lc.cell_path("ltx-2.3", "fp8-hooks") != p


# ---------------------------------------------------------------------------
# verdict: production verify() semantics verbatim + drift parity
# ---------------------------------------------------------------------------


def test_store_verdict_hit(tmp_path):
    cell = _make_cell(tmp_path / "cell.tar.gz")
    assert lc.store_verdict(cell, "fam", _Pipe(), _Cfg()) == ""


@pytest.mark.parametrize(
    "overrides, expect",
    [
        ({"torch": "0.0.0+nope"}, "torch"),
        ({"gen_worker": "0.0.0-not-this"}, "gen_worker"),
        ({"sku": "not-this-gpu"}, "sku"),
        ({"format": 99}, "format"),
        ({"weight_lane": "fp8-hooks"}, "weight_lane"),   # lane drift, symmetric
        ({"compile_mode": "regional"}, "compile_mode"),
        ({"low_vram_mode": "vae_only"}, "low_vram_mode"),
    ],
)
def test_store_verdict_mismatches(tmp_path, overrides, expect):
    cell = _make_cell(tmp_path / "cell.tar.gz", **overrides)
    assert expect in lc.store_verdict(cell, "fam", _Pipe(), _Cfg())


def test_store_verdict_family_mismatch(tmp_path):
    cell = _make_cell(tmp_path / "cell.tar.gz")
    assert "family" in lc.store_verdict(cell, "other-family", _Pipe(), _Cfg())


def test_store_verdict_garbage_artifact(tmp_path):
    bad = tmp_path / "cell.tar.gz"
    bad.write_bytes(b"not a tarball")
    assert "unreadable" in lc.store_verdict(bad, "fam", _Pipe(), _Cfg())


# ---------------------------------------------------------------------------
# mint: save/load round-trip through the real pack/unpack/seed machinery
# ---------------------------------------------------------------------------


def _fake_compile_and_warm(pipe, cfg, **kw):
    """Stand-in for the GPU-only capture: write files where the ACTIVE
    TORCHINDUCTOR/TRITON cache env points (what inductor would do)."""
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"], "graph.so").write_bytes(b"\x7fELF-ish")
    Path(os.environ["TRITON_CACHE_DIR"], "kern.cubin").write_bytes(b"cubin")


def test_mint_saves_atomically_and_roundtrips(monkeypatch, tmp_path):
    monkeypatch.setenv(lc.ENV_STORE_DIR, str(tmp_path))
    monkeypatch.setattr(lc, "_compile_and_warm", _fake_compile_and_warm)
    pipe, cfg = _Pipe(), _Cfg()
    target = lc.cell_path("fam")

    saved = lc._mint(pipe, cfg, target, "fam")

    assert saved == target and target.exists()
    assert not list(target.parent.glob("*.part"))  # atomic: no partials left
    # the saved cell is adoptable by this same runtime (the re-boot path)
    assert lc.store_verdict(target, "fam", pipe, cfg) == ""
    # and unpacks through the production consumer machinery
    meta = cc.unpack(target, tmp_path / "seed-root")
    assert meta["family"] == "fam"
    assert meta["source_ref"] == "local-mint"
    assert (tmp_path / "seed-root" / "inductor" / "graph.so").exists()
    assert cc.verify(meta, family="fam") == ""


def test_mint_refuses_empty_capture(monkeypatch, tmp_path):
    monkeypatch.setenv(lc.ENV_STORE_DIR, str(tmp_path))
    monkeypatch.setattr(lc, "_compile_and_warm", lambda pipe, cfg, **kw: None)
    with pytest.raises(RuntimeError, match="captured nothing"):
        lc._mint(_Pipe(), _Cfg(), lc.cell_path("fam"), "fam")
    assert not lc.cell_path("fam").exists()  # nothing saved on failure


# ---------------------------------------------------------------------------
# enable_compiled orchestration: hit adopts, mismatch re-mints, miss mints
# ---------------------------------------------------------------------------


@pytest.fixture()
def _local_env(monkeypatch, tmp_path):
    monkeypatch.setenv(lc.ENV_STORE_DIR, str(tmp_path / "store"))
    monkeypatch.delenv(cc.ENV_CACHE_PATH, raising=False)
    monkeypatch.delenv(cc.ENV_CACHE_URL, raising=False)
    monkeypatch.delenv(cc.ENV_ALLOW_COLD, raising=False)
    monkeypatch.setattr(lc, "_cuda_ready", lambda: True)
    # keep the delivered-artifact leg deterministic on GPU-less CI
    from gen_worker.models import provision

    monkeypatch.setattr(provision, "enable_compiled", lambda *a, **k: False)
    return tmp_path / "store"


def test_stored_cell_adopts_through_production_path(_local_env, monkeypatch):
    pipe, cfg = _Pipe(), _Cfg()
    _make_cell(lc.cell_path("fam"))
    calls = {}

    def fake_enable(p, c, cache_dir=None, artifact=None):
        calls["artifact"] = artifact
        return True

    monkeypatch.setattr(cc, "enable", fake_enable)
    minted = []
    monkeypatch.setattr(lc, "_mint", lambda *a, **k: minted.append(a))

    assert lc.enable_compiled(pipe, cfg) is True
    assert calls["artifact"] == lc.cell_path("fam")  # delivered-cell code path
    assert minted == []                              # no re-mint on a hit


def test_key_mismatch_remints_and_replaces(_local_env, monkeypatch):
    pipe, cfg = _Pipe(), _Cfg()
    target = lc.cell_path("fam")
    _make_cell(target, gen_worker="0.0.0-stale")     # stale-key cell in store
    stale_bytes = target.read_bytes()
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)

    def fake_mint(p, c, tgt, family):
        _make_cell(tgt)                              # fresh, matching cell
        return tgt

    monkeypatch.setattr(lc, "_mint", fake_mint)
    enables = []

    def fake_enable(p, c, cache_dir=None, artifact=None):
        enables.append(artifact)
        return True

    monkeypatch.setattr(cc, "enable", fake_enable)

    assert lc.enable_compiled(pipe, cfg) is True
    assert target.read_bytes() != stale_bytes        # stale cell replaced
    assert lc.store_verdict(target, "fam", pipe, cfg) == ""
    assert enables == [target]                       # armed from the NEW cell


def test_miss_mints_once(_local_env, monkeypatch):
    pipe, cfg = _Pipe(), _Cfg()
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(lc, "_compile_and_warm", _fake_compile_and_warm)
    monkeypatch.setattr(cc, "enable", lambda *a, **k: True)

    assert lc.enable_compiled(pipe, cfg) is True
    assert lc.cell_path("fam").exists()


def test_no_toolchain_stays_eager(_local_env, monkeypatch, capsys):
    monkeypatch.setattr(cc, "toolchain_present", lambda: False)
    assert lc.enable_compiled(_Pipe(), _Cfg()) is False
    assert not lc.cell_path("fam").exists()
    assert "no C compiler" in capsys.readouterr().err


def test_mint_failure_serves_eager(_local_env, monkeypatch):
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)

    def boom(pipe, cfg, **kw):
        raise RuntimeError("compile exploded")

    monkeypatch.setattr(lc, "_compile_and_warm", boom)
    assert lc.enable_compiled(_Pipe(), _Cfg()) is False
    assert not lc.cell_path("fam").exists()


def test_no_family_no_mint(_local_env):
    assert lc.enable_compiled(_Pipe(), _Cfg(family="")) is False


# ---------------------------------------------------------------------------
# gw#564: w8a8 fail-closed x local mint. Production's no-delivered-cell
# refusal (CompiledLaneUnavailableError) must fall through to the LOCAL
# store/mint path — found live on a 4090 where the raise aborted the mint —
# and every exit that cannot produce a cell keeps the refusal TYPED.
# ---------------------------------------------------------------------------


class _W8a8Pipe(_Pipe):
    _cozy_weight_lane = "w8a8"


@pytest.fixture()
def _w8a8_env(_local_env, monkeypatch):
    """The live-bug trigger: the delivered-artifact leg RAISES the w8a8
    fail-closed refusal (no cell anywhere) instead of returning False."""
    from gen_worker.models import provision

    def refuse(*a, **k):
        raise cc.CompiledLaneUnavailableError("no delivered w8a8 cell")

    monkeypatch.setattr(provision, "enable_compiled", refuse)
    return _local_env


def test_w8a8_delivered_refusal_falls_through_to_mint(_w8a8_env, monkeypatch):
    pipe, cfg = _W8a8Pipe(), _Cfg()
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(lc, "_compile_and_warm", _fake_compile_and_warm)
    monkeypatch.setattr(cc, "enable", lambda *a, **k: True)

    assert lc.enable_compiled(pipe, cfg) is True
    target = lc.cell_path("fam", "w8a8")
    assert target.exists()
    assert "-w8a8" in target.name  # epilogue/rowwise lanes share the token


def test_w8a8_no_toolchain_refusal_stays_typed(_w8a8_env, monkeypatch):
    monkeypatch.setattr(cc, "toolchain_present", lambda: False)
    with pytest.raises(cc.CompiledLaneUnavailableError, match="C compiler"):
        lc.enable_compiled(_W8a8Pipe(), _Cfg())


def test_w8a8_mint_failure_refusal_stays_typed(_w8a8_env, monkeypatch):
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)

    def boom(pipe, cfg, **kw):
        raise RuntimeError("compile exploded")

    monkeypatch.setattr(lc, "_compile_and_warm", boom)
    with pytest.raises(cc.CompiledLaneUnavailableError, match="mint failed"):
        lc.enable_compiled(_W8a8Pipe(), _Cfg())
    assert not lc.cell_path("fam", "w8a8").exists()


def test_w8a8_stored_cell_adopts_after_delivered_refusal(_w8a8_env, monkeypatch):
    pipe, cfg = _W8a8Pipe(), _Cfg()
    _make_cell(lc.cell_path("fam", "w8a8"), weight_lane="w8a8")
    calls = {}

    def fake_enable(p, c, cache_dir=None, artifact=None):
        calls["artifact"] = artifact
        return True

    monkeypatch.setattr(cc, "enable", fake_enable)
    minted = []
    monkeypatch.setattr(lc, "_mint", lambda *a, **k: minted.append(a))

    assert lc.enable_compiled(pipe, cfg) is True
    assert calls["artifact"] == lc.cell_path("fam", "w8a8")
    assert minted == []


def test_plain_lane_miss_policy_unchanged(_local_env, monkeypatch):
    """Plain lanes keep the never-raise eager fallback."""
    monkeypatch.setattr(cc, "toolchain_present", lambda: False)
    assert lc.enable_compiled(_Pipe(), _Cfg()) is False


# ---------------------------------------------------------------------------
# trust boundary: no publish path exists, structurally
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORTS = {
    "gen_worker.s3_transfer", "gen_worker.presigned_upload",
    "gen_worker._upload_transport", "gen_worker.media_transfer",
    "gen_worker.net", "gen_worker.callout", "gen_worker.transport",
    "urllib", "urllib.request", "requests", "httpx", "boto3", "aiohttp",
}


def test_local_cells_has_no_publish_path():
    """local cells are user-generated executable artifacts (supply-chain
    boundary): the store module must have no network/upload machinery."""
    src = Path(lc.__file__).read_text()
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            base = f"gen_worker.{mod.lstrip('.')}" if node.level else mod
            imported.add(base)
            imported.update(f"{base}.{a.name}" for a in node.names)
    hits = {
        i for i in imported
        if i in _FORBIDDEN_IMPORTS or any(i.startswith(f + ".") for f in _FORBIDDEN_IMPORTS)
    }
    assert not hits, f"local_cells must never grow a publish path: {hits}"
    idents = {
        n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)
    } | {
        n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    } | {
        n.name.lower() for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for word in ("upload", "publish", "presign", "put_object", "post"):
        bad = {i for i in idents if word in i}
        assert not bad, f"local_cells code references {bad}"


def test_production_executor_never_reaches_local_cells():
    """The mint entry is the local CLI only: the production executor and
    lifecycle must not reference local_cells (fetch-only stays fetch-only)."""
    root = Path(lc.__file__).parent
    for name in ("executor.py", "lifecycle.py", "worker.py", "entrypoint.py"):
        assert "local_cells" not in (root / name).read_text(), name


def test_local_cli_is_the_only_mint_caller():
    root = Path(lc.__file__).parent
    callers = [
        p for p in root.rglob("*.py")
        if p.name != "local_cells.py" and "local_cells" in p.read_text()
    ]
    assert {p.relative_to(root).as_posix() for p in callers} == {"cli/run.py"}
