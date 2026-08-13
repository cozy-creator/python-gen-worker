"""pgw#984 + pgw#985 — the two mint recipes, held to one contract.

Both defects were MEASURED by the pgw#978 micro-mint rig inside its first hour,
at ~20 s a cycle, by running the same machinery twice with only
``MintRequest.recipe`` changed. They are the same kind of finding: the two
recipes had drifted into two different answers to one question.

**pgw#984 — the AOT recipe never entered the endpoint.** A measured AOT mint's
phase table was ``{'load': 4.4, 'trace_graph': 8.1, 'seal_publish': 0.9,
'finalize': 0.0}`` — no ``warmup_forward`` row at all, because
``torch.export`` traces the declared modules directly and the handler is never
called. So a green AOT mint proved the family's graphs export and said nothing
about whether the forward those graphs serve can run. pgw#969's crash
(``ctx.slots["pipeline"]``, 0.0 s into ``warmup_forward``, twice on L40S pods)
is *unreachable* on that recipe: it would have sealed and published a cell for
an endpoint whose first real request dies.

**pgw#985 — a deterministic arm decline was a crash.** On the same box, the
same pipeline, the (then-existing) dynamo recipe raised::

    RuntimeError: no compile targets resolved on TinyDiffusionPipeline

...out of the cold arm — about a pipeline whose ``.unet``
``has_compile_target`` had resolved one frame earlier. Two things were wrong
and each is a class:

1. the arm answered TWO facts with ONE sentence. The pipeline
   owned its declared target; what actually declined was ``apply``, because
   the process had no CUDA. Two computations of one fact disagree eventually
   (§1.29, th#1616) — these two had, and the survivor lied.
2. A bare ``RuntimeError`` exits 1, which classifies ``CRASHED`` — the
   retryable class. On a pod that is a second billed mint for a condition the
   first attempt had already settled. The AOT recipe types the identical
   condition ``PreflightRefused`` -> ``EXIT_REFUSED`` -> terminal.

pgw#1010 deleted the dynamo recipe's ARTIFACT (it had no consumer), so the
child now mints one kind. Both findings survive intact and are asserted where
they still live: the arm that answers with one sentence is
``compile_cache.arm_jit_intake`` — the JIT INTAKE arm a serving pod takes
instead — and the terminal classification is the child's, on the one recipe it
has.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import msgspec
import pytest

from gen_worker import child_preflight
from gen_worker import child_contract
from gen_worker import compile_cache as cc
from gen_worker import mint_child, mint_delegate
from gen_worker import mint_process as mp
from gen_worker.api.binding import ModelRef
from gen_worker.registry import CompileCell

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
ENDPOINT_MODULE = "harness.tiny_diffusion_endpoint"
FUNCTION = "rig-generate"
GIB = 1 << 30


# ---------------------------------------------------------------------------
# 1. ONE target authority, ONE precondition authority
# ---------------------------------------------------------------------------


class _Unet:
    def forward(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None


class _Pipe:
    def __init__(self) -> None:
        self.unet = _Unet()


def _cfg(*targets: str) -> Any:
    return SimpleNamespace(
        targets=targets, shapes=((64, 64),), family="pgw985", regional=False,
        dynamic=(), lora_bucket=0, guidance_scales=(), text_lens=(),
        text_len=77)


@pytest.fixture
def cardless(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that can see no card — the rig's own condition, forced so the
    row measures the same thing on a laptop with a card and on CI without."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_the_two_target_resolutions_are_one_function() -> None:
    """RED at HEAD: ``has_compile_target`` scanned ``cfg.targets`` and
    ``apply`` scanned them AGAIN, in its own loop, and ``arm_jit_intake``
    called both and reported whichever failed under the first one's sentence.
    """
    pipe, cfg = _Pipe(), _cfg("unet")
    rows = cc.resolve_targets(pipe, cfg)
    assert [name for name, *_ in rows] == ["unet"]
    owner, attr = rows[0][1], rows[0][2]
    assert owner is pipe.unet and attr == "forward"
    assert cc.has_compile_target(pipe, cfg) is True
    assert cc.resolve_targets(pipe, _cfg("no_such_module")) == []
    assert cc.has_compile_target(pipe, _cfg("no_such_module")) is False


def test_every_target_reader_moves_with_the_one_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proof that it is ONE relation and not three that happen to agree:
    move ``resolve_targets`` and all three readers move with it."""
    pipe, cfg = _Pipe(), _cfg("unet")
    monkeypatch.setattr(cc, "resolve_targets", lambda *a, **k: [])
    assert cc.has_compile_target(pipe, cfg) is False
    assert cc.apply(pipe, cfg, cache_ready=False, allow_cold=True) is False
    with pytest.raises(cc.CompileArmRefused) as exc:
        cc.arm_jit_intake(pipe, cfg)
    assert "no compile target resolves" in str(exc.value)


def test_a_cardless_process_is_named_as_the_environment_fact_it_is(
    cardless: None,
) -> None:
    """RED at HEAD: this raised ``no compile targets resolved on _Pipe`` — the
    WIRING sentence — about a pipeline that owns its declared target. The
    mint child's operator then had a message that ruled out the only thing it
    was not.
    """
    pipe, cfg = _Pipe(), _cfg("unet")
    assert cc.has_compile_target(pipe, cfg) is True
    block = cc.arming_block(pipe, cfg, cache_ready=False, allow_cold=True)
    assert "CUDA" in block

    with pytest.raises(cc.CompileArmRefused) as exc:
        cc.arm_jit_intake(pipe, cfg)
    text = str(exc.value)
    assert "['unet']" in text and block in text and "pgw985" in text
    assert "no compile target resolves" not in text, (
        f"the wiring sentence must not stand in for the environment fact: {text}")


def test_the_refusal_leaves_the_process_global_cache_dir_where_it_was(
    cardless: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gw#608's invariant, now true BY CONSTRUCTION: pgw#1010 deleted the
    capture dir the arm used to re-point the interpreter at, so a decline (or
    a success) cannot strand the cache dir anywhere."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "seeded"))
    with pytest.raises(cc.CompileArmRefused):
        cc.arm_jit_intake(_Pipe(), _cfg("unet"))
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "seeded")
    assert "capture" not in inspect.signature(cc.arm_jit_intake).parameters


# ---------------------------------------------------------------------------
# 2. The child's request, built through the REAL parent chain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from harness.tiny_diffusion import build_checkpoint

    return build_checkpoint(tmp_path_factory.mktemp("microrig") / "checkpoint")


def _request(
    checkpoint: Path, workdir: Path, *,
    targets: Tuple[str, ...] = ("unet",),
) -> mp.MintRequest:
    from harness import tiny_diffusion_endpoint as ep

    workdir.mkdir(parents=True, exist_ok=True)
    cfg = CompileCell(
        shapes=(ep.PIXEL_SHAPE,), targets=targets, family=ep.FAMILY,
        regional=False, text_len=ep.TEXT_LEN, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=())
    pending = SimpleNamespace(
        family=ep.FAMILY, arm_token="arm1-recipe-parity", cfg=cfg,
        target=workdir / "cell.tar.gz", mint_root=workdir)
    task = mint_delegate.MintTask(
        pending=pending, pipe=None, function=FUNCTION,
        modules=(ENDPOINT_MODULE,),
        slots={"pipeline": child_contract.MintSlot(
            ref=ModelRef(source="tensorhub", path="rig/tiny-diffusion",
                         tag="prod"),
            path=str(checkpoint))},
        device=-1, execution_lane="", configs={})
    request = mint_delegate.build_request(task, workdir=workdir)
    # The boundary IS a file: round-trip it exactly as the child decodes it.
    return msgspec.json.decode(msgspec.json.encode(request), type=mp.MintRequest)


def _child_env(request: mp.MintRequest, *, cardless: bool) -> Dict[str, str]:
    env = dict(mp.child_env(request))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), str(HERE.parent / "src"), env.get("PYTHONPATH", "")])
    if cardless:
        # Deterministic on any host, and the honest production shape of the
        # condition: a mint child that can see no card.
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


# ---------------------------------------------------------------------------
# 3. pgw#985 — the dynamo recipe's deterministic refusal is TERMINAL and TYPED
# ---------------------------------------------------------------------------


def test_the_child_refuses_a_deterministic_environment_decline(
    checkpoint: Path, tmp_path: Path,
) -> None:
    """The real entrypoint, in a real child, on the real request file.

    RED at HEAD: exit 1, ``CRASHED``, detail ``RuntimeError: no compile
    targets resolved on TinyDiffusionPipeline`` — the retryable class, for a
    fact no retry can change.
    """
    request = _request(checkpoint, tmp_path / "w")
    outcome = asyncio.run(mp.run_mint(
        request, workdir=tmp_path / "w", python=sys.executable,
        env=_child_env(request, cardless=True)))

    assert outcome.status == mp.REFUSED, (
        f"{outcome.status}: {outcome.detail or outcome.stderr_tail}")
    assert outcome.exit_code == mp.EXIT_REFUSED
    assert outcome.retryable is False, "a deterministic refusal buys no second pod"
    report = outcome.report
    assert report is not None and report.status == "refused"
    for fact in ("sm",):
        assert fact in report.detail, f"{fact!r} missing from: {report.detail}"
    assert not Path(request.target).exists()
    # And the phase it died in survives into the report. This was
    # ALWAYS "" — `_close_phases()` closes the open phase, and the field was
    # read after it in the same call.
    # The phase it died in survives into the report. WHICH phase is
    # the child's business — a cardless box refuses inside the export's
    # packaging step — so the assertion is that a phase is NAMED, not which.
    assert report.phase, report.phase


def test_a_missing_target_is_refused_terminally_not_crashed(
    checkpoint: Path, tmp_path: Path,
) -> None:
    """One condition, one classification. This used to be asserted across both
    recipes; pgw#1010 left one, and the vocabulary it is held to is the same."""
    workdir = tmp_path / "aot"
    request = _request(checkpoint, workdir, targets=("no_such_module",))
    outcome = asyncio.run(mp.run_mint(
        request, workdir=workdir, python=sys.executable,
        env=_child_env(request, cardless=True)))
    assert outcome.status == mp.REFUSED, (
        f"{outcome.status}: {outcome.detail or outcome.stderr_tail}")
    assert outcome.exit_code == mp.EXIT_REFUSED
    assert outcome.retryable is False
    assert "no compile target resolved" in (outcome.detail or "")


# ---------------------------------------------------------------------------
# 4. pgw#984 — the AOT recipe runs the endpoint's own warm plan before it seals
# ---------------------------------------------------------------------------


@pytest.fixture
def aot_without_the_export(
    monkeypatch: pytest.MonkeyPatch,
) -> List[Dict[str, Any]]:
    """Everything the AOT recipe does up to the export, and nothing after.

    The export + AOTInductor compile is the rig's job (and a LOCAL-ONLY row);
    what is under test here is the ORDER — whether the endpoint's own forward
    has run by the time a cell can be sealed.
    """
    from harness import tiny_diffusion_endpoint as ep

    ep.reset()
    mint_child._PHASE_SPANS.clear()
    monkeypatch.setattr(mint_child, "_PHASE_OPEN", ("", 0.0), raising=False)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(cc, "cxx_toolchain_present", lambda: True)

    exported: List[Dict[str, Any]] = []

    def _stub(request: Any, pipe: Any, cfg: Any, target: Path, **kwargs: Any):
        exported.append({"family": cfg.family, "target": target})
        return mp.MintReport(
            status="minted", artifact=str(target), cell_key=request.arm_token,
            phase="finalize", phases=mint_child._close_phases())

    monkeypatch.setattr(mint_child, "_mint_aot", _stub)
    return exported


def test_the_aot_recipe_runs_the_endpoints_own_forward_before_it_exports(
    checkpoint: Path, tmp_path: Path, aot_without_the_export: List[Any],
) -> None:
    """RED at HEAD: ``exported`` was populated and ``RESOLVED_REFS`` was empty
    — the recipe reached the seal without ever entering the handler, and the
    phase table it published had no ``warmup_forward`` row to say so.
    """
    from harness import tiny_diffusion_endpoint as ep

    report = mint_child.mint(_request(checkpoint, tmp_path / "w"))

    assert ep.RESOLVED_REFS == ["rig/tiny-diffusion"], (
        "the endpoint's own handler must have run, through ctx.slots — "
        f"got {ep.RESOLVED_REFS}")
    assert aot_without_the_export, "the export must still happen, after the proof"
    assert "warmup_forward" in report.phases, (
        f"a reader of the phase table must be able to tell: {report.phases}")


def test_an_aot_mint_cannot_seal_for_a_handler_that_cannot_run(
    checkpoint: Path, tmp_path: Path, aot_without_the_export: List[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#969's crash class, made reachable on the recipe it was invisible on.

    RED at HEAD: this minted. ``torch.export`` does not care that the handler
    raises, so the cell sealed, published, and every pod that adopted it hit
    the ValueError on its first real request instead.
    """
    from harness import tiny_diffusion_endpoint as ep

    def _broken(self: Any, ctx: Any, data: Any) -> Any:
        raise ValueError(
            "slot 'pipeline': no resolved model ref for this request")

    monkeypatch.setattr(ep.MicroRigEndpoint, "rig_generate", _broken)

    with pytest.raises(child_preflight.PreflightRefused) as exc:
        mint_child.mint(_request(checkpoint, tmp_path / "w"))

    text = str(exc.value)
    assert "warm plan does not run" in text, text
    # The recipe left the identity line with the recipe axis — the
    # child mints one kind, so a name for it on every refusal is noise.
    assert "microrig" in text and FUNCTION in text, text
    assert "no resolved model ref" in text, text
    assert not aot_without_the_export, (
        "nothing may be exported once the endpoint's forward has failed")


def test_a_resource_shortfall_in_the_warm_plan_is_still_retryable(
    checkpoint: Path, tmp_path: Path, aot_without_the_export: List[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line the refusal must not cross. A named refusal is terminal, so
    classifying an OOM as one would strand a mint the next attempt could make
    at a narrower width."""
    from harness import tiny_diffusion_endpoint as ep

    def _oom(self: Any, ctx: Any, data: Any) -> Any:
        raise MemoryError("host RAM shortfall in the warm forward")

    monkeypatch.setattr(ep.MicroRigEndpoint, "rig_generate", _oom)

    with pytest.raises(MemoryError):
        mint_child.mint(_request(checkpoint, tmp_path / "w"))
    assert not aot_without_the_export
