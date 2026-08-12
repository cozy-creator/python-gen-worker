"""pgw#1075 — a declared value this SDK's own vocabulary rejects is a REFUSAL.

``api.errors.ValidationError`` means, verbatim, *bad user input; do not retry*.
Inside the mint child the "user input" IS the declaration, so every property
the parent's retry policy reads off a refusal already holds for it: it is
deterministic, identical on the next card, and fixed by editing the
declaration and by nothing else. It nonetheless took the one exit that says
none of that — an unclassified crash.

**Measured on the rig, 2026-08-09.** A vehicle declaring ``lora_bucket=8``
(``RANK_BUCKETS = (16, 32, 64, 128)``) makes ``enable_lora_branches`` raise its
typed ``ValidationError`` — the refusal is CORRECT — and the rig reported
``mint-child crashed: the mint process exited 1`` with a truncated traceback
tail. Two things died at the process boundary: the class (crash, so RETRYABLE
— a second billed pod for a fact the first one settled) and the sentence
carrying the author's fix.

pgw#999's rule is that refusals carry a class. pgw#1062's is that a refusal's
message IS the authoring contract and is carried whole — that is why
``export_program`` hands torch's own ``register_fake`` guidance through
unedited. This is the same wrapper, one layer down.

**RED VERIFICATION.** Delete the ``except ValidationError`` branch from
``mint_child.main``:

    test_a_declaration_the_vocabulary_rejects_is_refused_not_crashed fails at
    `outcome.status = 'crashed', want 'refused'`, and (once that assert is
    removed) at `outcome.retryable is True` and at the message — the detail
    reads `the mint process exited 1: <traceback tail>` and the words
    `invalid lora rank bucket 8` are nowhere in it.

The child runs as a REAL subprocess on the REAL request file, so the
classification is proved across the boundary it was lost at rather than at the
raise site.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import msgspec
import pytest

from gen_worker import mint_child, mint_delegate
from gen_worker import mint_process as mp
from gen_worker.api.binding import ModelRef
from gen_worker.api.errors import IllegalCombination, ValidationError
from gen_worker.registry import CompileCell

pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
ENDPOINT_MODULE = "harness.tiny_diffusion_endpoint"
FUNCTION = "rig-generate"

#: Not in ``models.w8a8_lora.RANK_BUCKETS``. The rig's own reproduction.
BAD_BUCKET = 8


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from harness.tiny_diffusion import build_checkpoint

    return build_checkpoint(tmp_path_factory.mktemp("pgw1075") / "checkpoint")


def _request(checkpoint: Path, workdir: Path, *, bucket: int) -> mp.MintRequest:
    """The real parent chain, round-tripped through the file the child decodes.

    ``CompileCell`` is built directly, exactly as the rig's vehicle builds it:
    the ``@endpoint`` decorator screens ``lora_bucket`` at decoration time
    (``decorators.py``), so a bad bucket can only arrive from a path that is
    not the decorator — an operator mint request, or a rig vehicle. That is
    the shape this issue was found in.
    """
    from harness import tiny_diffusion_endpoint as ep

    workdir.mkdir(parents=True, exist_ok=True)
    cfg = CompileCell(
        shapes=(ep.PIXEL_SHAPE,), targets=("unet",), family=ep.FAMILY,
        regional=False, text_len=ep.TEXT_LEN, dynamic=(), lora_bucket=bucket,
        guidance_scales=(), text_lens=())
    pending = SimpleNamespace(
        family=ep.FAMILY, arm_token="arm1-pgw1075", cfg=cfg,
        target=workdir / "cell.tar.gz", mint_root=workdir)
    task = mint_delegate.MintTask(
        pending=pending, pipe=None, function=FUNCTION,
        modules=(ENDPOINT_MODULE,),
        slots={"pipeline": mp.MintSlot(
            ref=ModelRef(source="tensorhub", path="rig/tiny-diffusion",
                         tag="prod"),
            path=str(checkpoint))},
        device=-1, execution_lane="", configs={})
    request = mint_delegate.build_request(task, workdir=workdir)
    return msgspec.json.decode(msgspec.json.encode(request), type=mp.MintRequest)


def _child_env(request: mp.MintRequest) -> Dict[str, str]:
    env = dict(mp.child_env(request))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), str(HERE.parent / "src"), env.get("PYTHONPATH", "")])
    # Deterministic on any host, and the honest production shape: a mint child
    # that can see no card. The refusal under test fires long before anything
    # needs one.
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _run(checkpoint: Path, workdir: Path, *, bucket: int) -> Tuple[mp.MintOutcome, mp.MintRequest]:
    request = _request(checkpoint, workdir, bucket=bucket)
    outcome = asyncio.run(mp.run_mint(
        request, workdir=workdir, python=sys.executable,
        env=_child_env(request)))
    return outcome, request


def test_a_declaration_the_vocabulary_rejects_is_refused_not_crashed(
    checkpoint: Path, tmp_path: Path,
) -> None:
    """The whole issue, on the real entrypoint in a real child process."""
    outcome, request = _run(checkpoint, tmp_path / "w", bucket=BAD_BUCKET)

    assert outcome.status == mp.REFUSED, (
        f"{outcome.status}: {outcome.detail or outcome.stderr_tail}\n\n"
        "A ValidationError is the SDK saying 'bad declared input; do not "
        "retry'. Reporting it as an unclassified crash makes it RETRYABLE, "
        "which on a pod is a second billed mint for a fact the first attempt "
        "already settled.")
    assert outcome.exit_code == mp.EXIT_REFUSED
    assert outcome.retryable is False, (
        "a declaration defect is identical on the next card and the one after")

    detail = outcome.detail or ""
    assert "invalid lora rank bucket 8" in detail, (
        f"the author's fix must survive the boundary whole: {detail!r}")
    assert "16, 32, 64, 128" in detail, (
        f"including the vocabulary it was measured against: {detail!r}")
    assert "ValidationError" in detail, (
        f"and the type, so the refusal is groupable: {detail!r}")

    report = outcome.report
    assert report is not None and report.status == "refused", (
        "the child's own report must say refused, not failed — the rig and "
        "the hub both read the report before the exit code")
    assert report.phase, "the phase it refused in is named (th#1322)"
    assert not Path(request.target).exists(), "nothing may be sealed"


def test_the_classification_discriminates(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE CONTROL, on the real ``main``: only a ValidationError is refused.

    A classification that widens is worse than none — it would report a
    genuine BUG as a deterministic refusal and strand a mint a retry could
    make. So the same entrypoint is driven twice with only the exception type
    changed, and the two must land on different exits.

    In-process rather than through a child: what is under test is ``main``'s
    exception table, and re-running a whole CPU mint to change one raise
    would buy 4 minutes of nothing. The BOUNDARY is proved above.
    """
    request = _request(checkpoint, tmp_path / "w", bucket=BAD_BUCKET)
    request_file = tmp_path / "request.json"
    request_file.write_bytes(msgspec.json.encode(request))

    def _run_main(exc: BaseException) -> Tuple[int, mp.MintReport]:
        def _raise(_req: mp.MintRequest) -> mp.MintReport:
            raise exc

        monkeypatch.setattr(mint_child, "mint", _raise)
        code = mint_child.main([str(request_file)])
        return code, msgspec.json.decode(
            Path(request.report).read_bytes(), type=mp.MintReport)

    code, report = _run_main(ValidationError("invalid lora rank bucket 8"))
    assert code == mp.EXIT_REFUSED and report.status == "refused", (
        f"exit={code} status={report.status!r}: a declared value the "
        "vocabulary rejects is terminal, and the exit code is what the "
        "parent's retry policy reads")
    assert "invalid lora rank bucket 8" in report.detail

    code, report = _run_main(RuntimeError("a real bug, from real code"))
    assert code == 1 and report.status == "failed", (
        f"exit={code} status={report.status!r}: a RuntimeError is NOT a "
        "declaration refusal. Classifying one would strand a mint that a "
        "retry could make — the mirror of the defect this issue closes")


def test_the_raise_site_accepts_every_declared_bucket() -> None:
    """The other side of the control, where the sentence comes from: a LEGAL
    bucket must not raise at all, or the refusal above is just a broken
    vocabulary check being reported politely."""
    torch = pytest.importorskip("torch")
    from gen_worker.models import w8a8_lora

    for bucket in w8a8_lora.RANK_BUCKETS:
        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        w8a8_lora.enable_lora_branches(model, bucket)
        assert w8a8_lora.branch_bucket(model) == bucket

    with pytest.raises(ValidationError) as exc:
        w8a8_lora.enable_lora_branches(
            torch.nn.Sequential(torch.nn.Linear(8, 8)), BAD_BUCKET)
    assert f"invalid lora rank bucket {BAD_BUCKET}" in str(exc.value)


def test_the_refusal_type_is_the_sdks_own_bad_input_type() -> None:
    """Why ``ValidationError`` and not a new exception: the SDK already owns
    exactly this meaning, and the mint child is a second reader of it rather
    than a second definition. ``IllegalCombination`` — an endpoint declaring a
    payload combination outside its contract — is a subclass, so it is
    classified by the same branch without naming it."""
    assert issubclass(IllegalCombination, ValidationError)
    assert ValidationError.__doc__ and "do not retry" in ValidationError.__doc__

    refusal = mint_child._declaration_refusal(
        ValidationError("invalid lora rank bucket 8 (valid: (16, 32, 64, 128))"))
    assert isinstance(refusal, mint_child.MintChildRefused), (
        "it must land on the type the child already exits EXIT_REFUSED for — "
        "one refusal path, not two")
    assert "invalid lora rank bucket 8" in str(refusal)
    assert "ValidationError" in str(refusal)
