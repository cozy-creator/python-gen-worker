import json
import os
import subprocess
from typing import Iterator, Optional

import msgspec

from gen_worker import ActionContext, ResourceRequirements, worker_function


class CodexExecInput(msgspec.Struct):
    prompt: str
    model: Optional[str] = None
    sandbox: str = "read-only"


class CodexEventDelta(msgspec.Struct):
    # Raw JSONL line from `codex exec --json`.
    jsonl: str


@worker_function(ResourceRequirements())
def codex_exec(ctx: ActionContext, payload: CodexExecInput) -> Iterator[CodexEventDelta]:
    """
    Run Codex in headless mode and stream JSONL events as incremental deltas.

    Auth is provided via env at runtime:
      - CODEX_API_KEY (preferred for `codex exec` headless runs)
    """
    api_key = (os.getenv("CODEX_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("missing CODEX_API_KEY (must be injected at runtime)")

    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise ValueError("prompt cannot be empty")

    cmd = [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        payload.sandbox,
        prompt,
    ]
    if payload.model:
        cmd[2:2] = ["--model", payload.model]

    # Keep stderr separate: `--json` streams events to stdout, while progress/debug
    # may still go to stderr.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_lines: list[str] = []
    try:
        for line in proc.stdout:
            if ctx.is_canceled():
                proc.kill()
                raise InterruptedError("Task cancelled")
            s = line.rstrip("\n")
            if not s:
                continue
            # Best-effort validation that it's JSONL (keeps output sane).
            try:
                json.loads(s)
            except Exception:
                # Still stream it; callers can debug.
                pass
            yield CodexEventDelta(jsonl=s)
    finally:
        try:
            # Drain stderr for error reporting.
            stderr_lines.extend(proc.stderr.read().splitlines())
        except Exception:
            pass

    rc = proc.wait()
    if rc != 0:
        tail = "\n".join(stderr_lines[-50:])
        raise RuntimeError(f"codex exec failed with exit code {rc}\n{tail}")

