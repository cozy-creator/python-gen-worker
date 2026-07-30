#!/usr/bin/env python3
"""pgw#788 publish gate: the shipped wheel BOOTS on a worker with no torch.

`entrypoint.py` calls `env_seal.establish()` on every boot regardless of
`accelerator`. From 0.70.3 through 0.78.0 that chain bare-imported torch at
three unguarded call sites, so every torchless CPU endpoint died at
`phase=env_seal` before advertising a single function — and no gate noticed,
because every environment we test in has torch.

This runs in the wheel-contract venv, which installs the built wheel with NO
extras and is therefore genuinely torch-free. That is the point: the in-suite
test (`tests/test_torchless_boot_pgw788.py`) simulates the absence with a
`sys.meta_path` finder, and a simulation can drift from a real environment.

Asserted:
  A. this interpreter really has no torch — otherwise the gate is vacuous;
  B. `gen_worker` resolves from site-packages, not a checkout;
  C. the public entrypoint module IMPORTS;
  D. `env_seal.establish()` completes and seals `torch: "absent"`, and the ISA
     clamp and guard posture no-op instead of raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FAILURES: list[str] = []


def _check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        _FAILURES.append(f"{name}: {detail}")


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        _check("torchless-env", True, f"torch is genuinely absent ({exc})")
    else:
        _check(
            "torchless-env", False,
            "torch IS importable here — this gate proves nothing; it must run "
            "in the no-extras wheel venv",
        )
        return _verdict()

    import gen_worker

    loc = Path(gen_worker.__file__).resolve()
    repo_src = (Path(__file__).resolve().parent.parent / "src").resolve()
    _check("installed", repo_src not in loc.parents,
           f"gen_worker resolves from {loc}")

    try:
        import gen_worker.entrypoint  # noqa: F401
        _check("entrypoint", True, "the public entrypoint imports without torch")
    except Exception as exc:  # noqa: BLE001 - the whole point of the gate
        _check("entrypoint", False, f"import failed: {exc!r}")
        return _verdict()

    from gen_worker import env_seal, guard_closure, host_isa, torch_capability

    try:
        seal = env_seal.establish()
    except Exception as exc:  # noqa: BLE001
        _check("env_seal", False,
               f"establish() raised on a torchless worker: {exc!r} — this is "
               "the pgw#788 regression, the pod would die at phase=env_seal")
        return _verdict()

    _check("env_seal", seal.get("config", {}).get("torch") == torch_capability.ABSENT,
           f"config.torch = {seal.get('config', {}).get('torch')!r}")
    _check("env_seal", seal.get("inductor") == torch_capability.ABSENT,
           f"inductor = {seal.get('inductor')!r}")
    _check("env_seal", len(env_seal.seal_digest(seal)) == 16,
           "the absent seal still digests to a key axis")
    _check("host_isa", host_isa.impose() == {} and host_isa.effective() == {},
           "the ISA clamp no-ops instead of raising")
    _check("guard_closure",
           guard_closure.establish_posture() == {"torch": torch_capability.ABSENT},
           "the guard posture seals the absence instead of raising")
    return _verdict()


def _verdict() -> int:
    if _FAILURES:
        print("\nFAILED:", file=sys.stderr)
        for f in _FAILURES:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\ntorchless boot contract holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
