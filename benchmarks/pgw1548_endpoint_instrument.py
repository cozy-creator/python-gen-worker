"""The endpoint-side instrument for pgw#1548's LoRA amortization arm.

**This file is never imported by the harness.** It is COPIED into a per-arm
workspace copy of the endpoint as ``src/<pkg>/_bench1548.py`` and activated by
a single appended line in that copy's ``main.py``. The shared checkout is never
touched (coordinator, 2026-08-20).

## Why an endpoint-side instrument exists at all

Two facts, both read off the code rather than assumed:

1. **`gen-worker run` has NO channel for adapter picks.** `cli/run.py` sends
   `{"function", "payload"}`; `cli/daemon.py:434` reads only those two and
   calls `host.dispatch(function, payload, ...)` — never passing its
   `loras: Sequence[Any] = ()` kwarg (`serving/host.py:526`). So a locally
   `up`'d endpoint fills every plural adapter slot with `[]`, always. This is
   NOT a hub-only design: `envelope.AdapterRow.path` materializes a plain
   local `Path`, so the worker takes local adapter bytes natively — the local
   CLI simply has no plumbing to state one. Filed as an author-surface gap.
2. **No endpoint wires the fold path.** `sdxl/main.py:235-247` `Model.adapters`
   is the EAGER peft path (`load_lora_weights` / `set_adapters` /
   `unload_lora_weights`). `lora_fold.folded` has zero call sites outside its
   own module (se#808 is the owed rewire). So "compiled + fold-ahead LoRA" is
   not a flag; it is a patch, and this file is its prototype.

## What it measures, and why through a file rather than the payload

The mode has to vary PER REQUEST while the daemon boots ONCE per arm, so an
env var is fixed too early. A control file is read fresh at each request; a
trace file is appended at the END of each request, which is the only point at
which the RESTORE wall exists — restore happens in `adapters().__exit__`,
after the entrypoint has already returned its value.

Emitted per request (one JSON object per line):

* ``denoise_s`` — the `pipe(...)` call alone, so per-step ms is
  ``denoise_s / steps`` and is not contaminated by VAE decode or encode.
* ``save_s`` — decode + encode + write, kept separate rather than folded in.
* ``fold_s`` / ``restore_s`` — the RECURRING per-request cost Paul asked for:
  every request pays both, because the pipeline must be byte-identical for the
  next one.
* ``rearm_calls`` — how many armed constant tables `rearm_constants` re-installed.
  ZERO on a compiled arm would mean the fold never met the artifact.
* ``mode`` — off / eager / fold, so a row cannot be read without knowing which
  path produced it.
"""

from __future__ import annotations

INSTRUMENT_SOURCE = r'''
"""pgw#1548 benchmark instrumentation — INJECTED, not committed to the endpoint."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

_CONTROL = os.environ.get("PGW1548_BENCH_CONTROL", "")
_TRACE = os.environ.get("PGW1548_BENCH_TRACE", "")

#: Per-request scratch, filled by the patched pieces and flushed by whichever
#: one runs last (the adapter scope's exit, which is after the entrypoint).
_pending: dict = {}


def _control() -> dict:
    """Read the mode FRESH per request. The daemon boots once per arm; an env
    var would pin the mode to the boot and there would be no A/B at all."""
    if not _CONTROL:
        return {"mode": "off"}
    try:
        return json.loads(Path(_CONTROL).read_text())
    except Exception:
        return {"mode": "off"}


def _flush() -> None:
    if not _TRACE or not _pending:
        return
    with open(_TRACE, "a") as handle:
        handle.write(json.dumps(_pending) + "\n")
    _pending.clear()


def _adapter_bytes(path: str, ref: str):
    """Through the platform's own loader, not a bare safetensors.load_file —
    the zero-delta and key-grammar refusals live there (lora_fold's docstring
    says so explicitly), and a benchmark that skips them could measure a fold
    of nothing."""
    from gen_worker.utils.lora import load_adapter_state_dict

    return load_adapter_state_dict(Path(path), ref=ref)


#: Loaded once per boot and reused: reading 50-400 MB of safetensors off disk
#: is I/O, not the fold, and charging it to every request would price the fold
#: as whatever the page cache felt like doing.
_state_cache: dict = {}


def install() -> None:
    from . import main as _main

    model_cls = None
    for value in vars(_main).values():
        if isinstance(value, type) and hasattr(value, "adapters") and \
                value.__module__ == _main.__name__:
            model_cls = value
            break
    if model_cls is None:
        raise RuntimeError("pgw1548 instrument: no Model class with .adapters")

    original_adapters = model_cls.adapters
    original_run = _main._run

    @contextmanager
    def _bench_adapters(self, applied: Sequence[Any]) -> Iterator[None]:
        control = _control()
        mode = str(control.get("mode", "off"))
        _pending["mode"] = mode
        if mode == "off" or not control.get("lora_path"):
            _pending["fold_s"] = 0.0
            _pending["rearm_calls"] = 0
            with original_adapters(self, applied):
                yield
            _pending["restore_s"] = 0.0
            _flush()
            return

        path = str(control["lora_path"])
        ref = str(control.get("ref") or path)
        scale = float(control.get("scale", 1.0))
        key = (path, ref)
        if key not in _state_cache:
            _state_cache[key] = _adapter_bytes(path, ref)
        state = _state_cache[key]

        if mode == "eager":
            # The REFERENCE point: diffusers' own adapter ops. On a
            # compiled-armed unet the P0 guard drops this to loud eager --
            # that is the guard working, and it is exactly the cost the fold
            # exists to avoid paying.
            started = time.perf_counter()
            self.pipe.load_lora_weights(path, adapter_name="bench")
            self.pipe.set_adapters(["bench"], adapter_weights=[scale])
            _pending["fold_s"] = time.perf_counter() - started
            _pending["rearm_calls"] = 0
            try:
                yield
            finally:
                started = time.perf_counter()
                self.pipe.unload_lora_weights()
                _pending["restore_s"] = time.perf_counter() - started
                _flush()
            return

        if mode != "fold":
            raise RuntimeError(f"pgw1548 instrument: unknown mode {mode!r}")

        from gen_worker.models import lora_fold
        from gen_worker.serving import adapter_guard

        rearmed: list = []

        def _rebind(module: Any) -> Any:
            count = adapter_guard.rearm_constants(module)
            rearmed.append(int(count or 0))
            return count

        # The fold wall is the ENTER of the scope; the restore wall is its
        # EXIT. Both are per-request and recurring (Paul: "we need to fuse /
        # unfuse the weights after every request"), so they are timed
        # separately and never summed into one "adapter overhead".
        started = time.perf_counter()
        scope = lora_fold.folded(self.pipe, [(state, scale, ref)], rebind=_rebind)
        stats = scope.__enter__()
        _pending["fold_s"] = time.perf_counter() - started
        _pending["fold_stats"] = dict(stats or {})
        # rearm fires on the way in AND on the way out; the entry count is what
        # says the fold met the compiled artifact at all.
        _pending["rearm_calls"] = sum(rearmed)
        try:
            yield
        finally:
            started = time.perf_counter()
            scope.__exit__(None, None, None)
            _pending["restore_s"] = time.perf_counter() - started
            _pending["rearm_calls_total"] = sum(rearmed)
            _flush()

    def _bench_run(model, ctx, *, steps: int, fmt, seed, **call_kwargs):
        import threading

        import torch

        generator = (
            torch.Generator(device=model.pipe.device).manual_seed(seed)
            if seed is not None else None
        )
        _pending["steps"] = int(steps)

        # --- the pgw#1586 probe, riding a request that was happening anyway ---
        # QUESTION: does AOTI's request-time workspace allocate OUTSIDE
        # PyTorch's caching allocator (a direct cudaMalloc), making it invisible
        # to `max_memory_allocated` and therefore invisible to any residency
        # ledger built on allocator statistics?
        #
        # The two readings answer it TOGETHER and neither answers it alone:
        #   allocator_peak_delta -- what PyTorch believes it allocated
        #   driver_used_peak_delta -- what the DRIVER handed out, sampled
        #     because a before/after pair cannot see a peak that is released
        #     before the request returns.
        # If the driver delta materially exceeds the allocator delta, the
        # ledger must sample driver-level in the compiled regime.
        device = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info(device)
        allocated_before = torch.cuda.memory_allocated()
        min_free = [free_before]
        sampling = threading.Event()
        sampling.set()

        def _sample() -> None:
            while sampling.is_set():
                try:
                    free, _ = torch.cuda.mem_get_info(device)
                except Exception:  # noqa: BLE001 — a probe must never kill a request
                    return
                if free < min_free[0]:
                    min_free[0] = free
                time.sleep(0.05)

        sampler = threading.Thread(target=_sample, daemon=True)
        sampler.start()

        with torch.inference_mode():
            started = time.perf_counter()
            result = model.pipe(
                num_inference_steps=steps,
                generator=generator,
                callback_on_step_end=ctx.step_callback(steps),
                **call_kwargs,
            )
            torch.cuda.synchronize()
            _pending["denoise_s"] = time.perf_counter() - started

        sampling.clear()
        sampler.join(timeout=1.0)
        _pending["vram"] = {
            "total_bytes": int(total),
            "free_before_bytes": int(free_before),
            "driver_min_free_bytes": int(min_free[0]),
            "driver_used_peak_delta_bytes": int(free_before - min_free[0]),
            "allocator_before_bytes": int(allocated_before),
            "allocator_peak_bytes": int(torch.cuda.max_memory_allocated()),
            "allocator_peak_delta_bytes": int(
                torch.cuda.max_memory_allocated() - allocated_before),
        }
        started = time.perf_counter()
        asset = ctx.save_image(result.images[0], format=fmt)
        _pending["save_s"] = time.perf_counter() - started
        return asset

    model_cls.adapters = _bench_adapters
    _main._run = _bench_run
    _main._PGW1548_ORIGINALS = (original_adapters, original_run)
'''

#: Appended verbatim to the workspace copy's `main.py`. One line, at EOF, so
#: the patch cannot land in the middle of a function and cannot depend on any
#: string in the endpoint's body staying put.
ACTIVATION = (
    "\n\n# pgw#1548 benchmark instrumentation (INJECTED into a workspace COPY)\n"
    "from . import _bench1548 as _bench1548  # noqa: E402\n"
    "_bench1548.install()\n"
)


def install_into(package_dir, main_py) -> None:
    """Write the instrument beside the endpoint's `main.py` and activate it."""

    from pathlib import Path

    package_dir = Path(package_dir)
    main_py = Path(main_py)
    (package_dir / "_bench1548.py").write_text(INSTRUMENT_SOURCE)
    text = main_py.read_text()
    if "_bench1548" not in text:
        main_py.write_text(text + ACTIVATION)


__all__ = ["INSTRUMENT_SOURCE", "ACTIVATION", "install_into"]
