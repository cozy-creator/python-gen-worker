- **A wedged runtime mint says so on the WIRE, and so does a graph this pod minted but cannot
  serve.** Closing pgw#1383: the mint's progress guard already condemned correctly, but the
  condemnation went to `logger.error` and a serve pod's stdout goes nowhere (pgw#760) — so from the
  hub's side a condemned mint is simply a mint that stopped emitting, which is indistinguishable
  from a busy healthy pod. `self_mint_wedged` (`phase=no_measured_progress`) now carries the holes,
  the width, what landed and the guard's own sentence; `self_mint_arm_missed` carries the graph that
  is minted and in the store but that this pod's live dispatch did not take, which is why it keeps
  serving eager for a graph it just paid to compile.

  **The root cause this closes, banked in `serving/mint.py`'s docstring so it is not re-lost.** Pod
  `j56tate13oav13` (A40, $0.44/hr) billed for thirty minutes on a mint whose compile children had
  both finished. 0.126.0's terminalise-on-every-exit narrowed it — every branch already emitted
  something — and the remainder was one blocking call in the now-retired supervisor: the terminus
  adopt (an arm plus a numerics verify per graph class, on the live pipeline) ran straight from a
  coroutine, and `activity._emit`'s bound sink ships every report as a task on that same event loop.
  The `seal_publish` transition declared one line earlier was created and never ran, nor was any
  heartbeat, nor any abort. The hub's stall detector fired four times reading
  `inductor_compile step 2/2` and was right every time — that genuinely was the last thing to reach
  it. **The arm may never run on a loop that also ships the worker's reports.** The runtime mint
  already honours that structurally (every arm on a mint worker thread under `arm_lock`, the guard
  sampling on its own cadence beside them); what it was missing was the report.
