# Multi-GPU — measured results (pgw#748 / th#1285)

A pod carries `gpu_count = G x D` GPUs as **G execution groups of degree D**.
Serving slots = G, parallel degree = D. Everything below is a measurement or a
ruling derived from one; the wire contract itself
(`WORKER_EXECUTION_TOPOLOGY`) is read in `procsplit` / the topology parser.

> Predates th#1426 (gen-worker 0.91.0), which added
> `max_gpus_per_execution_group` to `Resources`. The "Prod pins gen-worker
> 0.79.0" transition note that used to live here has aged out.

## DP: one OS process per execution group

**One CPython process cannot multiplex N GPUs.** Measured (pgw#782/#783),
same pod, same weights:

- four execution groups in **one** interpreter → **0.94x** of serial, **21 %**
  util on every card.
- four **processes** with one group each → **4.00x**, **91-93 %** util.

Hence: each execution group gets its own OS process with
`CUDA_VISIBLE_DEVICES` scoped to its own cards, under a torch-free control
parent that owns the hub stream. A group's process death is attributed to ITS
request as a typed FATAL and only ITS group respawns.

## SP: it is bit-identical, and it scales

Measured on **4xH100-80 SXM**, same seed, against degree 1:

| arm | speedup |
|---|---|
| degree 2 | **1.80x** |
| degree 4 | **3.42x** |
| `2x2`, both groups concurrent | **1.71x / 1.70x** |

`max|delta| = 0.0` in fp64 on **every** arm — sequence parallelism is
bit-identical, not approximately equal.

## The hard fabric constraint

**SP requires measured NVLink. There is no second option.** From a **17-probe
/ 9-host** fleet survey:

- NVLink: **241.9-273.9 GB/s** achieved, **388.2-389.8 GB/s** D2D.
- Everything else: **<= 30.2 GB/s** achieved, **<= 52.9 GB/s** D2D. PCIe P2P
  is dead fleet-wide (8-14.5 GB/s, host-staged in practice); "NVL"-branded
  hosts measured **unbridged**; consumer GeForce P2P is driver-disabled.

Three rulings fall out:

1. **Admission is `interconnect == "nvlink"` AND `peer_gbps >= 200`.** The two
   populations leave an empty band between them, so class alone would pass a
   degraded NV4 host. The boot canary always reports `peer_gbps`.
2. **`nvidia-smi topo -m` is anti-predictive.** On the same machine and SKU,
   `SYS` beat `PIX` **twice**. Report the measured number; never derive a class
   from wiring. `peer_access` alone means nothing — every non-NVLink host in
   the survey reported it true.
3. **The cheap D2D leg runs BEFORE any collective.** `peer_access == True`
   with `peer_gbps == 0.0` is an NCCL wedge, **reproduced twice**: the
   collective hangs with **no error, no exception, no timeout**. A canary that
   hangs is worse than one that fails, because the pod reaches serving and
   strands every request routed to it.

`parallel="internal"` (the model's own device map) is never fabric-gated — a
fabric miss must not strip a working multi-device model of its cards.

## Priced non-refusals

Two costs that are deliberately paid rather than refused:

- **Compile is eager-only at `D > 1`.** Once the CP hooks are installed every
  forward through the sharded modules issues collectives, and the only
  participant-supplying seam is the pipeline-level SP gate — a warm compile, a
  mint seed, or a probe forward would hang the group. No compile selection is
  fetched, no target installs, no cell adopts. `sp_degree` in the cell key is
  a later phase.
- **The degraded fit ladder is refused, not adapted.** It picks an offload
  rung from one card's free VRAM; two ranks that OOM differently execute
  different numbers of collectives and hang — or agree and silently produce
  wrong output. The group fails as a group.

## Failure doctrine (pgw#792)

A rank death is **not** yet a typed group failure. Rank 0's collective runs
the full **300 s** ceiling and raises **nothing**; NCCL then takes the process
down. `RankGroup.check_alive()` brackets a call, and the whole stall is inside
one.

**Staff on "one rank death = one process restart, ~6 minutes", not "one
request fails."**
