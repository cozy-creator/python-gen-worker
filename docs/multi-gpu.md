# Multi-GPU (pgw#748 / th#1285)

A pod carries `gpu_count = G x D` GPUs as **G execution groups of degree D**.
`4x1` is four independent serving slots (data parallel). `1x4` is one slot,
4-way sequence parallel. `2x2` is both. **Serving slots = G. Parallel degree =
D.** DP and SP are two projections of one number pair, not two features.

## The hub decides, the worker never invents

The whole contract is one env var (`WORKER_` is a hub-reserved prefix, so this
is a trusted wire fact and not an operator knob):

    WORKER_EXECUTION_TOPOLOGY={"gpu_count":4,"gpus_per_execution_group":2,"execution_groups":2,"parallel":"sequence"}

Derivation is identical on both sides: `D = gpus_per_execution_group`, `G = gpu_count/D`,
group `g` owns devices `[g*D, (g+1)*D)`. `ResolvedCompute.gpu_index` on the job
wire names the group's **rank-0 device** (`0, D, 2D, ...`), so at `D == 1`
nothing about dispatch moves. Absent is legal and means one slot — every CPU
pod and every pre-topology pod. Present but not fully recognised is a typed
refusal: the only producer is the hub.

`gpu_count = execution_groups x gpus_per_execution_group` encodes a **partition
invariant** — a group exclusively owns its devices — so introducing GPU sharing
between groups breaks that arithmetic loudly instead of silently.

The field set is **closed**: an unrecognised key is `topology_unknown_field`,
never a field read as absent. Absent means one slot, so shrugging at an
unknown key is how a hub that bought degree 2 gets served degree 1 in silence.
Growing the contract is therefore its own two-release transition.

**th#1375 rename, in transition.** `group_degree` -> `gpus_per_execution_group`
and `groups` -> `execution_groups`. The hub emits BOTH and the worker accepts
both (they must agree, else `topology_alias_disagree`), so deploy order is
irrelevant. The new spelling shipped in 0.91.0 (`topology.py`
`KEY_GPUS_PER_GROUP` / `KEY_EXECUTION_GROUPS`); the dual-emission window exists
only for pods still on an older tag. th#1376 deletes the old spelling, after
which it becomes an unknown key and is refused by the rule above — check
whether any pod still predates 0.91 before assuming this paragraph is live.

An **execution group** is the serving unit — the cards that cooperate on one
request. It is *not* a PyTorch `ProcessGroup`, which is the communication
handle those ranks collect over; each execution group owns one. Coming from
another stack: `gpus_per_execution_group` is `tensor_parallel_size`
(vLLM/SGLang), `context_parallel_size` (Megatron), `sequence_parallel_size`
(DeepSpeed-Ulysses); `execution_groups` is `data_parallel_size`, `num_replicas`
(Ray Serve), or instance count (Triton). Those names fuse the *width* of a
group with the *technique* filling it; `parallel` is a separate field here so
the width is named independently of the mechanism.

`parallel` names WHO shards:

| value | meaning | platform installs |
|---|---|---|
| `internal` | the model's own device map — what `Resources(gpu_count=N)` has always meant | nothing; never fabric-gated |
| `sequence` | Ulysses context parallel over the group | `enable_parallelism` |
| `cfg` | CFG-parallel (future) | — |

## DP (`G x 1`) — throughput

G groups serve G concurrent jobs over one hub connection. One resident
instance and one residency ledger per group; free VRAM is reported MIN within
a group, MAX across groups — never the pod sum. Weights are mmap'd, so host
RAM does not scale with G.

**One CPython process cannot multiplex N GPUs.** Measured: four groups in one
interpreter serve 0.94x of serial (21% util on every card); four processes with
one group each serve 4.00x at 91-93% util, same pod, same weights (pgw#782/#783).
So each execution group gets its own OS process, `CUDA_VISIBLE_DEVICES` scoped
to its own cards, under a torch-free control parent that owns the hub stream.
A group's process death is attributed to ITS request as a typed FATAL and ITS
group respawns; siblings are untouched.

## SP (`1 x D`, `G x D`) — latency

Ulysses context parallel via `diffusers.ContextParallelConfig` +
`ModelMixin.enable_parallelism()`. Rank 0 is the worker process; D-1 rank
siblings load from the same on-disk store and run the identical forward. Rank 0
decides and broadcasts the plan; nothing below rank 0 measures its own card and
adapts.

**It is bit-identical.** Measured on 4xH100-80 SXM against the same seed at
degree 1: degree 2 = 1.80x, degree 4 = 3.42x, `2x2` both groups concurrent =
1.71x/1.70x, `max|delta| = 0.0` in fp64 on every arm.

Preconditions, each a typed refusal so a mis-staffed pod fails loudly rather
than under-delivering: exactly one class-annotated pipeline slot; a `_cp_plan`
on the sharded component; no CPU offload (diffusers #12533); shape divisible by
the degree; `rowwise` if w8a8; sync handlers only.

Two things that are not refusals but must be priced:

- **Compile is eager-only at `D > 1`.** Once the CP hooks are installed, every
  forward through the sharded modules issues collectives, and the only
  participant-supplying seam is the pipeline-level SP gate — a warm compile, a
  mint seed or a probe forward would hang the group. So no compile selection is
  fetched, no target installs, no cell adopts. Cells and `sp_degree` in the key
  are a later phase.
- **The degraded ladder is refused, not adapted.** It picks an offload rung
  from one card's free VRAM; two ranks that OOM differently execute different
  numbers of collectives and hang — or agree and silently produce wrong output.
  The group fails as a group.

## The hard fabric constraint

**SP requires measured NVLink. There is no second option.** From a 17-probe /
9-host fleet survey:

- NVLink: 241.9-273.9 GB/s achieved, 388.2-389.8 GB/s D2D.
- Everything else: <= 30.2 GB/s achieved, <= 52.9 GB/s D2D. PCIe P2P is dead
  fleet-wide (8-14.5 GB/s, host-staged in practice); "NVL"-branded hosts
  measured unbridged; consumer GeForce P2P is driver-disabled.

Two consequences:

1. **Admission is `interconnect == "nvlink"` AND `peer_gbps >= 200`.** The
   populations leave an empty band between them, so class alone would pass a
   degraded NV4 host. The boot canary always reports `peer_gbps`.
2. **`nvidia-smi topo -m` is anti-predictive.** On the same machine and SKU,
   `SYS` beat `PIX` twice. Report the measured number; do not derive a class
   from wiring. `peer_access` alone means nothing — every non-NVLink host in the
   survey reported it.

`peer_access == True` with `peer_gbps == 0.0` is an NCCL wedge (reproduced
twice: peer access true, collective hangs with no error, no exception, no
timeout). The cheap D2D leg runs BEFORE any collective for exactly this reason:
a canary that hangs is worse than one that fails, because the pod reaches
serving and strands every request routed to it.

`parallel="internal"` is never fabric-gated — a fabric miss must not strip a
working multi-device model of its cards.

## Tier

There are no `*_fast` function variants. The same function carries a
request-level `tier` (`standard` | `fast` | `flex`), platform-reserved, never
visible to endpoint code. The hub maps tier to degree: `standard` = 1, `fast` =
2 or 4. Providers price multi-GPU linearly, so **the fast tier's price
multiplier IS the degree** — no invented pricing knob.

Degree is chosen by the hub's objective score. The worker's only job is to
tolerate whatever it is delivered, including a delivered slot count that
differs from what it booted with (a sharded pod whose measured interconnect is
not NVLink is demoted to `G x 1` rather than retired).

## Failure doctrine

A rank death is not yet a typed group failure (pgw#792): rank 0's collective
runs to the full 300 s ceiling, raises nothing, and NCCL then takes the process
down. `RankGroup.check_alive()` brackets a call; the whole stall is inside one.
Staff on the assumption **"one rank death = one process restart, ~6 minutes"**,
not "one request fails".
