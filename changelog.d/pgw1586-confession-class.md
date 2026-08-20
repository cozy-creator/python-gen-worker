- **pgw#1586: EVERY rung now confesses the free VRAM its DECISION saw, not a post-placement
  re-read.** pgw#1595's fix threaded the plan-time figure into the `partial_resident`
  confession only, leaving six sibling call sites — `model_offload`, `sequential`,
  `partial_stream`, both `cpu` arms and the fall-through — still re-reading free VRAM at report
  time, after the weights had landed. Within hours the pgw#1548 lane read `free_gb=0.4` off a
  `model_offload` line on a card with **7.9 GiB free at boot** and reached for a boot-ordering
  cause — the same wrong conclusion pgw#1595 was filed on, from the same artefact, on a rung
  the first fix had not covered. Free VRAM is now read **once, before anything is placed**, and
  carried to every confession. The regression test asserts the **class** by reading the source
  (no `_report_offload_engaged` call may omit `plan_free_gb`), because a per-rung test would
  have passed for `partial_resident` and missed the other six — which is exactly how the first
  fix shipped incomplete.
