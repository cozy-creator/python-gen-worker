- **pgw#1564 (close): the armed-zero and mint-terminal verdicts are DURABLE
  rows, not log lines — and they state the executed code identity.** The L4
  rental that falsified the read-once fix could neither read the log naming
  WHY (SSH dead) nor prove which bytes ran ("pin inferred from the build
  chain"). `SelfMint._settle` now emits a terminal activity row per mint
  (state, landed/failed vs ARMED count — a divergence between armed and
  run-processed is named on the row as the 13/23 ms no-op class —
  `gen_worker_version` + parent contract digest); the host's armed-zero
  warning also rides the wire with its deduped hole reasons. The next $0.07
  rental reads WHY and WHAT-RAN from the hub instead of buying a blind leg.
