- **pgw#1567 (instrument): "armed and never entered" now names its first
  divergence.** torchcg tcg#76 (re-vendored, `748115fc`): the first call no
  armed graph matches logs ONCE per module, at WARNING, the best-matching
  record and its first failing input — expected name/dtype/shape vs received
  py-type/dtype/shape — from the same predicate the guard decides with. Three
  confident root causes died against counter readings in one night while the
  dispatcher knew the divergent input on every call and said nothing; the
  trace line ends that class in one read.
