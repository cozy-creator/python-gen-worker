- `ExecutionSpec.arm.graph_contract_digest` now follows the artifact: required on an
  `aot_cell` arm (pgw#903's pre-dlopen fence compares it), refused on `dynamo` and
  `eager_only` arms. A non-cell arm has no traced graph the hub could name ahead of
  dispatch, and `aot_identity.expected_from_plan` already returns `None` there — so the
  old unconditional requirement made every non-cell dispatch unbuildable hub-side while
  the value it demanded had no reader. Paired with tensorhub `1457-arm-graph-contract`;
  both halves must ship together.
