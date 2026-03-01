# Issue 80 Plan: Class-Only Trainer Contract (Hard Cut)

Goal: make stateful class trainers the only supported training plugin model in `python-gen-worker`.

## Out Of Scope

- `python-gen-worker` must not add model-family-specific training backends.
- Runtime may add only low-level reusable helpers.

## Contract (Canonical)

All custom trainers must implement:

- `setup(ctx) -> None`
- `configure(ctx) -> state`
- `prepare_batch(raw_batch, state, ctx) -> prepared_batch`
- `train_step(prepared_batch, state, ctx) -> StepResult`
- `state_dict(state) -> dict[str, object]`
- `load_state_dict(state, payload, ctx) -> None`

## Ownership Split

Runtime (`python-gen-worker`) owns:
- lifecycle, cancellation, retry/terminal state
- model/data/checkpoint IO
- cadence/reporting/artifact upload

Endpoint trainer class owns:
- dataset semantics and batch shaping
- prompt/mask policies
- train-step math and state evolution

## Model Compatibility Policy (Current Scope)

- Buyer beware for now.
- Runtime does not enforce endpoint-to-model family/type compatibility in this issue.
- Endpoint trainer code must fail fast in `configure()` when model layout/components are incompatible.
- A typed compatibility declaration/validation system is deferred to a separate future issue.

## Rollout Phases

- Phase A: class-only runtime cutover
  - Define runtime-side `TrainerEndpointContract` protocol in `src/gen_worker/trainer/contracts.py`.
  - In `src/gen_worker/trainer/api.py`, remove:
    - `trainer_step_function` decorator path
    - `StepFunctionPlugin`
    - callable symbol fallback
  - In `src/gen_worker/trainer/loop.py`, replace transitional methods with canonical calls:
    - `setup` (once)
    - `configure` (once)
    - per-step `prepare_batch` then `train_step`
  - Resume/checkpoint must use `state_dict/load_state_dict`.

- Phase B: endpoint example cleanup
  - Remove transitional wrappers from examples once runtime cutover lands.
  - Ensure examples only use canonical class methods.
  - Remove function-style trainer docs/snippets.

- Phase C: helper extraction from `~/gen-trainer` and `~/ostris-ai-toolkit`
  - Extract only low-level reusable utilities.
  - Keep model-family backend code endpoint-owned.

## Acceptance Criteria

- Function-based trainer plugins cannot load.
- Non-conforming class plugins fail with clear shape validation errors.
- Both example endpoints in `~/cozy/training-endpoints` run through class-only hooks.
- Runtime tests pass with no references to function-style trainer path.
- No runtime code path depends on endpoint-specific model logic.

## Failure Behavior Contract

- While compatibility is buyer-beware, endpoint code must fail in `configure()` with a clear error when model/repo is incompatible.
- Typed compatibility gating is deferred to future issue `#202`.

## Example Status

`~/cozy/training-endpoints` examples already expose canonical class hooks and can be mapped directly to this contract.
They currently retain runtime compatibility wrappers only because `python-gen-worker` has not yet hard-cut its loop/API.
