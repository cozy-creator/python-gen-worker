# Endpoint/Release V1 Spec (Minimal)

Status: approved implementation spec for issue #62.

Scope:
- Immediate cutover.
- No legacy compatibility routes.
- Public invoke model is endpoint-first.

## Terms

- `tenant_slug`: tenant namespace.
- `endpoint_name`: public endpoint identity inside a tenant.
- `release_id`: random immutable internal id created on each publish.
- `function_name`: internal execution selector inside a worker release.

## Runtime and Traffic Model

- Every publish creates a new `release_id`.
- A release can have multiple image variants for environment compatibility.
- Live traffic defaults to tag `prod`.
- Publish does not move `prod` by default.
- Promote explicitly moves `prod` to a target release.
- First successful publish auto-creates `prod` if unset.
- Rollback is pointer flip: set `prod` to an older release.

## Minimal Database Model (Cozy Hub)

Keep this intentionally small.

### 1) `releases`

One row per publish.

Columns:
- `id uuid primary key default gen_random_uuid()`
- `tenant_id text not null`
- `status text not null` (`queued|building|ready|failed`)
- `source_tarball_path text not null`
- `manifest_json jsonb not null default '{}'::jsonb`
- `error_message text null`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Indexes:
- `(tenant_id, created_at desc)`

### 2) `release_images`

One row per release variant.

Columns:
- `release_id uuid not null references releases(id) on delete cascade`
- `tenant_id text not null`
- `accelerator text not null` (`cpu|cuda|rocm|mps`)
- `accelerator_version text not null default ''`
- `backend text not null` (v1 constraint: `pytorch`)
- `backend_version text not null`
- `architecture text not null` (`amd64|arm64`)
- `python_version text not null`
- `image_ref text not null`
- `image_digest text not null`
- `status text not null` (`building|ready|failed`)
- `error_message text null`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Primary key:
- `(release_id, accelerator, accelerator_version, backend, backend_version, architecture, python_version)`

Indexes:
- `(tenant_id, status, accelerator, architecture, backend)`
- `(tenant_id, release_id)`

### 3) `endpoint_pointers`

Tag pointers from endpoint -> release.

Columns:
- `tenant_id text not null`
- `endpoint_name text not null`
- `tag text not null` (`prod` required, arbitrary tags allowed)
- `release_id uuid not null references releases(id)`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Primary key:
- `(tenant_id, endpoint_name, tag)`

Indexes:
- `(tenant_id, endpoint_name)`
- `(release_id)`

Notes:
- No separate active table: `prod` is the active pointer.
- No separate release_endpoints table in v1: endpoint membership lives in `releases.manifest_json`.

## Minimal API Surface (Cozy Hub)

### Publish/Read

1. `POST /api/v1/endpoints/publish`
- Input: tarball upload + optional publish options.
- Effect: create `release_id`, enqueue build(s), store queued status.
- Output: `{ release_id, status }`.

2. `GET /api/v1/releases/:release_id`
- Output: release status, manifest summary, variant rows.

3. `GET /api/v1/releases/:release_id/logs`
- Output: build logs.

### Endpoint Pointers

4. `GET /api/v1/endpoints`
- Output: endpoint list for tenant + current `prod` release.

5. `GET /api/v1/endpoints/:endpoint_name`
- Output: tag map (`prod`, others) + release summaries.

6. `PUT /api/v1/endpoints/:endpoint_name/tags/:tag`
- Input: `{ "release_id": "..." }`.
- Effect: create/update pointer.

7. `DELETE /api/v1/endpoints/:endpoint_name/tags/:tag`
- Effect: delete non-`prod` pointer.

8. `POST /api/v1/endpoints/:endpoint_name/promote`
- Input: `{ "release_id": "..." }`.
- Effect: set `prod` pointer.

9. `GET /api/v1/endpoints/:endpoint_name/releases`
- Output: release history relevant to endpoint.

Note:
- Rollback uses promote endpoint by passing an older `release_id`.

## Orchestrator Invoke Contract

- Default invoke: `POST /{tenant}/{project}/{endpoint}` -> resolve `tag=prod`.
- Tagged invoke: `POST /{tenant}/{project}/{endpoint}@{tag}`.
- Resolution chain:
  1. `(tenant, endpoint, tag)` -> `release_id`
  2. `release_id` + host capabilities -> compatible `release_images` row
  3. execute target `function_name` inside that release

Realtime path should use the same pointer resolution rules.

## Build/Publish Rules

- Build all required variants for the release.
- Push each variant to Docker Hub.
- Persist immutable digest for each pushed variant.
- Release status is aggregate of variant statuses.
- Backend is fixed to `pytorch` in v1.

## Non-goals for V1

- No legacy deploy-id public model.
- No compatibility routes.
- No migration/backfill complexity.
- No explicit version-id layer beyond `release_id`.

## Acceptance Criteria

- Tenant can publish and receive a new `release_id`.
- Tenant can promote that release to `prod`.
- Invoke routes resolve correctly for default and tagged calls.
- Rollback works by retargeting `prod` to an older release.
- Digests are persisted and returned by release read APIs.
