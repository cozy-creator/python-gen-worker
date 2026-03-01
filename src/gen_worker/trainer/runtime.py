from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import inspect
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping

from .api import load_trainer_plugin
from .arrow_feed import ArrowFeedConfig, ParquetArrowBatchFeeder
from .contracts import StepContext, TrainingJobSpec
from .loop import run_training_loop
from .orchestrated import (
    InputMaterializationError,
    JsonHttpArtifactUploader,
    RuntimeCancelPolicy,
    RuntimeInputDownloader,
    StartupContractError,
    UploadEndpoints,
    is_truthy,
)

logger = logging.getLogger("TrainerRuntime")


@dataclass(frozen=True)
class RuntimeArtifactLayout:
    checkpoints_dir: str
    samples_dir: str
    metrics_dir: str
    events_path: str | None = None


@dataclass(frozen=True)
class TrainerRuntimeConfig:
    trainer_import: str
    job_spec_path: str
    artifacts: RuntimeArtifactLayout
    capability_token: str | None = None
    orchestrated: bool = False
    max_runtime_seconds: int = 0
    cancel_file_path: str | None = None
    upload_endpoints: UploadEndpoints = UploadEndpoints()


@dataclass(frozen=True)
class SampleRequest:
    name: str
    task: str
    prompt: str
    negative_prompt: str = ""
    instruction: str | None = None
    source_image: str | None = None
    seed: int | None = None


class LocalTrainingReporter:
    def __init__(self, events_path: str | None = None, *, cancel_policy: RuntimeCancelPolicy | None = None) -> None:
        self._events_path = events_path
        self._cancel_policy = cancel_policy
        self._run_id = ""
        self._seq = 0
        self._last_completed_step = 0
        self._last_final_checkpoint: str | None = None
        self._last_failed_step = 0
        self._last_error = ""
        if self._events_path:
            Path(self._events_path).parent.mkdir(parents=True, exist_ok=True)

    def _event(self, event: str, **payload: Any) -> None:
        self._seq += 1
        body = {
            "schema_version": "trainer_event.v1",
            "event": event,
            "seq": self._seq,
            "run_id": self._run_id,
            "timestamp_ms": int(time.time() * 1000),
            **payload,
        }
        logger.info("trainer.event %s", json.dumps(body, separators=(",", ":"), sort_keys=True))
        if self._events_path:
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(body, separators=(",", ":"), sort_keys=True))
                f.write("\n")

    def started(self, *, run_id: str) -> None:
        self._run_id = run_id
        if self._cancel_policy is not None and self._cancel_policy.started_monotonic_s <= 0:
            self._cancel_policy.start()
        self._event("started", run_id=run_id)

    def metric(self, *, name: str, value: float, step: int) -> None:
        self._event("metric", name=name, value=value, step=step)

    def checkpoint(self, *, path: str, step: int) -> None:
        self._event("checkpoint", path=path, step=step)

    def sample(self, *, path: str, step: int) -> None:
        self._event("sample", path=path, step=step)

    def completed(self, *, step: int, final_checkpoint: str | None) -> None:
        self._last_completed_step = int(step)
        self._last_final_checkpoint = final_checkpoint
        self._event("completed", step=step, final_checkpoint=final_checkpoint)

    def failed(self, *, step: int, error: str) -> None:
        self._last_failed_step = int(step)
        self._last_error = str(error)
        self._event("failed", step=step, error=error)

    def is_canceled(self) -> bool:
        if self._cancel_policy is not None and self._cancel_policy.is_canceled():
            return True
        cancel_flag = (os.getenv("TRAINER_CANCELLED") or "").strip().lower()
        return cancel_flag in {"1", "true", "t", "yes", "y"}

    @property
    def cancel_reason(self) -> str:
        if self._cancel_policy is not None:
            return self._cancel_policy.reason
        return "canceled"

    @property
    def completed_step(self) -> int:
        return self._last_completed_step

    @property
    def final_checkpoint(self) -> str | None:
        return self._last_final_checkpoint

    @property
    def failed_step(self) -> int:
        return self._last_failed_step

    @property
    def failed_error(self) -> str:
        return self._last_error


class LocalArtifactWriter:
    def __init__(
        self,
        *,
        checkpoints_dir: str,
        samples_dir: str,
        sample_requests: tuple[SampleRequest, ...] = (),
        sample_fixed_seed: int | None = None,
    ) -> None:
        self._checkpoints_dir = Path(checkpoints_dir)
        self._samples_dir = Path(samples_dir)
        self._sample_requests = sample_requests
        self._sample_fixed_seed = sample_fixed_seed
        self._last_final_checkpoint_path: str | None = None
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._samples_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _checkpoint_hook(trainer: object) -> Any:
        hook = getattr(trainer, "save_checkpoint", None)
        if callable(hook):
            return hook
        return None

    @staticmethod
    def _resolve_primary_path(meta: Mapping[str, Any], output_dir: Path) -> str | None:
        raw = str(meta.get("primary_path") or "").strip()
        if raw == "":
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = output_dir / p
        return str(p)

    def write_checkpoint(
        self,
        *,
        step: int,
        state_payload: dict[str, object],
        trainer: object,
        state: Any,
        ctx: StepContext,
    ) -> str:
        out = self._checkpoints_dir / f"step-{step:08d}.json"
        payload: dict[str, Any] = {"step": step, "run_id": ctx.job.run_id, "state": state_payload}
        hook = self._checkpoint_hook(trainer)
        if hook is not None:
            output_dir = self._checkpoints_dir / f"step-{step:08d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            raw = hook(state=state, step=step, output_dir=str(output_dir), final=False, ctx=ctx)
            if isinstance(raw, Mapping):
                meta = {str(k): v for (k, v) in raw.items()}
                payload["trainer_checkpoint"] = meta
                primary = self._resolve_primary_path(meta, output_dir)
                _atomic_write_json(out, payload)
                if primary:
                    return primary
        _atomic_write_json(out, payload)
        return str(out)

    def write_samples(self, *, step: int, state: Any, ctx: StepContext) -> list[str]:
        if not self._sample_requests:
            out = self._samples_dir / f"step-{step:08d}.txt"
            _atomic_write_json(
                out.with_suffix(".json"),
                {"step": step, "run_id": ctx.job.run_id, "task": "t2i", "prompt": ""},
            )
            out.write_text(f"sample step={step} run_id={ctx.job.run_id}\n", encoding="utf-8")
            return [str(out)]

        requests = self._sample_requests
        out_paths: list[str] = []
        for idx, req in enumerate(requests):
            out = self._samples_dir / f"step-{step:08d}-{idx:02d}.json"
            seed = req.seed if req.seed is not None else self._sample_fixed_seed
            payload = {
                "step": step,
                "run_id": ctx.job.run_id,
                "task": req.task,
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "instruction": req.instruction,
                "source_image": req.source_image,
                "seed": seed,
            }
            generated = _try_generate_sample(state=state, ctx=ctx, step=step, request=req, seed=seed)
            if isinstance(generated, Mapping):
                payload.update({str(k): v for (k, v) in generated.items()})
            _atomic_write_json(out, payload)
            out_paths.append(str(out))
        return out_paths

    def finalize(
        self,
        *,
        state_payload: dict[str, object],
        trainer: object,
        state: Any,
        ctx: StepContext,
    ) -> str | None:
        out = self._checkpoints_dir / "final.json"
        payload: dict[str, Any] = {"final": True, "run_id": ctx.job.run_id, "state": state_payload}
        hook = self._checkpoint_hook(trainer)
        primary: str | None = None
        if hook is not None:
            output_dir = self._checkpoints_dir / "final"
            output_dir.mkdir(parents=True, exist_ok=True)
            raw = hook(state=state, step=int(state_payload.get("step", 0)), output_dir=str(output_dir), final=True, ctx=ctx)
            if isinstance(raw, Mapping):
                meta = {str(k): v for (k, v) in raw.items()}
                payload["trainer_checkpoint"] = meta
                primary = self._resolve_primary_path(meta, output_dir)
        _atomic_write_json(out, payload)
        self._last_final_checkpoint_path = primary or str(out)
        return primary or str(out)

    @property
    def last_final_checkpoint_path(self) -> str | None:
        return self._last_final_checkpoint_path


def _read_job_spec(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trainer job spec not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("trainer job spec must be a JSON object")
    return data


def _runtime_config_from_env() -> TrainerRuntimeConfig:
    job_spec_path = (os.getenv("TRAINER_JOB_SPEC_PATH") or "/app/.cozy/trainer_job.json").strip()
    if not job_spec_path:
        raise StartupContractError("startup.missing_job_spec_path", "TRAINER_JOB_SPEC_PATH is required")
    trainer_import = (os.getenv("TRAINER_PLUGIN") or "").strip()
    artifacts_root = (os.getenv("TRAINER_ARTIFACTS_DIR") or "/tmp/training").strip()
    checkpoints_dir = (os.getenv("TRAINER_CHECKPOINTS_DIR") or f"{artifacts_root}/checkpoints").strip()
    samples_dir = (os.getenv("TRAINER_SAMPLES_DIR") or f"{artifacts_root}/samples").strip()
    metrics_dir = (os.getenv("TRAINER_METRICS_DIR") or f"{artifacts_root}/metrics").strip()
    events_path = (os.getenv("TRAINER_EVENTS_PATH") or f"{metrics_dir}/events.jsonl").strip() or None
    capability_token = (os.getenv("TRAINER_CAPABILITY_TOKEN") or "").strip() or None
    orchestrated = is_truthy(os.getenv("TRAINER_ORCHESTRATED"))
    max_runtime_seconds = int((os.getenv("TRAINER_MAX_RUNTIME_SECONDS") or "0").strip() or "0")
    cancel_file_path = (os.getenv("TRAINER_CANCEL_FILE") or "").strip() or None

    upload_endpoints = UploadEndpoints(
        metrics_url=(os.getenv("TRAINER_UPLOAD_METRICS_URL") or "").strip(),
        checkpoint_url=(os.getenv("TRAINER_UPLOAD_CHECKPOINT_URL") or "").strip(),
        sample_url=(os.getenv("TRAINER_UPLOAD_SAMPLE_URL") or "").strip(),
        terminal_url=(os.getenv("TRAINER_UPLOAD_TERMINAL_URL") or "").strip(),
    )
    if not trainer_import:
        payload = _read_job_spec(job_spec_path)
        trainer_import = str(payload.get("trainer") or "").strip()
        if not capability_token:
            capability_token = str(payload.get("capability_token") or "").strip() or None
        if not orchestrated:
            orchestrated = bool(payload.get("orchestrated", False))
        if max_runtime_seconds <= 0:
            max_runtime_seconds = int(payload.get("max_runtime_seconds") or 0)
        if not cancel_file_path:
            cancel_file_path = str(payload.get("cancel_file_path") or "").strip() or None
        up = payload.get("uploads")
        if isinstance(up, dict):
            upload_endpoints = UploadEndpoints(
                metrics_url=str(up.get("metrics_url") or upload_endpoints.metrics_url or "").strip(),
                checkpoint_url=str(up.get("checkpoint_url") or upload_endpoints.checkpoint_url or "").strip(),
                sample_url=str(up.get("sample_url") or upload_endpoints.sample_url or "").strip(),
                terminal_url=str(up.get("terminal_url") or upload_endpoints.terminal_url or "").strip(),
            )
    if not trainer_import:
        raise StartupContractError("startup.missing_trainer_import", "trainer import is required via TRAINER_PLUGIN or job spec field 'trainer'")
    if not checkpoints_dir or not samples_dir or not metrics_dir:
        raise StartupContractError("startup.invalid_artifact_paths", "artifact directories must be non-empty")
    if max_runtime_seconds < 0:
        raise StartupContractError("startup.invalid_timeout", "max runtime seconds must be >= 0")
    if orchestrated and not capability_token:
        raise StartupContractError(
            "startup.missing_capability_token",
            "TRAINER_CAPABILITY_TOKEN (or job spec `capability_token`) is required in orchestrated mode",
        )
    return TrainerRuntimeConfig(
        trainer_import=trainer_import,
        job_spec_path=job_spec_path,
        artifacts=RuntimeArtifactLayout(
            checkpoints_dir=checkpoints_dir,
            samples_dir=samples_dir,
            metrics_dir=metrics_dir,
            events_path=events_path,
        ),
        capability_token=capability_token,
        orchestrated=orchestrated,
        max_runtime_seconds=max_runtime_seconds,
        cancel_file_path=cancel_file_path,
        upload_endpoints=upload_endpoints,
    )


def _resolve_batches(spec: dict[str, Any]) -> Any:
    dataset = spec.get("dataset")
    if isinstance(dataset, dict):
        parquet_paths = dataset.get("parquet_paths")
        if isinstance(parquet_paths, list) and parquet_paths:
            cfg = ArrowFeedConfig(
                batch_size=int(dataset.get("batch_size", 32)),
                readahead=int(dataset.get("readahead", 2)),
                columns=tuple(dataset.get("columns") or ()),
            )
            return ParquetArrowBatchFeeder([str(p) for p in parquet_paths], cfg)
    batches = spec.get("mock_batches")
    if isinstance(batches, list):
        return batches
    return []


def _materialize_input_refs(spec: dict[str, Any], cfg: TrainerRuntimeConfig) -> dict[str, object]:
    """
    Materialize orchestrator-provided refs/URLs to local paths.

    Supported job spec fields:
      inputs.base_model_ref
      inputs.dataset_parquet_refs (list[str])
      inputs.resume_checkpoint_ref
    """
    inputs = spec.get("inputs")
    if not isinstance(inputs, dict):
        return {}

    downloader = RuntimeInputDownloader(
        root_dir=str(Path(cfg.artifacts.metrics_dir) / "materialized-inputs"),
        capability_token=cfg.capability_token,
    )

    model_handles: dict[str, object] = {}
    base_model_ref = str(inputs.get("base_model_ref") or inputs.get("base_model_url") or "").strip()
    if base_model_ref:
        model_dir = downloader.download_weights(base_model_ref)
        model_handles["base_model_dir"] = model_dir
        model_handles["base_model_ref"] = base_model_ref

    dataset_refs_raw = inputs.get("dataset_parquet_refs")
    if dataset_refs_raw is None:
        dataset_refs_raw = inputs.get("dataset_parquet_urls")
    dataset_refs: list[str] = []
    if isinstance(dataset_refs_raw, list):
        dataset_refs = [str(x).strip() for x in dataset_refs_raw if str(x).strip()]
    if dataset_refs:
        dataset = spec.get("dataset")
        if not isinstance(dataset, dict):
            dataset = {}
            spec["dataset"] = dataset
        dataset["parquet_paths"] = [downloader.download_dataset_parquet(ref) for ref in dataset_refs]

    resume_ref = str(inputs.get("resume_checkpoint_ref") or inputs.get("resume_checkpoint_url") or "").strip()
    if resume_ref:
        spec["resume_checkpoint_path"] = downloader.download_resume_checkpoint(resume_ref)
        spec["resume_from_latest"] = True

    return model_handles


_STEP_FILE_RE = re.compile(r"^step-(\d+)\.json$")


def _step_from_checkpoint_name(path: Path) -> int:
    m = _STEP_FILE_RE.match(path.name)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _is_valid_checkpoint_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    state = payload.get("state")
    return isinstance(state, dict)


def _read_resume_state_payload(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a JSON object: {path}")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint payload missing object field `state`: {path}")
    return {str(k): v for (k, v) in state.items()}


def _detect_resume_checkpoint(checkpoints_dir: str, spec: dict[str, Any]) -> tuple[int, str | None]:
    if not bool(spec.get("resume_from_latest", False)):
        return 0, None
    p = Path(checkpoints_dir)
    if not p.exists():
        return 0, None

    explicit = str(spec.get("resume_checkpoint_path") or "").strip()
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists() and _is_valid_checkpoint_file(explicit_path):
            return _step_from_checkpoint_name(explicit_path), str(explicit_path)

    best_step = 0
    best_path: str | None = None
    for child in p.iterdir():
        step = _step_from_checkpoint_name(child)
        if step <= 0:
            continue
        if not _is_valid_checkpoint_file(child):
            continue
        if step > best_step:
            best_step = step
            best_path = str(child)
    return best_step, best_path


def _already_completed_from_final(checkpoints_dir: str, spec: dict[str, Any]) -> bool:
    if not bool(spec.get("resume_from_latest", False)):
        return False
    p = Path(checkpoints_dir) / "final.json"
    if not p.exists():
        return False
    return _is_valid_checkpoint_file(p)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(tmp)


def _call_hook_with_supported_args(fn: Any, **payload: Any) -> Any:
    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    call_kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if name in payload:
            call_kwargs[name] = payload[name]
    if accepts_kwargs:
        for (k, v) in payload.items():
            if k not in call_kwargs:
                call_kwargs[k] = v
    return fn(**call_kwargs)


def _try_generate_sample(
    *,
    state: Any,
    ctx: StepContext,
    step: int,
    request: SampleRequest,
    seed: int | None,
) -> Mapping[str, Any] | None:
    hook = None
    if hasattr(state, "generate_sample"):
        hook = getattr(state, "generate_sample")
    elif isinstance(state, Mapping):
        candidate = state.get("generate_sample")
        if callable(candidate):
            hook = candidate
    if not callable(hook):
        return None
    payload = _call_hook_with_supported_args(
        hook,
        step=step,
        ctx=ctx,
        request=request,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        instruction=request.instruction,
        source_image=request.source_image,
        seed=seed,
        task=request.task,
    )
    if isinstance(payload, Mapping):
        return payload
    return None


def _parse_sample_requests(spec: dict[str, Any]) -> tuple[tuple[SampleRequest, ...], int | None]:
    fixed_seed_raw = spec.get("sample_seed")
    fixed_seed = int(fixed_seed_raw) if fixed_seed_raw is not None else None
    raw = spec.get("sample_prompts")
    if not isinstance(raw, list) or not raw:
        return (), fixed_seed

    requests: list[SampleRequest] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            requests.append(SampleRequest(name=f"sample-{idx:02d}", task="t2i", prompt=item))
            continue
        if not isinstance(item, dict):
            continue
        seed_raw = item.get("seed")
        requests.append(
            SampleRequest(
                name=str(item.get("name") or f"sample-{idx:02d}"),
                task=str(item.get("task") or "t2i"),
                prompt=str(item.get("prompt") or ""),
                negative_prompt=str(item.get("negative_prompt") or ""),
                instruction=str(item.get("instruction")) if item.get("instruction") is not None else None,
                source_image=str(item.get("source_image")) if item.get("source_image") is not None else None,
                seed=int(seed_raw) if seed_raw is not None else None,
            )
        )
    return tuple(requests), fixed_seed


def _classify_runtime_error(exc: Exception, *, phase: str) -> tuple[str, str, str]:
    msg = str(exc)
    if isinstance(exc, StartupContractError):
        return ("startup", msg, repr(exc))
    if isinstance(exc, InputMaterializationError):
        return ("input", "failed to materialize training inputs", repr(exc))
    if "authorization" in msg.lower() or "unauthorized" in msg.lower() or "forbidden" in msg.lower():
        return ("auth", "authorization failed", repr(exc))
    if phase == "upload":
        return ("upload", "failed to upload training artifacts", repr(exc))
    if phase == "model-load":
        return ("model-load", "failed to initialize trainer model/backend", repr(exc))
    if phase == "train-loop":
        return ("train-step", msg, repr(exc))
    return (phase, msg, repr(exc))


def run_training_runtime_from_env() -> int:
    cfg = _runtime_config_from_env()
    spec = _read_job_spec(cfg.job_spec_path)

    cancel_policy = RuntimeCancelPolicy(
        cancel_file_path=cfg.cancel_file_path,
        max_runtime_seconds=max(0, int(cfg.max_runtime_seconds)),
    )
    reporter = LocalTrainingReporter(events_path=cfg.artifacts.events_path, cancel_policy=cancel_policy)

    model_handles: dict[str, object] = {}
    try:
        model_handles.update(_materialize_input_refs(spec, cfg))
    except Exception as exc:
        category, safe_message, debug = _classify_runtime_error(exc if isinstance(exc, Exception) else Exception(str(exc)), phase="input")
        logger.exception("trainer.runtime.%s debug=%s", category, debug)
        reporter.failed(step=0, error=f"{category}:{safe_message}")
        raise

    job = TrainingJobSpec(
        run_id=str(spec.get("run_id") or ""),
        trainer_api_version=str(spec.get("trainer_api_version") or "v1"),
        max_steps=int(spec.get("max_steps", 0)),
        metric_every=int(spec.get("metric_every", 10)),
        checkpoint_every=int(spec.get("checkpoint_every", 0)),
        sample_every=int(spec.get("sample_every", 0)),
        owner=str(spec.get("owner") or ""),
        release_ref=str(spec.get("release_ref") or ""),
        hyperparams=spec.get("hyperparams") or {},
    )

    try:
        plugin = load_trainer_plugin(cfg.trainer_import)
    except Exception as exc:
        category, safe_message, debug = _classify_runtime_error(exc, phase="model-load")
        logger.exception("trainer.runtime.%s debug=%s", category, debug)
        reporter.failed(step=0, error=f"{category}:{safe_message}")
        raise

    if _already_completed_from_final(cfg.artifacts.checkpoints_dir, spec):
        final_path = str(Path(cfg.artifacts.checkpoints_dir) / "final.json")
        reporter.started(run_id=job.run_id)
        reporter.completed(step=int(job.max_steps), final_checkpoint=final_path)
        return 0

    batches = _resolve_batches(spec)
    sample_requests, sample_fixed_seed = _parse_sample_requests(spec)
    writer = LocalArtifactWriter(
        checkpoints_dir=cfg.artifacts.checkpoints_dir,
        samples_dir=cfg.artifacts.samples_dir,
        sample_requests=sample_requests,
        sample_fixed_seed=sample_fixed_seed,
    )
    start_step, resume_checkpoint_path = _detect_resume_checkpoint(cfg.artifacts.checkpoints_dir, spec)
    resume_state_payload = _read_resume_state_payload(resume_checkpoint_path) if resume_checkpoint_path else None

    uploader = None
    if cfg.upload_endpoints.enabled():
        uploader = JsonHttpArtifactUploader(
            run_id=job.run_id,
            token=cfg.capability_token,
            endpoints=cfg.upload_endpoints,
            tensorhub_url=os.getenv("TENSORHUB_URL"),
            owner=job.owner,
        )

    ctx = StepContext(
        job=job,
        dataset=batches,
        is_canceled=reporter.is_canceled,
        model_handles=model_handles,
    )
    if start_step > 0:
        reporter.metric(name="trainer/resumed_from_step", value=float(start_step), step=start_step)

    try:
        terminal_step = run_training_loop(
            job=job,
            ctx=ctx,
            trainer=plugin,
            batches=batches,
            reporter=reporter,
            artifact_writer=writer,
            uploader=uploader,
            start_step=start_step,
            resume_state_payload=resume_state_payload,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    except Exception as exc:
        category, safe_message, debug = _classify_runtime_error(exc if isinstance(exc, Exception) else Exception(str(exc)), phase="train-loop")
        logger.exception("trainer.runtime.%s debug=%s", category, debug)
        if uploader is not None and hasattr(uploader, "upload_terminal"):
            try:
                uploader.upload_terminal(
                    status="failed",
                    step=getattr(reporter, "failed_step", 0) or 0,
                    final_checkpoint=getattr(reporter, "final_checkpoint", None),
                    error=f"{category}:{safe_message}",
                )
            except Exception:
                logger.exception("trainer.runtime.upload terminal failure emission failed")
        raise

    if uploader is not None:
        try:
            final_path = writer.last_final_checkpoint_path
            if final_path:
                uploader.upload_checkpoint(local_path=final_path, step=int(terminal_step))
            if hasattr(uploader, "upload_terminal"):
                uploader.upload_terminal(
                    status="completed",
                    step=int(terminal_step),
                    final_checkpoint=final_path,
                    error="",
                )
        except Exception as exc:
            category, safe_message, debug = _classify_runtime_error(exc if isinstance(exc, Exception) else Exception(str(exc)), phase="upload")
            logger.exception("trainer.runtime.%s debug=%s", category, debug)
            reporter.failed(step=int(terminal_step), error=f"{category}:{safe_message}")
            raise
    return 0


__all__ = ["run_training_runtime_from_env"]
