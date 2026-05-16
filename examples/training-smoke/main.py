from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, cast

from gen_worker.trainer import (
    StepContext,
    StepResult,
    build_default_adamw_bundle,
    load_trainable_module_checkpoint,
    save_trainable_module_checkpoint,
    seed_everything,
    to_float_scalar,
)


DEFAULT_MODEL_NAME = "hf-internal-testing/tiny-random-BertForSequenceClassification"


class RecordBatchLike(Protocol):
    def to_pydict(self) -> dict[str, list[object]]:
        ...


@dataclass(frozen=True)
class TextBatch:
    inputs: dict[str, Any]
    batch_size: int


@dataclass
class TextFineTuneState:
    model: Any
    tokenizer: Any
    optimizer: Any
    scheduler: Any | None
    step: int
    seed: int
    learning_rate: float
    best_loss: float | None
    model_name_or_path: str
    num_labels: int
    max_length: int
    device: str
    dtype: str
    uses_huggingface: bool

    def generate_sample(self, prompt: str, **_: object) -> Mapping[str, object]:
        torch = _require_torch()
        self.model.eval()
        inputs = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for (name, value) in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
        logits = getattr(output, "logits", output)
        prediction = int(logits.argmax(dim=-1).detach().cpu()[0])
        return {"prompt": prompt, "prediction": prediction, "step": self.step}


class TextClassificationFineTuner:
    """
    Fine-tune a text classifier with a native PyTorch loop.

    The syntax intentionally mirrors the common Hugging Face pattern:
    tokenizer -> batch dict -> model(**batch) -> loss.backward() -> optimizer.step().
    TensorHub/python-gen-worker owns lifecycle, cadence, uploads, cancellation,
    and resume detection around these endpoint-owned hooks.
    """

    def setup(self, ctx: StepContext) -> None:
        seed_everything(int(_hp(ctx).get("seed", 42)))

    def configure(self, ctx: StepContext) -> TextFineTuneState:
        torch = _require_torch()
        hp = _hp(ctx)
        seed = int(hp.get("seed", 42))
        num_labels = int(hp.get("num_labels", 2))
        max_length = int(hp.get("max_length", 128))
        model_name_or_path = str(hp.get("model_name_or_path") or DEFAULT_MODEL_NAME).strip()
        device = _resolve_device(str(hp.get("device", "auto")))
        dtype = str(hp.get("dtype", "fp32")).strip().lower()

        model, tokenizer, uses_huggingface = _load_model_and_tokenizer(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
        )
        model = _move_model(model=model, device=device, dtype=dtype)
        model.train()

        trainable_params = [p for p in model.parameters() if bool(getattr(p, "requires_grad", False))]
        if not trainable_params:
            raise ValueError("no trainable parameters; for LoRA/PEFT, make sure an adapter is attached and active")

        optimizer_hp = dict(hp)
        optimizer_hp.setdefault("max_steps", ctx.job.max_steps)
        bundle = build_default_adamw_bundle(trainable_params, hyperparams=optimizer_hp)
        if bundle.optimizer is None:
            raise RuntimeError("failed to build optimizer; install torch in the training image")

        return TextFineTuneState(
            model=model,
            tokenizer=tokenizer,
            optimizer=bundle.optimizer,
            scheduler=bundle.scheduler,
            step=0,
            seed=seed,
            learning_rate=float(hp.get("learning_rate", hp.get("lr", 2e-5))),
            best_loss=None,
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            max_length=max_length,
            device=device,
            dtype=dtype,
            uses_huggingface=uses_huggingface,
        )

    def prepare_batch(self, raw_batch: Any, state: TextFineTuneState, ctx: StepContext) -> TextBatch:
        _ = ctx
        cols = _batch_to_columns(raw_batch)
        texts = _string_column(cols, "text")
        labels = _int_column(cols, "label")
        if len(texts) != len(labels):
            raise ValueError(f"text batch size {len(texts)} does not match label batch size {len(labels)}")

        inputs = state.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=state.max_length,
            return_tensors="pt",
        )
        torch = _require_torch()
        inputs = {name: value.to(state.device) for (name, value) in inputs.items()}
        inputs["labels"] = torch.tensor(labels, dtype=torch.long, device=state.device)
        return TextBatch(inputs=inputs, batch_size=len(texts))

    def train_step(self, batch: TextBatch, state: TextFineTuneState, ctx: StepContext) -> StepResult:
        torch = _require_torch()
        hp = _hp(ctx)
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)

        output = state.model(**batch.inputs)
        loss = getattr(output, "loss", None)
        if loss is None:
            logits = getattr(output, "logits", output)
            loss = torch.nn.functional.cross_entropy(logits, batch.inputs["labels"])

        loss.backward()
        max_grad_norm = float(hp.get("max_grad_norm", 1.0) or 0.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_grad_norm)
        state.optimizer.step()
        if state.scheduler is not None:
            state.scheduler.step()

        state.step += 1
        loss_value = to_float_scalar(loss)
        state.best_loss = loss_value if state.best_loss is None else min(state.best_loss, loss_value)
        lr = float(state.optimizer.param_groups[0].get("lr", state.learning_rate))
        return StepResult(
            metrics={
                "train/loss": loss_value,
                "train/lr": lr,
                "train/best_loss": float(state.best_loss),
            },
            debug={
                "step": state.step,
                "batch_size": batch.batch_size,
                "model_name_or_path": state.model_name_or_path,
                "uses_huggingface": state.uses_huggingface,
            },
        )

    def state_dict(self, state: TextFineTuneState) -> dict[str, object]:
        return {
            "step": state.step,
            "seed": state.seed,
            "learning_rate": state.learning_rate,
            "best_loss": state.best_loss,
            "model_name_or_path": state.model_name_or_path,
            "num_labels": state.num_labels,
            "max_length": state.max_length,
            "device": state.device,
            "dtype": state.dtype,
            "uses_huggingface": state.uses_huggingface,
        }

    def load_state_dict(self, state: TextFineTuneState, payload: dict[str, object], ctx: StepContext) -> None:
        _ = ctx
        state.step = int(payload.get("step", state.step))
        state.seed = int(payload.get("seed", state.seed))
        state.learning_rate = float(payload.get("learning_rate", state.learning_rate))
        raw_best = payload.get("best_loss", state.best_loss)
        state.best_loss = None if raw_best is None else float(raw_best)
        state.model_name_or_path = str(payload.get("model_name_or_path", state.model_name_or_path))
        state.num_labels = int(payload.get("num_labels", state.num_labels))
        state.max_length = int(payload.get("max_length", state.max_length))
        state.device = str(payload.get("device", state.device))
        state.dtype = str(payload.get("dtype", state.dtype))
        state.uses_huggingface = bool(payload.get("uses_huggingface", state.uses_huggingface))

    def save_checkpoint(
        self,
        *,
        state: TextFineTuneState,
        step: int,
        output_dir: str,
        final: bool,
        ctx: StepContext,
    ) -> Mapping[str, object]:
        _ = ctx
        if state.uses_huggingface and callable(getattr(state.model, "save_pretrained", None)):
            out = Path(output_dir)
            state.model.save_pretrained(out / "adapter-or-model")
            if callable(getattr(state.tokenizer, "save_pretrained", None)):
                state.tokenizer.save_pretrained(out / "tokenizer")

        return save_trainable_module_checkpoint(
            module=state.model,
            optimizer=state.optimizer,
            output_dir=output_dir,
            step=step,
            final=final,
            model_name_or_path=state.model_name_or_path,
            metadata={
                "task": "text-classification",
                "num_labels": state.num_labels,
                "max_length": state.max_length,
                "best_loss": state.best_loss,
                "uses_huggingface": state.uses_huggingface,
            },
        )

    def load_checkpoint(
        self,
        *,
        state: TextFineTuneState,
        checkpoint_dir: str,
        payload: Mapping[str, object],
        ctx: StepContext,
    ) -> None:
        _ = payload
        _ = ctx
        load_trainable_module_checkpoint(
            module=state.model,
            optimizer=state.optimizer,
            checkpoint_dir=checkpoint_dir,
        )


def _load_model_and_tokenizer(*, model_name_or_path: str, num_labels: int) -> tuple[Any, Any, bool]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
        )
        return model, tokenizer, True
    except Exception as exc:
        raise RuntimeError(
            "failed to load Hugging Face text-classification model; install transformers "
            f"and verify model_name_or_path={model_name_or_path!r}"
        ) from exc


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("this training endpoint requires torch") from exc
    return torch


def _hp(ctx: StepContext) -> dict[str, object]:
    return {str(k): v for (k, v) in dict(getattr(ctx.job, "hyperparams", {}) or {}).items()}


def _resolve_device(requested: str) -> str:
    torch = _require_torch()
    value = requested.strip().lower()
    if value in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return value


def _move_model(*, model: Any, device: str, dtype: str) -> Any:
    torch = _require_torch()
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    target_dtype = dtype_map.get(dtype)
    if target_dtype is None:
        raise ValueError("hyperparams.dtype must be fp32, bf16, or fp16")
    if device == "cpu" and target_dtype in {torch.float16, torch.bfloat16}:
        target_dtype = torch.float32
    return model.to(device=device, dtype=target_dtype)


def _batch_to_columns(raw_batch: Any) -> dict[str, object]:
    if hasattr(raw_batch, "to_pydict"):
        return cast(RecordBatchLike, raw_batch).to_pydict()
    if isinstance(raw_batch, Mapping):
        return {str(k): v for (k, v) in raw_batch.items()}
    raise TypeError("raw training batch must be a dict or pyarrow RecordBatch-like object")


def _string_column(cols: Mapping[str, object], name: str) -> list[str]:
    raw = cols.get(name)
    if not isinstance(raw, list):
        raise ValueError(f"training batch must include list column {name!r}")
    return [str(item) for item in raw]


def _int_column(cols: Mapping[str, object], name: str) -> list[int]:
    raw = cols.get(name)
    if not isinstance(raw, list):
        raise ValueError(f"training batch must include list column {name!r}")
    return [int(item) for item in raw]
