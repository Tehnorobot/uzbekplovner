"""Training loop for LaBSE with EfficientGlobalPointer."""

import gc
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from ner_core.contracts import read_jsonl, write_jsonl
from ner_core.runner import ExperimentContext
from scripts.evaluate import calculate_metrics, print_metrics, write_metrics

from .augmentation import apply_cluster_augmentation
from .data import GlobalPointerCollator, GlobalPointerDataset, validate_window
from .model import (
    LABELS,
    LaBSEEfficientGlobalPointer,
    global_pointer_loss,
    load_model_bundle,
    save_model_bundle,
)
from .predict import predict_records

JsonObject = dict[str, Any]


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and reject an unavailable explicitly requested CUDA device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def set_seed(seed: int, *, seed_cuda: bool) -> None:
    """Seed Python and PyTorch generators for reproducible training."""

    random.seed(seed)  # noqa: S311 - deterministic ML training, not cryptography
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move all tensors in a collated batch to the training device."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _model_logits(
    model: LaBSEEfficientGlobalPointer,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Call the model with only encoder and span-mask inputs."""

    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        span_mask=batch["span_mask"],
    )


def _autocast(device: torch.device, enabled: bool) -> torch.amp.autocast:
    """Create an autocast context for the active device."""

    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def train_epoch(
    model: LaBSEEfficientGlobalPointer,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    amp_enabled: bool,
) -> float:
    """Train one epoch and return mean batch loss."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    progress = tqdm(loader, desc="Train", unit="batch", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        batch = _move_batch(batch, device)
        with _autocast(device, amp_enabled):
            logits = _model_logits(model, batch)
            loss = global_pointer_loss(batch["span_labels"], logits)
            group_start = ((batch_index - 1) // gradient_accumulation_steps) * (
                gradient_accumulation_steps
            )
            group_size = min(gradient_accumulation_steps, len(loader) - group_start)
            scaled_loss = loss / group_size
        scaler.scale(scaled_loss).backward()

        should_update = batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().item())
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate_loss(
    model: LaBSEEfficientGlobalPointer,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> float:
    """Calculate mean GlobalPointer loss for a tokenized split."""

    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Validation loss", unit="batch", leave=False):
        batch = _move_batch(batch, device)
        with _autocast(device, amp_enabled):
            logits = _model_logits(model, batch)
            loss = global_pointer_loss(batch["span_labels"], logits)
        total_loss += float(loss.item())
    return total_loss / len(loader)


def exact_span_metrics(
    gold_records: list[JsonObject],
    predictions: list[JsonObject],
) -> JsonObject:
    """Calculate official per-label, micro and macro exact-span metrics."""

    gold = {
        record["hash"]: {
            "text": record["text"],
            "entities": {
                (entity["label"], entity["start"], entity["end"]) for entity in record["entities"]
            },
        }
        for record in gold_records
    }
    predicted = {
        record["hash"]: {
            (entity["label"], entity["start"], entity["end"]) for entity in record["entities"]
        }
        for record in predictions
    }
    return calculate_metrics(gold, predicted)


def _print_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_metrics: JsonObject | None,
    dev_loss: float,
    dev_metrics: JsonObject,
) -> None:
    """Print loss and complete exact-span metrics after one epoch."""

    print(f"Epoch {epoch}: train_loss={train_loss:.6f}, dev_loss={dev_loss:.6f}")
    if train_metrics is not None:
        print("Train exact-span metrics:")
        print_metrics(train_metrics)
    print("Validation exact-span metrics:")
    print_metrics(dev_metrics)


def _write_history(output_dir: Path, history: list[JsonObject]) -> None:
    """Persist the aggregated train/dev history after every epoch."""

    (output_dir / "metrics_history.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matching": "same hash and exact label/start/end",
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _checkpoint_state(
    model: LaBSEEfficientGlobalPointer,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    *,
    epoch: int,
    best_dev_f1: float,
    best_epoch: int,
    history: list[JsonObject],
) -> JsonObject:
    """Collect model, optimizer, scheduler, AMP and RNG state for resume."""

    state: JsonObject = {
        "epoch": epoch,
        "best_dev_f1": best_dev_f1,
        "best_epoch": best_epoch,
        "history": history,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "data_generator": generator.get_state(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def _resolve_resume_path(value: Any, root_dir: Path) -> Path | None:
    """Resolve an optional checkpoint path relative to the project root."""

    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _validate_settings(config: dict[str, Any]) -> None:
    """Validate experiment-only options not present in the common Pydantic schema."""

    pointer = config.get("global_pointer", {})
    head_size = int(pointer.get("head_size", 64))
    if head_size < 2 or head_size % 2:
        raise ValueError("global_pointer.head_size must be a positive even integer")
    batch_size = int(config["inference"].get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("inference.batch_size must be positive")


def run(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Fine-tune LaBSE and save the best dev-F1 model plus dev predictions."""

    if context.run_dir is None:
        raise ValueError("training requires an active run directory")
    _validate_settings(config)
    output_dir = context.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    augmentation = config.get("augmentation", {})
    pointer = config.get("global_pointer", {})
    seed = int(training["seed"])
    device = resolve_device(str(training["device"]))
    amp_enabled = bool(training.get("use_amp", True)) and device.type == "cuda"
    set_seed(seed, seed_cuda=device.type == "cuda")

    raw_train_records = read_jsonl(context.path(config, "train"), require_entities=True)
    dev_records = read_jsonl(context.path(config, "dev"), require_entities=True)
    model_name = str(config["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with offset_mapping support is required")
    max_length = int(training["max_length"])
    stride = int(training["stride"])
    validate_window(tokenizer, max_length, stride)

    encoder = AutoModel.from_pretrained(model_name)
    dropout_value = config["model"].get("dropout")
    model = LaBSEEfficientGlobalPointer(
        encoder,
        num_labels=len(LABELS),
        head_size=int(pointer.get("head_size", 64)),
        dropout=float(dropout_value) if dropout_value is not None else None,
    ).to(device)
    train_records, augmentation_stats = apply_cluster_augmentation(
        raw_train_records,
        tokenizer,
        model.encoder,
        enabled=bool(augmentation.get("enabled", False)),
        probability=float(augmentation.get("probability", 0.0)),
        num_clusters=int(augmentation.get("num_clusters", 8)),
        max_candidates=int(augmentation.get("max_candidates", 5)),
        seed=seed,
    )
    if augmentation_stats["added_records"]:
        write_jsonl(output_dir / "augmented_train.jsonl", train_records)

    train_dataset = GlobalPointerDataset(
        train_records,
        tokenizer,
        max_length=max_length,
        stride=stride,
        description="Tokenize train",
    )
    dev_dataset = GlobalPointerDataset(
        dev_records,
        tokenizer,
        max_length=max_length,
        stride=stride,
        description="Tokenize dev",
    )
    collator = GlobalPointerCollator(tokenizer, len(LABELS))
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation_steps = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(training["warmup_ratio"])),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    history: list[JsonObject] = []
    best_dev_f1 = -1.0
    best_epoch = 0
    resume_model_dir: Path | None = None
    resume_path = _resolve_resume_path(training.get("resume"), context.root_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        generator.set_state(checkpoint["data_generator"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if "cuda_rng" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dev_f1 = float(checkpoint["best_dev_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        history = list(checkpoint["history"])
        resume_model_dir = resume_path.parent / "model"
        print(f"Resumed from: {resume_path} (next epoch: {start_epoch})")

    inference_batch_size = int(config["inference"].get("batch_size", 1))
    threshold = float(pointer.get("threshold", 0.0))
    evaluate_train = bool(training.get("evaluate_train_each_epoch", True))
    model_dir = output_dir / "model"
    model_metadata: JsonObject = {
        "schema_version": 1,
        "architecture": "LaBSEEfficientGlobalPointer",
        "base_model": model_name,
        "labels": list(LABELS),
        "head_size": int(pointer.get("head_size", 64)),
        "dropout": float(model.dropout.p),
        "span_threshold": threshold,
        "max_length": max_length,
        "stride": stride,
    }
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Train: {len(train_records)} documents, {len(train_dataset)} windows")
    print(f"Dev: {len(dev_records)} documents, {len(dev_dataset)} windows")
    print(f"Cluster augmentation: {json.dumps(augmentation_stats, ensure_ascii=False)}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            gradient_accumulation_steps=accumulation_steps,
            max_grad_norm=float(training["max_grad_norm"]),
            amp_enabled=amp_enabled,
        )
        dev_loss = evaluate_loss(model, dev_loader, device, amp_enabled=amp_enabled)
        train_metrics = None
        if evaluate_train:
            train_predictions = predict_records(
                model,
                tokenizer,
                train_records,
                max_length=max_length,
                stride=stride,
                batch_size=inference_batch_size,
                threshold=threshold,
                device=device,
            )
            train_metrics = exact_span_metrics(train_records, train_predictions)
        dev_predictions = predict_records(
            model,
            tokenizer,
            dev_records,
            max_length=max_length,
            stride=stride,
            batch_size=inference_batch_size,
            threshold=threshold,
            device=device,
        )
        dev_metrics = exact_span_metrics(dev_records, dev_predictions)
        epoch_result: JsonObject = {
            "epoch": epoch,
            "train": {"loss": train_loss, "exact_span": train_metrics},
            "dev": {"loss": dev_loss, "exact_span": dev_metrics},
        }
        history.append(epoch_result)
        _print_epoch_metrics(epoch, train_loss, train_metrics, dev_loss, dev_metrics)
        _write_history(output_dir, history)

        current_dev_f1 = float(dev_metrics["micro"]["f1"])
        if current_dev_f1 > best_dev_f1:
            best_dev_f1 = current_dev_f1
            best_epoch = epoch
            save_model_bundle(
                model,
                tokenizer,
                model_dir,
                {
                    **model_metadata,
                    "best_epoch": best_epoch,
                    "best_dev_micro_f1": best_dev_f1,
                },
            )
            write_jsonl(output_dir / "best_epoch_prediction.jsonl", dev_predictions)
        torch.save(
            _checkpoint_state(
                model,
                optimizer,
                scheduler,
                scaler,
                generator,
                epoch=epoch,
                best_dev_f1=best_dev_f1,
                best_epoch=best_epoch,
                history=history,
            ),
            output_dir / "checkpoint.pt",
        )

    if not model_dir.is_dir():
        if resume_model_dir is None or not resume_model_dir.is_dir():
            raise RuntimeError("training produced no best-model artifacts")
        shutil.copytree(resume_model_dir, model_dir)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best_model, best_tokenizer, saved_metadata = load_model_bundle(model_dir, device)
    final_predictions = predict_records(
        best_model,
        best_tokenizer,
        dev_records,
        max_length=int(saved_metadata["max_length"]),
        stride=int(saved_metadata["stride"]),
        batch_size=inference_batch_size,
        threshold=float(saved_metadata["span_threshold"]),
        device=device,
    )
    final_metrics = exact_span_metrics(dev_records, final_predictions)
    write_jsonl(output_dir / "prediction.jsonl", final_predictions)
    write_metrics(output_dir / "dev_metrics.json", final_metrics)
    summary: JsonObject = {
        **model_metadata,
        "raw_train_records": len(raw_train_records),
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_windows": len(train_dataset),
        "dev_windows": len(dev_dataset),
        "partial_entity_train_windows": train_dataset.partial_entity_windows,
        "partial_entity_dev_windows": dev_dataset.partial_entity_windows,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "best_epoch": best_epoch,
        "best_dev_micro_f1": best_dev_f1,
        "augmentation_stats": augmentation_stats,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Best validation metrics:")
    print_metrics(final_metrics)
    print(f"Predictions: {output_dir / 'prediction.jsonl'}")
    print(f"Best model: {model_dir}")
    return model_dir
