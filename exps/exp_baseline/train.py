import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from ner_core.contracts import write_jsonl

from .cluster_augmentation import apply_cluster_augmentation
from .common import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_STRIDE,
    TAGS,
    TokenizedNerDataset,
    load_fast_tokenizer,
    read_records,
    resolve_device,
    set_seed,
    validate_window,
)


def parse_args() -> argparse.Namespace:
    """Разбирает параметры обучения baseline."""

    parser = argparse.ArgumentParser(description="Train a minimal Transformer NER baseline.")
    parser.add_argument("--train", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("dev.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exps/exp_baseline/artifacts/runs/standalone"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Отсекает некорректные числовые параметры до загрузки модели."""

    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "gradient-accumulation-steps": args.gradient_accumulation_steps,
        "max-length": args.max_length,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    optional_positive = {
        "max-train-records": args.max_train_records,
        "max-dev-records": args.max_dev_records,
    }
    for name, value in optional_positive.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    augmentation = getattr(args, "augmentation", {})
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation must be a mapping")
    probability = float(augmentation.get("probability", 0.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation probability must be in [0, 1]")
    if int(augmentation.get("num_clusters", 8)) < 1:
        raise ValueError("augmentation num_clusters must be positive")
    if int(augmentation.get("max_candidates", 5)) < 1:
        raise ValueError("augmentation max_candidates must be positive")


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Создаёт пустой каталог запуска или требует явное разрешение перезаписи."""

    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(
            f"output directory is not empty: {path}; use --overwrite-output-dir to reuse it"
        )
    path.mkdir(parents=True, exist_ok=True)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Переносит все тензоры batch на выбранное устройство."""

    return {key: value.to(device) for key, value in batch.items()}


def _loss_weight(batch: dict[str, torch.Tensor]) -> int:
    """Возвращает число непустых token labels для усреднения loss."""

    return int((batch["labels"] != -100).sum().item())


@torch.inference_mode()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Считает средний dev loss с весом по числу размеченных токенов."""

    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for batch in tqdm(loader, desc="Dev loss", unit="batch", leave=False):
        batch = _move_batch(batch, device)
        output = model(**batch)
        weight = _loss_weight(batch)
        weighted_loss += float(output.loss.item()) * weight
        token_count += weight
    if not token_count:
        raise RuntimeError("dev dataset contains no labeled tokens")
    return weighted_loss / token_count


@torch.inference_mode()
def evaluate_entity_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Calculate exact-span metrics for one split using the current model."""

    from scripts.evaluate import calculate_metrics

    from . import predict as baseline_predict

    windows = baseline_predict._build_windows(
        records,
        tokenizer,
        max_length=max_length,
        stride=stride,
    )
    scores = baseline_predict._predict_token_scores(
        model,
        tokenizer,
        windows,
        len(records),
        batch_size=batch_size,
        device=device,
    )
    id2label = baseline_predict._model_labels(model)
    predictions = baseline_predict._decode_records(records, scores, id2label)

    gold = {
        record["hash"]: {
            "text": record["text"],
            "entities": {
                (entity["label"], entity["start"], entity["end"]) for entity in record["entities"]
            },
        }
        for record in records
    }
    predicted = {
        record["hash"]: {
            (entity["label"], entity["start"], entity["end"]) for entity in record["entities"]
        }
        for record in predictions
    }
    return calculate_metrics(gold, predicted)


def print_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_metrics: dict[str, Any],
    val_loss: float,
    val_metrics: dict[str, Any],
) -> None:
    """Print loss and complete exact-span metrics for both data splits."""

    from scripts.evaluate import print_metrics

    print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    print("Train exact-span metrics:")
    print_metrics(train_metrics)
    print("Validation exact-span metrics:")
    print_metrics(val_metrics)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    *,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    """Выполняет одну эпоху обучения и возвращает средний token loss."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    token_count = 0

    progress = tqdm(loader, desc="Train", unit="batch", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        batch = _move_batch(batch, device)
        output = model(**batch)
        loss = output.loss
        group_start = ((batch_index - 1) // gradient_accumulation_steps) * (
            gradient_accumulation_steps
        )
        group_size = min(gradient_accumulation_steps, len(loader) - group_start)
        (loss / group_size).backward()

        weight = _loss_weight(batch)
        weighted_loss += float(loss.detach().item()) * weight
        token_count += weight
        should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader)
        if should_step:
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    if not token_count:
        raise RuntimeError("train dataset contains no labeled tokens")
    return weighted_loss / token_count


def _save_model(
    model: torch.nn.Module,
    tokenizer: Any,
    model_dir: Path,
    config: dict[str, Any],
) -> None:
    """Сохраняет веса, tokenizer и параметры оконного инференса."""

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "baseline_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    generator: torch.Generator,
    epoch: int,
    best_dev_loss: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect all state needed to continue training deterministically."""

    state: dict[str, Any] = {
        "epoch": epoch,
        "best_dev_loss": best_dev_loss,
        "history": history,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "data_generator": generator.get_state(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def run(args: argparse.Namespace) -> Path:
    """Обучает baseline и сохраняет checkpoint с минимальным dev loss."""

    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite_output_dir)
    device = resolve_device(args.device)
    set_seed(args.seed, seed_cuda=device.type == "cuda")

    train_records = read_records(
        args.train.expanduser().resolve(),
        require_entities=True,
        limit=args.max_train_records,
    )
    dev_records = read_records(
        args.dev.expanduser().resolve(),
        require_entities=True,
        limit=args.max_dev_records,
    )
    tokenizer = load_fast_tokenizer(args.model_name)
    validate_window(tokenizer, args.max_length, args.stride)

    id2label = dict(enumerate(TAGS))
    label2id = {tag: index for index, tag in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(TAGS),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    input_train_records = len(train_records)
    augmentation_config = getattr(args, "augmentation", {})
    train_records, augmentation_stats = apply_cluster_augmentation(
        train_records,
        tokenizer,
        model,
        enabled=bool(augmentation_config.get("enabled", False)),
        probability=float(augmentation_config.get("probability", 0.0)),
        num_clusters=int(augmentation_config.get("num_clusters", 8)),
        max_candidates=int(augmentation_config.get("max_candidates", 5)),
        seed=args.seed,
    )
    if augmentation_stats["added_records"]:
        write_jsonl(output_dir / "augmented_train.jsonl", train_records)
        print(
            "Cluster augmentation: "
            f"added={augmentation_stats['added_records']}, "
            f"replaced_entities={augmentation_stats['replaced_entities']}"
        )

    train_dataset = TokenizedNerDataset(
        train_records,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        description="Tokenize train",
    )
    dev_dataset = TokenizedNerDataset(
        dev_records,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        description="Tokenize dev",
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_dev_loss = float("inf")
    if args.resume is not None:
        checkpoint_path = args.resume.expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["data_generator"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if "cuda_rng" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dev_loss = float(checkpoint["best_dev_loss"])
        history = list(checkpoint["history"])
        print(f"Resumed from: {checkpoint_path} (next epoch: {start_epoch})")

    print(f"Device: {device}")
    print(f"Train: {len(train_records)} documents, {len(train_dataset)} windows")
    print(f"Dev: {len(dev_records)} documents, {len(dev_dataset)} windows")

    model_dir = output_dir / "model"
    baseline_config = {
        "schema_version": 1,
        "base_model": args.model_name,
        "tags": list(TAGS),
        "max_length": args.max_length,
        "stride": args.stride,
        "seed": args.seed,
        "augmentation": {
            "enabled": bool(augmentation_config.get("enabled", False)),
            "probability": float(augmentation_config.get("probability", 0.0)),
            "num_clusters": int(augmentation_config.get("num_clusters", 8)),
            "max_candidates": int(augmentation_config.get("max_candidates", 5)),
        },
    }
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        dev_loss = evaluate_loss(model, dev_loader, device)
        train_metrics = evaluate_entity_metrics(
            model,
            tokenizer,
            train_records,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
        )
        val_metrics = evaluate_entity_metrics(
            model,
            tokenizer,
            dev_records,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
        )
        epoch_result = {
            "epoch": epoch,
            "train": {"loss": train_loss, "exact_span": train_metrics},
            "val": {"loss": dev_loss, "exact_span": val_metrics},
            "train_loss": train_loss,
            "dev_loss": dev_loss,
        }
        history.append(epoch_result)
        print_epoch_metrics(epoch, train_loss, train_metrics, dev_loss, val_metrics)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            _save_model(model, tokenizer, model_dir, baseline_config)
        torch.save(
            _checkpoint_state(
                model,
                optimizer,
                scheduler,
                generator,
                epoch,
                best_dev_loss,
                history,
            ),
            output_dir / "checkpoint.pt",
        )
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

    if not model_dir.exists():
        _save_model(model, tokenizer, model_dir, baseline_config)

    run_summary = {
        **baseline_config,
        "input_train_records": input_train_records,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_windows": len(train_dataset),
        "dev_windows": len(dev_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "best_dev_loss": best_dev_loss,
        "augmentation_stats": augmentation_stats,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Best model: {model_dir}")
    return model_dir


def main() -> int:
    """Запускает CLI обучения с компактным сообщением об ошибке."""

    try:
        run(parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
