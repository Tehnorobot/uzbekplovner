"""Stages for the official Transformer baseline experiment."""

import json
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any

from ner_core.contracts import read_jsonl, write_jsonl
from ner_core.runner import ExperimentContext


def _path(context: ExperimentContext, config: dict[str, Any], name: str) -> Path:
    return context.path(config, name)


def _train_args(config: dict[str, Any], context: ExperimentContext) -> Namespace:
    """Translate the experiment YAML into the official baseline CLI namespace."""

    training = config["training"]
    return Namespace(
        train=_path(context, config, "train"),
        dev=_path(context, config, "dev"),
        output_dir=context.run_dir,
        model_name=config["model"]["name"],
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        max_grad_norm=float(training["max_grad_norm"]),
        max_length=int(training["max_length"]),
        stride=int(training["stride"]),
        seed=int(training["seed"]),
        device=str(training["device"]),
        num_workers=int(training["num_workers"]),
        max_train_records=None,
        max_dev_records=None,
        # The common runner writes run metadata before the training stage starts.
        overwrite_output_dir=True,
        resume=(
            (context.root_dir / Path(training["resume"])).resolve()
            if training.get("resume") and not Path(training["resume"]).is_absolute()
            else Path(training["resume"]).expanduser().resolve()
            if training.get("resume")
            else None
        ),
        augmentation={
            "enabled": bool(config.get("augmentation", {}).get("enabled", False)),
            "probability": float(config.get("augmentation", {}).get("probability", 0.0)),
            "num_clusters": int(config.get("augmentation", {}).get("num_clusters", 8)),
            "max_candidates": int(config.get("augmentation", {}).get("max_candidates", 5)),
        },
    )


def train(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Train the baseline and update the experiment's current model."""

    from .train import run as train_baseline

    if context.run_dir is None:
        raise ValueError("training requires an active run directory")
    model_dir = train_baseline(_train_args(config, context))
    current_model_dir = _path(context, config, "model_dir")
    current_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, current_model_dir, dirs_exist_ok=True)
    print(f"Run artifacts: {context.run_dir}")
    print(f"Current model: {current_model_dir}")
    return current_model_dir


def _model_dir(config: dict[str, Any], context: ExperimentContext) -> Path:
    model_dir = _path(context, config, "model_dir")
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"model directory does not exist: {model_dir}; run the train stage first"
        )
    return model_dir


class BaselinePredictor:
    """Load the current experiment model once and predict complete batches."""

    def __init__(self, config: dict[str, Any], context: ExperimentContext) -> None:
        from transformers import AutoModelForTokenClassification

        from .common import load_fast_tokenizer, resolve_device

        model_dir = _model_dir(config, context)
        service = config["service"]
        self.device = resolve_device(str(service.get("device", "auto")))
        self.tokenizer = load_fast_tokenizer(str(model_dir), local_files_only=True)
        model_config = model_dir / "baseline_config.json"
        stored = (
            json.loads(model_config.read_text(encoding="utf-8")) if model_config.exists() else {}
        )
        self.max_length = int(stored.get("max_length", config["training"]["max_length"]))
        self.stride = int(stored.get("stride", config["training"]["stride"]))
        self.batch_size = int(config["inference"].get("batch_size", 16))
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        ).to(self.device)

    def __call__(self, records: list[dict[str, str]]) -> list[dict[str, Any]]:
        from . import predict as baseline_predict

        windows = baseline_predict._build_windows(
            records,
            self.tokenizer,
            max_length=self.max_length,
            stride=self.stride,
        )
        scores = baseline_predict._predict_token_scores(
            self.model,
            self.tokenizer,
            windows,
            len(records),
            batch_size=self.batch_size,
            device=self.device,
        )
        labels = baseline_predict._model_labels(self.model)
        return baseline_predict._decode_records(records, scores, labels)


def build_predictor(config: dict[str, Any], context: ExperimentContext) -> BaselinePredictor:
    """Construct a predictor with local-only model loading."""

    return BaselinePredictor(config, context)


def predict(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Run inference with the current experiment model on a JSONL file."""

    predictor = build_predictor(config, context)
    input_path = _path(context, config, "inference_input")
    output_value = config["inference"].get("output")
    output_path = (
        Path(output_value).expanduser()
        if output_value
        else (context.run_dir or context.experiment_dir / "artifacts") / "dev_predictions.jsonl"
    )
    if not output_path.is_absolute():
        output_path = context.root_dir / output_path
    predictions = predictor(read_jsonl(input_path, require_entities=False))
    write_jsonl(output_path, predictions)
    print(f"Predictions: {output_path}")
    return output_path


def evaluate(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Calculate official exact-span metrics for the configured predictions."""

    from scripts.evaluate import evaluate_files, print_metrics, write_metrics

    gold_path = _path(context, config, "dev")
    prediction_value = config["inference"].get("output")
    prediction_path = (
        Path(prediction_value).expanduser()
        if prediction_value
        else (context.run_dir or context.experiment_dir / "artifacts") / "dev_predictions.jsonl"
    )
    if not prediction_path.is_absolute():
        prediction_path = context.root_dir / prediction_path
    metrics_path = (context.run_dir or context.experiment_dir / "artifacts") / "dev_metrics.json"
    metrics = evaluate_files(gold_path, prediction_path)
    print_metrics(metrics)
    write_metrics(metrics_path, metrics)
    print(f"Metrics: {metrics_path}")
    return metrics_path


def serve(config: dict[str, Any], context: ExperimentContext) -> None:
    """Start the contract-compatible HTTP service."""

    import uvicorn

    from ner_core.service import create_app

    predictor = build_predictor(config, context)
    service = config["service"]
    app = create_app(
        predictor,
        normalization_aliases=config.get("normalization", {}).get("aliases", {}),
    )
    uvicorn.run(
        app,
        host=str(service.get("host", "0.0.0.0")),
        port=int(service.get("port", 8000)),
        log_level="info",
    )
