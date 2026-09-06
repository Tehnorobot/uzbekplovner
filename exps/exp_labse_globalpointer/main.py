"""Runner stages for the LaBSE EfficientGlobalPointer experiment."""

import shutil
from pathlib import Path
from typing import Any

from ner_core.contracts import read_jsonl, write_jsonl
from ner_core.runner import ExperimentContext

from .data import validate_window
from .model import load_model_bundle
from .predict import predict_records
from .train import resolve_device


def _model_dir(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Return the current model directory or explain how to create it."""

    model_dir = context.path(config, "model_dir")
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"model directory does not exist: {model_dir}; run the train stage first"
        )
    return model_dir


class GlobalPointerPredictor:
    """Load the local-only model bundle once and predict complete request batches."""

    def __init__(
        self,
        config: dict[str, Any],
        context: ExperimentContext,
        *,
        device_name: str,
    ) -> None:
        self.device = resolve_device(device_name)
        self.model, self.tokenizer, stored = load_model_bundle(
            _model_dir(config, context),
            self.device,
        )
        self.max_length = int(stored["max_length"])
        self.stride = int(stored["stride"])
        self.threshold = float(stored["span_threshold"])
        self.batch_size = int(config["inference"].get("batch_size", 1))
        validate_window(self.tokenizer, self.max_length, self.stride)

    def __call__(self, records: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Predict exact spans for records while preserving their order and hashes."""

        return predict_records(
            self.model,
            self.tokenizer,
            records,
            max_length=self.max_length,
            stride=self.stride,
            batch_size=self.batch_size,
            threshold=self.threshold,
            device=self.device,
        )


def build_predictor(
    config: dict[str, Any],
    context: ExperimentContext,
) -> GlobalPointerPredictor:
    """Construct the contract-compatible service predictor."""

    return GlobalPointerPredictor(
        config,
        context,
        device_name=str(config["service"].get("device", "auto")),
    )


def train(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Train the experiment and update its current local model bundle."""

    from .train import run

    model_dir = run(config, context)
    current_model_dir = context.path(config, "model_dir")
    current_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, current_model_dir, dirs_exist_ok=True)
    print(f"Run artifacts: {context.run_dir}")
    print(f"Current model: {current_model_dir}")
    return current_model_dir


def predict(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Run local-only inference and write a contract-compatible JSONL file."""

    predictor = GlobalPointerPredictor(
        config,
        context,
        device_name=str(config["inference"].get("device", "auto")),
    )
    input_path = context.path(config, "inference_input")
    output_value = config["inference"].get("output")
    output_path = (
        Path(output_value).expanduser()
        if output_value
        else (context.run_dir or context.experiment_dir / "artifacts") / "prediction.jsonl"
    )
    if not output_path.is_absolute():
        output_path = context.root_dir / output_path
    predictions = predictor(read_jsonl(input_path, require_entities=False))
    write_jsonl(output_path, predictions)
    print(f"Predictions: {output_path}")
    return output_path


def evaluate(config: dict[str, Any], context: ExperimentContext) -> Path:
    """Predict dev data and calculate official exact-span metrics in one stage."""

    from scripts.evaluate import evaluate_files, print_metrics, write_metrics

    predictor = GlobalPointerPredictor(
        config,
        context,
        device_name=str(config["inference"].get("device", "auto")),
    )
    gold_path = context.path(config, "dev")
    records = read_jsonl(gold_path, require_entities=False)
    prediction_path = (context.run_dir or context.experiment_dir / "artifacts") / (
        "prediction.jsonl"
    )
    write_jsonl(prediction_path, predictor(records))
    metrics = evaluate_files(gold_path, prediction_path)
    metrics_path = (context.run_dir or context.experiment_dir / "artifacts") / (
        "dev_metrics.json"
    )
    write_metrics(metrics_path, metrics)
    print_metrics(metrics)
    print(f"Predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")
    return metrics_path


def serve(config: dict[str, Any], context: ExperimentContext) -> None:
    """Start the required health and prediction HTTP endpoints."""

    import uvicorn

    from ner_core.service import create_app

    service = config["service"]
    app = create_app(
        build_predictor(config, context),
        normalization_aliases=config.get("normalization", {}).get("aliases", {}),
    )
    uvicorn.run(
        app,
        host=str(service.get("host", "0.0.0.0")),
        port=int(service.get("port", 8000)),
        log_level="info",
    )
