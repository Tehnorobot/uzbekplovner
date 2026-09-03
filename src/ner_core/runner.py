"""Experiment discovery, configuration precedence and stage dispatch."""

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import new_run_dir, write_run_metadata
from .config import (
    ExperimentSettings,
)

EXP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class ExperimentContext:
    """Paths and runtime information available to an experiment stage."""

    root_dir: Path
    experiment_dir: Path
    run_dir: Path | None

    def path(self, config: dict[str, Any], name: str) -> Path:
        """Resolve a configured path relative to the project root."""

        value = config.get("paths", {}).get(name)
        if value is None:
            raise ValueError(f"missing paths.{name} in experiment config")
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root_dir / path


def validate_exp_name(name: str) -> str:
    """Validate an experiment name before using it in an import path."""

    if not EXP_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid experiment name: {name!r}")
    return name


def load_experiment_config(
    root_dir: Path,
    experiment_name: str,
    config_path: Path | None,
    overrides: list[str],
) -> tuple[dict[str, Any], Path]:
    """Load defaults, YAML, environment values and CLI overrides."""

    experiment_name = validate_exp_name(experiment_name)
    experiment_dir = root_dir / "exps" / experiment_name
    if not experiment_dir.is_dir():
        raise ValueError(f"experiment does not exist: {experiment_name}")

    selected_config = config_path or experiment_dir / "configs" / "default.yaml"
    if not selected_config.is_absolute():
        selected_config = root_dir / selected_config
    ExperimentSettings.yaml_path = selected_config
    ExperimentSettings.cli_overrides = overrides
    settings = ExperimentSettings(_env_file=root_dir / ".env")
    config = settings.model_dump(mode="python")
    config["experiment"]["name"] = experiment_name
    return config, selected_config


def dispatch(
    *,
    root_dir: Path,
    experiment_name: str,
    stage: str,
    config_path: Path | None,
    overrides: list[str],
    run_id: str | None,
    run_dir: Path | None,
) -> Any:
    """Resolve one experiment and execute its requested stage."""

    config, selected_config = load_experiment_config(
        root_dir,
        experiment_name,
        config_path,
        overrides,
    )
    experiment_dir = root_dir / "exps" / experiment_name
    active_run_dir: Path | None = None
    if stage in {"train", "predict", "evaluate"}:
        configured_runs_dir = Path(
            config.get("paths", {}).get("runs_dir", experiment_dir / "artifacts/runs")
        )
        if not configured_runs_dir.is_absolute():
            configured_runs_dir = root_dir / configured_runs_dir
        active_run_dir = (
            run_dir.expanduser().resolve()
            if run_dir is not None
            else new_run_dir(configured_runs_dir, run_id)
        )
        config["runtime"] = {"run_dir": str(active_run_dir), "config_file": str(selected_config)}
        write_run_metadata(
            active_run_dir,
            config,
            data_paths=[
                root_dir / str(config.get("paths", {}).get("train", "data/train.jsonl")),
                root_dir / str(config.get("paths", {}).get("dev", "data/dev.jsonl")),
            ],
        )

    module = importlib.import_module(f"exps.{validate_exp_name(experiment_name)}.main")
    handler = getattr(module, stage, None)
    if handler is None or not callable(handler):
        raise ValueError(f"experiment {experiment_name!r} does not implement stage {stage!r}")
    context = ExperimentContext(root_dir, experiment_dir, active_run_dir)
    return handler(config, context)
