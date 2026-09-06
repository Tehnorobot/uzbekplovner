"""Pydantic-backed YAML, environment and CLI configuration loader."""

from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class BaseConfig(BaseModel):
    """Common Pydantic configuration behavior for experiment sections."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class BootstrapSettings(BaseSettings):
    """Minimal settings used before the experiment name is selected."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    exp_name: str | None = Field(default=None, validation_alias="EXP_NAME")


class CliOverridesSource(PydanticBaseSettingsSource):
    """Pydantic settings source for the runner's `--set key=value` syntax."""

    def __init__(self, settings_cls: type[BaseSettings], overrides: list[str]) -> None:
        super().__init__(settings_cls)
        self.overrides = overrides

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Satisfy the source interface; values are returned in `__call__`."""

        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Convert CLI overrides into a nested mapping for Pydantic to merge."""

        result: dict[str, Any] = {}
        for override in self.overrides:
            if "=" not in override:
                raise ValueError(f"override must have KEY=VALUE syntax: {override!r}")
            key, raw_value = override.split("=", 1)
            set_dotted(result, key, parse_override(raw_value))
        return result


class ExperimentSection(BaseConfig):
    """Identity and version of one experiment."""

    name: str = "exp_baseline"
    version: str = "1.0"


class PathsConfig(BaseConfig):
    """Dataset, model and run locations."""

    train: Path = Path("data/train.jsonl")
    dev: Path = Path("data/dev.jsonl")
    inference_input: Path = Path("data/dev.jsonl")
    model_dir: Path = Path("exps/exp_baseline/artifacts/model")
    runs_dir: Path = Path("exps/exp_baseline/artifacts/runs")


class ModelConfig(BaseConfig):
    """Base model settings."""

    name: str = "distilbert/distilbert-base-multilingual-cased"


class TrainingConfig(BaseConfig):
    """Validated training hyperparameters."""

    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    max_length: int = Field(default=256, ge=1)
    stride: int = Field(default=64, ge=0)
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = Field(default=0, ge=0)
    resume: Path | None = None


class AugmentationConfig(BaseConfig):
    """Optional cluster-based training data augmentation settings."""

    enabled: bool = False
    probability: float = Field(default=0.0, ge=0, le=1)
    num_clusters: int = Field(default=8, ge=1)
    max_candidates: int = Field(default=5, ge=1)


class InferenceConfig(BaseConfig):
    """Batch inference settings."""

    output: Path | None = None
    batch_size: int = Field(default=16, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class NormalizationConfig(BaseConfig):
    """Canonical aliases used by the optional normalized prediction endpoint."""

    aliases: dict[Literal["ORG", "NAME", "GEO"], dict[str, str]] = Field(default_factory=dict)


class ServiceConfig(BaseConfig):
    """HTTP server settings."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class ExperimentConfig(BaseConfig):
    """Complete common schema for an experiment YAML file."""

    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)


class ExperimentSettings(BaseSettings, ExperimentConfig):
    """Complete settings model assembled from CLI, ENV, `.env` and YAML."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="allow",
        case_sensitive=False,
    )

    yaml_path: ClassVar[Path | None] = None
    cli_overrides: ClassVar[list[str]] = []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define source priority, matching the project's configuration policy."""

        sources: list[PydanticBaseSettingsSource] = [
            CliOverridesSource(settings_cls, cls.cli_overrides),
            env_settings,
            dotenv_settings,
        ]
        if cls.yaml_path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=cls.yaml_path))
        sources.append(init_settings)
        return tuple(sources)


def parse_override(value: Any) -> Any:
    """Parse a CLI override with YAML scalar/list/object semantics."""

    if not isinstance(value, str):
        return value
    parsed = yaml.safe_load(value)
    return value if parsed is None and value.lower() not in {"null", "~"} else parsed


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    """Set a nested key such as ``training.learning_rate``."""

    parts = key.split(".")
    if not key or any(not part for part in parts):
        raise ValueError(f"invalid override key: {key!r}")
    target = config
    for part in parts[:-1]:
        current = target.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"cannot override nested key below scalar: {part!r}")
        target = current
    target[parts[-1]] = value
