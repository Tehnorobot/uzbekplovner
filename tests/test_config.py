import pytest

from ner_core.config import ExperimentConfig, parse_override, set_dotted


def test_dotted_cli_override_is_typed() -> None:
    config = {"training": {"epochs": 3}}
    set_dotted(config, "training.epochs", parse_override("5"))
    set_dotted(config, "training.use_amp", parse_override("true"))
    assert config == {"training": {"epochs": 5, "use_amp": True}}


def test_pydantic_validates_training_constraints() -> None:
    config = ExperimentConfig.model_validate({"training": {"epochs": 2}})
    assert config.training.epochs == 2

    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({"training": {"epochs": 0}})


def test_pydantic_validates_cluster_augmentation_settings() -> None:
    config = ExperimentConfig.model_validate(
        {"augmentation": {"enabled": True, "probability": 0.25}}
    )
    assert config.augmentation.enabled is True
    assert config.augmentation.probability == 0.25

    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({"augmentation": {"probability": 1.1}})


def test_pydantic_validates_normalization_alias_labels() -> None:
    config = ExperimentConfig.model_validate(
        {"normalization": {"aliases": {"GEO": {"Toshkentda": "Toshkent"}}}}
    )
    assert config.normalization.aliases["GEO"]["Toshkentda"] == "Toshkent"

    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(
            {"normalization": {"aliases": {"UNKNOWN": {"value": "canonical"}}}}
        )
