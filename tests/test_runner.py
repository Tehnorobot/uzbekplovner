from pathlib import Path

import pytest

from exps.exp_baseline.main import _train_args
from ner_core.runner import load_experiment_config, validate_exp_name


def test_cli_override_has_priority_over_experiment_yaml() -> None:
    config, _ = load_experiment_config(
        Path(__file__).resolve().parents[1],
        "exp_baseline",
        None,
        ["training.epochs=7"],
    )
    assert config["training"]["epochs"] == 7


def test_experiment_name_cannot_escape_import_namespace() -> None:
    with pytest.raises(ValueError):
        validate_exp_name("../outside")


def test_runtime_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVICE__PORT", "9001")
    monkeypatch.setenv("SERVICE__DEVICE", "cpu")
    config, _ = load_experiment_config(
        Path(__file__).resolve().parents[1],
        "exp_baseline",
        None,
        [],
    )
    assert config["service"]["port"] == 9001
    assert config["service"]["device"] == "cpu"


def test_current_model_is_experiment_local() -> None:
    config, _ = load_experiment_config(
        Path(__file__).resolve().parents[1],
        "exp_baseline",
        None,
        [],
    )
    assert config["paths"]["model_dir"] == Path("exps/exp_baseline/artifacts/model")
    assert "promotion" not in config


def test_baseline_training_allows_runner_metadata(tmp_path: Path) -> None:
    from ner_core.runner import ExperimentContext

    root_dir = Path(__file__).resolve().parents[1]
    config, _ = load_experiment_config(root_dir, "exp_baseline", None, [])
    context = ExperimentContext(
        root_dir=root_dir,
        experiment_dir=root_dir / "exps" / "exp_baseline",
        run_dir=tmp_path / "run",
    )
    args = _train_args(config, context)

    assert args.output_dir == context.run_dir
    assert args.overwrite_output_dir is True
