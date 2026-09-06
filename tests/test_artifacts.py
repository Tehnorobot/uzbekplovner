from pathlib import Path

from ner_core.artifacts import write_run_metadata


def test_run_metadata_serializes_path_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run_metadata(
        run_dir,
        {"paths": {"train": Path("data/train.jsonl")}},
    )

    resolved_config = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    assert str(Path("data/train.jsonl")) in resolved_config
