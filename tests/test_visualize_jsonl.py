import json

import pytest

from scripts.visualize_jsonl import (
    load_comparison_records,
    render_comparison_text,
    render_svg,
    read_records,
    render_text,
)


def test_render_text_marks_entities_and_keeps_offsets() -> None:
    text = "Ali Toshkentda ishlaydi."
    entities = [
        {"label": "NAME", "start": 0, "end": 3},
        {"label": "GEO", "start": 4, "end": 14},
    ]

    assert render_text(text, entities, color=False) == (
        "[NAME:Ali] [GEO:Toshkentda] ishlaydi."
    )


def test_read_records_rejects_invalid_offsets(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "hash": "1",
                "text": "Ali",
                "entities": [{"label": "NAME", "start": 0, "end": 4}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid character offsets"):
        read_records(path)


def test_render_svg_contains_escaped_text_and_entity_color() -> None:
    records = [
        {
            "hash": "1",
            "text": "Ali & Toshkent",
            "entities": [
                {"label": "NAME", "start": 0, "end": 3},
                {"label": "GEO", "start": 6, "end": 14},
            ],
        }
    ]

    image = render_svg(records)

    assert image.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "&amp;" in image
    assert "#0891b2" in image
    assert "#16a34a" in image


def test_comparison_marks_correct_missed_and_false_positive(tmp_path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "hash": "1",
                "text": "Ali Toshkentda ishlaydi.",
                "entities": [{"label": "NAME", "start": 0, "end": 3}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_path.write_text(
        json.dumps(
            {
                "hash": "1",
                "entities": [
                    {"label": "NAME", "start": 0, "end": 3},
                    {"label": "GEO", "start": 4, "end": 14},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_comparison_records(gold_path, prediction_path)
    rendered = render_comparison_text(
        records[0]["text"],
        records[0]["gold_entities"],
        records[0]["pred_entities"],
        color=False,
    )

    assert "[correct:Ali]" in rendered
    assert "[false_positive:Toshkentda]" in rendered
