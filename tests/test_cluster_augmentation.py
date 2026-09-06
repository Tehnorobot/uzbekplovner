import pytest
import torch

from exps.exp_baseline.cluster_augmentation import (
    _make_augmented_record,
    apply_cluster_augmentation,
)


class FakeTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        encoded = [[ord(char) % 100 + 1 for char in text] for text in texts]
        width = max(len(item) for item in encoded)
        return {
            "input_ids": [item + [0] * (width - len(item)) for item in encoded],
            "attention_mask": [[1] * len(item) + [0] * (width - len(item)) for item in encoded],
        }


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding


def _record(record_hash: str, name: str, location: str) -> dict[str, object]:
    text = f"{name} {location}"
    return {
        "hash": record_hash,
        "text": text,
        "entities": [
            {"label": "NAME", "start": 0, "end": len(name)},
            {
                "label": "GEO",
                "start": len(name) + 1,
                "end": len(text),
            },
        ],
    }


def test_disabled_augmentation_does_not_touch_records() -> None:
    records = [_record("one", "Ali", "Toshkentda")]
    result, stats = apply_cluster_augmentation(
        records,
        tokenizer=None,
        model=None,
        enabled=False,
        probability=1.0,
        num_clusters=1,
        max_candidates=5,
        seed=42,
    )

    assert result == records
    assert stats["added_records"] == 0
    assert stats["reason"] == "disabled"


def test_cluster_augmentation_recalculates_offsets() -> None:
    records = [
        _record("one", "Ali", "Toshkentda"),
        _record("two", "Vali", "Samarqandda"),
    ]
    result, stats = apply_cluster_augmentation(
        records,
        FakeTokenizer(),
        FakeModel(),
        enabled=True,
        probability=1.0,
        num_clusters=1,
        max_candidates=5,
        seed=42,
    )

    assert stats["added_records"] == 2
    assert len(result) == 4
    for record in result[2:]:
        assert len(record["entities"]) == 2
        for entity in record["entities"]:
            assert record["text"][entity["start"] : entity["end"]]


def test_partial_replacement_preserves_unmodified_entities() -> None:
    record = _record("one", "Ali", "Toshkentda")
    replacement = [(record["entities"][0], "Alisher")]

    augmented = _make_augmented_record(record, replacement, variant_index=1)

    assert augmented["text"] == "Alisher Toshkentda"
    assert augmented["entities"] == [
        {"label": "NAME", "start": 0, "end": 7},
        {"label": "GEO", "start": 8, "end": 18},
    ]


def test_cluster_augmentation_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        apply_cluster_augmentation(
            [],
            None,
            None,
            enabled=True,
            probability=1.1,
            num_clusters=1,
            max_candidates=1,
            seed=42,
        )
