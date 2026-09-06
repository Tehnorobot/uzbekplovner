import torch

from exps.exp_labse_globalpointer.augmentation import _make_augmented_record
from exps.exp_labse_globalpointer.data import GlobalPointerDataset
from exps.exp_labse_globalpointer.model import EfficientGlobalPointer, global_pointer_loss
from exps.exp_labse_globalpointer.predict import select_non_overlapping


class TwoWindowTokenizer:
    def __call__(self, *_: object, **__: object) -> dict[str, list[list[object]]]:
        return {
            "input_ids": [[101, 1, 2, 3, 102], [101, 3, 4, 102]],
            "attention_mask": [[1, 1, 1, 1, 1], [1, 1, 1, 1]],
            "offset_mapping": [
                [(0, 0), (0, 2), (2, 4), (8, 10), (0, 0)],
                [(0, 0), (8, 10), (10, 12), (0, 0)],
            ],
        }


def test_efficient_global_pointer_masks_invalid_and_reverse_spans() -> None:
    head = EfficientGlobalPointer(hidden_size=8, num_labels=3, head_size=4)
    hidden = torch.randn(2, 5, 8)
    span_mask = torch.tensor([[False, True, True, True, False], [False, True, True, False, False]])

    logits = head(hidden, span_mask)

    assert logits.shape == (2, 3, 5, 5)
    assert torch.isfinite(logits[0, :, 1, 3]).all()
    assert (logits[0, :, 3, 1] == torch.finfo(logits.dtype).min).all()
    assert (logits[0, :, 0, :] == torch.finfo(logits.dtype).min).all()


def test_global_pointer_loss_is_finite() -> None:
    head = EfficientGlobalPointer(hidden_size=8, num_labels=3, head_size=4)
    logits = head(torch.randn(1, 4, 8), torch.tensor([[False, True, True, False]]))
    labels = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
    labels[0, 1, 1, 2] = True

    loss = global_pointer_loss(labels, logits)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_decoder_keeps_highest_scoring_non_overlapping_spans() -> None:
    candidates = [
        {"label": "NAME", "start": 0, "end": 5, "score": 3.0},
        {"label": "ORG", "start": 0, "end": 10, "score": 2.0},
        {"label": "GEO", "start": 11, "end": 15, "score": 1.0},
    ]

    assert select_non_overlapping(candidates) == [
        {"label": "NAME", "start": 0, "end": 5},
        {"label": "GEO", "start": 11, "end": 15},
    ]


def test_partial_cluster_replacement_preserves_other_entities() -> None:
    record = {
        "hash": "one",
        "text": "Ali Toshkentda",
        "entities": [
            {"label": "NAME", "start": 0, "end": 3},
            {"label": "GEO", "start": 4, "end": 14},
        ],
    }

    augmented = _make_augmented_record(record, {0: "Alisher"}, variant_index=1)

    assert augmented["text"] == "Alisher Toshkentda"
    assert augmented["entities"] == [
        {"label": "NAME", "start": 0, "end": 7},
        {"label": "GEO", "start": 8, "end": 18},
    ]


def test_partial_entity_does_not_discard_complete_entities_in_same_window() -> None:
    records = [
        {
            "hash": "long-record",
            "text": "abcd    efgh",
            "entities": [
                {"label": "NAME", "start": 0, "end": 2},
                {"label": "GEO", "start": 8, "end": 12},
            ],
        }
    ]

    dataset = GlobalPointerDataset(
        records,
        TwoWindowTokenizer(),
        max_length=5,
        stride=1,
        description="test",
    )

    assert len(dataset) == 2
    assert dataset.partial_entity_windows == 1
    assert dataset[0]["span_labels"] == [(1, 1, 1)]
    assert dataset[0]["span_mask"] == [False, True, True, False, False]
    assert dataset[1]["span_labels"] == [(2, 1, 2)]
