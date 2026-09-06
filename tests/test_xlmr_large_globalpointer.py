from pathlib import Path

import torch

from exps.exp_xlmr_large_globalpointer.augmentation import _make_augmented_record
from exps.exp_xlmr_large_globalpointer.data import _span_targets
from exps.exp_xlmr_large_globalpointer.model import GlobalPointer, global_pointer_loss
from exps.exp_xlmr_large_globalpointer.preprocessing import (
    normalize_uzbek_apostrophes,
    preprocess_records,
)
from ner_core.runner import load_experiment_config


def test_global_pointer_masks_special_and_reverse_spans() -> None:
    head = GlobalPointer(hidden_size=8, num_labels=3, head_size=4)
    hidden = torch.randn(2, 5, 8)
    span_mask = torch.tensor([[False, True, True, True, False], [False, True, True, False, False]])

    logits = head(hidden, span_mask)

    assert logits.shape == (2, 3, 5, 5)
    assert torch.isfinite(logits[0, :, 1, 3]).all()
    assert (logits[0, :, 3, 1] == torch.finfo(logits.dtype).min).all()
    assert (logits[0, :, 0, :] == torch.finfo(logits.dtype).min).all()


def test_global_pointer_loss_is_finite() -> None:
    head = GlobalPointer(hidden_size=8, num_labels=3, head_size=4)
    logits = head(torch.randn(1, 4, 8), torch.tensor([[False, True, True, False]]))
    targets = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
    targets[0, 1, 1, 2] = True

    loss = global_pointer_loss(targets, logits)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_inexact_gold_boundary_is_still_represented_by_token_span() -> None:
    targets, represented = _span_targets(
        [(0, 0), (0, 6), (0, 0)],
        [{"label": "ORG", "start": 0, "end": 5}],
    )

    assert targets == [(0, 1, 1)]
    assert represented == {("ORG", 0, 5)}


def test_apostrophe_preprocessing_preserves_offsets_and_english_possessive() -> None:
    original = "G'ulom Domino's bilan O‘zbekistonda"
    normalized, replacements = normalize_uzbek_apostrophes(original)

    assert normalized == "Gʻulom Domino's bilan Oʻzbekistonda"
    assert len(normalized) == len(original)
    assert sum(replacements.values()) == 2

    records, stats = preprocess_records(
        [
            {
                "hash": "one",
                "text": original,
                "entities": [{"label": "NAME", "start": 0, "end": 6}],
            }
        ]
    )
    assert records[0]["entities"] == [{"label": "NAME", "start": 0, "end": 6}]
    assert stats["changed_records"] == 1


def test_partial_cluster_replacement_retains_all_entities() -> None:
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


def test_default_config_matches_xlmr_globalpointer_experiment() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    config, _ = load_experiment_config(
        root_dir,
        "exp_xlmr_large_globalpointer",
        None,
        [],
    )

    assert config["model"]["name"] == "FacebookAI/xlm-roberta-large"
    assert config["training"]["max_length"] == 512
    assert config["training"]["stride"] == 128
    assert config["global_pointer"]["head_size"] == 128
    assert config["augmentation"]["enabled"] is True
    assert config["augmentation"]["probability"] > 0
