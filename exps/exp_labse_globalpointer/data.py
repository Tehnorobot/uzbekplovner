"""Sliding-window tokenization and exact-span target construction."""

from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .model import LABELS

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
JsonObject = dict[str, Any]
ModelFeature = dict[str, list[int]]
Offsets = list[tuple[int, int]]


def validate_window(tokenizer: Any, max_length: int, stride: int) -> None:
    """Validate sliding-window sizes against tokenizer special tokens."""

    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length < 1:
        raise ValueError("training.max_length is too small for tokenizer special tokens")
    if not 0 <= stride < content_length:
        raise ValueError(f"training.stride must be between 0 and {content_length - 1}")


def tokenize_windows(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[tuple[ModelFeature, Offsets]]:
    """Tokenize text into overlapping windows while retaining character offsets."""

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )
    input_chunks = encoded["input_ids"]
    offset_chunks = encoded["offset_mapping"]
    if input_chunks and isinstance(input_chunks[0], int):
        input_chunks = [input_chunks]
        offset_chunks = [offset_chunks]

    windows: list[tuple[ModelFeature, Offsets]] = []
    for chunk_index, raw_offsets in enumerate(offset_chunks):
        feature: ModelFeature = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key not in encoded:
                continue
            values = encoded[key]
            feature[key] = values[chunk_index] if isinstance(values[0], list) else values
        offsets = [(int(start), int(end)) for start, end in raw_offsets]
        windows.append((feature, offsets))
    return windows


def window_character_bounds(offsets: Offsets) -> tuple[int, int] | None:
    """Return the character interval covered by non-special tokens."""

    valid_offsets = [(start, end) for start, end in offsets if start != end]
    if not valid_offsets:
        return None
    return (
        min(start for start, _ in valid_offsets),
        max(end for _, end in valid_offsets),
    )


def window_cuts_entity(offsets: Offsets, entities: list[JsonObject]) -> bool:
    """Check whether a window contains only part of a gold entity."""

    _, partial_entities = partition_window_entities(offsets, entities)
    return bool(partial_entities)


def partition_window_entities(
    offsets: Offsets,
    entities: list[JsonObject],
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Separate fully contained entities from entities cut by a window edge."""

    bounds = window_character_bounds(offsets)
    if bounds is None:
        return [], []
    window_start, window_end = bounds
    complete: list[JsonObject] = []
    partial: list[JsonObject] = []
    for entity in entities:
        overlaps = entity["start"] < window_end and entity["end"] > window_start
        fully_contained = window_start <= entity["start"] and entity["end"] <= window_end
        if fully_contained:
            complete.append(entity)
        elif overlaps:
            partial.append(entity)
    return complete, partial


def build_span_mask(
    offsets: Offsets,
    partial_entities: list[JsonObject],
) -> list[bool]:
    """Mask special tokens and tokens belonging to boundary-cut gold entities."""

    return [
        start != end
        and not any(start < entity["end"] and end > entity["start"] for entity in partial_entities)
        for start, end in offsets
    ]


def entity_token_span(
    offsets: Offsets,
    entity: JsonObject,
) -> tuple[int, int] | None:
    """Map one exact character span to inclusive token start/end indices."""

    indices = [
        token_index
        for token_index, (start, end) in enumerate(offsets)
        if start != end and start < entity["end"] and end > entity["start"]
    ]
    if not indices:
        return None
    token_start, token_end = indices[0], indices[-1]
    if offsets[token_start][0] != entity["start"] or offsets[token_end][1] != entity["end"]:
        raise ValueError(
            "gold character span is not exactly representable by tokenizer tokens: "
            f"entity={entity}, token_span="
            f"({offsets[token_start][0]}, {offsets[token_end][1]})"
        )
    return token_start, token_end


def build_span_targets(
    offsets: Offsets,
    entities: list[JsonObject],
) -> list[tuple[int, int, int]]:
    """Build sparse ``(label, token_start, token_end)`` targets for one window."""

    targets: list[tuple[int, int, int]] = []
    for entity in entities:
        token_span = entity_token_span(offsets, entity)
        if token_span is not None:
            targets.append((LABEL_TO_ID[entity["label"]], *token_span))
    return targets


class GlobalPointerDataset(Dataset):
    """Store tokenized windows and sparse exact-span labels."""

    def __init__(
        self,
        records: list[JsonObject],
        tokenizer: Any,
        *,
        max_length: int,
        stride: int,
        description: str,
    ) -> None:
        self.features: list[JsonObject] = []
        self.partial_entity_windows = 0
        represented: list[set[tuple[str, int, int]]] = [set() for _ in records]
        for record_index, record in enumerate(tqdm(records, desc=description, unit="doc")):
            for feature, offsets in tokenize_windows(
                tokenizer,
                record["text"],
                max_length=max_length,
                stride=stride,
            ):
                complete_entities, partial_entities = partition_window_entities(
                    offsets,
                    record["entities"],
                )
                if partial_entities:
                    self.partial_entity_windows += 1
                span_mask = build_span_mask(offsets, partial_entities)
                if not any(span_mask):
                    continue
                targets = build_span_targets(offsets, complete_entities)
                for label_id, token_start, token_end in targets:
                    represented[record_index].add(
                        (
                            LABELS[label_id],
                            offsets[token_start][0],
                            offsets[token_end][1],
                        )
                    )
                self.features.append(
                    {
                        **feature,
                        "span_labels": targets,
                        "span_mask": span_mask,
                    }
                )

        if not self.features:
            raise ValueError(f"{description}: tokenization produced no trainable windows")
        for record_index, record in enumerate(records):
            expected = {
                (entity["label"], entity["start"], entity["end"]) for entity in record["entities"]
            }
            missing = expected - represented[record_index]
            if missing:
                raise ValueError(
                    f"{description}: {record['hash']} has entities not represented in any "
                    f"complete token window: {sorted(missing)[:3]}"
                )

    def __len__(self) -> int:
        """Return the number of tokenized windows."""

        return len(self.features)

    def __getitem__(self, index: int) -> JsonObject:
        """Return one tokenized window."""

        return self.features[index]


class GlobalPointerCollator:
    """Pad windows and expand sparse span labels into a dense boolean tensor."""

    def __init__(self, tokenizer: Any, num_labels: int) -> None:
        self.tokenizer = tokenizer
        self.num_labels = num_labels

    def __call__(self, features: list[JsonObject]) -> dict[str, torch.Tensor]:
        """Create one padded training/evaluation batch."""

        model_features: list[JsonObject] = []
        sparse_labels: list[list[tuple[int, int, int]]] = []
        span_masks: list[list[bool]] = []
        for feature in features:
            current = dict(feature)
            sparse_labels.append(current.pop("span_labels"))
            span_masks.append(current.pop("span_mask"))
            model_features.append(current)

        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        sequence_length = int(batch["input_ids"].shape[1])
        labels = torch.zeros(
            len(features),
            self.num_labels,
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        )
        valid_tokens = torch.zeros(len(features), sequence_length, dtype=torch.bool)
        for row, (targets, mask) in enumerate(zip(sparse_labels, span_masks, strict=True)):
            valid_tokens[row, : len(mask)] = torch.tensor(mask, dtype=torch.bool)
            for label_id, token_start, token_end in targets:
                labels[row, label_id, token_start, token_end] = True
        batch["span_labels"] = labels
        batch["span_mask"] = valid_tokens
        return batch
