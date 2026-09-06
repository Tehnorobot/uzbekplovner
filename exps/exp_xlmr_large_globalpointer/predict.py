"""Sliding-window inference and exact-span decoding for GlobalPointer."""

from typing import Any

import torch
from tqdm.auto import tqdm

from .data import tokenize_windows
from .model import LABELS, XLMRobertaGlobalPointer

JsonObject = dict[str, Any]


def _spans_overlap(left: JsonObject, right: JsonObject) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]


def select_non_overlapping(candidates: list[JsonObject]) -> list[JsonObject]:
    """Greedily retain the highest-scoring compatible spans."""

    selected: list[JsonObject] = []
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item["score"],
            item["start"],
            -(item["end"] - item["start"]),
            item["label"],
        ),
    )
    for candidate in ordered:
        if any(_spans_overlap(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    return [
        {"label": item["label"], "start": item["start"], "end": item["end"]} for item in selected
    ]


@torch.inference_mode()
def predict_records(
    model: XLMRobertaGlobalPointer,
    tokenizer: Any,
    records: list[JsonObject],
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> list[JsonObject]:
    """Predict spans and merge duplicate candidates from overlapping windows."""

    model.eval()
    aggregated: list[dict[tuple[str, int, int], float]] = [dict() for _ in records]
    pending: list[tuple[int, dict[str, list[int]], list[tuple[int, int]]]] = []

    def process_batch(
        items: list[tuple[int, dict[str, list[int]], list[tuple[int, int]]]],
    ) -> None:
        features = [item[1] for item in items]
        offsets_batch = [item[2] for item in items]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        sequence_length = int(batch["input_ids"].shape[1])
        span_mask = torch.zeros(len(items), sequence_length, dtype=torch.bool)
        for row, offsets in enumerate(offsets_batch):
            span_mask[row, : len(offsets)] = torch.tensor(
                [start != end for start, end in offsets],
                dtype=torch.bool,
            )
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits = (
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                span_mask=span_mask.to(device, non_blocking=True),
            )
            .float()
            .cpu()
        )

        for row, (record_index, _, offsets) in enumerate(items):
            positive = torch.nonzero(logits[row] > threshold, as_tuple=False)
            for label_id, token_start, token_end in positive.tolist():
                if token_start >= len(offsets) or token_end >= len(offsets):
                    continue
                character_start = offsets[token_start][0]
                character_end = offsets[token_end][1]
                if character_start == character_end:
                    continue
                key = (LABELS[label_id], character_start, character_end)
                score = float(logits[row, label_id, token_start, token_end].item())
                previous = aggregated[record_index].get(key)
                if previous is None or score > previous:
                    aggregated[record_index][key] = score

    for record_index, record in enumerate(tqdm(records, desc="Predict", unit="doc")):
        for feature, offsets in tokenize_windows(
            tokenizer,
            record["text"],
            max_length=max_length,
            stride=stride,
        ):
            pending.append((record_index, feature, offsets))
            if len(pending) == batch_size:
                process_batch(pending)
                pending = []
    if pending:
        process_batch(pending)

    predictions: list[JsonObject] = []
    for record, spans in zip(records, aggregated, strict=True):
        candidates = [
            {"label": label, "start": start, "end": end, "score": score}
            for (label, start, end), score in spans.items()
        ]
        predictions.append({"hash": record["hash"], "entities": select_non_overlapping(candidates)})
    return predictions
