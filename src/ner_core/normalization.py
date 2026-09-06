"""Conservative postprocessing for canonical entity values."""

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .contracts import LABELS

AliasIndex = dict[str, dict[str, str]]

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "`": "'",
        "´": "'",
        "‘": "'",
        "’": "'",
        "ʼ": "'",
        "ʻ": "'",
        "ʹ": "'",
    }
)
_EDGE_PUNCTUATION = ' \t\r\n.,;:!?()[]{}"«»“”'


def clean_entity_value(value: str) -> str:
    """Normalize harmless typography while preserving entity spelling and case."""

    normalized = unicodedata.normalize("NFKC", value).translate(_APOSTROPHE_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(_EDGE_PUNCTUATION)


def build_alias_index(aliases: Mapping[str, Mapping[str, str]]) -> AliasIndex:
    """Validate aliases and build a case-insensitive canonical lookup table."""

    index: AliasIndex = {}
    for label, values in aliases.items():
        if label not in LABELS:
            raise ValueError(f"normalization alias has unsupported label: {label!r}")
        label_index: dict[str, str] = {}
        for source, canonical in values.items():
            source_key = clean_entity_value(source).casefold()
            canonical_value = clean_entity_value(canonical)
            if not source_key or not canonical_value:
                raise ValueError("normalization aliases must not contain empty values")
            previous = label_index.get(source_key)
            if previous is not None and previous != canonical_value:
                raise ValueError(f"conflicting canonical values for {label} alias {source!r}")
            label_index[source_key] = canonical_value
        index[label] = label_index
    return index


def normalize_prediction_batch(
    inputs: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    aliases: AliasIndex,
) -> list[dict[str, Any]]:
    """Add source and canonical text to validated span predictions."""

    result: list[dict[str, Any]] = []
    for item, prediction in zip(inputs, predictions, strict=True):
        text = item["text"]
        entities = []
        for entity in prediction["entities"]:
            surface = text[entity["start"] : entity["end"]]
            cleaned = clean_entity_value(surface)
            normalized = aliases.get(entity["label"], {}).get(cleaned.casefold(), cleaned)
            entities.append({**entity, "text": surface, "normalized": normalized})
        result.append({"hash": prediction["hash"], "entities": entities})
    return result
