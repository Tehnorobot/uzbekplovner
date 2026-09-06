"""Length-preserving preprocessing shared by training and inference."""

from collections import Counter
from typing import Any

UZBEK_APOSTROPHE = "ʻ"
UZBEK_APOSTROPHE_VARIANTS = frozenset({"'", "‘", "’", "ʼ", "`"})


def normalize_uzbek_apostrophes(text: str) -> tuple[str, Counter[str]]:
    """Unify apostrophes in Uzbek O/G letters without changing offsets."""

    characters = list(text)
    replacements: Counter[str] = Counter()
    for index in range(1, len(characters) - 1):
        current = characters[index]
        terminal_s_possessive = characters[index + 1] in "sS" and (
            index + 2 == len(characters) or not characters[index + 2].isalpha()
        )
        if (
            current in UZBEK_APOSTROPHE_VARIANTS
            and characters[index - 1] in "oOgG"
            and characters[index + 1].isascii()
            and characters[index + 1].isalpha()
            and not terminal_s_possessive
        ):
            characters[index] = UZBEK_APOSTROPHE
            replacements[current] += 1
    normalized = "".join(characters)
    if len(normalized) != len(text):
        raise AssertionError("Uzbek apostrophe normalization changed text length")
    return normalized, replacements


def preprocess_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize record text and report replacements while retaining all spans."""

    result: list[dict[str, Any]] = []
    total: Counter[str] = Counter()
    changed_records = 0
    for record in records:
        normalized, replacements = normalize_uzbek_apostrophes(record["text"])
        if replacements:
            changed_records += 1
            total.update(replacements)
        result.append({**record, "text": normalized})
    stats = {
        "method": "length_preserving_uzbek_apostrophe_normalization",
        "records": len(records),
        "changed_records": changed_records,
        "replacements": sum(total.values()),
        "by_variant": dict(sorted(total.items(), key=lambda item: ord(item[0]))),
    }
    return result, stats
