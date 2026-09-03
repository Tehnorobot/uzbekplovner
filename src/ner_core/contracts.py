"""Validation and serialization for the official exact-span NER contract."""

import json
from pathlib import Path
from typing import Any

LABELS = frozenset({"ORG", "NAME", "GEO"})
Entity = dict[str, Any]
Record = dict[str, Any]


class ContractError(ValueError):
    """Raised when an input or prediction violates the project contract."""


def validate_input_batch(raw: Any) -> list[dict[str, str]]:
    """Validate an HTTP request batch and return its canonical representation."""

    if not isinstance(raw, list) or not raw:
        raise ContractError("request body must be a non-empty JSON array")

    result: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for index, item in enumerate(raw):
        source = f"request[{index}]"
        if not isinstance(item, dict):
            raise ContractError(f"{source}: item must be an object")
        record_hash = item.get("hash")
        text = item.get("text")
        if not isinstance(record_hash, str) or not record_hash:
            raise ContractError(f"{source}: hash must be a non-empty string")
        if record_hash in seen_hashes:
            raise ContractError(f"{source}: duplicate hash {record_hash!r}")
        if not isinstance(text, str):
            raise ContractError(f"{source}: text must be a string")
        seen_hashes.add(record_hash)
        result.append({"hash": record_hash, "text": text})
    return result


def validate_entities(raw: Any, text: str, *, source: str) -> list[Entity]:
    """Validate, deduplicate-check and sort exact spans for one document."""

    if not isinstance(raw, list):
        raise ContractError(f"{source}: entities must be an array")

    entities: list[Entity] = []
    seen: set[tuple[str, int, int]] = set()
    for index, entity in enumerate(raw):
        entity_source = f"{source}/entities[{index}]"
        if not isinstance(entity, dict):
            raise ContractError(f"{entity_source}: entity must be an object")
        label = entity.get("label")
        start = entity.get("start")
        end = entity.get("end")
        if label not in LABELS:
            raise ContractError(f"{entity_source}: invalid label {label!r}")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
        ):
            raise ContractError(f"{entity_source}: invalid Unicode character offsets")
        key = (label, start, end)
        if key in seen:
            raise ContractError(f"{entity_source}: duplicate entity")
        seen.add(key)
        entities.append({key_name: entity[key_name] for key_name in ("label", "start", "end")})

    entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    for left, right in zip(entities, entities[1:], strict=False):
        if right["start"] < left["end"]:
            raise ContractError(f"{source}: overlapping entities are not supported")
    return entities


def validate_prediction_batch(
    raw: Any,
    inputs: list[dict[str, str]],
) -> list[Record]:
    """Validate an experiment result while preserving input order."""

    if not isinstance(raw, list) or len(raw) != len(inputs):
        raise ContractError("prediction count must match request count")

    result: list[Record] = []
    for index, (prediction, item) in enumerate(zip(raw, inputs, strict=True)):
        source = f"prediction[{index}]"
        if not isinstance(prediction, dict) or prediction.get("hash") != item["hash"]:
            raise ContractError(f"{source}: hash or result order differs from request")
        entities = validate_entities(prediction.get("entities"), item["text"], source=source)
        result.append({"hash": item["hash"], "entities": entities})
    return result


def read_jsonl(path: Path, *, require_entities: bool) -> list[Record]:
    """Read canonical train/dev JSONL records with UTF-8 validation."""

    records: list[Record] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ContractError(f"{path}:{line_number}: empty line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(raw, dict):
                raise ContractError(f"{path}:{line_number}: record must be an object")
            record_hash = raw.get("hash")
            text = raw.get("text")
            if not isinstance(record_hash, str) or not record_hash:
                raise ContractError(f"{path}:{line_number}: invalid hash")
            if record_hash in seen_hashes:
                raise ContractError(f"{path}:{line_number}: duplicate hash {record_hash!r}")
            if not isinstance(text, str):
                raise ContractError(f"{path}:{line_number}: text must be a string")
            record: Record = {"hash": record_hash, "text": text}
            if require_entities:
                record["entities"] = validate_entities(
                    raw.get("entities"), text, source=f"{path}:{line_number}"
                )
            records.append(record)
            seen_hashes.add(record_hash)
    if not records:
        raise ContractError(f"{path}: no records")
    return records


def write_jsonl(path: Path, records: list[Record]) -> None:
    """Write records as compact UTF-8 JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
