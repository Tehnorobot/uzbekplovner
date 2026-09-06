"""Merge the Mendeley Uzbek NER data into the project JSONL dataset."""

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_LABELS = ("ORG", "NAME", "GEO")
SOURCE_TO_TARGET = {
    "PER": "NAME",
    "LOC": "GEO",
    "ORG": "ORG",
}
DEFAULT_ARCHIVE = Path(
    "useful_files/A Multi-Source Synthetic Dataset for Uzbek Sentime.zip"
)
DEFAULT_OUTPUT_DIR = Path("data/merged_mendeley_v5")
DEFAULT_MEMBER_SUFFIX = "synthetic_dataset_10000_v6.jsonl"
ENTITY_CONTINUATION_CHARS = frozenset("'’ʻʼ-_")
JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the merge operation."""

    parser = argparse.ArgumentParser(
        description="Merge Mendeley Uzbek NER records into the project train split."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[JsonObject]:
    """Read project JSONL records and check their basic structure."""

    records: list[JsonObject] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            record_hash = record.get("hash")
            text = record.get("text")
            entities = record.get("entities")
            if not isinstance(record_hash, str) or not record_hash:
                raise ValueError(f"{path}:{line_number}: hash must be a non-empty string")
            if record_hash in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record_hash}")
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: text must be a string")
            validate_target_entities(entities, len(text), f"{path}:{line_number}")
            seen_hashes.add(record_hash)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def validate_target_entities(raw_entities: Any, text_length: int, source: str) -> None:
    """Validate entities against the target label and offset contract."""

    if not isinstance(raw_entities, list):
        raise ValueError(f"{source}: entities must be an array")
    seen: set[tuple[str, int, int]] = set()
    spans: list[tuple[int, int]] = []
    for index, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            raise ValueError(f"{source}/entities[{index}]: entity must be an object")
        label = entity.get("label")
        start = entity.get("start")
        end = entity.get("end")
        if label not in TARGET_LABELS:
            raise ValueError(f"{source}/entities[{index}]: invalid label {label!r}")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= text_length
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        key = (label, start, end)
        if key in seen:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        seen.add(key)
        spans.append((start, end))
    for left, right in zip(
        sorted(spans),
        sorted(spans)[1:],
        strict=False,
    ):
        if right[0] < left[1]:
            raise ValueError(f"{source}: overlapping entities are not supported")


def choose_jsonl_member(archive: zipfile.ZipFile) -> str:
    """Find the primary synthetic NER JSONL member in the archive."""

    members = [name for name in archive.namelist() if not name.endswith("/")]
    exact = [name for name in members if name.endswith(DEFAULT_MEMBER_SUFFIX)]
    if len(exact) == 1:
        return exact[0]
    jsonl_members = [name for name in members if name.lower().endswith(".jsonl")]
    if len(jsonl_members) == 1:
        return jsonl_members[0]
    raise ValueError(
        "Could not identify the source JSONL. Expected one member ending with "
        f"{DEFAULT_MEMBER_SUFFIX!r}; found: {jsonl_members}"
    )


def parse_source_arrays(record: JsonObject, source: str) -> tuple[list[str], list[str]]:
    """Read aligned surface-form and source-label arrays from one source row."""

    surfaces = record.get("entities")
    labels = record.get("entity_type")
    if not isinstance(surfaces, list) or not isinstance(labels, list):
        raise ValueError(f"{source}: entities and entity_type must be arrays")
    if len(surfaces) != len(labels):
        raise ValueError(f"{source}: entities and entity_type lengths differ")
    if any(not isinstance(surface, str) or not surface for surface in surfaces):
        raise ValueError(f"{source}: entities must contain non-empty strings")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError(f"{source}: entity_type must contain non-empty strings")
    return surfaces, labels


def make_source_hash(source_id: Any, text: str) -> str:
    """Create a deterministic project-compatible hash for an external record."""

    payload = f"{source_id}\0{text}".encode("utf-8")
    return f"mendeley-{hashlib.sha256(payload).hexdigest()}"


def find_occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    """Return all exact occurrences of a source surface form."""

    occurrences: list[tuple[int, int]] = []
    search_start = 0
    while True:
        start = text.find(surface, search_start)
        if start < 0:
            return occurrences
        end = start + len(surface)
        occurrences.append((start, end))
        search_start = start + 1


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two half-open character spans overlap."""

    return left[0] < right[1] and right[0] < left[1]


def assign_source_occurrences(text: str, surfaces: list[str]) -> dict[int, tuple[int, int]]:
    """Assign source entities, preferring long non-overlapping occurrences."""

    assignments: dict[int, tuple[int, int]] = {}
    used_spans: set[tuple[int, int]] = set()
    assigned_spans: list[tuple[int, int]] = []
    order = sorted(range(len(surfaces)), key=lambda index: (-len(surfaces[index]), index))
    for entity_index in order:
        surface = surfaces[entity_index]
        candidates = [
            occurrence
            for occurrence in find_occurrences(text, surface)
            if occurrence not in used_spans
        ]
        if not candidates:
            raise ValueError(f"surface form {surface!r} has no unused exact occurrence in text")
        non_overlapping = [
            occurrence
            for occurrence in candidates
            if not any(spans_overlap(occurrence, assigned) for assigned in assigned_spans)
        ]
        occurrence = (non_overlapping or candidates)[0]
        assignments[entity_index] = occurrence
        used_spans.add(occurrence)
        assigned_spans.append(occurrence)
    return assignments


def expand_attached_suffix(text: str, end: int) -> int:
    """Include a suffix attached to an entity according to the project guide."""

    while end < len(text) and (
        text[end].isalnum() or text[end] in ENTITY_CONTINUATION_CHARS
    ):
        end += 1
    return end


def remove_overlapping_entities(
    entities: list[JsonObject],
    counters: Counter[str],
) -> list[JsonObject]:
    """Keep the longest entity when source annotations remain nested."""

    kept: list[JsonObject] = []
    candidates = sorted(
        entities,
        key=lambda entity: (
            -(entity["end"] - entity["start"]),
            entity["start"],
            entity["end"],
        ),
    )
    for candidate in candidates:
        span = (candidate["start"], candidate["end"])
        if any(spans_overlap(span, (item["start"], item["end"])) for item in kept):
            counters["dropped_overlapping_entity"] += 1
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda entity: (entity["start"], entity["end"], entity["label"]))


def convert_source_record(
    record: JsonObject,
    source: str,
    counters: Counter[str],
) -> JsonObject:
    """Convert one Mendeley row to the project exact-span JSON schema."""

    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{source}: text must be a string")
    surfaces, source_labels = parse_source_arrays(record, source)

    entities: list[JsonObject] = []
    assignments = assign_source_occurrences(text, surfaces)
    for index, (surface, source_label) in enumerate(zip(surfaces, source_labels, strict=True)):
        start, surface_end = assignments[index]
        end = expand_attached_suffix(text, surface_end)
        source_label = source_label.upper()
        counters[f"source_label:{source_label}"] += 1
        target_label = SOURCE_TO_TARGET.get(source_label)
        if target_label is None:
            counters[f"dropped_label:{source_label}"] += 1
            continue
        entities.append({"label": target_label, "start": start, "end": end})

    entities = remove_overlapping_entities(entities, counters)
    result = {
        "hash": make_source_hash(record.get("id", source), text),
        "text": text,
        "entities": entities,
    }
    validate_target_entities(result["entities"], len(text), source)
    return result


def load_external_records(archive_path: Path) -> tuple[list[JsonObject], str, Counter[str]]:
    """Load and convert the primary JSONL file from the ZIP archive."""

    counters: Counter[str] = Counter()
    records: list[JsonObject] = []
    with zipfile.ZipFile(archive_path) as archive:
        member = choose_jsonl_member(archive)
        with archive.open(member, "r") as binary_stream:
            text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8-sig")
            for line_number, line in enumerate(text_stream, start=1):
                if not line.strip():
                    raise ValueError(f"{member}:{line_number}: empty line")
                try:
                    source_record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{member}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(source_record, dict):
                    raise ValueError(f"{member}:{line_number}: record must be an object")
                records.append(
                    convert_source_record(
                        source_record,
                        f"{member}:{line_number}",
                        counters,
                    )
                )
    if not records:
        raise ValueError(f"{archive_path}: source JSONL is empty")
    return records, member, counters


def deduplicate_external_records(
    records: list[JsonObject],
    existing_texts: set[str],
    counters: Counter[str],
) -> list[JsonObject]:
    """Remove duplicate texts already present in the project or source data."""

    result: list[JsonObject] = []
    seen_texts = set(existing_texts)
    seen_hashes: set[str] = set()
    for record in records:
        text = record["text"]
        record_hash = record["hash"]
        if text in seen_texts:
            counters["skipped_duplicate_text"] += 1
            continue
        if record_hash in seen_hashes:
            raise ValueError(f"duplicate generated hash: {record_hash}")
        seen_texts.add(text)
        seen_hashes.add(record_hash)
        result.append(record)
    return result


def write_jsonl(path: Path, records: list[JsonObject]) -> None:
    """Write records as UTF-8 JSONL."""

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory or reject accidental overwrites."""

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {path}; use --overwrite to reuse it"
        )
    path.mkdir(parents=True, exist_ok=True)


def build_manifest(
    archive_path: Path,
    source_member: str,
    train_path: Path,
    dev_path: Path,
    output_dir: Path,
    original_train: list[JsonObject],
    original_dev: list[JsonObject],
    external_records: list[JsonObject],
    counters: Counter[str],
) -> JsonObject:
    """Build an auditable description of the merge result."""

    entity_counts = Counter(
        entity["label"]
        for record in external_records
        for entity in record["entities"]
    )
    return {
        "schema_version": 1,
        "operation": "merge_external_ner_into_train",
        "source_archive": str(archive_path),
        "source_member": source_member,
        "source_license": "CC BY 4.0",
        "source_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "input_train": str(train_path),
        "input_dev": str(dev_path),
        "policy": "append converted external records to train; keep original dev unchanged",
        "label_mapping": SOURCE_TO_TARGET,
        "span_policy": (
            "include alphanumeric and apostrophe/hyphen suffixes attached directly to a source surface form"
        ),
        "dropped_source_labels": sorted(
            key.removeprefix("dropped_label:")
            for key in counters
            if key.startswith("dropped_label:")
        ),
        "counts": {
            "original_train_records": len(original_train),
            "original_dev_records": len(original_dev),
            "external_records_after_deduplication": len(external_records),
            "merged_train_records": len(original_train) + len(external_records),
            "merged_dev_records": len(original_dev),
            "external_entities_by_target_label": dict(sorted(entity_counts.items())),
            **dict(sorted(counters.items())),
        },
        "outputs": {
            "train": str(output_dir / "train.jsonl"),
            "dev": str(output_dir / "dev.jsonl"),
            "manifest": str(output_dir / "merge_manifest.json"),
        },
    }


def run(args: argparse.Namespace) -> int:
    """Merge external records and write the derived dataset."""

    archive_path = args.archive.expanduser().resolve()
    train_path = args.train.expanduser().resolve()
    dev_path = args.dev.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (archive_path, train_path, dev_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    original_train = read_jsonl(train_path)
    original_dev = read_jsonl(dev_path)
    original_texts = {record["text"] for record in original_train + original_dev}
    external_records, source_member, counters = load_external_records(archive_path)
    external_records = deduplicate_external_records(
        external_records,
        original_texts,
        counters,
    )
    merged_train = original_train + external_records

    prepare_output_dir(output_dir, args.overwrite)
    output_train = output_dir / "train.jsonl"
    output_dev = output_dir / "dev.jsonl"
    output_manifest = output_dir / "merge_manifest.json"
    write_jsonl(output_train, merged_train)
    write_jsonl(output_dev, original_dev)
    manifest = build_manifest(
        archive_path,
        source_member,
        train_path,
        dev_path,
        output_dir,
        original_train,
        original_dev,
        external_records,
        counters,
    )
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Original train: {len(original_train)}")
    print(f"Original dev: {len(original_dev)}")
    print(f"Added external records: {len(external_records)}")
    print(f"Merged train: {len(merged_train)}")
    print(f"Dev remains unchanged: {len(original_dev)}")
    print(f"Output train: {output_train}")
    print(f"Output dev: {output_dev}")
    print(f"Manifest: {output_manifest}")
    print(f"Dropped source labels: {manifest['dropped_source_labels'] or 'none'}")
    return 0


def main() -> int:
    """Run the merge and print a concise error on invalid input."""

    try:
        return run(parse_args())
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
