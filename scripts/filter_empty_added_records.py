"""Remove a deterministic fraction of empty records originating from added datasets."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.merge_mendeley_ner import (
        prepare_output_dir,
        read_jsonl,
        write_jsonl,
    )
except ModuleNotFoundError:
    from merge_mendeley_ner import prepare_output_dir, read_jsonl, write_jsonl


DEFAULT_TRAIN = Path("data/merged_mendeley_uzner_v2/train.jsonl")
DEFAULT_DEV = Path("data/merged_mendeley_uzner_v2/dev.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/merged_mendeley_uzner_v2_empty75")
DEFAULT_ADDED_PREFIXES = ("mendeley-", "uzner-v2-")
JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse filtering paths, fraction, seed and provenance prefixes."""

    parser = argparse.ArgumentParser(
        description=(
            "Remove 75% of entity-empty records only when their hash identifies "
            "an externally added dataset. Original records and all non-empty records stay."
        )
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--drop-fraction",
        type=float,
        default=0.75,
        help="Fraction of eligible empty added records to remove (default: 0.75).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--added-prefix",
        dest="added_prefixes",
        action="append",
        help=(
            "Hash prefix identifying an added source; repeat for multiple sources. "
            "Defaults to mendeley- and uzner-v2-."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_prefix(record_hash: str, added_prefixes: tuple[str, ...]) -> str | None:
    """Return the matching added-source prefix, if any."""

    return next(
        (prefix for prefix in added_prefixes if record_hash.startswith(prefix)),
        None,
    )


def _entity_counts(records: list[JsonObject]) -> dict[str, int]:
    """Count retained target entities by label."""

    counts = Counter(
        entity["label"]
        for record in records
        for entity in record["entities"]
    )
    return dict(sorted(counts.items()))


def filter_empty_added_records(
    records: list[JsonObject],
    *,
    drop_fraction: float,
    seed: int,
    added_prefixes: tuple[str, ...] = DEFAULT_ADDED_PREFIXES,
) -> tuple[list[JsonObject], dict[str, Any]]:
    """Drop a seeded sample only from empty records with added-source hashes."""

    if not 0.0 <= drop_fraction <= 1.0:
        raise ValueError("drop_fraction must be in [0, 1]")
    if not added_prefixes or any(not prefix for prefix in added_prefixes):
        raise ValueError("at least one non-empty added-source prefix is required")
    if len(set(added_prefixes)) != len(added_prefixes):
        raise ValueError("added-source prefixes must be unique")

    eligible_indices: list[int] = []
    input_by_source: Counter[str] = Counter()
    empty_by_source: Counter[str] = Counter()
    original_records = 0
    input_empty_records = 0
    for index, record in enumerate(records):
        is_empty = not record["entities"]
        input_empty_records += int(is_empty)
        source = _source_prefix(record["hash"], added_prefixes)
        if source is None:
            original_records += 1
            continue
        input_by_source[source] += 1
        if is_empty:
            empty_by_source[source] += 1
            eligible_indices.append(index)

    remove_count = math.floor(len(eligible_indices) * drop_fraction)
    randomizer = random.Random(seed)
    removed_indices = set(randomizer.sample(eligible_indices, remove_count))
    filtered = [record for index, record in enumerate(records) if index not in removed_indices]

    removed_by_source: Counter[str] = Counter()
    for index in removed_indices:
        record = records[index]
        source = _source_prefix(record["hash"], added_prefixes)
        if source is None or record["entities"]:
            raise AssertionError("filter selected a non-empty or original record")
        removed_by_source[source] += 1

    before_entities = _entity_counts(records)
    after_entities = _entity_counts(filtered)
    if before_entities != after_entities:
        raise AssertionError("entity counts changed while removing empty records")

    stats: dict[str, Any] = {
        "input_records": len(records),
        "output_records": len(filtered),
        "removed_records": remove_count,
        "input_empty_records": input_empty_records,
        "input_empty_share": input_empty_records / len(records) if records else 0.0,
        "output_empty_records": input_empty_records - remove_count,
        "output_empty_share": (
            (input_empty_records - remove_count) / len(filtered) if filtered else 0.0
        ),
        "drop_fraction": drop_fraction,
        "seed": seed,
        "added_prefixes": list(added_prefixes),
        "original_records": original_records,
        "added_records_by_prefix": dict(sorted(input_by_source.items())),
        "eligible_empty_added_by_prefix": dict(sorted(empty_by_source.items())),
        "removed_empty_added_by_prefix": dict(sorted(removed_by_source.items())),
        "eligible_empty_added_records": len(eligible_indices),
        "kept_empty_added_records": len(eligible_indices) - remove_count,
        "entities_by_label_before": before_entities,
        "entities_by_label_after": after_entities,
    }
    return filtered, stats


def run(args: argparse.Namespace) -> int:
    """Read merged data, filter train only, copy dev, and write an audit manifest."""

    train_path = args.train.expanduser().resolve()
    dev_path = args.dev.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (train_path, dev_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    added_prefixes = tuple(args.added_prefixes or DEFAULT_ADDED_PREFIXES)
    train_records = read_jsonl(train_path)
    dev_records = read_jsonl(dev_path)
    filtered_train, stats = filter_empty_added_records(
        train_records,
        drop_fraction=args.drop_fraction,
        seed=args.seed,
        added_prefixes=added_prefixes,
    )

    prepare_output_dir(output_dir, args.overwrite)
    output_train = output_dir / "train.jsonl"
    output_dev = output_dir / "dev.jsonl"
    output_manifest = output_dir / "filter_manifest.json"
    write_jsonl(output_train, filtered_train)
    write_jsonl(output_dev, dev_records)
    manifest = {
        "schema_version": 1,
        "operation": "drop_empty_added_records",
        "policy": (
            "drop a deterministic sample only when entities is empty and hash starts "
            "with one of added_prefixes; preserve record order; keep dev unchanged"
        ),
        "input_train": str(train_path),
        "input_dev": str(dev_path),
        "counts": {
            **stats,
            "dev_records": len(dev_records),
        },
        "outputs": {
            "train": str(output_train),
            "dev": str(output_dev),
            "manifest": str(output_manifest),
        },
    }
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input train records: {stats['input_records']}")
    print(f"Eligible empty added records: {stats['eligible_empty_added_records']}")
    print(f"Removed empty added records: {stats['removed_records']}")
    print(f"Kept train records: {stats['output_records']}")
    print(f"Dev records kept unchanged: {len(dev_records)}")
    print(f"Output train: {output_train}")
    print(f"Manifest: {output_manifest}")
    return 0


def main() -> int:
    """Run filtering and report input or validation errors compactly."""

    try:
        return run(parse_args())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
