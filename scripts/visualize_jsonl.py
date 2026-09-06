"""Print a random sample of project NER JSONL records with highlighted entities."""

import argparse
import json
import random
import sys
from html import escape
from pathlib import Path
from typing import Any

LABEL_COLORS = {
    "ORG": "\033[95m",
    "NAME": "\033[96m",
    "GEO": "\033[92m",
}
RESET = "\033[0m"
DEFAULT_COLOR = "\033[93m"
STATUS_COLORS = {
    "correct": "\033[92m",
    "missed": "\033[91m",
    "false_positive": "\033[94m",
    "boundary_error": "\033[93m",
}
SVG_COLORS = {
    "ORG": "#c026d3",
    "NAME": "#0891b2",
    "GEO": "#16a34a",
}
SVG_DEFAULT_COLOR = "#ca8a04"
SVG_STATUS_COLORS = {
    "correct": "#16a34a",
    "missed": "#dc2626",
    "false_positive": "#2563eb",
    "boundary_error": "#d97706",
}
JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse visualization options."""

    parser = argparse.ArgumentParser(
        description="Visualize random NER records from a UTF-8 JSONL file."
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input JSONL file.")
    parser.add_argument("--gold", type=Path, help="Gold JSONL for comparison mode.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Prediction JSONL for comparison mode. Use together with --gold.",
    )
    parser.add_argument("-n", "--number", type=int, default=5, help="Number of records.")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional UTF-8 output file. Without it, print to the terminal.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "svg"),
        default="text",
        help="Output format. SVG requires --output.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Use readable markers instead of ANSI colors.",
    )
    return parser.parse_args()


def _validate_entity(entity: Any, text: str, source: str) -> tuple[str, int, int]:
    """Validate one entity and return its label and character offsets."""

    if not isinstance(entity, dict):
        raise ValueError(f"{source}: entity must be an object")
    label = entity.get("label")
    start = entity.get("start")
    end = entity.get("end")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{source}: label must be a non-empty string")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError(f"{source}: invalid character offsets")
    return label, start, end


def read_records(path: Path) -> list[JsonObject]:
    """Read JSONL records and validate text and entity offsets."""

    records: list[JsonObject] = []
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
            text = record.get("text")
            entities = record.get("entities")
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: text must be a string")
            if not isinstance(entities, list):
                raise ValueError(f"{path}:{line_number}: entities must be an array")

            spans: list[tuple[int, int]] = []
            for entity_index, entity in enumerate(entities):
                _, start, end = _validate_entity(
                    entity,
                    text,
                    f"{path}:{line_number}/entities[{entity_index}]",
                )
                spans.append((start, end))
            for left, right in zip(sorted(spans), sorted(spans)[1:], strict=False):
                if right[0] < left[1]:
                    raise ValueError(f"{path}:{line_number}: overlapping entities")
            records.append(record)

    if not records:
        raise ValueError(f"{path}: no records")
    return records


def read_prediction_records(path: Path) -> list[JsonObject]:
    """Read prediction JSONL records that may not contain the source text."""

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
            if not isinstance(record_hash, str) or not record_hash:
                raise ValueError(f"{path}:{line_number}: hash must be a non-empty string")
            if record_hash in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record_hash}")
            entities = record.get("entities")
            if not isinstance(entities, list):
                raise ValueError(f"{path}:{line_number}: entities must be an array")
            for entity_index, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    raise ValueError(
                        f"{path}:{line_number}/entities[{entity_index}]: entity must be an object"
                    )
                label = entity.get("label")
                start = entity.get("start")
                end = entity.get("end")
                if not isinstance(label, str) or not label:
                    raise ValueError(
                        f"{path}:{line_number}/entities[{entity_index}]: invalid label"
                    )
                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or not 0 <= start < end
                ):
                    raise ValueError(
                        f"{path}:{line_number}/entities[{entity_index}]: invalid offsets"
                    )
            seen_hashes.add(record_hash)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no prediction records")
    return records


def load_comparison_records(gold_path: Path, predictions_path: Path) -> list[JsonObject]:
    """Join gold and prediction records by hash and validate their offsets."""

    gold_records = read_records(gold_path)
    prediction_records = read_prediction_records(predictions_path)
    predictions = {record["hash"]: record for record in prediction_records}
    gold_hashes = {record["hash"] for record in gold_records}
    prediction_hashes = set(predictions)
    if gold_hashes != prediction_hashes:
        raise ValueError(
            "gold/prediction hashes differ: "
            f"gold_only={len(gold_hashes - prediction_hashes)}, "
            f"prediction_only={len(prediction_hashes - gold_hashes)}"
        )

    comparisons: list[JsonObject] = []
    for record in gold_records:
        prediction = predictions[record["hash"]]
        text = record["text"]
        if "text" in prediction and prediction["text"] != text:
            raise ValueError(f"prediction text differs from gold for {record['hash']}")
        predicted_entities = prediction["entities"]
        for entity_index, entity in enumerate(predicted_entities):
            _validate_entity(
                entity,
                text,
                f"{predictions_path}:{record['hash']}/entities[{entity_index}]",
            )
        predicted_spans = sorted((entity["start"], entity["end"]) for entity in predicted_entities)
        if any(
            right[0] < left[1]
            for left, right in zip(predicted_spans, predicted_spans[1:], strict=False)
        ):
            raise ValueError(f"prediction entities overlap for {record['hash']}")
        comparisons.append(
            {
                "hash": record["hash"],
                "text": text,
                "gold_entities": record["entities"],
                "pred_entities": predicted_entities,
            }
        )
    return comparisons


def render_text(text: str, entities: list[Any], *, color: bool) -> str:
    """Render text with entity spans highlighted by label."""

    normalized = sorted(
        [
            (
                entity["label"],
                int(entity["start"]),
                int(entity["end"]),
            )
            for entity in entities
        ],
        key=lambda item: (item[1], item[2], item[0]),
    )
    parts: list[str] = []
    cursor = 0
    for label, start, end in normalized:
        parts.append(text[cursor:start])
        surface = text[start:end]
        if color:
            color_code = LABEL_COLORS.get(label, DEFAULT_COLOR)
            parts.append(f"{color_code}{label}:{surface}{RESET}")
        else:
            parts.append(f"[{label}:{surface}]")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def format_record(record: JsonObject, index: int, total: int, *, color: bool) -> str:
    """Format one sampled record for human inspection."""

    text = record["text"]
    entities = record["entities"]
    lines = [
        f"=== Sample {index}/{total} | hash={record.get('hash', '<missing>')} ===",
        f"Text: {render_text(text, entities, color=color)}",
        f"Entities: {len(entities)}",
    ]
    for entity in sorted(entities, key=lambda item: (item["start"], item["end"])):
        start = entity["start"]
        end = entity["end"]
        lines.append(f"  - {entity['label']}: [{start}:{end}] {text[start:end]!r}")
    return "\n".join(lines)


def _entity_key(entity: JsonObject) -> tuple[str, int, int]:
    """Return the comparable key of one entity."""

    return entity["label"], int(entity["start"]), int(entity["end"])


def _overlaps(left: JsonObject, right: JsonObject) -> bool:
    """Return whether two entity spans overlap."""

    return left["start"] < right["end"] and right["start"] < left["end"]


def _entity_status(entity: JsonObject, other_entities: list[JsonObject]) -> str:
    """Classify an entity as correct, missed, false-positive, or boundary error."""

    if _entity_key(entity) in {_entity_key(other) for other in other_entities}:
        return "correct"
    if any(_overlaps(entity, other) for other in other_entities):
        return "boundary_error"
    return "missed" if entity.get("source") == "gold" else "false_positive"


def _comparison_segments(
    text: str,
    gold_entities: list[JsonObject],
    predicted_entities: list[JsonObject],
) -> list[tuple[str, str | None]]:
    """Split text into regions colored by gold/prediction agreement."""

    boundaries = {0, len(text)}
    for entity in gold_entities + predicted_entities:
        boundaries.update((int(entity["start"]), int(entity["end"])))
    sorted_boundaries = sorted(boundaries)
    segments: list[tuple[str, str | None]] = []
    predicted_keys = {_entity_key(entity) for entity in predicted_entities}
    for start, end in zip(sorted_boundaries, sorted_boundaries[1:], strict=False):
        if start == end:
            continue
        gold = next(
            (entity for entity in gold_entities if entity["start"] <= start < entity["end"]),
            None,
        )
        predicted = next(
            (entity for entity in predicted_entities if entity["start"] <= start < entity["end"]),
            None,
        )
        if gold is not None and predicted is not None:
            status = "correct" if _entity_key(gold) in predicted_keys else "boundary_error"
        elif gold is not None:
            status = "missed"
        elif predicted is not None:
            status = "false_positive"
        else:
            status = None
        if segments and segments[-1][1] == status:
            segments[-1] = (segments[-1][0] + text[start:end], status)
        else:
            segments.append((text[start:end], status))
    return segments


def render_comparison_text(
    text: str,
    gold_entities: list[JsonObject],
    predicted_entities: list[JsonObject],
    *,
    color: bool,
) -> str:
    """Render text with colors showing gold/prediction agreement."""

    parts: list[str] = []
    for content, status in _comparison_segments(text, gold_entities, predicted_entities):
        if status is None:
            parts.append(content)
        elif color:
            parts.append(f"{STATUS_COLORS[status]}{content}{RESET}")
        else:
            parts.append(f"[{status}:{content}]")
    return "".join(parts)


def format_comparison_record(
    record: JsonObject,
    index: int,
    total: int,
    *,
    color: bool,
) -> str:
    """Format one gold/prediction comparison for human inspection."""

    text = record["text"]
    gold_entities = record["gold_entities"]
    predicted_entities = record["pred_entities"]
    lines = [
        f"=== Sample {index}/{total} | hash={record['hash']} ===",
        "Legend: green=correct, red=missed, blue=false positive, yellow=boundary or label error",
        "Text: "
        + render_comparison_text(
            text,
            gold_entities,
            predicted_entities,
            color=color,
        ),
        "Gold:",
    ]
    for entity in gold_entities:
        start = entity["start"]
        end = entity["end"]
        lines.append(
            f"  - {entity['label']}: [{start}:{end}] {text[start:end]!r} "
            f"({_entity_status({**entity, 'source': 'gold'}, predicted_entities)})"
        )
    lines.append("Predicted:")
    for entity in predicted_entities:
        start = entity["start"]
        end = entity["end"]
        lines.append(
            f"  - {entity['label']}: [{start}:{end}] {text[start:end]!r} "
            f"({_entity_status({**entity, 'source': 'prediction'}, gold_entities)})"
        )
    return "\n".join(lines)


def _svg_segments(text: str, entities: list[Any]) -> list[tuple[str, str | None]]:
    """Split text into plain and entity segments for SVG rendering."""

    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for entity in sorted(entities, key=lambda item: (item["start"], item["end"])):
        start = int(entity["start"])
        end = int(entity["end"])
        if start > cursor:
            segments.append((text[cursor:start], None))
        segments.append((text[start:end], str(entity["label"])))
        cursor = end
    if cursor < len(text):
        segments.append((text[cursor:], None))
    return segments


def _wrap_svg_segments(
    segments: list[tuple[str, str | None]],
    *,
    max_chars: int = 120,
) -> list[list[tuple[str, str | None]]]:
    """Wrap annotated text into SVG lines without losing entity colors."""

    lines: list[list[tuple[str, str | None]]] = [[]]
    line_length = 0
    for content, label in segments:
        for character in content:
            if character == "\n" or line_length >= max_chars:
                lines.append([])
                line_length = 0
                if character == "\n":
                    continue
            current = lines[-1]
            if current and current[-1][1] == label:
                current[-1] = (current[-1][0] + character, label)
            else:
                current.append((character, label))
            line_length += 1
    return [line for line in lines if line]


def render_svg(records: list[JsonObject]) -> str:
    """Render sampled records as a standalone SVG image."""

    width = 1600
    y = 42
    blocks: list[str] = []
    if any("gold_entities" in record for record in records):
        blocks.append(
            '<text x="40" y="28" font-family="Segoe UI, Arial, sans-serif" '
            'font-size="15" fill="#334155">Legend:</text>'
        )
        legend_x = 105
        for status, title in (
            ("correct", "correct"),
            ("missed", "missed"),
            ("false_positive", "false positive"),
            ("boundary_error", "boundary/label error"),
        ):
            blocks.append(
                f'<rect x="{legend_x}" y="16" width="12" height="12" '
                f'fill="{SVG_STATUS_COLORS[status]}"/>'
            )
            blocks.append(
                f'<text x="{legend_x + 17}" y="28" '
                'font-family="Segoe UI, Arial, sans-serif" font-size="15" '
                f'fill="#334155">{title}</text>'
            )
            legend_x += 150 if status != "boundary_error" else 220
        y += 30
    for index, record in enumerate(records, start=1):
        comparison = "gold_entities" in record
        if comparison:
            text_segments = _comparison_segments(
                record["text"],
                record["gold_entities"],
                record["pred_entities"],
            )
            entity_lines = len(record["gold_entities"]) + len(record["pred_entities"])
        else:
            text_segments = _svg_segments(record["text"], record["entities"])
            entity_lines = len(record["entities"])
        lines = _wrap_svg_segments(text_segments)
        comparison_details_height = 48 if comparison else 0
        block_height = (
            58 + len(lines) * 27 + comparison_details_height + max(1, entity_lines) * 24 + 20
        )
        top = y - 27
        blocks.append(
            f'<rect x="20" y="{top}" width="1560" height="{block_height}" '
            'rx="10" fill="#f8fafc" stroke="#cbd5e1"/>'
        )
        blocks.append(
            f'<text x="40" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="18" font-weight="600" fill="#0f172a">'
            f"Sample {index}/{len(records)} | "
            f"hash={escape(str(record.get('hash', '<missing>')))}</text>"
        )
        y += 30
        for line in lines:
            tspans: list[str] = []
            for content, label in line:
                palette = SVG_STATUS_COLORS if comparison else SVG_COLORS
                default_color = SVG_DEFAULT_COLOR if not comparison else "#0f172a"
                color = palette.get(label or "", default_color) if label else "#0f172a"
                tspans.append(f'<tspan fill="{color}">{escape(content)}</tspan>')
            blocks.append(
                f'<text x="40" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
                f'font-size="17">{"".join(tspans)}</text>'
            )
            y += 27
        if comparison:
            for title, entities in (
                ("Gold", record["gold_entities"]),
                ("Predicted", record["pred_entities"]),
            ):
                blocks.append(
                    f'<text x="40" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
                    f'font-size="15" font-weight="600" fill="#334155">{title}:</text>'
                )
                y += 24
                for entity in entities:
                    start = int(entity["start"])
                    end = int(entity["end"])
                    label = escape(str(entity["label"]))
                    surface = escape(record["text"][start:end])
                    status = _entity_status(
                        {**entity, "source": "gold" if title == "Gold" else "prediction"},
                        record["pred_entities"] if title == "Gold" else record["gold_entities"],
                    )
                    blocks.append(
                        f'<text x="40" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
                        f'font-size="15" fill="{SVG_STATUS_COLORS[status]}">'
                        f"{label}: [{start}:{end}] {surface} ({status})</text>"
                    )
                    y += 24
        elif not record["entities"]:
            blocks.append(
                f'<text x="40" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
                'font-size="15" fill="#64748b">Entities: 0</text>'
            )
            y += 24
        else:
            for entity in sorted(record["entities"], key=lambda item: (item["start"], item["end"])):
                start = int(entity["start"])
                end = int(entity["end"])
                label = escape(str(entity["label"]))
                surface = escape(record["text"][start:end])
                blocks.append(
                    f'<text x="40" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
                    f'font-size="15" fill="#334155">{label}: [{start}:{end}] '
                    f"{surface}</text>"
                )
                y += 24
        y += 20

    height = max(100, y)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>' + "".join(blocks) + "</svg>\n"
    )


def run(args: argparse.Namespace) -> int:
    """Sample records and write their highlighted representation."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.number <= 0:
        raise ValueError("number must be positive")

    comparison_mode = args.gold is not None or args.predictions is not None
    if comparison_mode:
        if args.input is not None:
            raise ValueError("input cannot be used together with --gold/--predictions")
        if args.gold is None or args.predictions is None:
            raise ValueError("--gold and --predictions must be used together")
        for path in (args.gold, args.predictions):
            if not path.is_file():
                raise FileNotFoundError(path)
        records = load_comparison_records(args.gold, args.predictions)
    else:
        if args.input is None:
            raise ValueError("input JSONL or --gold/--predictions is required")
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        records = read_records(args.input)
    if args.number > len(records):
        raise ValueError(f"number={args.number} exceeds available records={len(records)}")
    sampled = random.Random(args.seed).sample(records, args.number)
    if args.format == "svg":
        if args.output is None:
            raise ValueError("--format svg requires --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_svg(sampled), encoding="utf-8")
        print(f"Saved visualization: {args.output}")
        return 0

    color = not args.no_color and args.output is None
    if comparison_mode:
        output = "\n\n".join(
            format_comparison_record(record, index, len(sampled), color=color)
            for index, record in enumerate(sampled, start=1)
        )
    else:
        output = "\n\n".join(
            format_record(record, index, len(sampled), color=color)
            for index, record in enumerate(sampled, start=1)
        )

    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Saved visualization: {args.output}")
    return 0


def main() -> int:
    """Run the visualizer and report input errors."""

    try:
        return run(parse_args())
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
