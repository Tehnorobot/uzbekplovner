from collections import Counter

from scripts.merge_uzner_style import _convert_sentence


def test_bioes_rows_become_exact_target_spans() -> None:
    rows = [
        (1, "source", "style", "Sentence 1", "Ali", "S-PER", "general"),
        (2, "source", "style", "Sentence 1", "Toshkentda", "S-LOC", "general"),
        (3, "source", "style", "Sentence 1", "ishlaydi", "O", "general"),
        (4, "source", "style", "Sentence 1", ".", "O", "general"),
    ]
    record = _convert_sentence("Dataset_1", "Sentence 1", rows, Counter())

    assert record["text"] == "Ali Toshkentda ishlaydi."
    assert [
        (entity["label"], record["text"][entity["start"] : entity["end"]])
        for entity in record["entities"]
    ] == [("NAME", "Ali"), ("GEO", "Toshkentda")]


def test_unsupported_source_entities_are_dropped_and_counted() -> None:
    counters = Counter()
    rows = [(1, "source", "style", "Sentence 1", "2026", "S-DATE", "general")]

    record = _convert_sentence("Dataset_1", "Sentence 1", rows, counters)

    assert record["entities"] == []
    assert counters["dropped_source_entities:DATE"] == 1
