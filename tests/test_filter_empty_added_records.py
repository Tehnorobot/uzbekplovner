import pytest

from scripts.filter_empty_added_records import filter_empty_added_records


def _record(record_hash: str, *, empty: bool) -> dict[str, object]:
    return {
        "hash": record_hash,
        "text": "Ali",
        "entities": [] if empty else [{"label": "NAME", "start": 0, "end": 3}],
    }


def test_filter_removes_only_empty_added_records() -> None:
    records = [
        _record("original-empty-one", empty=True),
        _record("mendeley-empty-one", empty=True),
        _record("uzner-v2-empty-one", empty=True),
        _record("uzner-v2-name", empty=False),
        _record("original-empty-two", empty=True),
        _record("uzner-v2-empty-two", empty=True),
        _record("uzner-v2-empty-three", empty=True),
    ]

    filtered, stats = filter_empty_added_records(
        records,
        drop_fraction=0.75,
        seed=42,
    )

    hashes = {record["hash"] for record in filtered}
    assert stats["eligible_empty_added_records"] == 4
    assert stats["removed_records"] == 3
    assert "original-empty-one" in hashes
    assert "original-empty-two" in hashes
    assert "uzner-v2-name" in hashes
    assert stats["entities_by_label_before"] == stats["entities_by_label_after"]
    assert [record["hash"] for record in filtered] == [
        record["hash"] for record in records if record["hash"] in hashes
    ]


def test_filter_is_deterministic_for_the_same_seed() -> None:
    records = [_record(f"uzner-v2-{index}", empty=True) for index in range(20)]

    first, _ = filter_empty_added_records(records, drop_fraction=0.75, seed=7)
    second, _ = filter_empty_added_records(records, drop_fraction=0.75, seed=7)

    assert first == second


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_filter_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="drop_fraction"):
        filter_empty_added_records([], drop_fraction=fraction, seed=42)
