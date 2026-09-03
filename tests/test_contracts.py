from ner_core.contracts import ContractError, validate_entities, validate_input_batch


def test_unicode_offsets_are_validated_as_python_character_indexes() -> None:
    text = "Али Toshkent"
    entities = validate_entities(
        [{"label": "NAME", "start": 0, "end": 3}],
        text,
        source="test",
    )
    assert text[entities[0]["start"] : entities[0]["end"]] == "Али"


def test_overlapping_entities_are_rejected() -> None:
    try:
        validate_entities(
            [
                {"label": "NAME", "start": 0, "end": 4},
                {"label": "GEO", "start": 3, "end": 8},
            ],
            "Ali Toshkent",
            source="test",
        )
    except ContractError:
        return
    raise AssertionError("overlapping entities must be rejected")


def test_input_batch_preserves_order_and_rejects_duplicate_hashes() -> None:
    items = validate_input_batch(
        [
            {"hash": "a", "text": "one"},
            {"hash": "b", "text": "two"},
        ]
    )
    assert [item["hash"] for item in items] == ["a", "b"]

    try:
        validate_input_batch(
            [
                {"hash": "a", "text": "one"},
                {"hash": "a", "text": "two"},
            ]
        )
    except ContractError:
        return
    raise AssertionError("duplicate hashes must be rejected")
