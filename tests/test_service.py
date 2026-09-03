from fastapi.testclient import TestClient

from ner_core.service import create_app


def test_http_adapter_returns_contract_compatible_response() -> None:
    def predictor(items: list[dict[str, str]]) -> list[dict[str, object]]:
        return [
            {
                "hash": item["hash"],
                "entities": (
                    [{"label": "NAME", "start": 0, "end": 3}]
                    if item["text"].startswith("Ali")
                    else []
                ),
            }
            for item in items
        ]

    client = TestClient(create_app(predictor))
    assert client.get("/healthz").json() == {"status": "ok"}
    response = client.post(
        "/api/v1/predict",
        json=[{"hash": "x", "text": "Ali works."}],
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["entities"] == [{"label": "NAME", "start": 0, "end": 3}]


def test_http_adapter_rejects_invalid_request() -> None:
    client = TestClient(create_app(lambda _: []))
    response = client.post("/api/v1/predict", json=[])
    assert response.status_code == 400
