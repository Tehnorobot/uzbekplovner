"""Shared FastAPI adapter for experiment predictors."""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .contracts import ContractError, validate_input_batch, validate_prediction_batch
from .normalization import build_alias_index, normalize_prediction_batch

Predictor = Callable[[list[dict[str, str]]], list[dict[str, Any]]]


def create_app(
    predictor: Predictor,
    *,
    normalization_aliases: dict[str, dict[str, str]] | None = None,
) -> FastAPI:
    """Create mandatory inference endpoints and canonicalized entity output."""

    app = FastAPI(title="NER inference service", version="1.0")
    alias_index = build_alias_index(normalization_aliases or {})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    async def run_prediction(request: Request, *, normalize: bool) -> JSONResponse:
        try:
            inputs = validate_input_batch(await request.json())
            predictions = validate_prediction_batch(predictor(inputs), inputs)
            data = (
                normalize_prediction_batch(inputs, predictions, alias_index)
                if normalize
                else predictions
            )
        except (ContractError, TypeError, ValueError) as error:
            return JSONResponse(status_code=400, content={"detail": str(error)})
        except RuntimeError as error:
            return JSONResponse(status_code=503, content={"detail": str(error)})
        return JSONResponse(content={"data": data})

    @app.post("/api/v1/predict")
    async def predict(request: Request) -> JSONResponse:
        return await run_prediction(request, normalize=False)

    @app.post("/api/v1/predict/normalized")
    async def predict_normalized(request: Request) -> JSONResponse:
        return await run_prediction(request, normalize=True)

    return app
