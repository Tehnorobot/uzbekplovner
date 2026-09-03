"""Shared FastAPI adapter for experiment predictors."""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .contracts import ContractError, validate_input_batch, validate_prediction_batch

Predictor = Callable[[list[dict[str, str]]], list[dict[str, Any]]]


def create_app(predictor: Predictor) -> FastAPI:
    """Create the mandatory `/healthz` and `/api/v1/predict` endpoints."""

    app = FastAPI(title="NER inference service", version="1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/predict")
    async def predict(request: Request) -> JSONResponse:
        try:
            inputs = validate_input_batch(await request.json())
            predictions = validate_prediction_batch(predictor(inputs), inputs)
        except (ContractError, TypeError, ValueError) as error:
            return JSONResponse(status_code=400, content={"detail": str(error)})
        except RuntimeError as error:
            return JSONResponse(status_code=503, content={"detail": str(error)})
        return JSONResponse(content={"data": predictions})

    return app
