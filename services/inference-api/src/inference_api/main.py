from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Body, Depends, FastAPI, HTTPException, Response, status

from inference_api import __version__
from inference_api.model import LoadedModel, get_model, try_load
from inference_api.schemas import (
    PREDICT_EXAMPLE,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    Prediction,
)
from inference_api.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.model = try_load(settings) if settings.load_model_on_startup else None
    yield


app = FastAPI(
    title="Cubierta Forestal · Inference API",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"], response_model=HealthResponse)
def health(response: Response, model: LoadedModel | None = Depends(get_model)) -> HealthResponse:
    """Liveness probe used by the container healthcheck: unhealthy until a model is loaded."""
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", model=None, version=None)
    return HealthResponse(status="ok", model=f"{model.name}@{model.alias}", version=model.version)


@app.post("/predict", tags=["inference"], response_model=PredictResponse)
def predict(
    payload: PredictRequest = Body(
        openapi_examples={
            "two-test-rows": {
                "summary": "Two real rows from the test split",
                "description": "Expected prediction: cover_type 1 for the first row, 2 for the second.",
                "value": PREDICT_EXAMPLE,
            }
        }
    ),
    model: LoadedModel | None = Depends(get_model),
) -> PredictResponse:
    """Predict the forest cover type for a batch of raw feature rows."""
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not available in the registry")
    frame = pd.DataFrame([instance.model_dump() for instance in payload.instances])
    output = model.pyfunc.predict(frame)
    return PredictResponse(
        predictions=[
            Prediction(cover_type=int(row.cover_type), confidence=float(row.confidence))
            for row in output.itertuples(index=False)
        ],
        model=ModelInfo(name=model.name, alias=model.alias, version=model.version),
    )
