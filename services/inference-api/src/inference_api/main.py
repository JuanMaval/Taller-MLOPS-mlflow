import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient

from inference_api import __version__
from inference_api.config import Settings
from inference_api.schemas import ModelInfo, PredictRequest, PredictResponse

logger = logging.getLogger("uvicorn.error")


@dataclass
class LoadedModel:
    model: mlflow.pyfunc.PyFuncModel
    info: ModelInfo


def load_model(settings: Settings) -> LoadedModel:
    """Resolve the registry alias to a concrete version and load that version.

    Loading by version (not by alias) keeps ``/model-info`` consistent with the
    model actually in memory even if the alias moves after startup.
    """
    version = MlflowClient().get_model_version_by_alias(
        settings.model_name, settings.model_alias
    )
    model = mlflow.pyfunc.load_model(
        f"models:/{settings.model_name}/{version.version}"
    )
    info = ModelInfo(
        model_name=settings.model_name,
        model_alias=settings.model_alias,
        model_version=str(version.version),
        run_id=version.run_id,
    )
    logger.info("Loaded model %s", info)
    return LoadedModel(model=model, info=info)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail fast: if the model cannot be loaded the process exits and the
    # container restarts instead of reporting healthy without a model.
    app.state.loaded = load_model(Settings())
    yield


app = FastAPI(
    title="Cubierta Forestal · Inference API",
    version=__version__,
    lifespan=lifespan,
)


def _loaded(request: Request) -> LoadedModel:
    loaded: LoadedModel | None = getattr(request.app.state, "loaded", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return loaded


@app.get("/health", tags=["ops"])
def health(request: Request) -> dict[str, str]:
    """Readiness probe: 200 only when the model is loaded."""
    loaded = _loaded(request)
    return {"status": "ok", "model_version": loaded.info.model_version}


@app.get("/model-info", tags=["ops"])
def model_info(request: Request) -> ModelInfo:
    return _loaded(request).info


@app.post("/predict", tags=["inference"])
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Predict ``cover_type`` for one or more observations."""
    loaded = _loaded(request)
    frame = pd.DataFrame([row.model_dump() for row in body.instances])
    predictions = loaded.model.predict(frame)
    return PredictResponse(
        predictions=[int(p) for p in predictions],
        model_name=loaded.info.model_name,
        model_version=loaded.info.model_version,
    )
