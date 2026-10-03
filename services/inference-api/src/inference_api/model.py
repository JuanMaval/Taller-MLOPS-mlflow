"""Model Registry access: resolve models:/<name>@<alias> once and keep it in app state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from inference_api.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    pyfunc: Any  # mlflow.pyfunc.PyFuncModel (or a stand-in with the same predict contract)
    name: str
    alias: str
    version: str

    @property
    def uri(self) -> str:
        return f"models:/{self.name}@{self.alias}"


def load_from_registry(settings: Settings) -> LoadedModel:
    """Download and deserialise the model the alias points to. Raises on any failure."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    version = mlflow.MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)
    uri = f"models:/{settings.model_name}@{settings.model_alias}"
    pyfunc = mlflow.pyfunc.load_model(uri)
    logger.info("loaded %s (version %s)", uri, version.version)
    return LoadedModel(pyfunc=pyfunc, name=settings.model_name, alias=settings.model_alias, version=version.version)


def try_load(settings: Settings) -> LoadedModel | None:
    try:
        return load_from_registry(settings)
    except Exception:  # noqa: BLE001 - startup must not crash-loop on a missing model
        logger.exception("could not load models:/%s@%s", settings.model_name, settings.model_alias)
        return None


def get_model(request: Request) -> LoadedModel | None:
    """FastAPI dependency. Retries the registry lazily if startup could not load the model."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        model = try_load(request.app.state.settings)
        request.app.state.model = model
    return model
