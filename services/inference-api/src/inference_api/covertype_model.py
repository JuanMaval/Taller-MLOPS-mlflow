"""MLflow pyfunc wrapper around the AutoGluon Covertype predictor.

Shared by training (Jupyter logs and registers the model) and serving (the API
loads it from the Model Registry). The model input contract is the RAW feature
set below: the deterministic feature engineering used in training is applied
here, so callers never have to replicate it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import mlflow.pyfunc

NUMERIC_FEATURES: list[str] = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
]
CATEGORICAL_FEATURES: list[str] = ["wilderness_area", "soil_type"]
RAW_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cover_type"


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, fit-free transformations identical to the training notebook."""
    missing = [c for c in RAW_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    out = raw[RAW_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype(str).str.strip().str.lower().astype("category")
    out["euclidean_distance_to_hydrology"] = np.sqrt(
        out["horizontal_distance_to_hydrology"].astype(float) ** 2
        + out["vertical_distance_to_hydrology"].astype(float) ** 2
    ).round(3)
    return out


def predict_frame(predictor: Any, raw: pd.DataFrame) -> pd.DataFrame:
    """Raw features in, one row per input with the predicted class and its probability.

    `predictor` is any object exposing AutoGluon's `predict` / `predict_proba`.
    """
    features = engineer_features(raw)
    labels = predictor.predict(features)
    proba = predictor.predict_proba(features)
    return pd.DataFrame(
        {
            TARGET: labels.astype(int).to_numpy(),
            "confidence": proba.max(axis=1).astype(float).to_numpy(),
        },
        index=raw.index,
    )


class CovertypeModel(mlflow.pyfunc.PythonModel):
    """pyfunc flavor. The predictor is loaded from the logged artifacts, never pickled."""

    def __init__(self) -> None:
        self._predictor: Any = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from autogluon.tabular import TabularPredictor  # heavy import, serving only

        self._predictor = TabularPredictor.load(context.artifacts["predictor"], require_py_version_match=False)

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        return predict_frame(self._predictor, model_input)
