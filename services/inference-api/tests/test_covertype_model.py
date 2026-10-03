import numpy as np
import pandas as pd
import pytest

from inference_api.covertype_model import (
    CATEGORICAL_FEATURES,
    CovertypeModel,
    RAW_FEATURES,
    engineer_features,
    predict_frame,
)


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "elevation": [2596, 2804],
            "aspect": [51, 139],
            "slope": [3, 9],
            "horizontal_distance_to_hydrology": [3, 4],
            "vertical_distance_to_hydrology": [4, 3],
            "horizontal_distance_to_roadways": [510, 3180],
            "hillshade_9am": [221, 234],
            "hillshade_noon": [232, 238],
            "hillshade_3pm": [148, 122],
            "horizontal_distance_to_fire_points": [6279, 6121],
            "wilderness_area": ["wilderness_1", "wilderness_1"],
            "soil_type": ["soil_29", "soil_30"],
        }
    )


class _FakePredictor:
    """Stands in for an AutoGluon TabularPredictor."""

    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict(self, data: pd.DataFrame) -> pd.Series:
        self.seen = data
        return pd.Series([1, 4], index=data.index)

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({0: [0.1, 0.2], 1: [0.8, 0.1], 4: [0.1, 0.7]}, index=data.index)


def test_engineer_features_adds_euclidean_distance_and_categories() -> None:
    out = engineer_features(_raw_rows())

    assert list(out.columns) == RAW_FEATURES + ["euclidean_distance_to_hydrology"]
    assert out["euclidean_distance_to_hydrology"].tolist() == [5.0, 5.0]
    for column in CATEGORICAL_FEATURES:
        assert str(out[column].dtype) == "category"


def test_engineer_features_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="soil_type"):
        engineer_features(_raw_rows().drop(columns=["soil_type"]))


def test_predict_returns_label_and_confidence_from_raw_rows() -> None:
    model = CovertypeModel()
    fake = _FakePredictor()
    model._predictor = fake

    out = model.predict(context=None, model_input=_raw_rows())

    assert list(out.columns) == ["cover_type", "confidence"]
    assert out["cover_type"].tolist() == [1, 4]
    assert np.allclose(out["confidence"].tolist(), [0.8, 0.7])
    assert "euclidean_distance_to_hydrology" in fake.seen.columns


def test_predict_frame_matches_wrapper_output() -> None:
    out = predict_frame(_FakePredictor(), _raw_rows())

    assert out["cover_type"].tolist() == [1, 4]
    assert np.allclose(out["confidence"].tolist(), [0.8, 0.7])
