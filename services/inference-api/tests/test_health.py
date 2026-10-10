import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inference_api import main
from inference_api.schemas import ModelInfo

ROW = {
    "elevation": 3037,
    "aspect": 330,
    "slope": 12,
    "horizontal_distance_to_hydrology": 492,
    "vertical_distance_to_hydrology": -83,
    "horizontal_distance_to_roadways": 1806,
    "hillshade_9am": 192,
    "hillshade_noon": 226,
    "hillshade_3pm": 173,
    "horizontal_distance_to_fire_points": 2023,
    "wilderness_area": "commanche",
    "soil_type": "c7756",
}


class FakeModel:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        self.frames.append(frame)
        return pd.Series([3] * len(frame))


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    info = ModelInfo(
        model_name="cubierta-forestal",
        model_alias="production",
        model_version="36",
        run_id="abc",
    )
    monkeypatch.setattr(
        main, "load_model", lambda settings: main.LoadedModel(model, info)
    )
    return model


@pytest.fixture
def client(fake_model: FakeModel) -> TestClient:
    with TestClient(main.app) as c:
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "36"}


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_alias"] == "production"


def test_predict_returns_one_prediction_per_instance(
    client: TestClient, fake_model: FakeModel
) -> None:
    response = client.post("/predict", json={"instances": [ROW, ROW]})

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [3, 3],
        "model_name": "cubierta-forestal",
        "model_version": "36",
    }
    assert list(fake_model.frames[0].columns) == list(ROW)


def test_predict_rejects_missing_feature(client: TestClient) -> None:
    row = {k: v for k, v in ROW.items() if k != "soil_type"}

    response = client.post("/predict", json={"instances": [row]})

    assert response.status_code == 422


def test_predict_rejects_empty_batch(client: TestClient) -> None:
    response = client.post("/predict", json={"instances": []})

    assert response.status_code == 422
