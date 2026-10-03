import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inference_api.main import app
from inference_api.model import LoadedModel, get_model


class _FakePyfunc:
    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        self.seen = data
        return pd.DataFrame({"cover_type": [1, 4][: len(data)], "confidence": [0.98, 0.71][: len(data)]})


RAW_INSTANCE = {
    "elevation": 2596,
    "aspect": 51,
    "slope": 3,
    "horizontal_distance_to_hydrology": 258,
    "vertical_distance_to_hydrology": 0,
    "horizontal_distance_to_roadways": 510,
    "hillshade_9am": 221,
    "hillshade_noon": 232,
    "hillshade_3pm": 148,
    "horizontal_distance_to_fire_points": 6279,
    "wilderness_area": "rawah",
    "soil_type": "c7745",
}


@pytest.fixture
def fake() -> _FakePyfunc:
    return _FakePyfunc()


@pytest.fixture
def client(fake: _FakePyfunc):
    app.dependency_overrides[get_model] = lambda: LoadedModel(
        pyfunc=fake, name="cubierta-forestal", alias="production", version="7"
    )
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_predict_returns_one_prediction_per_instance(client: TestClient, fake: _FakePyfunc) -> None:
    second = {**RAW_INSTANCE, "elevation": 2804, "wilderness_area": "commanche"}

    response = client.post("/predict", json={"instances": [RAW_INSTANCE, second]})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == {"name": "cubierta-forestal", "alias": "production", "version": "7"}
    assert body["predictions"] == [
        {"cover_type": 1, "confidence": 0.98},
        {"cover_type": 4, "confidence": 0.71},
    ]
    assert list(fake.seen.columns) == list(RAW_INSTANCE)
    assert fake.seen["wilderness_area"].tolist() == ["rawah", "commanche"]


def test_predict_rejects_missing_feature(client: TestClient) -> None:
    incomplete = {k: v for k, v in RAW_INSTANCE.items() if k != "soil_type"}

    response = client.post("/predict", json={"instances": [incomplete]})

    assert response.status_code == 422
    assert "soil_type" in response.text


def test_predict_rejects_empty_batch(client: TestClient) -> None:
    response = client.post("/predict", json={"instances": []})

    assert response.status_code == 422


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "cubierta-forestal@production", "version": "7"}


def test_openapi_example_is_a_valid_request(client: TestClient) -> None:
    from inference_api.schemas import PREDICT_EXAMPLE, PredictRequest

    PredictRequest.model_validate(PREDICT_EXAMPLE)  # must not raise

    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]
    assert body["examples"]["two-test-rows"]["value"] == PREDICT_EXAMPLE

    response = client.post("/predict", json=PREDICT_EXAMPLE)
    assert response.status_code == 200
