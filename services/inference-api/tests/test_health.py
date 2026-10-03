from fastapi.testclient import TestClient

from inference_api.main import app
from inference_api.model import get_model


def test_health_without_model_reports_degraded() -> None:
    app.dependency_overrides[get_model] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "model": None, "version": None}
