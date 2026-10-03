# Cubierta Forestal · Inference API (s6)

FastAPI service that serves the model registered in MLflow as
`models:/<MODEL_NAME>@<MODEL_ALIAS>` (defaults: `cubierta-forestal@production`).
The alias is resolved once at startup; promote a new version by moving the alias
in the Model Registry and restarting the service.

## Endpoints

| Method | Path       | Purpose |
|--------|------------|---------|
| GET    | `/health`  | `200 {"status":"ok", "model":"cubierta-forestal@production", "version":"1"}` once the model is loaded, `503 {"status":"degraded"}` otherwise (the container healthcheck relies on it). |
| POST   | `/predict` | Batch prediction, 1 to 10 000 instances per call. |
| GET    | `/docs`    | OpenAPI UI. |

### `POST /predict`

Input is the **raw** feature set stored in `training.covertype_features`
(10 integers + 2 strings). Feature engineering and categorical encoding live
inside the registered model, so callers never replicate them.

```bash
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "instances": [
    {"elevation": 3037, "aspect": 330, "slope": 12,
     "horizontal_distance_to_hydrology": 492, "vertical_distance_to_hydrology": -83,
     "horizontal_distance_to_roadways": 1806,
     "hillshade_9am": 192, "hillshade_noon": 226, "hillshade_3pm": 173,
     "horizontal_distance_to_fire_points": 2023,
     "wilderness_area": "commanche", "soil_type": "c7756"},
    {"elevation": 2616, "aspect": 176, "slope": 27,
     "horizontal_distance_to_hydrology": 351, "vertical_distance_to_hydrology": 88,
     "horizontal_distance_to_roadways": 927,
     "hillshade_9am": 223, "hillshade_noon": 242, "hillshade_3pm": 134,
     "horizontal_distance_to_fire_points": 1047,
     "wilderness_area": "commanche", "soil_type": "c2703"}
  ]
}'
```

Both rows come from the `test` split of `training.covertype_features`
(true classes 1 and 2). The same payload is the request example in `/docs`.

```json
{
  "predictions": [
    {"cover_type": 1, "confidence": 0.99996},
    {"cover_type": 2, "confidence": 0.99998}
  ],
  "model": {"name": "cubierta-forestal", "alias": "production", "version": "1"}
}
```

`cover_type` is the 0-6 class used in training; `confidence` is the probability
of the predicted class. Missing or out-of-range fields return `422`.

## Configuration (environment)

| Variable                | Default                  |
|-------------------------|--------------------------|
| `MLFLOW_TRACKING_URI`   | `http://localhost:5001`  |
| `MODEL_NAME`            | `cubierta-forestal`      |
| `MODEL_ALIAS`           | `production`             |
| `LOAD_MODEL_ON_STARTUP` | `true` (tests set false) |

## Development

```bash
uv sync --group api --group dev
uv run --group dev pytest -q          # fake model injected through dependency overrides
uv run uvicorn inference_api.main:app --reload   # needs a reachable MLflow with the alias set
```

Dependencies are shared with the training notebook through this `pyproject.toml`
(`uv.lock`), so the library versions that serialise a model are the ones that
deserialise it. The image needs `libgomp1` because the model's LightGBM booster
loads the OpenMP runtime at import time.
