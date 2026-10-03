"""Request / response contracts for the prediction endpoint.

The input is the RAW feature set the registered model was signed with
(see inference_api.covertype_model.RAW_FEATURES): feature engineering lives
inside the model, so callers send what the business database stores.
"""

from pydantic import BaseModel, ConfigDict, Field


class CovertypeFeatures(BaseModel):
    elevation: int = Field(description="Elevation in meters")
    aspect: int = Field(ge=0, le=360, description="Aspect in degrees azimuth")
    slope: int = Field(ge=0, le=90, description="Slope in degrees")
    horizontal_distance_to_hydrology: int = Field(ge=0)
    vertical_distance_to_hydrology: int
    horizontal_distance_to_roadways: int = Field(ge=0)
    hillshade_9am: int = Field(ge=0, le=255)
    hillshade_noon: int = Field(ge=0, le=255)
    hillshade_3pm: int = Field(ge=0, le=255)
    horizontal_distance_to_fire_points: int = Field(ge=0)
    wilderness_area: str = Field(min_length=1, description="Wilderness area name, e.g. 'rawah'")
    soil_type: str = Field(min_length=1, description="Soil type code, e.g. 'c7745'")


# Two real rows from the `test` split of training.covertype_features. The
# registered model predicts cover_type 1 for the first and 2 for the second.
PREDICT_EXAMPLE: dict = {
    "instances": [
        {
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
        },
        {
            "elevation": 2616,
            "aspect": 176,
            "slope": 27,
            "horizontal_distance_to_hydrology": 351,
            "vertical_distance_to_hydrology": 88,
            "horizontal_distance_to_roadways": 927,
            "hillshade_9am": 223,
            "hillshade_noon": 242,
            "hillshade_3pm": 134,
            "horizontal_distance_to_fire_points": 1047,
            "wilderness_area": "commanche",
            "soil_type": "c2703",
        },
    ]
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [PREDICT_EXAMPLE]})

    instances: list[CovertypeFeatures] = Field(min_length=1, max_length=10_000)


class Prediction(BaseModel):
    cover_type: int = Field(ge=0, le=6, description="Predicted forest cover type (0-6)")
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted class")


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: ModelInfo


class HealthResponse(BaseModel):
    status: str
    model: str | None
    version: str | None
