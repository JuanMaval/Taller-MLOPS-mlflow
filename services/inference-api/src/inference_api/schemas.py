"""Request / response contracts for the prediction endpoint.

The input is the RAW feature set the registered model was signed with
(see inference_api.covertype_model.RAW_FEATURES): feature engineering lives
inside the model, so callers send what the business database stores.
"""

from pydantic import BaseModel, Field


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


class PredictRequest(BaseModel):
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
