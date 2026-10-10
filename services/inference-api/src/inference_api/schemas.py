from pydantic import BaseModel, Field

# First row of the input example logged with the production model.
_EXAMPLE = {
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


class CovertypeFeatures(BaseModel):
    """One observation; mirrors the signature of the registered model."""

    model_config = {"json_schema_extra": {"example": _EXAMPLE}}

    elevation: int
    aspect: int
    slope: int
    horizontal_distance_to_hydrology: int
    vertical_distance_to_hydrology: int
    horizontal_distance_to_roadways: int
    hillshade_9am: int
    hillshade_noon: int
    hillshade_3pm: int
    horizontal_distance_to_fire_points: int
    wilderness_area: str
    soil_type: str


class PredictRequest(BaseModel):
    instances: list[CovertypeFeatures] = Field(min_length=1)


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predictions: list[int]
    model_name: str
    model_version: str


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    model_alias: str
    model_version: str
    run_id: str | None
