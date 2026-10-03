"""Runtime configuration, read from the environment (see docker-compose.yml)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5001"
    model_name: str = "cubierta-forestal"
    model_alias: str = "production"
    # Set to false in tests so the app starts without a tracking server.
    load_model_on_startup: bool = True
