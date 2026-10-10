from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration read from the environment (see docker-compose)."""

    model_config = SettingsConfigDict(protected_namespaces=())

    model_name: str = "cubierta-forestal"
    # Registry alias resolved at startup: models:/<model_name>@<model_alias>
    model_alias: str = "production"
