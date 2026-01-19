from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="datagent_")

    db_path: str = "./db.duckdb"
    vec_size: int = 1024

settings = Settings()
