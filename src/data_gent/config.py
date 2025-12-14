from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="datagent_")

    db_path: str = "/data-dent/db.duckdb"
    docs_dir: str = "/data-gent/load/docs"
    vec_size: int = 128

settings = Settings()