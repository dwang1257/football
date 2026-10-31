from functools import cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOOTBALL_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    repo_dir: Path = Path(".")
    seed: int = 20261016
    http_timeout_s: float = 30.0
    user_agent: str = "football-predictor/0.1 (personal research)"
    opinion_weight: float = 0.12
    opinion_team_weight: float = 0.5
    opinion_cap: float = 0.35
    opinion_cap_max_conf: float = 0.5


@cache
def get_settings() -> Settings:
    return Settings()
