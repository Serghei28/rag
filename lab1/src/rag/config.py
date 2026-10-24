"""Конфигурация проекта.

Два источника настроек:
- `.env` — секреты и адреса сервисов (ключи Langfuse, адрес Ollama). Не коммитится.
- `configs/*.yaml` — параметры pipeline и экспериментов. Коммитятся, чтобы любой
  эксперимент можно было воспроизвести по имени конфига.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Settings(BaseSettings):
    """Секреты и адреса внешних сервисов из `.env` / переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    ollama_host: str = "http://localhost:11434"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")
    experiments_dir: Path = Path("experiments")

    def resolve(self, root: Path) -> "PathsConfig":
        """Сделать относительные пути абсолютными относительно корня проекта."""
        return PathsConfig(
            **{name: p if p.is_absolute() else root / p for name, p in self}
        )


class AppConfig(BaseModel):
    """Параметры pipeline. Новые секции добавляются по мере реализации этапов."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)
    config.paths = config.paths.resolve(PROJECT_ROOT)
    return config


@lru_cache
def get_settings() -> Settings:
    return Settings()
