"""Конфигурация проекта.

Два источника настроек:
- `.env` — секреты и адреса сервисов (ключи Langfuse, адрес Ollama). Не коммитится.
- `configs/*.yaml` — параметры pipeline и экспериментов. Коммитятся, чтобы любой
  эксперимент можно было воспроизвести по имени конфига.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
        return PathsConfig(**{name: p if p.is_absolute() else root / p for name, p in self})


class CategorySeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    depth: int = 1  # 0 — только статьи самой категории, 1 — плюс подкатегории и т.д.


class GrabberConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_url: str = "https://ru.wikipedia.org/w/api.php"
    # Wikimedia требует User-Agent с контактом: https://meta.wikimedia.org/wiki/User-Agent_policy
    user_agent: str = "rag-md-lab/0.1 (https://github.com/Serghei28/rag; student project)"
    categories: list[CategorySeed] = []
    titles: list[str] = []
    exclude_categories: list[str] = []  # регулярные выражения по названию подкатегории
    exclude_titles: list[str] = []  # регулярные выражения по названию статьи
    max_pages: int | None = None
    request_delay: float = 0.1  # пауза между запросами, секунды
    timeout: float = 30.0
    max_retries: int = 5


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Служебные секции статей, в которых нет содержательного текста
    drop_sections: list[str] = [
        "Примечания",
        "Комментарии",
        "Ссылки",
        "Литература",
        "Источники",
        "См. также",
        "Галерея",
    ]
    keep_infobox: bool = True  # карточка статьи → строки «ключ: значение»
    keep_tables: bool = True  # таблицы → строки «ячейка | ячейка»


class ChunkingStrategy(BaseModel):
    """Одна стратегия разбиения. Размеры — в символах (≈ 4 символа на токен для русского)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    method: Literal["fixed", "paragraph", "section"]
    size: int = 800
    overlap: int = 0  # только для method=fixed
    min_chars: int = 40  # более короткие chunks отбрасываются


class AppConfig(BaseModel):
    """Параметры pipeline. Новые секции добавляются по мере реализации этапов."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    grabber: GrabberConfig = GrabberConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    chunking: list[ChunkingStrategy] = []


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)
    config.paths = config.paths.resolve(PROJECT_ROOT)
    return config


@lru_cache
def get_settings() -> Settings:
    return Settings()
