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


class EmbeddingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)  # frozen → можно кэшировать

    name: str  # короткое имя: входит в название коллекции
    model_id: str  # id на Hugging Face
    query_prefix: str = ""  # e5 обучена с префиксами "query: " / "passage: "
    passage_prefix: str = ""
    max_seq_length: int = 512  # токенов; длиннее — обрезается
    fp16: bool = True  # половинная точность на GPU: в ~2 раза быстрее, качество то же


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[EmbeddingModel] = []
    batch_size: int = 32
    device: str = "auto"  # auto → cuda, если есть, иначе cpu
    # Добавлять к тексту chunk заголовок «Статья. Раздел» перед векторизацией
    with_header: bool = True


class RetrievalConfig(BaseModel):
    """Первичный поиск и фильтры до reranker'а."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = 20  # сколько кандидатов достаём из векторного поиска
    # Порог cosine similarity для каждой embedding-модели: у моделей разные шкалы score
    score_thresholds: dict[str, float] = {}
    min_chars: int = 0  # отбросить слишком короткие chunks
    dedup_threshold: float | None = 0.8  # Jaccard по словам; None — не дедуплицировать
    max_per_document: int | None = None  # не больше N chunks из одной статьи


class RerankerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    model_id: str
    max_length: int = 512  # токенов на пару (вопрос + chunk)
    fp16: bool = True


class RerankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    models: list[RerankerModel] = []
    top_n: int = 5  # сколько chunks уходит в LLM
    # Порог релевантности после reranker'а (0..1) для каждой модели; нет ключа — не отсекать
    score_thresholds: dict[str, float] = {}
    batch_size: int = 16


class ActiveConfig(BaseModel):
    """Какая комбинация используется RAG по умолчанию (UI, генерация)."""

    model_config = ConfigDict(extra="forbid")

    embedding: str = ""
    chunking: str = ""
    reranker: str = ""


class AppConfig(BaseModel):
    """Параметры pipeline. Новые секции добавляются по мере реализации этапов."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    grabber: GrabberConfig = GrabberConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    chunking: list[ChunkingStrategy] = []
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    active: ActiveConfig = ActiveConfig()

    def reranker_model(self, name: str) -> RerankerModel:
        for m in self.reranking.models:
            if m.name == name:
                return m
        raise KeyError(f"нет reranker'а {name!r} в reranking.models")

    def embedding_model(self, name: str) -> EmbeddingModel:
        for m in self.embeddings.models:
            if m.name == name:
                return m
        raise KeyError(f"нет модели {name!r} в embeddings.models")

    def chunking_strategy(self, name: str) -> ChunkingStrategy:
        for s in self.chunking:
            if s.name == name:
                return s
        raise KeyError(f"нет стратегии {name!r} в chunking")


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)
    config.paths = config.paths.resolve(PROJECT_ROOT)
    return config


@lru_cache
def get_settings() -> Settings:
    return Settings()
