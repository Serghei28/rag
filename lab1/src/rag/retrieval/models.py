"""Структуры данных этапа retrieval."""

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel
from qdrant_client import models

from rag.preprocessing.models import Chunk


@dataclass
class Candidate:
    chunk: Chunk
    vector_score: float  # cosine similarity из векторного поиска
    vector_rank: int  # место в выдаче векторного поиска (с 1)
    rerank_score: float | None = None  # релевантность по reranker'у (0..1)

    @property
    def score(self) -> float:
        """Итоговый score: reranker'а, если он был, иначе векторный."""
        return self.vector_score if self.rerank_score is None else self.rerank_score


@dataclass
class RetrievalResult:
    query: str
    collection: str
    chunks: list[Candidate]  # итог: уходит в LLM
    candidates: list[Candidate]  # все кандидаты векторного поиска, до фильтров
    dropped: dict[str, list[Candidate]] = field(default_factory=dict)  # причина → кандидаты
    timings_ms: dict[str, float] = field(default_factory=dict)
    reranker: str | None = None

    @property
    def is_empty(self) -> bool:
        """Ничего релевантного не нашлось — сигнал для ответа «информации недостаточно»."""
        return not self.chunks


class MetadataFilter(BaseModel):
    """Фильтр по metadata. Применяется в самом Qdrant, до отбора top_k."""

    titles: list[str] = []  # искать только в этих статьях
    exclude_titles: list[str] = []
    updated_after: datetime | None = None  # статьи, отредактированные после даты

    def to_qdrant(self) -> models.Filter | None:
        must, must_not = [], []
        if self.titles:
            must.append(models.FieldCondition(key="title", match=models.MatchAny(any=self.titles)))
        if self.exclude_titles:
            must_not.append(
                models.FieldCondition(key="title", match=models.MatchAny(any=self.exclude_titles))
            )
        if self.updated_after:
            must.append(
                models.FieldCondition(
                    key="updated_at", range=models.DatetimeRange(gt=self.updated_after)
                )
            )
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)
