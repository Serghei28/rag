"""Фильтры контекста между векторным поиском и reranker'ом.

Каждый фильтр — чистая функция `(кандидаты) → (оставленные, отброшенные)`. Порядок
кандидатов сохраняется. Отброшенные возвращаются, чтобы pipeline мог показать, почему
chunk не попал в контекст (в UI, Langfuse и отчёте об экспериментах).
"""

import re

from rag.retrieval.models import Candidate

WORD = re.compile(r"\w+")

FilterResult = tuple[list[Candidate], list[Candidate]]


def by_score_threshold(candidates: list[Candidate], threshold: float | None) -> FilterResult:
    """Отсечь кандидатов с cosine similarity ниже порога."""
    if threshold is None:
        return candidates, []
    kept = [c for c in candidates if c.vector_score >= threshold]
    return kept, [c for c in candidates if c.vector_score < threshold]


def by_min_length(candidates: list[Candidate], min_chars: int) -> FilterResult:
    """Отсечь слишком короткие chunks: обрывки списков, одиночные строки."""
    kept = [c for c in candidates if len(c.chunk.text) >= min_chars]
    return kept, [c for c in candidates if len(c.chunk.text) < min_chars]


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def deduplicate(candidates: list[Candidate], threshold: float | None) -> FilterResult:
    """Убрать почти одинаковые chunks, оставив тот, что выше в выдаче.

    Источники дубликатов в корпусе: перекрывающиеся окна (стратегия с overlap) и
    статьи-близнецы (например, «Гагаузия» и «Республика Гагаузия»).
    Сходство — Jaccard по множествам слов: 1.0 — одинаковый набор слов.
    """
    if threshold is None:
        return candidates, []
    kept: list[Candidate] = []
    kept_words: list[set[str]] = []
    dropped = []
    for c in candidates:
        words = set(WORD.findall(c.chunk.text.lower()))
        if any(jaccard(words, w) >= threshold for w in kept_words):
            dropped.append(c)
        else:
            kept.append(c)
            kept_words.append(words)
    return kept, dropped


def limit_per_document(candidates: list[Candidate], limit: int | None) -> FilterResult:
    """Не больше `limit` chunks из одной статьи — чтобы в контексте были разные источники."""
    if limit is None:
        return candidates, []
    counts: dict[str, int] = {}
    kept, dropped = [], []
    for c in candidates:
        doc = c.chunk.document_id
        if counts.get(doc, 0) < limit:
            counts[doc] = counts.get(doc, 0) + 1
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped


def by_rerank_threshold(candidates: list[Candidate], threshold: float | None) -> FilterResult:
    """После reranker'а: отсечь кандидатов, которых reranker счёл нерелевантными."""
    if threshold is None:
        return candidates, []
    kept = [c for c in candidates if (c.rerank_score or 0) >= threshold]
    return kept, [c for c in candidates if (c.rerank_score or 0) < threshold]
