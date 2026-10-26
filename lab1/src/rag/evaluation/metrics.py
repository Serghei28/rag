"""Метрики качества поиска (на уровне статей).

Разметка набора вопросов — названия статей с ответом (`relevant_titles`), а не id
chunks: chunks меняются при каждой смене стратегии chunking, а статьи — нет. Поэтому
chunk считается релевантным, если он из релевантной статьи.

`ranked` — названия статей найденных chunks в порядке выдачи (повторы возможны: из
одной статьи может прийти несколько chunks).

| Метрика      | Вопрос, на который отвечает                                   |
|--------------|---------------------------------------------------------------|
| Hit@K        | есть ли среди первых K хоть один chunk из нужной статьи        |
| Recall@K     | какая доля нужных статей попала в первые K (важно для multi_doc) |
| Precision@K  | какая доля первых K chunks — из нужных статей (сколько шума)   |
| MRR          | насколько высоко первый релевантный chunk (1 / его место)      |
"""

from collections.abc import Sequence


def first_relevant_rank(ranked: Sequence[str], relevant: set[str]) -> int | None:
    return next((i for i, title in enumerate(ranked, start=1) if title in relevant), None)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(title in relevant for title in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Делим на K, а не на число найденных: недобор chunks — тоже потеря точности."""
    return sum(title in relevant for title in ranked[:k]) / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    rank = first_relevant_rank(ranked, relevant)
    return 1 / rank if rank else 0.0


def retrieval_metrics(ranked: Sequence[str], relevant: set[str], ks: Sequence[int]) -> dict:
    metrics = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
    return metrics


def mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None
