"""Текст chunk для векторизации. Отдельный модуль без torch: его импортирует индексатор
и команды работы с базой, которым нейросети не нужны."""

from rag.preprocessing.models import Chunk


def passage_text(chunk: Chunk, with_header: bool) -> str:
    """Текст chunk для векторизации. Заголовок даёт контекст коротким фрагментам:
    «Президент: Майя Санду» без названия статьи неясно, о какой стране речь."""
    if not with_header:
        return chunk.text
    return f"{chunk.title}. {chunk.section}\n{chunk.text}"
