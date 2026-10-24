"""Трейсинг в Langfuse.

Каждый запрос к RAG будет записываться в Langfuse как trace с шагами
retrieval → rerank → filter → generation. Если ключи не заданы, трейсинг
отключается и pipeline работает без него.
"""

from functools import lru_cache

from langfuse import Langfuse

from rag.config import get_settings


@lru_cache
def get_langfuse() -> Langfuse | None:
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
