"""Трейсинг в Langfuse."""

from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def observe(name: str, **kwargs) -> Iterator:
    """Записать шаг trace в Langfuse или пропустить его без настроенных ключей."""
    client = get_langfuse()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(name=name, **kwargs) as observation:
        yield observation


@contextmanager
def session(session_id: str, tags: list[str] | None = None, metadata: dict | None = None):
    """Все трейсы внутри блока попадают в одну сессию Langfuse (например, прогон оценки)."""
    if get_langfuse() is None:
        yield
        return
    from langfuse import propagate_attributes

    with propagate_attributes(session_id=session_id, tags=tags, metadata=metadata):
        yield


def current_trace_id() -> str | None:
    client = get_langfuse()
    return client.get_current_trace_id() if client is not None else None


def score(trace_id: str | None, name: str, value: float, comment: str = "") -> None:
    """Прикрепить оценку (например, LLM-судьи) к трейсу ответа."""
    client = get_langfuse()
    if client is not None and trace_id:
        client.create_score(trace_id=trace_id, name=name, value=value, comment=comment or None)


def flush() -> None:
    """Отправить накопленные трейсы перед завершением короткой CLI-команды."""
    client = get_langfuse()
    if client is not None:
        client.flush()
