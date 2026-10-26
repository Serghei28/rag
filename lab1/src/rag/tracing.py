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


def flush() -> None:
    """Отправить накопленные трейсы перед завершением короткой CLI-команды."""
    client = get_langfuse()
    if client is not None:
        client.flush()
