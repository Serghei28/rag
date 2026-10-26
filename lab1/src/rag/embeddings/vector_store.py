"""Векторное хранилище на Qdrant (встроенный режим, без сервера).

Почему Qdrant:
- встроенный режим `QdrantClient(path=...)` работает без Docker и сервера, а тот же код
  переключается на сервер заменой `path=` на `url=`;
- хранит payload (metadata chunk) рядом с вектором и умеет фильтровать по нему —
  это нужно для фильтров по metadata на этапе retrieval;
- FAISS хранит только векторы: metadata и фильтрацию пришлось бы писать самим.

Одна коллекция = одна пара (стратегия chunking × embedding-модель), например
`section_800__bge-m3`. Так эксперименты сравнивают комбинации без переиндексации.
"""

import atexit
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from rag.preprocessing.models import Chunk

UPSERT_BATCH = 256


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity: 1 — совпадение, 0 — не связано


def collection_name(strategy: str, model: str, with_header: bool = True) -> str:
    """Без заголовка «Статья. Раздел» векторы другие — это отдельная коллекция."""
    return f"{strategy}__{model}" + ("" if with_header else "__nohdr")


def point_id(chunk_id: str) -> str:
    # Qdrant принимает id только как число или UUID → детерминированный UUID из chunk_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))

    def collections(self) -> list[str]:
        return sorted(c.name for c in self.client.get_collections().collections)

    def exists(self, name: str) -> bool:
        return self.client.collection_exists(name)

    def count(self, name: str) -> int:
        return self.client.count(name, exact=True).count

    def ensure_collection(self, name: str, dim: int) -> None:
        """Создать коллекцию; пересоздать, если у модели другая размерность."""
        if self.exists(name):
            current = self.client.get_collection(name).config.params.vectors.size
            if current == dim:
                return
            self.client.delete_collection(name)
        self.client.create_collection(
            name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    def existing_hashes(self, name: str) -> dict[str, str]:
        """chunk_id → хеш, с которым chunk был проиндексирован."""
        result, offset = {}, None
        while True:
            points, offset = self.client.scroll(
                name,
                limit=1000,
                offset=offset,
                with_payload=["chunk_id", "embed_hash"],
                with_vectors=False,
            )
            result.update({p.payload["chunk_id"]: p.payload["embed_hash"] for p in points})
            if offset is None:
                return result

    def upsert(
        self, name: str, chunks: list[Chunk], vectors: np.ndarray, hashes: dict[str, str]
    ) -> None:
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            self.client.upsert(
                name,
                points=[
                    models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector=vectors[i + j].tolist(),
                        payload={**c.model_dump(mode="json"), "embed_hash": hashes[c.chunk_id]},
                    )
                    for j, c in enumerate(batch)
                ],
            )

    def delete_chunks(self, name: str, chunk_ids: Iterable[str]) -> None:
        ids = [point_id(cid) for cid in chunk_ids]
        if ids:
            self.client.delete(name, points_selector=models.PointIdsList(points=ids))

    def search(
        self,
        name: str,
        vector: np.ndarray,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[SearchHit]:
        response = self.client.query_points(
            name, query=vector.tolist(), limit=limit, query_filter=query_filter, with_payload=True
        )
        hits = []
        for p in response.points:
            payload = {k: v for k, v in p.payload.items() if k != "embed_hash"}
            hits.append(SearchHit(chunk=Chunk.model_validate(payload), score=p.score))
        return hits

    def close(self) -> None:
        self.client.close()


@lru_cache
def get_store(path: Path) -> VectorStore:
    """Встроенный Qdrant блокирует папку: в одном процессе — один клиент."""
    store = VectorStore(path)
    # Закрыть до завершения интерпретатора, иначе Qdrant пишет ошибку из __del__
    atexit.register(store.close)
    return store
