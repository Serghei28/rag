"""Инкрементальная индексация chunks в векторное хранилище.

У каждого проиндексированного chunk хранится `embed_hash` — хеш от текста, который ушёл
в модель, и от настроек модели. При повторном запуске:
- хеш совпал → chunk не пересчитывается;
- хеш изменился (новый текст, другая модель/префикс/заголовок) → пересчёт;
- chunk исчез из файла (статью удалили или изменили chunking) → удаляется из индекса.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag.config import EmbeddingModel
from rag.embeddings.encoder import passage_text
from rag.embeddings.vector_store import VectorStore
from rag.preprocessing.models import Chunk


class PassageEncoder(Protocol):
    spec: EmbeddingModel

    @property
    def dim(self) -> int: ...

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray: ...


@dataclass
class IndexResult:
    collection: str
    total: int
    added: int
    updated: int
    deleted: int
    unchanged: int
    seconds: float


def embed_hash(spec: EmbeddingModel, text: str) -> str:
    signature = f"{spec.model_id}|{spec.passage_prefix}|{spec.max_seq_length}|{text}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def index_chunks(
    store: VectorStore,
    encoder: PassageEncoder,
    collection: str,
    chunks: list[Chunk],
    *,
    with_header: bool = True,
    show_progress: bool = False,
) -> IndexResult:
    started = time.perf_counter()
    texts = {c.chunk_id: passage_text(c, with_header) for c in chunks}
    hashes = {cid: embed_hash(encoder.spec, t) for cid, t in texts.items()}

    store.ensure_collection(collection, encoder.dim)
    existing = store.existing_hashes(collection)

    to_embed = [c for c in chunks if existing.get(c.chunk_id) != hashes[c.chunk_id]]
    stale = existing.keys() - hashes.keys()

    if to_embed:
        vectors = encoder.encode_passages([texts[c.chunk_id] for c in to_embed], show_progress)
        store.upsert(collection, to_embed, vectors, hashes)
    store.delete_chunks(collection, stale)

    added = sum(c.chunk_id not in existing for c in to_embed)
    return IndexResult(
        collection=collection,
        total=len(chunks),
        added=added,
        updated=len(to_embed) - added,
        deleted=len(stale),
        unchanged=len(chunks) - len(to_embed),
        seconds=round(time.perf_counter() - started, 1),
    )
