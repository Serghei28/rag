"""Индексация и поиск на настоящем Qdrant (встроенный режим во временной папке),
но с фейковым encoder вместо нейросети: векторы = мешок слов, быстро и детерминированно."""

import hashlib
from datetime import UTC, datetime

import numpy as np
import pytest

from rag.config import EmbeddingModel
from rag.embeddings import VectorStore, index_chunks
from rag.preprocessing.models import Chunk

DIM = 64


class BagOfWordsEncoder:
    def __init__(self, model_id: str = "fake/model"):
        self.spec = EmbeddingModel(name="fake", model_id=model_id)
        self.calls: list[int] = []

    @property
    def dim(self) -> int:
        return DIM

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
        return v / (np.linalg.norm(v) or 1)

    def encode_passages(self, texts, show_progress=False):
        self.calls.append(len(texts))
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, query):
        return self._vec(query)


def chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"ruwiki:{i}#s:0",
        document_id=f"ruwiki:{i}",
        title=f"Статья {i}",
        url="https://example.org",
        section="Введение",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        strategy="s",
        position=0,
        text=text,
    )


CHUNKS = [
    chunk(1, "столица молдовы город кишинёв"),
    chunk(2, "президент молдовы избирается на четыре года"),
    chunk(3, "парламент состоит из депутатов"),
]


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "qdrant")
    yield s
    s.close()


def test_first_index_adds_all_and_search_finds_relevant(store):
    enc = BagOfWordsEncoder()
    r = index_chunks(store, enc, "c", CHUNKS)
    assert (r.added, r.updated, r.unchanged, r.deleted) == (3, 0, 0, 0)
    assert store.count("c") == 3

    hits = store.search("c", enc.encode_query("какая столица молдовы"), limit=2)
    assert hits[0].chunk.chunk_id == "ruwiki:1#s:0"
    assert hits[0].score > hits[1].score
    assert hits[0].chunk.title == "Статья 1"  # metadata возвращается вместе с вектором


def test_reindex_is_incremental(store):
    enc = BagOfWordsEncoder()
    index_chunks(store, enc, "c", CHUNKS)
    enc.calls.clear()

    changed = [CHUNKS[0], chunk(2, "президент молдовы избирается всенародно")]  # 3 удалён
    r = index_chunks(store, enc, "c", changed)

    assert (r.added, r.updated, r.unchanged, r.deleted) == (0, 1, 1, 1)
    assert enc.calls == [1]  # пересчитан только изменённый chunk
    assert store.count("c") == 2


def test_nothing_to_do_on_repeat(store):
    enc = BagOfWordsEncoder()
    index_chunks(store, enc, "c", CHUNKS)
    enc.calls.clear()
    r = index_chunks(store, enc, "c", CHUNKS)
    assert r.unchanged == 3 and enc.calls == []


def test_changing_model_or_header_reembeds_everything(store):
    index_chunks(store, BagOfWordsEncoder(), "c", CHUNKS)

    r = index_chunks(store, BagOfWordsEncoder("fake/other-model"), "c", CHUNKS)
    assert r.updated == 3

    r = index_chunks(store, BagOfWordsEncoder("fake/other-model"), "c", CHUNKS, with_header=False)
    assert r.updated == 3


def test_dimension_change_recreates_collection(store):
    store.ensure_collection("c", 8)
    store.ensure_collection("c", 16)
    assert store.client.get_collection("c").config.params.vectors.size == 16
