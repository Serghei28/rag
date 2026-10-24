from datetime import UTC, datetime

import pytest

from rag.config import AppConfig, EmbeddingModel, EmbeddingsConfig, RetrievalConfig
from rag.embeddings import VectorStore, index_chunks
from rag.preprocessing.models import Chunk
from rag.retrieval import Candidate, MetadataFilter, Retriever
from rag.retrieval import filters as f
from tests.test_embeddings import BagOfWordsEncoder


def chunk(i: int, text: str, doc: int | None = None, title: str | None = None) -> Chunk:
    doc = doc if doc is not None else i
    return Chunk(
        chunk_id=f"ruwiki:{doc}#s:{i}",
        document_id=f"ruwiki:{doc}",
        title=title or f"Статья {doc}",
        url="https://example.org",
        section="Введение",
        updated_at=datetime(2026, 1, doc % 28 + 1, tzinfo=UTC),
        strategy="s",
        position=i,
        text=text,
    )


def cand(i: int, score: float, text: str = "", doc: int | None = None) -> Candidate:
    return Candidate(
        chunk=chunk(i, text or f"текст {i} " * 10, doc), vector_score=score, vector_rank=i
    )


def ids(cands):
    return [c.chunk.chunk_id.split(":")[-1] for c in cands]


# --------------------------------------------------------------- filters ---


def test_score_threshold():
    kept, dropped = f.by_score_threshold([cand(1, 0.9), cand(2, 0.4), cand(3, 0.6)], 0.5)
    assert ids(kept) == ["1", "3"] and ids(dropped) == ["2"]
    assert f.by_score_threshold([cand(1, 0.1)], None) == ([cand(1, 0.1)], [])


def test_min_length():
    kept, dropped = f.by_min_length([cand(1, 1, "коротко"), cand(2, 1, "д" * 100)], 60)
    assert ids(kept) == ["2"] and ids(dropped) == ["1"]


def test_dedup_keeps_higher_ranked_copy():
    a = "президент молдовы избирается прямым голосованием граждан на четыре года"
    near = a + " страны"  # почти тот же набор слов
    other = "парламент молдовы однопалатный и состоит из 101 депутата"
    kept, dropped = f.deduplicate([cand(1, 0.9, a), cand(2, 0.8, near), cand(3, 0.7, other)], 0.8)
    assert ids(kept) == ["1", "3"] and ids(dropped) == ["2"]


def test_limit_per_document():
    cs = [cand(1, 0.9, doc=7), cand(2, 0.8, doc=7), cand(3, 0.7, doc=7), cand(4, 0.6, doc=8)]
    kept, dropped = f.limit_per_document(cs, 2)
    assert ids(kept) == ["1", "2", "4"] and ids(dropped) == ["3"]


def test_metadata_filter_to_qdrant():
    assert MetadataFilter().to_qdrant() is None
    flt = MetadataFilter(titles=["Молдова"], exclude_titles=["Гагаузия"]).to_qdrant()
    assert flt.must[0].key == "title" and flt.must_not[0].key == "title"


# ------------------------------------------------------------- retriever ---


class FakeReranker:
    """Считает релевантным chunk, в котором есть слово «кишинёв» — переворачивает порядок."""

    def rerank(self, query, candidates, with_header=True):
        for c in candidates:
            c.rerank_score = 0.95 if "кишинёв" in c.chunk.text else 0.05
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


CORPUS = [
    chunk(1, "столица молдовы столица молдовы столица молдовы находится на реке бык", 1, "Молдова"),
    chunk(2, "столица молдовы город кишинёв крупнейший город страны", 2, "Кишинёв"),
    chunk(3, "парламент состоит из депутатов избираемых на четыре года", 3, "Парламент"),
]


@pytest.fixture
def retriever(tmp_path):
    enc = BagOfWordsEncoder()
    store = VectorStore(tmp_path / "q")
    index_chunks(store, enc, "s__fake", CORPUS, with_header=False)
    cfg = AppConfig(
        embeddings=EmbeddingsConfig(models=[EmbeddingModel(name="fake", model_id="fake/model")]),
        retrieval=RetrievalConfig(top_k=3, min_chars=10, score_thresholds={"fake": 0.2}),
    )
    yield Retriever(
        cfg, embedding="fake", chunking="s", store=store, encoder=enc, reranker_model=FakeReranker()
    )
    store.close()


def test_reranker_reorders_candidates(retriever):
    without = retriever.retrieve("столица молдовы", rerank=False, top_n=2)
    assert without.chunks[0].chunk.title == "Молдова"  # векторный поиск: больше совпадений слов

    result = retriever.retrieve("столица молдовы", top_n=2)
    assert result.chunks[0].chunk.title == "Кишинёв"
    assert result.chunks[0].vector_rank == 2  # поднялся со 2-го места
    assert result.reranker is not None


def test_irrelevant_query_returns_empty_result(retriever):
    result = retriever.retrieve("рецепт борща со сметаной")
    assert result.is_empty
    assert "score_threshold" in result.dropped or not result.candidates


def test_rerank_threshold_and_top_n(retriever):
    result = retriever.retrieve("столица молдовы", rerank_threshold=0.5)
    assert [c.chunk.title for c in result.chunks] == ["Кишинёв"]
    assert "rerank_threshold" in result.dropped


def test_metadata_filter_restricts_search(retriever):
    result = retriever.retrieve(
        "столица молдовы",
        rerank=False,
        metadata_filter=MetadataFilter(titles=["Парламент"]),
        score_threshold=None,
    )
    assert {c.chunk.title for c in result.candidates} == {"Парламент"}


def test_unknown_collection_raises(tmp_path):
    cfg = AppConfig(embeddings=EmbeddingsConfig(models=[EmbeddingModel(name="x", model_id="x")]))
    store = VectorStore(tmp_path / "empty")
    with pytest.raises(LookupError):
        Retriever(cfg, embedding="x", chunking="none", store=store, encoder=object())
    store.close()


def test_metadata_filter_by_update_date(retriever):
    result = retriever.retrieve(
        "столица молдовы парламент",
        rerank=False,
        score_threshold=None,
        metadata_filter=MetadataFilter(updated_after=datetime(2026, 1, 3, 12, tzinfo=UTC)),
    )
    assert {c.chunk.title for c in result.candidates} == {"Парламент"}  # правка 4 января
