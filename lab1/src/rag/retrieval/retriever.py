"""Retriever: вопрос → релевантные chunks для LLM.

    вопрос
      ↓ embedding-модель (bi-encoder)
    векторный поиск в Qdrant (+ фильтр по metadata) → top_k кандидатов
      ↓ порог similarity → короткие → дубликаты → лимит на статью
      ↓ reranker (cross-encoder) → сортировка по релевантности
      ↓ порог reranker'а
    top_n chunks → LLM

Все параметры можно переопределить при вызове — это использует код экспериментов.
"""

import time
from dataclasses import dataclass, replace

from rag.config import AppConfig, RetrievalConfig
from rag.embeddings.encoder import get_encoder
from rag.embeddings.vector_store import VectorStore, collection_name, get_store
from rag.retrieval import filters
from rag.retrieval.models import Candidate, MetadataFilter, RetrievalResult


@dataclass
class RetrievalParams:
    top_k: int
    top_n: int
    score_threshold: float | None
    min_chars: int
    dedup_threshold: float | None
    max_per_document: int | None
    rerank: bool
    rerank_threshold: float | None


class Retriever:
    def __init__(
        self,
        config: AppConfig,
        *,
        embedding: str | None = None,
        chunking: str | None = None,
        reranker: str | None = None,
        store: VectorStore | None = None,
        encoder=None,
        reranker_model=None,
    ):
        self.config = config
        self.embedding_spec = config.embedding_model(embedding or config.active.embedding)
        self.chunking = chunking or config.active.chunking
        self.collection = collection_name(
            self.chunking, self.embedding_spec.name, config.embeddings.with_header
        )
        self.store = store or get_store(config.paths.index_dir / "qdrant")
        if not self.store.exists(self.collection):
            raise LookupError(f"коллекции {self.collection} нет — запустите `pixi run index`")

        emb = config.embeddings
        self.encoder = encoder or get_encoder(self.embedding_spec, emb.device, emb.batch_size)

        self.reranker_name = reranker or config.active.reranker
        self._reranker = reranker_model  # грузится лениво: без rerank модель не нужна

        self.defaults = self._params_from(config.retrieval)

    def _params_from(self, r: RetrievalConfig) -> RetrievalParams:
        rr = self.config.reranking
        return RetrievalParams(
            top_k=r.top_k,
            top_n=rr.top_n,
            score_threshold=r.score_thresholds.get(self.embedding_spec.name),
            min_chars=r.min_chars,
            dedup_threshold=r.dedup_threshold,
            max_per_document=r.max_per_document,
            rerank=rr.enabled,
            rerank_threshold=rr.score_thresholds.get(self.reranker_name),
        )

    @property
    def reranker(self):
        if self._reranker is None:
            from rag.reranking.reranker import get_reranker

            rr = self.config.reranking
            spec = self.config.reranker_model(self.reranker_name)
            self._reranker = get_reranker(spec, self.config.embeddings.device, rr.batch_size)
        return self._reranker

    def retrieve(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        **overrides,
    ) -> RetrievalResult:
        """`overrides` — любые поля RetrievalParams, например top_k=10, rerank=False."""
        p = replace(self.defaults, **overrides)
        timings: dict[str, float] = {}
        dropped: dict[str, list[Candidate]] = {}

        t = time.perf_counter()
        vector = self.encoder.encode_query(query)
        timings["embed"] = _ms(t)

        t = time.perf_counter()
        hits = self.store.search(
            self.collection,
            vector,
            limit=p.top_k,
            query_filter=metadata_filter.to_qdrant() if metadata_filter else None,
        )
        timings["search"] = _ms(t)
        candidates = [
            Candidate(chunk=h.chunk, vector_score=h.score, vector_rank=i)
            for i, h in enumerate(hits, start=1)
        ]

        t = time.perf_counter()
        current = candidates
        for reason, step in [
            ("score_threshold", lambda c: filters.by_score_threshold(c, p.score_threshold)),
            ("too_short", lambda c: filters.by_min_length(c, p.min_chars)),
            ("duplicate", lambda c: filters.deduplicate(c, p.dedup_threshold)),
            ("per_document_limit", lambda c: filters.limit_per_document(c, p.max_per_document)),
        ]:
            current, out = step(current)
            if out:
                dropped[reason] = out
        timings["filters"] = _ms(t)

        reranker_name = None
        if p.rerank and current:
            t = time.perf_counter()
            current = self.reranker.rerank(query, current, self.config.embeddings.with_header)
            timings["rerank"] = _ms(t)
            reranker_name = self.reranker_name
            current, out = filters.by_rerank_threshold(current, p.rerank_threshold)
            if out:
                dropped["rerank_threshold"] = out

        if len(current) > p.top_n:
            dropped["beyond_top_n"] = current[p.top_n :]
            current = current[: p.top_n]

        return RetrievalResult(
            query=query,
            collection=self.collection,
            chunks=current,
            candidates=candidates,
            dropped=dropped,
            timings_ms=timings,
            reranker=reranker_name,
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
