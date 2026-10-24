"""Reranker на cross-encoder.

Bi-encoder (embedding-модель) кодирует вопрос и chunk по отдельности и сравнивает
векторы — это быстро, поэтому подходит для поиска по всему корпусу. Cross-encoder
читает вопрос и chunk вместе, одной последовательностью, и видит, как слова вопроса
соотносятся со словами текста. Это точнее, но в десятки раз дороже, поэтому
cross-encoder переоценивает только top-20 кандидатов, а не весь корпус.
"""

import logging
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from rag.config import RerankerModel
from rag.embeddings.encoder import passage_text, resolve_device
from rag.retrieval.models import Candidate

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, spec: RerankerModel, device: str = "auto", batch_size: int = 16):
        self.spec = spec
        self.batch_size = batch_size
        self.device = resolve_device(device)
        log.info("загрузка reranker %s на %s", spec.model_id, self.device)
        dtype = torch.float16 if spec.fp16 and self.device == "cuda" else torch.float32
        self.model = CrossEncoder(
            spec.model_id,
            device=self.device,
            max_length=spec.max_length,
            model_kwargs={"torch_dtype": dtype},
            # sigmoid переводит логит в 0..1: score сопоставим между вопросами и пригоден для порога
            activation_fn=torch.nn.Sigmoid(),
        )
        # Первый вызов на GPU занимает секунды (инициализация CUDA-ядер). Прогреваем сразу,
        # чтобы первый вопрос пользователя и замеры времени не включали эту задержку.
        self.score("прогрев", ["прогрев модели"])

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]

    def rerank(
        self, query: str, candidates: list[Candidate], with_header: bool = True
    ) -> list[Candidate]:
        """Проставить rerank_score и отсортировать по нему (по убыванию)."""
        texts = [passage_text(c.chunk, with_header) for c in candidates]
        for c, s in zip(candidates, self.score(query, texts), strict=True):
            c.rerank_score = s
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


@lru_cache(maxsize=2)
def get_reranker(spec: RerankerModel, device: str = "auto", batch_size: int = 16) -> Reranker:
    return Reranker(spec, device, batch_size)
